"""Historical collector: Pump program history via RPC.

Pipeline (backfill runs newest -> oldest; ``catchup`` fetches everything newer than the last run)::

    getSignaturesForAddress(program, before=cursor, limit=1000)       (paged)
      -> filter failed txs (configurable)
      -> batched getTransaction (JSON-RPC batch of tx_batch_size, max_concurrency in flight)
      -> EventDecoder.events_from_transaction (self-CPI events preferred, logs fallback)
      -> slot-clock millisecond timestamps + SOL/USD + optional block hashes
      -> ParquetEventStore.write (day partitions, checksummed)  -> checkpoint

Resumability: the checkpoint (oldest/newest signature, counts) is only advanced after the
corresponding events are flushed to disk, so a crash can at worst re-download one buffer (the
store's compaction removes duplicates). Transactions that fail to fetch go to the
``pending_signatures`` retry queue; ``retry_pending`` drains it. Slot discontinuities larger
than ``collector.gap_slot_threshold`` between consecutive signatures are recorded as gaps.

Note: public RPC endpoints rate-limit ``getTransaction`` heavily and prune old history; use a
dedicated/archival RPC for deep backfills.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from pumpfun_hft.api.rpc import RpcError, SolanaRpcClient
from pumpfun_hft.collectors.slotclock import assign_timestamps
from pumpfun_hft.collectors.sol_price import SolPriceProvider
from pumpfun_hft.collectors.storage import ParquetEventStore
from pumpfun_hft.core.events import EventDecoder
from pumpfun_hft.core.types import Event, events_to_frame
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.utils.logging import get_logger

log = get_logger("system")


@dataclass(slots=True)
class CollectStats:
    signatures: int = 0
    transactions: int = 0
    events: int = 0
    failed_tx_skipped: int = 0
    fetch_errors: int = 0
    gaps: int = 0
    pages: int = 0
    done: bool = False
    kinds: dict[str, int] = field(default_factory=dict)


class HistoricalCollector:
    """Resumable, batched, concurrent history downloader for one program address."""

    def __init__(self, rpc: SolanaRpcClient, decoder: EventDecoder, store: ParquetEventStore, meta: MetaStore,
                 cfg: Any, program_id: str, sol_price: SolPriceProvider, slot_ms: float = 400.0) -> None:
        self.rpc = rpc
        self.decoder = decoder
        self.store = store
        self.meta = meta
        self.cfg = cfg
        self.program_id = program_id
        self.sol_price = sol_price
        self.slot_ms = slot_ms
        self.name = f"hist:{program_id}"
        self._buffer: list[Event] = []
        self._pending_ck: dict[str, Any] = {}

    async def _fetch_batch(self, sigs: list[dict[str, Any]], sem: asyncio.Semaphore) -> list[tuple[dict[str, Any], Any]]:
        async with sem:
            try:
                res = await self.rpc.get_transactions([s["signature"] for s in sigs])
            except Exception as exc:  # noqa: BLE001 - whole batch failed -> retry queue
                return [(s, RpcError(-1, repr(exc)[:200])) for s in sigs]
        return list(zip(sigs, res, strict=True))

    def _flush(self, stats: CollectStats) -> None:
        if self._buffer:
            df = events_to_frame(self._buffer)
            df = assign_timestamps(df, self.slot_ms)
            df = df.with_columns(pl.col("ts_ms").map_elements(self.sol_price.price_at, return_dtype=pl.Float64).alias("sol_usd"))
            self.store.write(df)
            for k, n in df.group_by("kind").len().iter_rows():
                stats.kinds[k] = stats.kinds.get(k, 0) + int(n)
            self._buffer.clear()
        if self._pending_ck:
            self.meta.save_checkpoint(self.name, **self._pending_ck)
            self._pending_ck = {}

    async def run(self, max_signatures: int | None = None, mode: str = "backfill") -> CollectStats:
        """Collect up to ``max_signatures`` signatures (``backfill`` older, ``catchup`` newer)."""
        stats = CollectStats()
        limit_total = max_signatures if max_signatures is not None else self.cfg.max_signatures
        ck = self.meta.get_checkpoint(self.name) or {}
        before = ck.get("oldest_signature") if mode == "backfill" else None
        until = ck.get("newest_signature") if mode == "catchup" else None
        index_base = int(ck.get("n_signatures") or 0)
        newest_sig, newest_slot = ck.get("newest_signature"), ck.get("newest_slot")
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        prev_slot: int | None = None
        try:
            while stats.signatures < limit_total:
                page = await self.rpc.get_signatures_for_address(
                    self.program_id, before=before, until=until, limit=min(self.cfg.signatures_page_limit, limit_total - stats.signatures))
                stats.pages += 1
                if not page:
                    stats.done = True
                    self._pending_ck["done"] = 1 if mode == "backfill" else ck.get("done", 0)
                    break
                if mode == "backfill" and newest_sig is None:
                    newest_sig, newest_slot = page[0]["signature"], page[0]["slot"]
                seq_of: dict[str, int] = {}
                keep: list[dict[str, Any]] = []
                for j, item in enumerate(page):
                    slot = int(item["slot"])
                    if prev_slot is not None and prev_slot - slot > self.cfg.gap_slot_threshold:
                        self.meta.add_gap("historical", "slot_gap", start_slot=slot, end_slot=prev_slot,
                                          detail=f"no {self.program_id[:6]} txs for {prev_slot - slot} slots")
                        stats.gaps += 1
                    prev_slot = slot
                    # newest-first paging => chronological order is descending global index
                    seq_of[item["signature"]] = -((index_base + stats.signatures + j) % 2_000_000_000)
                    if item.get("err") is not None and not self.cfg.include_failed_tx:
                        stats.failed_tx_skipped += 1
                        continue
                    keep.append(item)
                batches = [keep[i:i + self.cfg.tx_batch_size] for i in range(0, len(keep), self.cfg.tx_batch_size)]
                results = await asyncio.gather(*(self._fetch_batch(b, sem) for b in batches))
                failed: list[tuple[str, int | None]] = []
                block_hashes: dict[int, str | None] = {}
                if self.cfg.fetch_block_hash:
                    slots = sorted({int(s["slot"]) for s in keep})
                    for i in range(0, len(slots), self.cfg.tx_batch_size):
                        block_hashes.update(await self.rpc.get_block_hashes(slots[i:i + self.cfg.tx_batch_size]))
                for batch in results:
                    for item, tx in batch:
                        if tx is None or isinstance(tx, RpcError):
                            failed.append((item["signature"], item.get("slot")))
                            continue
                        stats.transactions += 1
                        evs = self.decoder.events_from_transaction(
                            tx, seq=seq_of[item["signature"]], block_hash=block_hashes.get(int(tx.get("slot") or 0)))
                        for e in evs:
                            e.ts_ms = 0  # assigned by the slot clock at flush time
                        self._buffer.extend(evs)
                        stats.events += len(evs)
                if failed:
                    self.meta.add_pending(failed, "fetch failed")
                    stats.fetch_errors += len(failed)
                stats.signatures += len(page)
                if mode == "backfill":
                    before = page[-1]["signature"]
                    self._pending_ck.update(oldest_signature=before, oldest_slot=int(page[-1]["slot"]),
                                            newest_signature=newest_sig, newest_slot=newest_slot,
                                            n_signatures=index_base + stats.signatures)
                else:
                    if stats.pages == 1:
                        self._pending_ck.update(newest_signature=page[0]["signature"], newest_slot=int(page[0]["slot"]))
                    until_done = len(page) < self.cfg.signatures_page_limit
                    before = page[-1]["signature"]
                    self._pending_ck["n_signatures"] = index_base + stats.signatures
                    if until_done:
                        stats.done = True
                        break
                if len(self._buffer) >= self.cfg.flush_rows:
                    self._flush(stats)
                log.info("history page", extra={"data": {"sigs": stats.signatures, "events": stats.events, "oldest_slot": page[-1]["slot"]}})
        finally:
            self._flush(stats)
        return stats

    async def retry_pending(self, limit: int = 1000) -> int:
        """Re-fetch signatures from the retry queue; returns number recovered."""
        pending = self.meta.pending(limit)
        if not pending:
            return 0
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        items = [{"signature": p["signature"], "slot": p["slot"]} for p in pending]
        batches = [items[i:i + self.cfg.tx_batch_size] for i in range(0, len(items), self.cfg.tx_batch_size)]
        recovered: list[str] = []
        stats = CollectStats()
        for batch in await asyncio.gather(*(self._fetch_batch(b, sem) for b in batches)):
            for item, tx in batch:
                if tx is None or isinstance(tx, RpcError):
                    self.meta.add_pending([(item["signature"], item["slot"])], "retry failed")
                    continue
                evs = self.decoder.events_from_transaction(tx, seq=0)
                for e in evs:
                    e.ts_ms = 0
                self._buffer.extend(evs)
                recovered.append(item["signature"])
        self._flush(stats)
        self.meta.resolve_pending(recovered)
        return len(recovered)
