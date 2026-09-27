"""Live stream collector (Solana WebSockets).

Subscriptions
    * ``logsSubscribe`` mentioning the Pump program -> launches (CreateEvent), swaps
      (TradeEvent), curve completion (CompleteEvent) and migrations (CompletePumpAmmMigrationEvent)
    * ``logsSubscribe`` mentioning PumpSwap -> post-migration swaps / pool creation
    * ``slotSubscribe`` -> chain clock (slot lag of each event, observed slot time = congestion)
    * optional ``logsSubscribe`` per watched wallet -> raw wallet activity (funding, transfers)

Latency
    Each notification is timestamped on arrival by the WS reader; the collector records
    ``live.event_to_dispatch`` (arrival -> all subscriber queues fed, budget 100 ms) and
    ``live.slot_lag`` (current slot - event slot). Large buys/sells above
    ``collector.large_trade_sol`` are logged to the ``signals`` channel.

Durability
    Events are buffered and flushed to the Parquet store every ``flush_interval_s`` or
    ``flush_rows``. Reconnect gaps and truncated logs are recorded in the gap registry; with a
    historical collector attached, reconnect gaps are backfilled (``backfill_on_reconnect``) and
    truncated transactions are queued and re-fetched over RPC every 30 s.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from pumpfun_hft.collectors.historical import HistoricalCollector
from pumpfun_hft.collectors.sol_price import SolPriceProvider
from pumpfun_hft.collectors.storage import ParquetEventStore
from pumpfun_hft.api.ws import SolanaWsClient
from pumpfun_hft.core.events import EventDecoder
from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Event, EventKind, events_to_frame
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger

log = get_logger("system")
sig_log = get_logger("signals")


class LiveStreamCollector:
    """Decode Pump/PumpSwap activity from WebSockets and fan it out to subscribers.

    Example::

        collector = LiveStreamCollector(ws, decoder, store, meta, settings.collector, settings.protocol, latency, price)
        q = collector.subscribe()            # asyncio.Queue[Event]
        asyncio.create_task(collector.run())
        ev = await q.get()
    """

    def __init__(self, ws: SolanaWsClient, decoder: EventDecoder, store: ParquetEventStore | None, meta: MetaStore | None,
                 cfg: Any, protocol: Any, latency: LatencyTracker, sol_price: SolPriceProvider,
                 commitment: str = "processed", historical: HistoricalCollector | None = None,
                 queue_size: int = 200_000) -> None:
        self.ws = ws
        self.decoder = decoder
        self.store = store
        self.meta = meta
        self.cfg = cfg
        self.protocol = protocol
        self.latency = latency
        self.sol_price = sol_price
        self.commitment = commitment
        self.historical = historical
        self.queue_size = queue_size
        self.latency.set_budget("live.event_to_dispatch", cfg.live_latency_budget_ms)
        self._queues: list[asyncio.Queue[Event]] = []
        self._callbacks: list[Callable[[Event], Awaitable[None] | None]] = []
        self._wallet_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._buffer: list[Event] = []
        self._seq: dict[int, int] = {}
        self.current_slot = 0
        self._last_slot_ms: float | None = None
        self.dropped = 0
        self.events = 0
        self.truncated = 0
        self.watched_wallets: list[str] = []
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ fan-out
    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self.queue_size)
        self._queues.append(q)
        return q

    def add_callback(self, cb: Callable[[Event], Awaitable[None] | None]) -> None:
        """Synchronous-path consumer called inline (keep it fast)."""
        self._callbacks.append(cb)

    def subscribe_wallet_activity(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        self._wallet_queues.append(q)
        return q

    def watch_wallets(self, wallets: list[str]) -> None:
        """Watch up to ``collector.watch_wallets_max`` wallets (one WS subscription each)."""
        self.watched_wallets = list(dict.fromkeys(wallets))[: self.cfg.watch_wallets_max]

    def _next_seq(self, slot: int) -> int:
        n = self._seq.get(slot, 0)
        self._seq[slot] = n + 1
        if len(self._seq) > 4096:  # bounded memory
            for s in sorted(self._seq)[:2048]:
                del self._seq[s]
        return n

    async def _dispatch(self, ev: Event) -> None:
        for q in self._queues:
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                self.dropped += 1
            q.put_nowait(ev)
        for cb in self._callbacks:
            res = cb(ev)
            if asyncio.iscoroutine(res):
                await res

    # ------------------------------------------------------------------ handlers
    async def on_logs(self, result: dict[str, Any], recv_ns: int, recv_ms: int) -> None:
        value = result.get("value") or {}
        slot = int((result.get("context") or {}).get("slot") or 0)
        if value.get("err") is not None and not self.cfg.include_failed_tx:
            return
        sig = value.get("signature")
        events, parsed = self.decoder.events_from_logs(
            value.get("logs") or [], slot=slot, seq=self._next_seq(slot), ts_ms=recv_ms, signature=sig,
            sol_usd=self.sol_price.price_at(recv_ms))
        if parsed.truncated:
            self.truncated += 1
            if self.meta is not None:
                self.meta.add_gap("live", "truncated_logs", start_slot=slot, end_slot=slot, detail=str(sig))
                if sig:  # re-fetched over RPC by the repair loop (self-CPI events are immune to log truncation)
                    self.meta.add_pending([(str(sig), slot)], "truncated_logs")
        for ev in events:
            self.events += 1
            await self._dispatch(ev)
            self._buffer.append(ev)
            if ev.kind == EventKind.TRADE.value and (ev.sol_amount or 0) >= self.cfg.large_trade_sol * LAMPORTS_PER_SOL:
                sig_log.info("large trade", extra={"data": {"mint": ev.mint, "user": ev.user, "is_buy": ev.is_buy,
                                                             "sol": (ev.sol_amount or 0) / LAMPORTS_PER_SOL, "slot": slot}})
            elif ev.kind in (EventKind.CREATE.value, EventKind.COMPLETE.value, EventKind.MIGRATE.value):
                sig_log.info(ev.kind, extra={"data": {"mint": ev.mint, "creator": ev.creator, "slot": slot}})
        if events:
            self.latency.record_ns("live.event_to_dispatch", recv_ns)
            if self.current_slot:
                self.latency.record("live.slot_lag", float(max(0, self.current_slot - slot)))
        if len(self._buffer) >= self.cfg.flush_rows:
            self.flush()

    async def on_slot(self, result: dict[str, Any], recv_ns: int, recv_ms: int) -> None:
        slot = int(result.get("slot") or 0)
        if slot > self.current_slot:
            if self._last_slot_ms is not None and self.current_slot:
                per_slot = (recv_ms - self._last_slot_ms) / max(1, slot - self.current_slot)
                self.latency.record("chain.slot_ms", per_slot)
            self.current_slot = slot
            self._last_slot_ms = recv_ms

    async def on_wallet_logs(self, result: dict[str, Any], recv_ns: int, recv_ms: int) -> None:
        value = result.get("value") or {}
        item = {"signature": value.get("signature"), "slot": (result.get("context") or {}).get("slot"),
                "err": value.get("err"), "recv_ms": recv_ms, "logs": value.get("logs") or []}
        for q in self._wallet_queues:
            if not q.full():
                q.put_nowait(item)

    async def on_gap(self, disconnected_ms: float, reconnected_ms: float) -> None:
        log.warning("live gap", extra={"data": {"from_ms": disconnected_ms, "to_ms": reconnected_ms}})
        if self.meta is not None:
            self.meta.add_gap("live", "ws_disconnect", start_ms=int(disconnected_ms), end_ms=int(reconnected_ms),
                              start_slot=self.current_slot)
        if self.cfg.backfill_on_reconnect and self.historical is not None:
            asyncio.create_task(self.historical.run(max_signatures=self.cfg.max_signatures, mode="catchup"))

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        opts = {"commitment": self.commitment}
        self.ws.add_subscription("pump_logs", "logsSubscribe", [{"mentions": [self.protocol.pump_program_id]}, opts], self.on_logs)
        if self.cfg.subscribe_amm:
            self.ws.add_subscription("amm_logs", "logsSubscribe", [{"mentions": [self.protocol.pump_amm_program_id]}, opts], self.on_logs)
        self.ws.add_subscription("slots", "slotSubscribe", [], self.on_slot)
        for w in self.watched_wallets:
            self.ws.add_subscription(f"wallet:{w}", "logsSubscribe", [{"mentions": [w]}, opts], self.on_wallet_logs)
        self.ws.on_gap(self.on_gap)

    def flush(self) -> None:
        if self._buffer and self.store is not None:
            self.store.write(events_to_frame(self._buffer))
        self._buffer.clear()

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.cfg.flush_interval_s)
            self.flush()

    async def _repair_loop(self, interval_s: float = 30.0) -> None:
        """Periodically re-fetch truncated / failed signatures through the historical collector."""
        while not self._stop.is_set():
            await asyncio.sleep(interval_s)
            if self.historical is None:
                continue
            try:
                n = await self.historical.retry_pending(limit=200)
                if n:
                    log.info("repaired transactions", extra={"data": {"recovered": n}})
            except Exception as exc:  # noqa: BLE001 - repair is best effort; the gap stays registered
                log.warning("repair failed", extra={"data": {"error": repr(exc)[:200]}})

    async def run(self) -> None:
        self.setup()
        tasks = [asyncio.create_task(self._flush_loop())]
        if self.historical is not None:
            tasks.append(asyncio.create_task(self._repair_loop()))
        try:
            await self.ws.run_forever()
        finally:
            self._stop.set()
            for t in tasks:
                t.cancel()
            self.flush()

    async def stop(self) -> None:
        self._stop.set()
        await self.ws.stop()

    def status(self) -> dict[str, Any]:
        snap = self.latency.snapshot()
        return {"events": self.events, "dropped": self.dropped, "truncated": self.truncated, "slot": self.current_slot,
                "reconnects": self.ws.reconnects, "latency": {k: v for k, v in snap.items() if k.startswith(("live.", "chain.", "ws."))}}
