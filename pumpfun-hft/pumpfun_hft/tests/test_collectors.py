"""Data layer: Parquet store integrity, historical backfill with resume and retry, live decoding and gap handling."""

from __future__ import annotations

import base64
import os

import polars as pl
import pytest

from pumpfun_hft.api.rpc import RpcError
from pumpfun_hft.collectors.historical import HistoricalCollector
from pumpfun_hft.collectors.live import LiveStreamCollector
from pumpfun_hft.collectors.slotclock import assign_timestamps
from pumpfun_hft.collectors.sol_price import StaticSolPrice
from pumpfun_hft.collectors.storage import ParquetEventStore
from pumpfun_hft.core.events import EventDecoder
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.utils.base58 import b58encode
from pumpfun_hft.utils.latency import LatencyTracker

PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def pk() -> str:
    return b58encode(os.urandom(32))


# --------------------------------------------------------------------------- store
@pytest.fixture()
def store(tmp_path) -> ParquetEventStore:
    return ParquetEventStore(tmp_path / "events", MetaStore(tmp_path / "meta.sqlite"))


def test_write_compact_dedupe_and_read(store, events) -> None:
    part = events.head(3_000)
    store.write(part)
    store.write(part.head(500))  # a re-delivered batch (e.g. overlapping backfill)
    assert store.read().height == 3_500
    reports = store.compact()
    assert sum(r.duplicates for r in reports) == 500
    out = store.read()
    assert out.height == 3_000
    assert out.select(["slot", "seq", "ev_idx"]).equals(out.select(["slot", "seq", "ev_idx"]).sort(["slot", "seq", "ev_idx"]))
    t0, t1 = int(part["ts_ms"].quantile(0.25)), int(part["ts_ms"].quantile(0.75))
    window = store.read(t0, t1)
    assert window["ts_ms"].min() >= t0 and window["ts_ms"].max() < t1
    batches = list(store.iter_batches())
    assert sum(b.height for b in batches) == 3_000


def test_checksums_detect_corruption(store, events) -> None:
    store.write(events.head(1_000))
    store.compact()
    assert store.verify() == []
    f = next(store.root.glob("date=*/*.parquet"))
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    assert store.verify() and str(f.name) in store.verify()[0]


def test_gap_detection(store, events) -> None:
    ev = events.head(2_000)
    cut = ev["slot"][1_000]
    shifted = ev.with_columns(pl.when(pl.col("slot") >= cut).then(pl.col("slot") + 10_000).otherwise(pl.col("slot")).alias("slot"))
    store.write(shifted)
    gaps = store.detect_gaps(500)
    assert gaps.height >= 1 and int(gaps["gap_slots"].max()) >= 10_000


def test_slot_clock_timestamps_are_monotone() -> None:
    df = pl.DataFrame({"slot": [100, 101, 105, 110, 111], "block_time": [1_000, None, None, 1_004, None], "ts_ms": [0] * 5})
    out = assign_timestamps(df, 400.0)
    ts = out["ts_ms"].to_list()
    assert ts == sorted(ts)
    # block times have 1-second resolution: estimates stay inside the reported second
    assert 1_000_000 <= ts[0] < 1_001_000 and 1_004_000 <= ts[3] < 1_005_000


# --------------------------------------------------------------------------- historical backfill
def _tx(decoder: EventDecoder, sig: str, slot: int) -> dict:
    codec = decoder.codecs[PUMP]
    f = {n: 0 for n in codec.struct_field_names("TradeEvent")}
    f.update(mint=pk(), user=pk(), fee_recipient=pk(), creator=pk(), is_buy=True, track_volume=False, ix_name="buy",
             sol_amount=10**8, token_amount=10**12, virtual_sol_reserves=30_100_000_000, virtual_token_reserves=10**15,
             real_sol_reserves=10**8, real_token_reserves=7 * 10**14)
    raw = codec.encode_event("TradeEvent", f, truncate_after="ix_name")
    logs = [f"Program {PUMP} invoke [1]", f"Program data: {base64.b64encode(raw).decode()}", f"Program {PUMP} success"]
    return {"slot": slot, "blockTime": 1_788_220_800 + slot // 3, "transaction": {"signatures": [sig], "message": {"accountKeys": []}},
            "meta": {"err": None, "logMessages": logs, "innerInstructions": []}}


class FakeRpc:
    """Newest-first signature history with optional transient fetch failures."""

    def __init__(self, decoder: EventDecoder, n: int, fail: set[str] | None = None) -> None:
        self.history = [{"signature": f"sig{i:05d}", "slot": 1_000 + i, "err": None, "blockTime": None} for i in range(n)][::-1]
        self.decoder = decoder
        self.fail = fail or set()
        self.fetched: list[str] = []

    async def get_signatures_for_address(self, address: str, before: str | None = None, until: str | None = None, limit: int = 1000):
        sigs = [h["signature"] for h in self.history]
        start = sigs.index(before) + 1 if before else 0
        end = sigs.index(until) if until else len(sigs)
        return self.history[start:end][:limit]

    async def get_transactions(self, signatures: list[str]):
        out = []
        for s in signatures:
            self.fetched.append(s)
            if s in self.fail:
                out.append(RpcError(-32000, "temporarily unavailable"))
            else:
                out.append(_tx(self.decoder, s, 1_000 + int(s[3:])))
        return out

    async def get_block_hashes(self, slots):
        return {}


async def test_backfill_resumes_and_retries(settings, tmp_path) -> None:
    decoder = EventDecoder.from_settings(settings)
    meta = MetaStore(tmp_path / "meta.sqlite")
    st = ParquetEventStore(tmp_path / "events", meta)
    rpc = FakeRpc(decoder, 250, fail={"sig00100", "sig00101"})
    cfg = settings.collector.model_copy(update={"signatures_page_limit": 50, "tx_batch_size": 20, "flush_rows": 60})
    col = HistoricalCollector(rpc, decoder, st, meta, cfg, PUMP, StaticSolPrice(150.0), 400.0)
    first = await col.run(max_signatures=120, mode="backfill")
    assert first.signatures == 120 and first.fetch_errors == 0
    fetched_first = set(rpc.fetched)
    second = await col.run(max_signatures=1_000, mode="backfill")  # resumes where the first run stopped
    assert second.done and not (set(rpc.fetched[len(fetched_first):]) & fetched_first)
    assert second.fetch_errors == 2 and {p["signature"] for p in meta.pending()} == {"sig00100", "sig00101"}
    rpc.fail.clear()
    assert await col.retry_pending() == 2 and not meta.pending()
    st.compact()
    df = st.read()
    assert df.height == 250 and df["signature"].n_unique() == 250
    assert df["ts_ms"].min() > 0  # timestamps assigned by the slot clock


# --------------------------------------------------------------------------- live stream
async def test_live_collector_records_truncation_and_flushes(settings, tmp_path) -> None:
    decoder = EventDecoder.from_settings(settings)
    meta = MetaStore(tmp_path / "meta.sqlite")
    st = ParquetEventStore(tmp_path / "events", meta)
    col = LiveStreamCollector(None, decoder, st, meta, settings.collector, settings.protocol, LatencyTracker(), StaticSolPrice(150.0))  # type: ignore[arg-type]
    q = col.subscribe()
    good = _tx(decoder, "a", 5)["meta"]["logMessages"]
    await col.on_logs({"context": {"slot": 5}, "value": {"signature": "a", "err": None, "logs": good}}, 0, 1_788_220_800_000)
    await col.on_logs({"context": {"slot": 6}, "value": {"signature": "b", "err": None, "logs": good[:1] + ["Log truncated"]}}, 0,
                      1_788_220_800_400)
    await col.on_logs({"context": {"slot": 7}, "value": {"signature": "c", "err": {"InstructionError": [0, "x"]}, "logs": good}}, 0,
                      1_788_220_800_800)
    assert q.qsize() == 1 and col.truncated == 1
    assert [p["signature"] for p in meta.pending()] == ["b"]  # queued for an RPC re-fetch
    assert any(g["kind"] == "truncated_logs" for g in meta.gaps())
    await col.on_slot({"slot": 10}, 0, 1_000)
    await col.on_slot({"slot": 12}, 0, 1_800)
    assert col.latency.percentile("chain.slot_ms", 50) == pytest.approx(400.0)
    col.flush()
    assert st.read().height == 1
