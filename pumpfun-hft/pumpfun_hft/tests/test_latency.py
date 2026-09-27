"""Latency: the tracker itself, and the in-process budgets that the <100 ms event-to-dispatch and
<250 ms quote-to-submit targets depend on (network time is measured live by `latency-probe`)."""

from __future__ import annotations

import base64
import os
import time

import numpy as np
import pytest

from pumpfun_hft.collectors.live import LiveStreamCollector
from pumpfun_hft.collectors.sol_price import StaticSolPrice
from pumpfun_hft.core.curve import BondingCurve
from pumpfun_hft.core.events import EventDecoder
from pumpfun_hft.utils.base58 import b58encode
from pumpfun_hft.utils.latency import LatencyTracker

pytestmark = pytest.mark.latency
PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def test_tracker_percentiles_budget_and_ring_buffer() -> None:
    lat = LatencyTracker(window=100)
    lat.set_budget("x", 50.0)
    for v in range(1, 201):
        lat.record("x", float(v))
    snap = lat.snapshot()["x"]
    assert snap["count"] == 200 and snap["max"] == 200.0
    assert lat.samples("x").size == 100  # only the most recent window is kept
    assert snap["p50"] == pytest.approx(np.percentile(np.arange(101, 201), 50))
    assert snap["breaches"] == 150
    with lat.measure("block"):
        time.sleep(0.01)
    assert lat.percentile("block", 50) >= 9.0


def _log_lines(decoder: EventDecoder) -> list[str]:
    codec = decoder.codecs[PUMP]
    f = {n: 0 for n in codec.struct_field_names("TradeEvent")}
    f.update(mint=b58encode(os.urandom(32)), user=b58encode(os.urandom(32)), fee_recipient=b58encode(os.urandom(32)),
             creator=b58encode(os.urandom(32)), is_buy=True, track_volume=False, ix_name="buy", sol_amount=10**9,
             token_amount=10**13, virtual_sol_reserves=31 * 10**9, virtual_token_reserves=10**15, real_token_reserves=7 * 10**14)
    raw = codec.encode_event("TradeEvent", f, truncate_after="ix_name")
    return [f"Program {PUMP} invoke [1]", "Program log: Instruction: Buy", f"Program data: {base64.b64encode(raw).decode()}",
            f"Program {PUMP} consumed 40000 of 200000 compute units", f"Program {PUMP} success"]


def test_log_decoding_is_far_below_the_dispatch_budget(settings) -> None:
    decoder = EventDecoder.from_settings(settings)
    lines = _log_lines(decoder)
    t = []
    for i in range(2_000):
        t0 = time.perf_counter_ns()
        evs, _ = decoder.events_from_logs(lines, slot=i, seq=0, ts_ms=0, signature="s")
        t.append((time.perf_counter_ns() - t0) / 1e6)
        assert len(evs) == 1
    assert np.percentile(t, 99) < 5.0  # ms; the whole event-to-dispatch budget is 100 ms


async def test_live_collector_dispatch_latency(settings) -> None:
    lat = LatencyTracker()
    decoder = EventDecoder.from_settings(settings)
    col = LiveStreamCollector(None, decoder, None, None, settings.collector, settings.protocol, lat, StaticSolPrice(150.0))  # type: ignore[arg-type]
    q = col.subscribe()
    lines = _log_lines(decoder)
    for i in range(500):
        recv_ns, recv_ms = time.perf_counter_ns(), int(time.time() * 1000)
        await col.on_logs({"context": {"slot": 1_000 + i}, "value": {"signature": f"s{i}", "err": None, "logs": lines}}, recv_ns, recv_ms)
    assert q.qsize() == 500 and col.events == 500
    p99 = lat.percentile("live.event_to_dispatch", 99)
    assert p99 < settings.collector.live_latency_budget_ms
    assert lat.snapshot()["live.event_to_dispatch"]["breaches"] == 0


def test_quote_is_sub_millisecond(settings) -> None:
    curve = BondingCurve.from_config(settings.protocol.curve, settings.protocol.curve_fee_tiers)
    st = curve.state_from_tokens_sold(300_000_000_000_000)
    t0 = time.perf_counter_ns()
    for i in range(10_000):
        curve.buy_with_budget(st, 100_000_000 + i)
    per_quote_ms = (time.perf_counter_ns() - t0) / 1e6 / 10_000
    assert per_quote_ms < 0.2  # quote -> submit budget is 250 ms; local quoting uses a negligible share


def test_strategy_evaluation_throughput(backtest_result) -> None:
    assert backtest_result.events_per_second > 2_000  # events/s with full state, features and two strategies
