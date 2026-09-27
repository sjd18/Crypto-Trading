"""Live trader with the paper gateway on a replayed stream: decisions, fills, snapshots, flatten on stop,
and the market-time replay of recorded events through the whole live stack."""

from __future__ import annotations

import asyncio
import time

import polars as pl

from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource
from pumpfun_hft.core.clock import ReplayClock, VirtualTimeEventLoop
from pumpfun_hft.core.types import EVENT_COLUMNS, Action, Event, Order, OrderType, Side, Urgency
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.execution.engine import LiveTrader, _priority
from pumpfun_hft.execution.gateway import PaperGateway
from pumpfun_hft.execution.replay import replay_live
from pumpfun_hft.tests.conftest import make_settings
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.timeutil import now_ms


def _order(side: Side, urgency: Urgency, oid: int) -> Order:
    return Order(oid, "M", side, Action.BUY if side is Side.BUY else Action.EXIT, OrderType.MARKET, 0, "t", "r", urgency)


async def test_order_queue_puts_exits_first() -> None:
    q: asyncio.PriorityQueue = asyncio.PriorityQueue()
    orders = [_order(Side.BUY, Urgency.HIGH, 1), _order(Side.SELL, Urgency.NORMAL, 2), _order(Side.SELL, Urgency.EXIT, 3),
              _order(Side.BUY, Urgency.NORMAL, 4)]
    for i, o in enumerate(orders):
        await q.put((_priority(o), i, o))
    out = [(await q.get())[2].id for _ in orders]
    assert out == [3, 2, 1, 4]  # urgent exits, then other sells, then entries in arrival order


async def test_paper_trader_on_a_replayed_stream(events, tmp_path) -> None:
    s = make_settings(**{
        "simulation.latency.inclusion_ms.median_ms": 5.0, "simulation.latency.confirmation_ms.median_ms": 5.0,
        "simulation.latency.network_ms.median_ms": 1.0, "simulation.latency.rpc_ms.median_ms": 1.0,
        "simulation.latency.spike_prob": 0.0, "live.workers": 2, "live.state_snapshot_s": 0.2,
        "backtest.sweep_interval_ms": 200, "strategy.active": ["momentum_ignition", "smart_money"]})
    meta = MetaStore(tmp_path / "meta.sqlite")
    queue: asyncio.Queue = asyncio.Queue()
    lat = LatencyTracker()
    holder: dict = {}
    trader = LiveTrader(s, queue, lambda sim: PaperGateway(sim, lambda: holder["t"].features.tps, lat), meta=meta, latency=lat,
                        initial_capital_sol=10.0)
    holder["t"] = trader
    rows = events.sort(["slot", "seq", "ev_idx"]).head(6_000).select(list(EVENT_COLUMNS)).rows()
    run = asyncio.create_task(trader.run())
    for r in rows:  # replay "live": stamp each event with the wall clock as it arrives
        ev = Event(*r)
        ev.ts_ms = now_ms()
        await queue.put(ev)
        await asyncio.sleep(0)
    for _ in range(400):
        if trader.events_processed >= len(rows) and trader.queue.empty():
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.5)  # let a snapshot happen
    assert trader.events_processed == len(rows)
    submitted = sum(v for k, v in trader.runtime.counts.items() if k.endswith(":submitted"))
    assert submitted > 0 and trader.fills
    snap = meta.all_state().get("live")
    assert snap and snap["value"]["events_processed"] > 0 and snap["value"]["equity_sol"] > 0
    assert lat.count("live.event_to_decision") == len(rows)
    await trader.stop(flatten=True, timeout_s=10.0)
    run.cancel()
    assert not any(p.tokens > 0 for p in trader.portfolio.positions.values())  # flattened before shutdown
    eq = trader.portfolio.equity_lamports() / 1e9
    booked = sum(f.sol_delta for f in trader.fills) / 1e9
    assert abs(eq - (10.0 + booked)) < 1e-9  # cash ledger matches every booked fill
    fills = pl.DataFrame([f.to_dict() for f in trader.fills])
    assert (fills["land_ms"] >= fills["decision_ms"]).all()


def test_virtual_time_loop_jumps_instead_of_waiting() -> None:
    async def main() -> tuple[int, int]:
        clock = ReplayClock(start_ms=1_000_000)
        await clock.sleep(3_600_000)  # an hour of market time
        a = clock.now_ms()
        await asyncio.gather(clock.sleep(500), clock.sleep(1_500))  # concurrent timers resolve in order
        return a, clock.now_ms()

    t0 = time.perf_counter()
    a, b = asyncio.run(main(), loop_factory=VirtualTimeEventLoop)
    assert a == 1_000_000 + 3_600_000 and b == a + 1_500
    assert time.perf_counter() - t0 < 1.0  # no real waiting


def test_replay_through_the_live_engine_matches_the_backtest(settings, events, metadata, tmp_path) -> None:
    """The live stack (queue, workers, paper gateway, snapshots) on the recorded market, on virtual time.

    Deterministic, and close to the backtest of the same events: the live engine is asynchronous, so
    the simulator's random draws come in a different order, but the timing semantics are the same.
    """
    meta = MetaStore(tmp_path / "meta.sqlite")
    rep = replay_live(settings, events, metadata, ["smart_money"], meta=meta)
    again = replay_live(settings, events, metadata, ["smart_money"])
    assert rep.events == events.height and rep.open_positions == 0  # every event processed, flattened at the end
    assert (rep.round_trips, rep.equity_sol) == (again.round_trips, again.equity_sol)  # deterministic
    assert rep.latency["live.queue_wait"]["p99"] == 0.0  # each event is decided at its own market time
    bt = BacktestEngine(settings, DataSource(frame=events), ["smart_money"], metadata=metadata, seed=7).run()
    assert bt.trades.height >= 10
    assert abs(rep.round_trips - bt.trades.height) <= max(3, 0.2 * bt.trades.height)
    assert abs(rep.total_return - bt.metrics["total_return"]) < 0.03
    booked = rep.fills_frame["sol_delta"].sum() / 1e9
    assert abs(rep.equity_sol - (rep.initial_capital_sol + booked)) < 1e-9  # cash ledger matches every booked fill
    snap = meta.all_state()["live"]["value"]
    assert snap["events_processed"] == events.height and snap["closed_trades"] == rep.round_trips
