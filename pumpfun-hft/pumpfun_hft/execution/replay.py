"""Replay recorded events through the live trading engine with the paper gateway.

Purpose
    An offline, end-to-end run of the *live* code path — event queue, decision stack, priority
    order queue, async workers, paper gateway, circuit breakers, state snapshots, flatten on stop —
    on events from the Parquet store. It is the check to run after changing execution or strategy
    code and before pointing ``paper`` at the real stream.

Timing
    By default the engine runs on a :class:`~pumpfun_hft.core.clock.VirtualTimeEventLoop` with a
    :class:`~pumpfun_hft.core.clock.ReplayClock`: events are released when market time reaches
    their timestamp, every latency / sweep / retry wait is a timer in market time, and whenever
    all tasks are waiting the clock jumps to the next timer. A day of market replays in about a
    minute, deterministically, with the backtester's timing semantics. ``realtime=True`` runs on an
    ordinary loop instead, pacing the replay at 1x (useful for watching the Live Monitor).

    The live engine is asynchronous: orders pass through a priority queue and a worker pool, so
    the simulator's random draws (latency, failures) happen in a different order than in the
    backtester. Results therefore agree with a backtest of the same window statistically, not
    trade by trade; the accounting rules are identical.

Inputs / Outputs
    Inputs: an event frame (``EVENT_COLUMNS``), token metadata, strategy names and an optional
    MetaStore for Live Monitor snapshots. Output: :class:`ReplaySummary` (events, fills, round
    trips, PnL, equity, latency snapshot, wall time) plus the final snapshot in the MetaStore.

Example
    summary = replay_live(settings, events, metadata, ["smart_money"], meta=meta)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from pumpfun_hft.core.clock import ReplayClock, VirtualTimeEventLoop
from pumpfun_hft.core.types import EVENT_COLUMNS, LAMPORTS_PER_SOL, Event, OrderStatus
from pumpfun_hft.execution.engine import LiveTrader
from pumpfun_hft.execution.gateway import PaperGateway
from pumpfun_hft.utils.latency import LatencyTracker


@dataclass
class ReplaySummary:
    events: int
    market_span_s: float
    wall_s: float
    fills: int
    failed_or_dropped: int
    round_trips: int
    realized_pnl_sol: float
    equity_sol: float
    initial_capital_sol: float
    open_positions: int
    latency: dict[str, Any] = field(default_factory=dict)
    trades: pl.DataFrame = field(default_factory=pl.DataFrame)
    fills_frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    ml_funnel: str = ""  # ml_signal filter counts, when that strategy ran

    @property
    def total_return(self) -> float:
        return self.equity_sol / self.initial_capital_sol - 1.0


def replay_live(settings: Any, events: pl.DataFrame, metadata: dict[str, Any] | None, strategies: list[str] | None,
                meta: Any = None, flatten: bool = True, realtime: bool = False, snapshot_every_s: float = 60.0,
                progress: Callable[[int, int], None] | None = None) -> ReplaySummary:
    """Replay ``events`` through a :class:`LiveTrader` with the paper gateway (see module docstring).

    ``snapshot_every_s`` is the market-time interval of Live Monitor snapshots in virtual time
    (the configured ``live.state_snapshot_s`` applies in real time). ``progress(done, total)`` is
    called every 1 % of the events. Call from synchronous code: the replay runs its own event loop.
    """
    rows = events.sort(["slot", "seq", "ev_idx"]).select(list(EVENT_COLUMNS)).rows()
    if not rows:
        raise ValueError("no events to replay")
    if not realtime:
        live = settings.live.model_copy(update={"state_snapshot_s": max(snapshot_every_s, settings.live.state_snapshot_s)})
        settings = settings.model_copy(update={"live": live})
    coro = _replay(settings, rows, metadata, strategies, meta, flatten, progress)
    if realtime:
        return asyncio.run(coro)
    return asyncio.run(coro, loop_factory=VirtualTimeEventLoop)


async def _replay(settings: Any, rows: list[tuple[Any, ...]], metadata: dict[str, Any] | None, strategies: list[str] | None,
                  meta: Any, flatten: bool, progress: Callable[[int, int], None] | None) -> ReplaySummary:
    ts_idx = EVENT_COLUMNS.index("ts_ms")
    t0_market, t1_market = int(rows[0][ts_idx]), int(rows[-1][ts_idx])
    clock = ReplayClock(t0_market)
    latency = LatencyTracker()
    queue: asyncio.Queue[Event] = asyncio.Queue()
    holder: dict[str, LiveTrader] = {}
    trader = LiveTrader(settings, queue, lambda sim: PaperGateway(sim, lambda: holder["t"].features.tps, latency, clock),
                        strategies=strategies, meta=meta, latency=latency, metadata=metadata, clock=clock)
    holder["t"] = trader
    trader.session = "paper-replay"
    wall0 = time.perf_counter()
    run = asyncio.create_task(trader.run())
    step = max(1, len(rows) // 100)
    try:
        for i, row in enumerate(rows):
            ev = Event(*row)
            wait = ev.ts_ms - clock.now_ms()
            if wait > 0:
                await clock.sleep(wait)
            await queue.put(ev)
            if progress is not None and (i + 1) % step == 0:
                progress(i + 1, len(rows))
        while trader.events_processed < len(rows):  # let the engine drain the queue
            await asyncio.sleep(0.001)
        await trader.stop(flatten=flatten, timeout_s=120.0)
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
    snap = trader.snapshot()
    if meta is not None:
        meta.set_state("live", snap)
    fills = trader.fills
    pf = trader.portfolio
    ml = trader.runtime.by_name.get("ml_signal")
    return ReplaySummary(
        events=trader.events_processed, market_span_s=(t1_market - t0_market) / 1000.0, wall_s=time.perf_counter() - wall0,
        fills=sum(1 for f in fills if f.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)),
        failed_or_dropped=sum(1 for f in fills if f.status in (OrderStatus.FAILED, OrderStatus.DROPPED, OrderStatus.EXPIRED)),
        round_trips=len(pf.trades), realized_pnl_sol=sum(t.pnl_sol for t in pf.trades),
        equity_sol=pf.equity_lamports() / LAMPORTS_PER_SOL, initial_capital_sol=pf.initial / LAMPORTS_PER_SOL,
        open_positions=sum(1 for p in pf.positions.values() if p.tokens > 0), latency=latency.snapshot(),
        ml_funnel=ml.funnel(getattr(trader.runtime, "signal_records", None)) if ml is not None and hasattr(ml, "funnel") else "",
        trades=pf.trades_frame(), fills_frame=pl.DataFrame([f.to_dict() for f in fills], infer_schema_length=None) if fills else pl.DataFrame())
