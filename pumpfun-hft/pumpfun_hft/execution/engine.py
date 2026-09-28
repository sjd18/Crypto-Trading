"""Live / paper trading engine.

``LiveTrader`` runs the *same* decision stack as the backtester on real-time data:

    LiveStreamCollector queue -> MarketState / WalletIntel / FeatureEngine / Discovery
    -> StrategyRuntime.evaluate -> OrderQueue (EXIT > SELL > BUY priority)
    -> N async workers -> ExecutionGateway (paper or live) -> Portfolio / RiskEngine
    -> StrategyRuntime.on_result (retries after backoff)

Background tasks: periodic sweeps (time exits, flatten, outcome resolution), equity marks and
state snapshots to SQLite for the dashboard's Live Monitor, and circuit-breaker feeds (RPC
latency p90, observed slot time). ``stop(flatten=True)`` exits all positions before shutdown.

Live mode requires ``app.mode: live`` **and** the ``--confirm-live`` CLI flag; paper mode needs
no wallet.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any

import numpy as np

from pumpfun_hft.analytics.wallet_intel import WalletIntel
from pumpfun_hft.backtester.execution_sim import ExecutionSimulator
from pumpfun_hft.core.amm import ConstantProductAmm
from pumpfun_hft.core.clock import WallClock
from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Event, EventKind, Fill, Order, OrderStatus, Side, Urgency
from pumpfun_hft.discovery.creator import CreatorBook
from pumpfun_hft.discovery.lifecycle import OutcomeResolver
from pumpfun_hft.discovery.scanner import TokenDiscoveryEngine, classify_sector
from pumpfun_hft.features.market import MarketState
from pumpfun_hft.features.online import FeatureEngine
from pumpfun_hft.ml.rug_model import build_rug_scorer
from pumpfun_hft.risk.engine import RiskEngine
from pumpfun_hft.risk.portfolio import Portfolio
from pumpfun_hft.risk.position_manager import PositionManager
from pumpfun_hft.risk.sizing import Sizer
from pumpfun_hft.strategies.base import build_strategy
from pumpfun_hft.strategies.runtime import StrategyRuntime
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger

log = get_logger("trades")
lat_log = get_logger("latency")
_TRADES = frozenset({EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value})


def _priority(order: Order) -> int:
    if order.side is Side.SELL:
        return 0 if order.urgency is Urgency.EXIT else 1
    return 2


class LiveTrader:
    """Real-time orchestrator (see module docstring).

    Example::

        trader = LiveTrader(settings, events_queue, gateway_factory=lambda sim: PaperGateway(sim, ...))
        await trader.run()
    """

    def __init__(self, settings: Any, events: asyncio.Queue[Event], gateway_factory: Any, *, strategies: list[str] | None = None,
                 meta: Any = None, latency: LatencyTracker | None = None, metadata: dict[str, Any] | None = None,
                 initial_capital_sol: float | None = None, collector_status: Any = None, clock: Any = None) -> None:
        s = settings
        self.s = s
        self.events = events
        # market time: the wall clock when trading live / paper, a ReplayClock when replaying recorded events
        self.clock = clock or WallClock()
        self.meta = meta
        self.latency = latency or LatencyTracker()
        self.collector_status = collector_status
        self.curve = BondingCurve.from_config(s.protocol.curve, s.protocol.curve_fee_tiers)
        flat = s.protocol.amm_flat_fees
        self.amm = ConstantProductAmm(FeeSchedule.from_config(s.protocol.amm_fee_tiers), FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))
        sec = s.risk.sectors
        self.market = MarketState(self.curve, lambda n, sym: classify_sector(n, sym, sec.keywords, sec.default), metadata)
        self.wallets = WalletIntel(s.wallet_intel, s.features)
        self.features = FeatureEngine(s.features, self.curve, self.wallets)
        self.creators = CreatorBook(s.discovery)
        self.resolver = OutcomeResolver(s.discovery, self.creators, self.wallets)
        self.discovery = TokenDiscoveryEngine(s, self.market, self.creators)
        cap = s.live.paper_initial_capital_sol if initial_capital_sol is None else initial_capital_sol
        self.portfolio = Portfolio(int(cap * LAMPORTS_PER_SOL), s.fees.close_token_account_on_exit)
        self.risk = RiskEngine(s.risk, self.portfolio, int(s.backtest.min_order_sol * LAMPORTS_PER_SOL))
        self.sizer = Sizer(s.sizing, s.risk.limits, s.position, s.backtest)
        self.posmgr = PositionManager(s.position)
        self.sim = ExecutionSimulator(s, self.curve, self.amm, self.market, np.random.default_rng(s.app.seed))
        names = strategies or list(s.strategy.active)
        built = [build_strategy(n, s) for n in names]
        overlay = build_strategy(s.strategy.exit_overlay, s) if s.strategy.exit_overlay else None
        self.runtime = StrategyRuntime(s, built, overlay, self.market, self.features, self.wallets, self.creators,
                                       build_rug_scorer(s.rug_model), self.portfolio, self.risk, self.sizer, self.posmgr,
                                       self.sim, self.discovery, record_signals=True, max_signal_records=10_000, signal_ring=True)
        self.gateway = gateway_factory(self.sim)
        self.queue: asyncio.PriorityQueue[tuple[int, int, Order]] = asyncio.PriorityQueue(maxsize=s.live.queue_maxsize)
        self._seq = itertools.count()
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop = asyncio.Event()
        self._order_ctx: dict[int, dict[str, Any]] = {}
        self.fills: list[Fill] = []
        self.events_processed = 0
        self.session = s.app.mode  # shown on the Live monitor: "paper", "live" or "paper-replay"
        self.stopped = False       # set by stop(); the final snapshot carries it so the dashboard can say so

    # ------------------------------------------------------------------ event path
    def on_event(self, ev: Event) -> list[Order]:
        t0_ns = time.perf_counter_ns()
        t = self.clock.now_ms()
        now = max(ev.ts_ms, t) if ev.ts_ms else t
        if ev.ts_ms:  # time the event spent waiting in the queue (backlog indicator)
            self.latency.record("live.queue_wait", float(max(0, t - ev.ts_ms)))
        st = self.market.on_event(ev)
        if ev.kind == EventKind.CREATE.value and st is not None:
            self.wallets.on_create(ev)
            self.resolver.on_create(st)
            self.discovery.on_create(st)
        self.features.on_event(ev, st)
        if ev.kind in _TRADES and st is not None:
            self.wallets.on_trade(ev, st.created_ms, st.created_slot, st.creator)
        self.events_processed += 1
        if st is None:
            return []
        orders = self.runtime.evaluate(ev, st, now)
        self.latency.record_ns("live.event_to_decision", t0_ns)
        return orders

    async def enqueue(self, order: Order, delay_ms: int = 0) -> None:
        if delay_ms:
            await self.clock.sleep(delay_ms)
        pos = self.portfolio.position(order.mint)
        st = self.market.get(order.mint)
        self._order_ctx[order.id] = {"strategy": order.strategy if order.side is Side.BUY or pos is None else pos.strategy,
                                     "creator": st.creator if st else None, "sector": st.sector if st else "other",
                                     "stop_frac": self.sizer.stop_fraction(None), "exit_overrides": order.exit_overrides}
        await self.queue.put((_priority(order), next(self._seq), order))

    async def _consume(self) -> None:
        while not self._stop.is_set():
            ev = await self.events.get()
            for o in self.on_event(ev):
                await self.enqueue(o)

    def _book(self, fill: Fill) -> None:
        ctx = self._order_ctx.pop(fill.order_id, {})
        self.portfolio.apply_fill(fill, strategy=ctx.get("strategy", fill.strategy), creator=ctx.get("creator"),
                                  sector=ctx.get("sector", "other"), stop_frac=ctx.get("stop_frac", 0.25),
                                  exit_overrides=ctx.get("exit_overrides"))
        failed = fill.status in (OrderStatus.FAILED, OrderStatus.DROPPED, OrderStatus.EXPIRED)
        self.risk.on_fill(fill.filled, failed, fill.slippage_bps, self.clock.now_ms())
        self.fills.append(fill)

    async def _worker(self, wid: int) -> None:
        while not self._stop.is_set():
            _, _, order = await self.queue.get()
            try:
                try:
                    fill = await self.gateway.execute(order)
                except Exception as exc:  # noqa: BLE001 - a gateway bug must not kill the worker
                    log.exception("gateway error", extra={"data": {"order": order.id}})
                    t = self.clock.now_ms()
                    from pumpfun_hft.core.types import Venue

                    fill = Fill(order.id, order.mint, order.side, order.action, OrderStatus.REJECTED, order.strategy, order.reason,
                                order.created_ms, t, t, t, 0, Venue.CURVE, failure=f"gateway_error:{type(exc).__name__}")
                self._book(fill)
                retry = self.runtime.on_result(fill, self.clock.now_ms())
                if retry is not None:
                    self._spawn(self.enqueue(retry, self.s.simulation.retries.backoff_ms))
            except Exception:  # noqa: BLE001 - booking must never stall the queue (flatten waits on it)
                log.exception("order handling failed", extra={"data": {"order": order.id, "worker": wid}})
            finally:
                self.queue.task_done()

    def _spawn(self, coro: Any) -> None:
        """Fire-and-forget task that is tracked (so stop() cancels it and exceptions are not lost)."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)

    async def _sweep_loop(self) -> None:
        while not self._stop.is_set():
            await self.clock.sleep(self.s.backtest.sweep_interval_ms)
            now = self.clock.now_ms()
            for o in self.runtime.on_sweep(now):
                await self.enqueue(o)
            self.resolver.advance(now, self.market.tokens)

    def snapshot(self) -> dict[str, Any]:
        now = self.clock.now_ms()
        for mint, pos in list(self.portfolio.positions.items()):
            if pos.tokens > 0:
                self.portfolio.mark(mint, self.sim.liquidation_value(mint, pos.tokens), self.sim.spot_price(mint), now)
        eq = self.portfolio.record_equity(now)
        self.risk.on_equity(now, eq)
        if self.risk.breakers.flatten_requested:
            self.risk.breakers.flatten_requested = False
            self.runtime.flatten = True
        return {
            "ts_ms": now,
            "session": self.session,
            "stopped": self.stopped,
            "equity_sol": eq / LAMPORTS_PER_SOL,
            "cash_sol": self.portfolio.cash / LAMPORTS_PER_SOL,
            "positions": [{"mint": p.mint, "strategy": p.strategy, "tokens": p.tokens, "value_sol": p.last_value / LAMPORTS_PER_SOL,
                           "cost_sol": p.cost_lamports / LAMPORTS_PER_SOL, "ret": p.economic_return(), "entry_ms": p.entry_ms}
                          for p in self.portfolio.positions.values() if p.tokens > 0],
            "closed_trades": len(self.portfolio.trades),
            "realized_pnl_sol": sum(t.pnl_sol for t in self.portfolio.trades),
            "pending_orders": len(self.runtime.pending),
            "queue": self.queue.qsize(),
            "events_processed": self.events_processed,
            "risk": self.risk.status(now),
            "latency": self.latency.snapshot(),
            "collector": self.collector_status() if self.collector_status else {},
            "recent_signals": list(self.runtime.signal_records)[-50:],
        }

    async def _snapshot_loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            await asyncio.sleep(self.s.live.state_snapshot_s)
            snap = self.snapshot()
            if self.meta is not None:
                self.meta.set_state("live", snap)
            n += 1
            if n % max(1, int(10 / max(self.s.live.state_snapshot_s, 1e-3))) == 0:  # ~every 10 s
                lat_log.info("latency snapshot", extra={"data": snap["latency"]})

    async def _breaker_feed(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)  # wall time: RPC latency and slot time are properties of the real network
            now = self.clock.now_ms()
            for name in self.latency.names():
                if name.startswith("api.rpc"):
                    self.risk.breakers.on_rpc_latency(self.latency.percentile(name, 90), now)
            slot_ms = self.latency.samples("chain.slot_ms")
            if slot_ms.size >= 10:
                self.risk.breakers.on_slot_time(float(slot_ms[-50:].mean()), now)

    async def run(self) -> None:
        self._tasks = [asyncio.create_task(self._consume()), asyncio.create_task(self._sweep_loop()),
                       asyncio.create_task(self._snapshot_loop()), asyncio.create_task(self._breaker_feed())]
        self._tasks += [asyncio.create_task(self._worker(i)) for i in range(self.s.live.workers)]
        log.info("live trader started", extra={"data": {"mode": self.s.app.mode, "strategies": [st.name for st in self.runtime.strategies]}})
        await self._stop.wait()

    async def stop(self, flatten: bool = False, timeout_s: float = 30.0) -> None:
        if flatten:
            self.runtime.flatten = True
            for o in self.runtime.on_sweep(self.clock.now_ms()):
                await self.enqueue(o)
            try:
                await asyncio.wait_for(self.queue.join(), timeout=timeout_s)
            except TimeoutError:
                log.warning("flatten timed out")
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.stopped = True
        if self.meta is not None:  # final snapshot: the Live monitor must not keep showing pre-shutdown positions
            try:
                self.meta.set_state("live", self.snapshot())
            except Exception:  # noqa: BLE001 - shutdown must finish even if the store is unavailable
                log.exception("final snapshot failed")
        log.info("live trader stopped", extra={"data": {"flattened": flatten, "fills": len(self.fills),
                                                        "closed_trades": len(self.portfolio.trades)}})
