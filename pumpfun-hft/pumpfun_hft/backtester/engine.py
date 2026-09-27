"""Event-driven backtest engine (zero look-ahead by construction).

Loop invariant: when a market event with time ``t`` is applied, every internal event
(order landing, confirmation, retry, periodic sweep, equity sample, candle close) with time
``< t`` has already been processed, and nothing with time ``>= t`` has. Orders decided at ``t``
can therefore only execute against state produced by events at or after ``t + latency`` —
trades execute strictly after their signals, behind every trade that landed before them.

Per market event:
    MarketState -> (create: WalletIntel, OutcomeResolver, Discovery) -> FeatureEngine (uses the
    wallet DB *before* this trade) -> WalletIntel.on_trade -> marks for open positions ->
    strategy evaluation according to the replay mode -> orders -> ExecutionSimulator.submit

Internal events:
    land     -> ExecutionSimulator.execute -> Portfolio.apply_fill (economic truth at landing)
                -> RiskEngine.on_fill -> confirm scheduled
    confirm  -> StrategyRuntime.on_result (the runtime learns about fills only now) -> retries
    result   -> terminal non-landed outcomes (dropped / expired / rejected) -> retries
    sweep    -> time-based exits, drawdown flatten, outcome resolution
    equity   -> liquidation-value marks, equity sample, drawdown breaker

Determinism: every random draw comes from one ``numpy.random.Generator`` seeded by ``seed``.
"""

from __future__ import annotations

import heapq
import time
import uuid
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.analytics.metrics import compute_metrics
from pumpfun_hft.analytics.wallet_intel import WalletIntel
from pumpfun_hft.backtester.execution_sim import ExecutionSimulator
from pumpfun_hft.backtester.replay import DataSource, load_metadata
from pumpfun_hft.backtester.results import BacktestResult
from pumpfun_hft.core.amm import ConstantProductAmm
from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
from pumpfun_hft.core.types import (
    EVENT_COLUMNS,
    LAMPORTS_PER_SOL,
    Action,
    Event,
    EventKind,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Urgency,
)
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
from pumpfun_hft.strategies.base import Strategy, build_strategy
from pumpfun_hft.strategies.runtime import StrategyRuntime
from pumpfun_hft.utils.logging import get_logger

log = get_logger("backtests")
INF = 1 << 62
_TRADE_KINDS = frozenset({EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value})
# heap priorities for simultaneous internal events
_P_LAND, _P_RESULT, _P_CONFIRM, _P_SUBMIT = 0, 1, 2, 3


class BacktestEngine:
    """Replay events through the full trading stack.

    Example::

        src = DataSource(frame=events_df)
        res = BacktestEngine(settings, src, ["momentum_ignition"], metadata=meta, seed=7).run()
        res.metrics["sharpe"], res.trades
    """

    def __init__(self, settings: Any, source: DataSource, strategies: list[str | Strategy] | None = None, *,
                 metadata: Any = None, seed: int | None = None, run_id: str | None = None,
                 latency_scale: float = 1.0, failure_scale: float = 1.0, trade_start_ms: int | None = None,
                 allowed_mints: set[str] | None = None, strategy_overrides: dict[str, dict[str, Any]] | None = None,
                 synthetic: bool = False, rug_scorer: Any = None, record_signals: bool | None = None) -> None:
        s = settings
        self.s = s
        self.source = source
        self.seed = s.app.seed if seed is None else seed
        self.run_id = run_id or f"bt-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.synthetic = synthetic
        self.trade_start_ms = trade_start_ms
        self.curve = BondingCurve.from_config(s.protocol.curve, s.protocol.curve_fee_tiers)
        flat = s.protocol.amm_flat_fees
        self.amm = ConstantProductAmm(FeeSchedule.from_config(s.protocol.amm_fee_tiers),
                                      FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))
        sectors = s.risk.sectors
        metadata = load_metadata(metadata) if metadata is not None else None  # accepts {mint: row}, a DataFrame or a path
        self.market = MarketState(self.curve, lambda n, sym: classify_sector(n, sym, sectors.keywords, sectors.default), metadata)
        self.wallets = WalletIntel(s.wallet_intel, s.features)
        self.features = FeatureEngine(s.features, self.curve, self.wallets)
        self.creators = CreatorBook(s.discovery)
        self.resolver = OutcomeResolver(s.discovery, self.creators, self.wallets)
        self.discovery = TokenDiscoveryEngine(s, self.market, self.creators)
        self.rug = rug_scorer if rug_scorer is not None else build_rug_scorer(s.rug_model)
        self.portfolio = Portfolio(int(s.backtest.initial_capital_sol * LAMPORTS_PER_SOL), s.fees.close_token_account_on_exit)
        self.risk = RiskEngine(s.risk, self.portfolio, int(s.backtest.min_order_sol * LAMPORTS_PER_SOL))
        self.risk.breakers.verbose = False  # simulated trips are recorded in the diagnostics, not logged
        self.sizer = Sizer(s.sizing, s.risk.limits, s.position, s.backtest)
        self.posmgr = PositionManager(s.position)
        self.rng = np.random.default_rng(self.seed)
        self.sim = ExecutionSimulator(s, self.curve, self.amm, self.market, self.rng, latency_scale, failure_scale)
        names = strategies if strategies is not None else list(s.strategy.active)
        overrides = strategy_overrides or {}
        built: list[Strategy] = [x if isinstance(x, Strategy) else build_strategy(x, s, overrides.get(x)) for x in names]
        overlay = build_strategy(s.strategy.exit_overlay, s, overrides.get(s.strategy.exit_overlay)) if s.strategy.exit_overlay else None
        rec = s.backtest.record_signals if record_signals is None else record_signals
        self.runtime = StrategyRuntime(s, built, overlay, self.market, self.features, self.wallets, self.creators, self.rug,
                                       self.portfolio, self.risk, self.sizer, self.posmgr, self.sim, self.discovery, rec)
        self.runtime.allowed_mints = allowed_mints
        self.runtime.log_signals = False  # simulated signals are kept in result.signals, not written to the live log
        self.discovery.verbose = False
        self.strategy_names = [st.name for st in built]
        self.mode = s.backtest.replay_mode
        self._heap: list[tuple[int, int, int, str, Any]] = []
        self._seq = 0
        self._next_sweep = INF
        self._next_equity = INF
        self._next_candle = INF
        self._candle_touched: dict[str, Event] = {}
        self._tick_slot = -1
        self._tick_ts = 0
        self._tick_buf: dict[str, Event] = {}
        self._order_ctx: dict[int, dict[str, Any]] = {}
        self.now = 0
        self.last_slot = 0
        self.n_events = 0

    # ------------------------------------------------------------------ scheduling helpers
    def _push(self, t: int, prio: int, kind: str, payload: Any) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (int(t), prio, self._seq, kind, payload))

    def _submit(self, order: Order, now: int) -> None:
        pos = self.portfolio.position(order.mint)
        st = self.market.get(order.mint)
        self._order_ctx[order.id] = {
            "strategy": order.strategy if order.side is Side.BUY or pos is None else pos.strategy,
            "creator": st.creator if st else None, "sector": st.sector if st else "other",
            "stop_frac": self.sizer.stop_fraction(None) if not order.exit_overrides
            else max(0.01, order.exit_overrides.get("stop_loss_pct", self.s.position.stop_loss_pct) / 100.0),
            "exit_overrides": order.exit_overrides,
        }
        kind, t, payload = self.sim.submit(order, now, self.features.tps)
        if kind == "land":
            self._push(t, _P_LAND, "land", order)
        else:
            self._push(t, _P_RESULT, "result", payload)

    def _book(self, fill: Fill) -> None:
        ctx = self._order_ctx.get(fill.order_id, {})
        self.portfolio.apply_fill(fill, strategy=ctx.get("strategy", fill.strategy), creator=ctx.get("creator"),
                                  sector=ctx.get("sector", "other"), stop_frac=ctx.get("stop_frac", 0.25),
                                  exit_overrides=ctx.get("exit_overrides"))
        failed = fill.status in (OrderStatus.FAILED, OrderStatus.DROPPED, OrderStatus.EXPIRED)
        self.risk.on_fill(fill.filled, failed, fill.slippage_bps, fill.land_ms)

    # ------------------------------------------------------------------ internal event processing
    def _handle(self, t: int, kind: str, payload: Any) -> None:
        if kind == "land":
            order: Order = payload
            fill = self.sim.execute(order, t, self.last_slot)
            self._book(fill)
            self._push(fill.confirm_ms, _P_CONFIRM, "confirm", fill)
        elif kind == "result":
            fill = payload
            self._book(fill)
            self._after_result(fill, t)
        elif kind == "confirm":
            self._after_result(payload, t)
        elif kind == "submit":
            self._submit(payload, t)

    def _after_result(self, fill: Fill, t: int) -> None:
        retry = self.runtime.on_result(fill, t)
        if retry is not None:
            self._push(t + self.s.simulation.retries.backoff_ms, _P_SUBMIT, "submit", retry)

    def _mark_all(self, t: int) -> None:
        for mint, pos in list(self.portfolio.positions.items()):
            if pos.tokens > 0:
                self.portfolio.mark(mint, self.sim.liquidation_value(mint, pos.tokens), self.sim.spot_price(mint), t)

    def _advance(self, t: int) -> None:
        """Process every internal event strictly before ``t`` in time order."""
        while True:
            nh = self._heap[0][0] if self._heap else INF
            nxt = min(nh, self._next_candle, self._next_sweep, self._next_equity)
            if nxt >= t or nxt >= INF:
                return
            self.now = max(self.now, nxt)
            if nh == nxt:
                tt, _, _, kind, payload = heapq.heappop(self._heap)
                self._handle(tt, kind, payload)
            elif self._next_candle == nxt:
                self._candle_close(nxt)
                self._next_candle += self.s.backtest.candle_interval_ms
            elif self._next_sweep == nxt:
                for o in self.runtime.on_sweep(nxt):
                    self._submit(o, nxt)
                self.resolver.advance(nxt, self.market.tokens)
                self._next_sweep += self.s.backtest.sweep_interval_ms
            else:
                self._mark_all(nxt)
                eq = self.portfolio.record_equity(nxt)
                self.risk.on_equity(nxt, eq)
                if self.risk.breakers.flatten_requested:
                    self.risk.breakers.flatten_requested = False
                    self.runtime.flatten = True
                self._next_equity += self.s.backtest.equity_sample_ms

    def _evaluate(self, ev: Event, now: int) -> None:
        st = self.market.get(ev.mint) if ev.mint else None
        if st is None:
            return
        for o in self.runtime.evaluate(ev, st, now):
            self._submit(o, now)

    def _candle_close(self, t: int) -> None:
        touched, self._candle_touched = self._candle_touched, {}
        for ev in touched.values():
            self._evaluate(ev, t)

    def _flush_tick(self) -> None:
        buf, self._tick_buf = self._tick_buf, {}
        for ev in buf.values():
            self._evaluate(ev, self._tick_ts)

    # ------------------------------------------------------------------ market events
    def _on_event(self, ev: Event) -> None:
        t = ev.ts_ms
        self.now = t
        self.last_slot = ev.slot
        st = self.market.on_event(ev)
        kind = ev.kind
        if kind == EventKind.CREATE.value and st is not None:
            self.wallets.on_create(ev)
            self.resolver.on_create(st)
            self.discovery.on_create(st)
        self.features.on_event(ev, st)
        if kind in _TRADE_KINDS and st is not None:
            self.wallets.on_trade(ev, st.created_ms, st.created_slot, st.creator)
            pos = self.portfolio.positions.get(st.mint)
            if pos is not None and pos.tokens > 0:
                self.portfolio.mark(st.mint, self.sim.liquidation_value(st.mint, pos.tokens), self.sim.spot_price(st.mint), t)
        if st is None:
            return
        self.runtime.trading_enabled = self.trade_start_ms is None or t >= self.trade_start_ms
        mode = self.mode
        if mode == "event":
            self._evaluate(ev, t)
        elif mode == "trade":
            if kind in _TRADE_KINDS or self.portfolio.position(st.mint) is not None:
                self._evaluate(ev, t)
        elif mode == "tick":
            self._tick_buf[st.mint] = ev
            self._tick_ts = t
        else:  # candle
            self._candle_touched[st.mint] = ev

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        t0 = time.perf_counter()
        first_ts: int | None = None
        last_ts = 0
        max_events = self.s.backtest.max_events
        stop = False
        for batch in self.source.batches():
            for row in batch.select(list(EVENT_COLUMNS)).iter_rows():
                ev = Event(*row)
                if first_ts is None:
                    first_ts = ev.ts_ms
                    self._next_sweep = first_ts + self.s.backtest.sweep_interval_ms
                    self._next_equity = first_ts
                    if self.mode == "candle":
                        iv = self.s.backtest.candle_interval_ms
                        self._next_candle = first_ts - first_ts % iv + iv
                if self.mode == "tick" and self._tick_buf and ev.slot != self._tick_slot:
                    self._advance(self._tick_ts + 1)
                    self._flush_tick()
                self._tick_slot = ev.slot
                self._advance(ev.ts_ms)
                self._on_event(ev)
                last_ts = ev.ts_ms
                self.n_events += 1
                if max_events is not None and self.n_events >= max_events:
                    stop = True
                    break
            if stop:
                break
        if self.mode == "tick" and self._tick_buf:
            self._advance(self._tick_ts + 1)
            self._flush_tick()
        drain_end = last_ts + 2 * self.s.simulation.failures.blockhash_ttl_ms
        self._advance(drain_end)
        end_ts = max(last_ts, self.now)
        self._final_close(end_ts)
        self._mark_all(end_ts)
        self.portfolio.record_equity(end_ts)
        self.resolver.advance(end_ts, self.market.tokens)
        elapsed = time.perf_counter() - t0
        return self._result(first_ts or 0, end_ts, elapsed)

    def _final_close(self, t: int) -> None:
        """Liquidate remaining positions at the final effective state (flagged as end_of_data)."""
        self.runtime.pending.clear()
        for mint, pos in list(self.portfolio.positions.items()):
            if pos.tokens <= 0:
                continue
            order = Order(id=self.runtime._next_id(), mint=mint, side=Side.SELL, action=Action.EXIT, order_type=OrderType.MARKET,
                          created_ms=t, strategy=pos.strategy, reason="end_of_data", urgency=Urgency.EXIT,
                          token_amount=pos.tokens, compute_units=self.s.priority_fee.compute_unit_limit,
                          priority_micro_lamports=self.s.priority_fee.fixed_micro_lamports)
            order.meta.update(pos_tokens=pos.tokens, rent_paid=pos.rent_paid, quote_avg=0.0)
            self._order_ctx[order.id] = {"strategy": pos.strategy, "creator": pos.creator, "sector": pos.sector, "stop_frac": 0.25}
            fill = self.sim.force_close(order, t, self.last_slot)
            if not fill.filled:
                # untradable at the end of data (e.g. awaiting migration): book an estimated close
                value = self.sim.liquidation_value(mint, pos.tokens)
                fill = Fill(order.id, mint, Side.SELL, Action.EXIT, OrderStatus.FILLED, pos.strategy, "end_of_data_estimated",
                            t, t, t, t, self.last_slot, fill.venue, token_amount=pos.tokens, sol_amount=value,
                            sol_delta=value + pos.rent_paid, rent=-pos.rent_paid,
                            price=value / pos.tokens / 1000.0 if pos.tokens else 0.0, failure="estimated")
            self._book(fill)

    def _result(self, start: int, end: int, elapsed: float) -> BacktestResult:
        pf = self.portfolio
        trades, fills, equity = pf.trades_frame(), pf.fills_frame(), pf.equity_frame()
        b = self.s.backtest
        metrics = compute_metrics(equity, trades, b.initial_capital_sol, b.returns_bar_ms, b.annualization_days, fills)
        metrics["unattributed_costs_sol"] = pf.unattributed_costs / LAMPORTS_PER_SOL
        lat = np.array(self.sim.latencies) if self.sim.latencies else np.empty(0)
        diagnostics = {
            "events_per_second": self.n_events / elapsed if elapsed > 0 else None,
            "sim_counts": dict(self.sim.stats),
            "runtime_counts": dict(self.runtime.counts),
            "risk": self.risk.status(end),
            "breaker_log": list(self.risk.breakers.log),
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p90": float(np.percentile(lat, 90)),
                           "p99": float(np.percentile(lat, 99))} if lat.size else {},
            "tokens_seen": len(self.market.tokens),
            "wallets_tracked": len(self.wallets),
            "outcomes": self.resolver.to_frame().group_by("label").len().to_dicts() if self.resolver.outcomes else [],
        }
        signals = pl.DataFrame(self.runtime.signal_records) if self.runtime.signal_records else pl.DataFrame(
            schema={"ts_ms": pl.Int64, "mint": pl.Utf8, "strategy": pl.Utf8, "action": pl.Utf8, "confidence": pl.Float64,
                    "reason": pl.Utf8, "outcome": pl.Utf8})
        params = {st.name: st.params_dict() for st in self.runtime.strategies}
        res = BacktestResult(
            run_id=self.run_id, strategies=self.strategy_names, config_hash=self.s.fingerprint(), data_hash=self.source.fingerprint(),
            seed=self.seed, start_ms=start, end_ms=end, n_events=self.n_events, elapsed_s=elapsed,
            initial_capital_sol=b.initial_capital_sol, metrics=metrics, trades=trades, fills=fills, equity=equity,
            signals=signals, diagnostics=diagnostics, params=params, synthetic=self.synthetic,
        )
        log.info("backtest finished", extra={"data": {"run_id": self.run_id, "events": self.n_events, "elapsed_s": round(elapsed, 2),
                                                      **{k: v for k, v in res.summary().items() if v is not None}}})
        return res
