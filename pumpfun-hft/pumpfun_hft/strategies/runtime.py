"""Strategy runtime: the single decision layer shared by backtests, paper and live trading.

Per evaluation of a token (driven by the replay engine or the live event loop):

* **open position** -> gather candidate signals from the exit overlay (rug avoidance), the
  position manager (stop / take-profit ladder / trailing / time) and the owning strategy;
  pick the highest-priority action (EXIT > SELL > SCALE_OUT > SCALE_IN); SCALE_IN must pass the
  pyramiding rules. Nothing is sent while a completed curve awaits migration.
* **flat** -> evaluate every active strategy; take the most confident BUY above
  ``sizing.min_confidence``; skip tokens fully exited less than ``strategy.reentry_cooldown_s``
  ago (stops fee-burning churn); apply the overlay veto; size it (with the liquidity cap); apply
  the **cost gate** (expected return must exceed ``cost_gate_multiple`` x the exact round-trip
  cost of that size: fees both ways + own price impact both ways + tx costs); pass the risk
  engine; emit the order.

At most one order per token is in flight (``strategy.one_order_in_flight_per_token``); the
runtime learns about fills only when they are *confirmed* (like a live client), and decides
retries for dropped / expired / slippage-failed transactions.

``QuoteProvider`` abstracts quoting so the same runtime runs on the simulator (with persistent
own-impact) or on live curve state.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from typing import Any, Protocol

from pumpfun_hft.core.types import (
    ACTION_PRIORITY,
    LAMPORTS_PER_SOL,
    Action,
    Event,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Signal,
    Urgency,
)
from pumpfun_hft.features.market import TokenState
from pumpfun_hft.ml.dataset import model_features
from pumpfun_hft.ml.rug_model import TrainedRugModel, rug_features
from pumpfun_hft.strategies.base import Strategy, StrategyContext
from pumpfun_hft.utils.logging import get_logger

sig_log = get_logger("signals")

_RETRY_KEY = {OrderStatus.DROPPED: "dropped", OrderStatus.EXPIRED: "expired", OrderStatus.REJECTED: "rejected"}


class QuoteProvider(Protocol):
    def tradable(self, mint: str) -> bool: ...
    def spot_price(self, mint: str) -> float: ...
    def quote_buy(self, mint: str, budget_lamports: int) -> tuple[int, float]: ...
    def quote_sell(self, mint: str, tokens: int) -> tuple[int, float]: ...
    def liquidation_value(self, mint: str, tokens: int) -> int: ...
    def impact_bps(self, mint: str, budget_lamports: int) -> float: ...


class StrategyRuntime:
    """Turns strategy signals into orders under sizing, risk and cost constraints."""

    def __init__(self, settings: Any, strategies: list[Strategy], overlay: Strategy | None, market: Any, features: Any,
                 wallets: Any, creators: Any, rug_scorer: Any, portfolio: Any, risk: Any, sizer: Any, posmgr: Any,
                 quotes: QuoteProvider, discovery: Any = None, record_signals: bool = True, max_signal_records: int = 200_000,
                 signal_ring: bool = False) -> None:
        self.s = settings
        self.strategies = strategies
        self.by_name = {st.name: st for st in strategies}
        self.overlay = overlay
        if overlay is not None:
            self.by_name.setdefault(overlay.name, overlay)
        self.market = market
        self.features = features
        self.wallets = wallets
        self.creators = creators
        self.rug_scorer = rug_scorer
        self.pf = portfolio
        self.risk = risk
        self.sizer = sizer
        self.posmgr = posmgr
        self.quotes = quotes
        self.discovery = discovery
        self.pending: dict[str, Order] = {}
        self.resting: dict[str, Order] = {}
        self.pending_lamports = 0
        self.closed_returns: list[float] = []
        self.flatten = False
        self.trading_enabled = True
        self.allowed_mints: set[str] | None = None
        self._oid = 0
        self.counts: Counter[str] = Counter()
        self.record_signals = record_signals
        self.max_signal_records = max_signal_records
        # backtests keep the first ``max_signal_records``; live sessions keep the most recent ones (ring buffer)
        self.signal_ring = signal_ring
        self.signal_records: Any = deque(maxlen=max_signal_records) if signal_ring else []
        self.log_signals = True  # submitted signals go to logs/signals.log (backtests keep them in the result instead)
        self.order_meta: dict[int, dict[str, Any]] = {}
        pf_cfg = settings.priority_fee
        self._cu = pf_cfg.compute_unit_limit
        self._micro = pf_cfg.fixed_micro_lamports
        self.priority_override: Any = None  # live: callable(urgency) -> micro-lamports (dynamic estimator)
        self._cooldown_ms = int(settings.strategy.reentry_cooldown_s * 1000)
        self.cooldown_until: dict[str, int] = {}  # mint -> no new entries before this time (after a full exit)

    # ------------------------------------------------------------------ helpers used by StrategyContext
    def creator_score(self, creator: str | None) -> Any:
        return self.creators.score(creator)

    def rug_probability(self, ctx: StrategyContext) -> float:
        x = rug_features(ctx.f, ctx.token, ctx.creator_score)
        try:
            if isinstance(self.rug_scorer, TrainedRugModel):
                # a trained model learned from the full training row (online features + rug_* columns),
                # not just the raw rug features the heuristic uses
                return float(self.rug_scorer.predict({**x, **model_features(ctx.f, ctx.token, ctx.creator_score)}, ctx.now_ms))
            return float(self.rug_scorer.predict(x, ctx.now_ms))
        except Exception:  # noqa: BLE001 - e.g. LookAheadError from a model trained after `now`
            self.counts["rug_model_unavailable"] += 1
            return float(self._heuristic_fallback(x))

    def _heuristic_fallback(self, x: dict[str, float]) -> float:
        from pumpfun_hft.ml.rug_model import HeuristicRugScorer

        return HeuristicRugScorer(self.s.rug_model.heuristic_weights).predict(x)

    def position_return(self, pos: Any) -> float:
        return pos.economic_return(self.quotes.liquidation_value(pos.mint, pos.tokens))

    def round_trip_cost(self, mint: str, size_sol: float) -> float:
        """Exact fractional cost of buying ``size_sol`` and selling it straight back (no price change)."""
        budget = int(size_sol * LAMPORTS_PER_SOL)
        if budget <= 0:
            return 0.0
        tokens, _ = self.quotes.quote_buy(mint, budget)
        if tokens <= 0:
            return 1.0
        net = self.quotes.round_trip_net(mint, budget, tokens) if hasattr(self.quotes, "round_trip_net") else self.quotes.quote_sell(mint, tokens)[0]
        tx = 2 * (self.s.fees.base_fee_lamports_per_signature + self._cu * self._micro // 1_000_000)
        return max(0.0, 1.0 - (net - tx) / budget)

    def _next_id(self) -> int:
        self._oid += 1
        return self._oid

    def _record(self, now: int, mint: str, sig: Signal, outcome: str) -> None:
        self.counts[f"{sig.strategy or '?'}:{sig.action.value}:{outcome}"] += 1
        if self.record_signals and (self.signal_ring or len(self.signal_records) < self.max_signal_records):
            self.signal_records.append({"ts_ms": now, "mint": mint, "strategy": sig.strategy, "action": sig.action.value,
                                        "confidence": round(sig.confidence, 2), "reason": sig.reason, "outcome": outcome})
        if outcome == "submitted" and self.log_signals:
            sig_log.info("signal", extra={"data": {"ts": now, "mint": mint, "strategy": sig.strategy, "action": sig.action.value,
                                                   "confidence": round(sig.confidence, 1), "reason": sig.reason}})

    def _context(self, ev: Event, st: TokenState, now: int) -> StrategyContext | None:
        f = self.features.view(st.mint, now, st)
        if f is None:
            return None
        pos = self.pf.position(st.mint)
        disc = self.discovery.discovered.get(st.mint) if self.discovery is not None else None
        return StrategyContext(self, now, ev, st, f, pos, st.mint in self.pending, self.wallets,
                               self.pf.equity_lamports() / LAMPORTS_PER_SOL, self.pf.exposure_lamports() / LAMPORTS_PER_SOL,
                               self.pf.n_positions, disc)

    # ------------------------------------------------------------------ main entry points
    def evaluate(self, ev: Event, st: TokenState, now: int) -> list[Order]:
        mint = st.mint
        if mint in self.pending and self.s.strategy.one_order_in_flight_per_token:
            return []
        resting = self.resting.get(mint)
        if resting is not None:
            return self._check_resting(resting, now)
        ctx = self._context(ev, st, now)
        if ctx is None:
            return []
        pos = ctx.position
        if pos is not None:
            return self._manage_position(ctx, pos, now)
        # strategies see every event (they may keep internal trackers) even if entries are disabled
        best: Signal | None = None
        for strat in self.strategies:
            if ev.kind not in strat.entry_events:
                continue
            sig = strat.generate_signal(ctx)
            if sig.action is Action.BUY:
                sig.strategy = sig.strategy or strat.name
                if best is None or sig.confidence > best.confidence:
                    best = sig
        if best is None:
            return []
        if not self.trading_enabled or self.flatten or (self.allowed_mints is not None and mint not in self.allowed_mints):
            self._record(now, mint, best, "trading_disabled")
            return []
        if st.venue is None or not self.quotes.tradable(mint):
            self._record(now, mint, best, "not_tradable")
            return []
        if best.confidence < self.s.sizing.min_confidence:
            self._record(now, mint, best, "low_confidence")
            return []
        until = self.cooldown_until.get(mint)
        if until is not None:
            if now < until:
                self._record(now, mint, best, "reentry_cooldown")
                return []
            del self.cooldown_until[mint]
        if self.overlay is not None:
            reason = self.overlay.veto(ctx)
            if reason:
                self._record(now, mint, best, "vetoed")
                return []
        return self._entry(ctx, best, now)

    def _manage_position(self, ctx: StrategyContext, pos: Any, now: int) -> list[Order]:
        st = ctx.token
        if st.venue is None or not self.quotes.tradable(st.mint):
            return []  # completed curve awaiting migration: cannot sell yet
        value = self.quotes.liquidation_value(st.mint, pos.tokens)
        self.pf.mark(st.mint, value, self.quotes.spot_price(st.mint), now)
        cands: list[Signal] = []
        if self.flatten:
            cands.append(Signal(Action.EXIT, 100, "risk flatten (drawdown breaker)", "risk", urgency=Urgency.EXIT))
        if self.overlay is not None:
            s = self.overlay.generate_signal(ctx)
            if not s.is_hold:
                cands.append(s)
        s = self.posmgr.check(pos, value, ctx.f.price, now)
        if s is not None:
            cands.append(s)
        owner = self.by_name.get(pos.strategy)
        if owner is not None and owner is not self.overlay:
            s = owner.generate_signal(ctx)
            if s.action in (Action.SELL, Action.EXIT, Action.SCALE_OUT, Action.SCALE_IN):
                s.strategy = s.strategy or owner.name
                cands.append(s)
        if not cands:
            return []
        best = max(cands, key=lambda x: (ACTION_PRIORITY[x.action], x.confidence))
        if best.action is Action.SCALE_IN:
            if not self.posmgr.allow_scale_in(pos, value, best.confidence) or self.flatten:
                return []
            best = replace(best, size_sol=(pos.total_cost / LAMPORTS_PER_SOL) * self.posmgr.scale_in_fraction())
            return self._entry(ctx, best, now, action=Action.SCALE_IN)
        frac = 1.0 if best.action in (Action.EXIT, Action.SELL) and best.size_frac is None else (best.size_frac or 1.0)
        tokens = pos.tokens if frac >= 1.0 else int(pos.tokens * frac)
        if tokens <= 0:
            return []
        order = self._make_order(st.mint, Side.SELL, best, now, token_amount=tokens)
        order.meta["tp"] = best.action is Action.SCALE_OUT and best.strategy == "position_manager"
        self.risk.check_exit(st.mint, now)
        pos.exit_reason = best.reason
        self._record(now, st.mint, best, "submitted")
        return [self._activate(order)]

    def _entry(self, ctx: StrategyContext, sig: Signal, now: int, action: Action = Action.BUY) -> list[Order]:
        st = ctx.token
        mint = st.mint
        pos = ctx.position
        vol = max(ctx.f.atr_pct, ctx.f.rv_medium)
        lamports = self.sizer.size(
            sig, equity_lamports=self.pf.equity_lamports(), cash_lamports=self.pf.cash - self.pending_lamports,
            exposure_lamports=self.pf.cost_exposure_lamports() + self.pending_lamports,
            token_exposure_lamports=pos.cost_lamports if pos is not None else 0, vol=vol,
            closed_returns=self.closed_returns, impact_bps_fn=lambda b: self.quotes.impact_bps(mint, b))
        if lamports <= 0:
            self._record(now, mint, sig, "size_zero")
            return []
        if sig.expected_return is not None:
            cost = self.round_trip_cost(mint, lamports / LAMPORTS_PER_SOL)
            if sig.expected_return < self.s.strategy.cost_gate_multiple * cost:
                self._record(now, mint, sig, "cost_gate")
                return []
        pend_creator = sum(o.sol_budget for o in self.pending.values() if o.meta.get("creator") == st.creator)
        pend_sector = sum(o.sol_budget for o in self.pending.values() if o.meta.get("sector") == st.sector)
        dec = self.risk.check_entry(mint, st.creator, st.sector, lamports, now, self.pending_lamports, pend_creator, pend_sector)
        if not dec.ok:
            self._record(now, mint, sig, f"risk:{dec.reason.split(':')[0]}")
            return []
        order = self._make_order(mint, Side.BUY, sig, now, sol_budget=dec.lamports, action=action)
        order.meta.update(creator=st.creator, sector=st.sector)
        self._record(now, mint, sig, "submitted")
        if order.order_type is OrderType.LIMIT:
            self.resting[mint] = order
            order.status = OrderStatus.RESTING
            order.expires_ms = now + self.s.backtest.limit_ttl_ms
            return self._check_resting(order, now)
        return [self._activate(order)]

    def _make_order(self, mint: str, side: Side, sig: Signal, now: int, *, sol_budget: int = 0, token_amount: int = 0,
                    action: Action | None = None) -> Order:
        urgency = sig.urgency
        slip_cfg = self.s.slippage
        if side is Side.BUY:
            otype = sig.order_type or OrderType(self.s.backtest.default_order_type)
            slip = slip_cfg.buy_bps
        else:
            otype = sig.order_type or OrderType.MARKET
            slip = slip_cfg.exit_bps if urgency is Urgency.EXIT else slip_cfg.sell_bps
        mult = self.s.priority_fee.urgency_multiplier.get(urgency.value, 1.0)
        micro = int(self.priority_override(urgency) if self.priority_override else self._micro * mult)
        micro = max(self.s.priority_fee.min_micro_lamports, min(self.s.priority_fee.max_micro_lamports, micro))
        spot = self.quotes.spot_price(mint)
        order = Order(
            id=self._next_id(), mint=mint, side=side, action=action or sig.action, order_type=otype, created_ms=now,
            strategy=sig.strategy, reason=sig.reason, urgency=urgency, sol_budget=sol_budget, token_amount=token_amount,
            slippage_bps=slip, priority_micro_lamports=micro, compute_units=self._cu,
            use_jito=self.s.jito.enabled, jito_tip_lamports=self.s.jito.tip_lamports if self.s.jito.enabled else 0,
            decision_price=spot, confidence=sig.confidence, exit_overrides=sig.exit_overrides,
        )
        off = self.s.backtest.limit_offset_bps / 1e4
        if otype in (OrderType.LIMIT, OrderType.IOC, OrderType.FOK):
            order.limit_price = sig.limit_price or (spot * (1 + off) if side is Side.BUY else spot * (1 - off))
        self._requote(order)
        return order

    def _requote(self, order: Order) -> None:
        if order.side is Side.BUY:
            order.quote_tokens, avg = self.quotes.quote_buy(order.mint, order.sol_budget)
            order.meta["quote_avg"] = avg
        else:
            order.quote_sol, avg = self.quotes.quote_sell(order.mint, order.token_amount)
            order.meta["quote_avg"] = avg
        order.decision_price = self.quotes.spot_price(order.mint)

    def _activate(self, order: Order) -> Order:
        order.status = OrderStatus.SUBMITTED
        pos = self.pf.position(order.mint)
        order.meta["pos_tokens"] = pos.tokens if pos is not None else 0
        order.meta["rent_paid"] = pos.rent_paid if pos is not None else 0
        self.pending[order.mint] = order
        self.pending_lamports += order.sol_budget
        return order

    def _check_resting(self, order: Order, now: int) -> list[Order]:
        if order.expires_ms is not None and now >= order.expires_ms:
            del self.resting[order.mint]
            self.counts["limit_expired"] += 1
            return []
        self._requote(order)
        avg = order.meta.get("quote_avg") or 0.0
        marketable = (avg <= (order.limit_price or 0.0)) if order.side is Side.BUY else (avg >= (order.limit_price or 0.0))
        if not marketable:
            return []
        del self.resting[order.mint]
        order.created_ms = now
        return [self._activate(order)]

    def on_sweep(self, now: int) -> list[Order]:
        """Periodic checks for positions whose tokens produced no events (time exits, flatten)."""
        out: list[Order] = []
        for mint, pos in list(self.pf.positions.items()):
            if pos.tokens <= 0 or mint in self.pending:
                continue
            st = self.market.get(mint)
            if st is None or st.venue is None or not self.quotes.tradable(mint):
                continue
            value = self.quotes.liquidation_value(mint, pos.tokens)
            self.pf.mark(mint, value, self.quotes.spot_price(mint), now)
            sig = Signal(Action.EXIT, 100, "risk flatten (drawdown breaker)", "risk", urgency=Urgency.EXIT) if self.flatten \
                else self.posmgr.check(pos, value, self.quotes.spot_price(mint), now)
            if sig is None or sig.action not in (Action.EXIT, Action.SCALE_OUT):
                continue
            frac = sig.size_frac if sig.action is Action.SCALE_OUT and sig.size_frac else 1.0
            tokens = pos.tokens if frac >= 1.0 else int(pos.tokens * frac)
            if tokens <= 0:
                continue
            order = self._make_order(mint, Side.SELL, sig, now, token_amount=tokens)
            order.meta["tp"] = sig.action is Action.SCALE_OUT
            pos.exit_reason = sig.reason
            self._record(now, mint, sig, "submitted")
            out.append(self._activate(order))
        for mint, order in list(self.resting.items()):
            if order.expires_ms is not None and now >= order.expires_ms:
                del self.resting[mint]
                self.counts["limit_expired"] += 1
        if self.flatten and not any(p.tokens > 0 for p in self.pf.positions.values()):
            self.flatten = False
        return out

    def on_result(self, fill: Fill, now: int) -> Order | None:
        """Confirmed fill or terminal failure of the in-flight order; may return a retry order."""
        order = self.pending.pop(fill.mint, None)
        if order is None or order.id != fill.order_id:
            if order is not None:  # stale result for an older order: keep the current one pending
                self.pending[fill.mint] = order
            return None
        self.pending_lamports -= order.sol_budget
        owner = self.by_name.get(order.strategy)
        if fill.filled:
            self.counts[f"fill:{fill.status.value}"] += 1
            pos = self.pf.positions.get(fill.mint)
            if order.meta.get("tp") and pos is not None:
                pos.tp_hits += 1
            if fill.side is Side.SELL and (pos is None or pos.tokens == 0):
                if self._cooldown_ms > 0:
                    self.cooldown_until[fill.mint] = now + self._cooldown_ms
                if self.pf.trades and self.pf.trades[-1].mint == fill.mint:
                    self.closed_returns.append(self.pf.trades[-1].ret)
            if owner is not None:
                owner.on_fill(fill, None)
            return None
        self.counts[f"fail:{fill.status.value}:{fill.failure}"] += 1
        key = _RETRY_KEY.get(fill.status)
        if fill.status is OrderStatus.FAILED:
            key = "slippage" if fill.failure == "slippage" else "failed"
        retry = self.s.simulation.retries
        if key is None or key not in retry.retry_on or order.attempt >= retry.max_retries:
            return None
        if order.side is Side.SELL and self.pf.position(order.mint) is None:
            return None
        new = replace(order, id=self._next_id(), attempt=order.attempt + 1, created_ms=now, parent_id=order.id,
                      status=OrderStatus.NEW, meta=dict(order.meta))
        if key == "slippage":
            new.slippage_bps = min(self.s.slippage.max_bps, order.slippage_bps + self.s.slippage.retry_widen_bps)
        new.priority_micro_lamports = min(self.s.priority_fee.max_micro_lamports,
                                          int(order.priority_micro_lamports * self.s.priority_fee.retry_bump_factor))
        if new.side is Side.SELL:
            pos = self.pf.position(new.mint)
            new.token_amount = min(new.token_amount, pos.tokens) if pos is not None else 0
            if new.token_amount <= 0:
                return None
        if not self.quotes.tradable(new.mint):
            return None
        self._requote(new)
        self.counts["retries"] += 1
        return self._activate(new)
