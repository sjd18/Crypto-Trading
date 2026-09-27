"""Execution simulator: the path from a decision to an on-chain fill.

Timeline of an order decided at ``t``::

    t --decision+network+RPC--> submit --inclusion (priority-fee / Jito dependent)--> land --confirmation--> confirm
                                  |                                              |
                        outage / rate limit -> reject or delay         execute against the *effective*
                        drop (never lands) -> detected at blockhash    state at ``land`` (every market
                        expiry, no fee                                  event with ts <= land applied)

Realism features
    * lognormal latency components, latency spikes, priority-fee speed-up, Jito bundles
    * drops (no fee), landed failures (base + priority fee charged), congestion multipliers,
      blockhash expiry, Jito bundle failures (atomic: no fee, no tip)
    * API outages (Poisson windows) and client rate limits (token bucket in simulated time)
    * exact curve / AMM math with the fee tier (or the fee bps recorded on-chain) at landing time
    * slippage-tolerance failures mirroring on-chain ``min_tokens_out`` / ``max_sol_cost`` / ``min_sol_output``
    * order types: MARKET, LIMIT (FOK at the limit when it lands), IOC (partial up to the limit), FOK
    * partial fills when the curve sells out, liquidity exhaustion on sells, no trading while a
      completed curve awaits migration
    * **persistent own impact**: our net tokens shift the historical reserves along the curve's
      constant-product hyperbola, so entries move the price we later exit into
    * costs: base fee, priority fee (CU x micro-lamports), Jito tip, platform fee, ATA rent
      (refunded on the closing sell)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pumpfun_hft.api.rate_limit import TokenBucket
from pumpfun_hft.core.amm import ConstantProductAmm, PoolState, shift_pool
from pumpfun_hft.core.curve import PRICE_SCALE, BondingCurve, CurveState, FeeBps, fee_amount, shift_state
from pumpfun_hft.core.types import (
    BPS,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Venue,
)
from pumpfun_hft.features.market import MarketState, TokenState


class ExecutionSimulator:
    """Latency / failure / cost model plus exact venue math. Also a ``QuoteProvider``."""

    def __init__(self, settings: Any, curve: BondingCurve, amm: ConstantProductAmm, market: MarketState,
                 rng: np.random.Generator, latency_scale: float = 1.0, failure_scale: float = 1.0) -> None:
        self.s = settings
        self.sim = settings.simulation
        self.curve = curve
        self.amm = amm
        self.market = market
        self.rng = rng
        self.latency_scale = latency_scale
        self.failure_scale = failure_scale
        self.impact_curve: dict[str, int] = {}
        self.impact_pool: dict[str, int] = {}
        rl = self.sim.rate_limit
        self.bucket = TokenBucket(rl.max_orders_per_s, rl.burst, now_s=0.0)
        self._outage_end = -1
        self._next_outage: float | None = None
        self.stats: dict[str, int] = {}
        self.latencies: list[float] = []
        pf = settings.priority_fee
        self._prio_min, self._prio_max = pf.min_micro_lamports, pf.max_micro_lamports
        self.persistent = self.sim.impact_model == "persistent"
        self.platform_bps = settings.effective_platform_fee_bps

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    # ------------------------------------------------------------------ effective state (own impact)
    def eff_curve(self, st: TokenState) -> CurveState:
        n = self.impact_curve.get(st.mint, 0) if self.persistent else 0
        return shift_state(st.curve, n) if n else st.curve

    def eff_pool(self, st: TokenState) -> PoolState | None:
        if st.pool is None:
            return None
        n = self.impact_pool.get(st.mint, 0) if self.persistent else 0
        return shift_pool(st.pool, n) if n else st.pool

    def _fees_curve(self, st: TokenState, eff: CurveState) -> FeeBps:
        if self.s.fees.source == "observed_if_available" and st.fee_bps is not None:
            return st.fee_bps
        return self.curve.fees_for(eff)

    def _fees_pool(self, st: TokenState, pool: PoolState) -> FeeBps:
        if self.s.fees.source == "observed_if_available" and st.amm_fee_bps is not None:
            return st.amm_fee_bps
        return self.amm.fees_for(pool)

    # ------------------------------------------------------------------ QuoteProvider
    def tradable(self, mint: str) -> bool:
        st = self.market.get(mint)
        return st is not None and st.venue is not None

    def spot_price(self, mint: str) -> float:
        st = self.market.get(mint)
        if st is None:
            return 0.0
        if st.venue is Venue.AMM:
            p = self.eff_pool(st)
            return p.price if p else 0.0
        return self.eff_curve(st).price

    def _platform(self, lamports: int) -> int:
        bps = self.platform_bps
        return fee_amount(lamports, bps) if bps else 0

    def quote_buy(self, mint: str, budget_lamports: int) -> tuple[int, float]:
        st = self.market.get(mint)
        if st is None or st.venue is None or budget_lamports <= 0:
            return 0, 0.0
        platform = self._platform(budget_lamports)
        budget = budget_lamports - platform
        # all-in average price (venue fees + router platform fee), comparable with Fill.price
        if st.venue is Venue.AMM:
            pool = self.eff_pool(st)
            qa = self.amm.buy_base_for_quote(pool, budget, self._fees_pool(st, pool))  # type: ignore[arg-type]
            return qa.base_out, ((qa.total + platform) / qa.base_out * PRICE_SCALE) if qa.base_out else 0.0
        eff = self.eff_curve(st)
        q = self.curve.buy_with_budget(eff, budget, self._fees_curve(st, eff))
        return q.tokens, ((q.total + platform) / q.tokens * PRICE_SCALE) if q.tokens else 0.0

    def quote_sell(self, mint: str, tokens: int) -> tuple[int, float]:
        st = self.market.get(mint)
        if st is None or st.venue is None or tokens <= 0:
            return 0, 0.0
        if st.venue is Venue.AMM:
            pool = self.eff_pool(st)
            q = self.amm.sell_quote_for_base(pool, tokens, self._fees_pool(st, pool))  # type: ignore[arg-type]
            net = q.net - self._platform(q.net)
        else:
            eff = self.eff_curve(st)
            q = self.curve.sell_proceeds_for_tokens(eff, tokens, self._fees_curve(st, eff))
            net = min(q.net, eff.r_sol) - self._platform(q.net)
        net = max(net, 0)
        return net, (net / tokens / 1000.0) if tokens else 0.0  # SOL per whole token

    def round_trip_net(self, mint: str, budget_lamports: int, tokens: int) -> int:
        """Net lamports from buying with ``budget`` then immediately selling the tokens back."""
        st = self.market.get(mint)
        if st is None or st.venue is None:
            return 0
        budget = budget_lamports - self._platform(budget_lamports)
        if st.venue is Venue.AMM:
            pool = self.eff_pool(st)
            f = self._fees_pool(st, pool)  # type: ignore[arg-type]
            qb = self.amm.buy_base_for_quote(pool, budget, f)  # type: ignore[arg-type]
            after = self.amm.apply_buy(pool, qb)  # type: ignore[arg-type]
            qs = self.amm.sell_quote_for_base(after, qb.base_out, f)
            return qs.net - self._platform(qs.net)
        eff = self.eff_curve(st)
        f = self._fees_curve(st, eff)
        qb = self.curve.buy_with_budget(eff, budget, f)
        after = self.curve.apply_buy(eff, qb)
        qs = self.curve.sell_proceeds_for_tokens(after, qb.tokens, f)
        return qs.net - self._platform(qs.net)

    def liquidation_value(self, mint: str, tokens: int) -> int:
        """Net SOL from selling ``tokens`` now (for a completed curve awaiting migration: at the last curve state)."""
        st = self.market.get(mint)
        if st is None or tokens <= 0:
            return 0
        if st.venue is Venue.AMM:
            return self.quote_sell(mint, tokens)[0]
        eff = self.eff_curve(st)
        q = self.curve.sell_proceeds_for_tokens(eff, tokens, self._fees_curve(st, eff))
        return max(0, min(q.net, eff.r_sol))

    def impact_bps(self, mint: str, budget_lamports: int) -> float:
        st = self.market.get(mint)
        if st is None or st.venue is None or budget_lamports <= 0:
            return 0.0
        spot = self.spot_price(mint)
        tokens, avg = self.quote_buy(mint, budget_lamports)
        if tokens <= 0 or spot <= 0:
            return float("inf")
        return (avg / spot - 1.0) * BPS

    # ------------------------------------------------------------------ latency & failures
    def _ln(self, spec: Any) -> float:
        return spec.median_ms * math.exp(spec.sigma * self.rng.standard_normal())

    def _outage(self, t: int) -> int | None:
        """End time of an outage covering ``t`` (outage windows are generated lazily, Poisson)."""
        o = self.sim.outages
        if o.rate_per_hour <= 0:
            return None
        if self._next_outage is None:
            self._next_outage = t + self.rng.exponential(3_600_000 / o.rate_per_hour)
        while t >= self._next_outage:
            start = self._next_outage
            self._outage_end = int(start + self.rng.exponential(o.mean_duration_s * 1000))
            self._next_outage = self._outage_end + self.rng.exponential(3_600_000 / o.rate_per_hour)
        return self._outage_end if t < self._outage_end else None

    def _reject(self, order: Order, t: int, reason: str) -> Fill:
        self._count(f"rejected:{reason}")
        return Fill(order.id, order.mint, order.side, order.action, OrderStatus.REJECTED, order.strategy, order.reason,
                    order.created_ms, t, t, t, 0, Venue.CURVE, failure=reason, attempt=order.attempt,
                    decision_price=order.decision_price)

    def submit(self, order: Order, now: int, activity_eps: float) -> tuple[str, int, Any]:
        """Schedule an order: ('land', land_ms, meta) | ('result', t, Fill) | ('delay', t, None)."""
        sim = self.sim
        t_submit = now
        end = self._outage(now)
        if end is not None:
            if sim.outages.behavior == "reject":
                return "result", now, self._reject(order, now, "api_outage")
            t_submit = end
        if not self.bucket.try_acquire(t_submit / 1000.0):
            if sim.rate_limit.behavior == "reject":
                return "result", t_submit, self._reject(order, t_submit, "rate_limited")
            t_submit = int(self.bucket.next_available(t_submit / 1000.0) * 1000) + 1
            self.bucket.try_acquire(t_submit / 1000.0)
        lat = sim.latency
        pre = self._ln(lat.decision_ms) + self._ln(lat.network_ms) + self._ln(lat.rpc_ms)
        level = (order.priority_micro_lamports - self._prio_min) / max(1, self._prio_max - self._prio_min)
        level = min(1.0, max(0.0, level))
        incl = self._ln(lat.jito_inclusion_ms if order.use_jito else lat.inclusion_ms) * (1.0 - lat.priority_speedup * level)
        total = pre + incl
        if self.rng.random() < lat.spike_prob:
            total *= lat.spike_multiplier
        total *= self.latency_scale
        land = t_submit + int(total)
        congested = activity_eps > sim.failures.congestion_events_per_s
        mult = (sim.failures.congestion_multiplier if congested else 1.0) * self.failure_scale
        if order.use_jito:
            if self.rng.random() < min(1.0, sim.failures.jito_bundle_fail_prob * mult):
                self._count("jito_bundle_failed")
                fill = self._terminal(order, land, OrderStatus.DROPPED, "bundle_not_landed", t_submit)
                return "result", land, fill
        elif self.rng.random() < min(1.0, sim.failures.drop_prob * mult):
            self._count("dropped")
            detect = t_submit + sim.failures.blockhash_ttl_ms
            return "result", detect, self._terminal(order, detect, OrderStatus.DROPPED, "not_landed", t_submit)
        if land - t_submit > sim.failures.blockhash_ttl_ms:
            self._count("expired")
            detect = t_submit + sim.failures.blockhash_ttl_ms
            return "result", detect, self._terminal(order, detect, OrderStatus.EXPIRED, "blockhash_expired", t_submit)
        order.meta["submit_ms"] = t_submit
        order.meta["landed_fail"] = self.rng.random() < min(1.0, sim.failures.landed_fail_prob * mult)
        order.meta["confirm_delay"] = int(self._ln(lat.confirmation_ms) * self.latency_scale)
        return "land", land, None

    def _terminal(self, order: Order, t: int, status: OrderStatus, reason: str, submit_ms: int) -> Fill:
        return Fill(order.id, order.mint, order.side, order.action, status, order.strategy, order.reason, order.created_ms,
                    submit_ms, t, t, 0, Venue.CURVE, failure=reason, attempt=order.attempt, decision_price=order.decision_price,
                    latency_ms=float(t - order.created_ms))

    # ------------------------------------------------------------------ execution at landing
    def _tx_costs(self, order: Order, success: bool) -> tuple[int, int, int]:
        base = self.s.fees.base_fee_lamports_per_signature
        prio = order.compute_units * order.priority_micro_lamports // 1_000_000
        tip = order.jito_tip_lamports if (order.use_jito and success) else 0
        return base, prio, tip

    def execute(self, order: Order, land_ms: int, slot: int) -> Fill:  # noqa: C901, PLR0911, PLR0912, PLR0915
        """Execute a landed order against the effective state at ``land_ms``."""
        st = self.market.get(order.mint)
        submit_ms = int(order.meta.get("submit_ms", order.created_ms))
        confirm = land_ms + int(order.meta.get("confirm_delay", 0))
        base_fill = dict(order_id=order.id, mint=order.mint, side=order.side, action=order.action, strategy=order.strategy,
                         reason=order.reason, decision_ms=order.created_ms, submit_ms=submit_ms, land_ms=land_ms,
                         confirm_ms=confirm, slot=slot, attempt=order.attempt, decision_price=order.decision_price,
                         latency_ms=float(land_ms - order.created_ms))

        def failed(reason: str, venue: Venue) -> Fill:
            self._count(f"failed:{reason}")
            if order.use_jito:  # atomic bundle: a failing tx is simply not included
                return Fill(status=OrderStatus.DROPPED, venue=venue, failure=reason, **base_fill)
            base, prio, _ = self._tx_costs(order, False)
            return Fill(status=OrderStatus.FAILED, venue=venue, network_fee=base, priority_fee=prio,
                        sol_delta=-(base + prio), failure=reason, **base_fill)

        if st is None or st.venue is None:
            return failed("curve_complete_awaiting_migration" if st is not None else "unknown_token", Venue.CURVE)
        venue = st.venue
        if order.meta.get("landed_fail"):
            return failed("tx_error", venue)
        quote_avg = float(order.meta.get("quote_avg") or 0.0)
        status = OrderStatus.FILLED
        if order.side is Side.BUY:
            platform = self._platform(order.sol_budget)
            budget = order.sol_budget - platform
            if venue is Venue.CURVE:
                eff = self.eff_curve(st)
                f = self._fees_curve(st, eff)
                if order.order_type is OrderType.IOC and order.limit_price:
                    t_lim = self.curve.max_buy_tokens_at_avg_price(eff, order.limit_price, f)
                    t_bud = self.curve.buy_tokens_for_sol(eff, budget, f)
                    tokens = min(t_lim, t_bud)
                    if tokens <= 0:
                        return failed("ioc_no_liquidity_at_limit", venue)
                    q = self.curve.buy_cost_for_tokens(eff, tokens, f)
                    if tokens < t_bud:
                        status = OrderStatus.PARTIAL
                elif self.s.fees.buy_instruction == "buy" and order.order_type is OrderType.MARKET:
                    want = order.quote_tokens
                    q = self.curve.buy_cost_for_tokens(eff, want, f)
                    if q.total > budget * (1 + order.slippage_bps / BPS):
                        return failed("slippage", venue)
                    if q.tokens < want:
                        status = OrderStatus.PARTIAL
                else:
                    q = self.curve.buy_with_budget(eff, budget, f)
                    if q.tokens <= 0:
                        return failed("no_liquidity", venue)
                    if order.order_type in (OrderType.FOK, OrderType.LIMIT) and order.limit_price:
                        if q.avg_price > order.limit_price:
                            return failed("limit_not_met", venue)
                    elif q.tokens < order.quote_tokens * (1 - order.slippage_bps / BPS):
                        return failed("slippage", venue)
                    if q.tokens == eff.r_tok and q.total < budget * 0.99:
                        status = OrderStatus.PARTIAL  # the curve sold out
                self.impact_curve[order.mint] = self.impact_curve.get(order.mint, 0) + q.tokens
                tokens, sol_leg, pfee, cfee, lpfee = q.tokens, q.sol_curve, q.protocol_fee, q.creator_fee, 0
            else:
                pool = self.eff_pool(st)
                assert pool is not None
                f = self._fees_pool(st, pool)
                qa = self.amm.buy_base_for_quote(pool, budget, f)
                if qa.base_out <= 0:
                    return failed("no_liquidity", venue)
                if order.order_type in (OrderType.FOK, OrderType.LIMIT, OrderType.IOC) and order.limit_price:
                    if qa.avg_price > order.limit_price:
                        return failed("limit_not_met", venue)
                elif qa.base_out < order.quote_tokens * (1 - order.slippage_bps / BPS):
                    return failed("slippage", venue)
                self.impact_pool[order.mint] = self.impact_pool.get(order.mint, 0) + qa.base_out
                tokens, sol_leg, pfee, cfee, lpfee = qa.base_out, qa.quote_in, qa.protocol_fee, qa.creator_fee, qa.lp_fee
            base, prio, tip = self._tx_costs(order, True)
            rent = self.s.fees.token_account_rent_lamports if (self.s.fees.charge_token_account_rent and not order.meta.get("pos_tokens")) else 0
            spent = sol_leg + pfee + cfee + lpfee + platform
            price = spent / tokens / 1000.0
            slip = (price / quote_avg - 1.0) * BPS if quote_avg > 0 else 0.0
            self._count(f"filled:{status.value}")
            self.latencies.append(float(land_ms - order.created_ms))
            return Fill(status=status, venue=venue, token_amount=tokens, sol_amount=sol_leg, protocol_fee=pfee, creator_fee=cfee,
                        lp_fee=lpfee, platform_fee=platform, network_fee=base, priority_fee=prio, jito_tip=tip, rent=rent,
                        sol_delta=-(spent + base + prio + tip + rent), price=price, slippage_bps=slip, **base_fill)
        # ---- sells
        tokens = order.token_amount
        if venue is Venue.CURVE:
            eff = self.eff_curve(st)
            f = self._fees_curve(st, eff)
            if order.order_type is OrderType.IOC and order.limit_price:
                tokens = self.curve.max_sell_tokens_at_avg_price(eff, order.limit_price, tokens, f)
                if tokens <= 0:
                    return failed("ioc_no_liquidity_at_limit", venue)
                if tokens < order.token_amount:
                    status = OrderStatus.PARTIAL
            q = self.curve.sell_proceeds_for_tokens(eff, tokens, f)
            if q.sol_curve > eff.r_sol:  # liquidity exhaustion: sell only what the curve can pay for
                lo, hi = 0, tokens
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.curve.sell_proceeds_for_tokens(eff, mid, f).sol_curve <= eff.r_sol:
                        lo = mid
                    else:
                        hi = mid - 1
                tokens = lo
                if tokens <= 0:
                    return failed("no_liquidity", venue)
                q = self.curve.sell_proceeds_for_tokens(eff, tokens, f)
                status = OrderStatus.PARTIAL
            net = q.net
            if order.order_type in (OrderType.FOK, OrderType.LIMIT) and order.limit_price:
                if q.avg_price < order.limit_price:
                    return failed("limit_not_met", venue)
            elif order.order_type is OrderType.MARKET and net < order.quote_sol * tokens / max(1, order.token_amount) * (1 - order.slippage_bps / BPS):
                return failed("slippage", venue)
            self.impact_curve[order.mint] = self.impact_curve.get(order.mint, 0) - tokens
            sol_leg, pfee, cfee, lpfee = q.sol_curve, q.protocol_fee, q.creator_fee, 0
        else:
            pool = self.eff_pool(st)
            assert pool is not None
            f = self._fees_pool(st, pool)
            qa = self.amm.sell_quote_for_base(pool, tokens, f)
            net = qa.net
            if qa.quote_out <= 0:
                return failed("no_liquidity", venue)
            if order.order_type in (OrderType.FOK, OrderType.LIMIT, OrderType.IOC) and order.limit_price:
                if qa.avg_price < order.limit_price:
                    return failed("limit_not_met", venue)
            elif net < order.quote_sol * (1 - order.slippage_bps / BPS):
                return failed("slippage", venue)
            self.impact_pool[order.mint] = self.impact_pool.get(order.mint, 0) - tokens
            sol_leg, pfee, cfee, lpfee = qa.quote_out, qa.protocol_fee, qa.creator_fee, qa.lp_fee
        platform = self._platform(net)
        base, prio, tip = self._tx_costs(order, True)
        closing = tokens >= int(order.meta.get("pos_tokens", 0)) and self.s.fees.close_token_account_on_exit
        refund = int(order.meta.get("rent_paid", 0)) if closing else 0
        received = net - platform
        price = received / tokens / 1000.0 if tokens else 0.0
        slip = (1.0 - price / quote_avg) * BPS if quote_avg > 0 else 0.0
        self._count(f"filled:{status.value}")
        self.latencies.append(float(land_ms - order.created_ms))
        return Fill(status=status, venue=venue, token_amount=tokens, sol_amount=sol_leg, protocol_fee=pfee, creator_fee=cfee,
                    lp_fee=lpfee, platform_fee=platform, network_fee=base, priority_fee=prio, jito_tip=tip, rent=-refund,
                    sol_delta=received - base - prio - tip + refund, price=price, slippage_bps=slip, **base_fill)

    def force_close(self, order: Order, now: int, slot: int) -> Fill:
        """End-of-data liquidation: immediate market sell at the final effective state (fees charged)."""
        order.meta["submit_ms"] = now
        order.meta["landed_fail"] = False
        order.meta["confirm_delay"] = 0
        order.slippage_bps = BPS  # accept any price
        fill = self.execute(order, now, slot)
        if fill.status in (OrderStatus.FAILED, OrderStatus.DROPPED):
            # untradable (e.g. awaiting migration at the end of data): book at zero proceeds
            fill.failure = f"end_of_data_unrealizable:{fill.failure}"
        return fill
