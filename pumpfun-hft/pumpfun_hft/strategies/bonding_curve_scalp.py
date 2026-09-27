"""Bonding-curve scalping: buy transient sell-driven dips on otherwise healthy curves.

On a bonding curve the price is a deterministic function of reserves, so short-lived dips
are pure order-flow shocks. The strategy buys when a sell burst pushes price well below the
short-window high while medium-window flow is still net positive, targeting a partial
reversion. With ~2.5 % round-trip fees the runtime's cost gate rejects dips that are too
shallow to pay for themselves. Tight per-position exits come through ``exit_overrides``.
"""

from __future__ import annotations

from pumpfun_hft.core.types import Action, Signal, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class BondingCurveScalp(Strategy):
    name = "bonding_curve_scalp"

    class Params(StrategyParams):
        min_age_s: float
        max_age_s: float
        min_progress_pct: float
        max_progress_pct: float
        dip_pct: float
        min_medium_imbalance: float
        max_rug_prob: float
        target_pct: float
        stop_pct: float
        max_hold_s: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f = self.p, ctx.f
        if ctx.has_position or not ctx.is_trade or ctx.token.venue is not Venue.CURVE:
            return hold()
        if ctx.event.is_buy:  # only react to sells (the dip)
            return hold()
        if not (p.min_age_s <= ctx.age_s <= p.max_age_s):
            return hold()
        prog = f.progress_pct
        if not (p.min_progress_pct <= prog <= p.max_progress_pct):
            return hold()
        high = f.high_short
        price = f.price
        dip = 100.0 * (1.0 - price / high) if high > 0 else 0.0
        if dip < p.dip_pct or f.imbalance_medium < p.min_medium_imbalance:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        expected = min(p.target_pct / 100.0, high / price - 1.0)
        conf = 50.0 + 40.0 * min(1.0, dip / (2.0 * p.dip_pct)) + 10.0 * max(0.0, f.imbalance_medium)
        return Signal(Action.BUY, conf, f"dip {dip:.1f}% prog={prog:.0f}", self.name, expected_return=expected,
                      exit_overrides={"take_profit_pct": p.target_pct, "stop_loss_pct": p.stop_pct, "max_hold_s": p.max_hold_s})
