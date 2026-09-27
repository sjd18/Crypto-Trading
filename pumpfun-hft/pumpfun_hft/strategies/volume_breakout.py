"""Volume breakout: trade abnormal volume expansion that breaks the recent high.

Abnormal volume = short-window volume z-score (vs an EWMA of completed bar volumes, empty bars
included) above ``volume_z`` with at least ``min_short_volume_sol`` traded; a breakout requires
price within ``breakout_tolerance_pct`` of (or above) the medium-window high and net buying.
"""

from __future__ import annotations

from pumpfun_hft.core.types import Action, Signal, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class VolumeBreakout(Strategy):
    name = "volume_breakout"

    class Params(StrategyParams):
        min_age_s: float
        max_age_s: float
        volume_z: float
        min_short_volume_sol: float
        breakout_tolerance_pct: float
        min_imbalance: float
        max_rug_prob: float
        expected_return_pct: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f = self.p, ctx.f
        if ctx.has_position or not ctx.is_trade or ctx.token.venue is not Venue.CURVE:
            return hold()
        if not (p.min_age_s <= ctx.age_s <= p.max_age_s) or not ctx.event.is_buy:
            return hold()
        if f.volume_sol_short < p.min_short_volume_sol:
            return hold()
        vz = f.volume_z
        if vz < p.volume_z:
            return hold()
        if f.price < f.high_medium * (1.0 - p.breakout_tolerance_pct / 100.0) or f.imbalance_short < p.min_imbalance:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        conf = 50.0 + 35.0 * min(1.0, (vz - p.volume_z) / p.volume_z) + 15.0 * f.imbalance_short
        return Signal(Action.BUY, conf, f"volume z={vz:.1f} breakout", self.name, expected_return=p.expected_return_pct / 100.0)
