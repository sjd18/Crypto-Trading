"""Mean reversion on failed pumps.

A token that pumped at least ``min_pump_multiple`` x from launch and then retraced
``min_retrace_pct``-``max_retrace_pct`` % from its ATH, where selling has decelerated (short-window
sell rate well below the medium-window rate) and buyers are stepping back in, is bought for a
partial bounce (``target_retrace_frac`` of the distance back to the ATH). Tokens beyond the
maximum retrace, with concentrated holdings or a high rug probability are skipped.
"""

from __future__ import annotations

from pumpfun_hft.core.types import Action, Signal, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class MeanReversion(Strategy):
    name = "mean_reversion"

    class Params(StrategyParams):
        min_age_s: float
        min_pump_multiple: float
        min_retrace_pct: float
        max_retrace_pct: float
        sell_exhaustion_ratio: float
        min_short_imbalance: float
        max_top10_pct: float
        target_retrace_frac: float
        max_rug_prob: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f = self.p, ctx.f
        if ctx.has_position or not ctx.is_trade or ctx.token.venue is not Venue.CURVE:
            return hold()
        if ctx.age_s < p.min_age_s or f.ath_multiple < p.min_pump_multiple:
            return hold()
        dd = f.drawdown_from_ath_pct
        if not (p.min_retrace_pct <= dd <= p.max_retrace_pct):
            return hold()
        short_s, med_s = f.cfg.short_window_ms, f.cfg.medium_window_ms
        med_rate = f.sell_sol_medium * short_s / med_s
        if med_rate <= 0 or f.sell_sol_short > p.sell_exhaustion_ratio * med_rate:
            return hold()
        imb = f.imbalance_short
        if imb < p.min_short_imbalance or f.top10_pct > p.max_top10_pct:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        ath = ctx.token.ath_price
        expected = p.target_retrace_frac * (ath / f.price - 1.0) if f.price > 0 else 0.0
        conf = 50.0 + 30.0 * min(1.0, (dd - p.min_retrace_pct) / max(1e-9, p.max_retrace_pct - p.min_retrace_pct)) + 20.0 * imb
        return Signal(Action.BUY, conf, f"failed pump dd={dd:.0f}% ath={f.ath_multiple:.1f}x", self.name, expected_return=expected)
