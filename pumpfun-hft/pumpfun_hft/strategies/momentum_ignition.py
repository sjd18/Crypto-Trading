"""Momentum ignition: buy young tokens whose launch is turning explosive.

Detects (it does not create) ignition: strong one-sided buy flow from many distinct wallets with
positive medium-window returns early in the token's life, on reasonably distributed holdings
and an acceptable creator / rug profile. Exits discretionally when flow reverses; everything
else (stops, targets, trailing) is handled by the position manager.
"""

from __future__ import annotations

import math

from pumpfun_hft.core.types import Action, Signal, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class MomentumIgnition(Strategy):
    name = "momentum_ignition"

    class Params(StrategyParams):
        min_age_s: float
        max_age_s: float
        min_buy_sol: float
        min_imbalance: float
        min_unique_buyers: int
        min_momentum: float
        max_progress_pct: float
        max_rug_prob: float
        min_creator_score: float
        max_top10_pct: float
        exit_imbalance: float
        continuation: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f = self.p, ctx.f
        if ctx.has_position:
            if ctx.position_strategy == self.name and f.imbalance_medium <= p.exit_imbalance and f.ret_short < 0:
                return Signal(Action.SELL, 70, f"flow reversal imb={f.imbalance_medium:.2f}", self.name)
            return hold()
        if not ctx.is_trade or ctx.token.venue is not Venue.CURVE:
            return hold()
        age = ctx.age_s
        if not (p.min_age_s <= age <= p.max_age_s):
            return hold()
        buy, imb = f.buy_sol_medium, f.imbalance_medium
        if buy < p.min_buy_sol or imb < p.min_imbalance:
            return hold()
        ret = f.ret_medium
        if ret < p.min_momentum or f.unique_buyers_medium < p.min_unique_buyers:
            return hold()
        if f.progress_pct > p.max_progress_pct or f.top10_pct > p.max_top10_pct:
            return hold()
        if ctx.creator_score.score < p.min_creator_score:
            return hold()
        rug = ctx.rug_prob
        if rug > p.max_rug_prob:
            return hold()
        s1 = (imb - p.min_imbalance) / max(1e-9, 1.0 - p.min_imbalance)
        s2 = min(1.0, ret / (3.0 * p.min_momentum))
        s3 = min(1.0, buy / (3.0 * p.min_buy_sol))
        conf = (50.0 + 50.0 * (s1 + s2 + s3) / 3.0) * (1.0 - rug)
        exp_ret = math.expm1(p.continuation * ret)
        return Signal(Action.BUY, conf, f"ignition imb={imb:.2f} ret={ret:.2f} buy={buy:.1f}", self.name, expected_return=exp_ret)
