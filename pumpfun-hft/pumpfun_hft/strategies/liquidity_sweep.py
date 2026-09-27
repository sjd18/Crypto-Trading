"""Liquidity sweep detection: identify exhaustion after large buys.

A *sweep* is a single large buy that moves price sharply. If follow-through buying does not
arrive within ``exhaustion_s`` and short-term flow turns negative, the move is exhausted:
open positions are exited. Optionally the strategy re-enters after a deep pullback from the
sweep high, provided the sweeper still holds and buyers step back in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Action, Signal, Urgency, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@dataclass(slots=True)
class _Sweep:
    ts: int
    high: float
    user: str
    sol: float


@register
class LiquiditySweep(Strategy):
    name = "liquidity_sweep"

    class Params(StrategyParams):
        sweep_min_sol: float
        sweep_impact_pct: float
        exhaustion_s: float
        follow_through_frac: float
        enable_entries: bool
        entry_pullback_pct: float
        reentry_imbalance: float
        max_rug_prob: float
        target_pct: float
        max_age_s: float

    def __init__(self, params: object) -> None:
        super().__init__(params)  # type: ignore[arg-type]
        self.sweeps: dict[str, _Sweep] = {}

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f, ev = self.p, ctx.f, ctx.event
        mint = ctx.token.mint
        if not ctx.is_trade:
            return hold()
        sol = (ev.sol_amount or 0) / LAMPORTS_PER_SOL
        if ev.is_buy and sol >= p.sweep_min_sol and f.log_ret >= math.log1p(p.sweep_impact_pct / 100.0):
            self.sweeps[mint] = _Sweep(ctx.now_ms, f.price, ev.user or "", sol)
            return hold()
        sw = self.sweeps.get(mint)
        if sw is None:
            return hold()
        age_ms = ctx.now_ms - sw.ts
        if age_ms > 10 * p.exhaustion_s * 1000:
            del self.sweeps[mint]
            return hold()
        sw.high = max(sw.high, f.price)
        if age_ms < p.exhaustion_s * 1000:
            return hold()
        exhausted = f.buy_sol_short < p.follow_through_frac * sw.sol and f.imbalance_short < 0
        if ctx.has_position:
            if exhausted:
                return Signal(Action.EXIT, 75, f"sweep exhaustion ({sw.sol:.1f} SOL sweep)", self.name, urgency=Urgency.HIGH)
            return hold()
        if not p.enable_entries or ctx.token.venue is not Venue.CURVE or ctx.age_s > p.max_age_s:
            return hold()
        pullback = 100.0 * (1.0 - f.price / sw.high) if sw.high > 0 else 0.0
        if pullback < p.entry_pullback_pct or f.imbalance_short < p.reentry_imbalance:
            return hold()
        w = ctx.wallets.get(sw.user)
        if w is None or mint not in w.positions:  # the sweeper has exited: no support
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        expected = min(p.target_pct / 100.0, sw.high / f.price - 1.0)
        conf = 55.0 + 30.0 * min(1.0, pullback / (2 * p.entry_pullback_pct)) + 15.0 * f.imbalance_short
        return Signal(Action.BUY, conf, f"post-sweep pullback {pullback:.0f}%", self.name, expected_return=expected)
