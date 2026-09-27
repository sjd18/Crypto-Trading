"""Migration strategy: trade curves that are about to graduate to PumpSwap.

Pump.fun curves that sell out (``real_token_reserves == 0``) complete and their liquidity
migrates to a PumpSwap pool (Raydium before March 2025). The strategy buys late-stage curves
with strong net buying, holds through completion and migration (orders are not sent while a
completed curve awaits migration), and exits ``post_migration_exit_s`` after the pool opens.
"""

from __future__ import annotations

from pumpfun_hft.core.types import Action, Signal, Urgency, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class Migration(Strategy):
    name = "migration"

    class Params(StrategyParams):
        min_progress_pct: float
        max_progress_pct: float
        min_imbalance: float
        min_buy_sol_medium: float
        hold_through_migration: bool
        post_migration_exit_s: float
        expected_return_pct: float
        max_rug_prob: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, f, tok = self.p, ctx.f, ctx.token
        if ctx.has_position:
            if ctx.position_strategy != self.name:
                return hold()
            if tok.migrated and tok.migrated_ms is not None:
                if not p.hold_through_migration or ctx.now_ms - tok.migrated_ms >= p.post_migration_exit_s * 1000:
                    return Signal(Action.EXIT, 80, "post-migration exit", self.name, urgency=Urgency.HIGH)
            return hold()
        if not ctx.is_trade or tok.venue is not Venue.CURVE:
            return hold()
        prog = f.progress_pct
        if not (p.min_progress_pct <= prog <= p.max_progress_pct):
            return hold()
        if f.imbalance_medium < p.min_imbalance or f.buy_sol_medium < p.min_buy_sol_medium:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        conf = 50.0 + 30.0 * (prog - p.min_progress_pct) / max(1e-9, p.max_progress_pct - p.min_progress_pct) + 20.0 * f.imbalance_medium
        overrides = {"max_hold_s": 1e9} if p.hold_through_migration else None
        return Signal(Action.BUY, conf, f"pre-migration prog={prog:.0f}% ({f.sol_to_complete:.1f} SOL left)", self.name,
                      expected_return=p.expected_return_pct / 100.0, exit_overrides=overrides)
