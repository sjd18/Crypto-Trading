"""Whale follow: mirror large buys from *statistically profitable* wallets.

Size alone is not enough: the buyer must have a point-in-time smart score (posterior probability
of positive edge from realised round trips) above ``min_wallet_score`` with at least
``min_wallet_closed`` closed trades. The position is exited when the followed whale sells a
large fraction of their holdings.
"""

from __future__ import annotations

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Action, Signal, Urgency, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class WhaleFollow(Strategy):
    name = "whale_follow"

    class Params(StrategyParams):
        min_whale_buy_sol: float
        min_wallet_score: float
        min_wallet_closed: int
        max_age_s: float
        max_progress_pct: float
        exit_on_whale_sell_frac: float
        expected_return_pct: float
        max_rug_prob: float

    def __init__(self, params: object) -> None:
        super().__init__(params)  # type: ignore[arg-type]
        self.followed: dict[str, str] = {}

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, ev = self.p, ctx.event
        if not ctx.is_trade or not ev.user:
            return hold()
        mint = ctx.token.mint
        if ctx.has_position:
            if ctx.position_strategy == self.name and not ev.is_buy and self.followed.get(mint) == ev.user:
                w = ctx.wallets.get(ev.user)
                held = w.positions[mint].tokens if (w and mint in w.positions) else 0
                frac = (ev.token_amount or 0) / held if held else 1.0
                if frac >= p.exit_on_whale_sell_frac:
                    return Signal(Action.EXIT, 80, f"followed whale sold {frac:.0%}", self.name, urgency=Urgency.HIGH)
            return hold()
        if not ev.is_buy or (ev.sol_amount or 0) < p.min_whale_buy_sol * LAMPORTS_PER_SOL:
            return hold()
        if ctx.token.venue is not Venue.CURVE or ctx.age_s > p.max_age_s or ctx.f.progress_pct > p.max_progress_pct:
            return hold()
        w = ctx.wallets.get(ev.user)
        score = ctx.wallets.score(ev.user)
        if w is None or score is None or w.closed < p.min_wallet_closed or score < p.min_wallet_score:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        self.followed[mint] = ev.user
        conf = min(100.0, 100.0 * score)
        return Signal(Action.BUY, conf, f"follow whale wallet={ev.user[:6]} score={score:.2f}", self.name,
                      expected_return=p.expected_return_pct / 100.0)
