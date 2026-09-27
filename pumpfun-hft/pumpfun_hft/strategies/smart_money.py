"""Smart-money copy trading: enter after several elite wallets buy the same token.

Elite = point-in-time smart score >= ``min_smart_score``. When at least ``min_smart_buyers``
distinct elite wallets buy within ``window_s`` (and together spend ``min_total_smart_sol``),
the strategy enters. It exits when elite wallets have sold at least ``exit_smart_sell_frac`` of
what they bought since the entry cohort formed.
"""

from __future__ import annotations

from collections import deque

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Action, Signal, Urgency, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class SmartMoney(Strategy):
    name = "smart_money"

    class Params(StrategyParams):
        min_smart_buyers: int
        window_s: float
        min_smart_score: float
        min_total_smart_sol: float
        max_age_s: float
        max_progress_pct: float
        exit_smart_sell_frac: float
        expected_return_pct: float
        max_rug_prob: float

    def __init__(self, params: object) -> None:
        super().__init__(params)  # type: ignore[arg-type]
        self.buys: dict[str, deque[tuple[int, str, float, float]]] = {}
        self.flow: dict[str, list[float]] = {}  # mint -> [smart bought SOL, smart sold SOL] since entry

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, ev = self.p, ctx.event
        if not ctx.is_trade or not ev.user:
            return hold()
        mint = ctx.token.mint
        score = ctx.wallets.score(ev.user)
        is_smart = score is not None and score >= p.min_smart_score
        sol = (ev.sol_amount or 0) / LAMPORTS_PER_SOL
        if ctx.has_position and ctx.position_strategy == self.name:
            fl = self.flow.setdefault(mint, [0.0, 0.0])
            if is_smart:
                fl[0 if ev.is_buy else 1] += sol
            if fl[0] > 0 and fl[1] >= p.exit_smart_sell_frac * fl[0]:
                return Signal(Action.EXIT, 80, f"smart wallets sold {fl[1]:.1f}/{fl[0]:.1f} SOL", self.name, urgency=Urgency.HIGH)
            return hold()
        if not is_smart or not ev.is_buy:
            return hold()
        dq = self.buys.setdefault(mint, deque())
        dq.append((ctx.now_ms, ev.user, sol, float(score)))
        cutoff = ctx.now_ms - p.window_s * 1000
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        if ctx.has_position or ctx.token.venue is not Venue.CURVE or ctx.age_s > p.max_age_s:
            return hold()
        wallets = {u for _, u, _, _ in dq}
        total = sum(s for _, _, s, _ in dq)
        if len(wallets) < p.min_smart_buyers or total < p.min_total_smart_sol or ctx.f.progress_pct > p.max_progress_pct:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        avg_score = sum(sc for _, _, _, sc in dq) / len(dq)
        conf = min(100.0, 45.0 + 12.0 * (len(wallets) - 1) + 60.0 * (avg_score - 0.5))
        self.flow[mint] = [total, 0.0]
        return Signal(Action.BUY, conf, f"{len(wallets)} smart wallets bought {total:.1f} SOL", self.name,
                      expected_return=p.expected_return_pct / 100.0)
