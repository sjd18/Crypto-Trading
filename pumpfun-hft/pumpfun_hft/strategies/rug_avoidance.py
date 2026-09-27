"""Rug avoidance overlay: exit before rug signatures complete; veto risky entries.

Runs for every open position regardless of which strategy opened it (configured as
``strategy.exit_overlay``). EXIT triggers (urgency ``exit`` -> wider slippage, higher priority):

* rug probability >= ``max_rug_prob``
* the creator sells, having now sold >= ``creator_sell_pct`` % of the tokens they bought
* real SOL liquidity has dropped >= ``liquidity_drop_pct`` % from its peak *since the position was
  opened* AND by at least ``liquidity_drop_min_sol`` SOL (a single retail sell on a 2-SOL curve is
  not a rug; measuring from the token's all-time peak would stop out every dip entry at once)
* a top-3 holder that held >= ``top_holder_min_supply_pct`` % of supply dumps >=
  ``top_holder_dump_pct`` % of their balance in one trade (sniper flips of small bags are ignored)

``veto`` blocks new entries when the rug probability exceeds ``veto_entry_rug_prob`` or the
creator has already sold past the threshold.
"""

from __future__ import annotations

import heapq
from typing import Any

from pumpfun_hft.core.types import Action, Signal, Urgency, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class RugAvoidance(Strategy):
    name = "rug_avoidance"

    class Params(StrategyParams):
        max_rug_prob: float
        creator_sell_pct: float
        liquidity_drop_pct: float
        liquidity_drop_min_sol: float
        top_holder_dump_pct: float
        top_holder_min_supply_pct: float
        veto_entry_rug_prob: float

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self._liq_peak: dict[int, float] = {}  # trade_id -> highest real-SOL liquidity since the position opened

    def _position_liquidity_peak(self, ctx: StrategyContext) -> float:
        assert ctx.position is not None
        tid = ctx.position.trade_id
        liq = ctx.f.liquidity_sol
        peak = max(self._liq_peak.get(tid, liq), liq)
        self._liq_peak[tid] = peak
        if len(self._liq_peak) > 10_000:  # bounded memory: forget the oldest round trips
            for k in sorted(self._liq_peak)[:5_000]:
                del self._liq_peak[k]
        return peak

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        if not ctx.has_position:
            return hold()
        p, f, ev = self.p, ctx.f, ctx.event
        liq_peak = self._position_liquidity_peak(ctx)
        if ctx.is_trade and not ev.is_buy and ev.user:
            if ev.user == ctx.token.creator and f.creator_sold_pct >= p.creator_sell_pct:
                return Signal(Action.EXIT, 95, f"creator sold {f.creator_sold_pct:.0f}%", self.name, urgency=Urgency.EXIT)
            sold = ev.token_amount or 0
            after = f.tf.holders.get(ev.user, 0)
            before = after + sold
            supply = ctx.token.curve.supply or 1
            if (before > 0 and 100.0 * sold / before >= p.top_holder_dump_pct
                    and 100.0 * before / supply >= p.top_holder_min_supply_pct):
                top3 = heapq.nlargest(3, f.tf.holders.values())
                if not top3 or before >= top3[-1]:
                    return Signal(Action.EXIT, 85, f"top holder dumped {100.0 * sold / before:.0f}% "
                                  f"({100.0 * before / supply:.1f}% of supply)", self.name, urgency=Urgency.EXIT)
        drop_sol = liq_peak - f.liquidity_sol
        if (liq_peak > 0 and not ctx.token.complete and drop_sol >= p.liquidity_drop_min_sol
                and 100.0 * drop_sol / liq_peak >= p.liquidity_drop_pct):
            return Signal(Action.EXIT, 85, f"liquidity drop -{100.0 * drop_sol / liq_peak:.0f}% since entry peak", self.name,
                          urgency=Urgency.EXIT)
        rug = ctx.rug_prob
        if rug >= p.max_rug_prob:
            return Signal(Action.EXIT, min(100.0, 100.0 * rug), f"rug probability {rug:.2f}", self.name, urgency=Urgency.EXIT)
        return hold()

    def veto(self, ctx: StrategyContext) -> str | None:
        if ctx.f.creator_sold_pct >= self.p.creator_sell_pct:
            return "creator already selling"
        if ctx.rug_prob >= self.p.veto_entry_rug_prob:
            return f"rug probability {ctx.rug_prob:.2f}"
        return None
