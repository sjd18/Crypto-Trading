"""Sniper: enter in the first seconds after launch when the launch profile is clean.

Evaluated on the launch's first events (CreateEvent and early trades). Requirements: decision
within ``max_entry_age_s`` of creation, creator score above ``min_creator_score`` (point-in-time),
optional social links, a dev buy ("initial SOL deposited") inside a sane range, few bundled
creation-slot buyers (insider signature), no copy-cat name, acceptable rug probability.
Latency dominates this strategy: the backtester's latency model decides how many other
snipers you land behind, so results are very sensitive to ``simulation.latency``.
"""

from __future__ import annotations

from pumpfun_hft.core.types import EventKind, Action, Signal, Venue, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class Sniper(Strategy):
    name = "sniper"
    entry_events = frozenset({EventKind.CREATE.value, EventKind.TRADE.value})

    class Params(StrategyParams):
        max_entry_age_s: float
        min_creator_score: float
        require_socials: bool
        min_dev_buy_sol: float
        max_dev_buy_sol: float
        max_bundled_buyers: int
        allow_duplicate_names: bool
        size_sol: float
        expected_return_pct: float
        take_profit_pct: float
        stop_pct: float
        max_hold_s: float
        max_rug_prob: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p, tok = self.p, ctx.token
        if ctx.has_position or tok.venue is not Venue.CURVE or not tok.seen_create:
            return hold()
        if ctx.age_s > p.max_entry_age_s:
            return hold()
        if ctx.event.kind == EventKind.CREATE.value:
            return hold()  # wait for the dev buy in the creation slot
        dev = ctx.f.dev_buy_sol
        if not (p.min_dev_buy_sol <= dev <= p.max_dev_buy_sol):
            return hold()
        if ctx.f.bundled_unknown > p.max_bundled_buyers:
            return hold()
        meta = tok.metadata
        if p.require_socials and not (meta is not None and getattr(meta, "has_socials", False)):
            return hold()
        if not p.allow_duplicate_names and meta is not None and "duplicate_name" in (getattr(meta, "anomalies", None) or []):
            return hold()
        cs = ctx.creator_score
        if cs.score < p.min_creator_score:
            return hold()
        if ctx.rug_prob > p.max_rug_prob:
            return hold()
        conf = min(100.0, 40.0 + cs.score * 0.6)
        return Signal(Action.BUY, conf, f"snipe creator={cs.score:.0f} dev={dev:.2f}", self.name, size_sol=p.size_sol,
                      expected_return=p.expected_return_pct / 100.0,
                      exit_overrides={"take_profit_pct": p.take_profit_pct, "stop_loss_pct": p.stop_pct, "max_hold_s": p.max_hold_s})
