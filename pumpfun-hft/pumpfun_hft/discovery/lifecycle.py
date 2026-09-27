"""Token outcome resolution (feeds creator history and wallet intelligence).

Each launch is resolved exactly once, at ``created_ms + resolution_horizon_s`` (processed as
simulated/wall time passes), using only what happened up to that moment:

    migrated  : the curve completed and migrated
    ath_mult  : ATH price / launch price
    rug       : max *liquidity* drawdown (real SOL in the curve / pool, from its running peak) >=
                rug_drawdown_pct AND creator sold >= rug_creator_sold_pct. Price drawdown is not used:
                on a bonding curve the price cannot fall below the launch price, so a dev dump at 2x
                shows only a ~50 % price drawdown while most of the SOL has left.
    success   : migrated OR ath_mult >= success_ath_multiple (and not a rug)

Resolutions are pushed to :class:`~pumpfun_hft.discovery.creator.CreatorBook` and to
:class:`~pumpfun_hft.analytics.wallet_intel.WalletIntel` (creator / bundled-insider rug flags).
"""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from pumpfun_hft.discovery.creator import CreatorBook
from pumpfun_hft.features.market import TokenState


@dataclass(slots=True)
class TokenOutcome:
    mint: str
    creator: str | None
    resolved_ms: int
    migrated: bool
    ath_multiple: float
    max_drawdown_pct: float
    max_liq_drawdown_pct: float
    creator_sold_pct: float
    rug: bool
    success: bool

    @property
    def label(self) -> str:
        return "rug" if self.rug else ("success" if self.success else "neutral")


class OutcomeResolver:
    """Schedules and resolves token outcomes as time advances."""

    def __init__(self, cfg: Any, creators: CreatorBook, wallet_intel: Any | None = None) -> None:
        self.cfg = cfg
        self.creators = creators
        self.wallet_intel = wallet_intel
        self._heap: list[tuple[int, str]] = []
        self.outcomes: dict[str, TokenOutcome] = {}

    def on_create(self, st: TokenState) -> None:
        if st.creator:
            self.creators.record_launch(st.creator, st.created_ms or 0)
        heapq.heappush(self._heap, (int((st.created_ms or 0) + self.cfg.resolution_horizon_s * 1000), st.mint))

    def advance(self, now_ms: int, tokens: dict[str, TokenState]) -> list[TokenOutcome]:
        """Resolve every token whose horizon is <= ``now_ms``."""
        out: list[TokenOutcome] = []
        while self._heap and self._heap[0][0] <= now_ms:
            due, mint = heapq.heappop(self._heap)
            st = tokens.get(mint)
            if st is None or mint in self.outcomes:
                continue
            o = self.resolve(st, due)
            self.outcomes[mint] = o
            out.append(o)
            if st.creator:
                self.creators.record_outcome(st.creator, o.success, o.rug, o.migrated, o.ath_multiple)
            if self.wallet_intel is not None:
                self.wallet_intel.on_token_outcome(st.creator, o.rug, set(st.bundled_buyers))
        return out

    def resolve(self, st: TokenState, at_ms: int) -> TokenOutcome:
        c = self.cfg
        rug = st.max_liq_dd_pct >= c.rug_drawdown_pct and st.creator_sold_pct >= c.rug_creator_sold_pct
        migrated = st.migrated and (st.migrated_ms or 0) <= at_ms
        success = (migrated or st.ath_multiple >= c.success_ath_multiple) and not rug
        return TokenOutcome(st.mint, st.creator, at_ms, migrated, st.ath_multiple, st.max_dd_pct, st.max_liq_dd_pct,
                            st.creator_sold_pct, rug, success)

    def to_frame(self) -> pl.DataFrame:
        if not self.outcomes:
            return pl.DataFrame(schema={"mint": pl.Utf8})
        return pl.DataFrame([{**asdict(o), "label": o.label} for o in self.outcomes.values()])
