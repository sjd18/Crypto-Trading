"""Exact integer math of the Pump bonding curve.

Mirrors the official ``@pump-fun/pump-sdk`` (``bondingCurve.ts`` / ``fees.ts``) bit-for-bit;
``tests/test_curve.py`` checks 792 golden values generated from that SDK.

Model
    Constant product on *virtual* reserves: ``v_sol * v_tok`` is (up to rounding) invariant.
    Fees are charged on the SOL leg and never enter the reserves:

    * buy with a SOL budget ``B`` (fees included):
      ``x = (B - 1) * 10_000 // (10_000 + fee_bps)``; ``tokens = x * v_tok // (v_sol + x)``,
      capped at the real token reserves.
    * cost of ``t`` tokens: ``c = t * v_sol // (v_tok - t) + 1``; total ``c + ceil(c*p) + ceil(c*k)``.
    * sell ``t`` tokens: ``s = t * v_sol // (v_tok + t)``; net ``s - ceil(s*p) - ceil(s*k)``.

    Fee rates depend on market-cap tiers (``FeeConfig``); the market cap used for tier selection
    is ``v_sol * 1e15 // v_tok`` (one-billion supply basis). The creator fee is charged only when
    the curve has a creator set.

All amounts are Python ints (arbitrary precision; no overflow). Prices are SOL per whole token.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from pumpfun_hft.core.types import BPS, LAMPORTS_PER_SOL, TOKEN_UNIT

if TYPE_CHECKING:
    from pumpfun_hft.core.config import CurveCfg, FeeTierCfg

ONE_BILLION_SUPPLY = 1_000_000_000_000_000
PRICE_SCALE = TOKEN_UNIT / LAMPORTS_PER_SOL  # lamports per base unit -> SOL per whole token (1e-3)


def ceil_div(a: int, b: int) -> int:
    """Ceiling division for non-negative ints."""
    return (a + b - 1) // b


def fee_amount(amount: int, bps: int) -> int:
    """Fee on ``amount`` at ``bps`` basis points, rounded up (as on-chain)."""
    return ceil_div(amount * bps, BPS)


@dataclass(frozen=True, slots=True)
class FeeTier:
    """One market-cap fee tier. ``threshold`` is in lamports of market cap."""

    threshold: int
    protocol_bps: int
    creator_bps: int
    lp_bps: int = 0


@dataclass(frozen=True, slots=True)
class FeeBps:
    """Fee rates applied to one trade."""

    protocol: int
    creator: int
    lp: int = 0

    @property
    def total(self) -> int:
        return self.protocol + self.creator + self.lp


class FeeSchedule:
    """Tiered fee schedule (``pump-fees-math::calculate_fee_tier``)."""

    def __init__(self, tiers: list[FeeTier]) -> None:
        if not tiers:
            raise ValueError("fee tiers cannot be empty")
        self.tiers = sorted(tiers, key=lambda t: t.threshold)

    @classmethod
    def from_config(cls, tiers: list[FeeTierCfg]) -> FeeSchedule:
        return cls([
            FeeTier(int(round(t.mcap_sol * LAMPORTS_PER_SOL)), t.protocol_bps, t.creator_bps, t.lp_bps) for t in tiers
        ])

    def tier_for(self, market_cap_lamports: int) -> FeeTier:
        first = self.tiers[0]
        if market_cap_lamports < first.threshold:
            return first
        for tier in reversed(self.tiers):
            if market_cap_lamports >= tier.threshold:
                return tier
        return first


@dataclass(slots=True)
class CurveState:
    """Bonding-curve account state (the fields that matter for pricing)."""

    v_tok: int
    v_sol: int
    r_tok: int
    r_sol: int
    supply: int = ONE_BILLION_SUPPLY
    complete: bool = False
    has_creator: bool = True
    mayhem: bool = False

    def copy(self) -> CurveState:
        return replace(self)

    @property
    def price(self) -> float:
        """Marginal (spot) price in SOL per whole token."""
        return (self.v_sol / self.v_tok) * PRICE_SCALE if self.v_tok else 0.0

    @property
    def market_cap_lamports(self) -> int:
        return self.v_sol * self.supply // self.v_tok if self.v_tok else 0

    @property
    def market_cap_sol(self) -> float:
        return self.market_cap_lamports / LAMPORTS_PER_SOL

    def fee_market_cap_lamports(self) -> int:
        """Market cap used to select fee tiers (1B-supply basis unless mayhem mode)."""
        supply = self.supply if self.mayhem else ONE_BILLION_SUPPLY
        return self.v_sol * supply // self.v_tok if self.v_tok else 0


@dataclass(frozen=True, slots=True)
class BuyQuote:
    tokens: int
    sol_curve: int      # SOL entering the curve reserves
    protocol_fee: int
    creator_fee: int

    @property
    def total(self) -> int:
        return self.sol_curve + self.protocol_fee + self.creator_fee

    @property
    def avg_price(self) -> float:
        return (self.total / self.tokens) * PRICE_SCALE if self.tokens else float("inf")


@dataclass(frozen=True, slots=True)
class SellQuote:
    tokens: int
    sol_curve: int      # SOL leaving the curve reserves
    protocol_fee: int
    creator_fee: int

    @property
    def net(self) -> int:
        return self.sol_curve - self.protocol_fee - self.creator_fee

    @property
    def avg_price(self) -> float:
        return (self.net / self.tokens) * PRICE_SCALE if self.tokens else 0.0


class BondingCurve:
    """Pump bonding-curve calculator bound to a fee schedule and initial parameters."""

    def __init__(
        self,
        schedule: FeeSchedule,
        initial_virtual_token_reserves: int,
        initial_virtual_sol_reserves: int,
        initial_real_token_reserves: int,
        token_total_supply: int,
    ) -> None:
        self.schedule = schedule
        self.initial_v_tok = initial_virtual_token_reserves
        self.initial_v_sol = initial_virtual_sol_reserves
        self.initial_r_tok = initial_real_token_reserves
        self.supply = token_total_supply

    @classmethod
    def from_config(cls, curve: CurveCfg, tiers: list[FeeTierCfg]) -> BondingCurve:
        return cls(
            FeeSchedule.from_config(tiers),
            curve.initial_virtual_token_reserves,
            curve.initial_virtual_sol_reserves,
            curve.initial_real_token_reserves,
            curve.token_total_supply,
        )

    # ------------------------------------------------------------------ state helpers
    def new_state(self, has_creator: bool = True) -> CurveState:
        """State of a freshly created curve."""
        return CurveState(self.initial_v_tok, self.initial_v_sol, self.initial_r_tok, 0, self.supply, False, has_creator)

    def fees_for(self, state: CurveState) -> FeeBps:
        tier = self.schedule.tier_for(state.fee_market_cap_lamports())
        return FeeBps(tier.protocol_bps, tier.creator_bps if state.has_creator else 0, 0)

    def progress_pct(self, state: CurveState) -> float:
        """Share of the sellable supply already bought from the curve (pump.fun progress bar)."""
        return 100.0 * (self.initial_r_tok - state.r_tok) / self.initial_r_tok

    # ------------------------------------------------------------------ quotes
    def buy_tokens_for_sol(self, state: CurveState, sol_in: int, fees: FeeBps | None = None) -> int:
        """Tokens received for a SOL budget that includes fees (``getBuyTokenAmountFromSolAmount``)."""
        if sol_in <= 0 or state.v_tok == 0 or state.complete:
            return 0
        f = fees or self.fees_for(state)
        total_bps = f.protocol + f.creator
        x = (sol_in - 1) * BPS // (total_bps + BPS)
        tokens = x * state.v_tok // (state.v_sol + x)
        return min(tokens, state.r_tok)

    def buy_cost_for_tokens(self, state: CurveState, tokens: int, fees: FeeBps | None = None) -> BuyQuote:
        """Exact SOL cost of buying ``tokens`` (``getBuySolAmountFromTokenAmount``)."""
        if tokens <= 0 or state.v_tok == 0:
            return BuyQuote(0, 0, 0, 0)
        f = fees or self.fees_for(state)
        t = min(tokens, state.r_tok)
        if t <= 0:
            return BuyQuote(0, 0, 0, 0)
        sol = t * state.v_sol // (state.v_tok - t) + 1
        return BuyQuote(t, sol, fee_amount(sol, f.protocol), fee_amount(sol, f.creator))

    def sell_proceeds_for_tokens(self, state: CurveState, tokens: int, fees: FeeBps | None = None) -> SellQuote:
        """SOL received for selling ``tokens`` (``getSellSolAmountFromTokenAmount``)."""
        if tokens <= 0 or state.v_tok == 0:
            return SellQuote(0, 0, 0, 0)
        f = fees or self.fees_for(state)
        sol = tokens * state.v_sol // (state.v_tok + tokens)
        # NB: callers that execute trades must check ``sol <= state.r_sol`` (liquidity exhaustion);
        # the quote itself mirrors the SDK exactly.
        return SellQuote(tokens, sol, fee_amount(sol, f.protocol), fee_amount(sol, f.creator))

    def buy_with_budget(self, state: CurveState, sol_in: int, fees: FeeBps | None = None) -> BuyQuote:
        """Quote for ``buy_exact_sol_in``: tokens for the budget and their exact cost (<= budget)."""
        f = fees or self.fees_for(state)
        tokens = self.buy_tokens_for_sol(state, sol_in, f)
        return self.buy_cost_for_tokens(state, tokens, f)

    # ------------------------------------------------------------------ state transitions
    @staticmethod
    def apply_buy(state: CurveState, quote: BuyQuote) -> CurveState:
        """Return the post-trade state after a buy (fees never enter reserves)."""
        s = state.copy()
        s.v_tok -= quote.tokens
        s.r_tok -= quote.tokens
        s.v_sol += quote.sol_curve
        s.r_sol += quote.sol_curve
        s.complete = s.r_tok == 0
        return s

    @staticmethod
    def apply_sell(state: CurveState, quote: SellQuote) -> CurveState:
        s = state.copy()
        s.v_tok += quote.tokens
        s.r_tok += quote.tokens
        s.v_sol -= quote.sol_curve
        s.r_sol -= quote.sol_curve
        return s

    # ------------------------------------------------------------------ limit / liquidity helpers
    def max_buy_tokens_at_avg_price(self, state: CurveState, limit_price: float, fees: FeeBps | None = None) -> int:
        """Largest token amount whose average all-in price is <= ``limit_price`` (SOL/token)."""
        f = fees or self.fees_for(state)
        lo, hi = 0, state.r_tok
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.buy_cost_for_tokens(state, mid, f).avg_price <= limit_price:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def max_sell_tokens_at_avg_price(self, state: CurveState, limit_price: float, max_tokens: int, fees: FeeBps | None = None) -> int:
        """Largest amount (<= ``max_tokens``) whose average net price is >= ``limit_price``."""
        f = fees or self.fees_for(state)
        lo, hi = 0, max_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.sell_proceeds_for_tokens(state, mid, f).avg_price >= limit_price:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def sol_to_complete(self, state: CurveState, fees: FeeBps | None = None) -> int:
        """All-in SOL needed to buy every remaining real token (completes the curve)."""
        return self.buy_cost_for_tokens(state, state.r_tok, fees).total if state.r_tok else 0

    def price_impact_bps(self, state: CurveState, sol_in: int) -> float:
        """Price impact of a buy of ``sol_in`` (avg execution price vs spot, incl. fees)."""
        q = self.buy_with_budget(state, sol_in)
        if not q.tokens:
            return float("inf")
        return (q.avg_price / state.price - 1.0) * BPS

    def state_from_tokens_sold(self, tokens_sold: int, has_creator: bool = True) -> CurveState:
        """Curve state after ``tokens_sold`` net tokens left the curve (invariant-consistent)."""
        k = self.initial_v_tok * self.initial_v_sol
        v_tok = self.initial_v_tok - tokens_sold
        v_sol = k // v_tok
        r_tok = self.initial_r_tok - tokens_sold
        return CurveState(v_tok, v_sol, r_tok, v_sol - self.initial_v_sol, self.supply, r_tok == 0, has_creator)


def shift_state(hist: CurveState, our_net_tokens: int) -> CurveState:
    """Apply our own net position to historical reserves (persistent impact model).

    Every trade moves the state along the same ``v_sol * v_tok = k`` hyperbola (fees never enter
    reserves), so the counterfactual state that includes our net purchase ``n`` is the point on the
    historical hyperbola with ``v_tok - n`` virtual tokens. Other participants are assumed to have
    traded the same token quantities.
    """
    if our_net_tokens == 0:
        return hist
    v_tok = hist.v_tok - our_net_tokens
    if v_tok <= 0:
        v_tok = 1
    v_sol = hist.v_sol * hist.v_tok // v_tok
    r_tok = max(hist.r_tok - our_net_tokens, 0)
    r_sol = max(hist.r_sol + (v_sol - hist.v_sol), 0)
    return CurveState(v_tok, v_sol, r_tok, r_sol, hist.supply, hist.complete or r_tok == 0, hist.has_creator, hist.mayhem)


def prices_from_reserves(v_sol: np.ndarray, v_tok: np.ndarray) -> np.ndarray:
    """Vectorised spot price (SOL/token) from reserve arrays."""
    v_tok_f = np.asarray(v_tok, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v_tok_f > 0, np.asarray(v_sol, dtype=np.float64) / v_tok_f * PRICE_SCALE, np.nan)
