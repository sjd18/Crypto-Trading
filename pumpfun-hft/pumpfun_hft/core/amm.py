"""Constant-product AMM math (PumpSwap canonical pools; template for Raydium/Meteora/Orca).

Mirrors ``@pump-fun/pump-swap-sdk`` ``buyQuoteInput`` / ``sellBaseInput``:

* buy with quote budget ``Q`` (fees included):
  ``q = Q * 10_000 // (10_000 + fee_bps)`` then fees (ceil) on ``q``, reduced by any excess;
  ``base_out = base * (q - 1) // (quote_eff + q - 1)``.
* sell ``b`` base: ``out = quote_eff * b // (base + b)``; user receives ``out - lp - protocol - creator``.

The LP fee stays in the pool (it accrues to LPs), protocol/creator fees leave it.
Fee rates come from market-cap tiers for canonical pools, flat fees otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pumpfun_hft.core.curve import ONE_BILLION_SUPPLY, PRICE_SCALE, FeeBps, FeeSchedule, FeeTier, ceil_div, fee_amount
from pumpfun_hft.core.types import BPS


@dataclass(slots=True)
class PoolState:
    """Pool reserves; ``quote`` is the raw vault balance, ``virtual_quote`` the appended virtual reserve."""

    base: int
    quote: int
    supply: int = ONE_BILLION_SUPPLY
    virtual_quote: int = 0
    has_coin_creator: bool = True
    canonical: bool = True

    def copy(self) -> PoolState:
        return replace(self)

    @property
    def quote_eff(self) -> int:
        return self.quote + self.virtual_quote

    @property
    def price(self) -> float:
        return (self.quote_eff / self.base) * PRICE_SCALE if self.base else 0.0

    @property
    def market_cap_lamports(self) -> int:
        return self.quote_eff * self.supply // self.base if self.base else 0


@dataclass(frozen=True, slots=True)
class AmmBuyQuote:
    base_out: int
    quote_in: int        # quote entering the pool excluding fees
    lp_fee: int
    protocol_fee: int
    creator_fee: int

    @property
    def total(self) -> int:
        return self.quote_in + self.lp_fee + self.protocol_fee + self.creator_fee

    @property
    def avg_price(self) -> float:
        return (self.total / self.base_out) * PRICE_SCALE if self.base_out else float("inf")


@dataclass(frozen=True, slots=True)
class AmmSellQuote:
    base_in: int
    quote_out: int       # gross quote leaving the pool before fees
    lp_fee: int
    protocol_fee: int
    creator_fee: int

    @property
    def net(self) -> int:
        return self.quote_out - self.lp_fee - self.protocol_fee - self.creator_fee

    @property
    def avg_price(self) -> float:
        return (self.net / self.base_in) * PRICE_SCALE if self.base_in else 0.0


class ConstantProductAmm:
    """Quotes and state transitions for a PumpSwap-style constant-product pool."""

    def __init__(self, tiers: FeeSchedule, flat: FeeTier) -> None:
        self.tiers = tiers
        self.flat = flat

    def fees_for(self, pool: PoolState) -> FeeBps:
        tier = self.tiers.tier_for(pool.market_cap_lamports) if pool.canonical else self.flat
        return FeeBps(tier.protocol_bps, tier.creator_bps if pool.has_coin_creator else 0, tier.lp_bps)

    def buy_base_for_quote(self, pool: PoolState, quote_budget: int, fees: FeeBps | None = None) -> AmmBuyQuote:
        """Base tokens received for a quote budget including all fees."""
        if quote_budget <= 1 or pool.base == 0:
            return AmmBuyQuote(0, 0, 0, 0, 0)
        f = fees or self.fees_for(pool)
        eff = quote_budget * BPS // (BPS + f.total)
        lp, pr, cr = fee_amount(eff, f.lp), fee_amount(eff, f.protocol), fee_amount(eff, f.creator)
        total = eff + lp + pr + cr
        if total > quote_budget:
            eff -= total - quote_budget
        x = eff - 1
        if x <= 0:
            return AmmBuyQuote(0, 0, 0, 0, 0)
        base_out = pool.base * x // (pool.quote_eff + x)
        base_out = min(base_out, pool.base - 1)
        return AmmBuyQuote(base_out, eff, fee_amount(eff, f.lp), fee_amount(eff, f.protocol), fee_amount(eff, f.creator))

    def buy_quote_for_base(self, pool: PoolState, base_out: int, fees: FeeBps | None = None) -> AmmBuyQuote:
        """Quote needed (before fees) to buy exactly ``base_out``, with fees on top."""
        if base_out <= 0 or base_out >= pool.base:
            return AmmBuyQuote(0, 0, 0, 0, 0)
        f = fees or self.fees_for(pool)
        q = ceil_div(pool.quote_eff * base_out, pool.base - base_out)
        return AmmBuyQuote(base_out, q, fee_amount(q, f.lp), fee_amount(q, f.protocol), fee_amount(q, f.creator))

    def sell_quote_for_base(self, pool: PoolState, base_in: int, fees: FeeBps | None = None) -> AmmSellQuote:
        if base_in <= 0 or pool.base == 0:
            return AmmSellQuote(0, 0, 0, 0, 0)
        f = fees or self.fees_for(pool)
        out = pool.quote_eff * base_in // (pool.base + base_in)
        out = min(out, pool.quote)
        return AmmSellQuote(base_in, out, fee_amount(out, f.lp), fee_amount(out, f.protocol), fee_amount(out, f.creator))

    @staticmethod
    def apply_buy(pool: PoolState, q: AmmBuyQuote) -> PoolState:
        p = pool.copy()
        p.base -= q.base_out
        p.quote += q.quote_in + q.lp_fee
        return p

    @staticmethod
    def apply_sell(pool: PoolState, q: AmmSellQuote) -> PoolState:
        p = pool.copy()
        p.base += q.base_in
        p.quote -= q.quote_out - q.lp_fee
        return p


def shift_pool(hist: PoolState, our_net_tokens: int) -> PoolState:
    """Persistent-impact counterfactual for pools (approximate: constant ``k`` along our trades)."""
    if our_net_tokens == 0:
        return hist
    base = max(hist.base - our_net_tokens, 1)
    quote = hist.quote_eff * hist.base // base - hist.virtual_quote
    return PoolState(base, max(quote, 0), hist.supply, hist.virtual_quote, hist.has_coin_creator, hist.canonical)
