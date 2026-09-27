"""Point-in-time market state (per-token curve / pool state and lifecycle facts).

:class:`MarketState` consumes normalised events in on-chain order and maintains, for every
mint, the *historical* bonding-curve state (reserves reported by the latest TradeEvent), the
PumpSwap pool after migration, observed fee rates and lifecycle facts used by discovery,
features and the rug model (launch/ATH prices, drawdown, creator buys/sells, dev buy,
creation-slot bundle buyers, peak liquidity).

It never looks ahead: every field reflects only events already applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pumpfun_hft.core.amm import PoolState
from pumpfun_hft.core.curve import BondingCurve, CurveState, FeeBps
from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Event, EventKind, Venue

_TRADE = EventKind.TRADE.value
_CREATE = EventKind.CREATE.value
_COMPLETE = EventKind.COMPLETE.value
_MIGRATE = EventKind.MIGRATE.value
_AMM_BUY = EventKind.AMM_BUY.value
_AMM_SELL = EventKind.AMM_SELL.value


@dataclass(slots=True)
class TokenState:
    mint: str
    curve: CurveState
    creator: str | None = None
    name: str = ""
    symbol: str = ""
    uri: str = ""
    created_ms: int | None = None
    created_slot: int | None = None
    bonding_curve: str | None = None
    seen_create: bool = False
    complete: bool = False
    complete_ms: int | None = None
    migrated: bool = False
    migrated_ms: int | None = None
    pool: PoolState | None = None
    pool_addr: str | None = None
    fee_bps: FeeBps | None = None
    amm_fee_bps: FeeBps | None = None
    last_ts: int = 0
    last_slot: int = 0
    n_trades: int = 0
    first_trade_ms: int | None = None
    launch_price: float = 0.0
    price: float = 0.0
    prev_price: float = 0.0
    ath_price: float = 0.0
    ath_ms: int = 0
    max_dd_pct: float = 0.0
    creator_bought: int = 0
    creator_sold: int = 0
    dev_buy_lamports: int = 0
    bundled_buyers: set[str] = field(default_factory=set)
    peak_r_sol: int = 0
    peak_liq_lamports: int = 0      # highest real-SOL liquidity (curve r_sol, then pool quote) seen so far
    max_liq_dd_pct: float = 0.0     # deepest liquidity drawdown from that running peak (rug signature)
    volume_lamports: int = 0
    sector: str = "other"
    metadata: Any = None

    @property
    def venue(self) -> Venue | None:
        """Where the token can trade now (None while completed but not yet migrated)."""
        if self.migrated and self.pool is not None:
            return Venue.AMM
        if self.complete:
            return None
        return Venue.CURVE

    @property
    def age_ms(self) -> int:
        return self.last_ts - self.created_ms if self.created_ms is not None else 0

    @property
    def creator_sold_pct(self) -> float:
        return 100.0 * self.creator_sold / self.creator_bought if self.creator_bought else 0.0

    @property
    def ath_multiple(self) -> float:
        return self.ath_price / self.launch_price if self.launch_price else 1.0


class MarketState:
    """Applies events to per-token state. ``on_event`` returns the affected TokenState."""

    def __init__(self, curve: BondingCurve, sector_fn: Any = None, metadata: dict[str, Any] | None = None) -> None:
        self.curve = curve
        self.tokens: dict[str, TokenState] = {}
        self.sector_fn = sector_fn
        self.metadata = metadata or {}
        self.last_ts = 0
        self.last_slot = 0
        self.n_events = 0

    def _new(self, mint: str) -> TokenState:
        st = TokenState(mint=mint, curve=self.curve.new_state())
        st.launch_price = st.curve.price
        st.price = st.prev_price = st.ath_price = st.launch_price
        self.tokens[mint] = st
        return st

    def get(self, mint: str) -> TokenState | None:
        return self.tokens.get(mint)

    def on_event(self, ev: Event) -> TokenState | None:  # noqa: C901
        self.n_events += 1
        if ev.ts_ms > self.last_ts:
            self.last_ts = ev.ts_ms
        if ev.slot > self.last_slot:
            self.last_slot = ev.slot
        mint = ev.mint
        if mint is None:
            return None
        st = self.tokens.get(mint)
        kind = ev.kind
        if kind == _CREATE:
            if st is None:
                st = self._new(mint)
            st.seen_create = True
            st.creator = ev.creator or ev.user
            st.name, st.symbol, st.uri = ev.name or "", ev.symbol or "", ev.uri or ""
            st.created_ms, st.created_slot = ev.ts_ms, ev.slot
            st.bonding_curve = ev.bonding_curve
            if ev.v_tok and ev.v_sol:
                st.curve = CurveState(ev.v_tok, ev.v_sol, ev.r_tok or 0, ev.r_sol or 0,
                                      ev.token_amount or self.curve.supply, False, True)
                st.launch_price = st.price = st.prev_price = st.ath_price = st.curve.price
            st.ath_ms = ev.ts_ms
            st.metadata = self.metadata.get(mint)
            if self.sector_fn is not None:
                st.sector = self.sector_fn(st.name, st.symbol)
        elif kind == _TRADE:
            if st is None:  # history started after creation: partial state from reserves
                st = self._new(mint)
                st.created_ms = ev.ts_ms
                st.created_slot = ev.slot
                st.creator = ev.creator
            st.curve = CurveState(ev.v_tok or 0, ev.v_sol or 0, ev.r_tok or 0, ev.r_sol or 0,
                                  st.curve.supply, (ev.r_tok == 0), True if ev.creator is None else ev.creator != "11111111111111111111111111111111")
            if ev.fee_bps is not None:
                st.fee_bps = FeeBps(int(ev.fee_bps), int(ev.creator_fee_bps or 0))
            self._trade_common(st, ev)
            if st.creator is not None and ev.user == st.creator:
                if ev.is_buy:
                    st.creator_bought += ev.token_amount or 0
                    if st.created_slot is not None and ev.slot == st.created_slot:
                        st.dev_buy_lamports += ev.sol_amount or 0
                else:
                    st.creator_sold += ev.token_amount or 0
            elif ev.is_buy and st.created_slot is not None and ev.slot == st.created_slot:
                st.bundled_buyers.add(ev.user or "")
            if (ev.r_sol or 0) > st.peak_r_sol:
                st.peak_r_sol = ev.r_sol or 0
        elif kind == _COMPLETE:
            if st is None:
                st = self._new(mint)
            st.complete = True
            st.complete_ms = ev.ts_ms
            st.curve.complete = True
        elif kind == _MIGRATE:
            if st is None:
                st = self._new(mint)
            st.complete = True
            st.migrated = True
            st.migrated_ms = ev.ts_ms
            st.pool_addr = ev.pool
            st.pool = PoolState(int(ev.v_tok or ev.token_amount or 0), int(ev.r_sol or ev.sol_amount or 0), st.curve.supply)
        elif kind in (_AMM_BUY, _AMM_SELL):
            if st is None:
                st = self._new(mint)
                st.complete = st.migrated = True
            base, quote = int(ev.r_tok or 0), int(ev.r_sol or 0)
            st.pool = PoolState(base, quote, st.curve.supply, max(0, int((ev.v_sol or quote) - quote)))
            st.migrated = st.complete = True
            if ev.fee_bps is not None:
                st.amm_fee_bps = FeeBps(int(ev.fee_bps), int(ev.creator_fee_bps or 0), int(ev.lp_fee_bps or 0))
            if st.creator is not None and ev.user == st.creator and not ev.is_buy:
                st.creator_sold += ev.token_amount or 0
            self._trade_common(st, ev)
        else:
            return st
        st.last_ts = ev.ts_ms
        st.last_slot = ev.slot
        return st

    @staticmethod
    def _trade_common(st: TokenState, ev: Event) -> None:
        st.n_trades += 1
        if st.first_trade_ms is None:
            st.first_trade_ms = ev.ts_ms
        st.prev_price = st.price
        st.price = st.pool.price if (st.migrated and st.pool is not None) else st.curve.price
        st.volume_lamports += ev.sol_amount or 0
        if st.price > st.ath_price:
            st.ath_price, st.ath_ms = st.price, ev.ts_ms
        if st.ath_price > 0:
            dd = 100.0 * (1.0 - st.price / st.ath_price)
            if dd > st.max_dd_pct:
                st.max_dd_pct = dd
        # liquidity drawdown: on a bonding curve the price can never fall below the launch price, so a
        # price drawdown understates a rug; the share of real SOL that has left the curve / pool does not
        liq = st.pool.quote if (st.migrated and st.pool is not None) else st.curve.r_sol
        if liq > st.peak_liq_lamports:
            st.peak_liq_lamports = liq
        elif st.peak_liq_lamports > 0:
            ldd = 100.0 * (1.0 - liq / st.peak_liq_lamports)
            if ldd > st.max_liq_dd_pct:
                st.max_liq_dd_pct = ldd

    def liquidity_sol(self, mint: str) -> float:
        st = self.tokens.get(mint)
        if st is None:
            return 0.0
        if st.migrated and st.pool is not None:
            return st.pool.quote / LAMPORTS_PER_SOL
        return st.curve.r_sol / LAMPORTS_PER_SOL
