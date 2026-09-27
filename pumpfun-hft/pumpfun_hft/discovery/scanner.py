"""Token discovery engine: track every new token immediately and enrich it.

On every CreateEvent the engine synchronously produces a :class:`DiscoveredToken` with the
on-chain launch facts and the creator's *point-in-time* statistics (previous launches, win rate,
rugs, migrations, average ATH multiple, statistical creator score). Off-chain metadata
(description, image, Twitter, Telegram, website) is fetched asynchronously in live mode and
attached when it arrives; in backtests it is joined from the stored metadata table.

Launch facts that only exist after the creation slot (dev buy = "initial SOL deposited",
bundled buyers) are refreshed through :meth:`snapshot`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pumpfun_hft.api.pumpfun import MetadataFetcher, TokenMetadata
from pumpfun_hft.core.types import LAMPORTS_PER_SOL
from pumpfun_hft.discovery.creator import CreatorBook, CreatorScore
from pumpfun_hft.features.market import MarketState, TokenState
from pumpfun_hft.utils.logging import get_logger

log = get_logger("signals")


def classify_sector(name: str, symbol: str, keywords: dict[str, list[str]], default: str) -> str:
    """Narrative "sector" from name/symbol keywords (used for sector exposure limits)."""
    text = f"{name} {symbol}".lower()
    for sector, words in keywords.items():
        if any(w in text for w in words):
            return sector
    return default


def metadata_from_row(row: dict[str, Any], recent_names: set[str] | None = None) -> TokenMetadata:
    meta = TokenMetadata(mint=row.get("mint", ""), name=row.get("name") or "", symbol=row.get("symbol") or "",
                         uri=row.get("uri") or "", description=row.get("description") or "", image=row.get("image") or "",
                         twitter=row.get("twitter") or "", telegram=row.get("telegram") or "", website=row.get("website") or "",
                         fetched=True)
    meta.anomalies = MetadataFetcher.anomalies(meta, recent_names)
    return meta


@dataclass(slots=True)
class DiscoveredToken:
    mint: str
    creator: str | None
    created_slot: int | None
    created_ms: int | None
    name: str
    symbol: str
    uri: str
    bonding_curve: str | None
    initial_virtual_sol: int
    initial_virtual_tokens: int
    initial_real_tokens: int
    initial_supply: int
    initial_sol_deposited: float
    bundled_buyers: int
    progress_pct: float
    sector: str
    creator_score: CreatorScore
    metadata: TokenMetadata | None = None
    anomalies: list[str] = field(default_factory=list)

    @property
    def twitter(self) -> str:
        return self.metadata.twitter if self.metadata else ""

    @property
    def telegram(self) -> str:
        return self.metadata.telegram if self.metadata else ""

    @property
    def website(self) -> str:
        return self.metadata.website if self.metadata else ""

    @property
    def image_uri(self) -> str:
        return self.metadata.image if self.metadata else ""


class TokenDiscoveryEngine:
    """Discovery + enrichment. Call :meth:`on_create` from the event loop for each CreateEvent.

    Example::

        disc = TokenDiscoveryEngine(settings, market, creator_book, metadata_fetcher)
        tok = disc.on_create(token_state)     # immediate, point-in-time creator stats
        disc.snapshot(mint)                   # refreshed launch facts (dev buy, bundle)
    """

    def __init__(self, settings: Any, market: MarketState, creators: CreatorBook, fetcher: MetadataFetcher | None = None,
                 on_discovered: Callable[[DiscoveredToken], Any] | None = None) -> None:
        self.s = settings
        self.market = market
        self.creators = creators
        self.fetcher = fetcher
        self.on_discovered = on_discovered
        self.recent: deque[tuple[int, str]] = deque()
        self.discovered: dict[str, DiscoveredToken] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self.verbose = True  # log each discovery to the signals channel (backtests turn this off)

    def _recent_names(self, now_ms: int) -> set[str]:
        horizon = self.s.discovery.duplicate_name_window_s * 1000
        while self.recent and self.recent[0][0] < now_ms - horizon:
            self.recent.popleft()
        return {n for _, n in self.recent}

    def on_create(self, st: TokenState) -> DiscoveredToken:
        now = st.created_ms or 0
        recent = self._recent_names(now)
        if st.metadata is not None and isinstance(st.metadata, dict):
            st.metadata = metadata_from_row(st.metadata, recent)
        elif isinstance(st.metadata, TokenMetadata) and not st.metadata.anomalies:
            st.metadata.anomalies = MetadataFetcher.anomalies(st.metadata, recent)
        self.recent.append((now, st.name.strip().lower()))
        tok = DiscoveredToken(
            mint=st.mint, creator=st.creator, created_slot=st.created_slot, created_ms=st.created_ms, name=st.name,
            symbol=st.symbol, uri=st.uri, bonding_curve=st.bonding_curve, initial_virtual_sol=st.curve.v_sol,
            initial_virtual_tokens=st.curve.v_tok, initial_real_tokens=st.curve.r_tok, initial_supply=st.curve.supply,
            initial_sol_deposited=st.dev_buy_lamports / LAMPORTS_PER_SOL, bundled_buyers=len(st.bundled_buyers),
            progress_pct=self.market.curve.progress_pct(st.curve), sector=st.sector,
            creator_score=self.creators.score(st.creator), metadata=st.metadata if isinstance(st.metadata, TokenMetadata) else None,
        )
        tok.anomalies = list(tok.metadata.anomalies) if tok.metadata else []
        self.discovered[st.mint] = tok
        if tok.metadata is None and self.fetcher is not None and self.s.discovery.metadata.enabled and st.uri:
            try:
                task = asyncio.get_running_loop().create_task(self._enrich(st, recent))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            except RuntimeError:
                pass  # no running loop (synchronous backtest): metadata comes from the stored table
        if self.on_discovered is not None:
            self.on_discovered(tok)
        if self.verbose:
            log.info("discovered", extra={"data": {"mint": st.mint, "creator": st.creator, "score": round(tok.creator_score.score, 1),
                                                   "launches": tok.creator_score.launches, "rugs": tok.creator_score.rugs}})
        return tok

    async def _enrich(self, st: TokenState, recent: set[str]) -> None:
        assert self.fetcher is not None
        meta = await self.fetcher.fetch(st.mint, st.name, st.symbol, st.uri, recent)
        st.metadata = meta
        tok = self.discovered.get(st.mint)
        if tok is not None:
            tok.metadata = meta
            tok.anomalies = list(meta.anomalies)
            if self.on_discovered is not None:
                self.on_discovered(tok)

    def snapshot(self, mint: str) -> DiscoveredToken | None:
        tok = self.discovered.get(mint)
        st = self.market.get(mint)
        if tok is None or st is None:
            return tok
        tok.initial_sol_deposited = st.dev_buy_lamports / LAMPORTS_PER_SOL
        tok.bundled_buyers = len(st.bundled_buyers)
        tok.progress_pct = self.market.curve.progress_pct(st.curve)
        return tok
