"""Replay helpers: run events through the online state exactly like the backtester does.

``online_features`` returns, for every swap, the online feature values *immediately after* that
swap — the same numbers a strategy would see — so they can be compared with the vectorised
pipeline (:func:`pumpfun_hft.features.batch.batch_features`), inspected in notebooks or used as
a research dataset.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pumpfun_hft.analytics.wallet_intel import WalletIntel
from pumpfun_hft.core.curve import BondingCurve
from pumpfun_hft.core.types import EVENT_COLUMNS, SORT_KEYS, Event, EventKind
from pumpfun_hft.discovery.scanner import classify_sector
from pumpfun_hft.features.market import MarketState
from pumpfun_hft.features.online import FeatureEngine
from pumpfun_hft.features.registry import FEATURE_NAMES

_TRADES = frozenset({EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value})


def online_features(settings: Any, events: pl.DataFrame, names: list[str] | None = None,
                    metadata: dict[str, Any] | None = None) -> pl.DataFrame:
    """One row per swap: ``mint, ts_ms, slot`` + the requested online features after that swap.

    Example::

        df = online_features(settings, events, ["imbalance_medium", "ret_short"])
    """
    curve = BondingCurve.from_config(settings.protocol.curve, settings.protocol.curve_fee_tiers)
    sec = settings.risk.sectors
    market = MarketState(curve, lambda n, s: classify_sector(n, s, sec.keywords, sec.default), metadata)
    wallets = WalletIntel(settings.wallet_intel, settings.features)
    fe = FeatureEngine(settings.features, curve, wallets)
    wanted = list(names or FEATURE_NAMES)
    rows: list[dict[str, Any]] = []
    for r in events.sort(list(SORT_KEYS), maintain_order=True).select(list(EVENT_COLUMNS)).iter_rows():
        ev = Event(*r)
        st = market.on_event(ev)
        if ev.kind == EventKind.CREATE.value and st is not None:
            wallets.on_create(ev)
        fe.on_event(ev, st)  # wallet flags use the wallet database *before* this trade
        if ev.kind in _TRADES and st is not None:
            wallets.on_trade(ev, st.created_ms, st.created_slot, st.creator)
            view = fe.view(ev.mint or "", ev.ts_ms, st)
            if view is not None:
                rows.append({"mint": ev.mint, "ts_ms": ev.ts_ms, "slot": ev.slot, **view.as_dict(wanted)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
