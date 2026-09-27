"""Point-in-time ML datasets.

Snapshots are taken at ``created_ms + delay`` for each delay in ``rug_model.snapshot_delays_s``
by replaying events through the *same* online state as live trading (MarketState, WalletIntel,
FeatureEngine, CreatorBook) and recording features just before the first event at or after the
snapshot time — so a snapshot contains only information available at that instant.

Labels are computed afterwards from strictly later events in ``(snapshot, snapshot + horizon]``:

    rug          1 if real-SOL liquidity (curve r_sol / pool quote) falls to <= snapshot liquidity *
                 (1 - label_drawdown_pct/100) — liquidity, not price: a curve's price cannot fall below the
                 launch price, so price drawdowns understate dumps
    fwd_return   log(price at horizon / snapshot price) (+ binary ``fwd_up`` >= fwd_return_threshold)
    migrate      1 if the curve migrates within the horizon

Every row carries ``snapshot_ms`` and ``label_end_ms`` so cross-validation can purge samples
whose label window overlaps the test period.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import polars as pl

from pumpfun_hft.analytics.wallet_intel import WalletIntel
from pumpfun_hft.core.curve import BondingCurve
from pumpfun_hft.core.types import EVENT_COLUMNS, Event, EventKind
from pumpfun_hft.discovery.creator import CreatorBook
from pumpfun_hft.discovery.lifecycle import OutcomeResolver
from pumpfun_hft.discovery.scanner import TokenDiscoveryEngine, classify_sector
from pumpfun_hft.features.market import MarketState
from pumpfun_hft.features.online import FeatureEngine
from pumpfun_hft.features.registry import FEATURE_NAMES
from pumpfun_hft.ml.rug_model import RUG_FEATURES, rug_features

_TRADES = [EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value]


def build_snapshot_dataset(settings: Any, events: pl.DataFrame, metadata: dict[str, Any] | None = None,
                           delays_s: list[float] | None = None, horizon_s: float | None = None) -> pl.DataFrame:
    """Replay ``events`` and return one feature row per (token, snapshot delay) with labels."""
    s = settings
    delays = delays_s or list(s.rug_model.snapshot_delays_s)
    horizon_ms = int((horizon_s or s.rug_model.label_horizon_s) * 1000)
    curve = BondingCurve.from_config(s.protocol.curve, s.protocol.curve_fee_tiers)
    sec = s.risk.sectors
    market = MarketState(curve, lambda n, sym: classify_sector(n, sym, sec.keywords, sec.default), metadata)
    wallets = WalletIntel(s.wallet_intel, s.features)
    feats = FeatureEngine(s.features, curve, wallets)
    creators = CreatorBook(s.discovery)
    resolver = OutcomeResolver(s.discovery, creators, wallets)
    discovery = TokenDiscoveryEngine(s, market, creators)
    heap: list[tuple[int, int, str, float]] = []
    seq = 0
    rows: list[dict[str, Any]] = []

    def snapshot(t: int, mint: str, delay: float) -> None:
        st = market.get(mint)
        if st is None or st.venue is None:
            return
        view = feats.view(mint, t, st)
        if view is None:
            return
        cs = creators.score(st.creator)
        row: dict[str, Any] = {"mint": mint, "snapshot_ms": t, "delay_s": delay, "label_end_ms": t + horizon_ms,
                               "snap_price": st.price, "snap_liq": view.liquidity_sol, "creator_score": cs.score,
                               "creator_launches": cs.launches}
        row.update(view.as_dict(FEATURE_NAMES))
        row.update({f"rug_{k}": v for k, v in rug_features(view, st, cs).items()})
        rows.append(row)

    for r in events.select(list(EVENT_COLUMNS)).iter_rows():
        ev = Event(*r)
        while heap and heap[0][0] <= ev.ts_ms:
            t, _, mint, delay = heapq.heappop(heap)
            snapshot(t, mint, delay)
        resolver.advance(ev.ts_ms, market.tokens)
        st = market.on_event(ev)
        if ev.kind == EventKind.CREATE.value and st is not None:
            wallets.on_create(ev)
            resolver.on_create(st)
            discovery.on_create(st)
            for d in delays:
                seq += 1
                heapq.heappush(heap, (ev.ts_ms + int(d * 1000), seq, st.mint, d))
        feats.on_event(ev, st)
        if ev.kind in _TRADES and st is not None:
            wallets.on_trade(ev, st.created_ms, st.created_slot, st.creator)
    if not rows:
        return pl.DataFrame()
    snaps = pl.DataFrame(rows, infer_schema_length=None)
    return attach_labels(snaps, events, settings)


def attach_labels(snaps: pl.DataFrame, events: pl.DataFrame, settings: Any) -> pl.DataFrame:
    """Forward-looking labels from events strictly after each snapshot (see module docstring)."""
    dd = settings.rug_model.label_drawdown_pct / 100.0
    fwd_ms = int(settings.ml.fwd_return_horizon_s * 1000)
    tr = (events.filter(pl.col("kind").is_in(_TRADES))
          .select("mint", "ts_ms", (pl.col("v_sol").cast(pl.Float64) / pl.col("v_tok").cast(pl.Float64) / 1000.0).alias("px"),
                  (pl.col("r_sol").cast(pl.Float64) / 1e9).alias("liq")))
    mig = events.filter(pl.col("kind") == EventKind.MIGRATE.value).group_by("mint").agg(pl.col("ts_ms").min().alias("mig_ms"))
    j = snaps.select("mint", "snapshot_ms", "label_end_ms", "snap_price").join(tr, on="mint", how="left")
    fut = j.filter((pl.col("ts_ms") > pl.col("snapshot_ms")) & (pl.col("ts_ms") <= pl.col("label_end_ms")))
    agg = fut.group_by(["mint", "snapshot_ms"]).agg(pl.col("px").min().alias("min_px"), pl.col("liq").min().alias("min_liq"))
    fwd = (j.filter((pl.col("ts_ms") > pl.col("snapshot_ms")) & (pl.col("ts_ms") <= pl.col("snapshot_ms") + fwd_ms))
           .sort("ts_ms").group_by(["mint", "snapshot_ms"]).agg(pl.col("px").last().alias("fwd_px")))
    out = (snaps.join(agg, on=["mint", "snapshot_ms"], how="left").join(fwd, on=["mint", "snapshot_ms"], how="left")
           .join(mig, on="mint", how="left"))
    thr = settings.ml.fwd_return_threshold
    return out.with_columns(
        pl.col("min_px").fill_null(pl.col("snap_price")),
        pl.col("min_liq").fill_null(pl.col("snap_liq")),
        pl.col("fwd_px").fill_null(pl.col("snap_price")),
    ).with_columns(
        ((pl.col("snap_liq") > 0) & (pl.col("min_liq") <= pl.col("snap_liq") * (1.0 - dd))).cast(pl.Int8).alias("rug"),
        (pl.col("fwd_px") / pl.col("snap_price")).log().alias("fwd_return"),
        ((pl.col("mig_ms").is_not_null()) & (pl.col("mig_ms") > pl.col("snapshot_ms"))
         & (pl.col("mig_ms") <= pl.col("label_end_ms"))).cast(pl.Int8).alias("migrate"),
    ).with_columns((pl.col("fwd_return") >= math.log1p(thr)).cast(pl.Int8).alias("fwd_up"))


def feature_columns(df: pl.DataFrame, include_rug: bool = True) -> list[str]:
    base = [c for c in FEATURE_NAMES if c in df.columns] + ["creator_score", "creator_launches"]
    if include_rug:
        base += [f"rug_{k}" for k in RUG_FEATURES if f"rug_{k}" in df.columns and k not in FEATURE_NAMES]
    return base
