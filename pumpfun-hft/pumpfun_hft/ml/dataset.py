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

Every row carries ``snapshot_ms`` and ``label_end_ms`` (and ``fwd_end_ms`` for the fwd_* labels)
so cross-validation can purge samples whose label window overlaps the test period.

A label is only trustworthy if the data covers its whole window. ``label_complete`` /
``fwd_complete`` are false when the window runs past the last event (the end of the data, or of
``train-model --end``) or, with ``max_gap_s`` set (real data), across a stretch with no events at
all for longer than that - a recording gap between two ``stream`` sessions, where "no trade seen"
means "not recorded", not "price unchanged". Training uses only rows with a complete window.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
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


#: every column ``model_features`` produces (the inputs a trained model may use)
MODEL_FEATURE_NAMES: frozenset[str] = frozenset(("creator_score", "creator_launches", *FEATURE_NAMES,
                                                 *(f"rug_{k}" for k in RUG_FEATURES)))


def model_features(view: Any, st: Any, cs: Any) -> dict[str, float]:
    """Every model input for one token at one instant: online features, creator score, rug features.

    The single source of the feature row for both training (``build_snapshot_dataset``) and live
    scoring (``strategies.ml_signal``), so a trained model sees the same inputs in both.
    """
    row: dict[str, float] = {"creator_score": float(cs.score), "creator_launches": float(cs.launches)}
    row.update(view.as_dict(FEATURE_NAMES))
    row.update({f"rug_{k}": v for k, v in rug_features(view, st, cs).items()})
    return row


def build_snapshot_dataset(settings: Any, events: pl.DataFrame, metadata: dict[str, Any] | None = None,
                           delays_s: list[float] | None = None, horizon_s: float | None = None,
                           max_gap_s: float | None = None) -> pl.DataFrame:
    """Replay ``events`` and return one feature row per (token, snapshot delay) with labels.

    ``max_gap_s``: treat a market-wide silence longer than this as missing data (see module docstring);
    ``None`` for synthetic data, whose quiet stretches are real lulls rather than recording gaps."""
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
                               "snap_price": st.price, "snap_liq": view.liquidity_sol}
        row.update(model_features(view, st, cs))
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
    return attach_labels(snaps, events, settings, max_gap_s)


def window_complete(start_ms: np.ndarray, end_ms: np.ndarray, event_ts: np.ndarray, max_gap_ms: int | None) -> np.ndarray:
    """True where ``[start, end]`` ends by the last event and (with ``max_gap_ms``) spans no silence longer than it."""
    ts = np.unique(event_ts)
    if not len(ts):
        return np.zeros(len(start_ms), dtype=bool)
    ok = end_ms <= ts[-1]
    if max_gap_ms is not None and len(ts) > 1:
        gi = np.nonzero(np.diff(ts) > max_gap_ms)[0]
        g0, g1 = ts[gi], ts[gi + 1]  # silent stretches (g0, g1), sorted and disjoint
        if len(g0):
            k = np.searchsorted(g0, end_ms, side="left")  # gaps starting before the window ends
            last_end = np.where(k > 0, g1[np.maximum(k - 1, 0)], np.iinfo(np.int64).min)
            ok &= ~(last_end > start_ms)  # ... of which the latest still reaches into the window
    return ok


def attach_labels(snaps: pl.DataFrame, events: pl.DataFrame, settings: Any, max_gap_s: float | None = None) -> pl.DataFrame:
    """Forward-looking labels from events strictly after each snapshot (see module docstring)."""
    dd = settings.rug_model.label_drawdown_pct / 100.0
    fwd_ms = int(settings.ml.fwd_return_horizon_s * 1000)
    ts = events["ts_ms"].to_numpy()
    gap_ms = None if max_gap_s is None else int(max_gap_s * 1000)
    t0 = snaps["snapshot_ms"].to_numpy().astype(np.int64)
    snaps = snaps.with_columns(
        (pl.col("snapshot_ms") + fwd_ms).alias("fwd_end_ms"),
        pl.Series("label_complete", window_complete(t0, snaps["label_end_ms"].to_numpy().astype(np.int64), ts, gap_ms)),
        pl.Series("fwd_complete", window_complete(t0, t0 + fwd_ms, ts, gap_ms)),
    )
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
