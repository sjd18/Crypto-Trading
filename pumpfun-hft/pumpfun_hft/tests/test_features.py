"""Features: online/batch parity, point-in-time correctness, window mechanics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pumpfun_hft.features.batch import batch_features, candles
from pumpfun_hft.features.online import SlidingMax, TimeSeriesBuffer
from pumpfun_hft.features.registry import FEATURE_NAMES, FEATURES
from pumpfun_hft.features.replay import online_features

PARITY = ["buy_sol_short", "sell_sol_short", "buy_sol_medium", "sell_sol_medium", "delta_sol_medium", "volume_sol_long",
          "imbalance_medium", "imbalance_long", "n_trades_medium", "vwap_dist_medium", "ret_short", "ret_medium", "ret_long",
          "progress_pct", "liquidity_sol", "age_s", "hour_sin", "hour_cos"]


@pytest.fixture(scope="module")
def online(settings, events) -> pl.DataFrame:
    return online_features(settings, events)


def test_online_and_batch_features_agree_exactly(settings, events, online) -> None:
    batch = batch_features(events, settings.features, settings.protocol.curve.initial_real_token_reserves)
    assert online.height == batch.height > 1000
    assert online["mint"].to_list() == batch["mint"].to_list()
    for c in PARITY:
        a, b = online[c].to_numpy().astype(float), batch[c].to_numpy().astype(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), c


def test_every_registered_feature_is_computable(online) -> None:
    assert set(FEATURE_NAMES) <= set(online.columns)
    for name in FEATURE_NAMES:
        col = online[name].to_numpy().astype(float)
        assert not np.isinf(col).any(), name
    assert len({f.name for f in FEATURES}) == len(FEATURES)  # unique names


def test_features_are_point_in_time(settings, events, online) -> None:
    """Values up to time T must not change when everything after T is removed (no future leakage)."""
    t_cut = int(events["ts_ms"].quantile(0.5))
    early = online_features(settings, events.filter(pl.col("ts_ms") <= t_cut))
    ref = online.filter(pl.col("ts_ms") <= t_cut)
    assert early.height == ref.height > 100
    for c in FEATURE_NAMES:  # includes wallet-derived features (smart / fresh / bot shares)
        a, b = early[c].to_numpy().astype(float), ref[c].to_numpy().astype(float)
        assert np.allclose(a, b, rtol=0, atol=0, equal_nan=True), c


def test_batch_features_are_point_in_time(settings, events) -> None:
    t_cut = int(events["ts_ms"].quantile(0.4))
    full = batch_features(events, settings.features, settings.protocol.curve.initial_real_token_reserves)
    part = batch_features(events.filter(pl.col("ts_ms") <= t_cut), settings.features, settings.protocol.curve.initial_real_token_reserves)
    ref = full.filter(pl.col("ts_ms") <= t_cut)
    for c in PARITY:
        assert np.allclose(part[c].to_numpy().astype(float), ref[c].to_numpy().astype(float), equal_nan=True), c


def test_time_series_buffer_asof_and_eviction() -> None:
    b = TimeSeriesBuffer(span_ms=1_000)
    for t, v in [(0, 1.0), (500, 2.0), (1_000, 3.0), (1_600, 4.0)]:
        b.append(t, v)
    assert b.asof(499) == 1.0
    assert b.asof(1_000) == 3.0
    assert b.asof(-1) is None
    b.evict(2_000)
    assert b.asof(1_000) in (None, 3.0)  # evicted history may be dropped, never invented
    assert b.asof(2_000) == 4.0


def test_sliding_max_tracks_window_maximum() -> None:
    m = SlidingMax(span_ms=1_000)
    for t, v in [(0, 5.0), (200, 3.0), (400, 4.0), (1_100, 1.0)]:
        m.push(t, v)
    assert m.value(400) == 5.0
    assert m.value(1_100) == 4.0  # the 5.0 at t=0 left the window


def test_candles_ohlcv(events) -> None:
    c = candles(events, 60_000)
    assert c.height > 0
    row = c.row(0, named=True)
    assert row["high"] >= max(row["open"], row["close"]) and row["low"] <= min(row["open"], row["close"])
    assert all(v >= 0 for v in c["volume_sol"].to_list())
    assert not any(math.isnan(v) for v in c["close"].to_list())
