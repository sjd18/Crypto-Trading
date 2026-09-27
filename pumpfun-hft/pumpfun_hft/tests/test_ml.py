"""ML: purged cross-validation, point-in-time datasets, labels from the future only, look-ahead guard."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pumpfun_hft.ml.cv import PurgedForwardSplit
from pumpfun_hft.ml.dataset import build_snapshot_dataset, feature_columns
from pumpfun_hft.ml.models import train_and_evaluate
from pumpfun_hft.ml.rug_model import RUG_FEATURES, HeuristicRugScorer, LookAheadError, TrainedRugModel, build_rug_scorer


def test_purged_split_never_leaks() -> None:
    rng = np.random.default_rng(0)
    n = 600
    times = np.sort(rng.integers(0, 10_000_000, n))
    label_end = times + rng.integers(0, 400_000, n)
    groups = rng.integers(0, 150, n)
    cv = PurgedForwardSplit(5, embargo_ms=100_000, min_train=10)
    folds = list(cv.split(times, label_end, groups))
    assert len(folds) >= 3
    for train, test in folds:
        t0 = times[test].min()
        assert (label_end[train] < t0).all()                 # purged: labels end before the test block
        assert (times[train] < t0 - 100_000).all()           # embargo
        assert not set(groups[train]) & set(groups[test])    # no token on both sides
        assert times[train].max() < times[test].min()        # forward only


@pytest.fixture(scope="module")
def dataset(settings, events, metadata) -> pl.DataFrame:
    return build_snapshot_dataset(settings, events, metadata)


def test_snapshot_dataset_shape_and_labels(dataset, settings) -> None:
    assert dataset.height > 50
    assert set(dataset["delay_s"].unique().to_list()) <= set(settings.rug_model.snapshot_delays_s)
    assert (dataset["label_end_ms"] - dataset["snapshot_ms"] == int(settings.rug_model.label_horizon_s * 1000)).all()
    assert 0.02 < dataset["rug"].mean() < 0.9
    cols = feature_columns(dataset)
    assert all(f"rug_{k}" in dataset.columns for k in RUG_FEATURES)
    assert "rug" not in cols and "fwd_return" not in cols and "min_liq" not in cols  # labels never become features


def test_snapshot_features_ignore_the_future(dataset, settings, events, metadata) -> None:
    """Features of snapshots before T are identical when every event after T is removed."""
    t_cut = int(events["ts_ms"].quantile(0.5))
    early = build_snapshot_dataset(settings, events.filter(pl.col("ts_ms") <= t_cut), metadata)
    cols = feature_columns(dataset)
    a = dataset.filter(pl.col("snapshot_ms") < t_cut).sort(["snapshot_ms", "mint"]).select(["mint", "snapshot_ms", *cols])
    b = early.filter(pl.col("snapshot_ms") < t_cut).sort(["snapshot_ms", "mint"]).select(["mint", "snapshot_ms", *cols])
    assert a.height == b.height > 20
    for c in cols:
        assert np.allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float), equal_nan=True), c


def test_train_evaluate_and_guard(dataset, settings, tmp_path) -> None:
    ml = settings.ml.model_copy(update={"model": "logistic", "cv_folds": 4})
    rep = train_and_evaluate(dataset, feature_columns(dataset), "rug", ml, 1)
    assert rep.folds and 0.5 < rep.mean["auc"] <= 1.0  # the planted rug structure is learnable
    assert {"feature", "native", "permutation"} <= set(rep.importance.columns)
    model = TrainedRugModel.load(rep.save(tmp_path / "m.joblib"))
    x = {f: 0.0 for f in model.features}
    assert 0.0 <= model.predict(x, now_ms=rep.train_end_ms + 1) <= 1.0
    with pytest.raises(LookAheadError):
        model.predict(x, now_ms=rep.train_end_ms - 1)


def test_heuristic_scorer_is_monotone_in_risk_features(settings) -> None:
    h = build_rug_scorer(settings.rug_model)
    assert isinstance(h, HeuristicRugScorer)
    base = {k: 0.0 for k in RUG_FEATURES}
    risky = {**base, "creator_sold_pct": 60.0, "bundled_unknown": 4.0, "creator_rug_rate": 0.9, "missing_socials": 1.0}
    assert h.predict(risky) > h.predict(base)
    assert 0.0 < h.predict(base) < 0.1
