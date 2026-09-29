"""Deploying trained models: the ml_signal strategy (fwd_up models) and the trained rug scorer.

Checks that live scoring feeds a model exactly the columns it was trained on, that a model never
trades before its training cut-off, and that a model can't be plugged into the wrong slot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource
from pumpfun_hft.core.types import Side
from pumpfun_hft.ml.dataset import MODEL_FEATURE_NAMES, build_snapshot_dataset, feature_columns
from pumpfun_hft.ml.models import train_and_evaluate
from pumpfun_hft.ml.rug_model import TrainedRugModel
from pumpfun_hft.strategies.base import build_strategy
from pumpfun_hft.strategies.ml_signal import MlSignal, resolve_model_path


class RecordingModel:
    """Picklable stand-in classifier that records the rows it is asked to score."""

    def __init__(self, n_features: int, p: float) -> None:
        self.n = n_features
        self.p = p
        self.rows: list[np.ndarray] = []

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, self.n)
        self.rows.append(x[0].copy())
        return np.array([[1.0 - self.p, self.p]])


@pytest.fixture(scope="module")
def split(settings: Any, events: pl.DataFrame) -> int:
    return int(events["ts_ms"].min()) + (int(events["ts_ms"].max()) - int(events["ts_ms"].min())) // 2


@pytest.fixture(scope="module")
def fwd_model(settings: Any, events: pl.DataFrame, metadata: dict[str, Any], split: int, tmp_path_factory: Any) -> Path:
    """A logistic fwd_up model trained on the first half of the market, saved like `train-model` does."""
    ds = build_snapshot_dataset(settings, events.filter(pl.col("ts_ms") <= split), metadata)
    ml = settings.ml.model_copy(update={"model": "logistic", "cv_folds": 3})
    rep = train_and_evaluate(ds, feature_columns(ds), "fwd_up", ml, 1)
    rep.bundle.update({"snapshot_delays_s": list(settings.rug_model.snapshot_delays_s),
                       "fwd_return_horizon_s": settings.ml.fwd_return_horizon_s,
                       "fwd_return_threshold": settings.ml.fwd_return_threshold})
    d = tmp_path_factory.mktemp("models")
    return rep.save(d / f"logistic-fwd_up-{split}.joblib")


def _strategy(settings: Any, path: Path, **params: Any) -> MlSignal:
    s = settings.model_copy(deep=True)
    s.strategy.params["ml_signal"] = {**s.strategy.params["ml_signal"], "model_path": str(path), **params}
    return build_strategy("ml_signal", s)  # type: ignore[return-value]


def test_model_features_cover_every_training_column(settings: Any, events: pl.DataFrame, metadata: dict[str, Any]) -> None:
    ds = build_snapshot_dataset(settings, events.head(40_000), metadata)
    cols = feature_columns(ds)
    assert cols and set(cols) <= MODEL_FEATURE_NAMES  # live scoring can rebuild every column a model trains on


def test_latest_resolves_newest_fwd_up_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Train one first"):
        resolve_model_path("latest", tmp_path)
    (tmp_path / "lightgbm-rug-1.joblib").write_bytes(b"")
    old, new = tmp_path / "lightgbm-fwd_up-1.joblib", tmp_path / "lightgbm-fwd_up-2.joblib"
    old.write_bytes(b"")
    new.write_bytes(b"")
    import os

    os.utime(old, (1, 1))
    assert resolve_model_path("latest", tmp_path) == new
    assert resolve_model_path("lightgbm-fwd_up-1.joblib", tmp_path) == old


def test_models_cannot_go_in_the_wrong_slot(settings: Any, fwd_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ml_signal"):
        TrainedRugModel(joblib.load(fwd_model))  # a fwd_up model is not a rug model
    rug = {**joblib.load(fwd_model), "target": "rug"}
    joblib.dump(rug, tmp_path / "rug.joblib")
    with pytest.raises(ValueError, match="rug_model.model_path"):
        _strategy(settings, tmp_path / "rug.joblib")
    stale = {**joblib.load(fwd_model), "features": ["creator_score", "feature_from_another_version"]}
    joblib.dump(stale, tmp_path / "stale.joblib")
    with pytest.raises(ValueError, match="retrain"):
        _strategy(settings, tmp_path / "stale.joblib")


def test_ml_signal_trades_only_after_the_cutoff(settings: Any, events: pl.DataFrame, metadata: dict[str, Any], fwd_model: Path,
                                               tmp_path: Path) -> None:
    b = joblib.load(fwd_model)
    b["oof_fwd_return"] = np.full(len(b["oof_probs"]), 0.5, dtype=np.float32)  # an edge that clears the cost gate
    joblib.dump(b, tmp_path / "edge.joblib")
    strat = _strategy(settings, tmp_path / "edge.joblib", min_prob=1e-9, max_rug_prob=1.0)  # buy every scored snapshot
    cut = strat.train_end_ms
    assert int(events["ts_ms"].min()) < cut < int(events["ts_ms"].max())
    res = BacktestEngine(settings, DataSource(frame=events), [strat], metadata=metadata, seed=1).run()
    assert strat.skipped_before_cutoff > 0 and strat.scored > 0
    buys = res.fills.filter((pl.col("strategy") == "ml_signal") & (pl.col("side") == Side.BUY.value))
    assert buys.height > 0
    assert (buys["decision_ms"] >= cut).all()  # never an entry on data the model learned from


def test_live_rows_match_the_training_columns(settings: Any, events: pl.DataFrame, metadata: dict[str, Any], fwd_model: Path) -> None:
    """Every row scored in a backtest has the model's columns, in order, and real (non-default) values."""
    strat = _strategy(settings, fwd_model, min_prob=0.99)  # the recording model says 0.5: never buys, only scores
    rec = RecordingModel(len(strat.features), 0.5)
    strat.model = rec
    BacktestEngine(settings, DataSource(frame=events), [strat], metadata=metadata, seed=1).run()
    rows = np.array(rec.rows)
    assert rows.shape[0] == strat.scored > 20
    nonzero = (rows != 0).mean(axis=0)
    idx = {f: i for i, f in enumerate(strat.features)}
    for f in ("age_s", "creator_score", "rug_sell_pressure", "rug_missing_socials"):
        if f in idx:
            assert nonzero[idx[f]] > 0, f  # populated, not silently zero-filled
    # at most one score per token per snapshot delay
    assert strat.scored <= len(strat.delays_s) * len(strat._next)


def test_trained_rug_model_receives_the_full_training_row(settings: Any, events: pl.DataFrame, metadata: dict[str, Any],
                                                         fwd_model: Path) -> None:
    bundle = {**joblib.load(fwd_model), "target": "rug"}
    feats = bundle["features"]
    rec = RecordingModel(len(feats), 0.0)
    scorer = TrainedRugModel({**bundle, "model": rec, "train_end_ms": 0})
    BacktestEngine(settings, DataSource(frame=events.head(60_000)), ["momentum_ignition"], metadata=metadata, seed=1,
                   rug_scorer=scorer).run()
    rows = np.array(rec.rows)
    assert rows.shape[0] > 0
    filled = (rows != 0).any(axis=0)
    idx = {f: i for i, f in enumerate(feats)}
    for f in ("rug_sell_pressure", "rug_missing_socials", "creator_score"):
        if f in idx:
            assert filled[idx[f]], f  # previously these were never passed and silently read as 0


def test_threshold_comes_from_the_out_of_fold_scores(settings: Any, fwd_model: Path) -> None:
    """A model's probabilities sit near its base rate: the default threshold is its own top 10 %, not a fixed 0.6."""
    bundle = joblib.load(fwd_model)
    oof = np.asarray(bundle["oof_probs"], dtype=float)
    assert len(oof) == len(bundle["oof_target"]) == len(bundle["oof_fwd_return"]) > 20
    strat = _strategy(settings, fwd_model)
    assert strat.min_score == pytest.approx(float(np.quantile(oof, 0.9)))
    assert 88.0 <= strat.confidence(strat.min_score) <= 100.0  # confidence = percentile: the top 10 % clears min_confidence
    assert strat.confidence(float(oof.min()) - 1.0) == 0.0
    assert strat.oof_n >= 0.09 * len(oof) and strat.expected_return is not None
    assert any("top 10%" in line for line in strat.describe())
    assert _strategy(settings, fwd_model, top_frac=0.02).min_score >= strat.min_score
    assert _strategy(settings, fwd_model, min_prob=0.7).min_score == 0.7
    table = MlSignal.threshold_table(bundle)
    assert [r["top_frac"] for r in table] == [0.01, 0.02, 0.05, 0.1, 0.2]
    assert all(a["threshold"] >= b["threshold"] for a, b in zip(table, table[1:], strict=False))
    for bad in ({"top_frac": 0.0}, {"top_frac": 1.5}, {"min_prob": 1.0}):
        with pytest.raises(ValueError):
            _strategy(settings, fwd_model, **bad)


def test_models_without_out_of_fold_scores_fall_back(settings: Any, fwd_model: Path, tmp_path: Path) -> None:
    old = {k: v for k, v in joblib.load(fwd_model).items() if not k.startswith("oof_")}
    joblib.dump(old, tmp_path / "old.joblib")
    strat = _strategy(settings, tmp_path / "old.joblib")
    assert strat.min_score == 0.6 and strat.expected_return is None and strat.confidence(0.7) == pytest.approx(70.0)
    assert "retrain" in strat.describe()[0]


def test_negative_edge_is_reported_and_gated(settings: Any, fwd_model: Path, tmp_path: Path, events: pl.DataFrame,
                                             metadata: dict[str, Any]) -> None:
    b = joblib.load(fwd_model)
    b["oof_fwd_return"] = np.full(len(b["oof_probs"]), -0.05, dtype=np.float32)  # everything above the threshold lost money
    joblib.dump(b, tmp_path / "loser.joblib")
    strat = _strategy(settings, tmp_path / "loser.joblib", top_frac=1.0, max_rug_prob=1.0)
    assert strat.expected_return == pytest.approx(-0.05) and any("lost money" in x for x in strat.describe())
    eng = BacktestEngine(settings, DataSource(frame=events), [strat], metadata=metadata, seed=1)
    res = eng.run()
    assert res.trades.filter(pl.col("strategy") == "ml_signal").height == 0 if "strategy" in res.trades.columns else res.trades.height == 0
    funnel = strat.funnel(eng.runtime.signal_records)
    assert "signals ->" in funnel and "cost_gate" in funnel
