"""Rug-pull probability model.

Features (all point-in-time)
    creator ownership (creator_holding_pct), creator selling (creator_sold_pct), holder
    concentration (top10_pct, hhi), sell pressure (medium-window sell share), liquidity removal
    (liquidity_drop_pct from peak), wallet clustering / transaction graph (bundled_buyers: non-
    creator wallets buying in the creation slot; bundled_unknown: those not recognised as recurring
    snipers / bots, i.e. the likely insider bundle), metadata anomalies (missing_socials,
    duplicate_name) and the creator's posterior rug rate.

Scorers
    * :class:`HeuristicRugScorer` — logistic score with configurable weights (no training data
      needed; default).
    * :class:`TrainedRugModel` — any fitted scikit-learn-compatible classifier saved by
      ``ml.models.train_model``; refuses to score events earlier than its training cut-off
      (:class:`LookAheadError`), so a model can never be used on the data it learned from.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

RUG_FEATURES: tuple[str, ...] = (
    "creator_holding_pct", "creator_sold_pct", "top10_pct", "hhi", "sell_pressure", "liquidity_drop_pct",
    "bundled_buyers", "bundled_unknown", "missing_socials", "duplicate_name", "creator_rug_rate",
)


class LookAheadError(RuntimeError):
    """A model trained on data up to T was asked to score an event before T."""


def rug_features(view: Any, st: Any, creator_score: Any | None) -> dict[str, float]:
    """Assemble the rug feature vector from a FeatureView + TokenState + CreatorScore."""
    buy, sell = view.buy_sol_medium, view.sell_sol_medium  # medium window: a single 5 s sell burst is not a rug
    meta = getattr(st, "metadata", None)
    anomalies = getattr(meta, "anomalies", None) or []
    has_socials = bool(getattr(meta, "has_socials", False)) if meta is not None else False
    return {
        "creator_holding_pct": view.creator_holding_pct,
        "creator_sold_pct": view.creator_sold_pct,
        "top10_pct": view.top10_pct,
        "hhi": view.hhi,
        "sell_pressure": sell / (buy + sell) if (buy + sell) > 1e-12 else 0.0,
        "liquidity_drop_pct": view.liquidity_drop_pct,
        "bundled_buyers": float(view.bundled_buyers),
        "bundled_unknown": float(view.bundled_unknown),
        "missing_socials": 0.0 if has_socials else 1.0,
        "duplicate_name": 1.0 if "duplicate_name" in anomalies else 0.0,
        "creator_rug_rate": float(creator_score.p_rug) if creator_score is not None else 1.0 / 3.0,
    }


class HeuristicRugScorer:
    """sigmoid(intercept + sum_i w_i * x_i) with weights from ``rug_model.heuristic_weights``."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.intercept = float(weights.get("intercept", 0.0))
        self.w = {k: float(v) for k, v in weights.items() if k != "intercept"}

    def predict(self, x: dict[str, float], now_ms: int | None = None) -> float:
        z = self.intercept + sum(w * float(x.get(k, 0.0)) for k, w in self.w.items())
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)


class TrainedRugModel:
    """Wraps a persisted classifier bundle ``{"model", "features", "train_end_ms", "kind"}``."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        target = bundle.get("target", "rug")
        if target != "rug":  # e.g. a fwd_up model: P(up) read as P(rug) would block the best tokens
            raise ValueError(f"this model predicts {target!r}, not 'rug'; trade it with the ml_signal strategy "
                             "(strategy.params.ml_signal.model_path) instead of rug_model.model_path")
        self.model = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        self.train_end_ms = int(bundle["train_end_ms"])
        self.kind = bundle.get("kind", "unknown")

    @classmethod
    def load(cls, path: str | Path) -> TrainedRugModel:
        import joblib

        return cls(joblib.load(path))

    def predict(self, x: dict[str, float], now_ms: int | None = None) -> float:
        if now_ms is not None and now_ms < self.train_end_ms:
            raise LookAheadError(f"model trained through {self.train_end_ms} used at {now_ms}")
        import numpy as np

        row = np.array([[float(x.get(f, 0.0)) for f in self.features]])
        return float(self.model.predict_proba(row)[0, 1])


def resolve_rug_model_path(model_path: str, models_dir: Path | None = None) -> Path:
    """``rug_model.model_path``: absolute, or relative to the data set's models folder (``models_dir``;
    a bare file name such as ``lightgbm-rug-1759000000.joblib`` is enough), or - without a models
    folder - to the project root."""
    from pumpfun_hft.core.config import PROJECT_ROOT

    p = Path(model_path)
    if p.is_absolute():
        candidates = [p]
    elif models_dir is not None:
        candidates = [Path(models_dir) / p, Path(models_dir) / p.name]
    else:
        candidates = [PROJECT_ROOT / p]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"rug_model.use_trained_model is true but the model file was not found (looked for "
                            f"{', '.join(str(c) for c in candidates)}); train one with `train-model --target rug` and set "
                            "rug_model.model_path to its file name, or set use_trained_model: false")


def build_rug_scorer(cfg: Any, models_dir: Path | None = None, dataset: str = "") -> HeuristicRugScorer | TrainedRugModel:
    """Trained model when ``cfg.use_trained_model`` (a missing file is an error, never a silent fallback), heuristic otherwise.

    ``dataset`` is the active data set: a model trained on the other one is refused."""
    if not cfg.use_trained_model:
        return HeuristicRugScorer(cfg.heuristic_weights)
    import joblib

    path = resolve_rug_model_path(cfg.model_path, models_dir)
    bundle = joblib.load(path)
    trained_on = bundle.get("dataset")
    if dataset and trained_on and trained_on != dataset:
        raise ValueError(f"rug model {path.name} was trained on the {trained_on} data set; this run uses the {dataset} "
                         "data set. Train one on this data set with `train-model --target rug`.")
    return TrainedRugModel(bundle)
