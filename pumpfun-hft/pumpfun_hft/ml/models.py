"""Model training, validation and explainability.

Supported models (``ml.model``): logistic regression, random forest (scikit-learn), XGBoost,
LightGBM, CatBoost (optional dependencies — a clear error is raised if missing).

``train_and_evaluate`` runs purged forward-chaining CV (:mod:`pumpfun_hft.ml.cv`) and reports AUC,
log loss, Brier score, base rate and precision in the top decile per fold; then fits the final
model on all rows and records ``train_end_ms`` (the latest label time the model has seen). The
saved bundle is loaded by :class:`~pumpfun_hft.ml.rug_model.TrainedRugModel`, which refuses to
score any event earlier than ``train_end_ms``.

Explainability: native importance (|coefficients| of the standardised logistic model or tree
impurity importance), permutation importance on the last CV fold, and SHAP mean |value| when the
``shap`` package is installed.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pumpfun_hft.ml.cv import PurgedForwardSplit


def make_model(kind: str, params: dict[str, Any], seed: int) -> Any:
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=seed, **params))
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, **params)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(random_state=seed, eval_metric="logloss", **params)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, **params)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(random_seed=seed, **params)
    raise ValueError(f"unknown model kind {kind}")


def native_importance(model: Any, features: list[str]) -> dict[str, float]:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    elif hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    else:
        return {}
    total = vals.sum() or 1.0
    return {f: float(v / total) for f, v in zip(features, vals, strict=False)}


def shap_importance(model: Any, x: np.ndarray, features: list[str], n_samples: int, seed: int) -> dict[str, float] | None:
    try:
        import shap
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    xs = x[rng.choice(len(x), size=min(n_samples, len(x)), replace=False)]
    try:
        with warnings.catch_warnings():  # shap warns about its own output-format changes; the handling is below
            warnings.simplefilter("ignore", UserWarning)
            if hasattr(model, "steps"):  # scaled linear pipeline
                scaler, est = model.steps[0][1], model.steps[-1][1]
                xt = scaler.transform(xs)
                vals = shap.LinearExplainer(est, xt).shap_values(xt)
            else:
                vals = shap.TreeExplainer(model).shap_values(xs)
            if isinstance(vals, list):
                vals = vals[-1]
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[:, :, -1]
    except Exception:  # noqa: BLE001 - explainability must never break training
        return None
    imp = np.abs(np.asarray(vals)).mean(axis=0)
    return {f: float(v) for f, v in zip(features, imp, strict=False)}


@dataclass
class MlReport:
    kind: str
    target: str
    features: list[str]
    folds: list[dict[str, float]]
    mean: dict[str, float]
    importance: pl.DataFrame
    train_end_ms: int
    n_rows: int
    base_rate: float
    bundle: dict[str, Any] = field(repr=False, default_factory=dict)

    def save(self, path: str | Path) -> Path:
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, p)
        return p


def _fold_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"n_test": float(len(y)), "base_rate": float(y.mean())}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    else:
        out["auc"] = out["log_loss"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    k = max(1, len(p) // 10)
    top = np.argsort(-p)[:k]
    out["precision_top_decile"] = float(y[top].mean())
    return out


class NotEnoughTrainingData(ValueError):
    """No usable training rows: no label window fully covered by the data, or only one outcome among them."""


def train_and_evaluate(df: pl.DataFrame, features: list[str], target: str, cfg: Any, seed: int) -> MlReport:
    """Purged CV + final fit + importance (see module docstring).

    Only rows whose label window the data fully covers are used (``fwd_complete`` for the fwd_*
    targets, ``label_complete`` otherwise; see :mod:`pumpfun_hft.ml.dataset`). For fwd_* targets the
    label window, and so the purge and ``train_end_ms``, ends at ``fwd_end_ms``."""
    fwd = target in ("fwd_up", "fwd_return")
    data = df.drop_nulls(subset=[target])
    flag = "fwd_complete" if fwd else "label_complete"
    if flag in data.columns:
        data = data.filter(pl.col(flag))
    if data.is_empty():
        raise NotEnoughTrainingData(f"no snapshots with a complete {target!r} label window: the data is too short (or too gappy) "
                         "for the label horizon")
    data = data.sort("snapshot_ms")
    x = data.select(features).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    y = data[target].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        raise NotEnoughTrainingData(f"all {len(y):,} usable snapshots have {target} = {int(y[0])}: a classifier needs both "
                                    "outcomes; use more data")
    times = data["snapshot_ms"].to_numpy()
    ends = data["fwd_end_ms" if fwd and "fwd_end_ms" in data.columns else "label_end_ms"].to_numpy()
    groups = data["mint"].to_numpy()
    params = dict(cfg.params.get(cfg.model, {}))
    cv = PurgedForwardSplit(cfg.cv_folds, int(cfg.embargo_s * 1000))
    folds: list[dict[str, float]] = []
    last: tuple[Any, np.ndarray, np.ndarray] | None = None
    oof_idx: list[np.ndarray] = []
    oof_p: list[np.ndarray] = []
    for i, (tr, te) in enumerate(cv.split(times, ends, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model(cfg.model, params, seed)
        m.fit(x[tr], y[tr])
        p = m.predict_proba(x[te])[:, 1]
        oof_idx.append(np.asarray(te))
        oof_p.append(p)
        fm = _fold_metrics(y[te], p)
        fm.update(fold=float(i), n_train=float(len(tr)), train_end_ms=float(ends[tr].max()), test_start_ms=float(times[te].min()))
        folds.append(fm)
        last = (m, x[te], y[te])
    final = make_model(cfg.model, params, seed)
    final.fit(x, y)
    imp_native = native_importance(final, features)
    imp_perm: dict[str, float] = {}
    if last is not None and len(np.unique(last[2])) == 2:
        pi = permutation_importance(last[0], last[1], last[2], n_repeats=cfg.permutation_repeats, random_state=seed, scoring="roc_auc")
        imp_perm = {f: float(v) for f, v in zip(features, pi.importances_mean, strict=True)}
    imp_shap = shap_importance(final, x, features, cfg.shap_samples, seed) or {}
    importance = pl.DataFrame({
        "feature": features,
        "native": [imp_native.get(f, float("nan")) for f in features],
        "permutation": [imp_perm.get(f, float("nan")) for f in features],
        "shap": [imp_shap.get(f, float("nan")) for f in features],
    }).sort("native", descending=True, nulls_last=True)
    keys = ["auc", "log_loss", "brier", "precision_top_decile", "base_rate"]

    def _mean(k: str) -> float:
        v = np.array([f[k] for f in folds], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    mean = {k: _mean(k) for k in keys}
    train_end = int(ends.max())
    bundle = {"model": final, "features": features, "train_end_ms": train_end, "kind": cfg.model, "target": target,
              "cv_mean": mean}
    if oof_idx:
        # out-of-fold scores (each from a model that never saw that row) with what then happened: lets ml_signal
        # pick its threshold from the score distribution and estimate the return above it honestly
        idx = np.concatenate(oof_idx)
        bundle["oof_probs"] = np.concatenate(oof_p).astype(np.float32)
        bundle["oof_target"] = y[idx].astype(np.int8)
        if "fwd_return" in data.columns:
            bundle["oof_fwd_return"] = np.expm1(data["fwd_return"].to_numpy()[idx]).astype(np.float32)  # simple return
    return MlReport(cfg.model, target, features, folds, mean, importance, train_end, len(y), float(y.mean()) if len(y) else math.nan, bundle)
