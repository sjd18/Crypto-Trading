"""Data science / ML module (optional).

Purpose
    Point-in-time prediction models (rug pull, forward return, migration) that plug into the
    strategies without any possibility of learning from the future.

Architecture
    dataset.py    snapshot datasets from the online feature stack + forward labels with windows
    cv.py         PurgedForwardSplit: forward-chaining CV with label purging, embargo, token groups
    models.py     logistic / random forest / XGBoost / LightGBM / CatBoost; CV metrics; native,
                  permutation and SHAP importance; persisted bundles with ``train_end_ms``
    rug_model.py  rug-pull probability (heuristic logistic score or trained model with a
                  look-ahead guard that refuses predictions before the training cut-off)

Data flow
    events -> build_snapshot_dataset -> train_and_evaluate (purged CV) -> bundle.joblib
           -> rug_model.use_trained_model=true -> StrategyRuntime.rug_probability

Inputs / Outputs
    Inputs: an event frame and token metadata (snapshots are built with the same online
    feature engine as live trading); model type, CV and label windows from the ``ml`` section.
    Outputs: per-fold CV metrics (AUC, log loss, Brier, top-decile precision), importance tables (native, permutation,
    SHAP), and a saved model bundle that records the last training timestamp.

Example
    ds = build_snapshot_dataset(settings, events, metadata)
    rep = train_and_evaluate(ds, feature_columns(ds), "rug", settings.ml, seed=7)
    rep.mean["auc"], rep.importance.head()
"""
