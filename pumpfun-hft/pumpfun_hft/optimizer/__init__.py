"""Optimisation and validation engine.

Purpose
    Large-scale parameter search that never touches test data, plus statistics that quantify
    how much of a backtest is selection bias.

Architecture
    space.py        ParamSpace: float / int / log-float / categorical; sampling, grids, [0,1]^d encoding
    search.py       grid, random, Bayesian (GP Matérn-5/2 + EI, constant-liar batches), genetic
    objective.py    objective functions with min-trade and max-drawdown constraints
    splits.py       embargoed train / validation / test / live-sim splits (test & live-sim sealed),
                    rolling or anchored walk-forward folds
    overfitting.py  Probabilistic & Deflated Sharpe Ratio, PBO via CSCV
    study.py        OptimizationStudy and WalkForwardAnalysis (process-pool evaluation)

Data flow
    events -> make_splits -> search(train) -> top-k -> validation selection -> DSR/PBO
           -> (explicit unseal) test + live-sim, each evaluated exactly once

Inputs / Outputs
    Inputs: an event frame, token metadata, a strategy name and its search space from
    ``optimizer.spaces``; method, budget, splits and constraints from the ``optimizer`` section.
    Outputs: a trials table (params, train / validation metrics), the selected parameters, DSR,
    PBO, an unseal audit log, and (only when unsealed) one test and one live-sim evaluation.

Example
    res = OptimizationStudy(settings, events, meta_path, "momentum_ignition", method="bayesian").run(final_evaluation=True)
    wf = WalkForwardAnalysis(settings, events, meta_path, "momentum_ignition").run()
"""
