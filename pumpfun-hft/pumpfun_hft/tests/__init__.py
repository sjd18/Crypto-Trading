"""Test suite.

Layout (run ``pytest``; ``pytest -m "not slow"`` skips notebook execution)
    test_curve.py            exact bonding-curve math vs 792 SDK golden cases, invariants
    test_amm.py              PumpSwap constant product, mainnet fee tiers, own-impact shifts
    test_idl_events.py       Borsh codec round trips, old layouts, log / CPI extraction, AMM reserves
    test_features.py         online vs batch parity, windows, point-in-time (no future leakage)
    test_metrics.py          performance metrics, drawdowns, reason categories
    test_backtest.py         look-ahead canary, latency causality, ledger reconciliation, determinism,
                             order types, failures, own impact
    test_risk.py             sizing, risk limits, circuit breakers, position management
    test_optimizer.py        search on known functions, sealed splits, folds, DSR / PBO
    test_montecarlo.py       resampling, stress, ruin
    test_ml.py               purged CV, labels from the future only, look-ahead guard
    test_execution.py        live gateway against mocked RPC / Metis (sign, send, confirm, retry-after-expiry)
    test_live_engine.py      live trader with the paper gateway, queue priority, snapshots, virtual-time replay
    test_latency.py          latency tracker, decode / quote / evaluation budgets
    test_collectors.py       Parquet store, historical resume, live stream decoding, gaps
    test_config.py           strict config, overrides, secrets and redaction
    test_reports.py          report files, HTML escaping, dashboard pages and API
    test_cli.py              command-line workflows
    test_notebooks.py        executes the research notebooks (slow)
"""
