"""pumpfun_hft — quantitative research, backtesting and execution platform for Pump.fun.

Purpose
    End-to-end framework for discovering, researching, validating and (optionally) trading
    Pump.fun bonding-curve tokens on Solana with a strict separation between historical
    simulation and live execution.

Architecture (package map)
    core/         protocol primitives: exact bonding-curve & AMM math, IDL decoder, event model, config
    api/          network clients: Solana RPC/WS, QuickNode Metis (public + JWT), rate limiting, retries
    collectors/   historical + live collectors, Parquet store, synthetic market generator
    discovery/    token discovery, metadata, creator history & statistical creator scoring
    features/     incremental (online) and vectorized (batch) timestamp-safe feature pipelines
    strategies/   strategy framework + 10 reference strategies + shared strategy runtime
    backtester/   event-driven replay engine with a realistic execution simulator
    risk/         sizing, position management, limits and circuit breakers
    optimizer/    grid/random/Bayesian/genetic search, sealed splits, walk-forward, DSR/PBO
    analytics/    performance metrics, Monte Carlo, wallet intelligence, report generator
    ml/           point-in-time datasets, purged CV, models, rug-pull model, SHAP
    execution/    live execution engine (queue, retry, confirm, blockhash, priority fees, Jito)
    dashboard/    FastAPI + Plotly dashboard and static export
    database/     DuckDB warehouse + SQLite metadata schemas
    utils/        logging, latency tracking, retry, hashing, time helpers

Data flow
    collectors -> Parquet (day partitions) -> backtester replay -> strategy runtime
    -> execution simulator -> portfolio -> analytics/reports ; the same strategy runtime is
    driven by the live collector + live execution engine in paper/live mode.
"""

__version__ = "0.1.0"
