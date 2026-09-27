"""Analytics: wallet intelligence, performance metrics, Monte Carlo, charts and reports.

Purpose
    Turn raw activity and backtest results into decisions: which wallets are informed or
    dangerous, how good a run really is (after costs), how fragile it is under resampling and
    execution stress, and how to present all of it.

Architecture
    wallet_intel.py  online, point-in-time wallet profiles: round-trip PnL, posterior "smart"
                     score, sniper / whale / bot / market-maker / insider / rug-wallet labels,
                     union-find insider clusters
    metrics.py       Sharpe, Sortino, Omega, CAGR, Calmar, drawdown statistics, trade metrics
                     (win rate, profit factor, expectancy, Kelly, R multiples, MAE / MFE, streaks),
                     exit-reason categories, monthly / hourly profiles
    montecarlo.py    trade-level Monte Carlo (bootstrap / permutation / block bootstrap with
                     slippage, fee, latency and size perturbations; numba) and path-level
                     re-simulation through the event-driven backtester
    charts.py        shared Plotly theme and figure builders (validated palette, light / dark),
                     escaped sortable HTML tables
    report.py        HTML / PDF / CSV / JSON report generator for a BacktestResult

Data flow
    Events -> WalletIntel.on_trade / on_create / on_token_outcome (point-in-time)
    BacktestResult (equity, trades, fills) -> compute_metrics -> ReportGenerator -> files
    BacktestResult.trades -> trade_monte_carlo -> MonteCarloResult -> report / dashboard

Inputs / Outputs
    Inputs: normalised events, equity / trade / fill frames from the backtester or live session.
    Outputs: metric dicts, wallet tables, Monte Carlo distributions, report files and figures.

Example
    >>> from pumpfun_hft.analytics.metrics import reason_category
    >>> reason_category("take profit L2 +40.0%")
    'take profit'
"""
