# pumpfun-hft

A modular research, backtesting and execution platform for Pump.fun bonding-curve markets
(and their PumpSwap AMM pools after graduation) on Solana. Everything runs locally: Python 3.13,
asyncio, Polars / DuckDB / Parquet, Numba, Plotly, FastAPI, Typer.

> **Read this first.** The platform is engineered to make research honest (exact protocol math,
> zero look-ahead, every cost charged, sealed test data, overfitting statistics). It does not
> contain a profitable strategy, and the bundled synthetic market only exists to exercise the
> pipeline — results on it say nothing about live profitability. The live execution path has
> not been run against mainnet with real funds; treat it as code to review and paper-trade
> before any real deployment.

## What is in the box

| Area | Module | Highlights |
|---|---|---|
| Protocol | `core/` | Exact integer bonding-curve math matching `@pump-fun/pump-sdk` (792 golden cases, 0 mismatches), PumpSwap constant-product math with the 25 canonical fee tiers decoded from mainnet, IDL-driven Borsh codec for events / accounts / instructions, typed config with no hidden defaults |
| APIs | `api/` | Solana JSON-RPC + WebSocket clients, Metis client (Mode A public, Mode B JWT), pump.fun metadata / holder analytics, token-bucket + AIMD rate limiting, retries with jitter, circuit breaker |
| Data | `collectors/`, `database/` | Historical backfill (resumable checkpoints, batching, retries, gap registry), live stream (logs + slots, reconnect / resubscribe, gap backfill, dispatch-latency budget), Parquet by day with checksums, dedupe and compaction, DuckDB warehouse, SQLite state, synthetic market generator |
| Discovery | `discovery/` | New-token scanner with metadata / socials / sector enrichment, creator scoring (Beta posteriors, Wilson bounds), point-in-time outcome resolution |
| Features | `features/` | ~55 timestamp-safe features (price, volume, wallet, curve, time); online engine and vectorised batch pipeline with exact parity |
| Wallets | `analytics/wallet_intel.py` | Round-trip PnL, posterior "smart" score, sniper / whale / bot / market-maker / insider / rug-wallet labels, insider clusters |
| Strategies | `strategies/` | 10 strategies behind one `generate_signal()` interface (BUY / SELL / HOLD / EXIT / SCALE_IN / SCALE_OUT, confidence 0-100) plus a shared runtime (sizing, cost gate, risk, retries) |
| Backtester | `backtester/` | Event / trade / tick / candle replay, market / limit / IOC / FOK orders, persistent own impact, fees, priority fees, Jito tips, rent, lognormal latency, drops, failed transactions, congestion, blockhash expiry, outages, rate limits, partial fills |
| Risk | `risk/` | Fixed / fixed-risk / Kelly / volatility / confidence / max-exposure sizing, liquidity cap, stop loss, take-profit ladder, trailing stop, breakeven, max hold, pyramiding, loss limits, exposure limits, circuit breakers |
| Validation | `optimizer/`, `analytics/montecarlo.py` | Grid / random / Bayesian (GP + EI) / genetic search, train-validation-test-live-sim splits with a sealed test set and audit log, walk-forward, deflated Sharpe, PBO (CSCV), trade- and path-level Monte Carlo |
| ML | `ml/` | Point-in-time snapshot datasets, purged forward-chaining CV with embargo, logistic / RF / XGBoost / LightGBM / CatBoost, permutation importance and SHAP, a look-ahead guard on trained models |
| Live | `execution/` | Priority queue of orders, quote → build (Metis) → sign → send (RPC or Jito) → confirm, blockhash cache, dynamic priority fees, rebroadcast, retry only after the blockhash has expired, persisted order state, paper gateway |
| Reporting | `analytics/report.py`, `dashboard/` | HTML / PDF / CSV / JSON reports after every backtest; local FastAPI dashboard (10 pages) and a single-file static export |

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[live,ml,research,dev]"          # or: pip install -r requirements.txt (exact tested versions)

python -m pumpfun_hft.main init                   # folders + .env from .env.example
python -m pumpfun_hft.main synth --hours 12       # synthetic market (demo / tests only)
python -m pumpfun_hft.main backtest               # strategy.active; report in pumpfun_hft/reports/runs/<id>/report/
python -m pumpfun_hft.main backtest --strategy smart_money --strategy momentum_ignition
python -m pumpfun_hft.main dashboard              # http://127.0.0.1:8050
python -m pumpfun_hft.main dashboard-export       # one self-contained HTML file
pytest                                            # full test suite (pytest -m "not slow" to skip notebooks)
```

On Windows, `hft.ps1` saves retyping all of that: load it once per PowerShell window with
`. .\hft.ps1` (edit the two paths at the top first), then run `hft synth --hours 24`,
`hft backtest --strategy smart_money`, and so on. `hft-help` prints a cheat sheet of every command.

Real data instead of the synthetic market:

```bash
# put SOLANA_RPC_URL / SOLANA_WS_URL in .env (a paid RPC is strongly recommended)
python -m pumpfun_hft.main collect-history --max-signatures 20000    # resumable backfill
python -m pumpfun_hft.main stream --minutes 60                       # live recorder + latency stats
python -m pumpfun_hft.main verify-data                               # checksums + slot gaps
```

## Commands

| Command | What it does |
|---|---|
| `init`, `check-config` | Create folders and `.env`; validate config and show which secrets are present (never their values) |
| `synth` | Generate a synthetic market into the event store |
| `collect-history`, `stream`, `verify-data` | Historical backfill, live recording, integrity checks |
| `backtest` | Event-driven backtest; saves the run, the report (HTML / PDF / CSV / JSON) and a Monte Carlo summary |
| `optimize`, `walkforward` | Parameter search on train, selection on validation, optional sealed final evaluation; walk-forward |
| `montecarlo`, `report` | Re-run Monte Carlo or regenerate a report for a saved run |
| `train-model`, `wallets` | Train and evaluate an ML model with purged CV; show top wallets of a run |
| `dashboard`, `dashboard-export` | Local dashboard server; static single-file export |
| `paper-replay` | Recorded events through the live engine (paper gateway) on virtual time, compared with a backtest of the same events |
| `paper`, `live` | Paper trading on the live stream; live trading (requires `app.mode: live` **and** `--confirm-live`) |
| `latency-probe`, `update-idl`, `docs` | Measure RPC / Metis latency; refresh IDLs; regenerate `docs/MODULES.md` |

Global options: `--config my.yaml` (deep-merged over `configs/default.yaml`) and repeatable
`--set dotted.key=value` overrides, e.g. `--set backtest.initial_capital_sol=5`.

## Configuration and secrets

* Every tunable lives in `pumpfun_hft/configs/default.yaml`. The typed schema in
  `core/config.py` has **no defaults**: a missing key or an unknown key fails validation, so
  nothing is silently hardcoded. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
* Secrets are read only from `.env` / the environment (`PUMPFUN_JWT`, `METIS_URL`,
  `SOLANA_RPC_URL`, `SOLANA_WS_URL`, `PRIVATE_KEY`, …), wrapped in `SecretStr`, registered with
  the log redactor, and never written to disk, logs, reports or the dashboard.
* **Mode A** (`network.metis.mode: public`) uses `https://public.jupiterapi.com`. QuickNode has
  announced that this public endpoint shuts down on **2026-10-14**; it only serves
  `/pump-fun/swap` (quotes are computed locally with the exact curve math) and applies a
  platform fee that the backtester charges automatically in this mode.
* **Mode B** (`authenticated`) sends `Authorization: Bearer <JWT>` to `METIS_URL`, either a
  static `PUMPFUN_JWT` or short-lived tokens minted from `QN_JWT_PRIVATE_KEY_PATH` + `QN_JWT_KID`.

## Tests

`pytest` runs 954 tests: about a minute for the fast suite, plus roughly five minutes to execute
the four notebooks (`pytest -m "not slow"` skips those). Beyond unit tests, they check the
properties that make a backtest trustworthy:

* **No look-ahead.** Truncating the data at any time `t` leaves every decision and fill before
  `t` unchanged, and a canary strategy that reacts to a large buy always fills after that buy,
  at the post-print price or worse — never ahead of the event it reacted to.
* **Latency causality.** Every fill lands after its decision, and more latency never improves fill timing.
* **Exact accounting.** The cash ledger reconciles to the lamport against every booked fill, and
  bonding-curve quotes match the official SDK on 792 golden cases.
* **Feature parity.** The live (online) feature engine matches the batch research pipeline (tolerance 1e-9).
* **Live path.** The gateway is exercised against mocked RPC / Metis: sign once, rebroadcast, retry
  only after blockhash expiry, and map on-chain errors to reasons per program. The whole live
  engine replays recorded events on virtual time, deterministically, and tracks the backtest.
* **Safety.** Secrets never reach logs or CLI output, live trading refuses to start without both
  the config switch and `--confirm-live`, and token names from the chain are escaped everywhere.

## Documentation

| Document | Covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, event model, concurrency, performance |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Event schema, Parquet layout, DuckDB and SQLite schemas |
| [docs/BACKTEST_REALISM.md](docs/BACKTEST_REALISM.md) | Execution simulation, costs, latency, look-ahead guarantees and how they are tested |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | The ten strategies, the runtime, sizing, exits and risk |
| [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md) | Optimisation, walk-forward, overfitting statistics, Monte Carlo, ML |
| [docs/LIVE_TRADING.md](docs/LIVE_TRADING.md) | Modes A / B, execution pipeline, safety interlocks, runbook |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every config section |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | Pages, reports and the chart design rules |
| [docs/MODULES.md](docs/MODULES.md) | Generated module reference (purpose, architecture, data flow, inputs / outputs, example) |

Research notebooks live in `pumpfun_hft/notebooks/` (data and features, backtest and report,
optimisation and validation, rug model and ML).

## Project layout

```
pumpfun_hft/
  core/         protocol math, IDL codec, event decoding, config, types, PDAs
  api/          RPC / WebSocket / Metis / pump.fun clients, auth, rate limiting
  collectors/   historical + live collectors, Parquet store, synthetic market
  database/     SQLite state store, DuckDB warehouse, schemas
  discovery/    token scanner, creator scoring, outcome resolution
  features/     market state, online feature engine, batch features, registry
  strategies/   10 strategies + shared runtime
  backtester/   replay, execution simulator, engine, results
  risk/         portfolio ledger, sizing, position manager, risk engine, circuit breakers
  optimizer/    search spaces, splits, search methods, studies, overfitting statistics
  analytics/    wallet intelligence, metrics, Monte Carlo, charts, reports
  ml/           datasets, purged CV, models, rug model
  execution/    signer, blockhash / fee / confirmation infrastructure, gateways, live trader
  dashboard/    FastAPI app and page renderers
  configs/      default.yaml
  tests/        unit, integration, replay, execution, latency and correctness tests
  notebooks/    research workflow examples
  logs/ reports/
```

## Known limits

* Latency targets (<100 ms event-to-dispatch, <250 ms quote-to-submit) are measured
  continuously and enforced as budgets, but what you achieve depends on your RPC / WebSocket
  provider and location; `latency-probe` shows your numbers.
* The public WebSocket `logsSubscribe` stream can truncate logs and drop messages under load;
  the collector detects truncation and backfills gaps over RPC, but a dedicated provider
  (or a Geyser stream) is the realistic choice for production.
* Pump.fun and PumpSwap change fee schedules and account layouts from time to time. Run
  `update-idl`, re-check the fee tiers in the config, and re-run the golden tests after upgrades.
