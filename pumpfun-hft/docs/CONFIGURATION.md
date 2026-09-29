# Configuration

All parameters live in `pumpfun_hft/configs/default.yaml`. The schema (`core/config.py`) is
strict: **every key is required and unknown keys are rejected**, so a typo fails loudly instead
of silently falling back to a hidden default.

```bash
python -m pumpfun_hft.main --config my.yaml backtest                     # deep-merged over the defaults
python -m pumpfun_hft.main --set sizing.method=kelly --set backtest.initial_capital_sol=5 backtest
python -m pumpfun_hft.main check-config                                  # validates and prints the fingerprint
```

`Settings.fingerprint()` hashes the fully resolved config; every run, study and report records it
next to a fingerprint of the data it used.

Units follow the key suffix: `*_sol` in SOL, `*_lamports` in lamports, `*_bps` in basis points,
`*_ms` / `*_s` in milliseconds / seconds, `*_pct` in percent.

## Sections

| Section | What it controls |
|---|---|
| `app` | `mode` (`paper` / `live`), global `seed` |
| `paths` | data, events, metadata, DuckDB, SQLite, logs, reports, models, IDLs (relative to the project root) |
| `datasets` | folders of the `synthetic` and `real` data sets; `--dataset NAME` (or `active: NAME`) points data, events, metadata, DuckDB, SQLite, models and reports into that folder (`hft` / `hftr` in `hft.ps1`) |
| `network.metis` | Mode A / B, public URL and its sunset date, public platform fee, timeouts, rate limit, urgency → `priorityFeeLevel` |
| `network.rpc`, `network.ws` | timeouts, rate limits, commitment, batch size, reconnect back-off, queue size |
| `network.pumpfun_frontend` | optional unofficial pump.fun API (off by default) |
| `network.retry`, `network.http_breaker`, `network.jwt` | retry policy, HTTP circuit breaker, JWT algorithm / lifetime / refresh margin |
| `protocol` | program ids, curve constants, curve fee tiers, the 25 PumpSwap fee tiers, flat fees for non-canonical pools, migration fee |
| `fees` | fee source (observed vs schedule), base fee, platform fee (`null` = derive from Metis mode), rent, buy instruction |
| `slippage` | tolerance for buys, sells, urgent exits; maximum; widening on retries |
| `priority_fee` | fixed vs dynamic, compute-unit limit, percentile, bounds, urgency multipliers, retry bump |
| `jito` | enable, block engine, tip mode, tip bounds |
| `simulation` | slot time, impact model, lognormal latency components, spikes, drops, landed failures, congestion, blockhash TTL, Jito failures, outages, rate limits, retries |
| `backtest` | capital, replay mode, candle interval, mark-to-market, equity sampling, sweep interval, default order type, limit offset / TTL, minimum order, cash reserve, annualisation, return-bar size |
| `risk.limits`, `risk.breakers`, `risk.sectors` | loss and exposure limits, per-token order rate; breaker thresholds and cooldown; sector keywords |
| `sizing` | method and its parameters, confidence scaling, impact cap, maximum equity fraction |
| `position` | stop loss, take-profit ladder, trailing stop, max hold, breakeven, pyramiding |
| `features` | window lengths, EWMA half-lives, bar size, whale / aggressive thresholds, fresh-wallet window, smart threshold |
| `discovery` | outcome horizon and definitions (success multiple, rug liquidity drawdown, creator sell share), creator priors and score weights, metadata fetching |
| `wallet_intel` | ranking minimums, prior strength, sniper / whale / market-maker / bot / insider / rug-wallet rules |
| `rug_model` | trained model switch and path (a file name in the data set's models folder, or absolute; a missing file is an error), label horizon and drawdown, snapshot delays, heuristic weights |
| `strategy` | active strategies, exit overlay, cost-gate multiple, one-order-in-flight, re-entry cooldown, per-strategy `params` |
| `optimizer` | method, trials, objective, constraints, workers, selection rule, splits, walk-forward, per-method settings, search `spaces` |
| `montecarlo` | simulations, resampling method, perturbation sigmas, latency cost, ruin level, path-level settings |
| `live` | quote source, swap route, send route, queue / workers, confirmation, rebroadcast, blockhash refresh, retries, latency budget, snapshots |
| `collector` | page sizes, batching, concurrency, flush policy, gap threshold, latency budget, SOL/USD source |
| `logging` | level, rotation, console, channels |
| `dashboard` | host, port, live refresh, table size |
| `synthetic` | the synthetic market generator (tests / demos only) |
| `ml` | model, target, horizons, the longest market-wide silence treated as data rather than a recording gap (real data), CV folds, embargo, SHAP / permutation settings, per-model hyper-parameters |

## Parameters worth reviewing before any real use

| Key | Default | Why it matters |
|---|---|---|
| `protocol.curve_fee_tiers`, `protocol.amm_fee_tiers` | 0.95 % + 0.30 % on the curve; mainnet PumpSwap tiers (captured 2026-09-08) | Pump.fun changes fees; wrong fees make every backtest wrong |
| `network.metis.public_platform_fee_bps` | 20 | charged on every Mode A swap in backtests; confirm the current value |
| `simulation.latency.*` | ~0.6 s median decision → landing | set from your own `latency-probe` and paper results |
| `simulation.failures.*` | 3 % drops, 1.5 % landed failures | calibrate from paper / live fill statistics |
| `risk.limits.daily_loss_sol` | 2 SOL | hard stop for the day |
| `sizing.max_impact_bps` | 600 | caps size by the curve's depth |
| `strategy.cost_gate_multiple` | 1.5 | minimum edge / cost ratio to trade at all |

## Secrets (`.env`)

| Variable | Used for |
|---|---|
| `PUMPFUN_JWT` | Mode B bearer token (static) |
| `QN_JWT_PRIVATE_KEY_PATH`, `QN_JWT_KID` | Mode B: mint short-lived JWTs locally instead |
| `METIS_URL` | Mode B base URL |
| `SOLANA_RPC_URL`, `SOLANA_WS_URL` | collectors, live data, execution |
| `PRIVATE_KEY` | live trading only (base58 secret or JSON byte array) |
| `JITO_AUTH_UUID` | optional Jito authentication |
| `PUMPFUN_FRONTEND_JWT` | optional unofficial frontend API |

Secrets are never read from the YAML, never logged (redacted everywhere) and never written to
reports or the dashboard. `check-config` shows only whether each one is present.
