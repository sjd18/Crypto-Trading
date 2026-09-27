# Data model

## Events (Parquet)

Every on-chain fact becomes one row with this schema (`core/types.py: EVENT_SCHEMA`):

| Column | Type | Meaning |
|---|---|---|
| `kind` | str | `create`, `trade`, `complete`, `migrate`, `amm_buy`, `amm_sell`, `pool_create` |
| `slot` | i64 | slot of the transaction |
| `seq` | i32 | transaction order inside the slot (block order historically, arrival order live) |
| `ev_idx` | i16 | event order inside the transaction |
| `ts_ms` | i64 | event time (block time interpolated by slot for history; receive time live) |
| `block_time` | i64 | block time in seconds when known |
| `signature`, `block_hash` | str | transaction signature, block hash (optional) |
| `mint`, `user`, `creator` | str | token mint, acting wallet, token creator |
| `is_buy` | bool | swap direction |
| `sol_amount`, `token_amount` | i64 | lamports and token base units (6 decimals) of the swap |
| `v_sol`, `v_tok`, `r_sol`, `r_tok` | i64 | virtual and real reserves **after** the event (curve) or pool reserves (AMM; `v_sol` = effective quote) |
| `fee_bps`, `fee`, `creator_fee_bps`, `creator_fee`, `lp_fee_bps`, `lp_fee` | i64 | fee rates and amounts as recorded on chain |
| `name`, `symbol`, `uri` | str | token metadata (create events) |
| `bonding_curve`, `pool`, `token_program` | str | accounts |
| `ix_name` | str | instruction (`buy`, `buy_exact_sol_in`, `sell`, …) |
| `sol_usd` | f64 | SOL/USD at the event (static or from a CSV series) |

Sort key: `(slot, seq, ev_idx)`. Duplicate key: `(signature, ev_idx, kind)`.

Layout on disk:

```
data/events/date=YYYY-MM-DD/part-<ms>-<rand>.parquet    append-only writes (zstd)
data/events/date=YYYY-MM-DD/data.parquet                after compaction: sorted, de-duplicated
data/metadata/tokens.parquet                            token metadata (socials, description, anomalies)
data/metadata/synthetic_truth.parquet                   only for synthetic data (planted ground truth)
```

Every file is SHA-256 checksummed into the SQLite manifest (`verify-data` re-hashes).

## Operational state (SQLite, `data/meta.sqlite`)

| Table | Purpose |
|---|---|
| `collector_checkpoints` | resume points of historical collectors (oldest / newest signature and slot, counts) |
| `file_manifest` | every Parquet file with checksum, rows, slot range, bytes, compaction flag |
| `gaps` | slot gaps, WebSocket disconnects, truncated logs, missing transactions (with resolution flag) |
| `pending_signatures` | retry queue for failed or truncated transaction fetches |
| `live_orders` | every live order state transition (crash recovery) |
| `live_state` | JSON snapshots of the live session (dashboard Live monitor) |
| `runs` | registry of backtests, studies and sessions |
| `kv` | small key-value settings |

Full DDL: `pumpfun_hft/database/schema_sqlite.sql`.

## Analytical warehouse (DuckDB, `data/warehouse.duckdb`)

The `events` view reads the Parquet store directly (hive partitions, predicate pushdown). Tables:

| Table | Content |
|---|---|
| `tokens` | per-token lifecycle: creator, metadata, dev buy, creation-slot buyers, completion / migration, ATH multiple, price and liquidity drawdowns, creator sell share, outcome, sector |
| `token_metadata` | fetched metadata and anomaly flags |
| `creators` | launches, resolved outcomes, success / rug posteriors, score |
| `wallets` | wallet profiles: activity, round trips, realised PnL, smart score, labels, insider cluster |
| `runs`, `run_trades`, `run_fills`, `run_equity` | persisted backtests (written by every `backtest`) |
| `optimizer_trials` | optimisation trials |
| `ml_models` | trained model registry |

```bash
python -m pumpfun_hft.main query "select kind, count(*) n from events group by 1 order by n desc"
python -m pumpfun_hft.main query "select strategy, count(*) n, sum(pnl_sol) pnl from run_trades group by 1"
```

Full DDL: `pumpfun_hft/database/schema_duckdb.sql`.

## Run artefacts

```
pumpfun_hft/reports/runs/<run_id>/
  result.json          metrics, parameters, config + data fingerprints, diagnostics
  trades.parquet       closed round trips (entry / exit, costs, PnL, return, R, MAE / MFE, exit reason)
  fills.parquet        every fill and failed attempt (status, venue, amounts, every fee, latency, slippage)
  equity.parquet       equity, cash, exposure, open positions over time
  signals.parquet      every signal with its outcome (submitted, vetoed, cost_gate, risk:…, cooldown…)
  wallets.parquet      wallet database at the end of the run (point-in-time)
  tokens.parquet       tokens seen with their resolved outcomes
  report/              HTML, PDF, CSV, JSON report
pumpfun_hft/reports/optimize/<study_id>/   trials.parquet, study.json (splits, selection, DSR, PBO, unseal log)
pumpfun_hft/reports/walkforward/<id>/      folds.parquet
pumpfun_hft/reports/ml/<model_id>/         importance.parquet, cv.json
data/models/<model_id>.joblib              trained model bundle (with its training cut-off)
```

## Logs

JSON lines, one file per channel, rotated (`logging.max_bytes`, `backup_count`) in
`pumpfun_hft/logs/`: `api`, `trades`, `errors` (also receives every ERROR from any channel),
`latency`, `signals`, `backtests`, `system`. Every registered secret is redacted before a record
is written.
