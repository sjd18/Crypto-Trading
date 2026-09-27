"""Storage layer.

Purpose
    DuckDB is the analytical warehouse (derived tables + a view over the Parquet event store);
    SQLite holds small, frequently-updated operational metadata.

Architecture
    schema_duckdb.sql   tokens, token_metadata, creators, wallets, runs, run_trades, run_fills,
                        run_equity, optimizer_trials, ml_models (+ `events` view over Parquet)
    schema_sqlite.sql   collector_checkpoints, file_manifest, gaps, pending_signatures,
                        live_orders, live_state, runs, kv
    warehouse.py        Warehouse (DuckDB): views, upserts, run persistence, SQL -> Polars
    meta.py             MetaStore (SQLite): checkpoints, manifest, gaps, live state

Data flow
    collectors -> Parquet + MetaStore.manifest ; discovery / wallet intel -> Warehouse tables ;
    backtests -> Warehouse run tables ; live engine -> MetaStore.live_* -> dashboard.

Inputs / Outputs
    Inputs: Polars frames (events, runs, trades, fills, wallets, tokens) and operational records
    (checkpoints, file checksums, gaps, pending signatures, order and live state).
    Outputs: SQL query results as Polars frames, idempotent upserts, and the state a restarted
    collector or live session resumes from.

Example
    wh = Warehouse(settings.paths.resolve("duckdb_file"), settings.paths.events_dir)
    df = wh.query("select kind, count(*) n from events group by 1")
"""
