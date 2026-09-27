-- Analytical warehouse (DuckDB). Raw events stay in day-partitioned Parquet and are exposed
-- through the `events` view (created by Warehouse.refresh_views()); derived entities and run
-- outputs are stored as tables.

CREATE TABLE IF NOT EXISTS tokens (
    mint              VARCHAR PRIMARY KEY,
    creator           VARCHAR,
    name              VARCHAR,
    symbol            VARCHAR,
    uri               VARCHAR,
    created_ms        BIGINT,
    created_slot      BIGINT,
    bonding_curve     VARCHAR,
    dev_buy_lamports  BIGINT,
    bundled_buyers    INTEGER,
    complete          BOOLEAN,
    migrated          BOOLEAN,
    migrated_ms       BIGINT,
    ath_multiple      DOUBLE,
    max_drawdown_pct  DOUBLE,           -- price drawdown from ATH
    max_liq_drawdown_pct DOUBLE,        -- real-SOL liquidity drawdown from its peak (rug signature)
    creator_sold_pct  DOUBLE,
    outcome           VARCHAR,          -- success | rug | neutral | unresolved
    sector            VARCHAR,
    resolved_ms       BIGINT
);

CREATE TABLE IF NOT EXISTS token_metadata (
    mint        VARCHAR PRIMARY KEY,
    name        VARCHAR,
    symbol      VARCHAR,
    uri         VARCHAR,
    description VARCHAR,
    image       VARCHAR,
    twitter     VARCHAR,
    telegram    VARCHAR,
    website     VARCHAR,
    fetched     BOOLEAN,
    anomalies   VARCHAR                 -- comma separated flags
);

CREATE TABLE IF NOT EXISTS creators (
    creator           VARCHAR PRIMARY KEY,
    launches          INTEGER,
    resolved          INTEGER,
    successes         INTEGER,
    rugs              INTEGER,
    migrations        INTEGER,
    avg_ath_multiple  DOUBLE,
    p_success         DOUBLE,
    p_rug             DOUBLE,
    score             DOUBLE,
    updated_ms        BIGINT
);

CREATE TABLE IF NOT EXISTS wallets (
    address           VARCHAR PRIMARY KEY,
    first_seen_ms     BIGINT,
    last_seen_ms      BIGINT,
    n_trades          BIGINT,
    n_buys            BIGINT,
    n_sells           BIGINT,
    buy_sol           DOUBLE,
    sell_sol          DOUBLE,
    tokens_traded     BIGINT,
    closed            BIGINT,
    wins              BIGINT,
    realized_pnl_sol  DOUBLE,
    mean_ret          DOUBLE,
    smart_score       DOUBLE,
    labels            VARCHAR,
    cluster           VARCHAR,          -- insider-cluster id (root wallet of the union-find set)
    updated_ms        BIGINT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    kind         VARCHAR,
    strategy     VARCHAR,
    config_hash  VARCHAR,
    data_hash    VARCHAR,
    created_ms   BIGINT,
    metrics_json VARCHAR
);

CREATE TABLE IF NOT EXISTS run_trades (
    run_id VARCHAR, trade_id BIGINT, mint VARCHAR, strategy VARCHAR, entry_ms BIGINT, exit_ms BIGINT,
    cost_sol DOUBLE, proceeds_sol DOUBLE, pnl_sol DOUBLE, ret DOUBLE, r_multiple DOUBLE,
    mae DOUBLE, mfe DOUBLE, hold_s DOUBLE, exit_reason VARCHAR, n_fills INTEGER, fees_sol DOUBLE
);

CREATE TABLE IF NOT EXISTS run_fills (
    run_id VARCHAR, order_id BIGINT, mint VARCHAR, side VARCHAR, action VARCHAR, status VARCHAR, strategy VARCHAR,
    decision_ms BIGINT, land_ms BIGINT, confirm_ms BIGINT, slot BIGINT, venue VARCHAR, token_amount BIGINT,
    sol_amount BIGINT, sol_delta BIGINT, price DOUBLE, slippage_bps DOUBLE, latency_ms DOUBLE, failure VARCHAR,
    protocol_fee BIGINT, creator_fee BIGINT, lp_fee BIGINT, network_fee BIGINT, priority_fee BIGINT, jito_tip BIGINT
);

CREATE TABLE IF NOT EXISTS run_equity (
    run_id VARCHAR, ts_ms BIGINT, equity_sol DOUBLE, cash_sol DOUBLE, exposure_sol DOUBLE, n_positions INTEGER
);

CREATE TABLE IF NOT EXISTS optimizer_trials (
    study_id VARCHAR, trial INTEGER, method VARCHAR, strategy VARCHAR, params_json VARCHAR,
    train_score DOUBLE, val_score DOUBLE, n_trades INTEGER, created_ms BIGINT
);

CREATE TABLE IF NOT EXISTS ml_models (
    model_id VARCHAR PRIMARY KEY, kind VARCHAR, target VARCHAR, train_end_ms BIGINT,
    features_json VARCHAR, metrics_json VARCHAR, path VARCHAR, created_ms BIGINT
);
