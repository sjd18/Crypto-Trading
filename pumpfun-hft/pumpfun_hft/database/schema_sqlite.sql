-- Lightweight operational metadata (SQLite, WAL mode).
-- Collector checkpoints (resume), file manifest (checksums), gap registry, live order
-- state (crash recovery), live monitor snapshots and the run registry.

CREATE TABLE IF NOT EXISTS collector_checkpoints (
    name              TEXT PRIMARY KEY,      -- e.g. 'hist:6EF8r...'
    newest_signature  TEXT,                  -- newest processed signature (forward catch-up with `until`)
    oldest_signature  TEXT,                  -- oldest processed signature (backfill cursor with `before`)
    newest_slot       INTEGER,
    oldest_slot       INTEGER,
    n_signatures      INTEGER NOT NULL DEFAULT 0,
    done              INTEGER NOT NULL DEFAULT 0,
    updated_ms        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS file_manifest (
    path        TEXT PRIMARY KEY,            -- path relative to the events root
    day         TEXT NOT NULL,               -- YYYY-MM-DD partition
    sha256      TEXT NOT NULL,
    rows        INTEGER NOT NULL,
    min_slot    INTEGER,
    max_slot    INTEGER,
    bytes       INTEGER NOT NULL,
    compacted   INTEGER NOT NULL DEFAULT 0,
    created_ms  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_manifest_day ON file_manifest(day);

CREATE TABLE IF NOT EXISTS gaps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,               -- historical | live | store
    kind        TEXT NOT NULL,               -- slot_gap | missing_tx | ws_disconnect | truncated_logs
    start_slot  INTEGER,
    end_slot    INTEGER,
    start_ms    INTEGER,
    end_ms      INTEGER,
    detail      TEXT,
    resolved    INTEGER NOT NULL DEFAULT 0,
    created_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_signatures (
    signature   TEXT PRIMARY KEY,            -- signatures whose transaction fetch failed (retry queue)
    slot        INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS live_orders (
    order_id                TEXT PRIMARY KEY,
    mint                    TEXT NOT NULL,
    side                    TEXT NOT NULL,
    status                  TEXT NOT NULL,
    signature               TEXT,
    last_valid_block_height INTEGER,
    payload                 TEXT NOT NULL,   -- JSON order snapshot
    created_ms              INTEGER NOT NULL,
    updated_ms              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(status);

CREATE TABLE IF NOT EXISTS live_state (
    key         TEXT PRIMARY KEY,            -- positions | pnl | latency | breakers | ...
    value       TEXT NOT NULL,               -- JSON
    updated_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,               -- backtest | optimize | walkforward | montecarlo | paper | live
    strategy    TEXT,
    config_hash TEXT,
    data_hash   TEXT,
    created_ms  INTEGER NOT NULL,
    path        TEXT,
    metrics     TEXT,                        -- JSON headline metrics
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
