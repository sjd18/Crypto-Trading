"""DuckDB analytical warehouse."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
import orjson
import polars as pl

from pumpfun_hft.utils.timeutil import now_ms

SCHEMA_PATH = Path(__file__).with_name("schema_duckdb.sql")


class Warehouse:
    """DuckDB warehouse with an ``events`` view over the Parquet store.

    Example::

        wh = Warehouse("data/warehouse.duckdb", "data/events")
        wh.query("select count(*) n from events where kind = 'trade'")
        wh.upsert("wallets", wallets_df, key=["address"])
    """

    def __init__(self, path: str | Path, events_dir: str | Path | None = None, read_only: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events_dir = Path(events_dir) if events_dir else None
        self._lock = threading.RLock()
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.refresh_views()

    def close(self) -> None:
        with self._lock:
            self.con.close()

    def refresh_views(self) -> bool:
        """(Re)create the ``events`` view; returns False when no Parquet files exist yet."""
        if self.events_dir is None or not any(self.events_dir.glob("date=*/*.parquet")):
            return False
        glob = str(self.events_dir / "date=*" / "*.parquet").replace("'", "''")
        with self._lock:
            self.con.execute(
                f"CREATE OR REPLACE TEMP VIEW events AS SELECT * EXCLUDE (date) FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true)"
            )
        return True

    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        with self._lock:
            return self.con.execute(sql, params or []).pl()

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        with self._lock:
            self.con.execute(sql, params or [])

    def upsert(self, table: str, df: pl.DataFrame, key: list[str]) -> int:
        """Insert-or-replace rows of ``df`` into ``table`` keyed by ``key`` columns."""
        if df.is_empty():
            return 0
        with self._lock:
            self.con.register("_upsert_df", df.to_arrow())
            try:
                cond = " AND ".join(f"t.{k} = s.{k}" for k in key)
                self.con.execute(f"DELETE FROM {table} t USING _upsert_df s WHERE {cond}")
                cols = ", ".join(df.columns)
                self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _upsert_df")
            finally:
                self.con.unregister("_upsert_df")
        return df.height

    def append(self, table: str, df: pl.DataFrame) -> int:
        if df.is_empty():
            return 0
        with self._lock:
            self.con.register("_append_df", df.to_arrow())
            try:
                cols = ", ".join(df.columns)
                self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _append_df")
            finally:
                self.con.unregister("_append_df")
        return df.height

    def save_run(self, run_id: str, kind: str, strategy: str, config_hash: str, data_hash: str, metrics: dict[str, Any],
                 trades: pl.DataFrame | None = None, fills: pl.DataFrame | None = None, equity: pl.DataFrame | None = None) -> None:
        """Persist a run and its trades/fills/equity (idempotent per run_id)."""
        with self._lock:
            for t in ("runs", "run_trades", "run_fills", "run_equity"):
                self.con.execute(f"DELETE FROM {t} WHERE run_id = ?", [run_id])
            self.con.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                [run_id, kind, strategy, config_hash, data_hash, now_ms(),
                 orjson.dumps(metrics, default=str, option=orjson.OPT_SERIALIZE_NUMPY).decode()],
            )
        schema_cols = {
            "run_trades": [c[0] for c in self.con.execute("DESCRIBE run_trades").fetchall()],
            "run_fills": [c[0] for c in self.con.execute("DESCRIBE run_fills").fetchall()],
            "run_equity": [c[0] for c in self.con.execute("DESCRIBE run_equity").fetchall()],
        }
        for table, df in (("run_trades", trades), ("run_fills", fills), ("run_equity", equity)):
            if df is None or df.is_empty():
                continue
            d = df.with_columns(pl.lit(run_id).alias("run_id"))
            cols = [c for c in schema_cols[table] if c in d.columns]
            self.append(table, d.select(cols))

    def events(self, start_ms: int | None = None, end_ms: int | None = None, kinds: list[str] | None = None,
               mints: list[str] | None = None) -> pl.DataFrame:
        where, params = [], []
        if start_ms is not None:
            where.append("ts_ms >= ?")
            params.append(start_ms)
        if end_ms is not None:
            where.append("ts_ms < ?")
            params.append(end_ms)
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += kinds
        if mints:
            where.append(f"mint IN ({','.join('?' * len(mints))})")
            params += mints
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY slot, seq, ev_idx"
        return self.query(sql, params)
