"""SQLite metadata store (checkpoints, manifest, gaps, live state, run registry)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import orjson

from pumpfun_hft.utils.timeutil import now_ms

SCHEMA_PATH = Path(__file__).with_name("schema_sqlite.sql")


class MetaStore:
    """Thread-safe SQLite wrapper (WAL mode) for operational metadata.

    Example::

        meta = MetaStore("data/meta.sqlite")
        meta.save_checkpoint("hist:pump", oldest_signature=sig, n_signatures=1000)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def _rows(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------ checkpoints
    def get_checkpoint(self, name: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM collector_checkpoints WHERE name=?", (name,))
        return rows[0] if rows else None

    def save_checkpoint(self, name: str, **fields: Any) -> None:
        cur = self.get_checkpoint(name) or {"name": name, "n_signatures": 0, "done": 0}
        cur.update(fields)
        cur["updated_ms"] = now_ms()
        cols = ["name", "newest_signature", "oldest_signature", "newest_slot", "oldest_slot", "n_signatures", "done", "updated_ms"]
        self._exec(
            f"INSERT OR REPLACE INTO collector_checkpoints ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [cur.get(c) for c in cols],
        )

    # ------------------------------------------------------------------ manifest
    def add_file(self, path: str, day: str, sha256: str, rows: int, min_slot: int | None, max_slot: int | None,
                 nbytes: int, compacted: bool = False) -> None:
        self._exec(
            "INSERT OR REPLACE INTO file_manifest (path, day, sha256, rows, min_slot, max_slot, bytes, compacted, created_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (path, day, sha256, rows, min_slot, max_slot, nbytes, int(compacted), now_ms()),
        )

    def remove_file(self, path: str) -> None:
        self._exec("DELETE FROM file_manifest WHERE path=?", (path,))

    def files(self, day: str | None = None) -> list[dict[str, Any]]:
        if day is None:
            return self._rows("SELECT * FROM file_manifest ORDER BY day, path")
        return self._rows("SELECT * FROM file_manifest WHERE day=? ORDER BY path", (day,))

    # ------------------------------------------------------------------ gaps & retry queue
    def add_gap(self, source: str, kind: str, *, start_slot: int | None = None, end_slot: int | None = None,
                start_ms: int | None = None, end_ms: int | None = None, detail: str = "") -> None:
        self._exec(
            "INSERT INTO gaps (source, kind, start_slot, end_slot, start_ms, end_ms, detail, created_ms) VALUES (?,?,?,?,?,?,?,?)",
            (source, kind, start_slot, end_slot, start_ms, end_ms, detail, now_ms()),
        )

    def gaps(self, unresolved_only: bool = True) -> list[dict[str, Any]]:
        q = "SELECT * FROM gaps" + (" WHERE resolved=0" if unresolved_only else "") + " ORDER BY id"
        return self._rows(q)

    def resolve_gap(self, gap_id: int) -> None:
        self._exec("UPDATE gaps SET resolved=1 WHERE id=?", (gap_id,))

    def add_pending(self, items: list[tuple[str, int | None]], error: str = "") -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT INTO pending_signatures (signature, slot, attempts, last_error, created_ms) VALUES (?,?,1,?,?)"
                " ON CONFLICT(signature) DO UPDATE SET attempts=attempts+1, last_error=excluded.last_error",
                [(s, slot, error, now_ms()) for s, slot in items],
            )

    def pending(self, limit: int = 1000, max_attempts: int = 10) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM pending_signatures WHERE attempts < ? ORDER BY slot LIMIT ?", (max_attempts, limit))

    def resolve_pending(self, signatures: list[str]) -> None:
        with self._lock:
            self.conn.executemany("DELETE FROM pending_signatures WHERE signature=?", [(s,) for s in signatures])

    # ------------------------------------------------------------------ live orders & state
    def upsert_order(self, order_id: str, mint: str, side: str, status: str, payload: dict[str, Any],
                     signature: str | None = None, last_valid_block_height: int | None = None) -> None:
        t = now_ms()
        self._exec(
            "INSERT INTO live_orders (order_id, mint, side, status, signature, last_valid_block_height, payload, created_ms, updated_ms)"
            " VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(order_id) DO UPDATE SET status=excluded.status,"
            " signature=COALESCE(excluded.signature, signature), last_valid_block_height=COALESCE(excluded.last_valid_block_height,"
            " last_valid_block_height), payload=excluded.payload, updated_ms=excluded.updated_ms",
            (order_id, mint, side, status, signature, last_valid_block_height, orjson.dumps(payload, default=str).decode(), t, t),
        )

    def open_orders(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM live_orders WHERE status IN ('new','submitted','resting') ORDER BY created_ms")

    def set_state(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO live_state (key, value, updated_ms) VALUES (?,?,?)",
                   (key, orjson.dumps(value, default=str, option=orjson.OPT_SERIALIZE_NUMPY).decode(), now_ms()))

    def get_state(self, key: str) -> Any:
        rows = self._rows("SELECT value, updated_ms FROM live_state WHERE key=?", (key,))
        return orjson.loads(rows[0]["value"]) if rows else None

    def all_state(self) -> dict[str, Any]:
        return {r["key"]: {"value": orjson.loads(r["value"]), "updated_ms": r["updated_ms"]}
                for r in self._rows("SELECT * FROM live_state")}

    # ------------------------------------------------------------------ runs
    def register_run(self, run_id: str, kind: str, strategy: str | None, config_hash: str | None,
                     data_hash: str | None, path: str | None, metrics: dict[str, Any] | None = None, notes: str = "") -> None:
        self._exec(
            "INSERT OR REPLACE INTO runs (run_id, kind, strategy, config_hash, data_hash, created_ms, path, metrics, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, kind, strategy, config_hash, data_hash, now_ms(), path,
             orjson.dumps(metrics or {}, default=str, option=orjson.OPT_SERIALIZE_NUMPY).decode(), notes),
        )

    def runs(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            return self._rows("SELECT * FROM runs WHERE kind=? ORDER BY created_ms DESC", (kind,))
        return self._rows("SELECT * FROM runs ORDER BY created_ms DESC")

    def set_kv(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)", (key, orjson.dumps(value, default=str).decode()))

    def get_kv(self, key: str) -> Any:
        rows = self._rows("SELECT value FROM kv WHERE key=?", (key,))
        return orjson.loads(rows[0]["value"]) if rows else None
