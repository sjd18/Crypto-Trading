"""Replay data sources.

:class:`DataSource` yields canonical event batches in on-chain order from an in-memory
DataFrame or a :class:`~pumpfun_hft.collectors.storage.ParquetEventStore` (streamed one day at a
time, so arbitrarily long histories replay with bounded memory).

Replay *modes* (``backtest.replay_mode``) are implemented by the engine and control when
strategies are evaluated — market state is always updated event by event:

* ``event``  every event (creates, trades, completions, migrations)
* ``trade``  trade events only (lifecycle events still update state)
* ``tick``   once per token per slot, at the last event of the slot (slot-granular observer)
* ``candle`` once per token per candle, at the candle close (``candle_interval_ms``)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import polars as pl

from pumpfun_hft.collectors.storage import ParquetEventStore, normalise_frame
from pumpfun_hft.core.types import EVENT_COLUMNS, SORT_KEYS
from pumpfun_hft.utils.hashing import stable_hash


@dataclass
class DataSource:
    """Events for replay (DataFrame or Parquet store) restricted to ``[start_ms, end_ms)``."""

    frame: pl.DataFrame | None = None
    store: ParquetEventStore | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    batch_rows: int = 250_000

    def __post_init__(self) -> None:
        if (self.frame is None) == (self.store is None):
            raise ValueError("provide exactly one of frame or store")
        if self.frame is not None:
            f = normalise_frame(self.frame)
            if self.start_ms is not None:
                f = f.filter(pl.col("ts_ms") >= self.start_ms)
            if self.end_ms is not None:
                f = f.filter(pl.col("ts_ms") < self.end_ms)
            self.frame = f.sort(list(SORT_KEYS))

    def batches(self) -> Iterator[pl.DataFrame]:
        if self.frame is not None:
            for off in range(0, self.frame.height, self.batch_rows):
                yield self.frame.slice(off, self.batch_rows).select(list(EVENT_COLUMNS))
        else:
            assert self.store is not None
            yield from self.store.iter_batches(self.start_ms, self.end_ms, batch_rows=self.batch_rows)

    def fingerprint(self) -> str:
        """Content fingerprint for reproducibility records."""
        if self.frame is not None:
            f = self.frame
            if f.is_empty():
                return "empty"
            return stable_hash([f.height, int(f["slot"].min()), int(f["slot"].max()), f.hash_rows().sum()])
        assert self.store is not None
        files = sorted(str(p.relative_to(self.store.root)) + str(p.stat().st_size) for p in self.store.root.glob("date=*/*.parquet"))
        return stable_hash([files, self.start_ms, self.end_ms])

    def time_bounds(self) -> tuple[int, int] | None:
        if self.frame is not None:
            if self.frame.is_empty():
                return None
            return int(self.frame["ts_ms"].min()), int(self.frame["ts_ms"].max())
        assert self.store is not None
        lf = self.store.scan(self.start_ms, self.end_ms, columns=["ts_ms"])
        res = lf.select(pl.col("ts_ms").min().alias("a"), pl.col("ts_ms").max().alias("b")).collect()
        if res.is_empty() or res["a"][0] is None:
            return None
        return int(res["a"][0]), int(res["b"][0])


def load_metadata(path: Any) -> dict[str, dict[str, Any]]:
    """Token metadata table (Parquet path or DataFrame) -> {mint: row} for point-in-time joins at creation."""
    if isinstance(path, dict):
        return path
    if isinstance(path, pl.DataFrame):
        df = path
    else:
        try:
            df = pl.read_parquet(path)
        except Exception:  # noqa: BLE001 - absent metadata is fine
            return {}
    if df.is_empty() or "mint" not in df.columns:
        return {}
    return {r["mint"]: r for r in df.iter_rows(named=True)}
