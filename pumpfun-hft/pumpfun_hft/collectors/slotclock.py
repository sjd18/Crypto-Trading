"""Millisecond timestamps for historical events.

Solana exposes block time only at 1-second resolution. For latency-sensitive research each
event needs a finer, monotone time axis. :class:`SlotClock` fits ``ts_ms = a + b * slot`` by
least squares on (slot, block_time) anchors (block time mid-point ``+500 ms``) and clamps each
estimate into its block's second ``[bt*1000, bt*1000 + 999]``. Within a slot all events share
the slot timestamp; the ``seq`` column preserves transaction order.
"""

from __future__ import annotations

import numpy as np
import polars as pl


class SlotClock:
    """Slot -> epoch-ms estimator from (slot, block_time) anchors."""

    def __init__(self, slot_ms: float = 400.0) -> None:
        self.slot_ms = slot_ms
        self.a: float | None = None
        self.b: float = slot_ms

    def fit(self, slots: np.ndarray, block_times_s: np.ndarray) -> SlotClock:
        mask = ~np.isnan(block_times_s.astype(np.float64))
        s = slots[mask].astype(np.float64)
        t = block_times_s[mask].astype(np.float64) * 1000.0 + 500.0
        if s.size >= 2 and np.ptp(s) > 0:
            self.b, self.a = np.polyfit(s, t, 1)
        elif s.size >= 1:
            self.b = self.slot_ms
            self.a = float(t[0] - self.b * s[0])
        return self

    def estimate(self, slots: np.ndarray, block_times_s: np.ndarray | None = None) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("SlotClock not fitted")
        est = self.a + self.b * slots.astype(np.float64)
        if block_times_s is not None:
            bt = block_times_s.astype(np.float64) * 1000.0
            ok = ~np.isnan(bt)
            est = np.where(ok, np.clip(est, bt, bt + 999.0), est)
        return est.astype(np.int64)


def assign_timestamps(df: pl.DataFrame, slot_ms: float = 400.0) -> pl.DataFrame:
    """Fill ``ts_ms`` for rows from ``slot`` and ``block_time`` (keeps existing non-null ts_ms)."""
    if df.is_empty():
        return df
    slots = df["slot"].to_numpy()
    bts = df["block_time"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    clock = SlotClock(slot_ms).fit(slots, bts)
    if clock.a is None:
        return df
    est = clock.estimate(slots, bts)
    return df.with_columns(
        pl.when(pl.col("ts_ms").is_null() | (pl.col("ts_ms") == 0)).then(pl.Series(est)).otherwise(pl.col("ts_ms")).alias("ts_ms")
    )
