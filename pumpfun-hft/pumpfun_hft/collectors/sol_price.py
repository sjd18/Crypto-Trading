"""SOL/USD price providers (point-in-time lookups for USD-denominated features and reports)."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Protocol

import polars as pl


class SolPriceProvider(Protocol):
    def price_at(self, ts_ms: int) -> float: ...


class StaticSolPrice:
    """Constant SOL/USD price (for offline research without a price series)."""

    def __init__(self, usd: float) -> None:
        self.usd = float(usd)

    def price_at(self, ts_ms: int) -> float:
        return self.usd


class SeriesSolPrice:
    """As-of lookup into a (ts_ms, sol_usd) series: returns the last price at or before ``ts_ms``."""

    def __init__(self, ts_ms: list[int], usd: list[float]) -> None:
        if not ts_ms:
            raise ValueError("empty price series")
        order = sorted(range(len(ts_ms)), key=ts_ms.__getitem__)
        self.ts = [int(ts_ms[i]) for i in order]
        self.px = [float(usd[i]) for i in order]

    @classmethod
    def from_csv(cls, path: str | Path) -> SeriesSolPrice:
        df = pl.read_csv(path).select(pl.col("ts_ms").cast(pl.Int64), pl.col("sol_usd").cast(pl.Float64))
        return cls(df["ts_ms"].to_list(), df["sol_usd"].to_list())

    def price_at(self, ts_ms: int) -> float:
        i = bisect.bisect_right(self.ts, ts_ms) - 1
        return self.px[i] if i >= 0 else float("nan")  # before the series starts the price is unknown


def provider_from_settings(settings: object) -> SolPriceProvider:
    c = settings.collector  # type: ignore[attr-defined]
    if c.sol_price_source == "csv" and c.sol_price_csv:
        return SeriesSolPrice.from_csv(c.sol_price_csv)
    return StaticSolPrice(c.sol_price_static_usd)
