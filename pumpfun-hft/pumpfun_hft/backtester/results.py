"""Backtest result container and persistence (Parquet + JSON, reloadable for reports/dashboard)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import polars as pl


def _json_default(o: Any) -> Any:
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


@dataclass
class BacktestResult:
    run_id: str
    strategies: list[str]
    config_hash: str
    data_hash: str
    seed: int
    start_ms: int
    end_ms: int
    n_events: int
    elapsed_s: float
    initial_capital_sol: float
    metrics: dict[str, Any]
    trades: pl.DataFrame
    fills: pl.DataFrame
    equity: pl.DataFrame
    signals: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def events_per_second(self) -> float:
        return self.n_events / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    def summary(self) -> dict[str, Any]:
        keys = ["total_return", "pnl_sol", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate", "profit_factor",
                "expectancy_sol", "avg_r_multiple", "kelly_fraction", "fill_rate", "avg_latency_ms", "avg_slippage_bps"]
        return {k: self.metrics.get(k) for k in keys}

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.trades.write_parquet(d / "trades.parquet")
        self.fills.write_parquet(d / "fills.parquet")
        self.equity.write_parquet(d / "equity.parquet")
        self.signals.write_parquet(d / "signals.parquet")
        meta = {k: getattr(self, k) for k in ("run_id", "strategies", "config_hash", "data_hash", "seed", "start_ms", "end_ms",
                                               "n_events", "elapsed_s", "initial_capital_sol", "synthetic")}
        meta.update(metrics=self.metrics, diagnostics=self.diagnostics, params=self.params)
        (d / "result.json").write_bytes(orjson.dumps(meta, default=_json_default,
                                                      option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> BacktestResult:
        d = Path(directory)
        meta = json.loads((d / "result.json").read_text(encoding="utf-8"))  # UTF-8 on every OS

        def rd(name: str) -> pl.DataFrame:
            p = d / name
            return pl.read_parquet(p) if p.exists() else pl.DataFrame()

        return cls(run_id=meta["run_id"], strategies=meta["strategies"], config_hash=meta["config_hash"], data_hash=meta["data_hash"],
                   seed=meta["seed"], start_ms=meta["start_ms"], end_ms=meta["end_ms"], n_events=meta["n_events"],
                   elapsed_s=meta["elapsed_s"], initial_capital_sol=meta["initial_capital_sol"], metrics=meta["metrics"],
                   trades=rd("trades.parquet"), fills=rd("fills.parquet"), equity=rd("equity.parquet"),
                   signals=rd("signals.parquet"), diagnostics=meta.get("diagnostics", {}), params=meta.get("params", {}),
                   synthetic=meta.get("synthetic", False))
