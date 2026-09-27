"""Shared fixtures: settings, a small synthetic market, isolated paths."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

os.environ.setdefault("PUMPFUN_DISABLE_NUMBA", "0")

from pumpfun_hft.core.config import Settings, load_settings  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


def make_settings(**overrides: Any) -> Settings:
    return load_settings(None, overrides)


@pytest.fixture(scope="session")
def market(settings: Settings) -> Any:
    """A 3-hour synthetic market (deterministic)."""
    from pumpfun_hft.collectors.synthetic import SyntheticMarket

    s = load_settings(None, {"synthetic.duration_hours": 3.0})
    sm = SyntheticMarket(s)
    data = sm.generate()
    return SimpleNamespace(events=data.events, metadata=data.metadata, truth=data.truth, wallet_classes=sm.wallets)


@pytest.fixture(scope="session")
def events(market: Any) -> pl.DataFrame:
    return market.events


@pytest.fixture(scope="session")
def metadata(market: Any) -> dict[str, Any]:
    return {r["mint"]: r for r in market.metadata.iter_rows(named=True)}


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings whose every writable path lives under ``tmp_path``."""
    return load_settings(None, {
        "paths.data_dir": str(tmp_path / "data"),
        "paths.duckdb_file": str(tmp_path / "data" / "warehouse.duckdb"),
        "paths.sqlite_file": str(tmp_path / "data" / "meta.sqlite"),
        "paths.logs_dir": str(tmp_path / "logs"),
        "paths.reports_dir": str(tmp_path / "reports"),
        "paths.models_dir": str(tmp_path / "models"),
        "synthetic.duration_hours": 2.0,
    })


@pytest.fixture(scope="session")
def backtest_result(settings: Settings, events: pl.DataFrame, metadata: dict[str, Any]) -> Any:
    """One momentum backtest on the session market (shared by several tests)."""
    from pumpfun_hft.backtester.engine import BacktestEngine
    from pumpfun_hft.backtester.replay import DataSource

    eng = BacktestEngine(settings, DataSource(frame=events), ["momentum_ignition", "smart_money"], metadata=metadata, seed=7,
                         synthetic=True)
    res = eng.run()
    res.engine = eng  # type: ignore[attr-defined]
    return res
