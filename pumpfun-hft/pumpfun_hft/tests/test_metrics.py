"""Performance metrics on hand-checkable inputs."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pumpfun_hft.analytics.metrics import (
    bar_returns,
    compute_metrics,
    drawdown_series,
    hourly_profile,
    monthly_returns,
    reason_category,
    return_metrics,
    streaks,
    trade_metrics,
)

H = 3_600_000
DAY = 24 * H


def _equity(values: list[float], step_ms: int = H, start: int = 1_788_220_800_000) -> pl.DataFrame:
    n = len(values)
    vals = [float(v) for v in values]
    return pl.DataFrame({"ts_ms": [start + i * step_ms for i in range(n)], "equity_sol": vals, "cash_sol": vals,
                         "exposure_sol": [0.0] * n, "n_positions": [0] * n})


def _trades(pnls: list[float], cost: float = 1.0) -> pl.DataFrame:
    n = len(pnls)
    return pl.DataFrame({"trade_id": list(range(1, n + 1)), "entry_ms": [1_788_220_800_000 + i * H for i in range(n)],
                         "exit_ms": [1_788_220_800_000 + i * H + 60_000 for i in range(n)], "pnl_sol": pnls,
                         "ret": [p / cost for p in pnls], "r_multiple": [p / (0.25 * cost) for p in pnls],
                         "mae": [min(0.0, p) for p in pnls], "mfe": [max(0.0, p) for p in pnls], "hold_s": [60.0] * n,
                         "fees_sol": [0.025] * n, "tx_costs_sol": [0.0002] * n})


def test_max_drawdown_and_underwater() -> None:
    eq = _equity([10, 11, 9.9, 12, 6, 8, 12.5])
    m = compute_metrics(eq, pl.DataFrame(), 10.0, H, 365.0)
    assert m["max_drawdown"] == pytest.approx(0.5)  # 12 -> 6
    assert m["total_return"] == pytest.approx(0.25)
    assert m["pnl_sol"] == pytest.approx(2.5)
    assert 0 < m["pct_time_underwater"] < 1
    assert np.allclose(drawdown_series(np.array([10, 12, 6, 12.0])), [0, 0, -0.5, 0])


def test_short_span_does_not_annualise_cagr_or_calmar() -> None:
    m = compute_metrics(_equity([10, 10.5, 10.2, 11]), pl.DataFrame(), 10.0, H, 365.0)
    assert m["cagr_extrapolated"] and math.isnan(m["cagr"]) and math.isnan(m["calmar"])
    assert m["cagr_raw"] > 1.0  # the extrapolation is kept, but only under its honest name
    assert m["recovery_factor"] == pytest.approx(0.1 / (0.3 / 10.5))


def test_long_span_cagr_and_calmar() -> None:
    values = list(np.linspace(10, 12, 400))
    m = compute_metrics(_equity(values, DAY), pl.DataFrame(), 10.0, DAY, 365.0)
    days = 399
    assert m["cagr"] == pytest.approx(1.2 ** (365 / days) - 1, rel=1e-9)
    assert math.isnan(m["calmar"]) or m["max_drawdown"] == 0  # monotone equity has no drawdown


def test_bar_returns_forward_fill() -> None:
    ts = np.array([0, 10, 25, 70], dtype=np.int64)
    eq = np.array([100.0, 101.0, 102.0, 99.0])
    r = bar_returns(ts, eq, 20)
    # bars: [0,20) last=101, [20,40) last=102, [40,60) empty -> 102, [60,80) last=99
    assert np.allclose(r, [0.01, 102 / 101 - 1, 0.0, 99 / 102 - 1])


def test_return_metrics_known_values() -> None:
    r = np.array([0.01, -0.005, 0.02, 0.0, -0.01])
    m = return_metrics(r, 252)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    assert m["omega"] == pytest.approx(0.03 / 0.015)
    assert return_metrics(np.array([0.01]), 252)["sharpe"] != return_metrics(np.array([0.01]), 252)["sharpe"]  # NaN


def test_trade_metrics() -> None:
    t = _trades([0.3, -0.1, -0.1, 0.2, -0.1])
    m = trade_metrics(t)
    assert m["n_trades"] == 5
    assert m["win_rate"] == pytest.approx(0.4)
    assert m["profit_factor"] == pytest.approx(0.5 / 0.3)
    assert m["expectancy_sol"] == pytest.approx(0.04)
    payoff = 0.25 / 0.1
    assert m["kelly_fraction"] == pytest.approx(0.4 - 0.6 / payoff)
    assert m["max_win_streak"] == 1 and m["max_loss_streak"] == 2


def test_streaks() -> None:
    assert streaks([True, True, False, True, True, True, False, False]) == (3, 2)
    assert streaks([]) == (0, 0)


def test_monthly_and_hourly_profiles() -> None:
    eq = _equity([10.0, 11.0, 12.1], step_ms=40 * DAY, start=1_788_220_800_000)
    mon = monthly_returns(eq)
    assert mon.height >= 2 and abs(mon["ret"][-1]) > 0
    hp = hourly_profile(_trades([0.1, -0.2, 0.3]))
    assert hp["pnl_sol"].sum() == pytest.approx(0.2)
    assert set(hp.columns) >= {"weekday", "hour", "pnl_sol", "n"}


@pytest.mark.parametrize(("reason", "category"), [
    ("take profit L2 +40.0%", "take profit"), ("stop loss -25.0%", "stop loss"), ("rug probability 0.63", "rug probability"),
    ("3 smart wallets bought 12.0 SOL", "smart wallets bought"), ("liquidity drop -34% since entry peak", "liquidity drop since entry peak"),
    ("top holder dumped 100% (4.2% of supply)", "top holder dumped"), ("ignition imb=1.00 ret=0.26 buy=3.2", "ignition"),
    ("follow whale wallet=Abc123 score=0.80", "follow whale"), (None, "unknown"), ("", "unknown"),
])
def test_reason_category(reason: str | None, category: str) -> None:
    assert reason_category(reason) == category
