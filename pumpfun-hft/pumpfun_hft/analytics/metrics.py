"""Performance metrics.

Return-based metrics use the equity curve resampled on a fixed grid of ``returns_bar_ms``
(last equity per bar, forward-filled across empty bars) and are annualised with
``periods_per_year = annualization_days * 86_400_000 / returns_bar_ms`` (crypto trades 24/7).

    Sharpe   mean(r) / std(r, ddof=1) * sqrt(ppy)
    Sortino  mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(ppy)
    Omega    sum(max(r, 0)) / sum(max(-r, 0))                      (threshold 0)
    CAGR     (E_end / E_0) ** (365 d / span) - 1; NaN for spans < 30 days (``cagr_raw`` keeps the extrapolation)
    Calmar   CAGR / max drawdown (spans >= 30 days); ``recovery_factor`` = period return / max drawdown
    Max DD, longest time under water, % of time under water — on raw equity samples

Trade-based metrics use closed round trips: win rate, profit factor, expectancy (SOL and R),
Kelly fraction ``W - (1 - W) / R``, average R multiple, median trade, MAE / MFE, streaks.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.utils.numba_compat import njit


@njit(cache=True)
def _drawdown_stats(ts: np.ndarray, eq: np.ndarray) -> tuple[float, float, float]:
    """(max drawdown fraction, longest underwater duration ms, fraction of time underwater).

    An underwater period runs from the last equity peak until equity regains that peak.
    """
    n = eq.shape[0]
    peak = eq[0]
    max_dd = 0.0
    longest = 0.0
    under_time = 0.0
    last_peak_ts = ts[0]
    prev_under = False
    for i in range(n):
        if i > 0 and prev_under:
            under_time += ts[i] - ts[i - 1]
        if eq[i] >= peak:
            if prev_under:
                dur = ts[i] - last_peak_ts
                if dur > longest:
                    longest = dur
            peak = eq[i]
            last_peak_ts = ts[i]
            prev_under = False
        else:
            dd = 1.0 - eq[i] / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            prev_under = True
    if prev_under:
        dur = ts[n - 1] - last_peak_ts
        if dur > longest:
            longest = dur
    span = ts[n - 1] - ts[0]
    return max_dd, longest, (under_time / span if span > 0 else 0.0)


MIN_ANNUALISATION_DAYS = 30.0  # below this span CAGR/Calmar are reported as NaN (extrapolation is meaningless)


_PAREN = re.compile(r"\([^)]*\)")
_KEYVAL = re.compile(r"\b\w+=\S+")


def reason_category(reason: str | None) -> str:
    """Collapse a free-text signal / exit reason into a stable category for aggregation.

    Drops parenthesised details, ``key=value`` pairs and every token containing a digit, then a
    trailing ``SOL`` unit: ``"take profit L2 +40.0%"`` -> ``"take profit"``, ``"rug probability 0.63"``
    -> ``"rug probability"``, ``"3 smart wallets bought 12.0 SOL"`` -> ``"smart wallets bought"``.
    """
    if not reason:
        return "unknown"
    s = _KEYVAL.sub(" ", _PAREN.sub(" ", str(reason)))
    toks = [t for t in s.replace(",", " ").split() if not any(ch.isdigit() for ch in t)]
    while toks and toks[-1] in ("SOL", "-", ":", "from"):
        toks.pop()
    out = " ".join(toks).strip(" :-")
    return out or str(reason).split()[0]


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return np.where(peak > 0, equity / peak - 1.0, 0.0)


def bar_returns(ts: np.ndarray, equity: np.ndarray, bar_ms: int) -> np.ndarray:
    """Simple returns on a fixed time grid (last value per bar, forward-filled)."""
    if len(ts) < 2:
        return np.empty(0)
    start = ts[0] - ts[0] % bar_ms
    idx = (ts - start) // bar_ms
    n_bars = int(idx[-1]) + 1
    last = np.full(n_bars, np.nan)
    last[idx.astype(np.int64)] = equity  # later samples overwrite earlier ones in the same bar
    for i in range(1, n_bars):
        if np.isnan(last[i]):
            last[i] = last[i - 1]
    series = np.concatenate([[equity[0]], last])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = series[1:] / series[:-1] - 1.0
    return r[np.isfinite(r)]


def streaks(signs: list[bool]) -> tuple[int, int]:
    """Longest winning and losing streaks."""
    best_w = best_l = cur_w = cur_l = 0
    for s in signs:
        if s:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def _safe(x: float) -> float:
    return float(x) if x is not None and math.isfinite(x) else float("nan")


def return_metrics(r: np.ndarray, periods_per_year: float) -> dict[str, float]:
    if r.size < 2:
        return {"sharpe": float("nan"), "sortino": float("nan"), "omega": float("nan"), "vol_annual": float("nan")}
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    gains, losses = float(np.maximum(r, 0).sum()), float(np.maximum(-r, 0).sum())
    ann = math.sqrt(periods_per_year)
    return {
        "sharpe": mean / sd * ann if sd > 0 else float("nan"),
        "sortino": mean / downside * ann if downside > 0 else float("nan"),
        "omega": gains / losses if losses > 0 else float("inf") if gains > 0 else float("nan"),
        "vol_annual": sd * ann,
    }


def trade_metrics(trades: pl.DataFrame) -> dict[str, float]:
    n = trades.height
    if n == 0:
        return {"n_trades": 0}
    pnl = trades["pnl_sol"].to_numpy()
    ret = trades["ret"].to_numpy()
    rm = trades["r_multiple"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    win_rate = wins.size / n
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else float("nan")
    kelly = win_rate - (1 - win_rate) / payoff if payoff and math.isfinite(payoff) and payoff > 0 else float("nan")
    sw, sl = streaks(list(pnl > 0))
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.size and losses.sum() < 0 else float("inf") if wins.size else float("nan"),
        "expectancy_sol": float(pnl.mean()),
        "expectancy_ret": float(ret.mean()),
        "expectancy_r": float(rm.mean()),
        "avg_r_multiple": float(rm.mean()),
        "kelly_fraction": _safe(kelly),
        "payoff_ratio": _safe(payoff),
        "avg_win_sol": avg_win,
        "avg_loss_sol": avg_loss,
        "median_trade_sol": float(np.median(pnl)),
        "median_trade_ret": float(np.median(ret)),
        "largest_win_sol": float(pnl.max()),
        "largest_loss_sol": float(pnl.min()),
        "avg_mae": float(trades["mae"].mean()),
        "avg_mfe": float(trades["mfe"].mean()),
        "avg_hold_s": float(trades["hold_s"].mean()),
        "max_win_streak": sw,
        "max_loss_streak": sl,
        "total_fees_sol": float(trades["fees_sol"].sum()),
        "total_tx_costs_sol": float(trades["tx_costs_sol"].sum()),
    }


def compute_metrics(equity: pl.DataFrame, trades: pl.DataFrame, initial_sol: float, returns_bar_ms: int,
                    annualization_days: float, fills: pl.DataFrame | None = None) -> dict[str, Any]:
    """Full metric set for a backtest / live session."""
    out: dict[str, Any] = {"initial_equity_sol": initial_sol}
    if equity.height >= 1:
        ts = equity["ts_ms"].to_numpy().astype(np.int64)
        eq = equity["equity_sol"].to_numpy().astype(np.float64)
        final = float(eq[-1])
        span_ms = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        days = span_ms / 86_400_000
        total_ret = final / initial_sol - 1.0
        growth = final / initial_sol if initial_sol > 0 else float("nan")
        if days > 0 and growth > 0:
            log_cagr = (365.0 / days) * math.log(growth)
            cagr = math.expm1(log_cagr) if log_cagr < 700 else float("inf")
        else:
            cagr = float("nan")
        if len(ts) > 1:  # the initial capital is the first peak
            max_dd, longest_uw, frac_uw = _drawdown_stats(np.concatenate([[ts[0]], ts]).astype(np.float64),
                                                          np.concatenate([[initial_sol], eq]))
        else:
            max_dd, longest_uw, frac_uw = 0.0, 0.0, 0.0
        ppy = annualization_days * 86_400_000 / returns_bar_ms
        r = bar_returns(ts, eq, returns_bar_ms)
        out.update({
            "final_equity_sol": final, "total_return": total_ret, "pnl_sol": final - initial_sol,
            "span_days": days, "cagr": cagr if days >= MIN_ANNUALISATION_DAYS else float("nan"),
            "cagr_extrapolated": days < MIN_ANNUALISATION_DAYS, "cagr_raw": cagr,
            "max_drawdown": max_dd, "longest_underwater_h": longest_uw / 3_600_000, "pct_time_underwater": frac_uw,
            # Calmar needs a meaningful annual return; below MIN_ANNUALISATION_DAYS use the period
            # return over max drawdown (recovery factor) instead of a nonsensical extrapolation.
            "calmar": (cagr / max_dd) if max_dd > 0 and math.isfinite(cagr) and days >= MIN_ANNUALISATION_DAYS else float("nan"),
            "recovery_factor": (total_ret / max_dd) if max_dd > 0 else float("nan"),
            "n_return_bars": int(r.size), "avg_exposure_sol": float(equity["exposure_sol"].mean()),
        })
        out.update(return_metrics(r, ppy))
    out.update(trade_metrics(trades))
    if fills is not None and fills.height and "status" in fills.columns:
        st = fills["status"]
        n = fills.height
        out["fill_rate"] = float(st.is_in(["filled", "partial"]).sum() / n)
        out["failed_rate"] = float((st == "failed").sum() / n)
        out["dropped_rate"] = float(st.is_in(["dropped", "expired"]).sum() / n)
        out["rejected_rate"] = float((st == "rejected").sum() / n)
        filled = fills.filter(pl.col("status").is_in(["filled", "partial"]))
        if filled.height:
            out["avg_latency_ms"] = float(filled["latency_ms"].mean())
            out["p90_latency_ms"] = float(filled["latency_ms"].quantile(0.9))
            out["avg_slippage_bps"] = float(filled["slippage_bps"].mean())
        out["total_priority_fees_sol"] = float(fills["priority_fee"].sum()) / 1e9
        out["total_network_fees_sol"] = float(fills["network_fee"].sum()) / 1e9
        out["total_jito_tips_sol"] = float(fills["jito_tip"].sum()) / 1e9
    return out


def monthly_returns(equity: pl.DataFrame) -> pl.DataFrame:
    """Calendar-month returns from the equity curve."""
    if equity.height < 2:
        return pl.DataFrame(schema={"year": pl.Int32, "month": pl.Int8, "ret": pl.Float64})
    e = equity.with_columns(pl.from_epoch("ts_ms", time_unit="ms").alias("dt")).sort("dt")
    m = e.group_by_dynamic("dt", every="1mo").agg(pl.col("equity_sol").first().alias("start"), pl.col("equity_sol").last().alias("end"))
    prev_end = m["end"].shift(1).fill_null(m["start"])
    return m.with_columns((pl.col("end") / prev_end - 1.0).alias("ret"), pl.col("dt").dt.year().alias("year"),
                          pl.col("dt").dt.month().alias("month")).select("year", "month", "ret")


def hourly_profile(trades: pl.DataFrame) -> pl.DataFrame:
    """PnL by UTC hour of entry and weekday (for heatmaps)."""
    if trades.height == 0:
        return pl.DataFrame(schema={"weekday": pl.Int8, "hour": pl.Int8, "pnl_sol": pl.Float64, "n": pl.UInt32})
    t = trades.with_columns(pl.from_epoch("entry_ms", time_unit="ms").alias("dt"))
    return (t.group_by(pl.col("dt").dt.weekday().alias("weekday"), pl.col("dt").dt.hour().alias("hour"))
            .agg(pl.col("pnl_sol").sum(), pl.len().alias("n"), pl.col("ret").mean().alias("avg_ret"))
            .sort(["weekday", "hour"]))
