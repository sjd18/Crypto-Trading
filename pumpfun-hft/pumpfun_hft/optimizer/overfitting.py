"""Backtest-overfitting diagnostics.

* Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012) — probability that the true
  Sharpe exceeds a benchmark given sample length, skewness and kurtosis.
* Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — PSR against the *expected maximum*
  Sharpe of N independent trials with the observed cross-trial variance; corrects for selection
  bias from trying many parameter sets.
* Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu, 2017) —
  share of combinatorial in-sample/out-of-sample partitions in which the in-sample winner ranks
  below the out-of-sample median.

All Sharpe ratios here are *per period* (not annualised).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """PSR = Phi((SR - SR*) sqrt(n-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)); ``kurtosis`` is non-excess."""
    if n_obs < 2 or not math.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] of ``n_trials`` independent trials with Sharpe variance ``var_sr`` (null of zero skill)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: list[float]) -> dict[str, float]:
    """DSR of the selected strategy's per-period ``returns`` given the Sharpes of all trials."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    srs = np.array([s for s in trial_sharpes if math.isfinite(s)], dtype=float)
    sr = sharpe(r)
    n_trials = max(1, srs.size)
    var_sr = float(srs.var(ddof=1)) if srs.size > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, var_sr)
    skew = float(stats.skew(r)) if r.size > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False)) if r.size > 3 else 3.0
    return {"sharpe": sr, "sr_star": sr_star, "n_trials": float(n_trials), "n_obs": float(r.size),
            "skew": skew, "kurtosis": kurt, "dsr": probabilistic_sharpe_ratio(sr, sr_star, r.size, skew, kurt),
            "psr_zero": probabilistic_sharpe_ratio(sr, 0.0, r.size, skew, kurt)}


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> dict[str, float]:
    """Probability of backtest overfitting.

    ``perf`` is a T x N matrix of per-period returns (T periods, N trials). Rows are cut into
    ``n_splits`` contiguous groups; every combination of half the groups is used as in-sample.
    """
    m = np.asarray(perf, dtype=float)
    t, n = m.shape
    if n < 2 or t < n_splits:
        return {"pbo": float("nan"), "n_combinations": 0.0, "median_logit": float("nan")}
    s = n_splits - n_splits % 2
    groups = np.array_split(np.arange(t), s)
    logits = []
    for is_idx in itertools.combinations(range(s), s // 2):
        is_rows = np.concatenate([groups[i] for i in is_idx])
        oos_rows = np.concatenate([groups[i] for i in range(s) if i not in is_idx])

        def col_sharpe(rows: np.ndarray) -> np.ndarray:
            x = m[rows]
            sd = x.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(sd > 0, x.mean(axis=0) / sd, -np.inf)

        is_sr, oos_sr = col_sharpe(is_rows), col_sharpe(oos_rows)
        best = int(np.argmax(is_sr))
        rank = float(stats.rankdata(oos_sr)[best])  # 1 = worst
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1.0 - omega)))
    arr = np.array(logits)
    return {"pbo": float((arr <= 0).mean()), "n_combinations": float(arr.size), "median_logit": float(np.median(arr))}
