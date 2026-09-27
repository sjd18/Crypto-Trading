"""Monte Carlo validation.

Two complementary layers:

1. **Trade-level** (fast, thousands of paths, numba): resample the closed trades (i.i.d.
   bootstrap, permutation of order, or block bootstrap preserving streaks) and perturb each
   trade independently:

   * position size   x LogNormal(0, sizing_sigma)
   * slippage cost   x LogNormal(0, slippage_sigma) applied to the measured slippage of the trade
   * fees            x LogNormal(0, fee_sigma) applied to fees + transaction costs
   * latency         x LogNormal(0, latency_sigma); extra latency costs
                     ``latency_cost_bps_per_100ms`` per 100 ms of the trade's notional

   Paths accumulate PnL on the initial capital; we record final return, CAGR (over the backtest
   span), max drawdown, longest drawdown (in trades) and ruin (equity <= ``ruin_equity_frac``).

2. **Path-level** (slow, full fidelity): re-run the event-driven backtest ``path_sims`` times with
   new random seeds and latency / failure-rate multipliers drawn from the configured ranges.

Outputs: quantiles, probability of ruin, worst / best cases and a sample of equity paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.utils.numba_compat import njit


@njit(cache=True)
def _simulate(pnl: np.ndarray, slip: np.ndarray, fees: np.ndarray, notional: np.ndarray, idx: np.ndarray,
              size_m: np.ndarray, slip_m: np.ndarray, fee_m: np.ndarray, lat_extra_frac: np.ndarray,
              initial: float, ruin_level: float, keep_paths: int, path_out: np.ndarray) -> np.ndarray:
    n_sims, n_tr = idx.shape
    out = np.empty((n_sims, 4))  # final equity, max dd, ruin flag, longest dd (trades)
    for s in range(n_sims):
        eq = initial
        peak = initial
        max_dd = 0.0
        ruin = 0.0
        dd_len = 0
        longest = 0
        for k in range(n_tr):
            j = idx[s, k]
            adj = pnl[j] - (slip_m[s, k] - 1.0) * slip[j] - (fee_m[s, k] - 1.0) * fees[j] - lat_extra_frac[s, k] * notional[j]
            eq += size_m[s, k] * adj
            if eq > peak:
                peak = eq
                dd_len = 0
            else:
                dd_len += 1
                if dd_len > longest:
                    longest = dd_len
            dd = 1.0 - eq / peak if peak > 0 else 1.0
            if dd > max_dd:
                max_dd = dd
            if eq <= ruin_level:
                ruin = 1.0
            if s < keep_paths:
                path_out[s, k] = eq
        out[s, 0] = eq
        out[s, 1] = max_dd
        out[s, 2] = ruin
        out[s, 3] = longest
    return out


@dataclass
class MonteCarloResult:
    n_sims: int
    method: str
    quantiles: dict[str, dict[str, float]]
    prob_ruin: float
    prob_loss: float
    worst: dict[str, float]
    best: dict[str, float]
    distributions: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    paths: np.ndarray | None = field(repr=False, default=None)
    path_level: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"n_sims": self.n_sims, "method": self.method, "prob_ruin": self.prob_ruin, "prob_loss": self.prob_loss,
                "quantiles": self.quantiles, "worst": self.worst, "best": self.best,
                "path_level_runs": len(self.path_level)}


def _indices(rng: np.random.Generator, n_sims: int, n_tr: int, method: str, block: int) -> np.ndarray:
    if method == "permutation":
        return np.argsort(rng.random((n_sims, n_tr)), axis=1)
    if method == "block_bootstrap":
        n_blocks = int(math.ceil(n_tr / block))
        starts = rng.integers(0, max(1, n_tr - block + 1), size=(n_sims, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :n_tr]
        return np.minimum(idx, n_tr - 1)
    return rng.integers(0, n_tr, size=(n_sims, n_tr))


def trade_monte_carlo(trades: pl.DataFrame, initial_sol: float, span_days: float, cfg: Any, avg_latency_ms: float = 500.0,
                      n_sims: int | None = None, seed: int | None = None, keep_paths: int = 200) -> MonteCarloResult:
    """Trade-level Monte Carlo (see module docstring)."""
    n = n_sims or cfg.n_sims
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    if trades.height == 0:
        return MonteCarloResult(n, cfg.method, {}, 0.0, 0.0, {}, {})
    pnl = trades["pnl_sol"].to_numpy().astype(np.float64)
    slip = np.abs(trades["slippage_sol"].to_numpy().astype(np.float64)) if "slippage_sol" in trades.columns else np.zeros_like(pnl)
    fees = (trades["fees_sol"] + trades["tx_costs_sol"]).to_numpy().astype(np.float64)
    notional = trades["cost_sol"].to_numpy().astype(np.float64)
    n_tr = pnl.size
    idx = _indices(rng, n, n_tr, cfg.method, cfg.block_size).astype(np.int64)
    size_m = rng.lognormal(0.0, cfg.sizing_sigma, (n, n_tr))
    slip_m = rng.lognormal(0.0, cfg.slippage_sigma, (n, n_tr))
    fee_m = rng.lognormal(0.0, cfg.fee_sigma, (n, n_tr))
    lat_m = rng.lognormal(0.0, cfg.latency_sigma, (n, n_tr))
    lat_extra = (lat_m - 1.0) * avg_latency_ms / 100.0 * cfg.latency_cost_bps_per_100ms / 1e4
    kp = min(keep_paths, n)
    paths = np.zeros((kp, n_tr))
    res = _simulate(pnl, slip, fees, notional, idx, size_m, slip_m, fee_m, lat_extra, float(initial_sol),
                    float(initial_sol * cfg.ruin_equity_frac), kp, paths)
    final = res[:, 0]
    total_ret = final / initial_sol - 1.0
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        growth = np.maximum(final / initial_sol, 1e-12)
        cagr = np.expm1(np.minimum(np.log(growth) * (365.0 / max(span_days, 1e-9)), 700.0))
    if span_days < 30:  # annualising a short sample is meaningless: report NaN rather than astronomic numbers
        cagr = np.full_like(total_ret, np.nan)
    dists = {"total_return": total_ret, "cagr": cagr, "max_drawdown": res[:, 1], "longest_dd_trades": res[:, 3]}
    qs = cfg.quantiles
    quantiles = {k: {f"q{int(q * 100):02d}": (float(np.quantile(v, q)) if np.isfinite(v).any() else float("nan")) for q in qs}
                 for k, v in dists.items()}
    worst_i, best_i = int(np.argmin(final)), int(np.argmax(final))
    return MonteCarloResult(
        n_sims=n, method=cfg.method, quantiles=quantiles, prob_ruin=float(res[:, 2].mean()), prob_loss=float((total_ret < 0).mean()),
        worst={"total_return": float(total_ret[worst_i]), "max_drawdown": float(res[worst_i, 1])},
        best={"total_return": float(total_ret[best_i]), "max_drawdown": float(res[best_i, 1])},
        distributions=dists, paths=paths,
    )


def path_monte_carlo(settings: Any, events: pl.DataFrame, strategies: list[str], metadata: dict[str, Any] | None,
                     n_paths: int | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    """Full event-driven re-simulations with random seeds and perturbed latency / failure rates."""
    from pumpfun_hft.backtester.engine import BacktestEngine
    from pumpfun_hft.backtester.replay import DataSource

    mc = settings.montecarlo
    rng = np.random.default_rng(mc.seed if seed is None else seed)
    out = []
    src = DataSource(frame=events)
    for i in range(n_paths or mc.path_sims):
        lat = float(rng.uniform(*mc.path_latency_scale))
        fail = float(rng.uniform(*mc.path_failure_scale))
        sd = int(rng.integers(0, 2**31 - 1))
        res = BacktestEngine(settings, src, strategies, metadata=metadata, seed=sd, latency_scale=lat, failure_scale=fail,
                             record_signals=False).run()
        m = res.metrics
        out.append({"path": i, "seed": sd, "latency_scale": lat, "failure_scale": fail, "total_return": m.get("total_return"),
                    "max_drawdown": m.get("max_drawdown"), "sharpe": m.get("sharpe"), "n_trades": m.get("n_trades"),
                    "fill_rate": m.get("fill_rate")})
    return out
