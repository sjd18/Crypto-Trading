"""Monte Carlo: determinism, invariants of resampling, stress direction, ruin."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pumpfun_hft.analytics.montecarlo import trade_monte_carlo
from pumpfun_hft.tests.conftest import make_settings


def _trades(pnls: list[float]) -> pl.DataFrame:
    n = len(pnls)
    return pl.DataFrame({"pnl_sol": pnls, "slippage_sol": [0.002] * n, "fees_sol": [0.005] * n, "tx_costs_sol": [0.0002] * n,
                         "cost_sol": [0.2] * n})


def _cfg(**kw):
    return make_settings(**{f"montecarlo.{k}": v for k, v in kw.items()}).montecarlo


def test_deterministic_for_a_seed() -> None:
    t = _trades(list(np.random.default_rng(0).normal(0.001, 0.02, 200)))
    a = trade_monte_carlo(t, 10.0, 60.0, _cfg(), n_sims=500, seed=1)
    b = trade_monte_carlo(t, 10.0, 60.0, _cfg(), n_sims=500, seed=1)
    assert np.array_equal(a.distributions["total_return"], b.distributions["total_return"])


def test_permutation_without_perturbation_preserves_total_pnl() -> None:
    pnls = list(np.random.default_rng(1).normal(0.0, 0.05, 100))
    cfg = _cfg(method="permutation", slippage_sigma=0.0, fee_sigma=0.0, latency_sigma=0.0, sizing_sigma=0.0)
    mc = trade_monte_carlo(_trades(pnls), 10.0, 60.0, cfg, n_sims=300, seed=2)
    assert np.allclose(mc.distributions["total_return"], sum(pnls) / 10.0)  # order changes the path, never the sum
    assert mc.distributions["max_drawdown"].std() > 0  # ...but it does change the drawdown


def test_cost_stress_is_adverse_on_average() -> None:
    t = _trades([0.01] * 300)
    calm = trade_monte_carlo(t, 10.0, 60.0, _cfg(slippage_sigma=0.0, fee_sigma=0.0, latency_sigma=0.0, sizing_sigma=0.0), n_sims=400)
    stressed = trade_monte_carlo(t, 10.0, 60.0, _cfg(slippage_sigma=0.8, fee_sigma=0.5, latency_sigma=0.8, sizing_sigma=0.0), n_sims=400)
    assert stressed.distributions["total_return"].mean() < calm.distributions["total_return"].mean()


def test_probability_of_ruin_and_loss() -> None:
    losing = _trades([-0.2] * 40)
    mc = trade_monte_carlo(losing, 10.0, 60.0, _cfg(ruin_equity_frac=0.5), n_sims=200)
    assert mc.prob_loss == pytest.approx(1.0) and mc.prob_ruin > 0.9
    winning = _trades([0.05] * 40)
    mc2 = trade_monte_carlo(winning, 10.0, 60.0, _cfg(), n_sims=200)
    assert mc2.prob_ruin == 0.0 and mc2.prob_loss < 0.05


def test_short_spans_do_not_report_cagr() -> None:
    mc = trade_monte_carlo(_trades([0.01] * 50), 10.0, 0.5, _cfg(), n_sims=100)
    assert np.isnan(mc.distributions["cagr"]).all()
    assert set(mc.quantiles["total_return"]) == {"q05", "q25", "q50", "q75", "q95"}
    assert mc.paths is not None and mc.paths.shape == (100, 50)


def test_empty_trades() -> None:
    mc = trade_monte_carlo(_trades([]), 10.0, 1.0, _cfg(), n_sims=10)
    assert mc.prob_ruin == 0.0 and mc.quantiles == {}
