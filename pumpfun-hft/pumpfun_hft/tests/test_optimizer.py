"""Optimiser: search algorithms on known functions, sealed splits, folds, overfitting statistics."""

from __future__ import annotations

import numpy as np
import pytest

from pumpfun_hft.core.config import ParamSpec
from pumpfun_hft.optimizer.objective import PENALTY, objective_value
from pumpfun_hft.optimizer.overfitting import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    sharpe,
)
from pumpfun_hft.optimizer.search import run_search
from pumpfun_hft.optimizer.space import ParamSpace
from pumpfun_hft.optimizer.splits import SealedSplitError, make_splits, walk_forward_folds
from pumpfun_hft.tests.conftest import make_settings

DAY = 86_400_000


@pytest.fixture(scope="module")
def space() -> ParamSpace:
    return ParamSpace({
        "x": ParamSpec(type="float", low=-2.0, high=2.0),
        "y": ParamSpec(type="float", low=-2.0, high=2.0),
        "k": ParamSpec(type="int", low=1, high=5),
        "mode": ParamSpec(type="categorical", choices=["a", "b"]),
    })


def _objective(p: dict) -> float:  # maximum 3.0 at x=0.7, y=-0.3, k=3, mode="b"
    return 3.0 - (p["x"] - 0.7) ** 2 - (p["y"] + 0.3) ** 2 - 0.1 * (p["k"] - 3) ** 2 - (0.0 if p["mode"] == "b" else 0.5)


def _evaluate(batch: list[dict]) -> list[tuple[float, dict]]:
    return [(_objective(p), {"metrics": {}}) for p in batch]


def test_space_round_trip(space: ParamSpace) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = space.sample(rng)
        assert -2 <= p["x"] <= 2 and 1 <= p["k"] <= 5 and p["mode"] in ("a", "b")
        q = space.decode(space.encode(p))
        assert q["k"] == p["k"] and q["mode"] == p["mode"] and q["x"] == pytest.approx(p["x"], abs=1e-9)
    assert len(space.grid(3, 1000)) == 3 * 3 * 3 * 2


@pytest.mark.parametrize(("method", "n", "target"), [("random", 120, 2.3), ("bayesian", 40, 2.6), ("genetic", 80, 2.3),
                                                    ("grid", 400, 2.3)])
def test_search_finds_the_optimum(space: ParamSpace, method: str, n: int, target: float) -> None:
    cfg = make_settings().optimizer
    trials = run_search(method, space, _evaluate, cfg, seed=3, n_trials=n, batch=4)
    assert len(trials) <= max(n, 1) + 8
    best = max(trials, key=lambda t: t.score)
    assert best.score >= target
    assert best.params["mode"] == "b"


def test_bayesian_beats_random_on_the_same_budget(space: ParamSpace) -> None:
    cfg = make_settings().optimizer
    bo = [max(t.score for t in run_search("bayesian", space, _evaluate, cfg, seed=s, n_trials=30, batch=2)) for s in range(3)]
    rs = [max(t.score for t in run_search("random", space, _evaluate, cfg, seed=s, n_trials=30, batch=2)) for s in range(3)]
    assert np.mean(bo) > np.mean(rs)


def test_splits_are_ordered_embargoed_and_sealed() -> None:
    cfg = make_settings().optimizer.splits
    sp = make_splits(0, 10 * DAY, cfg)
    assert sp.train.end_ms + sp.embargo_ms == sp.validation.start_ms
    with pytest.raises(SealedSplitError):
        _ = sp.test.split
    test = sp.test.unseal("final evaluation")
    assert test.start_ms == sp.validation.end_ms + sp.embargo_ms
    assert sp.test.audit and sp.test.audit[0]["reason"] == "final evaluation"
    live = sp.live_sim.unseal("live-sim")
    assert live.start_ms >= test.end_ms + sp.embargo_ms and live.end_ms <= 10 * DAY


@pytest.mark.parametrize("anchored", [False, True])
def test_walk_forward_folds(anchored: bool) -> None:
    cfg = make_settings(**{"optimizer.walk_forward.anchored": anchored}).optimizer.walk_forward
    folds = walk_forward_folds(0, 30 * DAY, cfg)
    assert len(folds) == cfg.n_folds
    emb = int(cfg.embargo_s * 1000)
    for i, f in enumerate(folds):
        assert f.train.end_ms + emb <= f.validation.start_ms < f.validation.end_ms
        assert f.validation.end_ms + emb <= f.test.start_ms < f.test.end_ms
        if anchored:
            assert f.train.start_ms == 0
        if i:
            assert folds[i - 1].test.end_ms <= f.test.start_ms  # out-of-sample windows never overlap
    assert folds[-1].test.end_ms <= 30 * DAY


def test_pbo_detects_noise_and_real_edge() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=(500, 24))
    assert 0.3 <= pbo_cscv(noise)["pbo"] <= 0.75
    edge = noise.copy()
    edge[:, 5] += 0.4
    assert pbo_cscv(edge)["pbo"] < 0.1


def test_probabilistic_and_deflated_sharpe() -> None:
    assert probabilistic_sharpe_ratio(0.1, 0.0, 1000, 0.0, 3.0) > 0.99
    assert probabilistic_sharpe_ratio(0.01, 0.0, 50, 0.0, 3.0) < 0.6
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    rng = np.random.default_rng(2)
    r = rng.normal(0.002, 0.01, 2000)
    few = deflated_sharpe_ratio(r, [sharpe(rng.normal(0, 0.01, 2000)) for _ in range(3)])
    many = deflated_sharpe_ratio(r, [sharpe(rng.normal(0, 0.01, 2000)) for _ in range(300)])
    assert few["dsr"] >= many["dsr"]  # more trials -> a higher bar


def test_objective_constraints() -> None:
    ok = {"n_trades": 50, "max_drawdown": 0.1, "sharpe": 1.5}
    assert objective_value(ok, "sharpe", 20, 30.0) == pytest.approx(1.5)
    assert objective_value({**ok, "n_trades": 5}, "sharpe", 20, 30.0) == PENALTY + 5
    assert objective_value({**ok, "max_drawdown": 0.5}, "sharpe", 20, 30.0) < PENALTY / 2
    short = {"n_trades": 50, "max_drawdown": 0.1, "calmar": float("nan"), "recovery_factor": 0.8}
    assert objective_value(short, "calmar", 20, 30.0) == pytest.approx(0.8)


def test_gappy_recordings_split_by_event_count() -> None:
    from types import SimpleNamespace

    import numpy as np

    from pumpfun_hft.optimizer.splits import has_recording_gaps, make_splits, split_event_counts

    cfg = SimpleNamespace(train=0.5, validation=0.2, test=0.2, live_sim=0.1, embargo_s=60.0)
    day = 86_400_000
    # three recorded days spread over 14: clock-time validation (days 7-9.8) would hold no events at all
    ts = np.concatenate([np.arange(0, day, 1000), np.arange(10 * day, 11 * day, 1000), np.arange(13 * day, 14 * day, 1000)])
    assert has_recording_gaps(ts) and not has_recording_gaps(np.arange(0, 3 * day, 1000))
    by_time = split_event_counts(ts, make_splits(int(ts[0]), int(ts[-1]) + 1, cfg))
    assert by_time["validation"] == 0
    by_events = split_event_counts(ts, make_splits(int(ts[0]), int(ts[-1]) + 1, cfg, ts))
    total = ts.size
    for name, frac in (("train", 0.5), ("validation", 0.2), ("test", 0.2), ("live_sim", 0.1)):
        assert abs(by_events[name] / total - frac) < 0.01, (name, by_events)
    sp = make_splits(int(ts[0]), int(ts[-1]) + 1, cfg, ts)
    assert sp.train.end_ms + 60_000 == sp.validation.start_ms  # embargo kept, order kept


def test_event_splits_never_lose_a_split_to_the_embargo() -> None:
    from types import SimpleNamespace

    import numpy as np

    from pumpfun_hft.optimizer.splits import make_splits, split_event_counts

    cfg = SimpleNamespace(train=0.5, validation=0.2, test=0.2, live_sim=0.1, embargo_s=1800.0)
    hour = 3_600_000
    # a quiet day followed by a dense 30-minute recording that holds most of the events
    ts = np.concatenate([np.arange(0, 20 * hour, 60_000), np.arange(20 * hour, 20 * hour + hour // 2, 2)])
    sp = make_splits(int(ts[0]), int(ts[-1]) + 1, cfg, ts)
    counts = split_event_counts(ts, sp)
    assert all(counts[k] > 0.05 * ts.size for k in ("validation", "test", "live_sim")), counts
    assert 0 < sp.embargo_ms < 1_800_000

