"""Optimisation studies and walk-forward analysis on chronological splits.

``OptimizationStudy``
    1. split the data (train / validation / test / live-sim, embargoed; test & live-sim sealed)
    2. search the parameter space on **train** only (grid / random / Bayesian / genetic)
    3. re-evaluate the top-k candidates on **validation** and select (best validation score, or
       "plateau": prefer candidates whose train-score neighbourhood is robust, then best validation)
    4. report selection-bias diagnostics: Deflated Sharpe Ratio and PBO (CSCV) over all trials
    5. optionally unseal **test** and **live-sim** exactly once for the final, untouched estimate

``WalkForwardAnalysis``
    Repeats search -> validation selection -> out-of-sample test over rolling (or anchored) folds and
    stitches the out-of-sample periods; reports walk-forward efficiency (OOS / IS score) and
    parameter stability across folds.

Trials run in a process pool (``optimizer.n_workers``); each worker loads the event data once.
"""

from __future__ import annotations

import math
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.replay import DataSource, load_metadata
from pumpfun_hft.core.config import Settings
from pumpfun_hft.core.types import EventKind
from pumpfun_hft.optimizer.objective import objective_value
from pumpfun_hft.optimizer.overfitting import deflated_sharpe_ratio, pbo_cscv, sharpe
from pumpfun_hft.optimizer.search import Trial, run_search
from pumpfun_hft.optimizer.space import ParamSpace
from pumpfun_hft.optimizer.splits import TimeSplit, make_splits, walk_forward_folds
from pumpfun_hft.utils.logging import get_logger

log = get_logger("backtests")
_W: dict[str, Any] = {}
_METRIC_KEYS = ("total_return", "sharpe", "sortino", "max_drawdown", "n_trades", "win_rate", "profit_factor",
                "expectancy_ret", "avg_r_multiple", "fill_rate", "calmar")


def _init_worker(settings_json: str, events_path: str, meta_path: str | None) -> None:
    _W["settings"] = Settings.model_validate_json(settings_json)
    _W["events"] = pl.read_parquet(events_path)
    _W["meta"] = load_metadata(meta_path) if meta_path else {}
    _W["creates"] = (_W["events"].filter(pl.col("kind") == EventKind.CREATE.value).select("mint", "ts_ms"))


def _evaluate(task: tuple[str, dict[str, Any], int, int, int, int]) -> tuple[float, dict[str, Any]]:
    strategy, params, warm_start, start, end, seed = task
    s: Settings = _W["settings"]
    ev: pl.DataFrame = _W["events"]
    frame = ev.filter((pl.col("ts_ms") >= warm_start) & (pl.col("ts_ms") < end))
    cr = _W["creates"]
    allowed = set(cr.filter((pl.col("ts_ms") >= start) & (pl.col("ts_ms") < end))["mint"].to_list())
    eng = BacktestEngine(s, DataSource(frame=frame), [strategy], metadata=_W["meta"], seed=seed, trade_start_ms=start,
                         allowed_mints=allowed, strategy_overrides={strategy: params}, record_signals=False)
    res = eng.run()
    o = s.optimizer
    score = objective_value(res.metrics, o.objective, o.min_trades, o.max_drawdown_pct)
    eq = res.equity.filter(pl.col("ts_ms") >= start)
    e = eq["equity_sol"].to_numpy()
    rets = (e[1:] / e[:-1] - 1.0) if e.size > 1 else np.empty(0)
    info = {"metrics": {k: res.metrics.get(k) for k in _METRIC_KEYS}, "returns": rets.tolist()}
    return score, info


def _mp_method() -> str:
    """'spawn' (safe with threads) when the main module is importable or 'fork' does not exist (Windows),
    else 'fork' (code piped through stdin / a REPL, which 'spawn' cannot re-import)."""
    import multiprocessing
    import os
    import sys

    if "fork" not in multiprocessing.get_all_start_methods():
        return "spawn"
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file is None or os.path.exists(main_file):
        return "spawn"
    return "fork"


class SplitEvaluator:
    """Evaluates parameter batches for one strategy on one split (serial or process pool)."""

    def __init__(self, settings: Settings, events: pl.DataFrame, metadata_path: str | None, strategy: str,
                 n_workers: int, seed: int) -> None:
        self.settings = settings
        self.strategy = strategy
        self.seed = seed
        self.n_workers = max(1, n_workers)
        self._tmp = Path(tempfile.mkdtemp(prefix="pumpfun-opt-"))
        self._events_path = self._tmp / "events.parquet"
        events.write_parquet(self._events_path)
        args = (settings.model_dump_json(), str(self._events_path), metadata_path)
        self.pool: ProcessPoolExecutor | None = None
        if self.n_workers > 1:
            self.pool = ProcessPoolExecutor(self.n_workers, mp_context=get_context(_mp_method()), initializer=_init_worker, initargs=args)
        else:
            _init_worker(*args)
        self.split: TimeSplit | None = None
        self.warm_start = 0

    def use(self, split: TimeSplit, warm_start: int) -> SplitEvaluator:
        self.split, self.warm_start = split, warm_start
        return self

    def __call__(self, params_list: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
        assert self.split is not None
        tasks = [(self.strategy, p, self.warm_start, self.split.start_ms, self.split.end_ms, self.seed) for p in params_list]
        if self.pool is not None:
            return list(self.pool.map(_evaluate, tasks))
        return [_evaluate(t) for t in tasks]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
        for p in self._tmp.glob("*"):
            p.unlink(missing_ok=True)
        self._tmp.rmdir()


def _plateau_scores(space: ParamSpace, trials: list[Trial], k: int = 5) -> list[float]:
    x = np.array([space.encode(t.params) for t in trials])
    y = np.array([t.score for t in trials])
    out = []
    for i in range(len(trials)):
        d = np.sqrt(((x - x[i]) ** 2).sum(1))
        nn = np.argsort(d)[: min(k, len(trials))]
        out.append(float(np.median(y[nn])))
    return out


def _aligned(returns: list[list[float]]) -> np.ndarray:
    n = min((len(r) for r in returns if r), default=0)
    if n == 0:
        return np.empty((0, 0))
    return np.array([r[:n] for r in returns if len(r) >= n]).T


@dataclass
class StudyResult:
    study_id: str
    strategy: str
    method: str
    splits: list[str]
    trials: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    selected_params: dict[str, Any]
    selected_train_metrics: dict[str, Any]
    selected_val_metrics: dict[str, Any]
    dsr: dict[str, float]
    pbo: dict[str, float]
    test_metrics: dict[str, Any] | None = None
    live_sim_metrics: dict[str, Any] | None = None
    unseal_log: list[dict[str, Any]] = field(default_factory=list)

    def trials_frame(self) -> pl.DataFrame:
        rows = [{"trial": t["number"], "score": t["score"], **{f"p_{k}": v for k, v in t["params"].items()},
                 **{f"m_{k}": v for k, v in (t.get("metrics") or {}).items()}} for t in self.trials]
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


class OptimizationStudy:
    """Search on train, select on validation, report DSR/PBO, optionally unseal test once.

    Example::

        study = OptimizationStudy(settings, events, metadata_path, "momentum_ignition")
        res = study.run(final_evaluation=True)
        res.selected_params, res.dsr["dsr"], res.pbo["pbo"], res.test_metrics
    """

    def __init__(self, settings: Settings, events: pl.DataFrame, metadata_path: str | None, strategy: str,
                 method: str | None = None, n_trials: int | None = None, seed: int | None = None, n_workers: int | None = None) -> None:
        self.s = settings
        self.events = events
        self.metadata_path = metadata_path
        self.strategy = strategy
        self.method = method or settings.optimizer.method
        self.n_trials = n_trials or settings.optimizer.n_trials
        self.seed = settings.optimizer.seed if seed is None else seed
        self.n_workers = settings.optimizer.n_workers if n_workers is None else n_workers
        if strategy not in settings.optimizer.spaces:
            raise KeyError(f"no optimizer.spaces entry for {strategy}")
        self.space = ParamSpace(settings.optimizer.spaces[strategy])

    def run(self, final_evaluation: bool = False) -> StudyResult:
        o = self.s.optimizer
        start, end = int(self.events["ts_ms"].min()), int(self.events["ts_ms"].max()) + 1
        splits = make_splits(start, end, o.splits)
        ev = SplitEvaluator(self.s, self.events, self.metadata_path, self.strategy, self.n_workers, self.seed)
        try:
            ev.use(splits.train, start)
            trials = run_search(self.method, self.space, ev, o, self.seed, self.n_trials, batch=self.n_workers)
            ranked = sorted(trials, key=lambda t: t.score, reverse=True)
            if o.selection == "plateau":
                plateau = _plateau_scores(self.space, trials)
                by_plateau = sorted(range(len(trials)), key=lambda i: plateau[i], reverse=True)
                cands = [trials[i] for i in by_plateau[: o.top_k]]
            else:
                cands = ranked[: o.top_k]
            ev.use(splits.validation, start)
            val = ev([c.params for c in cands])
            cand_rows = [{"params": c.params, "train_score": c.score, "val_score": v[0], "val_metrics": v[1]["metrics"],
                          "train_metrics": c.info.get("metrics")} for c, v in zip(cands, val, strict=True)]
            best_i = int(np.argmax([r["val_score"] for r in cand_rows]))
            chosen = cands[best_i]
            trial_sharpes = [sharpe(np.array(t.info.get("returns") or [])) for t in trials]
            dsr = deflated_sharpe_ratio(np.array(chosen.info.get("returns") or []), trial_sharpes)
            pbo = pbo_cscv(_aligned([t.info.get("returns") or [] for t in trials]))
            result = StudyResult(
                study_id=f"study-{uuid.uuid4().hex[:8]}", strategy=self.strategy, method=self.method, splits=splits.describe(),
                trials=[{"number": t.number, "params": t.params, "score": t.score, "metrics": t.info.get("metrics")} for t in trials],
                candidates=cand_rows, selected_params=chosen.params, selected_train_metrics=chosen.info.get("metrics") or {},
                selected_val_metrics=cand_rows[best_i]["val_metrics"], dsr=dsr, pbo=pbo)
            if final_evaluation:
                test = splits.test.unseal(f"final evaluation of {self.strategy} {chosen.params}")
                ev.use(test, start)
                result.test_metrics = ev([chosen.params])[0][1]["metrics"]
                live = splits.live_sim.unseal(f"live-simulation of {self.strategy} {chosen.params}")
                ev.use(live, start)
                result.live_sim_metrics = ev([chosen.params])[0][1]["metrics"]
                result.unseal_log = splits.test.audit + splits.live_sim.audit
        finally:
            ev.close()
        log.info("study finished", extra={"data": {"strategy": self.strategy, "method": self.method, "trials": len(trials),
                                                   "selected": result.selected_params, "dsr": result.dsr.get("dsr"),
                                                   "pbo": result.pbo.get("pbo")}})
        return result


@dataclass
class FoldResult:
    index: int
    train: str
    validation: str
    test: str
    params: dict[str, Any]
    is_score: float
    val_score: float
    oos_score: float
    oos_metrics: dict[str, Any]
    oos_returns: list[float]


@dataclass
class WalkForwardResult:
    strategy: str
    method: str
    folds: list[FoldResult]
    wfe: float
    oos_total_return: float
    oos_sharpe_per_period: float
    param_stability: dict[str, float]

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame([{"fold": f.index, "is_score": f.is_score, "val_score": f.val_score, "oos_score": f.oos_score,
                              **{f"p_{k}": v for k, v in f.params.items()},
                              **{f"oos_{k}": v for k, v in f.oos_metrics.items()}} for f in self.folds], infer_schema_length=None)


class WalkForwardAnalysis:
    """Rolling / anchored walk-forward optimisation with stitched out-of-sample results."""

    def __init__(self, settings: Settings, events: pl.DataFrame, metadata_path: str | None, strategy: str,
                 method: str | None = None, n_trials: int | None = None, seed: int | None = None, n_workers: int | None = None) -> None:
        self.s = settings
        self.events = events
        self.metadata_path = metadata_path
        self.strategy = strategy
        self.method = method or settings.optimizer.method
        self.n_trials = n_trials or settings.optimizer.n_trials
        self.seed = settings.optimizer.seed if seed is None else seed
        self.n_workers = settings.optimizer.n_workers if n_workers is None else n_workers
        self.space = ParamSpace(settings.optimizer.spaces[strategy])

    def run(self) -> WalkForwardResult:
        o = self.s.optimizer
        start, end = int(self.events["ts_ms"].min()), int(self.events["ts_ms"].max()) + 1
        folds = walk_forward_folds(start, end, o.walk_forward)
        ev = SplitEvaluator(self.s, self.events, self.metadata_path, self.strategy, self.n_workers, self.seed)
        results: list[FoldResult] = []
        try:
            for f in folds:
                ev.use(f.train, start)
                trials = run_search(self.method, self.space, ev, o, self.seed + f.index, self.n_trials, batch=self.n_workers)
                cands = sorted(trials, key=lambda t: t.score, reverse=True)[: o.top_k]
                ev.use(f.validation, start)
                val = ev([c.params for c in cands])
                bi = int(np.argmax([v[0] for v in val]))
                chosen = cands[bi]
                ev.use(f.test, start)
                oos_score, oos_info = ev([chosen.params])[0]
                results.append(FoldResult(f.index, f.train.describe(), f.validation.describe(), f.test.describe(), chosen.params,
                                          chosen.score, val[bi][0], oos_score, oos_info["metrics"], oos_info["returns"]))
        finally:
            ev.close()
        is_scores = [r.is_score for r in results if math.isfinite(r.is_score) and r.is_score > -100]
        oos_scores = [r.oos_score for r in results if math.isfinite(r.oos_score) and r.oos_score > -100]
        wfe = (float(np.mean(oos_scores)) / float(np.mean(is_scores))) if is_scores and oos_scores and np.mean(is_scores) != 0 else float("nan")
        stitched = np.concatenate([np.array(r.oos_returns) for r in results]) if results else np.empty(0)
        total = float(np.prod(1.0 + stitched) - 1.0) if stitched.size else 0.0
        enc = np.array([self.space.encode(r.params) for r in results]) if results else np.empty((0, self.space.dim))
        stability = {n: float(enc[:, i].std()) for i, n in enumerate(self.space.names)} if len(results) > 1 else {}
        return WalkForwardResult(self.strategy, self.method, results, wfe, total, sharpe(stitched), stability)
