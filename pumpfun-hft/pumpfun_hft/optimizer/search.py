"""Search algorithms: grid, random, Bayesian (Gaussian process + expected improvement) and genetic.

All searchers maximise a scalar objective through a batch evaluation callback
``evaluate(list[params]) -> list[(score, info)]`` so trials can run in parallel worker processes.

Bayesian optimisation
    GP on the [0, 1]^d encoding with an ARD Matérn-5/2 kernel; hyper-parameters (log length-scales,
    signal and noise variance) maximise the log marginal likelihood (L-BFGS-B, random restarts).
    Acquisition: expected improvement with exploration ``xi`` over random + local candidates.
    Batches of ``q`` points use the constant-liar heuristic (pending points take the worst
    observed value), so parallel workers explore different regions.

Genetic algorithm
    Tournament selection, uniform crossover, Gaussian mutation in encoded space, elitism and
    memoisation of already-evaluated parameter sets.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats

from pumpfun_hft.optimizer.space import ParamSpace

EvalFn = Callable[[list[dict[str, Any]]], list[tuple[float, dict[str, Any]]]]


@dataclass
class Trial:
    number: int
    params: dict[str, Any]
    score: float
    info: dict[str, Any] = field(default_factory=dict)
    method: str = ""
    elapsed_s: float = 0.0


# ----------------------------------------------------------------------------- Gaussian process
class GaussianProcess:
    """Minimal GP regressor with an ARD Matérn-5/2 kernel on [0, 1]^d inputs."""

    def __init__(self, dim: int, restarts: int = 3, rng: np.random.Generator | None = None) -> None:
        self.dim = dim
        self.restarts = restarts
        self.rng = rng or np.random.default_rng(0)
        self.log_ls = np.zeros(dim) + math.log(0.3)
        self.log_sf2 = 0.0
        self.log_sn2 = math.log(1e-3)

    @staticmethod
    def _matern52(x1: np.ndarray, x2: np.ndarray, ls: np.ndarray, sf2: float) -> np.ndarray:
        d = np.sqrt(np.maximum(((x1[:, None, :] - x2[None, :, :]) / ls) ** 2, 0.0).sum(-1))
        s5 = math.sqrt(5.0) * d
        return sf2 * (1.0 + s5 + 5.0 / 3.0 * d * d) * np.exp(-s5)

    def _nll(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        ls = np.exp(theta[: self.dim])
        sf2, sn2 = math.exp(theta[self.dim]), math.exp(theta[self.dim + 1])
        k = self._matern52(x, x, ls, sf2) + (sn2 + 1e-9) * np.eye(len(x))
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        return float(0.5 * y @ alpha + np.log(np.diag(chol)).sum() + 0.5 * len(x) * math.log(2 * math.pi))

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianProcess:
        self.x = np.asarray(x, dtype=float)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y) or 1.0)
        yn = (np.asarray(y, dtype=float) - self.y_mean) / self.y_std
        bounds = [(math.log(0.02), math.log(5.0))] * self.dim + [(math.log(0.05), math.log(20.0)), (math.log(1e-6), math.log(0.5))]
        best_theta, best_val = np.concatenate([self.log_ls, [self.log_sf2, self.log_sn2]]), math.inf
        starts = [best_theta] + [np.array([self.rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(self.restarts)]
        for th0 in starts:
            res = optimize.minimize(self._nll, th0, args=(self.x, yn), method="L-BFGS-B", bounds=bounds)
            if res.fun < best_val:
                best_val, best_theta = res.fun, res.x
        self.log_ls, self.log_sf2, self.log_sn2 = best_theta[: self.dim], best_theta[self.dim], best_theta[self.dim + 1]
        ls, sf2, sn2 = np.exp(self.log_ls), math.exp(self.log_sf2), math.exp(self.log_sn2)
        k = self._matern52(self.x, self.x, ls, sf2) + (sn2 + 1e-9) * np.eye(len(self.x))
        self._chol = np.linalg.cholesky(k)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, yn))
        return self

    def predict(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ls, sf2 = np.exp(self.log_ls), math.exp(self.log_sf2)
        ks = self._matern52(np.asarray(xs, dtype=float), self.x, ls, sf2)
        mu = ks @ self._alpha
        v = np.linalg.solve(self._chol, ks.T)
        var = np.maximum(sf2 - (v * v).sum(0), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    imp = mu - best - xi
    z = np.where(sigma > 0, imp / sigma, 0.0)
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return np.where(sigma > 0, ei, 0.0)


# ----------------------------------------------------------------------------- searchers
def _run(evaluate: EvalFn, batch: list[dict[str, Any]], trials: list[Trial], method: str) -> None:
    t0 = time.perf_counter()
    results = evaluate(batch)
    el = (time.perf_counter() - t0) / max(1, len(batch))
    for p, (score, info) in zip(batch, results, strict=True):
        trials.append(Trial(len(trials), p, float(score), info, method, el))


def grid_search(space: ParamSpace, evaluate: EvalFn, points_per_dim: int, max_points: int, batch: int = 4) -> list[Trial]:
    trials: list[Trial] = []
    grid = space.grid(points_per_dim, max_points)
    for i in range(0, len(grid), batch):
        _run(evaluate, grid[i:i + batch], trials, "grid")
    return trials


def random_search(space: ParamSpace, evaluate: EvalFn, n_trials: int, rng: np.random.Generator, batch: int = 4) -> list[Trial]:
    trials: list[Trial] = []
    seen: set[tuple[Any, ...]] = set()
    while len(trials) < n_trials:
        cands = []
        attempts = 0
        while len(cands) < min(batch, n_trials - len(trials)) and attempts < 1000:
            p = space.sample(rng)
            attempts += 1
            if space.key(p) not in seen:
                seen.add(space.key(p))
                cands.append(p)
        if not cands:
            break
        _run(evaluate, cands, trials, "random")
    return trials


def bayesian_search(space: ParamSpace, evaluate: EvalFn, n_trials: int, rng: np.random.Generator, n_initial: int,
                    n_candidates: int, xi: float, restarts: int, batch: int = 1) -> list[Trial]:
    trials = random_search(space, evaluate, min(n_initial, n_trials), rng, batch=max(batch, 1))
    for t in trials:
        t.method = "bayesian-init"
    gp = GaussianProcess(space.dim, restarts, rng)
    seen = {space.key(t.params) for t in trials}
    while len(trials) < n_trials:
        x = np.array([space.encode(t.params) for t in trials])
        y = np.array([t.score for t in trials])
        y = np.maximum(y, np.percentile(y, 10) - 1.0)  # damp huge constraint penalties for the GP fit
        q = min(batch, n_trials - len(trials))
        chosen: list[dict[str, Any]] = []
        xs_pending, ys_pending = list(x), list(y)
        for _ in range(q):
            gp.fit(np.array(xs_pending), np.array(ys_pending))
            best = float(np.max(ys_pending))
            top = np.array(xs_pending)[np.argsort(ys_pending)[-3:]]
            cand = np.vstack([rng.random((n_candidates, space.dim)),
                              np.clip(top[rng.integers(0, len(top), n_candidates // 4)]
                                      + rng.normal(0, 0.05, (n_candidates // 4, space.dim)), 0, 1)])
            mu, sd = gp.predict(cand)
            ei = expected_improvement(mu, sd, best, xi)
            order = np.argsort(-ei)
            for idx in order:
                p = space.decode(cand[idx])
                if space.key(p) not in seen:
                    break
            else:
                p = space.sample(rng)
            seen.add(space.key(p))
            chosen.append(p)
            xs_pending.append(space.encode(p))
            ys_pending.append(float(np.min(ys_pending)))  # constant liar (pessimistic)
        _run(evaluate, chosen, trials, "bayesian")
    return trials


def genetic_search(space: ParamSpace, evaluate: EvalFn, rng: np.random.Generator, population: int, generations: int,
                   crossover_prob: float, mutation_prob: float, mutation_scale: float, tournament: int, elite: int,
                   batch: int = 4) -> list[Trial]:
    trials: list[Trial] = []
    cache: dict[tuple[Any, ...], float] = {}

    def score_all(params: list[dict[str, Any]]) -> list[float]:
        todo = [p for p in params if space.key(p) not in cache]
        uniq: dict[tuple[Any, ...], dict[str, Any]] = {space.key(p): p for p in todo}
        items = list(uniq.values())
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            n0 = len(trials)
            _run(evaluate, chunk, trials, "genetic")
            for t in trials[n0:]:
                cache[space.key(t.params)] = t.score
        return [cache[space.key(p)] for p in params]

    def pick(pop: list[np.ndarray], fitness: list[float]) -> np.ndarray:
        """Tournament selection: the fittest of ``tournament`` random members."""
        idx = rng.integers(0, population, tournament)
        return pop[int(idx[np.argmax([fitness[i] for i in idx])])]

    pop = [space.encode(space.sample(rng)) for _ in range(population)]
    fitness = score_all([space.decode(x) for x in pop])
    for _ in range(generations):
        order = np.argsort(fitness)[::-1]
        new_pop = [pop[i].copy() for i in order[:elite]]
        while len(new_pop) < population:
            a, b = pick(pop, fitness), pick(pop, fitness)
            child = np.where(rng.random(space.dim) < 0.5, a, b) if rng.random() < crossover_prob else a.copy()
            mut = rng.random(space.dim) < mutation_prob
            child = np.clip(child + mut * rng.normal(0, mutation_scale, space.dim), 0, 1)
            new_pop.append(child)
        pop = new_pop
        fitness = score_all([space.decode(x) for x in pop])
    return trials


def run_search(method: str, space: ParamSpace, evaluate: EvalFn, cfg: Any, seed: int, n_trials: int | None = None,
               batch: int = 1) -> list[Trial]:
    """Dispatch to the configured search method (``cfg`` is Settings.optimizer)."""
    rng = np.random.default_rng(seed)
    n = n_trials or cfg.n_trials
    if method == "grid":
        return grid_search(space, evaluate, cfg.grid.points_per_dim, min(cfg.grid.max_points, n), batch)
    if method == "random":
        return random_search(space, evaluate, n, rng, batch)
    if method == "bayesian":
        b = cfg.bayesian
        return bayesian_search(space, evaluate, n, rng, b.n_initial, b.n_candidates, b.xi, b.restarts, batch)
    if method == "genetic":
        g = cfg.genetic
        pop, gens = g.population, g.generations
        if n_trials:  # respect an explicit trial budget: ~pop * (gens + 1) evaluations
            pop = max(4, min(g.population, n // 2))
            gens = max(1, n // pop - 1)
        return genetic_search(space, evaluate, rng, pop, gens, g.crossover_prob, g.mutation_prob,
                              g.mutation_scale, g.tournament, min(g.elite, pop - 1), batch)
    raise ValueError(f"unknown optimisation method {method}")
