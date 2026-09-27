"""Parameter spaces: sampling, grids and [0, 1]^d encoding for model-based optimisers."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from pumpfun_hft.core.config import ParamSpec


class ParamSpace:
    """A product space of float / int / log-float / categorical parameters.

    Example::

        space = ParamSpace(settings.optimizer.spaces["momentum_ignition"])
        p = space.sample(np.random.default_rng(0)); x = space.encode(p); space.decode(x) == p
    """

    def __init__(self, specs: dict[str, ParamSpec]) -> None:
        if not specs:
            raise ValueError("empty parameter space")
        self.specs = dict(specs)
        self.names = list(specs)
        self.dim = len(self.names)

    def _decode_one(self, spec: ParamSpec, u: float) -> Any:
        u = min(1.0, max(0.0, float(u)))
        if spec.type == "categorical":
            choices = spec.choices or []
            return choices[min(len(choices) - 1, int(round(u * (len(choices) - 1))))]
        lo, hi = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        if spec.type == "log_float":
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u * (hi - lo)
        if spec.step:
            v = lo + round((v - lo) / spec.step) * spec.step
        if spec.type == "int":
            return int(round(v))
        return float(min(hi, max(lo, v)))

    def _encode_one(self, spec: ParamSpec, v: Any) -> float:
        if spec.type == "categorical":
            choices = spec.choices or []
            return choices.index(v) / max(1, len(choices) - 1)
        lo, hi = float(spec.low), float(spec.high)  # type: ignore[arg-type]
        if hi == lo:
            return 0.0
        if spec.type == "log_float":
            return (math.log(float(v)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (float(v) - lo) / (hi - lo)

    def decode(self, x: np.ndarray) -> dict[str, Any]:
        return {n: self._decode_one(self.specs[n], x[i]) for i, n in enumerate(self.names)}

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        return np.array([self._encode_one(self.specs[n], params[n]) for n in self.names], dtype=float)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.decode(rng.random(self.dim))

    def grid(self, points_per_dim: int, max_points: int) -> list[dict[str, Any]]:
        axes = []
        for n in self.names:
            spec = self.specs[n]
            if spec.type == "categorical":
                axes.append(list(spec.choices or []))
            else:
                us = np.linspace(0.0, 1.0, points_per_dim)
                vals = []
                for u in us:
                    v = self._decode_one(spec, u)
                    if v not in vals:
                        vals.append(v)
                axes.append(vals)
        combos = list(itertools.product(*axes))
        if len(combos) > max_points:
            idx = np.linspace(0, len(combos) - 1, max_points).round().astype(int)
            combos = [combos[i] for i in sorted(set(idx.tolist()))]
        return [dict(zip(self.names, c, strict=True)) for c in combos]

    def key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(round(params[n], 10) if isinstance(params[n], float) else params[n] for n in self.names)
