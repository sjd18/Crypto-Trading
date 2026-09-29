"""ML signal: trade the probability from a model trained with ``train-model``.

Loads a saved bundle (``train-model --target fwd_up``, or ``migrate``) and buys when the model's
probability of the positive outcome is at least ``min_prob``.

Where it scores
    The model learned from snapshots taken ``rug_model.snapshot_delays_s`` after each token's
    creation (10 / 30 / 60 / 120 s by default). It is only valid on that distribution, so the
    strategy scores each token once per snapshot delay: at the first trade at or after
    ``created + delay`` (if several delays have passed without a trade, it scores once for the
    latest). The feature row comes from ``ml.dataset.model_features``, the same function that
    built the training rows. Residual difference: training snapshots are taken just *before* the
    first event at/after the snapshot time, live scoring just *after* it.

Exits
    ``fwd_up`` means "price is at least +threshold after the horizon", so by default the position
    is held for at most the label horizon (``hold_s: 0`` = the model's ``fwd_return_horizon_s``).
    Confidence is ``100 * p``; note ``min_prob`` below ``sizing.min_confidence / 100`` has no effect
    beyond it. Stop loss, take-profit ladder and trailing stop come from the ``position`` config unless
    overridden here. The rug-avoidance overlay still applies.

Look-ahead guard
    The model refuses to trade any event before its training cut-off (the last label window of
    its training data): backtest it only on data after that date (``--start``), which
    ``train-model`` prints.

Cost gate
    ``expected_return = p * up_return - (1 - p) * down_return`` must beat the round-trip cost
    multiple (``strategy.cost_gate_multiple``). ``up_return: 0`` means the model's
    ``fwd_return_threshold``.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from pumpfun_hft.core.types import Action, Signal, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register
from pumpfun_hft.utils.logging import get_logger

log = get_logger("system")

#: targets whose positive class is a good outcome for a long position
TRADABLE_TARGETS = ("fwd_up", "migrate")


def resolve_model_path(model_path: str, models_dir: Path) -> Path:
    """``latest`` -> newest ``*fwd_up*.joblib`` in ``models_dir``; relative paths are under ``models_dir``."""
    if model_path in ("", "latest"):
        found = sorted(models_dir.glob("*-fwd_up-*.joblib"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"ml_signal: no fwd_up model in {models_dir}. Train one first: "
                                    "train-model --model lightgbm --target fwd_up")
        return found[-1]
    p = Path(model_path)
    if not p.is_absolute() and not p.exists():
        p = models_dir / p
    if not p.exists():
        raise FileNotFoundError(f"ml_signal: model file not found: {p}")
    return p


@register
class MlSignal(Strategy):
    name = "ml_signal"

    class Params(StrategyParams):
        model_path: str = "latest"     # "latest" or a path (relative paths are under paths.models_dir)
        min_prob: float = 0.6          # buy when P(positive outcome) >= this
        max_rug_prob: float = 0.5
        max_progress_pct: float = 90.0
        hold_s: float = 0.0            # 0 = the model's label horizon
        stop_loss_pct: float = 0.0     # 0 = position.stop_loss_pct
        take_profit_pct: float = 0.0   # 0 = the position.take_profit ladder
        up_return: float = 0.0         # 0 = the model's fwd_return_threshold
        down_return: float = 0.1

    def __init__(self, params: dict[str, Any] | StrategyParams) -> None:
        super().__init__(params)
        self.model: Any = None
        self.features: list[str] = []
        self.train_end_ms = 0
        self.delays_s: list[float] = []
        self.horizon_s = 300.0
        self.threshold = 0.2
        self.path: Path | None = None
        self._next: dict[str, tuple[int, int]] = {}  # mint -> (index of next delay to score, created_ms)
        self._warned_cutoff = False
        self.stats: Counter[str] = Counter()  # funnel: scored -> passed min_prob -> passed rug -> signals

    # ------------------------------------------------------------------ model loading
    def bind_settings(self, settings: Any) -> None:
        """Called by ``build_strategy``: load and validate the model bundle."""
        import joblib

        self.path = resolve_model_path(self.p.model_path, settings.paths.resolve("models_dir"))
        self.load_bundle(joblib.load(self.path), settings)

    def load_bundle(self, bundle: dict[str, Any], settings: Any) -> None:
        from pumpfun_hft.ml.dataset import MODEL_FEATURE_NAMES

        target = bundle.get("target")
        if target not in TRADABLE_TARGETS:
            raise ValueError(f"ml_signal trades models of {TRADABLE_TARGETS}; {self.path} predicts {target!r}"
                             + (" (a rug model belongs in rug_model.model_path)" if target == "rug" else ""))
        trained_on, active = bundle.get("dataset"), settings.datasets.active
        if active and trained_on and trained_on != active:  # e.g. a synthetic-market model on real data
            raise ValueError(f"ml_signal: {self.path} was trained on the {trained_on} data set; this run uses the {active} "
                             f"data set. Train one on it: {'hftr' if active == 'real' else 'hft'} train-model --target fwd_up")
        unknown = [f for f in bundle["features"] if f not in MODEL_FEATURE_NAMES]
        if unknown:  # trained by a different version of the feature code: refuse rather than feed zeros
            raise ValueError(f"ml_signal: model uses features this version does not compute: {unknown[:8]}; retrain it")
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.train_end_ms = int(bundle["train_end_ms"])
        self.delays_s = sorted(float(d) for d in bundle.get("snapshot_delays_s") or settings.rug_model.snapshot_delays_s)
        self.horizon_s = float(bundle.get("fwd_return_horizon_s") or settings.ml.fwd_return_horizon_s)
        if target == "migrate":
            self.horizon_s = float(bundle.get("label_horizon_s") or settings.rug_model.label_horizon_s)
        self.threshold = float(bundle.get("fwd_return_threshold") or settings.ml.fwd_return_threshold)

    @property
    def scored(self) -> int:
        return self.stats["scored"]

    @property
    def skipped_before_cutoff(self) -> int:
        return self.stats["before_cutoff"]

    def funnel(self) -> str:
        """One-line summary of where snapshots were filtered (printed by the CLI after a run)."""
        st = self.stats
        return (f"scored {st['scored']:,} · below min_prob {st['below_min_prob']:,} · rug-blocked {st['rug_blocked']:,} · "
                f"signals {st['signals']:,} · skipped before cut-off {st['before_cutoff']:,}")

    def predict(self, row: dict[str, float]) -> float:
        import numpy as np

        x = np.array([[row.get(f, 0.0) for f in self.features]], dtype=np.float64)
        x = np.where(np.isfinite(x), x, 0.0)  # training filled nulls / NaN / inf with 0
        return float(self.model.predict_proba(x)[0, 1])

    # ------------------------------------------------------------------ decisions
    def _due(self, ctx: StrategyContext) -> bool:
        """True on the first trade at or after each snapshot delay (see module docstring)."""
        mint, age = ctx.token.mint, ctx.age_s
        idx, created = self._next.get(mint, (0, ctx.token.created_ms or ctx.now_ms))
        if idx >= len(self.delays_s) or age < self.delays_s[idx]:
            return False
        while idx < len(self.delays_s) and age >= self.delays_s[idx]:
            idx += 1
        self._next[mint] = (idx, created)  # idx == len(delays) marks the token done: never re-scored
        if len(self._next) > 50_000:  # forget tokens far past the last delay
            horizon = ctx.now_ms - int((self.delays_s[-1] + 600.0) * 1000)
            self._next = {m: v for m, v in self._next.items() if v[1] >= horizon}
        return True

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        p = self.p
        if self.model is None or ctx.has_position or not ctx.is_trade or ctx.token.created_ms is None:
            return hold()
        if ctx.age_s > self.delays_s[-1] + 60.0 and ctx.token.mint not in self._next:
            return hold()
        if not self._due(ctx):
            return hold()
        if ctx.now_ms < self.train_end_ms:
            self.stats["before_cutoff"] += 1
            if not self._warned_cutoff:
                self._warned_cutoff = True
                log.warning("ml_signal: events before the model's training cut-off are not traded",
                            extra={"data": {"train_end_ms": self.train_end_ms}})
            return hold("before model training cut-off")
        if ctx.f.progress_pct > p.max_progress_pct:
            self.stats["progress_too_high"] += 1
            return hold()
        from pumpfun_hft.ml.dataset import model_features

        prob = self.predict(model_features(ctx.f, ctx.token, ctx.creator_score))
        self.stats["scored"] += 1
        if prob < p.min_prob:
            self.stats["below_min_prob"] += 1
            return hold()
        rug = ctx.rug_prob
        if rug > p.max_rug_prob:
            self.stats["rug_blocked"] += 1
            return hold()
        self.stats["signals"] += 1
        up = p.up_return or self.threshold
        exp_ret = prob * up - (1.0 - prob) * p.down_return
        overrides: dict[str, float] = {"max_hold_s": p.hold_s or self.horizon_s}
        if p.stop_loss_pct:
            overrides["stop_loss_pct"] = p.stop_loss_pct
        if p.take_profit_pct:
            overrides["take_profit_pct"] = p.take_profit_pct
        conf = 100.0 * prob  # the rug risk has its own gate above; keep min_prob and sizing.min_confidence independent
        return Signal(Action.BUY, conf, f"ml p={prob:.2f} age={ctx.age_s:.0f}s", self.name,
                      expected_return=exp_ret if math.isfinite(exp_ret) else None, exit_overrides=overrides)
