"""ML signal: trade the probability from a model trained with ``train-model``.

Loads a saved bundle (``train-model --target fwd_up``, or ``migrate``) and buys the tokens the model
scores highest.

Entry threshold
    A model's probabilities sit near its base rate: if 8 % of snapshots go up +20 % in 5 minutes, a
    good model's top scores are ~0.3, not 0.6, so a fixed probability cut-off usually means no trades.
    By default the threshold is taken from the model's own **out-of-fold** scores (each produced by a
    CV model that never saw that row): ``top_frac: 0.1`` buys a snapshot when its score is in the top
    10 % of those. ``min_prob > 0`` sets a fixed probability instead.

Confidence and cost gate
    Confidence is the score's percentile among the out-of-fold scores (the top 10 % -> 90-100), so it
    passes ``sizing.min_confidence`` whenever ``top_frac`` is below ``1 - min_confidence / 100``.
    The expected return used by the cost gate is what actually followed the out-of-fold snapshots
    above the threshold: their mean return over the label horizon (winsorised at the 95th percentile
    so a few extreme pumps do not dominate). If that is not above the round-trip cost multiple
    (``strategy.cost_gate_multiple``), the model has no edge after costs and the gate blocks every
    entry - as it should. Models trained before these statistics were stored fall back to
    ``p * up_return - (1 - p) * down_return`` and ``100 * p``.

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
    Stop loss, take-profit ladder and trailing stop come from the ``position`` config unless
    overridden here. The rug-avoidance overlay still applies; ``max_rug_prob`` (default 1 = off,
    since the model already sees the rug features) adds the rug scorer's veto on top.

Look-ahead guard
    The model refuses to trade any event before its training cut-off (the last label window of
    its training data): backtest it only on data after that date (``--start``), which
    ``train-model`` prints.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import model_validator

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
        top_frac: float = 0.1          # buy the top 10 % of scores (threshold from the model's out-of-fold scores)
        min_prob: float = 0.0          # > 0: a fixed probability threshold instead of top_frac
        max_rug_prob: float = 1.0      # 1 = off (the model already sees the rug features)
        max_progress_pct: float = 90.0
        hold_s: float = 0.0            # 0 = the model's label horizon
        stop_loss_pct: float = 0.0     # 0 = position.stop_loss_pct
        take_profit_pct: float = 0.0   # 0 = the position.take_profit ladder
        up_return: float = 0.0         # fallback cost gate (old models): 0 = the model's fwd_return_threshold
        down_return: float = 0.1

        @model_validator(mode="after")
        def _check(self) -> MlSignal.Params:
            if not 0.0 < self.top_frac <= 1.0:
                raise ValueError("ml_signal.top_frac must be in (0, 1]")
            if not 0.0 <= self.min_prob < 1.0:
                raise ValueError("ml_signal.min_prob must be in [0, 1) (0 = use top_frac)")
            return self

    def __init__(self, params: dict[str, Any] | StrategyParams) -> None:
        super().__init__(params)
        self.model: Any = None
        self.features: list[str] = []
        self.train_end_ms = 0
        self.delays_s: list[float] = []
        self.horizon_s = 300.0
        self.threshold = 0.2           # the label's return threshold (fwd_up: +20 %)
        self.min_score = 0.6           # entry threshold on the model's probability (set in load_bundle)
        self.threshold_source = ""
        self.oof_sorted: Any = None    # sorted out-of-fold scores (confidence = percentile among them)
        self.oof_n = 0                 # out-of-fold snapshots at or above the threshold ...
        self.oof_hit_rate: float | None = None   # ... how often their label was positive ...
        self.oof_base_rate: float | None = None
        self.expected_return: float | None = None  # ... and their mean return over the horizon
        self.min_confidence = 0.0
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
        self.min_confidence = float(settings.sizing.min_confidence)
        self._set_threshold(bundle, target)

    def _set_threshold(self, bundle: dict[str, Any], target: str) -> None:
        """Entry threshold, confidence scale and expected return from the out-of-fold scores (module docstring)."""
        import numpy as np

        p = self.p
        oof = bundle.get("oof_probs")
        probs = np.asarray(oof, dtype=np.float64) if oof is not None else np.empty(0)
        ok = np.isfinite(probs)
        self.oof_sorted = np.sort(probs[ok]) if ok.sum() >= 20 else None
        if p.min_prob > 0:
            self.min_score, self.threshold_source = p.min_prob, f"min_prob {p.min_prob:g}"
        elif self.oof_sorted is not None:
            self.min_score = float(np.quantile(self.oof_sorted, 1.0 - p.top_frac))
            self.threshold_source = f"top {p.top_frac:.0%} of {len(self.oof_sorted):,} out-of-sample scores"
        else:
            self.min_score = 0.6
            self.threshold_source = "fixed 0.6: this model has no out-of-sample scores, retrain it for an automatic threshold"
            log.warning("ml_signal: model has no out-of-fold scores; using min_prob 0.6 (retrain for top_frac)")
        self.oof_n, self.oof_hit_rate, self.oof_base_rate, self.expected_return = 0, None, None, None
        if not len(probs):
            return
        sel = ok & (probs >= self.min_score)
        self.oof_n = int(sel.sum())
        tgt = bundle.get("oof_target")
        if tgt is not None:
            y = np.asarray(tgt, dtype=np.float64)
            self.oof_base_rate = float(y[ok].mean()) if ok.any() else None
            self.oof_hit_rate = float(y[sel].mean()) if self.oof_n else None
        ret = bundle.get("oof_fwd_return")
        if target == "fwd_up" and ret is not None and self.oof_n >= 5:
            r = np.asarray(ret, dtype=np.float64)
            fin = np.isfinite(r)
            if (sel & fin).sum() >= 5:
                cap = float(np.quantile(r[fin], 0.95))
                self.expected_return = float(np.clip(r[sel & fin], -1.0, cap).mean())

    @staticmethod
    def threshold_table(bundle: dict[str, Any], fracs: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)) -> list[dict[str, float]]:
        """Out-of-fold statistics for a few ``top_frac`` values: threshold, snapshots, hit rate, mean return."""
        import numpy as np

        if bundle.get("oof_probs") is None:
            return []
        probs = np.asarray(bundle["oof_probs"], dtype=np.float64)
        ok = np.isfinite(probs)
        if ok.sum() < 20:
            return []
        y = np.asarray(bundle["oof_target"], dtype=np.float64) if bundle.get("oof_target") is not None else None
        r = np.asarray(bundle["oof_fwd_return"], dtype=np.float64) if bundle.get("oof_fwd_return") is not None else None
        cap = float(np.quantile(r[np.isfinite(r)], 0.95)) if r is not None and np.isfinite(r).any() else None
        rows = []
        for f in fracs:
            thr = float(np.quantile(probs[ok], 1.0 - f))
            sel = ok & (probs >= thr)
            row = {"top_frac": f, "threshold": thr, "n": float(sel.sum()), "hit_rate": float("nan"), "mean_return": float("nan")}
            if y is not None and sel.any():
                row["hit_rate"] = float(y[sel].mean())
            if r is not None and cap is not None and (sel & np.isfinite(r)).any():
                row["mean_return"] = float(np.clip(r[sel & np.isfinite(r)], -1.0, cap).mean())
            rows.append(row)
        return rows

    def confidence(self, prob: float) -> float:
        """0-100: the score's percentile among the out-of-fold scores (``100 * p`` without them)."""
        import numpy as np

        if self.oof_sorted is None:
            return 100.0 * prob
        return 100.0 * float(np.searchsorted(self.oof_sorted, prob, side="right")) / len(self.oof_sorted)

    def describe(self) -> list[str]:
        """How this model will trade (printed by the CLI before a run)."""
        h = f"{self.horizon_s / 60:g} min"
        lines = [f"entry: score >= {self.min_score:.3f} ({self.threshold_source})"]
        if self.oof_n:
            hit = f"hit rate {self.oof_hit_rate:.1%} vs base rate {self.oof_base_rate:.1%}" if self.oof_hit_rate is not None \
                and self.oof_base_rate is not None else ""
            ret = f", mean {h} return {self.expected_return:+.1%}" if self.expected_return is not None else ""
            lines.append(f"out of sample, {self.oof_n:,} snapshots scored that high: {hit}{ret}")
        if self.expected_return is not None and self.expected_return <= 0:
            lines.append("warning: out of sample, the snapshots above this threshold lost money on average; the cost "
                         "gate will block entries. Try a smaller top_frac, more data, or accept that the model has no edge.")
        if self.confidence(self.min_score) < self.min_confidence:
            lines.append(f"warning: confidence is the score's percentile, and scores below the {self.min_confidence:g}th "
                         f"percentile fall under sizing.min_confidence and are skipped: use top_frac <= "
                         f"{1 - self.min_confidence / 100:.2f}")
        return lines

    @property
    def scored(self) -> int:
        return self.stats["scored"]

    @property
    def skipped_before_cutoff(self) -> int:
        return self.stats["before_cutoff"]

    def funnel(self, signal_records: Any = None) -> str:
        """Where snapshots were filtered (printed by the CLI after a run). With the runtime's signal records it
        also says what happened to the signals: submitted, or skipped by the cost gate, confidence, veto, risk..."""
        st = self.stats
        out = (f"scored {st['scored']:,} · below threshold {st['below_min_prob']:,} · rug-blocked {st['rug_blocked']:,} · "
               f"signals {st['signals']:,} · skipped before cut-off {st['before_cutoff']:,}")
        if signal_records is not None:
            outcomes = Counter(r["outcome"] for r in signal_records if r.get("strategy") == self.name and r.get("action") == "BUY")
            if outcomes:
                out += "\n  signals -> " + " · ".join(f"{k} {v:,}" for k, v in outcomes.most_common())
        return out

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
        if not (ctx.token.price > 0 and math.isfinite(ctx.token.price)):  # the training rows never had this
            self.stats["no_price"] += 1
            return hold()
        if ctx.f.progress_pct > p.max_progress_pct:
            self.stats["progress_too_high"] += 1
            return hold()
        from pumpfun_hft.ml.dataset import model_features

        prob = self.predict(model_features(ctx.f, ctx.token, ctx.creator_score))
        self.stats["scored"] += 1
        if prob < self.min_score:
            self.stats["below_min_prob"] += 1
            return hold()
        if p.max_rug_prob < 1.0 and ctx.rug_prob > p.max_rug_prob:
            self.stats["rug_blocked"] += 1
            return hold()
        self.stats["signals"] += 1
        if self.expected_return is not None:
            exp_ret = self.expected_return
        else:
            up = p.up_return or self.threshold
            exp_ret = prob * up - (1.0 - prob) * p.down_return
        overrides: dict[str, float] = {"max_hold_s": p.hold_s or self.horizon_s}
        if p.stop_loss_pct:
            overrides["stop_loss_pct"] = p.stop_loss_pct
        if p.take_profit_pct:
            overrides["take_profit_pct"] = p.take_profit_pct
        conf = self.confidence(prob)
        return Signal(Action.BUY, conf, f"ml p={prob:.2f} age={ctx.age_s:.0f}s", self.name,
                      expected_return=exp_ret if math.isfinite(exp_ret) else None, exit_overrides=overrides)
