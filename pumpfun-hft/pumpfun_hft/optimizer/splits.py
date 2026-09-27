"""Chronological data splits with embargo, sealed hold-out sets and walk-forward folds.

* ``train`` / ``validation`` / ``test`` / ``live_sim`` are consecutive time ranges separated by an
  embargo gap (``embargo_s``) so that positions and label horizons cannot straddle a boundary.
* Only tokens *created inside* a split may be traded in it; state from earlier data (wallet
  intelligence, creator history) is used as warm-up — that is past information, never future.
* ``test`` and ``live_sim`` are :class:`SealedSplit` objects: reading them requires an explicit
  ``unseal(reason)`` which is logged and counted, making "never optimise on test data" auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pumpfun_hft.utils.logging import get_logger
from pumpfun_hft.utils.timeutil import ms_to_iso

log = get_logger("backtests")


class SealedSplitError(RuntimeError):
    """Raised when a sealed hold-out split is accessed without unsealing."""


@dataclass(frozen=True, slots=True)
class TimeSplit:
    name: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def describe(self) -> str:
        return f"{self.name}: {ms_to_iso(self.start_ms)} -> {ms_to_iso(self.end_ms)}"


@dataclass
class SealedSplit:
    """A hold-out split that must be explicitly unsealed (every access is audited)."""

    _split: TimeSplit
    audit: list[dict[str, Any]] = field(default_factory=list)

    @property
    def split(self) -> TimeSplit:
        raise SealedSplitError(f"{self._split.name} split is sealed; call unseal(reason) for the final evaluation")

    @property
    def name(self) -> str:
        return self._split.name

    def unseal(self, reason: str) -> TimeSplit:
        from pumpfun_hft.utils.timeutil import now_ms

        self.audit.append({"ts_ms": now_ms(), "reason": reason})
        log.warning("sealed split accessed", extra={"data": {"split": self._split.name, "reason": reason, "n_access": len(self.audit)}})
        return self._split


@dataclass
class DataSplits:
    train: TimeSplit
    validation: TimeSplit
    test: SealedSplit
    live_sim: SealedSplit
    embargo_ms: int

    def describe(self) -> list[str]:
        return [self.train.describe(), self.validation.describe(), f"{self.test.name}: <sealed>", f"{self.live_sim.name}: <sealed>"]


def make_splits(start_ms: int, end_ms: int, cfg: Any) -> DataSplits:
    """Chronological train/validation/test/live-sim splits with embargo gaps."""
    embargo = int(cfg.embargo_s * 1000)
    usable = end_ms - start_ms - 3 * embargo
    if usable <= 0:
        raise ValueError("data span too short for the configured embargo")
    fr = [cfg.train, cfg.validation, cfg.test, cfg.live_sim]
    names = ["train", "validation", "test", "live_sim"]
    bounds = []
    t = start_ms
    for i, f in enumerate(fr):
        dur = int(usable * f)
        bounds.append(TimeSplit(names[i], t, t + dur))
        t += dur + embargo
    return DataSplits(bounds[0], bounds[1], SealedSplit(bounds[2]), SealedSplit(bounds[3]), embargo)


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train: TimeSplit
    validation: TimeSplit
    test: TimeSplit


def walk_forward_folds(start_ms: int, end_ms: int, cfg: Any) -> list[Fold]:
    """Rolling (or anchored) walk-forward folds: [train][embargo][val][embargo][test] sliding forward.

    The test windows of consecutive folds are adjacent and non-overlapping, so stitched
    out-of-sample results cover a contiguous period.
    """
    embargo = int(cfg.embargo_s * 1000)
    n = cfg.n_folds
    tf, vf, sf = cfg.train_frac, cfg.val_frac, cfg.test_frac
    span = end_ms - start_ms
    # window length W such that n test windows fit after the first train+val block:
    # span = W*(tf+vf) + 2*embargo + n*W*sf
    w = (span - 2 * embargo) / (tf + vf + n * sf)
    if w <= 0:
        raise ValueError("data span too short for walk-forward configuration")
    folds = []
    for i in range(n):
        test_start = start_ms + int(w * (tf + vf)) + 2 * embargo + int(i * w * sf)
        test_end = test_start + int(w * sf)
        val_end = test_start - embargo
        val_start = val_end - int(w * vf)
        train_end = val_start - embargo
        train_start = start_ms if cfg.anchored else max(start_ms, train_end - int(w * tf))
        folds.append(Fold(i, TimeSplit(f"train{i}", train_start, train_end), TimeSplit(f"val{i}", val_start, val_end),
                          TimeSplit(f"test{i}", test_start, min(test_end, end_ms))))
    return folds
