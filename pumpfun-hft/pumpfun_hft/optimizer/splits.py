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

import numpy as np

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


def has_recording_gaps(ts: np.ndarray, min_gap_ms: int = 3_600_000, max_uncovered: float = 0.1) -> bool:
    """True when gaps longer than ``min_gap_ms`` (stretches nothing was recorded) add up to more than
    ``max_uncovered`` of the span: splitting such data by clock time can leave a split almost empty."""
    if ts.size < 2:
        return False
    t = np.sort(ts)
    gaps = np.diff(t)
    span = float(t[-1] - t[0])
    return span > 0 and float(gaps[gaps > min_gap_ms].sum()) / span > max_uncovered


def make_splits(start_ms: int, end_ms: int, cfg: Any, ts: np.ndarray | None = None) -> DataSplits:
    """Chronological train/validation/test/live-sim splits with embargo gaps.

    With ``ts`` (event timestamps) the boundaries fall at the event-count quantiles instead of the
    clock-time fractions, so each split holds its share of the *recorded* market even when the
    recording has long gaps (the order, the embargo and the sealing are unchanged)."""
    if ts is not None and ts.size:
        return _event_splits(start_ms, end_ms, cfg, np.sort(ts))
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


def _event_splits(start_ms: int, end_ms: int, cfg: Any, ts: np.ndarray) -> DataSplits:
    fr = np.cumsum([cfg.train, cfg.validation, cfg.test])
    cuts = [int(ts[min(ts.size - 1, int(f * ts.size))]) for f in fr]
    names = ["train", "validation", "test", "live_sim"]
    edges = [start_ms, *cuts, end_ms]
    # a dense recording can put validation / test / live-sim within minutes of each other: the embargo
    # (no-trade gap after each boundary) is capped at a quarter of the shortest of those splits, so it
    # never swallows a split whole
    shortest = min(edges[i + 1] - edges[i] for i in range(1, 4))
    embargo = min(int(cfg.embargo_s * 1000), max(0, shortest // 4))
    bounds = [TimeSplit(names[i], edges[i] + (embargo if i else 0), max(edges[i] + (embargo if i else 0), edges[i + 1]))
              for i in range(4)]
    return DataSplits(bounds[0], bounds[1], SealedSplit(bounds[2]), SealedSplit(bounds[3]), embargo)


def split_event_counts(ts: np.ndarray, splits: DataSplits) -> dict[str, int]:
    """Events inside each split (test / live-sim counted without unsealing them)."""
    t = np.sort(ts)
    out = {}
    for sp in (splits.train, splits.validation, splits.test._split, splits.live_sim._split):  # bounds only: not an evaluation
        out[sp.name] = int(np.searchsorted(t, sp.end_ms, "left") - np.searchsorted(t, sp.start_ms, "left"))
    return out
