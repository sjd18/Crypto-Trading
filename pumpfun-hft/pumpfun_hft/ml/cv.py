"""Purged, embargoed, group-aware forward-chaining cross-validation.

For fold ``k`` the test block is the k-th chronological slice; the training set contains only
samples whose *label window ends before the test block starts* (purging) and that precede it by
at least ``embargo_ms``; samples from tokens that appear in the test block are also removed, so
no token contributes to both sides. Training data therefore never contains information from the
test period or later — the model can never "predict using future data".
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedForwardSplit:
    """Forward-chaining CV with label-window purging, embargo and group exclusion."""

    def __init__(self, n_splits: int, embargo_ms: int, min_train: int = 30) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo_ms = embargo_ms
        self.min_train = min_train

    def split(self, times: np.ndarray, label_end: np.ndarray, groups: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        order = np.argsort(times, kind="stable")
        blocks = np.array_split(order, self.n_splits)
        for k in range(1, self.n_splits):
            test = blocks[k]
            if test.size == 0:
                continue
            t0 = times[test].min()
            cand = np.concatenate(blocks[:k])
            mask = (label_end[cand] < t0) & (times[cand] < t0 - self.embargo_ms)
            if groups is not None:
                test_groups = set(groups[test].tolist())
                mask &= np.array([g not in test_groups for g in groups[cand]])
            train = cand[mask]
            if train.size >= self.min_train:
                yield np.sort(train), np.sort(test)
