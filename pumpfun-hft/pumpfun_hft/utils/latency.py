"""Streaming latency measurement.

:class:`LatencyTracker` keeps, per named channel, a fixed-size ring buffer of the most recent
samples (for percentiles), an EWMA, lifetime count/max, and the number of samples that
breached an optional budget (e.g. ``live.event_to_dispatch`` must stay under 100 ms,
``exec.quote_to_submit`` under 250 ms).

It is used by the HTTP/WS clients, the live collector, the execution engine and the
circuit breakers, and is cheap enough to call on every event (O(1) per sample).

Example::

    lat = LatencyTracker()
    lat.set_budget("exec.quote_to_submit", 250)
    with lat.measure("exec.quote_to_submit"):
        ...
    lat.snapshot()["exec.quote_to_submit"]["p99"]
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from pumpfun_hft.utils.logging import get_logger

_log = get_logger("latency")


@dataclass(slots=True)
class _Channel:
    buf: np.ndarray
    idx: int = 0
    filled: int = 0
    count: int = 0
    ewma: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    breaches: int = 0
    budget_ms: float | None = None
    last_breach_log: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


class LatencyTracker:
    """Per-channel latency statistics with ring-buffer percentiles and budgets."""

    def __init__(self, window: int = 2048, ewma_alpha: float = 0.1) -> None:
        self.window = int(window)
        self.alpha = float(ewma_alpha)
        self._ch: dict[str, _Channel] = {}

    def _get(self, name: str) -> _Channel:
        ch = self._ch.get(name)
        if ch is None:
            ch = _Channel(buf=np.zeros(self.window, dtype=np.float64))
            self._ch[name] = ch
        return ch

    def set_budget(self, name: str, budget_ms: float) -> None:
        """Declare a latency budget; samples above it are counted as breaches."""
        self._get(name).budget_ms = float(budget_ms)

    def record(self, name: str, ms: float) -> None:
        """Record one latency sample in milliseconds."""
        ch = self._get(name)
        ch.buf[ch.idx] = ms
        ch.idx = (ch.idx + 1) % self.window
        ch.filled = min(ch.filled + 1, self.window)
        ch.count += 1
        ch.ewma = ms if ch.count == 1 else (1 - self.alpha) * ch.ewma + self.alpha * ms
        ch.last_ms = ms
        if ms > ch.max_ms:
            ch.max_ms = ms
        if ch.budget_ms is not None and ms > ch.budget_ms:
            ch.breaches += 1
            t = time.monotonic()
            if t - ch.last_breach_log >= 1.0:  # at most one log line per channel per second
                ch.last_breach_log = t
                _log.warning("latency budget exceeded", extra={"data": {"channel": name, "ms": round(ms, 3),
                                                                        "budget_ms": ch.budget_ms, "breaches": ch.breaches}})

    def record_ns(self, name: str, start_ns: int, end_ns: int | None = None) -> float:
        """Record ``end_ns - start_ns`` (perf_counter nanoseconds); returns milliseconds."""
        end = time.perf_counter_ns() if end_ns is None else end_ns
        ms = (end - start_ns) / 1e6
        self.record(name, ms)
        return ms

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Context manager recording the elapsed wall time of the block."""
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_ns(name, t0)

    def samples(self, name: str) -> np.ndarray:
        """Return the retained samples for a channel (most recent ``window``)."""
        ch = self._ch.get(name)
        if ch is None or ch.filled == 0:
            return np.empty(0)
        return ch.buf[: ch.filled].copy() if ch.filled < self.window else ch.buf.copy()

    def percentile(self, name: str, q: float) -> float:
        """Percentile ``q`` (0-100) over retained samples; NaN when empty."""
        s = self.samples(name)
        return float(np.percentile(s, q)) if s.size else float("nan")

    def count(self, name: str) -> int:
        ch = self._ch.get(name)
        return ch.count if ch else 0

    def ewma(self, name: str) -> float:
        ch = self._ch.get(name)
        return ch.ewma if ch else float("nan")

    def names(self) -> list[str]:
        return sorted(self._ch)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Summary statistics for every channel."""
        out: dict[str, dict[str, float]] = {}
        for name, ch in self._ch.items():
            s = self.samples(name)
            if s.size == 0:
                continue
            p50, p90, p99 = np.percentile(s, [50, 90, 99])
            out[name] = {
                "count": float(ch.count),
                "mean": float(s.mean()),
                "p50": float(p50),
                "p90": float(p90),
                "p99": float(p99),
                "max": float(ch.max_ms),
                "ewma": float(ch.ewma),
                "last": float(ch.last_ms),
                "budget_ms": float(ch.budget_ms) if ch.budget_ms is not None else float("nan"),
                "breaches": float(ch.breaches),
            }
        return out
