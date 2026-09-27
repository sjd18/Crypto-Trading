"""Rate limiting primitives.

* :class:`TokenBucket` — classic bucket with ``rate`` tokens/s and capacity ``burst``. Works
  in real time (``await acquire()``) or in simulated time (``try_acquire(now_s)`` /
  ``next_available(now_s)``), so the backtester models the same limits the live system obeys.
* :class:`AdaptiveRateLimiter` — AIMD wrapper: multiplicative decrease on HTTP 429 (or a
  ``Retry-After`` hint), additive recovery on success, bounded to ``[min_rps, max_rps]``.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Token bucket usable with a real or simulated clock."""

    def __init__(self, rate: float, burst: int, now_s: float | None = None) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic() if now_s is None else float(now_s)
        self._lock = asyncio.Lock()

    def _refill(self, now_s: float) -> None:
        if now_s > self.updated:
            self.tokens = min(self.capacity, self.tokens + (now_s - self.updated) * self.rate)
            self.updated = now_s

    # -- simulated time -------------------------------------------------------
    def try_acquire(self, now_s: float, n: float = 1.0) -> bool:
        """Take ``n`` tokens at simulated time ``now_s`` if available."""
        self._refill(now_s)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def next_available(self, now_s: float, n: float = 1.0) -> float:
        """Earliest time at which ``n`` tokens will be available."""
        self._refill(now_s)
        if self.tokens >= n:
            return now_s
        return now_s + (n - self.tokens) / self.rate

    # -- real time ------------------------------------------------------------
    async def acquire(self, n: float = 1.0) -> float:
        """Wait until ``n`` tokens are available; returns seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                self._refill(now)
                if self.tokens >= n:
                    self.tokens -= n
                    return waited
                delay = (n - self.tokens) / self.rate
                waited += delay
                await asyncio.sleep(delay)

    def set_rate(self, rate: float) -> None:
        self._refill(time.monotonic())
        self.rate = max(1e-6, float(rate))


class AdaptiveRateLimiter:
    """AIMD rate limiter around a :class:`TokenBucket` (429-aware)."""

    def __init__(self, max_rps: float, burst: int, min_rps: float | None = None,
                 decrease: float = 0.5, increase_per_s: float = 0.5) -> None:
        self.max_rps = float(max_rps)
        self.min_rps = float(min_rps if min_rps is not None else max(0.2, max_rps / 20))
        self.decrease = decrease
        self.increase_per_s = increase_per_s
        self.bucket = TokenBucket(max_rps, burst)
        self.current = float(max_rps)
        self._last_adjust = time.monotonic()
        self.throttled_until = 0.0

    async def acquire(self) -> float:
        now = time.monotonic()
        waited = 0.0
        if now < self.throttled_until:
            waited += self.throttled_until - now
            await asyncio.sleep(self.throttled_until - now)
        return waited + await self.bucket.acquire()

    def on_success(self) -> None:
        now = time.monotonic()
        if self.current < self.max_rps:
            self.current = min(self.max_rps, self.current + (now - self._last_adjust) * self.increase_per_s)
            self.bucket.set_rate(self.current)
        self._last_adjust = now

    def on_throttle(self, retry_after_s: float | None = None) -> None:
        self.current = max(self.min_rps, self.current * self.decrease)
        self.bucket.set_rate(self.current)
        self._last_adjust = time.monotonic()
        if retry_after_s:
            self.throttled_until = max(self.throttled_until, time.monotonic() + retry_after_s)
