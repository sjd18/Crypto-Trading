"""Exponential backoff with full jitter.

``delay = uniform(0, min(max_delay, base_delay * 2**attempt))`` (AWS "full jitter"), which
de-synchronises retry storms across workers. Callers decide which exceptions are retryable
and may supply a server-provided ``retry_after`` hint that overrides the computed delay.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Raise (or subclass) to signal that an operation may be retried.

    ``retry_after_s`` carries an optional server hint such as HTTP ``Retry-After``.
    """

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry schedule parameters."""

    max_attempts: int
    base_delay_s: float
    max_delay_s: float

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full-jitter delay before retry number ``attempt`` (0-based)."""
        r = rng or random
        cap = min(self.max_delay_s, self.base_delay_s * (2.0**attempt))
        return r.uniform(0.0, cap)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    rng: random.Random | None = None,
) -> T:
    """Await ``fn()`` retrying on ``retry_on`` exceptions according to ``policy``.

    The final exception is re-raised once ``policy.max_attempts`` attempts are exhausted.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except retry_on as exc:
            attempt += 1
            if attempt >= policy.max_attempts:
                raise
            hint = getattr(exc, "retry_after_s", None)
            delay = float(hint) if hint is not None else policy.delay(attempt - 1, rng)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
