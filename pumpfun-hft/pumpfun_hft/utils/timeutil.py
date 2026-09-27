"""Time helpers. All timestamps in the platform are UTC epoch milliseconds (int)."""

from __future__ import annotations

import time
from datetime import UTC, datetime


def now_ms() -> int:
    """Wall-clock UTC time in epoch milliseconds."""
    return time.time_ns() // 1_000_000


def mono_ns() -> int:
    """Monotonic clock in nanoseconds (for latency measurement only)."""
    return time.perf_counter_ns()


def ms_to_dt(ms: int) -> datetime:
    """Convert epoch milliseconds to an aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def ms_to_iso(ms: int) -> str:
    """Epoch milliseconds to ISO-8601 string with millisecond precision."""
    return ms_to_dt(ms).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def day_str(ms: int) -> str:
    """Epoch milliseconds to the UTC partition day ``YYYY-MM-DD``."""
    return ms_to_dt(ms).strftime("%Y-%m-%d")


def parse_iso_ms(text: str) -> int:
    """Parse an ISO-8601 timestamp (``Z`` suffix allowed) into epoch milliseconds."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def hour_of_day(ms: int) -> float:
    """Fractional UTC hour of day in [0, 24)."""
    return ((ms // 1000) % 86_400) / 3600.0
