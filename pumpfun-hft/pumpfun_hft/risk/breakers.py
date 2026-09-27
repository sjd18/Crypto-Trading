"""Circuit breakers: stop opening risk when the environment or the strategy misbehaves.

Breakers (each trips for ``cooldown_s``; exits are always allowed while tripped)
    rpc_latency   p90 of the last ``rpc_latency_window`` RPC latencies > ``rpc_latency_p90_ms``
    congestion    observed average slot time > ``congestion_slot_ms`` (Solana congestion)
    slippage      mean adverse slippage of the last ``slippage_window`` fills > ``slippage_bps_avg``
    failed_swaps  >= ``failed_swaps_max`` failed/dropped swaps within ``failed_swaps_window_s``
    drawdown      equity drawdown from peak > ``drawdown_pct`` (optionally flatten everything)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from pumpfun_hft.utils.logging import get_logger

log = get_logger("system")


@dataclass(slots=True)
class BreakerState:
    name: str
    tripped_until_ms: int = 0
    trips: int = 0
    last_reason: str = ""

    def active(self, now_ms: int) -> bool:
        return now_ms < self.tripped_until_ms


class CircuitBreakers:
    """Tracks metrics streams and trips named breakers."""

    NAMES = ("rpc_latency", "congestion", "slippage", "failed_swaps", "drawdown")

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.state = {n: BreakerState(n) for n in self.NAMES}
        self._lat: deque[float] = deque(maxlen=cfg.rpc_latency_window)
        self._slip: deque[float] = deque(maxlen=cfg.slippage_window)
        self._fails: deque[int] = deque()
        self.flatten_requested = False
        self.log: list[dict[str, Any]] = []
        self.verbose = True  # backtests set False: trips stay in ``log`` (and the run diagnostics) but are not logged

    def _trip(self, name: str, now_ms: int, reason: str) -> None:
        st = self.state[name]
        if not st.active(now_ms):
            st.trips += 1
            self.log.append({"ts_ms": now_ms, "breaker": name, "reason": reason})
            if self.verbose:
                log.warning("circuit breaker tripped", extra={"data": {"breaker": name, "reason": reason, "ts": now_ms}})
        st.tripped_until_ms = now_ms + int(self.cfg.cooldown_s * 1000)
        st.last_reason = reason

    def halted(self, now_ms: int) -> str | None:
        for st in self.state.values():
            if st.active(now_ms):
                return f"{st.name}: {st.last_reason}"
        return None

    def on_rpc_latency(self, ms: float, now_ms: int) -> None:
        self._lat.append(ms)
        if len(self._lat) >= max(5, self._lat.maxlen // 2):
            p90 = float(np.percentile(np.fromiter(self._lat, float), 90))
            if p90 > self.cfg.rpc_latency_p90_ms:
                self._trip("rpc_latency", now_ms, f"p90 {p90:.0f} ms")

    def on_slot_time(self, avg_slot_ms: float, now_ms: int) -> None:
        if avg_slot_ms > self.cfg.congestion_slot_ms:
            self._trip("congestion", now_ms, f"slot time {avg_slot_ms:.0f} ms")

    def on_fill(self, filled: bool, failed: bool, slippage_bps: float, now_ms: int) -> None:
        if filled:
            self._slip.append(slippage_bps)
            if len(self._slip) >= max(3, self._slip.maxlen // 2):
                avg = sum(self._slip) / len(self._slip)
                if avg > self.cfg.slippage_bps_avg:
                    self._trip("slippage", now_ms, f"avg slippage {avg:.0f} bps")
        if failed:
            self._fails.append(now_ms)
            cutoff = now_ms - int(self.cfg.failed_swaps_window_s * 1000)
            while self._fails and self._fails[0] < cutoff:
                self._fails.popleft()
            if len(self._fails) >= self.cfg.failed_swaps_max:
                self._trip("failed_swaps", now_ms, f"{len(self._fails)} failures")

    def on_equity(self, equity: float, peak: float, now_ms: int) -> None:
        if peak > 0:
            dd = 100.0 * (1.0 - equity / peak)
            if dd > self.cfg.drawdown_pct:
                self._trip("drawdown", now_ms, f"drawdown {dd:.1f}%")
                if self.cfg.flatten_on_drawdown:
                    self.flatten_requested = True

    def status(self, now_ms: int) -> dict[str, Any]:
        return {n: {"active": s.active(now_ms), "trips": s.trips, "reason": s.last_reason, "until_ms": s.tripped_until_ms}
                for n, s in self.state.items()}
