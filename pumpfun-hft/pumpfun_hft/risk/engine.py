"""Risk engine: hard pre-trade limits + circuit breakers.

Entry checks (``check_entry``), in order: circuit breakers, daily loss (equity vs UTC-day
start), hourly loss (equity vs 60 minutes ago), max open positions, per-token order rate, then
size clipping to the headroom under max exposure, per-token, per-creator and per-sector limits
(pending orders count toward exposure). Exits are never blocked (``check_exit`` always passes)
so risk can always be reduced.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from pumpfun_hft.core.types import LAMPORTS_PER_SOL
from pumpfun_hft.risk.breakers import CircuitBreakers
from pumpfun_hft.risk.portfolio import Portfolio


@dataclass(slots=True)
class RiskDecision:
    ok: bool
    lamports: int
    reason: str = ""


class RiskEngine:
    """Pre-trade limits and breakers bound to a portfolio.

    Example::

        risk = RiskEngine(settings.risk, portfolio, min_order_lamports)
        d = risk.check_entry(mint, creator, sector, lamports, now_ms, pending_lamports)
    """

    def __init__(self, cfg: Any, portfolio: Portfolio, min_order_lamports: int) -> None:
        self.cfg = cfg
        self.lim = cfg.limits
        self.pf = portfolio
        self.min_order = min_order_lamports
        self.breakers = CircuitBreakers(cfg.breakers)
        self._orders: dict[str, deque[int]] = defaultdict(deque)
        self._day: int | None = None
        self._day_start_equity = portfolio.initial
        self.rejections: dict[str, int] = defaultdict(int)

    def _reject(self, reason: str) -> RiskDecision:
        key = reason.split(":")[0]
        self.rejections[key] += 1
        return RiskDecision(False, 0, reason)

    def on_equity(self, ts: int, equity_lamports: int) -> None:
        day = ts // 86_400_000
        if self._day != day:
            self._day = day
            self._day_start_equity = equity_lamports
        self.breakers.on_equity(equity_lamports, self.pf.peak_equity, ts)

    def check_entry(self, mint: str, creator: str | None, sector: str, lamports: int, now_ms: int,
                    pending_lamports: int = 0, pending_by_creator: int = 0, pending_by_sector: int = 0) -> RiskDecision:
        halt = self.breakers.halted(now_ms)
        if halt:
            return self._reject(f"breaker: {halt}")
        lim = self.lim
        eq = self.pf.equity_lamports()
        if (self._day_start_equity - eq) / LAMPORTS_PER_SOL >= lim.daily_loss_sol:
            return self._reject("daily loss limit")
        hour_ago = self.pf.equity_at_or_before(now_ms - 3_600_000)
        if hour_ago is not None and (hour_ago - eq) / LAMPORTS_PER_SOL >= lim.hourly_loss_sol:
            return self._reject("hourly loss limit")
        if self.pf.n_positions >= lim.max_open_positions:
            return self._reject("max open positions")
        q = self._orders[mint]
        while q and q[0] < now_ms - 60_000:
            q.popleft()
        if len(q) >= lim.max_orders_per_token_per_min:
            return self._reject("token order rate")
        headroom = [
            lamports,
            int(lim.max_exposure_sol * LAMPORTS_PER_SOL) - self.pf.cost_exposure_lamports() - pending_lamports,
            int(lim.max_position_per_token_sol * LAMPORTS_PER_SOL) - self.pf.exposure_by("mint", mint),
            int(lim.max_creator_exposure_sol * LAMPORTS_PER_SOL) - self.pf.exposure_by("creator", creator) - pending_by_creator,
            int(lim.max_sector_exposure_sol * LAMPORTS_PER_SOL) - self.pf.exposure_by("sector", sector) - pending_by_sector,
        ]
        size = min(headroom)
        if size < self.min_order:
            names = ["size", "max exposure", "token limit", "creator limit", "sector limit"]
            return self._reject(f"limit: {names[headroom.index(size)]}")
        q.append(now_ms)
        return RiskDecision(True, size, "")

    def check_exit(self, mint: str, now_ms: int) -> RiskDecision:
        self._orders[mint].append(now_ms)
        return RiskDecision(True, 0, "")

    def on_fill(self, filled: bool, failed: bool, slippage_bps: float, now_ms: int) -> None:
        self.breakers.on_fill(filled, failed, slippage_bps, now_ms)

    def status(self, now_ms: int) -> dict[str, Any]:
        return {"breakers": self.breakers.status(now_ms), "rejections": dict(self.rejections),
                "day_start_equity_sol": self._day_start_equity / LAMPORTS_PER_SOL}
