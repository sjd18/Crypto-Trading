"""Position sizing.

Methods (``sizing.method``)
    fixed         constant ``fixed_sol``
    fixed_risk    ``risk_per_trade_sol / stop_fraction`` (stop from the signal's exit overrides or config)
    kelly         ``kelly_fraction * f*`` of equity, ``f* = W - (1 - W) / R`` from *closed* trades so far
                  (point-in-time; falls back to fixed_risk until ``kelly_min_trades``), capped at ``kelly_cap_frac``
    volatility    ``vol_target_sol / max(token volatility, vol_floor)`` (ATR% or realised vol)
    confidence    ``fixed_sol * (confidence / 100) ** confidence_exponent``
    max_exposure  whatever headroom is left under ``risk.limits.max_exposure_sol``

Pipeline: base size -> explicit signal size override -> optional confidence scaling -> caps
(per-token limit, ``max_equity_frac`` of equity, exposure headroom, free cash minus reserve) ->
liquidity cap (largest size whose own price impact stays under ``max_impact_bps``) -> minimum
order size.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Signal


def kelly_fraction(returns: list[float]) -> float | None:
    """Classic Kelly f* = W - (1-W)/R from per-trade returns (None if undefined)."""
    wins = [r for r in returns if r > 0]
    losses = [-r for r in returns if r < 0]
    if not wins or not losses:
        return None
    w = len(wins) / len(returns)
    r = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    return w - (1.0 - w) / r


class Sizer:
    """Computes order budgets in lamports.

    Example::

        sizer = Sizer(settings.sizing, settings.risk.limits, settings.position, settings.backtest)
        lamports = sizer.size(signal, equity, cash, exposure, token_exposure, vol, closed_returns, impact_fn)
    """

    def __init__(self, cfg: Any, limits: Any, position: Any, backtest: Any) -> None:
        self.cfg = cfg
        self.limits = limits
        self.position = position
        self.min_order = int(backtest.min_order_sol * LAMPORTS_PER_SOL)
        self.reserve = int(backtest.cash_reserve_sol * LAMPORTS_PER_SOL)

    def stop_fraction(self, signal: Signal | None) -> float:
        stop = self.position.stop_loss_pct
        if signal is not None and signal.exit_overrides and "stop_loss_pct" in signal.exit_overrides:
            stop = signal.exit_overrides["stop_loss_pct"]
        return max(0.01, stop / 100.0)

    def base_size_sol(self, signal: Signal, equity_sol: float, exposure_sol: float, vol: float,
                      closed_returns: list[float]) -> float:
        c = self.cfg
        m = c.method
        if m == "fixed":
            return c.fixed_sol
        if m == "fixed_risk":
            return c.risk_per_trade_sol / self.stop_fraction(signal)
        if m == "kelly":
            if len(closed_returns) >= c.kelly_min_trades:
                f = kelly_fraction(closed_returns)
                if f is not None:
                    return max(0.0, min(c.kelly_fraction * f, c.kelly_cap_frac)) * equity_sol
            return c.risk_per_trade_sol / self.stop_fraction(signal)
        if m == "volatility":
            return c.vol_target_sol / max(vol, c.vol_floor)
        if m == "confidence":
            return c.fixed_sol * (signal.confidence / 100.0) ** c.confidence_exponent
        if m == "max_exposure":
            return max(0.0, self.limits.max_exposure_sol - exposure_sol)
        raise ValueError(f"unknown sizing method {m}")

    def size(self, signal: Signal, *, equity_lamports: int, cash_lamports: int, exposure_lamports: int,
             token_exposure_lamports: int, vol: float, closed_returns: list[float],
             impact_bps_fn: Callable[[int], float] | None = None) -> int:
        """Final order budget in lamports (0 when below the minimum order size)."""
        c = self.cfg
        equity_sol = equity_lamports / LAMPORTS_PER_SOL
        exposure_sol = exposure_lamports / LAMPORTS_PER_SOL
        size_sol = signal.size_sol if signal.size_sol is not None else self.base_size_sol(signal, equity_sol, exposure_sol, vol, closed_returns)
        if c.confidence_scaling and c.method != "confidence" and signal.size_sol is None:
            size_sol *= (signal.confidence / 100.0) ** c.confidence_exponent
        caps = [
            size_sol,
            self.limits.max_position_per_token_sol - token_exposure_lamports / LAMPORTS_PER_SOL,
            c.max_equity_frac * equity_sol,
            self.limits.max_exposure_sol - exposure_sol,
            (cash_lamports - self.reserve) / LAMPORTS_PER_SOL,
        ]
        lamports = int(max(0.0, min(caps)) * LAMPORTS_PER_SOL)
        if impact_bps_fn is not None and lamports > 0 and impact_bps_fn(lamports) > c.max_impact_bps:
            lo, hi = 0, lamports
            for _ in range(24):  # binary search the largest size within the impact budget
                mid = (lo + hi) // 2
                if impact_bps_fn(mid) <= c.max_impact_bps:
                    lo = mid
                else:
                    hi = mid
            lamports = lo
        return lamports if lamports >= self.min_order else 0
