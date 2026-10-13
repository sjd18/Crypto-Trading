"""Position management: stops, take-profit ladder (partial exits), trailing exits, time
exits and pyramiding rules.

All thresholds are evaluated on the *economic return at liquidation value* (net of exit fees
and our own price impact), not on the mid price — a position that is up 10 % on paper but would
net -2 % after selling into the curve is treated as -2 %.

Per-position ``exit_overrides`` (from the entry signal) may replace ``stop_loss_pct``,
``max_hold_s`` and set a single full ``take_profit_pct``.

Profit lock (``position.profit_lock``): once the best net return reaches ``activation_pct``, the
stop moves up to the higher of ``floor_pct`` and ``trail_pct`` below the best value, e.g. with 20 /
5 / 10: a position that reached +20 % is closed at +8 % at the latest (10 % off its +20 % value),
one that reached +50 % at +35 %, and never below +5 %. Until then the normal stop loss applies.
The exit is decided at the liquidation value; the fill can still land lower (latency, a dump
between decision and landing), which the simulator charges like any other fill.
"""

from __future__ import annotations

from typing import Any

from pumpfun_hft.core.types import Action, Signal, Urgency
from pumpfun_hft.risk.portfolio import Position


class PositionManager:
    """Generates EXIT / SCALE_OUT signals for open positions and gates SCALE_IN."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.levels = sorted(cfg.take_profit, key=lambda lv: lv.at_pct)

    def check(self, pos: Position, value_lamports: int, price: float, now_ms: int) -> Signal | None:
        c = self.cfg
        ov = pos.exit_overrides or {}
        ret = pos.economic_return(value_lamports)
        stop = ov.get("stop_loss_pct", c.stop_loss_pct) / 100.0
        lock = self.profit_lock_level(pos, ret)
        if lock is not None and ret <= lock:
            best = max(pos.mfe, ret)
            return Signal(Action.EXIT, 92, f"profit lock {ret:+.1%} (best {best:+.1%}, locked {lock:+.1%})", "position_manager",
                          urgency=Urgency.HIGH)
        if c.breakeven_after_first_tp and pos.tp_hits > 0:
            stop = min(stop, 0.0)
            if ret <= 0.0:
                return Signal(Action.EXIT, 90, "breakeven stop", "position_manager", urgency=Urgency.HIGH)
        if ret <= -stop:
            return Signal(Action.EXIT, 95, f"stop loss {ret:+.1%}", "position_manager", urgency=Urgency.EXIT)
        max_hold = ov.get("max_hold_s", c.max_hold_s)
        if now_ms - pos.entry_ms >= max_hold * 1000:
            return Signal(Action.EXIT, 70, "max hold time", "position_manager", urgency=Urgency.HIGH)
        if "take_profit_pct" in ov:
            if ret >= ov["take_profit_pct"] / 100.0:
                return Signal(Action.EXIT, 85, f"take profit {ret:+.1%}", "position_manager")
        elif pos.tp_hits < len(self.levels):
            lv = self.levels[pos.tp_hits]
            if ret >= lv.at_pct / 100.0:
                last = pos.tp_hits == len(self.levels) - 1 or lv.sell_frac >= 1.0
                if last:
                    return Signal(Action.EXIT, 85, f"final take profit {ret:+.1%}", "position_manager")
                return Signal(Action.SCALE_OUT, 80, f"take profit L{pos.tp_hits + 1} {ret:+.1%}", "position_manager",
                              size_frac=lv.sell_frac)
        t = c.trailing_stop
        if t.enabled and pos.mfe >= t.activation_pct / 100.0 and pos.peak_price > 0:
            if price <= pos.peak_price * (1.0 - t.trail_pct / 100.0):
                return Signal(Action.EXIT, 85, f"trailing stop from peak {pos.peak_price:.3g}", "position_manager", urgency=Urgency.HIGH)
        return None

    def profit_lock_level(self, pos: Position, ret: float | None = None) -> float | None:
        """The net return the position is locked at, or None while the lock is not armed."""
        pl = self.cfg.profit_lock
        best = max(pos.mfe, pos.economic_return() if ret is None else ret)
        if not pl.enabled or best < pl.activation_pct / 100.0:
            return None
        return max(pl.floor_pct / 100.0, (1.0 + best) * (1.0 - pl.trail_pct / 100.0) - 1.0)

    def allow_scale_in(self, pos: Position, value_lamports: int, confidence: float) -> bool:
        py = self.cfg.pyramiding
        if not py.enabled or pos.n_adds >= py.max_adds or confidence < py.min_confidence:
            return False
        return pos.economic_return(value_lamports) >= py.add_trigger_pct / 100.0 * (pos.n_adds + 1)

    def scale_in_fraction(self) -> float:
        return self.cfg.pyramiding.add_size_frac
