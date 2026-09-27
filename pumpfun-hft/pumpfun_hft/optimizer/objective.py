"""Optimisation objectives with minimum-trade and drawdown constraints.

``calmar`` uses CAGR / max drawdown when the span is at least 30 days and the period return / max
drawdown (recovery factor) on shorter spans, where annualising is meaningless.
"""

from __future__ import annotations

import math
from typing import Any

PENALTY = -1e3


def objective_value(metrics: dict[str, Any], objective: str, min_trades: int, max_drawdown_pct: float) -> float:
    """Scalar to maximise. Constraint violations get a large negative value that still ranks
    candidates with more trades higher (so search can climb out of infeasible regions)."""
    n = int(metrics.get("n_trades") or 0)
    if n < min_trades:
        return PENALTY + n
    dd = float(metrics.get("max_drawdown") or 0.0) * 100.0
    if dd > max_drawdown_pct:
        return PENALTY / 2 - dd
    if objective == "composite":
        sh = metrics.get("sharpe")
        sh = sh if sh is not None and math.isfinite(sh) else 0.0
        v = sh * min(1.0, n / (2.0 * min_trades)) * (1.0 - dd / 100.0)
    else:
        key = {"sharpe": "sharpe", "sortino": "sortino", "profit_factor": "profit_factor", "expectancy": "expectancy_ret",
               "calmar": "calmar", "total_return": "total_return"}[objective]
        v = metrics.get(key)
        if objective == "calmar" and (v is None or not math.isfinite(float(v))):
            v = metrics.get("recovery_factor")  # spans < 30 days: period return / max drawdown
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return PENALTY / 4
        if objective == "profit_factor":
            v = min(float(v), 10.0)
        v = float(v)
    return v if math.isfinite(v) else (1e6 if v > 0 else PENALTY)
