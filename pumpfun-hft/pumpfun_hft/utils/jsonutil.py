"""Strict JSON output.

Python's ``json`` writes ``NaN`` / ``Infinity`` by default, which is not JSON: browsers'
``JSON.parse``, DuckDB, jq and most other consumers reject it. Metrics legitimately contain
non-finite values (a Calmar ratio on a 12-hour backtest, a profit factor with no losing trades),
so every JSON file and API response goes through :func:`jsonable`, which maps them to ``null``.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


def jsonable(o: Any) -> Any:
    """Recursively convert ``o`` into values that strict JSON can represent.

    Non-finite floats become ``None``; numpy scalars and arrays become Python numbers and lists;
    dates become ISO strings; paths and anything else unknown become ``str``.
    """
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):  # numpy scalar or array
        return jsonable(o.tolist())
    if hasattr(o, "item"):
        try:
            return jsonable(o.item())
        except (TypeError, ValueError):
            pass
    return str(o)


def dumps(o: Any, indent: int | None = 2) -> str:
    """``json.dumps`` that never emits NaN / Infinity (raises instead of writing invalid JSON)."""
    return json.dumps(jsonable(o), indent=indent, allow_nan=False)
