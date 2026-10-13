"""Tuned strategy parameters: the result of an optimisation study, saved from the dashboard's
Strategy Parameters page and applied to paper trading on the live stream (``hfts``).

The file ``tuned_params.json`` lives in the real data set folder (the data the study ran on):
``{strategy: {"params": {...}, "study_id", "saved_ms", "objective", "val_metrics", "baseline_val_metrics"}}``.
Only ``--profile hfts`` loads it; explicit ``--set strategy.params.*`` values still win, and backtests
(hft / hftr) keep the configured parameters, so the tuned values are never judged on the data they
were fitted to.

Example::

    save_tuned(root, "smart_money", {"params": {"min_smart_buyers": 3}, "study_id": "study-1a2b3c4d"})
    tuned_overrides(load_tuned(root))   # {"strategy.params.smart_money.min_smart_buyers": 3}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pumpfun_hft.utils import jsonutil

FILE_NAME = "tuned_params.json"


def tuned_path(root: str | Path) -> Path:
    return Path(root) / FILE_NAME


def load_tuned(root: str | Path) -> dict[str, dict[str, Any]]:
    p = tuned_path(root)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("params"), dict)}


def _write(root: str | Path, data: dict[str, Any]) -> None:
    p = tuned_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(jsonutil.dumps(data), encoding="utf-8")
    os.replace(tmp, p)


def save_tuned(root: str | Path, strategy: str, entry: dict[str, Any]) -> dict[str, Any]:
    data = load_tuned(root)
    data[strategy] = {**entry, "saved_ms": int(time.time() * 1000)}
    _write(root, data)
    return data[strategy]


def remove_tuned(root: str | Path, strategy: str) -> bool:
    data = load_tuned(root)
    if strategy not in data:
        return False
    del data[strategy]
    _write(root, data)
    return True


def tuned_overrides(tuned: dict[str, dict[str, Any]], strategies: list[str] | None = None) -> dict[str, Any]:
    """Dotted config overrides for the tuned strategies (all of them, or only ``strategies``)."""
    out: dict[str, Any] = {}
    for name, entry in tuned.items():
        if strategies is None or name in strategies:
            for k, v in entry["params"].items():
                out[f"strategy.params.{name}.{k}"] = v
    return out
