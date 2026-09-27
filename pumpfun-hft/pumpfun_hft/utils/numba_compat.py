"""Optional numba acceleration.

``njit`` compiles with numba when it is importable and falls back to the plain Python
function otherwise, so every accelerated routine remains importable and testable without
numba. Set ``PUMPFUN_DISABLE_NUMBA=1`` to force the fallback (useful when debugging).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

NUMBA_AVAILABLE = False
if os.environ.get("PUMPFUN_DISABLE_NUMBA") != "1":
    try:  # pragma: no cover - depends on environment
        import numba as _numba

        NUMBA_AVAILABLE = True
    except Exception:  # noqa: BLE001
        _numba = None
else:  # pragma: no cover
    _numba = None


def njit(*args: Any, **kwargs: Any) -> Any:
    """Drop-in for ``numba.njit`` supporting both ``@njit`` and ``@njit(cache=True)``."""
    if NUMBA_AVAILABLE:
        return _numba.njit(*args, **kwargs)  # type: ignore[union-attr]
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def wrap(fn: F) -> F:
        return fn

    return wrap
