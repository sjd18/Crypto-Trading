"""Shared utilities.

Purpose
    Cross-cutting helpers with no dependency on trading logic.

Architecture
    logging.py      JSON, rotating, per-channel logging behind a non-blocking QueueListener,
                    with a redaction filter that scrubs registered secrets.
    latency.py      streaming latency statistics (ring-buffer percentiles, EWMA, budgets).
    retry.py        exponential backoff with full jitter for async callables.
    timeutil.py     wall/monotonic clocks and date helpers (UTC, milliseconds).
    hashing.py      checksums and stable content hashes for reproducibility.
    numba_compat.py optional numba ``njit`` with a pure-Python fallback.
    base58.py       base58 encode/decode (fast path via solders when installed).

Data flow
    Every other package imports from here; nothing here imports trading code. Log records flow
    logger -> redaction -> QueueHandler -> background listener -> per-channel rotating JSON files.

Inputs / Outputs
    Pure functions and small classes; no I/O except logging handlers and file hashing.

Example
    >>> from pumpfun_hft.utils.latency import LatencyTracker
    >>> lt = LatencyTracker(window=128); lt.record("rpc", 12.5); round(lt.percentile("rpc", 50), 1)
    12.5
"""
