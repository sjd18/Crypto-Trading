"""Structured JSON logging with rotating per-channel files and secret redaction.

Channels (separate files under ``logs_dir``): ``api``, ``trades``, ``errors``, ``latency``,
``signals``, ``backtests`` and ``system``. Every ERROR-or-worse record from *any* channel is
also written to ``errors.log``.

Handlers never block the asyncio event loop: records are pushed onto a ``queue.Queue`` via
``QueueHandler`` and written by a background ``QueueListener`` thread.

Secrets (JWTs, private keys, RPC URLs carrying API keys) are registered with
:func:`register_secret` and replaced by ``***`` in every formatted message and field.

Example::

    setup_logging(cfg.logging, logs_dir=Path("pumpfun_hft/logs"))
    get_logger("trades").info("fill", extra={"data": {"mint": m, "sol": 0.25}})
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import orjson

CHANNELS: tuple[str, ...] = ("api", "trades", "errors", "latency", "signals", "backtests", "system")
_ROOT = "pumpfun"
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()
_LISTENER: logging.handlers.QueueListener | None = None
_CONFIGURED = False


def register_secret(value: str | None) -> None:
    """Register a secret string so it is redacted from all log output."""
    if value and len(value) >= 6:
        with _SECRETS_LOCK:
            _SECRETS.add(value)


def redact(text: str) -> str:
    """Replace every registered secret occurring in ``text`` with ``***``."""
    if not _SECRETS:
        return text
    for secret in _SECRETS:
        if secret in text:
            text = text.replace(secret, "***")
    return text


def _redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_obj(v) for v in obj]
    return obj


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line (orjson), redacting secrets."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "channel": record.name.removeprefix(_ROOT + "."),
            "msg": redact(record.getMessage()),
        }
        data = getattr(record, "data", None)
        if data is not None:
            payload["data"] = _redact_obj(data)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return orjson.dumps(payload, default=str, option=orjson.OPT_SERIALIZE_NUMPY).decode()


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable console format with redaction."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name.removeprefix(_ROOT + '.')}: {record.getMessage()}"
        data = getattr(record, "data", None)
        if data:
            base += " " + orjson.dumps(data, default=str, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        return redact(base)


# Library convention: no output unless the application configures logging (setup_logging). Without
# this, worker processes and notebooks would fall back to logging.lastResort and print warnings.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())


def get_logger(channel: str) -> logging.Logger:
    """Return the logger for a channel (``api``, ``trades``, ...)."""
    return logging.getLogger(f"{_ROOT}.{channel}")


def setup_logging(
    level: str = "INFO",
    logs_dir: str | Path = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
    channels: tuple[str, ...] | list[str] = CHANNELS,
) -> None:
    """Configure channel loggers. Idempotent: subsequent calls reconfigure cleanly."""
    global _LISTENER, _CONFIGURED
    shutdown_logging()
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    json_fmt = JsonFormatter()
    handlers: list[logging.Handler] = []
    routing: dict[str, logging.Handler] = {}
    for ch in channels:
        fh = logging.handlers.RotatingFileHandler(
            logs_path / f"{ch}.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(json_fmt)
        # errors.log: its own channel plus every ERROR+ record from any channel (one handler per
        # file, so rotation never races between two handlers on the same path)
        fh.addFilter(_ErrorsFilter() if ch == "errors" else _ChannelFilter(ch))
        routing[ch] = fh
        handlers.append(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(ConsoleFormatter())
        handlers.append(sh)

    q: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    root = logging.getLogger(_ROOT)
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(logging.handlers.QueueHandler(q))
    root.propagate = False
    _LISTENER = logging.handlers.QueueListener(q, *handlers, respect_handler_level=True)
    _LISTENER.start()
    _CONFIGURED = True


def shutdown_logging() -> None:
    """Flush and stop the background listener (safe to call repeatedly)."""
    global _LISTENER
    if _LISTENER is not None:
        try:
            _LISTENER.stop()
        finally:
            for h in _LISTENER.handlers:
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
            _LISTENER = None


atexit.register(shutdown_logging)


class _ChannelFilter(logging.Filter):
    def __init__(self, channel: str) -> None:
        super().__init__()
        self.name_full = f"{_ROOT}.{channel}"

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self.name_full


class _ErrorsFilter(logging.Filter):
    """Accept the ``errors`` channel and every ERROR+ record from any channel."""

    def __init__(self) -> None:
        super().__init__()
        self.name_full = f"{_ROOT}.errors"

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self.name_full or record.levelno >= logging.ERROR
