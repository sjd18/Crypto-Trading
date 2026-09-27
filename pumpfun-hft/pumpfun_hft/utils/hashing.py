"""Checksums and stable hashes used for data integrity and run reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import orjson


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, streaming in ``chunk_size`` blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of a JSON-serialisable object (sorted keys)."""
    payload = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY, default=str)
    return hashlib.sha256(payload).hexdigest()[:length]
