"""Live execution infrastructure: blockhash cache, priority-fee estimator, confirmation tracker, Jito.

* :class:`BlockhashCache` refreshes ``getLatestBlockhash`` in the background and tracks the chain
  block height so callers can tell when a blockhash (and every tx built on it) has expired.
* :class:`PriorityFeeEstimator` keeps an EWMA of the chosen percentile of
  ``getRecentPrioritizationFees`` for the Pump program's writable accounts, applies urgency
  multipliers and bumps on retries, and maps urgency to Metis ``priorityFeeLevel`` values.
* :class:`ConfirmationTracker` batches ``getSignatureStatuses`` for all in-flight signatures on a
  single polling loop and resolves per-signature futures (confirmed / failed / expired).
* :class:`JitoClient` sends bundles to the block engine (base64), queries statuses and tip accounts.
"""

from __future__ import annotations

import asyncio
import base64
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pumpfun_hft.api.http import AsyncHttpClient
from pumpfun_hft.api.rpc import SolanaRpcClient
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger

log = get_logger("api")


class BlockhashCache:
    """Background-refreshed latest blockhash + block height."""

    def __init__(self, rpc: SolanaRpcClient, refresh_ms: int, commitment: str = "confirmed") -> None:
        self.rpc = rpc
        self.refresh_s = refresh_ms / 1000.0
        self.commitment = commitment
        self.blockhash: str | None = None
        self.last_valid_block_height = 0
        self.block_height = 0
        self.fetched_at = 0.0
        self._task: asyncio.Task[None] | None = None

    async def refresh(self) -> None:
        bh, lvbh = await self.rpc.get_latest_blockhash(self.commitment)
        self.blockhash, self.last_valid_block_height = bh, lvbh
        self.block_height = await self.rpc.get_block_height(self.commitment)
        self.fetched_at = time.monotonic()

    async def get(self) -> tuple[str, int]:
        if self.blockhash is None or time.monotonic() - self.fetched_at > 2 * self.refresh_s:
            await self.refresh()
        assert self.blockhash is not None
        return self.blockhash, self.last_valid_block_height

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("blockhash refresh failed", extra={"data": {"err": repr(exc)[:200]}})
            await asyncio.sleep(self.refresh_s)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


class PriorityFeeEstimator:
    """Dynamic compute-unit price (micro-lamports) from recent prioritisation fees."""

    def __init__(self, rpc: SolanaRpcClient | None, cfg: Any, metis_levels: dict[str, str], accounts: list[str]) -> None:
        self.rpc = rpc
        self.cfg = cfg
        self.levels = metis_levels
        self.accounts = accounts
        self.current = float(cfg.fixed_micro_lamports)
        self.samples = 0
        self._task: asyncio.Task[None] | None = None

    def update_from(self, fees: list[dict[str, int]]) -> float:
        vals = np.array([f.get("prioritizationFee", 0) for f in fees], dtype=float)
        if vals.size:
            est = float(np.percentile(vals, self.cfg.dynamic_percentile))
            self.current = est if self.samples == 0 else 0.7 * self.current + 0.3 * est
            self.samples += 1
        return self.current

    async def refresh(self) -> float:
        if self.rpc is None or self.cfg.mode == "fixed":
            return self.current
        return self.update_from(await self.rpc.get_recent_prioritization_fees(self.accounts))

    def micro_lamports(self, urgency: str, attempt: int = 0) -> int:
        base = self.cfg.fixed_micro_lamports if self.cfg.mode == "fixed" else self.current
        v = base * self.cfg.urgency_multiplier.get(urgency, 1.0) * (self.cfg.retry_bump_factor ** attempt)
        return int(min(self.cfg.max_micro_lamports, max(self.cfg.min_micro_lamports, v)))

    def metis_level(self, urgency: str) -> str:
        return self.levels.get(urgency, "medium")

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("priority fee refresh failed", extra={"data": {"err": repr(exc)[:200]}})
            await asyncio.sleep(self.cfg.refresh_interval_s)

    def start(self) -> None:
        if self._task is None and self.cfg.mode == "dynamic" and self.rpc is not None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


@dataclass(slots=True)
class ConfirmResult:
    signature: str
    status: str            # confirmed | failed | expired | timeout
    slot: int | None = None
    err: Any = None
    elapsed_ms: float = 0.0


class ConfirmationTracker:
    """Batched signature-status polling shared by all in-flight transactions."""

    _RANK = {"processed": 0, "confirmed": 1, "finalized": 2}

    def __init__(self, rpc: SolanaRpcClient, poll_ms: int, commitment: str, latency: LatencyTracker | None = None) -> None:
        self.rpc = rpc
        self.poll_s = poll_ms / 1000.0
        self.need = self._RANK[commitment]
        self.latency = latency or LatencyTracker()
        self._waiting: dict[str, tuple[asyncio.Future[ConfirmResult], float]] = {}
        self._task: asyncio.Task[None] | None = None

    def track(self, signature: str) -> asyncio.Future[ConfirmResult]:
        fut = self._waiting.get(signature)
        if fut is not None:
            return fut[0]
        f: asyncio.Future[ConfirmResult] = asyncio.get_running_loop().create_future()
        self._waiting[signature] = (f, time.perf_counter())
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
        return f

    def cancel(self, signature: str, status: str = "expired") -> None:
        item = self._waiting.pop(signature, None)
        if item is not None and not item[0].done():
            item[0].set_result(ConfirmResult(signature, status, elapsed_ms=(time.perf_counter() - item[1]) * 1000))

    async def poll_once(self) -> None:
        sigs = list(self._waiting)
        for i in range(0, len(sigs), 256):
            chunk = sigs[i:i + 256]
            t0 = time.perf_counter_ns()
            statuses = await self.rpc.get_signature_statuses(chunk)
            self.latency.record_ns("rpc.signature_statuses", t0)
            for sig, st in zip(chunk, statuses, strict=True):
                if st is None:
                    continue
                rank = self._RANK.get(st.get("confirmationStatus") or "processed", 0)
                err = st.get("err")
                if err is not None or rank >= self.need:
                    fut, started = self._waiting.pop(sig)
                    if not fut.done():
                        fut.set_result(ConfirmResult(sig, "failed" if err is not None else "confirmed", st.get("slot"), err,
                                                     (time.perf_counter() - started) * 1000))

    async def _loop(self) -> None:
        while self._waiting:
            try:
                await self.poll_once()
            except Exception as exc:  # noqa: BLE001
                log.warning("status poll failed", extra={"data": {"err": repr(exc)[:200]}})
            await asyncio.sleep(self.poll_s)
        self._task = None


class JitoClient:
    """Jito block-engine JSON-RPC client (bundles of up to 5 base64 transactions)."""

    def __init__(self, http: AsyncHttpClient, tip_floor_http: AsyncHttpClient | None = None, tip_floor_field: str = "") -> None:
        self.http = http
        self.tip_floor_http = tip_floor_http
        self.tip_floor_field = tip_floor_field
        self._tip_accounts: list[str] = []

    async def _call(self, path: str, method: str, params: list[Any]) -> Any:
        data = await self.http.post_json(path, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        if data.get("error"):
            raise RuntimeError(f"jito {method}: {data['error']}")
        return data.get("result")

    async def tip_accounts(self) -> list[str]:
        if not self._tip_accounts:
            self._tip_accounts = list(await self._call("/api/v1/bundles", "getTipAccounts", []))
        return self._tip_accounts

    async def random_tip_account(self) -> str:
        return random.choice(await self.tip_accounts())

    async def send_bundle(self, txs: list[bytes]) -> str:
        if not 1 <= len(txs) <= 5:
            raise ValueError("a bundle holds 1-5 transactions")
        return str(await self._call("/api/v1/bundles", "sendBundle", [[base64.b64encode(t).decode() for t in txs], {"encoding": "base64"}]))

    async def bundle_statuses(self, bundle_ids: list[str]) -> Any:
        return await self._call("/api/v1/bundles", "getBundleStatuses", [bundle_ids])

    async def tip_floor_lamports(self) -> int | None:
        if self.tip_floor_http is None:
            return None
        data = await self.tip_floor_http.get_json("")
        row = data[0] if isinstance(data, list) and data else data
        v = row.get(self.tip_floor_field) if isinstance(row, dict) else None
        return int(float(v) * 1e9) if v is not None else None  # tip floor API reports SOL
