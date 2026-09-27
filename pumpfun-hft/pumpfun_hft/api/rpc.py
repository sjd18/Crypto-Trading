"""Solana JSON-RPC client (single and batched requests) with typed helpers.

Batching (``batch``) packs many calls into one HTTP request — the historical collector uses it
to fetch transactions ``collector.tx_batch_size`` at a time under a concurrency semaphore.
JSON-RPC level errors raise :class:`RpcError`; per-item errors inside a batch are returned as
``RpcError`` instances in the result list so one bad item does not fail the batch.
"""

from __future__ import annotations

import base64
from itertools import count
from typing import Any

from pumpfun_hft.api.http import AsyncHttpClient


class RpcError(Exception):
    """JSON-RPC error object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data

    @property
    def is_blockhash_not_found(self) -> bool:
        return "blockhash not found" in self.message.lower()


class SolanaRpcClient:
    """Typed Solana RPC wrapper around :class:`AsyncHttpClient`."""

    def __init__(self, http: AsyncHttpClient, url: str, commitment: str = "confirmed") -> None:
        self.http = http
        self.url = url
        self.commitment = commitment
        self._ids = count(1)

    async def call(self, method: str, params: list[Any] | None = None) -> Any:
        payload = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or []}
        data = await self.http.post_json(self.url, payload)
        if "error" in data and data["error"]:
            err = data["error"]
            raise RpcError(err.get("code", 0), err.get("message", ""), err.get("data"))
        return data.get("result")

    async def batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        """Execute several calls in one HTTP request; results keep the input order."""
        if not calls:
            return []
        ids = [next(self._ids) for _ in calls]
        payload = [{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in zip(ids, calls, strict=True)]
        data = await self.http.post_json(self.url, payload)
        if isinstance(data, dict):  # some providers return a single error object for the whole batch
            err = data.get("error") or {"code": -1, "message": "invalid batch response"}
            raise RpcError(err.get("code", -1), err.get("message", ""))
        by_id = {item.get("id"): item for item in data}
        out: list[Any] = []
        for i in ids:
            item = by_id.get(i)
            if item is None:
                out.append(RpcError(-1, "missing batch item"))
            elif item.get("error"):
                e = item["error"]
                out.append(RpcError(e.get("code", 0), e.get("message", ""), e.get("data")))
            else:
                out.append(item.get("result"))
        return out

    # ------------------------------------------------------------------ helpers
    async def get_slot(self, commitment: str | None = None) -> int:
        return int(await self.call("getSlot", [{"commitment": commitment or self.commitment}]))

    async def get_block_height(self, commitment: str | None = None) -> int:
        return int(await self.call("getBlockHeight", [{"commitment": commitment or self.commitment}]))

    async def get_latest_blockhash(self, commitment: str | None = None) -> tuple[str, int]:
        res = await self.call("getLatestBlockhash", [{"commitment": commitment or self.commitment}])
        v = res["value"]
        return v["blockhash"], int(v["lastValidBlockHeight"])

    async def get_account_info(self, address: str, commitment: str | None = None) -> bytes | None:
        res = await self.call("getAccountInfo", [address, {"encoding": "base64", "commitment": commitment or self.commitment}])
        value = res.get("value") if res else None
        if not value:
            return None
        return base64.b64decode(value["data"][0])

    async def get_multiple_accounts(self, addresses: list[str]) -> list[bytes | None]:
        res = await self.call("getMultipleAccounts", [addresses, {"encoding": "base64", "commitment": self.commitment}])
        return [base64.b64decode(v["data"][0]) if v else None for v in res["value"]]

    async def get_signatures_for_address(self, address: str, *, before: str | None = None, until: str | None = None,
                                         limit: int = 1000) -> list[dict[str, Any]]:
        opts: dict[str, Any] = {"limit": limit, "commitment": "confirmed" if self.commitment == "processed" else self.commitment}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        return list(await self.call("getSignaturesForAddress", [address, opts]) or [])

    @staticmethod
    def tx_params(signature: str) -> list[Any]:
        return [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"}]

    async def get_transaction(self, signature: str) -> dict[str, Any] | None:
        return await self.call("getTransaction", self.tx_params(signature))

    async def get_transactions(self, signatures: list[str]) -> list[Any]:
        return await self.batch([("getTransaction", self.tx_params(s)) for s in signatures])

    async def get_block_hashes(self, slots: list[int]) -> dict[int, str | None]:
        calls = [("getBlock", [s, {"transactionDetails": "none", "rewards": False, "maxSupportedTransactionVersion": 0}]) for s in slots]
        res = await self.batch(calls)
        return {s: (r.get("blockhash") if isinstance(r, dict) else None) for s, r in zip(slots, res, strict=True)}

    async def send_transaction(self, tx_bytes: bytes, *, skip_preflight: bool = True, max_retries: int | None = 0) -> str:
        opts: dict[str, Any] = {"encoding": "base64", "skipPreflight": skip_preflight, "preflightCommitment": self.commitment}
        if max_retries is not None:
            opts["maxRetries"] = max_retries
        return str(await self.call("sendTransaction", [base64.b64encode(tx_bytes).decode(), opts]))

    async def simulate_transaction(self, tx_bytes: bytes) -> dict[str, Any]:
        return await self.call("simulateTransaction", [base64.b64encode(tx_bytes).decode(),
                                                       {"encoding": "base64", "sigVerify": False, "replaceRecentBlockhash": True}])

    async def get_signature_statuses(self, signatures: list[str], search_history: bool = False) -> list[dict[str, Any] | None]:
        res = await self.call("getSignatureStatuses", [signatures, {"searchTransactionHistory": search_history}])
        return list(res["value"])

    async def get_recent_prioritization_fees(self, accounts: list[str] | None = None) -> list[dict[str, int]]:
        return list(await self.call("getRecentPrioritizationFees", [accounts or []]) or [])

    async def get_recent_performance_samples(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(await self.call("getRecentPerformanceSamples", [limit]) or [])

    async def get_token_largest_accounts(self, mint: str) -> list[dict[str, Any]]:
        res = await self.call("getTokenLargestAccounts", [mint, {"commitment": self.commitment}])
        return list(res["value"])

    async def get_token_supply(self, mint: str) -> int:
        res = await self.call("getTokenSupply", [mint, {"commitment": self.commitment}])
        return int(res["value"]["amount"])

    async def get_balance(self, address: str) -> int:
        res = await self.call("getBalance", [address, {"commitment": self.commitment}])
        return int(res["value"])
