"""QuickNode Metis client — Pump.fun and Jupiter swap APIs.

Modes (``network.metis.mode``)
    * ``public`` (Mode A): ``https://public.jupiterapi.com``. No credentials. Only
      ``/pump-fun/swap`` is served for Pump.fun (quotes are computed locally from on-chain curve
      state instead), a platform fee is applied to swaps, and the endpoint is scheduled to shut
      down on ``public_sunset_date`` (2026-10-14) — a warning is logged as the date approaches.
    * ``authenticated`` (Mode B): ``METIS_URL`` from ``.env`` with ``Authorization: Bearer <JWT>``
      (``PUMPFUN_JWT`` or auto-minted, see :mod:`pumpfun_hft.api.auth`). Enables
      ``/pump-fun/quote`` and ``/pump-fun/swap-instructions`` (paid Metis plans).

Endpoints
    GET  /pump-fun/quote              type=BUY|SELL, mint, amount (lamports / token base units)
    POST /pump-fun/swap               wallet, type, mint, inAmount, slippageBps, priorityFeeLevel -> {"tx": base64}
    POST /pump-fun/swap-instructions  same body -> {"instructions": [...]} (compose own tx / Jito tip)
    GET  /quote, POST /swap           Jupiter aggregator (routes migrated tokens via PumpSwap etc.)
"""

from __future__ import annotations

import base64
from datetime import UTC, date, datetime
from typing import Any

from pumpfun_hft.api.auth import JwtProvider
from pumpfun_hft.api.http import AsyncHttpClient, build_client
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger

log = get_logger("api")


class CapabilityUnavailable(RuntimeError):
    """The requested endpoint is not available in the configured mode/plan."""


class MetisClient:
    """Async client for Metis REST endpoints (see module docstring)."""

    def __init__(self, http: AsyncHttpClient, mode: str, sunset_date: str | None = None) -> None:
        self.http = http
        self.mode = mode
        self.sunset_date = sunset_date
        if mode == "public" and sunset_date:
            self._warn_sunset(sunset_date)

    @staticmethod
    def _warn_sunset(sunset: str) -> None:
        try:
            days = (date.fromisoformat(sunset) - datetime.now(UTC).date()).days
        except ValueError:
            return
        if days <= 0:
            log.error("public Metis endpoint is past its announced shutdown date; switch to network.metis.mode=authenticated",
                      extra={"data": {"sunset": sunset}})
        elif days <= 30:
            log.warning("public Metis endpoint shuts down soon", extra={"data": {"sunset": sunset, "days_left": days}})

    @classmethod
    def from_settings(cls, settings: Any, secrets: Any, latency: LatencyTracker | None = None, transport: Any = None) -> MetisClient:
        m = settings.network.metis
        auth = None
        if m.mode == "authenticated":
            base = secrets.get("metis_url")
            if not base:
                raise ValueError("network.metis.mode=authenticated requires METIS_URL in .env")
            if secrets.get("pumpfun_jwt") or secrets.get("qn_jwt_private_key_path"):
                jwt = settings.network.jwt
                auth = JwtProvider(secrets.get("pumpfun_jwt"), secrets.get("qn_jwt_private_key_path"), secrets.get("qn_jwt_kid"),
                                   jwt.algorithm, jwt.lifetime_s, jwt.refresh_margin_s).headers
        else:
            base = m.public_url
        http = build_client("metis", base.rstrip("/"), m.rate_limit, settings.network, m.timeout_s, latency, auth, transport)
        return cls(http, m.mode, m.public_sunset_date)

    def _require_auth(self, what: str) -> None:
        if self.mode != "authenticated":
            raise CapabilityUnavailable(f"{what} requires authenticated Metis (paid plan); public endpoint only serves /pump-fun/swap")

    # ------------------------------------------------------------------ pump.fun
    async def pump_quote(self, side: str, mint: str, amount: int, commitment: str | None = None) -> dict[str, Any]:
        """Quote a bonding-curve trade. ``amount`` is lamports (BUY) or token base units (SELL)."""
        self._require_auth("/pump-fun/quote")
        params: dict[str, Any] = {"type": side.upper(), "mint": mint, "amount": str(int(amount))}
        if commitment:
            params["commitment"] = commitment
        data = await self.http.get_json("/pump-fun/quote", params)
        return data.get("quote", data)

    def _swap_body(self, wallet: str, side: str, mint: str, in_amount: int, slippage_bps: int | None,
                   priority_fee_level: str | None, commitment: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {"wallet": wallet, "type": side.upper(), "mint": mint, "inAmount": str(int(in_amount))}
        if slippage_bps is not None:
            body["slippageBps"] = str(int(slippage_bps))
        if priority_fee_level:
            body["priorityFeeLevel"] = priority_fee_level
        if commitment:
            body["commitment"] = commitment
        return body

    async def pump_swap(self, wallet: str, side: str, mint: str, in_amount: int, *, slippage_bps: int | None = None,
                        priority_fee_level: str | None = None, commitment: str | None = None) -> bytes:
        """Build an unsigned (versioned) swap transaction; returns raw transaction bytes."""
        data = await self.http.post_json("/pump-fun/swap", self._swap_body(wallet, side, mint, in_amount, slippage_bps, priority_fee_level, commitment))
        tx = data.get("tx")
        if not tx:
            raise ValueError(f"unexpected /pump-fun/swap response keys: {sorted(data)}")
        return base64.b64decode(tx)

    async def pump_swap_instructions(self, wallet: str, side: str, mint: str, in_amount: int, *, slippage_bps: int | None = None,
                                     priority_fee_level: str | None = None, commitment: str | None = None) -> list[dict[str, Any]]:
        """Swap instructions (programId, keys, data) for composing a custom transaction (paid plans)."""
        self._require_auth("/pump-fun/swap-instructions")
        data = await self.http.post_json("/pump-fun/swap-instructions",
                                         self._swap_body(wallet, side, mint, in_amount, slippage_bps, priority_fee_level, commitment))
        return list(data.get("instructions") or [])

    # ------------------------------------------------------------------ Jupiter (migrated tokens / other DEXs)
    async def quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int, **extra: Any) -> dict[str, Any]:
        params = {"inputMint": input_mint, "outputMint": output_mint, "amount": str(int(amount)), "slippageBps": str(int(slippage_bps))}
        params.update({k: str(v) for k, v in extra.items()})
        return await self.http.get_json("/quote", params)

    async def swap(self, quote_response: dict[str, Any], user_public_key: str, *, prioritization_fee_lamports: int | None = None,
                   dynamic_compute_unit_limit: bool = True) -> bytes:
        body: dict[str, Any] = {"quoteResponse": quote_response, "userPublicKey": user_public_key,
                                "dynamicComputeUnitLimit": dynamic_compute_unit_limit, "wrapAndUnwrapSol": True}
        if prioritization_fee_lamports is not None:
            body["prioritizationFeeLamports"] = int(prioritization_fee_lamports)
        data = await self.http.post_json("/swap", body)
        return base64.b64decode(data["swapTransaction"])

    async def aclose(self) -> None:
        await self.http.aclose()
