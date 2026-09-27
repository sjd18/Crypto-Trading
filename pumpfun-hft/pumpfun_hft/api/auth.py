"""JWT bearer authentication for Mode B (authenticated Metis / QuickNode endpoints).

QuickNode endpoint JWT security expects ``Authorization: Bearer <JWT>`` signed with RS256 or
ES256 whose ``kid`` header matches the key registered on the endpoint. Two sources:

* ``PUMPFUN_JWT`` — a pre-issued token (static; the provider warns when it nears expiry);
* ``QN_JWT_PRIVATE_KEY_PATH`` + ``QN_JWT_KID`` — the provider mints short-lived tokens itself
  and refreshes them ``refresh_margin_s`` before expiry (requires ``pyjwt[crypto]``).

The token never appears in code or logs (it is registered with the log redactor).
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path

import orjson

from pumpfun_hft.utils.logging import get_logger, register_secret

log = get_logger("api")


def _jwt_exp(token: str) -> float | None:
    """Read the ``exp`` claim without verifying the signature (for refresh scheduling only)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = orjson.loads(base64.urlsafe_b64decode(payload))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001
        return None


class JwtProvider:
    """Supplies ``Authorization`` headers for authenticated requests."""

    def __init__(self, static_token: str | None = None, private_key_path: str | None = None,
                 kid: str | None = None, algorithm: str = "RS256", lifetime_s: int = 3600,
                 refresh_margin_s: int = 120) -> None:
        if not static_token and not private_key_path:
            raise ValueError("JwtProvider needs PUMPFUN_JWT or QN_JWT_PRIVATE_KEY_PATH")
        self._static = static_token
        self._key_path = private_key_path
        self._kid = kid
        self._alg = algorithm
        self._lifetime = lifetime_s
        self._margin = refresh_margin_s
        self._token: str | None = static_token
        self._exp: float | None = _jwt_exp(static_token) if static_token else None
        self._lock = asyncio.Lock()
        if static_token:
            register_secret(static_token)

    def _mint(self) -> str:
        import jwt  # pyjwt, optional dependency (extra "live")

        key = Path(self._key_path).read_text(encoding="utf-8")  # type: ignore[arg-type]
        now = int(time.time())
        token = jwt.encode({"iat": now, "exp": now + self._lifetime}, key, algorithm=self._alg,
                           headers={"kid": self._kid} if self._kid else None)
        register_secret(token)
        self._exp = now + self._lifetime
        return token

    async def token(self) -> str:
        async with self._lock:
            if self._key_path and (self._token is None or self._exp is None or time.time() > self._exp - self._margin):
                self._token = self._mint()
                log.info("jwt refreshed", extra={"data": {"exp": self._exp}})
            elif self._static and self._exp is not None and time.time() > self._exp - self._margin:
                log.warning("static PUMPFUN_JWT expires soon; rotate it or configure auto-minting",
                            extra={"data": {"exp": self._exp}})
            assert self._token is not None
            return self._token

    async def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self.token()}"}
