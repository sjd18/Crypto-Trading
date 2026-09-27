"""Pump.fun data facade: bonding-curve state, token metadata, holder analytics, profiles.

Sources, in order of trust:
    * on-chain (RPC): ``BondingCurve`` / ``FeeConfig`` accounts decoded with the bundled IDL,
      Token-2022 ``tokenMetadata`` extension or Metaplex metadata accounts, largest token accounts;
    * token URI JSON (IPFS / HTTPS) for description, image and social links;
    * optional pump.fun frontend API (unofficial; disabled by default) for platform analytics
      and user profiles.

``MetadataFetcher.anomalies`` flags metadata red flags used by discovery and the rug model
(missing socials, malformed links, zero-width / homoglyph characters, name reuse).
"""

from __future__ import annotations

import asyncio
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pumpfun_hft.api.http import AsyncHttpClient, HttpError, ServiceUnavailable
from pumpfun_hft.api.rpc import SolanaRpcClient
from pumpfun_hft.core.curve import CurveState
from pumpfun_hft.core.idl import IdlCodec
from pumpfun_hft.utils.logging import get_logger

log = get_logger("api")
_DEFAULT_PUBKEY = "11111111111111111111111111111111"
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_URL = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.I)


@dataclass(slots=True)
class TokenMetadata:
    mint: str
    name: str = ""
    symbol: str = ""
    uri: str = ""
    description: str = ""
    image: str = ""
    twitter: str = ""
    telegram: str = ""
    website: str = ""
    fetched: bool = False
    error: str = ""
    anomalies: list[str] = field(default_factory=list)

    @property
    def has_socials(self) -> bool:
        return bool(self.twitter or self.telegram or self.website)

    @property
    def n_socials(self) -> int:
        return int(bool(self.twitter)) + int(bool(self.telegram)) + int(bool(self.website))


@dataclass(slots=True)
class HolderStats:
    mint: str
    supply: int
    top_accounts: list[tuple[str, int]]
    top10_pct: float
    hhi_top: float
    excluded_curve: bool


class MetadataFetcher:
    """Fetches token URI JSON with IPFS gateway fallback, LRU caching and concurrency limits."""

    def __init__(self, http: AsyncHttpClient, gateways: list[str], max_concurrency: int, cache_size: int) -> None:
        self.http = http
        self.gateways = gateways
        self.sem = asyncio.Semaphore(max_concurrency)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.cache_size = cache_size

    def candidate_urls(self, uri: str) -> list[str]:
        uri = uri.strip()
        cid = None
        if uri.startswith("ipfs://"):
            cid = uri[len("ipfs://"):].removeprefix("ipfs/")
        else:
            m = re.search(r"/ipfs/([A-Za-z0-9]+.*)$", uri)
            if m:
                cid = m.group(1)
        if cid:
            urls = [g.rstrip("/") + "/" + cid for g in self.gateways]
            if uri.startswith("http"):
                urls.insert(0, uri)
            return urls
        return [uri] if uri.startswith("http") else []

    async def fetch_json(self, uri: str) -> dict[str, Any]:
        if uri in self.cache:
            self.cache.move_to_end(uri)
            return self.cache[uri]
        last: Exception | None = None
        async with self.sem:
            for url in self.candidate_urls(uri):
                try:
                    data = await self.http.get_json(url)
                    if isinstance(data, dict):
                        self.cache[uri] = data
                        if len(self.cache) > self.cache_size:
                            self.cache.popitem(last=False)
                        return data
                except (HttpError, ServiceUnavailable, ValueError) as exc:
                    last = exc
        raise ValueError(f"metadata unavailable: {last!r}")

    @staticmethod
    def extract_socials(data: dict[str, Any]) -> tuple[str, str, str]:
        """Pull twitter/telegram/website from common metadata layouts."""
        ext = data.get("extensions") if isinstance(data.get("extensions"), dict) else {}
        props = data.get("properties") if isinstance(data.get("properties"), dict) else {}

        def pick(*keys: str) -> str:
            for src in (data, ext, props):
                for k in keys:
                    v = src.get(k) if isinstance(src, dict) else None
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            return ""

        return pick("twitter", "x"), pick("telegram", "tg"), pick("website", "external_url", "site")

    @staticmethod
    def anomalies(meta: TokenMetadata, recent_names: set[str] | None = None) -> list[str]:
        out: list[str] = []
        if not meta.has_socials:
            out.append("missing_socials")
        for label, link in (("twitter", meta.twitter), ("telegram", meta.telegram), ("website", meta.website)):
            if link and not _URL.match(link) and not link.startswith("@"):
                out.append(f"malformed_{label}")
        text = meta.name + meta.symbol
        if _ZERO_WIDTH.search(text):
            out.append("zero_width_chars")
        if any(ord(c) > 0x2FF for c in text) and any(c.isascii() and c.isalpha() for c in text):
            out.append("mixed_script")
        if recent_names is not None and meta.name.strip().lower() in recent_names:
            out.append("duplicate_name")
        if not meta.image:
            out.append("missing_image")
        return out

    async def fetch(self, mint: str, name: str, symbol: str, uri: str, recent_names: set[str] | None = None) -> TokenMetadata:
        meta = TokenMetadata(mint=mint, name=name, symbol=symbol, uri=uri)
        try:
            data = await self.fetch_json(uri)
            meta.description = str(data.get("description") or "")[:2000]
            meta.image = str(data.get("image") or "")
            meta.twitter, meta.telegram, meta.website = self.extract_socials(data)
            meta.fetched = True
        except Exception as exc:  # noqa: BLE001
            meta.error = str(exc)[:200]
        meta.anomalies = self.anomalies(meta, recent_names)
        return meta


class PumpFunFrontendClient:
    """Optional client for the (unofficial) pump.fun frontend API. Paths are configurable."""

    def __init__(self, http: AsyncHttpClient, coin_path: str, user_path: str) -> None:
        self.http = http
        self.coin_path = coin_path
        self.user_path = user_path

    async def coin(self, mint: str) -> dict[str, Any]:
        return await self.http.get_json(self.coin_path.format(mint=mint))

    async def user(self, address: str) -> dict[str, Any]:
        return await self.http.get_json(self.user_path.format(address=address))


def decode_metaplex_metadata(data: bytes) -> tuple[str, str, str]:
    """Decode name/symbol/uri from a Metaplex ``Metadata`` account (fixed prefix layout)."""
    off = 1 + 32 + 32  # key, update_authority, mint
    out = []
    for _ in range(3):
        (n,) = struct.unpack_from("<I", data, off)
        off += 4
        out.append(data[off:off + n].decode("utf-8", "replace").rstrip("\x00").strip())
        off += n
    return out[0], out[1], out[2]


class PumpDataClient:
    """Facade combining RPC + IDL decoding + metadata + optional frontend API.

    Example::

        pdc = PumpDataClient(rpc, pump_codec, metadata_fetcher)
        state, raw = await pdc.bonding_curve_state(curve_address)
        holders = await pdc.holder_analytics(mint, curve_address)
    """

    def __init__(self, rpc: SolanaRpcClient, pump_codec: IdlCodec, metadata: MetadataFetcher | None = None,
                 frontend: PumpFunFrontendClient | None = None) -> None:
        self.rpc = rpc
        self.codec = pump_codec
        self.metadata = metadata
        self.frontend = frontend

    @staticmethod
    def curve_state_from_account(fields: dict[str, Any]) -> CurveState:
        v_sol = fields.get("virtual_quote_reserves", fields.get("virtual_sol_reserves"))
        r_sol = fields.get("real_quote_reserves", fields.get("real_sol_reserves"))
        creator = fields.get("creator")
        return CurveState(
            v_tok=int(fields["virtual_token_reserves"]), v_sol=int(v_sol), r_tok=int(fields["real_token_reserves"]),
            r_sol=int(r_sol), supply=int(fields["token_total_supply"]), complete=bool(fields["complete"]),
            has_creator=bool(creator) and creator != _DEFAULT_PUBKEY, mayhem=bool(fields.get("is_mayhem_mode") or False),
        )

    async def bonding_curve_state(self, curve_address: str) -> tuple[CurveState, dict[str, Any]] | None:
        data = await self.rpc.get_account_info(curve_address)
        if data is None:
            return None
        dec = self.codec.decode_account(data)
        if dec is None or dec[0] != "BondingCurve":
            raise ValueError("account is not a BondingCurve")
        return self.curve_state_from_account(dec[1]), dec[1]

    async def fee_config(self, fee_config_address: str, fees_codec: IdlCodec) -> dict[str, Any] | None:
        data = await self.rpc.get_account_info(fee_config_address)
        if data is None:
            return None
        dec = fees_codec.decode_account(data)
        return dec[1] if dec else None

    async def onchain_metadata(self, mint: str) -> tuple[str, str, str] | None:
        """(name, symbol, uri) from the Token-2022 metadata extension or the Metaplex account."""
        res = await self.rpc.call("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        value = (res or {}).get("value") or {}
        parsed = ((value.get("data") or {}).get("parsed") or {}) if isinstance(value.get("data"), dict) else {}
        for ext in (parsed.get("info") or {}).get("extensions") or []:
            if ext.get("extension") == "tokenMetadata":
                st = ext.get("state") or {}
                return st.get("name", ""), st.get("symbol", ""), st.get("uri", "")
        from pumpfun_hft.core.pda import metaplex_metadata_pda

        data = await self.rpc.get_account_info(metaplex_metadata_pda(mint))
        return decode_metaplex_metadata(data) if data else None

    async def token_metadata(self, mint: str, uri: str | None = None, name: str = "", symbol: str = "",
                             recent_names: set[str] | None = None) -> TokenMetadata:
        if uri is None:
            onchain = await self.onchain_metadata(mint)
            if onchain is None:
                return TokenMetadata(mint=mint, error="no metadata account")
            name, symbol, uri = onchain
        if self.metadata is None:
            return TokenMetadata(mint=mint, name=name, symbol=symbol, uri=uri)
        return await self.metadata.fetch(mint, name, symbol, uri, recent_names)

    async def holder_analytics(self, mint: str, curve_address: str | None = None, token_program: str | None = None) -> HolderStats:
        """Top-holder concentration from ``getTokenLargestAccounts`` (excludes the curve's vault)."""
        supply = await self.rpc.get_token_supply(mint)
        largest = await self.rpc.get_token_largest_accounts(mint)
        excluded = False
        curve_vault = None
        if curve_address:
            try:
                from pumpfun_hft.core.pda import TOKEN_PROGRAM, associated_token_address

                curve_vault = associated_token_address(curve_address, mint, token_program or TOKEN_PROGRAM)
            except ImportError:
                curve_vault = None
        accounts: list[tuple[str, int]] = []
        for acc in largest:
            if curve_vault and acc["address"] == curve_vault:
                excluded = True
                continue
            accounts.append((acc["address"], int(acc["amount"])))
        top10 = sum(a for _, a in accounts[:10])
        shares = [a / supply for _, a in accounts] if supply else []
        return HolderStats(mint, supply, accounts, 100.0 * top10 / supply if supply else 0.0,
                           float(sum(s * s for s in shares)), excluded)

    async def analytics(self, mint: str) -> dict[str, Any]:
        if self.frontend is None:
            raise RuntimeError("pump.fun frontend API disabled (network.pumpfun_frontend.enabled=false)")
        return await self.frontend.coin(mint)

    async def profile(self, address: str) -> dict[str, Any]:
        if self.frontend is None:
            raise RuntimeError("pump.fun frontend API disabled (network.pumpfun_frontend.enabled=false)")
        return await self.frontend.user(address)
