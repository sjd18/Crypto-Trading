"""Live execution against a mocked chain (RPC) and Metis: build, sign, send, confirm, rebroadcast,
retry only after expiry, program-aware error mapping and fill reconstruction."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest

pytest.importorskip("solders")
from solders.hash import Hash  # noqa: E402
from solders.instruction import AccountMeta, Instruction  # noqa: E402
from solders.message import MessageV0  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402
from solders.signature import Signature  # noqa: E402
from solders.transaction import VersionedTransaction  # noqa: E402

from pumpfun_hft.api.http import build_client  # noqa: E402
from pumpfun_hft.api.metis import CapabilityUnavailable, MetisClient  # noqa: E402
from pumpfun_hft.api.rpc import SolanaRpcClient  # noqa: E402
from pumpfun_hft.backtester.execution_sim import ExecutionSimulator  # noqa: E402
from pumpfun_hft.core.config import Secrets  # noqa: E402
from pumpfun_hft.core.curve import BondingCurve  # noqa: E402
from pumpfun_hft.core.events import EventDecoder  # noqa: E402
from pumpfun_hft.core.types import Action, Event, EventKind, Order, OrderStatus, OrderType, Side, Urgency  # noqa: E402
from pumpfun_hft.execution.gateway import LiveGateway  # noqa: E402
from pumpfun_hft.execution.infra import BlockhashCache, ConfirmationTracker, PriorityFeeEstimator  # noqa: E402
from pumpfun_hft.execution.signer import WalletSigner  # noqa: E402
from pumpfun_hft.features.market import MarketState  # noqa: E402
from pumpfun_hft.tests.conftest import make_settings  # noqa: E402
from pumpfun_hft.utils.latency import LatencyTracker  # noqa: E402

PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
RPC_URL = "https://rpc.test/"
MINT = str(Pubkey.new_unique())


class FakeChain:
    """Answers Solana JSON-RPC and Metis HTTP calls for one scripted transaction."""

    def __init__(self, signer: WalletSigner, decoder: EventDecoder, *, land_after_polls: int | None = 2,
                 valid_for_checks: int = 10**6, err: Any = None, tokens: int = 34_000_000_000_000, spent: int = 1_000_000_000) -> None:
        self.signer, self.decoder = signer, decoder
        self.land_after_polls, self.valid_for_checks, self.err = land_after_polls, valid_for_checks, err
        self.tokens, self.spent = tokens, spent
        self.sent: list[str] = []
        self.status_polls = 0
        self.validity_checks = 0
        self.swap_requests: list[dict] = []
        self.blockhash = str(Hash.new_unique())

    # ------------------------------------------------------------------ Metis
    def unsigned_swap_tx(self) -> bytes:
        payer = self.signer.keypair.pubkey()
        ix = Instruction(Pubkey.from_string(PUMP), bytes([102, 6, 61, 18, 1, 218, 235, 234]), [AccountMeta(payer, True, True)])
        msg = MessageV0.try_compile(payer, [ix], [], Hash.from_string(self.blockhash))
        return bytes(VersionedTransaction.populate(msg, [Signature.default()]))

    # ------------------------------------------------------------------ chain state
    def landed(self) -> bool:
        return self.land_after_polls is not None and self.status_polls >= self.land_after_polls

    def transaction(self, sig: str) -> dict:
        wallet = self.signer.pubkey
        codec = self.decoder.codecs[PUMP]
        fields = {n: 0 for n in codec.struct_field_names("TradeEvent")}
        fields.update(mint=MINT, sol_amount=987_654_321, token_amount=self.tokens, is_buy=True, user=wallet, timestamp=1,
                      virtual_sol_reserves=31_000_000_000, virtual_token_reserves=1_039_000_000_000_000, real_sol_reserves=987_654_321,
                      real_token_reserves=759_100_000_000_000, fee_recipient=str(Pubkey.new_unique()), fee_basis_points=95,
                      fee=9_382_716, creator=str(Pubkey.new_unique()), creator_fee_basis_points=30, creator_fee=2_962_963,
                      track_volume=False, ix_name="buy_exact_sol_in")
        raw = codec.encode_event("TradeEvent", fields, truncate_after="ix_name")
        fee = 5_000 + 70_000
        rent = 2_039_280
        pre = 5_000_000_000
        return {"slot": 777, "blockTime": 1_788_220_800,
                "transaction": {"signatures": [sig], "message": {"accountKeys": [wallet, PUMP, MINT]}},
                "meta": {"err": self.err, "fee": fee, "preBalances": [pre, 1, 1],
                         "postBalances": [pre - (0 if self.err else self.spent + rent) - fee, 1, 1],
                         "preTokenBalances": [],
                         "postTokenBalances": [] if self.err else [{"owner": wallet, "mint": MINT, "uiTokenAmount": {"amount": str(self.tokens)}}],
                         "logMessages": [f"Program {PUMP} invoke [1]", f"Program data: {base64.b64encode(raw).decode()}",
                                         f"Program {PUMP} success"],
                         "innerInstructions": []}}

    # ------------------------------------------------------------------ transport
    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.jupiterapi.com":
            body = json.loads(request.content)
            self.swap_requests.append(body)
            return httpx.Response(200, json={"tx": base64.b64encode(self.unsigned_swap_tx()).decode()})
        req = json.loads(request.content)
        method, params = req["method"], req.get("params") or []
        if method == "sendTransaction":
            self.sent.append(params[0])
            sig = str(VersionedTransaction.from_bytes(base64.b64decode(params[0])).signatures[0])
            return self._ok(req, sig)
        if method == "getSignatureStatuses":
            self.status_polls += 1
            value = [({"slot": 777, "confirmations": 1, "err": self.err, "confirmationStatus": "confirmed"} if self.landed() else None)
                     for _ in params[0]]
            return self._ok(req, {"context": {"slot": 800}, "value": value})
        if method == "isBlockhashValid":
            self.validity_checks += 1
            return self._ok(req, {"context": {"slot": 800}, "value": self.validity_checks <= self.valid_for_checks})
        if method == "getTransaction":
            return self._ok(req, self.transaction(params[0]) if self.landed() else None)
        return self._ok(req, None)

    @staticmethod
    def _ok(req: dict, result: Any) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": req["id"], "result": result})


def _settings():
    return make_settings(**{
        "network.rpc.rate_limit": {"rps": 10_000.0, "burst": 10_000}, "network.metis.rate_limit": {"rps": 10_000.0, "burst": 10_000},
        "live.confirm_poll_ms": 10, "live.rebroadcast_ms": 30, "live.confirm_timeout_ms": 5_000, "network.retry.max_attempts": 1})


def _gateway(chain_kwargs: dict | None = None):
    s = _settings()
    signer = WalletSigner.generate()
    decoder = EventDecoder.from_settings(s)
    chain = FakeChain(signer, decoder, **(chain_kwargs or {}))
    transport = httpx.MockTransport(chain)
    lat = LatencyTracker()
    rpc = SolanaRpcClient(build_client("rpc", "", s.network.rpc.rate_limit, s.network, 5.0, lat, None, transport), RPC_URL)
    metis = MetisClient.from_settings(s, Secrets(), lat, transport=transport)
    curve = BondingCurve.from_config(s.protocol.curve, s.protocol.curve_fee_tiers)
    market = MarketState(curve)
    st0 = curve.new_state()
    market.on_event(Event(EventKind.CREATE.value, 1, 0, 0, 0, mint=MINT, user="dev", creator="dev", v_sol=st0.v_sol, v_tok=st0.v_tok,
                          r_sol=0, r_tok=st0.r_tok, token_amount=st0.supply))
    sim = ExecutionSimulator(s, curve, None, market, np.random.default_rng(0))  # type: ignore[arg-type]
    gw = LiveGateway(s, rpc, metis, signer, BlockhashCache(rpc, 2_000),
                     PriorityFeeEstimator(None, s.priority_fee, s.network.metis.priority_fee_levels, [PUMP]),
                     ConfirmationTracker(rpc, s.live.confirm_poll_ms, "confirmed", lat), decoder, sim, lat)
    order = Order(1, MINT, Side.BUY, Action.BUY, OrderType.MARKET, 0, "t", "test", Urgency.NORMAL, sol_budget=1_000_000_000,
                  slippage_bps=1500, compute_units=140_000)
    order.quote_tokens, avg = sim.quote_buy(MINT, order.sol_budget)
    order.meta["quote_avg"] = avg
    return gw, chain, order, signer, lat


async def test_happy_path_fill_is_reconstructed_from_the_transaction() -> None:
    gw, chain, order, signer, lat = _gateway({"land_after_polls": 2})
    fill = await gw.execute(order)
    assert fill.status is OrderStatus.FILLED
    assert fill.token_amount == chain.tokens
    assert fill.sol_delta == -(chain.spent + 2_039_280 + 75_000)          # exact wallet delta
    assert fill.rent == 2_039_280 and fill.network_fee == 5_000 and fill.priority_fee == 70_000
    assert fill.protocol_fee == 9_382_716 and fill.creator_fee == 2_962_963
    assert fill.platform_fee == chain.spent - (987_654_321 + 9_382_716 + 2_962_963)  # router fee = the unexplained remainder
    assert fill.price == pytest.approx(chain.spent / chain.tokens / 1000)
    assert chain.swap_requests[0]["wallet"] == signer.pubkey and chain.swap_requests[0]["type"] == "BUY"
    assert chain.swap_requests[0]["inAmount"] == str(order.sol_budget)
    assert len(set(chain.sent)) == 1                                        # one signed transaction
    assert lat.count("exec.quote_to_submit") == 1


async def test_rebroadcasts_the_same_bytes_while_the_blockhash_is_valid() -> None:
    gw, chain, order, *_ = _gateway({"land_after_polls": 12})
    fill = await gw.execute(order)
    assert fill.status is OrderStatus.FILLED
    assert len(chain.sent) > 1 and len(set(chain.sent)) == 1                # re-sent, never re-signed


async def test_expired_only_after_the_blockhash_is_invalid_and_never_resigned() -> None:
    gw, chain, order, *_ = _gateway({"land_after_polls": None, "valid_for_checks": 3})
    fill = await gw.execute(order)
    assert fill.status is OrderStatus.EXPIRED and fill.failure == "blockhash_expired"
    assert chain.validity_checks == 4                                       # stopped exactly when it became invalid
    assert len(set(chain.sent)) == 1                                        # the gateway itself never double-sends


async def test_program_error_maps_to_slippage_and_charges_fees() -> None:
    err = {"InstructionError": [0, {"Custom": 6042}]}                      # pump: BuySlippageBelowMinTokensOut
    gw, chain, order, *_ = _gateway({"land_after_polls": 1, "err": err})
    fill = await gw.execute(order)
    assert fill.status is OrderStatus.FAILED and fill.failure == "slippage"
    assert fill.network_fee == 75_000 and fill.sol_delta == -75_000


def test_error_codes_are_resolved_per_program() -> None:
    gw, *_ = _gateway()
    amm = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
    err = {"InstructionError": [1, {"Custom": 6004}]}
    assert gw._error_reason(err, ["ComputeBudget111111111111111111111111111111", amm]) == "slippage"   # ExceededSlippage
    assert gw._error_reason(err, ["ComputeBudget111111111111111111111111111111", PUMP]) == "MintDoesNotMatchBondingCurve"
    assert gw._error_reason({"InstructionError": [0, {"Custom": 6001}]}, ["JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"]) == "slippage"
    assert gw._error_reason("BlockhashNotFound") == "BlockhashNotFound"


async def test_public_metis_refuses_paid_endpoints() -> None:
    s = _settings()
    m = MetisClient.from_settings(s, Secrets(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(CapabilityUnavailable):
        await m.pump_quote("BUY", MINT, 1_000)
    with pytest.raises(CapabilityUnavailable):
        await m.pump_swap_instructions("w", "BUY", MINT, 1_000)
    await m.aclose()


def test_signer_accepts_both_key_formats() -> None:
    w = WalletSigner.generate()
    raw = bytes(w.keypair)
    assert WalletSigner.from_secret(json.dumps(list(raw))).pubkey == w.pubkey
    assert WalletSigner.from_secret(str(w.keypair)).pubkey == w.pubkey
    with pytest.raises(ValueError):
        WalletSigner.from_secret("")
