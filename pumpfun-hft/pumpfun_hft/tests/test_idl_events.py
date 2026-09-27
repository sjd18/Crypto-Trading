"""IDL codec and event extraction: round trips, old layouts, log attribution, CPI events, AMM reserves."""

from __future__ import annotations

import base64
import os

import pytest

from pumpfun_hft.core.events import EVENT_IX_TAG, EventDecoder
from pumpfun_hft.core.idl import IdlDecodeError, load_codecs
from pumpfun_hft.core.types import EventKind
from pumpfun_hft.utils.base58 import b58decode, b58encode

PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def pk() -> str:
    return b58encode(os.urandom(32))


@pytest.fixture(scope="module")
def codecs(settings):
    return load_codecs(settings.paths.resolve("idl_dir"))


@pytest.fixture()
def decoder(settings) -> EventDecoder:
    return EventDecoder.from_settings(settings)


def trade_fields(mint: str, user: str, creator: str, is_buy: bool = True) -> dict:
    return {"mint": mint, "sol_amount": 1_000_000_000, "token_amount": 34_000_000_000_000, "is_buy": is_buy, "user": user,
            "timestamp": 1_788_220_000, "virtual_sol_reserves": 31_000_000_000, "virtual_token_reserves": 1_039_000_000_000_000,
            "real_sol_reserves": 1_000_000_000, "real_token_reserves": 759_100_000_000_000, "fee_recipient": pk(),
            "fee_basis_points": 95, "fee": 9_500_000, "creator": creator, "creator_fee_basis_points": 30, "creator_fee": 3_000_000,
            "track_volume": True, "total_unclaimed_tokens": 0, "total_claimed_tokens": 0, "current_sol_volume": 0,
            "last_update_timestamp": 0, "ix_name": "buy_exact_sol_in"}


def logs_for(program: str, payloads: list[bytes], nested_foreign: bytes | None = None) -> list[str]:
    lines = [f"Program {program} invoke [1]", "Program log: Instruction: Buy"]
    if nested_foreign is not None:  # an unrelated program emitting data inside our instruction
        lines += [f"Program {TOKEN_PROGRAM} invoke [2]", f"Program data: {base64.b64encode(nested_foreign).decode()}",
                  f"Program {TOKEN_PROGRAM} success"]
    lines += [f"Program data: {base64.b64encode(p).decode()}" for p in payloads]
    lines += [f"Program {program} consumed 50000 of 200000 compute units", f"Program {program} success"]
    return lines


def test_trade_event_round_trip_with_old_layout(codecs) -> None:
    pump = codecs[PUMP]
    f = trade_fields(pk(), pk(), pk())
    raw = pump.encode_event("TradeEvent", f, truncate_after="ix_name")  # a 2025-era, shorter layout
    name, dec = pump.decode_event(raw)
    assert name == "TradeEvent"
    for k in ("mint", "sol_amount", "token_amount", "is_buy", "user", "virtual_sol_reserves", "creator_fee", "ix_name"):
        assert dec[k] == f[k]
    assert dec["mayhem_mode"] is None and dec["shareholders"] is None  # fields added later decode as None
    with pytest.raises(IdlDecodeError):
        pump.decode_struct("TradeEvent", raw, 8, tolerant=False)  # strict mode refuses the short layout


def test_events_from_logs_attributes_to_the_invoking_program(decoder, codecs) -> None:
    mint, user, creator = pk(), pk(), pk()
    raw = codecs[PUMP].encode_event("TradeEvent", trade_fields(mint, user, creator), truncate_after="ix_name")
    foreign = codecs[PUMP].encode_event("TradeEvent", trade_fields(pk(), pk(), pk()), truncate_after="ix_name")
    events, parsed = decoder.events_from_logs(logs_for(PUMP, [raw], nested_foreign=foreign), slot=10, seq=3, ts_ms=1_000,
                                              signature="sig1")
    assert not parsed.truncated and not parsed.failed
    assert len(events) == 1  # the token program's data line is not attributed to Pump
    ev = events[0]
    assert (ev.kind, ev.mint, ev.user, ev.is_buy, ev.creator) == (EventKind.TRADE.value, mint, user, True, creator)
    assert (ev.slot, ev.seq, ev.ev_idx, ev.signature) == (10, 3, 0, "sig1")
    assert ev.v_sol == 31_000_000_000 and ev.r_tok == 759_100_000_000_000 and ev.fee == 9_500_000


def test_truncated_logs_are_detected(decoder, codecs) -> None:
    raw = codecs[PUMP].encode_event("TradeEvent", trade_fields(pk(), pk(), pk()), truncate_after="ix_name")
    lines = logs_for(PUMP, [raw])[:3] + ["Log truncated"]
    _, parsed = decoder.events_from_logs(lines, slot=1, seq=0, ts_ms=0, signature="s")
    assert parsed.truncated


def test_failed_instruction_is_flagged(decoder) -> None:
    lines = [f"Program {PUMP} invoke [1]", f"Program {PUMP} failed: custom program error: 0x1772"]
    _, parsed = decoder.events_from_logs(lines, slot=1, seq=0, ts_ms=0, signature="s")
    assert parsed.failed


def test_cpi_events_take_precedence_over_logs(decoder, codecs) -> None:
    mint, user, creator = pk(), pk(), pk()
    raw = codecs[PUMP].encode_event("TradeEvent", trade_fields(mint, user, creator), truncate_after="ix_name")
    tx = {
        "slot": 42, "blockTime": 1_788_220_000,
        "transaction": {"signatures": ["sigCPI"], "message": {"accountKeys": [user, PUMP, mint]}},
        "meta": {"err": None, "logMessages": logs_for(PUMP, [raw]),  # the same event also appears in the logs
                 "innerInstructions": [{"index": 0, "instructions": [{"programIdIndex": 1, "data": b58encode(EVENT_IX_TAG + raw)}]}]},
    }
    events = decoder.events_from_transaction(tx, seq=-5)
    assert len(events) == 1  # no double counting
    assert events[0].mint == mint and events[0].slot == 42 and events[0].ts_ms == 1_788_220_000_000
    assert decoder.parse_cpi_events(tx)[0].name == "TradeEvent"


def test_create_event_normalisation(decoder, codecs) -> None:
    mint, user = pk(), pk()
    f = {"name": "Test Coin", "symbol": "TEST", "uri": "ipfs://x", "mint": mint, "bonding_curve": pk(), "user": user,
         "creator": user, "timestamp": 1, "virtual_token_reserves": 1_073_000_000_000_000, "virtual_sol_reserves": 30_000_000_000,
         "real_token_reserves": 793_100_000_000_000, "token_total_supply": 10**15, "token_program": TOKEN_PROGRAM}
    raw = codecs[PUMP].encode_event("CreateEvent", f, truncate_after="token_program")
    events, _ = decoder.events_from_logs(logs_for(PUMP, [raw]), slot=5, seq=0, ts_ms=7, signature="c")
    ev = events[0]
    assert (ev.kind, ev.name, ev.symbol, ev.creator, ev.r_sol) == (EventKind.CREATE.value, "Test Coin", "TEST", user, 0)
    assert ev.v_sol == 30_000_000_000


def test_amm_events_map_pool_to_mint_and_convert_to_post_trade_reserves(decoder, codecs) -> None:
    pump, amm = codecs[PUMP], codecs[AMM]
    mint, pool, user = pk(), pk(), pk()
    mig_fields = {"user": user, "mint": mint, "mint_amount": 206_900_000_000_000, "sol_amount": 84_990_359_454,
                  "pool_migration_fee": 15_000_001, "bonding_curve": pk(), "timestamp": 1, "pool": pool}
    mig_raw = pump.encode_event("CompletePumpAmmMigrationEvent", _fill_defaults(pump, "CompletePumpAmmMigrationEvent", mig_fields))
    events, _ = decoder.events_from_logs(logs_for(PUMP, [mig_raw]), slot=1, seq=0, ts_ms=0, signature="m")
    assert events[0].kind == EventKind.MIGRATE.value and decoder.pools.mint_for(pool) == mint

    buy = dict(pool=pool, user=user, base_amount_out=1_000_000_000_000, quote_amount_in=411_000_000,
               pool_base_token_reserves=206_900_000_000_000, pool_quote_token_reserves=84_990_359_454, lp_fee=820_000,
               protocol_fee=205_000, coin_creator_fee=1_230_000, lp_fee_basis_points=20, protocol_fee_basis_points=5,
               coin_creator_fee_basis_points=30)
    raw = amm.encode_event("BuyEvent", _fill_defaults(amm, "BuyEvent", buy))
    events, _ = decoder.events_from_logs(logs_for(AMM, [raw]), slot=2, seq=0, ts_ms=0, signature="b")
    ev = events[0]
    assert ev.kind == EventKind.AMM_BUY.value and ev.mint == mint and ev.pool == pool
    # pre-trade reserves in the event -> post-trade reserves in the row (LP fee stays in the pool)
    assert ev.r_tok == 206_900_000_000_000 - 1_000_000_000_000
    assert ev.r_sol == 84_990_359_454 + 411_000_000 + 820_000


def _fill_defaults(codec, name: str, value: dict) -> dict:
    """Give every field not in ``value`` a type-appropriate zero value."""
    out = dict(value)
    for f in codec.types[name]["fields"]:
        if f["name"] in out:
            continue
        t = f["type"]
        if t in ("pubkey", "publicKey"):
            out[f["name"]] = b58encode(bytes(32))
        elif t == "bool":
            out[f["name"]] = False
        elif t == "string":
            out[f["name"]] = ""
        elif isinstance(t, dict) and "vec" in t:
            out[f["name"]] = []
        elif isinstance(t, dict) and "option" in t:
            out[f["name"]] = None
        else:
            out[f["name"]] = 0
    return out


def test_base58_round_trip() -> None:
    raw = os.urandom(32)
    assert b58decode(b58encode(raw)) == raw
    assert b58decode(PUMP) and len(b58decode(PUMP)) == 32
