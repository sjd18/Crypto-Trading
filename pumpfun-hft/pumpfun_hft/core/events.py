"""Extraction and normalisation of Pump / PumpSwap events from Solana transactions.

Two on-chain channels carry Anchor events and both are supported:

1. **Program logs** — ``Program data: <base64>`` lines (``emit!``). Available from
   ``logsSubscribe`` (live) and ``getTransaction`` (historical). Lines are attributed to the
   program on top of the invoke stack so events of unrelated programs are ignored. Logs can be
   truncated by the runtime ("Log truncated"), which is detected and reported.
2. **Self-CPI events** — inner instructions whose data starts with Anchor's
   ``EVENT_IX_TAG`` (``emit_cpi!``). Only available historically, but immune to log truncation.
   When a transaction carries CPI events they take precedence over logs (no double counting).

Normalisation maps decoded IDL structs to the flat :class:`~pumpfun_hft.core.types.Event`
schema. PumpSwap reserves are converted to post-trade values according to
``protocol.amm_event_reserves``; pools are mapped back to their base mint via
``CompletePumpAmmMigrationEvent`` / ``CreatePoolEvent`` (see :class:`PoolRegistry`).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pumpfun_hft.core.idl import IdlCodec, load_codecs
from pumpfun_hft.core.types import Event, EventKind
from pumpfun_hft.utils.base58 import b58decode

PROGRAM_DATA = "Program data: "
EVENT_IX_TAG = bytes.fromhex("e445a52e51cb9a1d")  # anchor EVENT_IX_TAG (u64 LE)


@dataclass(slots=True)
class DecodedEvent:
    program: str
    name: str
    fields: dict[str, Any]


@dataclass(slots=True)
class LogParseResult:
    events: list[DecodedEvent] = field(default_factory=list)
    truncated: bool = False
    failed: bool = False


class PoolRegistry:
    """Maps PumpSwap pool addresses to base mints (learned from migration/pool events)."""

    def __init__(self) -> None:
        self.pool_to_mint: dict[str, str] = {}

    def learn(self, pool: str | None, mint: str | None) -> None:
        if pool and mint:
            self.pool_to_mint[pool] = mint

    def mint_for(self, pool: str | None) -> str | None:
        return self.pool_to_mint.get(pool) if pool else None


class EventDecoder:
    """Decode and normalise Pump + PumpSwap events.

    Example::

        dec = EventDecoder.from_settings(settings)
        rows = dec.events_from_logs(logs, slot=..., signature=..., ts_ms=..., seq=0)
    """

    def __init__(
        self,
        codecs: dict[str, IdlCodec],
        pump_program_id: str,
        amm_program_id: str,
        amm_reserves: str = "pre",
        pools: PoolRegistry | None = None,
    ) -> None:
        self.codecs = codecs
        self.pump_id = pump_program_id
        self.amm_id = amm_program_id
        self.amm_reserves = amm_reserves
        self.pools = pools or PoolRegistry()
        if pump_program_id not in codecs:
            raise ValueError("pump IDL not loaded")

    @classmethod
    def from_idl_dir(cls, idl_dir: str | Path, pump_program_id: str, amm_program_id: str, amm_reserves: str = "pre") -> EventDecoder:
        return cls(load_codecs(idl_dir), pump_program_id, amm_program_id, amm_reserves)

    @classmethod
    def from_settings(cls, settings: Any) -> EventDecoder:
        p = settings.protocol
        return cls.from_idl_dir(settings.paths.resolve("idl_dir"), p.pump_program_id, p.pump_amm_program_id, p.amm_event_reserves)

    # ------------------------------------------------------------------ raw extraction
    def parse_logs(self, logs: list[str]) -> LogParseResult:
        """Decode ``Program data:`` lines, attributing each to the invoking program."""
        res = LogParseResult()
        stack: list[str] = []
        for line in logs:
            if line.startswith("Program "):
                if line.startswith(PROGRAM_DATA):
                    if not stack:
                        continue
                    codec = self.codecs.get(stack[-1])
                    if codec is None:
                        continue
                    try:
                        raw = base64.b64decode(line[len(PROGRAM_DATA):])
                    except ValueError:
                        continue
                    dec = codec.decode_event(raw)
                    if dec is not None:
                        res.events.append(DecodedEvent(stack[-1], dec[0], dec[1]))
                    continue
                parts = line.split(" ")
                if len(parts) >= 3 and parts[2] == "invoke":
                    stack.append(parts[1])
                elif len(parts) >= 3 and parts[2] in ("success", "failed:") or (len(parts) >= 3 and parts[2].startswith("failed")):
                    if parts[2].startswith("failed"):
                        res.failed = True
                    if stack and stack[-1] == parts[1]:
                        stack.pop()
            elif line == "Log truncated":
                res.truncated = True
        return res

    def parse_cpi_events(self, tx: dict[str, Any]) -> list[DecodedEvent]:
        """Decode self-CPI events from a ``getTransaction`` (json encoding) result."""
        meta = tx.get("meta") or {}
        msg = (tx.get("transaction") or {}).get("message") or {}
        keys: list[str] = list(msg.get("accountKeys") or [])
        loaded = meta.get("loadedAddresses") or {}
        keys += list(loaded.get("writable") or []) + list(loaded.get("readonly") or [])
        keys = [k if isinstance(k, str) else k.get("pubkey", "") for k in keys]
        out: list[DecodedEvent] = []
        for group in meta.get("innerInstructions") or []:
            for ix in group.get("instructions") or []:
                idx = ix.get("programIdIndex")
                if idx is None or idx >= len(keys):
                    continue
                program = keys[idx]
                codec = self.codecs.get(program)
                if codec is None:
                    continue
                data_field = ix.get("data")
                if not data_field:
                    continue
                try:
                    raw = b58decode(data_field)
                except ValueError:
                    continue
                if raw[:8] != EVENT_IX_TAG:
                    continue
                dec = codec.decode_event(raw[8:])
                if dec is not None:
                    out.append(DecodedEvent(program, dec[0], dec[1]))
        return out

    # ------------------------------------------------------------------ normalisation
    def normalise(self, ev: DecodedEvent, *, slot: int, seq: int, ev_idx: int, ts_ms: int,
                  signature: str | None, block_time: int | None = None, block_hash: str | None = None,
                  sol_usd: float | None = None) -> Event | None:
        """Map a decoded IDL event to the canonical flat Event (None if not market-relevant)."""
        f = ev.fields
        base = dict(slot=slot, seq=seq, ev_idx=ev_idx, ts_ms=ts_ms, block_time=block_time,
                    signature=signature, block_hash=block_hash, sol_usd=sol_usd)
        if ev.program == self.pump_id:
            if ev.name == "TradeEvent":
                return Event(kind=EventKind.TRADE.value, mint=f["mint"], user=f["user"], is_buy=f["is_buy"],
                             sol_amount=f["sol_amount"], token_amount=f["token_amount"],
                             v_sol=f["virtual_sol_reserves"], v_tok=f["virtual_token_reserves"],
                             r_sol=f.get("real_sol_reserves"), r_tok=f.get("real_token_reserves"),
                             fee_bps=f.get("fee_basis_points"), fee=f.get("fee"),
                             creator_fee_bps=f.get("creator_fee_basis_points"), creator_fee=f.get("creator_fee"),
                             creator=f.get("creator"), ix_name=f.get("ix_name"), **base)
            if ev.name == "CreateEvent":
                return Event(kind=EventKind.CREATE.value, mint=f["mint"], user=f["user"], creator=f.get("creator") or f["user"],
                             name=f["name"], symbol=f["symbol"], uri=f["uri"], bonding_curve=f["bonding_curve"],
                             v_sol=f.get("virtual_sol_reserves"), v_tok=f.get("virtual_token_reserves"),
                             r_tok=f.get("real_token_reserves"), r_sol=0 if f.get("real_token_reserves") is not None else None,
                             token_amount=f.get("token_total_supply"), token_program=f.get("token_program"), **base)
            if ev.name == "CompleteEvent":
                return Event(kind=EventKind.COMPLETE.value, mint=f["mint"], user=f["user"],
                             bonding_curve=f["bonding_curve"], **base)
            if ev.name == "CompletePumpAmmMigrationEvent":
                self.pools.learn(f.get("pool"), f.get("mint"))
                return Event(kind=EventKind.MIGRATE.value, mint=f["mint"], user=f["user"], pool=f.get("pool"),
                             token_amount=f.get("mint_amount"), sol_amount=f.get("sol_amount"),
                             v_tok=f.get("mint_amount"), v_sol=f.get("sol_amount"),
                             r_tok=f.get("mint_amount"), r_sol=f.get("sol_amount"),
                             fee=f.get("pool_migration_fee"), bonding_curve=f.get("bonding_curve"), **base)
            return None
        if ev.program == self.amm_id:
            if ev.name == "CreatePoolEvent":
                self.pools.learn(f.get("pool"), f.get("base_mint"))
                return Event(kind=EventKind.POOL_CREATE.value, mint=f.get("base_mint"), user=f.get("creator"),
                             pool=f.get("pool"), v_tok=f.get("pool_base_amount"), v_sol=f.get("pool_quote_amount"),
                             r_tok=f.get("pool_base_amount"), r_sol=f.get("pool_quote_amount"),
                             creator=f.get("coin_creator"), **base)
            if ev.name in ("BuyEvent", "SellEvent"):
                is_buy = ev.name == "BuyEvent"
                pool = f.get("pool")
                mint = self.pools.mint_for(pool)
                base_res = f.get("pool_base_token_reserves") or 0
                quote_res = f.get("pool_quote_token_reserves") or 0
                vq = f.get("virtual_quote_reserves") or 0
                lp_fee = f.get("lp_fee") or 0
                if is_buy:
                    base_amt = f.get("base_amount_out") or 0
                    quote_amt = f.get("quote_amount_in") or 0
                    if self.amm_reserves == "pre":
                        base_res, quote_res = base_res - base_amt, quote_res + quote_amt + lp_fee
                else:
                    base_amt = f.get("base_amount_in") or 0
                    quote_amt = f.get("quote_amount_out") or 0
                    if self.amm_reserves == "pre":
                        base_res, quote_res = base_res + base_amt, quote_res - (quote_amt - lp_fee)
                return Event(kind=(EventKind.AMM_BUY if is_buy else EventKind.AMM_SELL).value, mint=mint,
                             user=f.get("user"), is_buy=is_buy, pool=pool, sol_amount=quote_amt, token_amount=base_amt,
                             v_tok=base_res, v_sol=quote_res + vq, r_tok=base_res, r_sol=quote_res,
                             fee_bps=f.get("protocol_fee_basis_points"), fee=f.get("protocol_fee"),
                             creator_fee_bps=f.get("coin_creator_fee_basis_points"), creator_fee=f.get("coin_creator_fee"),
                             lp_fee_bps=f.get("lp_fee_basis_points"), lp_fee=lp_fee, creator=f.get("coin_creator"),
                             ix_name=f.get("ix_name"), **base)
        return None

    # ------------------------------------------------------------------ convenience
    def events_from_logs(self, logs: list[str], *, slot: int, seq: int, ts_ms: int, signature: str | None,
                         block_time: int | None = None, sol_usd: float | None = None) -> tuple[list[Event], LogParseResult]:
        res = self.parse_logs(logs)
        out: list[Event] = []
        for i, d in enumerate(res.events):
            e = self.normalise(d, slot=slot, seq=seq, ev_idx=i, ts_ms=ts_ms, signature=signature,
                               block_time=block_time, sol_usd=sol_usd)
            if e is not None:
                out.append(e)
        return out, res

    def events_from_transaction(self, tx: dict[str, Any], *, seq: int, ts_ms: int | None = None,
                                block_hash: str | None = None, sol_usd: float | None = None) -> list[Event]:
        """Events from a ``getTransaction`` result; prefers CPI events over (truncatable) logs."""
        meta = tx.get("meta") or {}
        slot = int(tx.get("slot") or 0)
        block_time = tx.get("blockTime")
        sigs = ((tx.get("transaction") or {}).get("signatures")) or [None]
        ts = ts_ms if ts_ms is not None else (int(block_time) * 1000 if block_time else 0)
        decoded = self.parse_cpi_events(tx)
        if not decoded:
            decoded = self.parse_logs(meta.get("logMessages") or []).events
        out: list[Event] = []
        for i, d in enumerate(decoded):
            e = self.normalise(d, slot=slot, seq=seq, ev_idx=i, ts_ms=ts, signature=sigs[0], block_time=block_time,
                               block_hash=block_hash, sol_usd=sol_usd)
            if e is not None:
                out.append(e)
        return out
