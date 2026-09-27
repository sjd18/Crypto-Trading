"""Core protocol layer.

Purpose
    Protocol-exact primitives shared by research, simulation and live trading.

Architecture
    config.py   typed YAML configuration (no hidden defaults) + secrets from .env
    types.py    Event/Order/Fill/Signal model and the canonical Parquet event schema
    curve.py    exact integer Pump bonding-curve math (verified against the official SDK)
    amm.py      constant-product AMM math (PumpSwap canonical pools; template for other DEXs)
    idl.py      IDL-driven Borsh codec (events, accounts, instructions; tolerant of old layouts)
    events.py   log / self-CPI event extraction and normalisation to the flat Event schema
    clock.py    simulated, wall and replay clocks behind one interface; a virtual-time asyncio loop
    idl/        bundled Anchor IDLs (pump, pump_amm, pump_fees) from pump-fun/pump-public-docs

Data flow
    raw tx / logs -> events.EventDecoder -> Event rows -> collectors / replay
    Event reserves -> curve.CurveState -> quotes (BuyQuote / SellQuote) -> simulator & live quotes

Inputs / Outputs
    Inputs: IDL JSON, config YAML, raw transaction JSON or log lines.
    Outputs: Event records, exact integer quotes and fee breakdowns.

Example
    >>> from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
    >>> bc = BondingCurve(FeeSchedule([FeeTier(0, 95, 30)]), 1073000000000000, 30000000000, 793100000000000, 10**15)
    >>> st = bc.new_state()
    >>> bc.buy_tokens_for_sol(st, 1_000_000_000) > 0
    True
"""
