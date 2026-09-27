"""Core domain types shared by research, simulation and live execution.

The flat :class:`Event` record is the canonical market-data unit: every collector emits it,
it is the Parquet schema (``EVENT_SCHEMA``), and every replay mode feeds it to the engine.
Amounts are integers in on-chain base units (lamports for SOL, 1e-6 for Pump tokens); prices
are floats in SOL per whole token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import polars as pl

LAMPORTS_PER_SOL = 1_000_000_000
TOKEN_UNIT = 1_000_000  # Pump tokens have 6 decimals
BPS = 10_000


class Action(StrEnum):
    """Strategy signal actions (``generate_signal`` return values)."""

    BUY = "BUY"            # open a new position (flat -> long)
    SELL = "SELL"          # discretionary close/reduce on signal reversal (normal urgency)
    HOLD = "HOLD"          # do nothing
    EXIT = "EXIT"          # risk exit: close everything now (exit slippage & priority)
    SCALE_IN = "SCALE_IN"  # add to an existing position (pyramiding)
    SCALE_OUT = "SCALE_OUT"  # partial profit-taking; position stays open


ACTION_PRIORITY: dict[Action, int] = {
    Action.EXIT: 5,
    Action.SELL: 4,
    Action.SCALE_OUT: 3,
    Action.SCALE_IN: 2,
    Action.BUY: 1,
    Action.HOLD: 0,
}


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"  # slippage-bounded swap
    LIMIT = "limit"    # client-side resting order, submitted when the limit is marketable
    IOC = "ioc"        # fill what is available at <= limit at landing time, cancel the rest
    FOK = "fok"        # fill the full size at <= limit or nothing


class OrderStatus(StrEnum):
    NEW = "new"
    RESTING = "resting"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    FAILED = "failed"        # landed on-chain but reverted (fees are charged)
    DROPPED = "dropped"      # never landed (no fees); detected at blockhash expiry
    EXPIRED = "expired"      # blockhash expired before landing / limit TTL elapsed
    REJECTED = "rejected"    # rejected before submission (risk, outage, rate limit)
    CANCELLED = "cancelled"


FILLED_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.PARTIAL})


class Urgency(StrEnum):
    NORMAL = "normal"
    HIGH = "high"
    EXIT = "exit"


class EventKind(StrEnum):
    CREATE = "create"
    TRADE = "trade"
    COMPLETE = "complete"
    MIGRATE = "migrate"
    AMM_BUY = "amm_buy"
    AMM_SELL = "amm_sell"
    POOL_CREATE = "pool_create"


class Venue(StrEnum):
    CURVE = "bonding_curve"
    AMM = "pumpswap"


# ----------------------------------------------------------------------------- events
EVENT_SCHEMA: dict[str, pl.DataType] = {
    "kind": pl.Utf8,
    "slot": pl.Int64,
    "seq": pl.Int32,           # order within the slot (transaction order)
    "ev_idx": pl.Int16,        # event order within the transaction
    "ts_ms": pl.Int64,         # event time, epoch ms (see collectors.slotclock)
    "block_time": pl.Int64,    # block time, epoch seconds (nullable)
    "signature": pl.Utf8,
    "block_hash": pl.Utf8,
    "mint": pl.Utf8,
    "user": pl.Utf8,
    "is_buy": pl.Boolean,
    "sol_amount": pl.Int64,    # curve/pool SOL amount excluding protocol fees (lamports)
    "token_amount": pl.Int64,  # token base units
    "v_sol": pl.Int64,         # post-event virtual SOL reserves (AMM: effective quote reserves)
    "v_tok": pl.Int64,         # post-event virtual token reserves (AMM: base reserves)
    "r_sol": pl.Int64,         # post-event real SOL reserves
    "r_tok": pl.Int64,         # post-event real token reserves
    "fee_bps": pl.Int64,       # protocol fee bps
    "fee": pl.Int64,
    "creator_fee_bps": pl.Int64,
    "creator_fee": pl.Int64,
    "lp_fee_bps": pl.Int64,
    "lp_fee": pl.Int64,
    "creator": pl.Utf8,
    "name": pl.Utf8,
    "symbol": pl.Utf8,
    "uri": pl.Utf8,
    "bonding_curve": pl.Utf8,
    "pool": pl.Utf8,
    "token_program": pl.Utf8,
    "ix_name": pl.Utf8,
    "sol_usd": pl.Float64,
}
EVENT_COLUMNS: tuple[str, ...] = tuple(EVENT_SCHEMA)
SORT_KEYS: tuple[str, ...] = ("slot", "seq", "ev_idx")


@dataclass(slots=True)
class Event:
    """One normalised on-chain market event (field order == ``EVENT_COLUMNS``)."""

    kind: str
    slot: int
    seq: int
    ev_idx: int
    ts_ms: int
    block_time: int | None = None
    signature: str | None = None
    block_hash: str | None = None
    mint: str | None = None
    user: str | None = None
    is_buy: bool | None = None
    sol_amount: int | None = None
    token_amount: int | None = None
    v_sol: int | None = None
    v_tok: int | None = None
    r_sol: int | None = None
    r_tok: int | None = None
    fee_bps: int | None = None
    fee: int | None = None
    creator_fee_bps: int | None = None
    creator_fee: int | None = None
    lp_fee_bps: int | None = None
    lp_fee: int | None = None
    creator: str | None = None
    name: str | None = None
    symbol: str | None = None
    uri: str | None = None
    bonding_curve: str | None = None
    pool: str | None = None
    token_program: str | None = None
    ix_name: str | None = None
    sol_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {c: getattr(self, c) for c in EVENT_COLUMNS}


def empty_events_frame() -> pl.DataFrame:
    """An empty DataFrame with the canonical event schema."""
    return pl.DataFrame(schema=EVENT_SCHEMA)


def events_to_frame(events: list[Event] | list[dict[str, Any]]) -> pl.DataFrame:
    """Build a canonical-schema DataFrame from Event objects or dicts."""
    if not events:
        return empty_events_frame()
    rows = [e.to_dict() if isinstance(e, Event) else e for e in events]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)


# ----------------------------------------------------------------------------- trading objects
@dataclass(slots=True)
class Signal:
    """Output of ``Strategy.generate_signal``. ``confidence`` is on a 0-100 scale."""

    action: Action
    confidence: float = 0.0
    reason: str = ""
    strategy: str = ""
    size_sol: float | None = None
    size_frac: float | None = None
    order_type: OrderType | None = None
    limit_price: float | None = None
    expected_return: float | None = None
    urgency: Urgency = Urgency.NORMAL
    exit_overrides: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 100.0:
            self.confidence = min(100.0, max(0.0, float(self.confidence)))

    @property
    def is_hold(self) -> bool:
        return self.action is Action.HOLD


def hold(reason: str = "") -> Signal:
    """Convenience HOLD signal."""
    return Signal(Action.HOLD, 0.0, reason)


@dataclass(slots=True)
class Order:
    """An order as decided by the strategy runtime (venue-agnostic)."""

    id: int
    mint: str
    side: Side
    action: Action
    order_type: OrderType
    created_ms: int
    strategy: str
    reason: str
    urgency: Urgency
    sol_budget: int = 0                 # buys: lamports to spend incl. protocol fees
    token_amount: int = 0               # sells: token base units
    limit_price: float | None = None    # SOL/token; buy = max avg price, sell = min avg price
    slippage_bps: int = 0
    priority_micro_lamports: int = 0
    compute_units: int = 0
    jito_tip_lamports: int = 0
    use_jito: bool = False
    decision_price: float = 0.0
    quote_tokens: int = 0
    quote_sol: int = 0
    confidence: float = 0.0
    attempt: int = 0
    parent_id: int | None = None
    status: OrderStatus = OrderStatus.NEW
    expires_ms: int | None = None
    exit_overrides: dict[str, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_exit(self) -> bool:
        return self.side is Side.SELL


@dataclass(slots=True)
class Fill:
    """Execution report for an order attempt (successful or not)."""

    order_id: int
    mint: str
    side: Side
    action: Action
    status: OrderStatus
    strategy: str
    reason: str
    decision_ms: int
    submit_ms: int
    land_ms: int
    confirm_ms: int
    slot: int
    venue: Venue
    token_amount: int = 0
    sol_amount: int = 0
    protocol_fee: int = 0
    creator_fee: int = 0
    lp_fee: int = 0
    platform_fee: int = 0
    network_fee: int = 0
    priority_fee: int = 0
    jito_tip: int = 0
    rent: int = 0
    sol_delta: int = 0
    price: float = 0.0
    decision_price: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    failure: str = ""
    attempt: int = 0
    signature: str = ""

    @property
    def filled(self) -> bool:
        return self.status in FILLED_STATUSES

    @property
    def total_costs(self) -> int:
        """All explicit costs in lamports (excludes price impact / slippage)."""
        return (
            self.protocol_fee + self.creator_fee + self.lp_fee + self.platform_fee
            + self.network_fee + self.priority_fee + self.jito_tip
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: (getattr(self, name).value if isinstance(getattr(self, name), StrEnum) else getattr(self, name))
                for name in self.__slots__}  # type: ignore[attr-defined]
