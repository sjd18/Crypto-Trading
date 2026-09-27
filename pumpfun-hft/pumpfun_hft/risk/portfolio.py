"""Portfolio ledger with exact lamport accounting (shared by backtests, paper and live).

Cash and costs are integers in lamports. Every fill (successful or failed) updates cash;
positions keep an average-cost basis that includes protocol/creator/LP/platform fees and
transaction costs (base fee, priority fee, Jito tip) of their entries. Token-account rent is
tracked separately because it is refunded when the account is closed after a full exit.

Round trips (flat -> open -> flat) become :class:`TradeRecord` rows with PnL, return, R
multiple (vs the initial risk budget), MAE/MFE (from liquidation-value marks), holding time,
exit reason and a cost breakdown. Costs of transactions that never produced a position (e.g. a
failed entry) are booked as ``unattributed_costs`` so equity stays exact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Fill, OrderStatus, Side


@dataclass(slots=True)
class Position:
    mint: str
    strategy: str
    creator: str | None
    sector: str
    trade_id: int
    entry_ms: int
    tokens: int = 0
    cost_lamports: int = 0          # cost basis of the tokens still held
    total_cost: int = 0             # all entry costs of the round trip (for returns)
    proceeds: int = 0               # net proceeds of exits so far
    realized: int = 0               # realised PnL of partial exits
    initial_risk: int = 0           # lamports at risk at entry (cost * stop fraction)
    entry_price: float = 0.0        # average all-in entry price (SOL/token)
    first_entry_price: float = 0.0
    peak_price: float = 0.0         # highest mark price since entry (trailing stops)
    mae: float = 0.0                # most adverse economic return seen
    mfe: float = 0.0                # most favourable economic return seen
    last_value: int = 0             # latest liquidation value of the held tokens
    last_mark_ms: int = 0
    n_fills: int = 0
    n_adds: int = 0
    n_partial_exits: int = 0
    tp_hits: int = 0
    rent_paid: int = 0
    fees: int = 0                   # protocol + creator + lp + platform
    tx_costs: int = 0               # network + priority + tip
    slippage_cost: float = 0.0      # SOL lost vs decision quotes
    exit_overrides: dict[str, float] | None = None
    exit_reason: str = ""

    def economic_return(self, value: int | None = None) -> float:
        v = self.last_value if value is None else value
        return (self.proceeds + v - self.total_cost) / self.total_cost if self.total_cost > 0 else 0.0


@dataclass(slots=True)
class TradeRecord:
    trade_id: int
    mint: str
    strategy: str
    creator: str | None
    sector: str
    entry_ms: int
    exit_ms: int
    cost_sol: float
    proceeds_sol: float
    pnl_sol: float
    ret: float
    r_multiple: float
    mae: float
    mfe: float
    hold_s: float
    exit_reason: str
    n_fills: int
    fees_sol: float
    tx_costs_sol: float
    slippage_sol: float
    entry_price: float
    exit_price: float


class Portfolio:
    """Cash, positions, closed trades and an equity history.

    Example::

        pf = Portfolio(10 * LAMPORTS_PER_SOL)
        pf.apply_fill(fill, strategy="momentum_ignition", creator=c, sector="ai", stop_frac=0.25)
        pf.mark(mint, value_lamports, price, ts)      # liquidation-value marks
        pf.equity_lamports()
    """

    def __init__(self, initial_lamports: int, close_account_refund: bool = True) -> None:
        self.initial = int(initial_lamports)
        self.cash = int(initial_lamports)
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeRecord] = []
        self.fills: list[Fill] = []
        self.equity_curve: list[tuple[int, float, float, float, int]] = []
        self.unattributed_costs = 0
        self.close_account_refund = close_account_refund
        self._trade_seq = 0
        self._equity_hist: deque[tuple[int, int]] = deque()
        self.peak_equity = int(initial_lamports)

    # ------------------------------------------------------------------ queries
    def position(self, mint: str) -> Position | None:
        p = self.positions.get(mint)
        return p if p is not None and p.tokens > 0 else None

    def equity_lamports(self) -> int:
        return self.cash + sum(p.last_value + (p.rent_paid if self.close_account_refund else 0) for p in self.positions.values())

    def exposure_lamports(self) -> int:
        return sum(p.last_value for p in self.positions.values())

    def cost_exposure_lamports(self) -> int:
        return sum(p.cost_lamports for p in self.positions.values())

    def exposure_by(self, attr: str, key: str | None) -> int:
        return sum(p.cost_lamports for p in self.positions.values() if getattr(p, attr) == key)

    @property
    def n_positions(self) -> int:
        return sum(1 for p in self.positions.values() if p.tokens > 0)

    # ------------------------------------------------------------------ updates
    def apply_fill(self, fill: Fill, *, strategy: str, creator: str | None, sector: str, stop_frac: float,
                   exit_overrides: dict[str, float] | None = None) -> TradeRecord | None:
        """Book a fill. Returns a TradeRecord when a round trip closes."""
        self.fills.append(fill)
        self.cash += fill.sol_delta
        pos = self.positions.get(fill.mint)
        tx = fill.network_fee + fill.priority_fee + fill.jito_tip
        fees = fill.protocol_fee + fill.creator_fee + fill.lp_fee + fill.platform_fee
        if not fill.filled:
            if pos is not None:
                pos.tx_costs += tx
                pos.cost_lamports += -fill.sol_delta if fill.sol_delta < 0 else 0
                pos.total_cost += -fill.sol_delta if fill.sol_delta < 0 else 0
            else:
                self.unattributed_costs += -fill.sol_delta if fill.sol_delta < 0 else 0
            return None
        if fill.side is Side.BUY:
            paid = -fill.sol_delta - fill.rent  # rent tracked separately (refundable)
            if pos is None or pos.tokens == 0:
                self._trade_seq += 1
                pos = Position(fill.mint, strategy, creator, sector, self._trade_seq, fill.land_ms,
                               first_entry_price=fill.price, peak_price=fill.price, exit_overrides=exit_overrides)
                self.positions[fill.mint] = pos
                pos.initial_risk = int(paid * stop_frac)
            else:
                pos.n_adds += 1
                pos.initial_risk += int(paid * stop_frac)
            pos.tokens += fill.token_amount
            pos.cost_lamports += paid
            pos.total_cost += paid
            pos.rent_paid += fill.rent
            # SOL per whole token = (lamports / 1e9) / (base units / 1e6)
            pos.entry_price = pos.cost_lamports / pos.tokens / 1000.0 if pos.tokens else 0.0
        else:
            if pos is None or pos.tokens <= 0:
                self.unattributed_costs += max(0, -fill.sol_delta)
                return None
            sold = min(fill.token_amount, pos.tokens)
            cost_part = pos.cost_lamports * sold // pos.tokens
            proceeds = fill.sol_delta + fill.rent  # exclude rent refund from trading proceeds
            pos.realized += proceeds - cost_part
            pos.proceeds += proceeds
            pos.cost_lamports -= cost_part
            pos.tokens -= sold
            pos.last_value = pos.last_value * pos.tokens // (pos.tokens + sold) if pos.tokens + sold else 0
            if pos.tokens > 0:
                pos.n_partial_exits += 1
        pos.n_fills += 1
        pos.fees += fees
        pos.tx_costs += tx
        if fill.decision_price > 0 and fill.token_amount > 0:  # adverse slippage in SOL (negative = price improvement)
            pos.slippage_cost += fill.slippage_bps / 1e4 * fill.price * fill.token_amount / 1e6
        if fill.side is Side.SELL and pos.tokens == 0:
            return self._close(pos, fill.land_ms, fill.reason, fill.price)
        return None

    def _close(self, pos: Position, ts: int, reason: str, exit_price: float) -> TradeRecord:
        pnl = pos.proceeds - pos.total_cost
        rec = TradeRecord(
            trade_id=pos.trade_id, mint=pos.mint, strategy=pos.strategy, creator=pos.creator, sector=pos.sector,
            entry_ms=pos.entry_ms, exit_ms=ts, cost_sol=pos.total_cost / LAMPORTS_PER_SOL,
            proceeds_sol=pos.proceeds / LAMPORTS_PER_SOL, pnl_sol=pnl / LAMPORTS_PER_SOL,
            ret=pnl / pos.total_cost if pos.total_cost else 0.0,
            r_multiple=pnl / pos.initial_risk if pos.initial_risk else 0.0,
            mae=min(pos.mae, pos.economic_return(0)), mfe=max(pos.mfe, pos.economic_return(0)),
            hold_s=(ts - pos.entry_ms) / 1000.0, exit_reason=reason or pos.exit_reason, n_fills=pos.n_fills,
            fees_sol=pos.fees / LAMPORTS_PER_SOL, tx_costs_sol=pos.tx_costs / LAMPORTS_PER_SOL,
            slippage_sol=pos.slippage_cost, entry_price=pos.first_entry_price, exit_price=exit_price,
        )
        self.trades.append(rec)
        del self.positions[pos.mint]
        return rec

    def mark(self, mint: str, value_lamports: int, price: float, ts: int) -> None:
        """Mark a position at its liquidation value; tracks MAE / MFE and the peak price."""
        pos = self.positions.get(mint)
        if pos is None:
            return
        pos.last_value = int(value_lamports)
        pos.last_mark_ms = ts
        r = pos.economic_return()
        if r < pos.mae:
            pos.mae = r
        if r > pos.mfe:
            pos.mfe = r
        if price > pos.peak_price:
            pos.peak_price = price

    def record_equity(self, ts: int) -> int:
        eq = self.equity_lamports()
        self.equity_curve.append((ts, eq / LAMPORTS_PER_SOL, self.cash / LAMPORTS_PER_SOL,
                                  self.exposure_lamports() / LAMPORTS_PER_SOL, self.n_positions))
        self._equity_hist.append((ts, eq))
        while self._equity_hist and self._equity_hist[0][0] < ts - 86_400_000:
            self._equity_hist.popleft()
        self.peak_equity = max(self.peak_equity, eq)
        return eq

    def equity_at_or_before(self, ts: int) -> int | None:
        best = None
        for t, e in self._equity_hist:
            if t <= ts:
                best = e
            else:
                break
        return best

    # ------------------------------------------------------------------ export
    def trades_frame(self) -> pl.DataFrame:
        if not self.trades:
            return pl.DataFrame(schema={f: pl.Float64 for f in TradeRecord.__slots__} | {  # type: ignore[attr-defined]
                "trade_id": pl.Int64, "mint": pl.Utf8, "strategy": pl.Utf8, "creator": pl.Utf8, "sector": pl.Utf8,
                "entry_ms": pl.Int64, "exit_ms": pl.Int64, "exit_reason": pl.Utf8, "n_fills": pl.Int64})
        return pl.DataFrame([asdict(t) for t in self.trades])

    def fills_frame(self) -> pl.DataFrame:
        if not self.fills:
            return pl.DataFrame(schema={"order_id": pl.Int64, "mint": pl.Utf8, "status": pl.Utf8})
        return pl.DataFrame([f.to_dict() for f in self.fills], infer_schema_length=None)

    def equity_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.equity_curve, schema=["ts_ms", "equity_sol", "cash_sol", "exposure_sol", "n_positions"], orient="row")

    def summary(self) -> dict[str, Any]:
        return {"cash_sol": self.cash / LAMPORTS_PER_SOL, "equity_sol": self.equity_lamports() / LAMPORTS_PER_SOL,
                "positions": self.n_positions, "closed_trades": len(self.trades),
                "unattributed_costs_sol": self.unattributed_costs / LAMPORTS_PER_SOL,
                "failed_fills": sum(1 for f in self.fills if f.status in (OrderStatus.FAILED,))}
