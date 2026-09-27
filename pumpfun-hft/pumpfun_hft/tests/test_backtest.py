"""Backtest correctness: causality, zero look-ahead, exact accounting, determinism, execution model."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pumpfun_hft.backtester.engine import BacktestEngine
from pumpfun_hft.backtester.execution_sim import ExecutionSimulator
from pumpfun_hft.backtester.replay import DataSource
from pumpfun_hft.core.amm import ConstantProductAmm
from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
from pumpfun_hft.core.types import (
    LAMPORTS_PER_SOL,
    Action,
    Event,
    EventKind,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Signal,
    Urgency,
    events_to_frame,
    hold,
)
from pumpfun_hft.features.market import MarketState
from pumpfun_hft.strategies.base import STRATEGY_REGISTRY, Strategy, StrategyContext, StrategyParams, register
from pumpfun_hft.tests.conftest import make_settings

SOL = LAMPORTS_PER_SOL


# --------------------------------------------------------------------------- helpers
class Tape:
    """Builds exact event sequences for hand-made scenarios."""

    def __init__(self, settings) -> None:
        self.curve = BondingCurve.from_config(settings.protocol.curve, settings.protocol.curve_fee_tiers)
        self.states: dict[str, object] = {}
        self.rows: list[Event] = []
        self.slot = 1000
        self.n = 0

    def _sig(self) -> str:
        self.n += 1
        return f"sig{self.n}"

    def create(self, mint: str, t: int, creator: str = "creator1") -> None:
        st = self.curve.new_state()
        self.states[mint] = st
        self.slot += 1
        self.rows.append(Event(EventKind.CREATE.value, self.slot, 0, 0, t, signature=self._sig(), mint=mint, user=creator,
                               creator=creator, name="Tape Coin", symbol="TAPE", v_sol=st.v_sol, v_tok=st.v_tok, r_sol=0,
                               r_tok=st.r_tok, token_amount=st.supply))

    def buy(self, mint: str, t: int, sol: float, user: str = "u") -> None:
        st = self.states[mint]
        q = self.curve.buy_with_budget(st, int(sol * SOL))
        st = self.curve.apply_buy(st, q)
        self.states[mint] = st
        self._trade(mint, t, True, q.sol_curve, q.tokens, q.protocol_fee, q.creator_fee, user)

    def sell(self, mint: str, t: int, tokens: int, user: str = "u") -> None:
        st = self.states[mint]
        q = self.curve.sell_proceeds_for_tokens(st, tokens)
        st = self.curve.apply_sell(st, q)
        self.states[mint] = st
        self._trade(mint, t, False, q.sol_curve, tokens, q.protocol_fee, q.creator_fee, user)

    def _trade(self, mint, t, is_buy, sol, tokens, pf, cf, user) -> None:
        st = self.states[mint]
        self.slot += 1
        self.rows.append(Event(EventKind.TRADE.value, self.slot, 0, 0, t, signature=self._sig(), mint=mint, user=user, is_buy=is_buy,
                               sol_amount=sol, token_amount=tokens, v_sol=st.v_sol, v_tok=st.v_tok, r_sol=st.r_sol, r_tok=st.r_tok,
                               fee_bps=95, fee=pf, creator_fee_bps=30, creator_fee=cf, creator="creator1"))

    def frame(self) -> pl.DataFrame:
        return events_to_frame(self.rows)


@pytest.fixture(scope="module", autouse=True)
def _test_strategies():
    """Strategies used only by these tests (registered once, removed afterwards)."""

    @register
    class BuyBigPrint(Strategy):
        """Buys right after any single buy of >= ``min_sol`` — the canary for latency causality."""

        name = "test_buy_big_print"

        class Params(StrategyParams):
            min_sol: float

        def generate_signal(self, ctx: StrategyContext) -> Signal:
            ev = ctx.event
            if ctx.has_position or not ctx.is_trade or not ev.is_buy or (ev.sol_amount or 0) < self.p.min_sol * SOL:
                return hold()
            return Signal(Action.BUY, 90, "big print", self.name, size_sol=0.5)

    yield
    STRATEGY_REGISTRY.pop("test_buy_big_print", None)


def _bare_settings(**extra):
    """No overlay, no failures, fixed size, no cost gate: isolates the execution path."""
    return make_settings(**{
        "strategy.exit_overlay": None, "strategy.params.test_buy_big_print": {"min_sol": 5.0},
        "simulation.failures.drop_prob": 0.0, "simulation.failures.landed_fail_prob": 0.0,
        "simulation.latency.spike_prob": 0.0, "simulation.outages.rate_per_hour": 0.0,
        "sizing.method": "fixed", "sizing.fixed_sol": 0.5, "sizing.confidence_scaling": False, **extra})


# --------------------------------------------------------------------------- causality
def test_order_decided_on_a_print_executes_after_it(settings) -> None:
    """The strategy sees a 10-SOL buy at t; its own buy must execute on the post-print curve, never before."""
    s = _bare_settings()
    tape = Tape(s)
    t0 = 1_788_220_800_000
    tape.create("MINT1", t0)
    tape.buy("MINT1", t0 + 1_000, 0.2)
    pre_print_price = tape.states["MINT1"].price
    tape.buy("MINT1", t0 + 5_000, 10.0, user="whale")
    post_print_price = tape.states["MINT1"].price
    for i in range(20):  # quiet tape afterwards
        tape.buy("MINT1", t0 + 6_000 + 1_000 * i, 0.05, user=f"r{i}")
    res = BacktestEngine(s, DataSource(frame=tape.frame()), ["test_buy_big_print"], seed=1).run()
    buys = res.fills.filter((pl.col("side") == "buy") & pl.col("status").is_in(["filled", "partial"]))
    assert buys.height == 1
    b = buys.row(0, named=True)
    assert b["decision_ms"] == t0 + 5_000
    assert b["land_ms"] > b["decision_ms"]
    assert b["price"] > post_print_price > pre_print_price * 1.5


def test_decisions_before_t_do_not_depend_on_data_after_t(settings, events, metadata) -> None:
    """Zero look-ahead: truncating the future leaves every earlier decision and fill unchanged."""
    s = settings
    t_cut = int(events["ts_ms"].quantile(0.6))
    full = BacktestEngine(s, DataSource(frame=events), ["momentum_ignition", "smart_money"], metadata=metadata, seed=3).run()
    part = BacktestEngine(s, DataSource(frame=events.filter(pl.col("ts_ms") <= t_cut)), ["momentum_ignition", "smart_money"],
                          metadata=metadata, seed=3).run()
    horizon = t_cut - 60_000  # orders decided just before the cut may land after it in the full run only
    cols = ["ts_ms", "mint", "strategy", "action", "reason", "outcome"]
    a = full.signals.filter(pl.col("ts_ms") < horizon).select(cols)
    b = part.signals.filter(pl.col("ts_ms") < horizon).select(cols)
    assert a.height > 20
    assert a.equals(b)
    fcols = ["order_id", "mint", "side", "status", "decision_ms", "land_ms", "token_amount", "sol_delta"]
    fa = full.fills.filter(pl.col("confirm_ms") < horizon).select(fcols)
    fb = part.fills.filter(pl.col("confirm_ms") < horizon).select(fcols)
    assert fa.height > 5 and fa.equals(fb)


def test_every_fill_lands_after_its_decision(backtest_result) -> None:
    f = backtest_result.fills
    landed = f.filter(pl.col("status").is_in(["filled", "partial", "failed"]))
    assert (landed["land_ms"] > landed["decision_ms"]).all()
    assert (f["confirm_ms"] >= f["land_ms"]).all()


def test_first_entries_never_beat_the_historical_price_at_landing(backtest_result, events) -> None:
    """A buy (fees + own impact) can never be cheaper than the last historical price before it landed."""
    f = backtest_result.fills.filter((pl.col("side") == "buy") & pl.col("status").is_in(["filled", "partial"]))
    tr = events.filter(pl.col("kind") == "trade").select("mint", "ts_ms", (pl.col("v_sol") / pl.col("v_tok") / 1000).alias("px"))
    checked = 0
    for row in f.head(60).iter_rows(named=True):
        hist = tr.filter((pl.col("mint") == row["mint"]) & (pl.col("ts_ms") <= row["land_ms"]))
        if hist.is_empty():
            continue
        assert row["price"] >= hist["px"][-1] * (1 - 1e-9)
        checked += 1
    assert checked > 10


# --------------------------------------------------------------------------- accounting
def test_ledger_reconciles_to_the_lamport(backtest_result) -> None:
    r = backtest_result
    m = r.metrics
    final = round(m["final_equity_sol"] * SOL)
    expected = round(r.initial_capital_sol * SOL) + round(r.trades["pnl_sol"].sum() * SOL) - round(m["unattributed_costs_sol"] * SOL)
    assert abs(final - expected) <= 2
    cash_moves = int(r.fills["sol_delta"].sum())
    assert abs(round(m["final_equity_sol"] * SOL) - (round(r.initial_capital_sol * SOL) + cash_moves)) <= 2  # flat at the end


def test_costs_are_charged(backtest_result) -> None:
    f = backtest_result.fills.filter(pl.col("status").is_in(["filled", "partial"]))
    assert (f["protocol_fee"] > 0).all()
    assert (f["network_fee"] > 0).all() and (f["priority_fee"] > 0).all()
    buys = f.filter(pl.col("side") == "buy")
    assert (buys["platform_fee"] > 0).all()  # Mode A public endpoint fee


def test_same_seed_same_result_different_seed_different_result(settings, events, metadata) -> None:
    runs = [BacktestEngine(settings, DataSource(frame=events), ["momentum_ignition"], metadata=metadata, seed=sd).run()
            for sd in (11, 11, 12)]
    a, b, c = (r.fills.select("order_id", "status", "land_ms", "sol_delta") for r in runs)
    assert a.equals(b)
    assert not a.equals(c)


@pytest.mark.parametrize("mode", ["trade", "tick", "candle"])
def test_replay_modes_run(settings, events, metadata, mode) -> None:
    s = make_settings(**{"backtest.replay_mode": mode})
    res = BacktestEngine(s, DataSource(frame=events), ["momentum_ignition"], metadata=metadata, seed=5).run()
    assert res.n_events == events.height
    assert np.isfinite(res.metrics["final_equity_sol"])


def test_higher_latency_never_improves_fill_timing(settings, events, metadata) -> None:
    fast = BacktestEngine(settings, DataSource(frame=events), ["smart_money"], metadata=metadata, seed=9, latency_scale=0.5).run()
    slow = BacktestEngine(settings, DataSource(frame=events), ["smart_money"], metadata=metadata, seed=9, latency_scale=3.0).run()
    assert slow.diagnostics["latency_ms"]["p50"] > 2 * fast.diagnostics["latency_ms"]["p50"]


# --------------------------------------------------------------------------- execution simulator units
@pytest.fixture()
def sim_env():
    s = _bare_settings()
    tape = Tape(s)
    t0 = 1_788_220_800_000
    tape.create("M", t0)
    tape.buy("M", t0 + 100, 2.0)
    market = MarketState(tape.curve)
    for ev in tape.rows:
        market.on_event(ev)
    flat = s.protocol.amm_flat_fees
    amm = ConstantProductAmm(FeeSchedule.from_config(s.protocol.amm_fee_tiers), FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))
    sim = ExecutionSimulator(s, tape.curve, amm, market, np.random.default_rng(0))
    return s, tape, market, sim, t0


def _order(side: Side, *, budget: int = 0, tokens: int = 0, otype: OrderType = OrderType.MARKET, limit: float | None = None,
           slip: int = 1500, jito: bool = False, oid: int = 1) -> Order:
    return Order(oid, "M", side, Action.BUY if side is Side.BUY else Action.EXIT, otype, 0, "t", "r", Urgency.NORMAL, sol_budget=budget,
                 token_amount=tokens, limit_price=limit, slippage_bps=slip, priority_micro_lamports=100_000, compute_units=140_000,
                 use_jito=jito, jito_tip_lamports=100_000 if jito else 0)


def _prep(sim, order, t):
    if order.side is Side.BUY:
        order.quote_tokens, avg = sim.quote_buy(order.mint, order.sol_budget)
    else:
        order.quote_sol, avg = sim.quote_sell(order.mint, order.token_amount)
    order.meta.update(quote_avg=avg, submit_ms=t, landed_fail=False, confirm_delay=500)
    return order


def test_market_buy_then_sell_round_trip(sim_env) -> None:
    s, tape, market, sim, t0 = sim_env
    spot0 = sim.spot_price("M")
    buy = sim.execute(_prep(sim, _order(Side.BUY, budget=SOL), t0), t0 + 700, 1)
    assert buy.status is OrderStatus.FILLED and buy.token_amount > 0
    assert -buy.sol_delta == SOL + buy.network_fee + buy.priority_fee + buy.rent  # the whole budget is spent
    assert sim.spot_price("M") > spot0  # persistent own impact
    expected_back = sim.quote_sell("M", buy.token_amount)[0]
    so = _prep(sim, _order(Side.SELL, tokens=buy.token_amount, oid=2), t0 + 800)
    so.meta.update(pos_tokens=buy.token_amount, rent_paid=buy.rent)  # closing sell: the token account rent comes back
    sell = sim.execute(so, t0 + 1500, 2)
    assert sell.status is OrderStatus.FILLED
    assert sell.sol_delta + sell.network_fee + sell.priority_fee - buy.rent == expected_back
    assert sim.spot_price("M") == pytest.approx(spot0)  # impact unwinds when we exit
    assert sell.sol_delta + buy.sol_delta < 0  # a round trip always costs money


def test_slippage_failure_when_the_price_runs_away(sim_env) -> None:
    s, tape, market, sim, t0 = sim_env
    order = _prep(sim, _order(Side.BUY, budget=SOL, slip=100), t0)  # 1 % tolerance
    tape.buy("M", t0 + 300, 15.0, user="front")  # a large buy lands first
    market.on_event(tape.rows[-1])
    fill = sim.execute(order, t0 + 700, 3)
    assert fill.status is OrderStatus.FAILED and fill.failure == "slippage"
    assert fill.sol_delta == -(fill.network_fee + fill.priority_fee)  # a failed tx still pays fees
    assert fill.token_amount == 0


def test_fok_limit_not_met_and_ioc_partial(sim_env) -> None:
    s, tape, market, sim, t0 = sim_env
    spot = sim.spot_price("M")
    fok = sim.execute(_prep(sim, _order(Side.BUY, budget=5 * SOL, otype=OrderType.FOK, limit=spot * 1.01), t0), t0 + 700, 4)
    assert fok.status is OrderStatus.FAILED and fok.failure == "limit_not_met"
    ioc = sim.execute(_prep(sim, _order(Side.BUY, budget=5 * SOL, otype=OrderType.IOC, limit=spot * 1.05, oid=5), t0), t0 + 700, 5)
    assert ioc.status is OrderStatus.PARTIAL
    assert 0 < ioc.token_amount and ioc.price <= spot * 1.05 * 1.01  # (price includes the platform fee)


def test_partial_fill_when_the_curve_sells_out(settings) -> None:
    s = _bare_settings()
    tape = Tape(s)
    t0 = 1_788_220_800_000
    tape.create("M", t0)
    tape.buy("M", t0 + 100, 80.0)  # ~84 SOL completes the curve: little is left
    market = MarketState(tape.curve)
    for ev in tape.rows:
        market.on_event(ev)
    flat = s.protocol.amm_flat_fees
    amm = ConstantProductAmm(FeeSchedule.from_config(s.protocol.amm_fee_tiers), FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))
    sim = ExecutionSimulator(s, tape.curve, amm, market, np.random.default_rng(0))
    left = market.get("M").curve.r_tok
    fill = sim.execute(_prep(sim, _order(Side.BUY, budget=20 * SOL, slip=5000), t0), t0 + 700, 9)
    assert fill.status is OrderStatus.PARTIAL
    assert fill.token_amount == left
    assert -fill.sol_delta < 20 * SOL  # only what the remaining supply cost


def test_landed_failure_and_jito_drop(sim_env) -> None:
    s, tape, market, sim, t0 = sim_env
    o = _prep(sim, _order(Side.BUY, budget=SOL), t0)
    o.meta["landed_fail"] = True
    fill = sim.execute(o, t0 + 700, 1)
    assert fill.status is OrderStatus.FAILED and fill.failure == "tx_error" and fill.sol_delta < 0
    j = _prep(sim, _order(Side.BUY, budget=SOL, jito=True, oid=7), t0)
    j.meta["landed_fail"] = True
    dropped = sim.execute(j, t0 + 700, 1)
    assert dropped.status is OrderStatus.DROPPED and dropped.sol_delta == 0  # atomic bundle: no fee, no tip


def test_submit_models_drops_and_expiry(settings) -> None:
    s = make_settings(**{"simulation.failures.drop_prob": 1.0})
    tape = Tape(s)
    tape.create("M", 0)
    market = MarketState(tape.curve)
    market.on_event(tape.rows[0])
    sim = ExecutionSimulator(s, tape.curve, None, market, np.random.default_rng(0))  # type: ignore[arg-type]
    kind, t, fill = sim.submit(_order(Side.BUY, budget=SOL), 0, 0.0)
    assert kind == "result" and fill.status is OrderStatus.DROPPED
    assert t == s.simulation.failures.blockhash_ttl_ms  # detected only when the blockhash expires
    s2 = make_settings(**{"simulation.failures.drop_prob": 0.0, "simulation.latency.inclusion_ms.median_ms": 120_000.0})
    sim2 = ExecutionSimulator(s2, tape.curve, None, market, np.random.default_rng(0))  # type: ignore[arg-type]
    kind2, _, fill2 = sim2.submit(_order(Side.BUY, budget=SOL), 0, 0.0)
    assert kind2 == "result" and fill2.status is OrderStatus.EXPIRED


def test_nothing_trades_while_a_completed_curve_awaits_migration(settings) -> None:
    s = _bare_settings()
    tape = Tape(s)
    tape.create("M", 0)
    tape.buy("M", 100, 200.0)  # completes the curve
    market = MarketState(tape.curve)
    for ev in tape.rows:
        market.on_event(ev)
    market.on_event(Event(EventKind.COMPLETE.value, tape.slot + 1, 0, 1, 200, mint="M", user="u"))
    sim = ExecutionSimulator(s, tape.curve, None, market, np.random.default_rng(0))  # type: ignore[arg-type]
    assert not sim.tradable("M")
    fill = sim.execute(_order(Side.SELL, tokens=10**9), 300, 1)
    assert fill.status is OrderStatus.FAILED and fill.failure == "curve_complete_awaiting_migration"
