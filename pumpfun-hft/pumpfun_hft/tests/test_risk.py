"""Sizing, pre-trade risk limits, circuit breakers and position management."""

from __future__ import annotations

import pytest

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Action, Fill, OrderStatus, Side, Signal, Venue
from pumpfun_hft.risk.breakers import CircuitBreakers
from pumpfun_hft.risk.engine import RiskEngine
from pumpfun_hft.risk.portfolio import Portfolio, Position
from pumpfun_hft.risk.position_manager import PositionManager
from pumpfun_hft.risk.sizing import Sizer, kelly_fraction
from pumpfun_hft.tests.conftest import make_settings

SOL = LAMPORTS_PER_SOL


def _sig(conf: float = 100.0, **kw) -> Signal:
    return Signal(Action.BUY, conf, "test", "t", **kw)


def _sizer(**overrides) -> Sizer:
    s = make_settings(**overrides)
    return Sizer(s.sizing, s.risk.limits, s.position, s.backtest)


def _size(sz: Sizer, sig: Signal, **kw) -> int:
    base = dict(equity_lamports=10 * SOL, cash_lamports=10 * SOL, exposure_lamports=0, token_exposure_lamports=0, vol=0.1,
                closed_returns=[])
    base.update(kw)
    return sz.size(sig, **base)


def test_fixed_risk_sizing_uses_the_stop_distance() -> None:
    sz = _sizer(**{"sizing.method": "fixed_risk", "sizing.risk_per_trade_sol": 0.05, "sizing.confidence_scaling": False,
                   "position.stop_loss_pct": 25.0})
    assert _size(sz, _sig()) == int(0.2 * SOL)
    tight = _sig(exit_overrides={"stop_loss_pct": 10.0})
    assert _size(sz, tight) == int(0.5 * SOL)


def test_confidence_scaling_and_minimum_order() -> None:
    sz = _sizer(**{"sizing.method": "fixed", "sizing.fixed_sol": 0.4, "sizing.confidence_scaling": True})
    assert _size(sz, _sig(50.0)) == int(0.2 * SOL)
    assert _size(sz, _sig(1.0)) == 0  # 0.004 SOL is below the minimum order size


def test_kelly_sizing_falls_back_until_enough_history() -> None:
    sz = _sizer(**{"sizing.method": "kelly", "sizing.kelly_min_trades": 10, "sizing.kelly_fraction": 0.5,
                   "sizing.kelly_cap_frac": 0.5, "sizing.confidence_scaling": False, "sizing.max_equity_frac": 1.0,
                   "risk.limits.max_position_per_token_sol": 100.0, "risk.limits.max_exposure_sol": 100.0})
    history = [0.2, -0.1] * 10  # W = 0.5, R = 2 -> f* = 0.25
    assert kelly_fraction(history) == pytest.approx(0.25)
    assert _size(sz, _sig(), closed_returns=history) == int(0.125 * 10 * SOL)
    fallback = _size(sz, _sig(), closed_returns=history[:4])
    assert fallback == int(sz.cfg.risk_per_trade_sol / 0.25 * SOL)


def test_caps_exposure_cash_and_price_impact() -> None:
    sz = _sizer(**{"sizing.method": "fixed", "sizing.fixed_sol": 5.0, "sizing.confidence_scaling": False,
                   "sizing.max_equity_frac": 1.0, "risk.limits.max_position_per_token_sol": 3.0,
                   "risk.limits.max_exposure_sol": 4.0, "sizing.max_impact_bps": 300.0})
    assert _size(sz, _sig()) == 3 * SOL  # per-token cap
    assert _size(sz, _sig(), exposure_lamports=2 * SOL) == 2 * SOL  # exposure headroom
    assert _size(sz, _sig(), cash_lamports=int(1.05 * SOL)) == int(1.0 * SOL)  # cash minus reserve
    capped = _size(sz, _sig(), impact_bps_fn=lambda lam: lam / SOL * 200.0)  # 200 bps per SOL
    assert capped == pytest.approx(1.5 * SOL, rel=1e-3)


def _portfolio_with_position(pf: Portfolio, mint: str, cost: int, creator: str = "c", sector: str = "ai") -> None:
    fill = Fill(1, mint, Side.BUY, Action.BUY, OrderStatus.FILLED, "t", "r", 0, 0, 1, 1, 1, Venue.CURVE,
                token_amount=10**12, sol_amount=cost, sol_delta=-cost, price=cost / 10**12 / 1000)
    pf.apply_fill(fill, strategy="t", creator=creator, sector=sector, stop_frac=0.25)


def test_risk_limits() -> None:
    s = make_settings(**{"risk.limits.max_open_positions": 2, "risk.limits.max_creator_exposure_sol": 0.6,
                         "risk.limits.max_orders_per_token_per_min": 2, "risk.limits.daily_loss_sol": 1.0})
    pf = Portfolio(10 * SOL)
    risk = RiskEngine(s.risk, pf, int(0.01 * SOL))
    now = 1_788_220_800_000
    risk.on_equity(now, pf.equity_lamports())
    ok = risk.check_entry("A", "c1", "ai", int(0.5 * SOL), now)
    assert ok.ok and ok.lamports == int(0.5 * SOL)
    _portfolio_with_position(pf, "A", int(0.5 * SOL), creator="c1")
    clipped = risk.check_entry("B", "c1", "ai", int(0.5 * SOL), now)
    assert clipped.ok and clipped.lamports == int(0.1 * SOL)  # creator exposure 0.6 - 0.5
    _portfolio_with_position(pf, "B", int(0.1 * SOL), creator="c2")
    assert risk.check_entry("C", "c3", "ai", SOL // 10, now).reason == "max open positions"
    # per-token order rate
    risk2 = RiskEngine(s.risk, Portfolio(10 * SOL), int(0.01 * SOL))
    for _ in range(2):
        assert risk2.check_entry("X", "c", "ai", SOL // 10, now).ok
    assert risk2.check_entry("X", "c", "ai", SOL // 10, now + 1).reason == "token order rate"
    assert risk2.check_entry("X", "c", "ai", SOL // 10, now + 61_000).ok
    # daily loss
    pf3 = Portfolio(10 * SOL)
    risk3 = RiskEngine(s.risk, pf3, int(0.01 * SOL))
    risk3.on_equity(now, pf3.equity_lamports())
    pf3.cash -= int(1.2 * SOL)
    assert risk3.check_entry("Y", "c", "ai", SOL // 10, now).reason == "daily loss limit"
    assert risk3.check_exit("Y", now).ok  # exits are never blocked


def test_circuit_breakers_trip_and_cool_down() -> None:
    s = make_settings(**{"risk.breakers.failed_swaps_max": 3, "risk.breakers.cooldown_s": 60.0,
                         "risk.breakers.drawdown_pct": 20.0, "risk.breakers.rpc_latency_p90_ms": 500.0})
    b = CircuitBreakers(s.risk.breakers)
    t = 1_000_000
    for i in range(3):
        b.on_fill(False, True, 0.0, t + i)
    assert b.halted(t + 10) and "failed_swaps" in b.halted(t + 10)
    assert b.halted(t + 2 + 60_000) is None  # cooldown elapsed
    b.on_equity(7.5, 10.0, t)
    assert b.flatten_requested and b.status(t + 1)["drawdown"]["active"]
    b2 = CircuitBreakers(s.risk.breakers)
    for _ in range(40):
        b2.on_rpc_latency(900.0, t)
    assert "rpc_latency" in (b2.halted(t + 1) or "")
    assert b2.status(t + 1)["rpc_latency"]["trips"] == 1


def _pos(total_cost: int = SOL, **kw) -> Position:
    p = Position("M", "t", "c", "ai", 1, 0, tokens=10**12, cost_lamports=total_cost, total_cost=total_cost, peak_price=1.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_position_manager_rules() -> None:
    s = make_settings(**{"position.stop_loss_pct": 25.0, "position.max_hold_s": 600.0,
                         "position.trailing_stop": {"enabled": True, "activation_pct": 30.0, "trail_pct": 20.0}})
    pm = PositionManager(s.position)
    assert pm.check(_pos(), int(0.74 * SOL), 1.0, 1_000).action is Action.EXIT  # stop loss
    assert pm.check(_pos(), int(0.9 * SOL), 1.0, 601_000).action is Action.EXIT  # max hold
    tp1 = pm.check(_pos(), int(1.45 * SOL), 1.0, 1_000)
    assert tp1.action is Action.SCALE_OUT and tp1.size_frac == pytest.approx(0.4)
    last = pm.check(_pos(tp_hits=2), int(3.6 * SOL), 1.0, 1_000)
    assert last.action is Action.EXIT and "final take profit" in last.reason
    assert pm.check(_pos(tp_hits=1), int(0.99 * SOL), 1.0, 1_000).reason == "breakeven stop"
    trail = pm.check(_pos(mfe=0.35, peak_price=2.0), int(1.2 * SOL), 1.55, 1_000)
    assert trail.action is Action.EXIT and "trailing stop" in trail.reason
    assert pm.check(_pos(mfe=0.35, peak_price=2.0), int(1.2 * SOL), 1.7, 1_000) is None


def test_exit_overrides_replace_defaults() -> None:
    s = make_settings()
    pm = PositionManager(s.position)
    ov = {"stop_loss_pct": 5.0, "take_profit_pct": 10.0, "max_hold_s": 30.0}
    assert pm.check(_pos(exit_overrides=ov), int(0.94 * SOL), 1.0, 1_000).action is Action.EXIT
    assert "take profit" in pm.check(_pos(exit_overrides=ov), int(1.11 * SOL), 1.0, 1_000).reason
    assert pm.check(_pos(exit_overrides=ov), SOL, 1.0, 31_000).reason == "max hold time"


def test_portfolio_round_trip_accounting() -> None:
    pf = Portfolio(10 * SOL)
    buy = Fill(1, "M", Side.BUY, Action.BUY, OrderStatus.FILLED, "t", "r", 0, 0, 1_000, 1_500, 1, Venue.CURVE,
               token_amount=10**12, sol_amount=990_000_000, protocol_fee=9_405_000, creator_fee=2_970_000, network_fee=5_000,
               priority_fee=14_000, rent=2_039_280, sol_delta=-(1_000_000_000 + 5_000 + 14_000 + 2_039_280), price=1.0e-6)
    pf.apply_fill(buy, strategy="t", creator="c", sector="ai", stop_frac=0.25)
    assert pf.positions["M"].total_cost == 1_000_019_000  # rent is tracked separately (refundable)
    sell = Fill(2, "M", Side.SELL, Action.EXIT, OrderStatus.FILLED, "t", "exit", 5_000, 5_000, 6_000, 6_500, 2, Venue.CURVE,
                token_amount=10**12, network_fee=5_000, priority_fee=14_000, rent=-2_039_280,
                sol_delta=1_100_000_000 - 19_000 + 2_039_280, price=1.1e-6)
    rec = pf.apply_fill(sell, strategy="t", creator="c", sector="ai", stop_frac=0.25)
    assert rec is not None and not pf.positions
    assert rec.pnl_sol == pytest.approx((1_099_981_000 - 1_000_019_000) / SOL)
    assert pf.cash == 10 * SOL + buy.sol_delta + sell.sol_delta
    assert pf.cash - 10 * SOL == round(rec.pnl_sol * SOL)  # rent paid and refunded nets to zero
