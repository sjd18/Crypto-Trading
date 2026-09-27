"""Bonding-curve math: exact parity with the official SDK and economic invariants."""

from __future__ import annotations

import json

import pytest

from pumpfun_hft.core.curve import (
    PRICE_SCALE,
    BondingCurve,
    CurveState,
    FeeBps,
    FeeSchedule,
    FeeTier,
    fee_amount,
    shift_state,
)
from pumpfun_hft.tests.conftest import FIXTURES

V_TOK, V_SOL, R_TOK, SUPPLY = 1_073_000_000_000_000, 30_000_000_000, 793_100_000_000_000, 10**15
GOLDEN = json.loads((FIXTURES / "curve_golden.json").read_text())


def _curve(kind: str) -> BondingCurve:
    tiers = [FeeTier(*t[:3]) for t in GOLDEN["fee_tiers"]] if kind == "tiered" else [FeeTier(0, *GOLDEN["global_fee"])]
    return BondingCurve(FeeSchedule(tiers), V_TOK, V_SOL, R_TOK, SUPPLY)


CURVES = {"global": _curve("global"), "tiered": _curve("tiered")}


def _state(case: dict) -> CurveState:
    v_tok, v_sol, r_tok, r_sol = (int(x) for x in case["state"])
    return CurveState(v_tok, v_sol, r_tok, r_sol, SUPPLY, False, bool(case["creator"]))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"{c['fn']}-{c['fee']}-{c['creator']}-{c['amount']}")
def test_golden_sdk_parity(case: dict) -> None:
    curve, st, amount, expected = CURVES[case["fee"]], _state(case), int(case["amount"]), int(case["result"])
    if case["fn"] == "buy_tokens_for_sol":
        got = curve.buy_tokens_for_sol(st, amount)
    elif case["fn"] == "buy_cost_for_tokens":
        got = curve.buy_cost_for_tokens(st, amount).total
    elif case["fn"] == "sell_proceeds_for_tokens":
        got = curve.sell_proceeds_for_tokens(st, amount).net
    else:
        got = st.market_cap_lamports
    assert got == expected


def test_golden_case_count() -> None:
    assert len(GOLDEN["cases"]) == 792


def test_fee_rounds_up() -> None:
    assert fee_amount(1, 95) == 1
    assert fee_amount(10_000, 95) == 95
    assert fee_amount(10_001, 95) == 96


def test_tier_selection_by_market_cap() -> None:
    sched = FeeSchedule([FeeTier(0, 95, 30), FeeTier(60_000_000_000, 90, 25), FeeTier(200_000_000_000, 80, 20)])
    assert sched.tier_for(0).protocol_bps == 95
    assert sched.tier_for(59_999_999_999).protocol_bps == 95
    assert sched.tier_for(60_000_000_000).protocol_bps == 90
    assert sched.tier_for(10**15).protocol_bps == 80


def test_no_creator_means_no_creator_fee() -> None:
    c = CURVES["global"]
    assert c.fees_for(c.new_state(has_creator=False)).creator == 0
    assert c.fees_for(c.new_state(has_creator=True)).creator == 30


def test_launch_price_and_market_cap() -> None:
    st = CURVES["global"].new_state()
    assert st.price == pytest.approx(V_SOL / V_TOK * PRICE_SCALE)
    assert st.price == pytest.approx(2.796e-8, rel=1e-3)
    assert st.market_cap_sol == pytest.approx(27.96, rel=1e-3)


@pytest.mark.parametrize("budget", [10_000_000, 250_000_000, 1_000_000_000, 5_000_000_000])
def test_budget_buy_never_exceeds_budget_and_round_trip_loses(budget: int) -> None:
    c = CURVES["global"]
    st = c.new_state()
    q = c.buy_with_budget(st, budget)
    assert 0 < q.total <= budget
    after = c.apply_buy(st, q)
    back = c.sell_proceeds_for_tokens(after, q.tokens)
    assert back.net < q.total  # fees both ways; the curve returns to its start
    assert c.apply_sell(after, back).v_tok == st.v_tok


def test_invariant_and_monotone_price() -> None:
    c = CURVES["global"]
    st = c.new_state()
    k = st.v_sol * st.v_tok
    prev = st.price
    for _ in range(20):
        q = c.buy_with_budget(st, 500_000_000)
        st = c.apply_buy(st, q)
        assert st.price > prev
        prev = st.price
        assert st.v_sol * st.v_tok >= k  # integer rounding only ever favours the curve
    fees = FeeBps(95, 30, 0)
    assert c.sell_proceeds_for_tokens(st, 10**12, fees).avg_price < st.price


def test_state_from_tokens_sold_is_on_the_hyperbola() -> None:
    c = CURVES["global"]
    st = c.state_from_tokens_sold(400_000_000_000_000)
    assert abs(st.v_sol * st.v_tok - V_SOL * V_TOK) < st.v_tok
    assert st.r_tok == R_TOK - 400_000_000_000_000
    assert c.progress_pct(st) == pytest.approx(100 * 400_000_000_000_000 / R_TOK)


def test_sol_to_complete_completes_the_curve() -> None:
    c = CURVES["global"]
    st = c.state_from_tokens_sold(700_000_000_000_000)
    need = c.sol_to_complete(st)
    q = c.buy_cost_for_tokens(st, st.r_tok)
    assert q.total == need
    assert c.apply_buy(st, q).complete


def test_limit_helpers_respect_limits() -> None:
    c = CURVES["global"]
    st = c.state_from_tokens_sold(100_000_000_000_000)
    limit = st.price * 1.10
    n = c.max_buy_tokens_at_avg_price(st, limit)
    assert c.buy_cost_for_tokens(st, n).avg_price <= limit < c.buy_cost_for_tokens(st, n + 10**9).avg_price
    floor = st.price * 0.9
    m = c.max_sell_tokens_at_avg_price(st, floor, 300_000_000_000_000)
    assert c.sell_proceeds_for_tokens(st, m).avg_price >= floor


def test_shift_state_models_persistent_own_impact() -> None:
    c = CURVES["global"]
    hist = c.state_from_tokens_sold(50_000_000_000_000)
    ours = 20_000_000_000_000
    shifted = shift_state(hist, ours)
    direct = c.state_from_tokens_sold(70_000_000_000_000)
    assert shifted.v_tok == direct.v_tok
    assert abs(shifted.v_sol - direct.v_sol) <= 1
    assert shifted.price > hist.price
    assert shift_state(hist, 0) is hist
    assert shift_state(shifted, -ours).v_tok == hist.v_tok


def test_price_impact_grows_with_size() -> None:
    c = CURVES["global"]
    st = c.new_state()
    small, large = c.price_impact_bps(st, 100_000_000), c.price_impact_bps(st, 5_000_000_000)
    assert 0 < small < large
