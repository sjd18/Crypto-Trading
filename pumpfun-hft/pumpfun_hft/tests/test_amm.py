"""PumpSwap constant-product math, fee tiers decoded from mainnet, and own-impact shifts."""

from __future__ import annotations

import pytest

from pumpfun_hft.core.amm import ConstantProductAmm, PoolState, shift_pool
from pumpfun_hft.core.curve import FeeSchedule, FeeTier
from pumpfun_hft.core.idl import load_codecs
from pumpfun_hft.tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def amm(settings) -> ConstantProductAmm:
    flat = settings.protocol.amm_flat_fees
    return ConstantProductAmm(FeeSchedule.from_config(settings.protocol.amm_fee_tiers), FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))


@pytest.fixture()
def pool() -> PoolState:
    # a freshly migrated pool: ~206.9M tokens against ~85 SOL
    return PoolState(base=206_900_000_000_000, quote=84_990_359_454)


def test_config_tiers_match_mainnet_fee_config(settings) -> None:
    codecs = load_codecs(settings.paths.resolve("idl_dir"))
    fees_prog = codecs[settings.protocol.pump_fees_program_id]
    name, acc = fees_prog.decode_account((FIXTURES / "pump_amm_fee_config_mainnet.bin").read_bytes())
    assert name == "FeeConfig"
    onchain = [(t["market_cap_lamports_threshold"], t["fees"]["lp_fee_bps"], t["fees"]["protocol_fee_bps"], t["fees"]["creator_fee_bps"])
               for t in acc["fee_tiers"]]
    configured = [(int(round(t.mcap_sol * 1e9)), t.lp_bps, t.protocol_bps, t.creator_bps) for t in settings.protocol.amm_fee_tiers]
    assert len(onchain) == 25
    assert onchain == configured
    flat = settings.protocol.amm_flat_fees
    assert (acc["flat_fees"]["lp_fee_bps"], acc["flat_fees"]["protocol_fee_bps"], acc["flat_fees"]["creator_fee_bps"]) == \
        (flat.lp_bps, flat.protocol_bps, flat.creator_bps)


def test_buy_and_sell_are_consistent(amm: ConstantProductAmm, pool: PoolState) -> None:
    q = amm.buy_base_for_quote(pool, 1_000_000_000)
    assert q.total <= 1_000_000_000 and q.base_out > 0
    after = amm.apply_buy(pool, q)
    assert after.price > pool.price
    s = amm.sell_quote_for_base(after, q.base_out)
    assert 0 < s.net < q.total
    back = amm.apply_sell(after, s)
    # the LP fee stays in the pool on both legs: LPs are richer, the base returns to where it was
    assert back.base == pool.base
    assert back.quote > pool.quote


def test_constant_product_never_decreases(amm: ConstantProductAmm, pool: PoolState) -> None:
    k = pool.base * pool.quote_eff
    p = pool
    for budget in (10**8, 5 * 10**8, 3 * 10**9):
        p = amm.apply_buy(p, amm.buy_base_for_quote(p, budget))
        assert p.base * p.quote_eff >= k
        k = p.base * p.quote_eff


def test_fees_follow_market_cap_tier(amm: ConstantProductAmm, pool: PoolState) -> None:
    f = amm.fees_for(pool)
    tier = amm.tiers.tier_for(pool.market_cap_lamports)
    assert (f.protocol, f.creator, f.lp) == (tier.protocol_bps, tier.creator_bps, tier.lp_bps)
    non_canonical = PoolState(pool.base, pool.quote, canonical=False)
    flat = amm.fees_for(non_canonical)
    assert (flat.protocol, flat.lp) == (5, 25)


def test_exact_output_buy_costs_at_least_the_budget_quote(amm: ConstantProductAmm, pool: PoolState) -> None:
    by_budget = amm.buy_base_for_quote(pool, 2_000_000_000)
    by_output = amm.buy_quote_for_base(pool, by_budget.base_out)
    assert by_output.base_out == by_budget.base_out
    assert by_output.total >= by_budget.total - 2


def test_shift_pool_moves_price_along_the_curve(pool: PoolState) -> None:
    ours = 5_000_000_000_000
    shifted = shift_pool(pool, ours)
    assert shifted.base == pool.base - ours
    assert shifted.price > pool.price
    assert abs(shifted.base * shifted.quote_eff - pool.base * pool.quote_eff) < shifted.base
    assert shift_pool(pool, 0) is pool or shift_pool(pool, 0) == pool
