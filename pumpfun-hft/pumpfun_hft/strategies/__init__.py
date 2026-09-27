"""HFT strategy framework and reference strategies.

Purpose
    Pluggable strategies exposing ``generate_signal(ctx) -> Signal`` (BUY / SELL / HOLD / EXIT /
    SCALE_IN / SCALE_OUT, confidence 0-100) and a shared runtime that turns signals into orders
    identically for backtests, paper and live trading.

Architecture
    base.py                 Strategy ABC, StrategyParams, StrategyContext, registry, build_strategy
    runtime.py              StrategyRuntime (signals -> cost gate -> sizing -> risk -> orders), CostModel
    momentum_ignition.py    buy early explosive launches
    bonding_curve_scalp.py  trade transient curve dips (fee-aware)
    liquidity_sweep.py      detect exhaustion after large buys (exit) / post-sweep pullback entries
    whale_follow.py         follow statistically profitable whales
    smart_money.py          enter after several elite wallets buy
    mean_reversion.py       trade failed pumps
    volume_breakout.py      trade abnormal volume expansion
    rug_avoidance.py        exit before rug signatures (overlay) and veto risky entries
    migration.py            trade curve completion and migration to PumpSwap
    sniper.py               enter in the first seconds after launch

Data flow
    Event -> MarketState / FeatureEngine -> StrategyContext -> Strategy.generate_signal
    -> StrategyRuntime (exit checks, veto, cost gate, sizing, risk) -> Order -> execution

Inputs / Outputs
    Inputs: a StrategyContext per event (token state, point-in-time features, wallet intel,
    position, equity) and each strategy's parameters from ``strategy.params`` in the YAML.
    Outputs: Signal objects (action, confidence 0-100, reason), then sized, risk-checked Orders;
    every decision is recorded with its outcome (submitted, vetoed, low_confidence, cost_gate,
    reentry_cooldown, risk:<reason>, ...).

Example
    strat = build_strategy("momentum_ignition", settings)
    sig = strat.generate_signal(ctx)
"""

from pumpfun_hft.strategies import (  # noqa: F401  (registration side effects)
    bonding_curve_scalp,
    liquidity_sweep,
    mean_reversion,
    migration,
    momentum_ignition,
    rug_avoidance,
    smart_money,
    sniper,
    volume_breakout,
    whale_follow,
)
from pumpfun_hft.strategies.base import STRATEGY_REGISTRY, Strategy, StrategyContext, build_strategy, register

__all__ = ["STRATEGY_REGISTRY", "Strategy", "StrategyContext", "build_strategy", "register"]
