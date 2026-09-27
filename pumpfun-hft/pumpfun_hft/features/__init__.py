"""Market state and feature engineering.

Purpose
    Point-in-time market state and "institutional" alpha features for Pump.fun tokens: price,
    volume / order-flow, wallet, bonding-curve and time features. Every feature is timestamp-safe
    (computed only from events at or before the evaluation time).

Architecture
    market.py    MarketState / TokenState: curve & pool state, fees, lifecycle facts
    online.py    FeatureEngine: O(1) incremental updates + lazy FeatureView (used live & in backtests)
    batch.py     vectorised Polars implementation (research / ML datasets) + OHLCV candles
    registry.py  feature catalogue (names, groups, units, definitions)

Data flow
    Event -> MarketState.on_event -> FeatureEngine.on_event -> FeatureView(now) -> strategies / ML
    (WalletIntel is queried for smart / whale / fresh / bot flags *before* it ingests the trade.)

Inputs / Outputs
    Inputs: Event stream, Settings.features, BondingCurve, WalletIntel. Outputs: FeatureView
    attributes / ``as_dict()`` rows; batch DataFrames.

Example
    ms = MarketState(curve); fe = FeatureEngine(settings.features, curve, wallet_intel)
    st = ms.on_event(ev); fe.on_event(ev, st); fe.view(ev.mint, ev.ts_ms, st).imbalance_medium
"""
