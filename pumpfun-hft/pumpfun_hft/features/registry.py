"""Feature catalogue: names, groups and definitions (single source for docs, ML and dashboards).

Every feature is computed from information available at (or before) the evaluation time — the
"timestamp-safe" column documents why. Windows: short / medium / long = ``features.*_window_ms``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    group: str
    unit: str
    description: str


FEATURES: tuple[FeatureSpec, ...] = (
    # ---- price
    FeatureSpec("price", "price", "SOL/token", "last traded spot price (post-trade reserves)"),
    FeatureSpec("log_ret", "price", "log", "log return of the last trade"),
    FeatureSpec("ret_short", "price", "log", "log(price / price at t-short) (as-of lookup; launch price if younger)"),
    FeatureSpec("ret_medium", "price", "log", "log return over the medium window"),
    FeatureSpec("ret_long", "price", "log", "log return over the long window"),
    FeatureSpec("momentum", "price", "log", "EWMA(log p, fast) - EWMA(log p, slow), time-decayed"),
    FeatureSpec("vwap_dist_medium", "price", "frac", "price / VWAP(medium) - 1"),
    FeatureSpec("atr_pct", "price", "frac", "Wilder ATR over completed bars / price"),
    FeatureSpec("rv_short", "price", "log", "realised volatility sqrt(sum r^2) in the short window"),
    FeatureSpec("rv_medium", "price", "log", "realised volatility in the medium window"),
    FeatureSpec("var_medium", "price", "log^2", "sample variance of trade log returns in the medium window"),
    FeatureSpec("micro_imbalance", "price", "frac", "(buy - sell) / (buy + sell) SOL volume, short window"),
    FeatureSpec("ath_multiple", "price", "x", "ATH price / launch price"),
    FeatureSpec("drawdown_from_ath_pct", "price", "%", "100 * (1 - price / ATH)"),
    FeatureSpec("high_medium_dist", "price", "frac", "price / max price in medium window - 1"),
    # ---- volume / order flow
    FeatureSpec("buy_sol_short", "volume", "SOL", "buy volume, short window"),
    FeatureSpec("sell_sol_short", "volume", "SOL", "sell volume, short window"),
    FeatureSpec("buy_sol_medium", "volume", "SOL", "buy volume, medium window"),
    FeatureSpec("sell_sol_medium", "volume", "SOL", "sell volume, medium window"),
    FeatureSpec("delta_sol_medium", "volume", "SOL", "buy - sell volume, medium window"),
    FeatureSpec("volume_sol_long", "volume", "SOL", "total volume, long window"),
    FeatureSpec("imbalance_medium", "volume", "frac", "order-flow imbalance, medium window"),
    FeatureSpec("imbalance_long", "volume", "frac", "order-flow imbalance, long window"),
    FeatureSpec("volume_accel", "volume", "SOL/s^2", "change in volume rate between the last two short windows"),
    FeatureSpec("volume_z", "volume", "z", "short-window volume z-score vs EWMA of completed bar volumes"),
    FeatureSpec("ofi_medium", "volume", "frac", "net flow (medium) / curve liquidity"),
    FeatureSpec("aggressive_buy_ratio", "volume", "frac", "share of medium-window buy volume from high-impact buys"),
    FeatureSpec("n_trades_medium", "volume", "count", "trades in the medium window"),
    FeatureSpec("unique_buyers_medium", "volume", "count", "distinct buyers in the medium window"),
    # ---- wallets
    FeatureSpec("whale_share", "wallet", "frac", "share of medium-window volume from whale-size trades / whale wallets"),
    FeatureSpec("smart_share", "wallet", "frac", "share of medium-window buy volume from wallets with smart score >= threshold"),
    FeatureSpec("fresh_wallet_pct", "wallet", "%", "% of medium-window buyers first seen within the fresh window"),
    FeatureSpec("bot_wallet_pct", "wallet", "%", "% of medium-window buyers flagged as bots"),
    FeatureSpec("hhi", "wallet", "frac", "Herfindahl index of holder balances (circulating supply)"),
    FeatureSpec("top10_pct", "wallet", "%", "% of total supply held by the 10 largest holders"),
    FeatureSpec("creator_holding_pct", "wallet", "%", "% of total supply held by the creator"),
    FeatureSpec("creator_sold_pct", "wallet", "%", "% of the creator's bought tokens already sold"),
    FeatureSpec("n_holders", "wallet", "count", "wallets with a positive reconstructed balance"),
    FeatureSpec("bundled_buyers", "wallet", "count", "non-creator buyers in the creation slot"),
    FeatureSpec("dev_buy_sol", "wallet", "SOL", "creator buys in the creation slot"),
    # ---- bonding curve
    FeatureSpec("progress_pct", "curve", "%", "share of sellable supply bought from the curve"),
    FeatureSpec("liquidity_sol", "curve", "SOL", "real SOL reserves (pool quote reserves after migration)"),
    FeatureSpec("liquidity_slope", "curve", "SOL/s", "change of real SOL reserves per second over the medium window"),
    FeatureSpec("liquidity_drop_pct", "curve", "%", "drop of real SOL reserves from their peak"),
    FeatureSpec("buy_pressure", "curve", "1/s", "short-window buy SOL per second / liquidity"),
    FeatureSpec("remaining_tokens_pct", "curve", "%", "real token reserves left / initial"),
    FeatureSpec("sol_to_complete", "curve", "SOL", "all-in SOL needed to complete the curve"),
    FeatureSpec("curve_accel", "curve", "%/s^2", "second difference of progress over consecutive short windows"),
    FeatureSpec("mcap_sol", "curve", "SOL", "market cap in SOL"),
    # ---- time
    FeatureSpec("age_s", "time", "s", "seconds since launch"),
    FeatureSpec("hour_sin", "time", "", "sin(2 pi hour/24), UTC"),
    FeatureSpec("hour_cos", "time", "", "cos(2 pi hour/24), UTC"),
    FeatureSpec("slot_velocity", "time", "slots/s", "EWMA slot advance rate"),
    FeatureSpec("tps", "time", "events/s", "EWMA market-wide Pump event rate (activity / congestion proxy)"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_GROUPS: dict[str, list[str]] = {}
for _f in FEATURES:
    FEATURE_GROUPS.setdefault(_f.group, []).append(_f.name)
