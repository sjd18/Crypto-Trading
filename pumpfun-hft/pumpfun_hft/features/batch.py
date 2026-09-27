"""Vectorised (batch) feature computation with Polars — for research and ML datasets.

Windowed sums use *cumulative sums + as-of joins* rather than rolling windows keyed on time:
for trade ``i`` at time ``t`` the window ``[t - w, t]`` sum is ``C_i - C_{j-1}`` where ``j`` is
the first trade with ``ts >= t - w`` (found with a forward as-of join on de-duplicated
timestamps). Unlike naive time-rolling windows this never includes later trades that share the
same millisecond, so values match the online engine exactly (see ``tests/test_features.py``).

Each output row describes the state *immediately after* the corresponding trade.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from pumpfun_hft.core.types import EventKind

TRADE_KINDS = [EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value]


def _window_sums(tr: pl.DataFrame, cols: list[str], window_ms: int, suffix: str) -> pl.DataFrame:
    cum = tr.with_columns([pl.col(c).cum_sum().over("mint").alias(f"_C_{c}") for c in cols])
    prev = cum.with_columns([(pl.col(f"_C_{c}") - pl.col(c)).alias(f"_P_{c}") for c in cols])
    lookup = (prev.group_by(["mint", "ts_ms"], maintain_order=True)
              .agg([pl.col(f"_P_{c}").first() for c in cols])
              .sort(["mint", "ts_ms"]))
    query = prev.select(["mint", "_i", (pl.col("ts_ms") - window_ms).alias("_lk")] + [f"_C_{c}" for c in cols]).sort(["mint", "_lk"])
    joined = query.join_asof(lookup, left_on="_lk", right_on="ts_ms", by="mint", strategy="forward", check_sortedness=False)
    out = joined.select(["_i", "mint"] + [(pl.col(f"_C_{c}") - pl.col(f"_P_{c}")).alias(f"{c}_{suffix}") for c in cols])
    return out


def batch_features(events: pl.DataFrame, feat_cfg: Any, initial_real_token_reserves: int) -> pl.DataFrame:
    """Compute a core feature subset for every trade row (post-trade state)."""
    creates = (events.filter(pl.col("kind") == EventKind.CREATE.value)
               .select("mint", pl.col("ts_ms").alias("created_ms"),
                       (pl.col("v_sol") / pl.col("v_tok") / 1000.0).alias("launch_price")))
    tr = (events.filter(pl.col("kind").is_in(TRADE_KINDS))
          .sort(["slot", "seq", "ev_idx"])
          .with_columns(
              (pl.col("v_sol").cast(pl.Float64) / pl.col("v_tok").cast(pl.Float64) / 1000.0).alias("price"),
              (pl.col("sol_amount").cast(pl.Float64) / 1e9).alias("sol"),
          )
          .with_columns(
              pl.when(pl.col("is_buy")).then(pl.col("sol")).otherwise(0.0).alias("buy"),
              pl.when(pl.col("is_buy")).then(0.0).otherwise(pl.col("sol")).alias("sell"),
              (pl.col("price") * pl.col("sol")).alias("pv"),
              pl.lit(1.0).alias("n"),
          )
          .with_columns(pl.int_range(pl.len()).alias("_i")))
    out = tr.select("_i", "mint", "ts_ms", "slot", "kind", "price", "r_sol", "r_tok")
    for name, w in (("short", feat_cfg.short_window_ms), ("medium", feat_cfg.medium_window_ms), ("long", feat_cfg.long_window_ms)):
        ws = _window_sums(tr, ["buy", "sell", "pv", "sol", "n"], w, name)
        out = out.join(ws.drop("mint"), on="_i", how="left")
        # as-of price at t - w (last price at or before), including the launch price at creation
        pts = pl.concat([  # create first: it precedes any trade sharing its timestamp
            creates.select("mint", pl.col("created_ms").alias("ts_ms"), pl.col("launch_price").alias("price")),
            tr.select("mint", "ts_ms", "price"),
        ]).sort(["mint", "ts_ms"], maintain_order=True)
        pts = pts.group_by(["mint", "ts_ms"], maintain_order=True).agg(pl.col("price").last()).sort(["mint", "ts_ms"])
        q = out.select("_i", "mint", (pl.col("ts_ms") - w).alias("_lk")).sort(["mint", "_lk"])
        lag = q.join_asof(pts.rename({"price": f"_p_{name}"}), left_on="_lk", right_on="ts_ms", by="mint",
                          strategy="backward", check_sortedness=False).select("_i", f"_p_{name}")
        out = out.join(lag, on="_i", how="left")
    out = out.join(creates, on="mint", how="left")
    out = out.with_columns(
        pl.col("_p_short").fill_null(pl.col("launch_price")),
        pl.col("_p_medium").fill_null(pl.col("launch_price")),
        pl.col("_p_long").fill_null(pl.col("launch_price")),
    )
    res = out.with_columns(
        pl.col("buy_short").alias("buy_sol_short"),
        pl.col("sell_short").alias("sell_sol_short"),
        pl.col("buy_medium").alias("buy_sol_medium"),
        pl.col("sell_medium").alias("sell_sol_medium"),
        (pl.col("buy_medium") - pl.col("sell_medium")).alias("delta_sol_medium"),
        pl.col("sol_long").alias("volume_sol_long"),
        pl.when((pl.col("buy_medium") + pl.col("sell_medium")) > 1e-12)
          .then((pl.col("buy_medium") - pl.col("sell_medium")) / (pl.col("buy_medium") + pl.col("sell_medium"))).otherwise(0.0)
          .alias("imbalance_medium"),
        pl.when((pl.col("buy_long") + pl.col("sell_long")) > 1e-12)
          .then((pl.col("buy_long") - pl.col("sell_long")) / (pl.col("buy_long") + pl.col("sell_long"))).otherwise(0.0)
          .alias("imbalance_long"),
        pl.col("n_medium").cast(pl.Int64).alias("n_trades_medium"),
        pl.when(pl.col("sol_medium") > 1e-12).then(pl.col("price") / (pl.col("pv_medium") / pl.col("sol_medium")) - 1.0)
          .otherwise(0.0).alias("vwap_dist_medium"),
        (pl.col("price") / pl.col("_p_short")).log().alias("ret_short"),
        (pl.col("price") / pl.col("_p_medium")).log().alias("ret_medium"),
        (pl.col("price") / pl.col("_p_long")).log().alias("ret_long"),
        pl.when(pl.col("kind") == EventKind.TRADE.value)
          .then(100.0 * (initial_real_token_reserves - pl.col("r_tok")) / initial_real_token_reserves).otherwise(100.0)
          .alias("progress_pct"),
        (pl.col("r_sol").cast(pl.Float64) / 1e9).alias("liquidity_sol"),
        ((pl.col("ts_ms") - pl.col("created_ms")) / 1000.0).alias("age_s"),
        (2 * math.pi * ((pl.col("ts_ms") // 1000) % 86_400) / 86_400).sin().alias("hour_sin"),
        (2 * math.pi * ((pl.col("ts_ms") // 1000) % 86_400) / 86_400).cos().alias("hour_cos"),
    )
    keep = ["mint", "ts_ms", "slot", "price", "buy_sol_short", "sell_sol_short", "buy_sol_medium", "sell_sol_medium",
            "delta_sol_medium", "volume_sol_long", "imbalance_medium", "imbalance_long", "n_trades_medium",
            "vwap_dist_medium", "ret_short", "ret_medium", "ret_long", "progress_pct", "liquidity_sol", "age_s",
            "hour_sin", "hour_cos"]
    return res.sort("_i").select(keep)


def candles(events: pl.DataFrame, interval_ms: int) -> pl.DataFrame:
    """OHLCV candles per mint from trades (buy/sell volume split, last reserves per bar)."""
    tr = (events.filter(pl.col("kind").is_in(TRADE_KINDS)).sort(["slot", "seq", "ev_idx"])
          .with_columns((pl.col("v_sol").cast(pl.Float64) / pl.col("v_tok").cast(pl.Float64) / 1000.0).alias("price"),
                        (pl.col("sol_amount").cast(pl.Float64) / 1e9).alias("sol"),
                        (pl.col("ts_ms") - pl.col("ts_ms") % interval_ms).alias("bar_ms")))
    return (tr.group_by(["mint", "bar_ms"], maintain_order=True)
            .agg(pl.col("price").first().alias("open"), pl.col("price").max().alias("high"), pl.col("price").min().alias("low"),
                 pl.col("price").last().alias("close"), pl.col("sol").sum().alias("volume_sol"),
                 pl.col("sol").filter(pl.col("is_buy")).sum().alias("buy_sol"),
                 pl.col("sol").filter(~pl.col("is_buy")).sum().alias("sell_sol"), pl.len().alias("n_trades"),
                 pl.col("user").n_unique().alias("unique_traders"), pl.col("slot").last().alias("slot"),
                 pl.col("v_sol").last(), pl.col("v_tok").last(), pl.col("r_sol").last(), pl.col("r_tok").last())
            .sort(["bar_ms", "mint"]))
