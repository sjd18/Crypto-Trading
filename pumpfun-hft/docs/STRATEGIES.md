# Strategies

Every strategy implements one method:

```python
def generate_signal(self, ctx: StrategyContext) -> Signal: ...
```

`Signal.action` is one of `BUY`, `SELL`, `HOLD`, `EXIT`, `SCALE_IN`, `SCALE_OUT`, with a
`confidence` from 0 to 100, a human-readable `reason`, an optional `expected_return` (used by the
cost gate), an `urgency` (priority fee and slippage tier), optional order type / limit price and
optional per-position exit overrides. Parameters live in `strategy.params.<name>` and are
validated by a pydantic `Params` model on the strategy class.

`StrategyContext` gives a strategy everything that is known *now* and nothing else: the current
event, the token state, a lazy `FeatureView` (≈55 features), the open position and pending order,
wallet intelligence, the point-in-time creator score and rug probability, portfolio exposure,
`position_return()` at liquidation value and `round_trip_cost(size)` for the exact fee + impact
cost of a round trip.

## The ten strategies

| Strategy | Enters when | Leaves when | Key parameters |
|---|---|---|---|
| `momentum_ignition` | young token (`min_age_s`–`max_age_s`), strong one-sided buy flow (`min_buy_sol`, `min_imbalance`) from many wallets (`min_unique_buyers`), positive momentum, curve not too advanced, acceptable creator / rug / concentration profile | flow reverses (`exit_imbalance`) or the position manager / overlay exits | `min_buy_sol`, `min_imbalance`, `min_unique_buyers`, `min_momentum`, `continuation` |
| `bonding_curve_scalp` | a sell burst pushes price `dip_pct` below the short-window high while medium-window flow is still positive, mid-curve (`min/max_progress_pct`) | tight per-position target / stop / max hold via exit overrides | `dip_pct`, `target_pct`, `stop_pct`, `max_hold_s` |
| `liquidity_sweep` | optional re-entry after a deep pullback from a sweep high when the sweeper still holds and buyers return | a single large buy (`sweep_min_sol`, `sweep_impact_pct`) gets no follow-through within `exhaustion_s` | `sweep_min_sol`, `exhaustion_s`, `entry_pullback_pct` |
| `whale_follow` | a large buy (`min_whale_buy_sol`) by a wallet with a point-in-time smart score ≥ `min_wallet_score` and ≥ `min_wallet_closed` closed trades | the followed whale sells ≥ `exit_on_whale_sell_frac` of its bag | `min_whale_buy_sol`, `min_wallet_score` |
| `smart_money` | ≥ `min_smart_buyers` distinct elite wallets (score ≥ `min_smart_score`) buy within `window_s`, spending ≥ `min_total_smart_sol` | the elite cohort has sold ≥ `exit_smart_sell_frac` of what it bought | `min_smart_buyers`, `window_s`, `min_smart_score` |
| `mean_reversion` | a failed pump (ATH ≥ `min_pump_multiple` × launch) retraced `min_retrace_pct`–`max_retrace_pct` %, selling has decelerated (`sell_exhaustion_ratio`) and buyers step back in | target = `target_retrace_frac` of the way back to the ATH, or the position manager | `min_pump_multiple`, `min_retrace_pct`, `sell_exhaustion_ratio` |
| `volume_breakout` | short-window volume z-score ≥ `volume_z` with ≥ `min_short_volume_sol`, price at the medium-window high, net buying | position manager / overlay | `volume_z`, `min_short_volume_sol`, `min_imbalance` |
| `rug_avoidance` | never enters: it is the **exit overlay** and **entry veto** for every other strategy | rug probability ≥ `max_rug_prob`; creator sells ≥ `creator_sell_pct` %; liquidity falls ≥ `liquidity_drop_pct` % *and* ≥ `liquidity_drop_min_sol` SOL below its peak since entry; a top-3 holder with ≥ `top_holder_min_supply_pct` % of supply dumps ≥ `top_holder_dump_pct` % of it | thresholds above, `veto_entry_rug_prob` |
| `migration` | late-stage curve (`min/max_progress_pct`) with strong net buying | `post_migration_exit_s` after the PumpSwap pool opens (holds through completion; nothing can trade while a completed curve waits for migration) | `min_progress_pct`, `post_migration_exit_s` |
| `sniper` | within `max_entry_age_s` of creation: creator score ≥ `min_creator_score`, socials (optional), dev buy inside `[min_dev_buy_sol, max_dev_buy_sol]`, ≤ `max_bundled_buyers` unknown (non-sniper, non-bot) creation-slot buyers, no copy-cat name | own take profit / stop / max hold | `min_creator_score`, `max_bundled_buyers`, `size_sol` |

## From signal to order: the runtime

`strategies/runtime.py` is the only place where signals become orders, in backtests and live:

1. **Open position** — candidates from the overlay, the position manager and the owning strategy;
   priority `EXIT > SELL > SCALE_OUT > SCALE_IN`, then confidence.
2. **Flat** — the most confident `BUY` across active strategies, then in order:
   minimum confidence (`sizing.min_confidence`) → re-entry cooldown (`strategy.reentry_cooldown_s`
   after fully exiting a token) → overlay veto → sizing → **cost gate** (`expected_return ≥
   cost_gate_multiple × round-trip cost` for that exact size, including own impact both ways) →
   risk engine → order.
3. At most one order per token in flight; results arrive at confirmation; dropped, expired and
   slippage-failed orders are retried up to `simulation.retries.max_retries` with wider slippage
   and a higher priority fee (live: only after the original blockhash has expired).

Every decision, including rejections and their reason, is recorded in the run's `signals`
table, so a funnel ("how many ignition signals were vetoed by the rug overlay?") is one
`group_by` away.

## Sizing (`risk/sizing.py`)

| `sizing.method` | Size |
|---|---|
| `fixed` | `fixed_sol` |
| `fixed_risk` | `risk_per_trade_sol / stop_fraction` (the stop distance turns risk into size) |
| `kelly` | `kelly_fraction` × f\* of equity, f\* = W − (1 − W) / R from the closed trades so far (point-in-time); `fixed_risk` until `kelly_min_trades` exist; capped at `kelly_cap_frac` |
| `volatility` | `vol_target_sol / max(ATR%, realised vol, vol_floor)` |
| `confidence` | `fixed_sol` × (confidence / 100) ^ `confidence_exponent` |
| `max_exposure` | whatever room is left under the exposure limits |

All methods are then capped by `max_equity_frac`, per-token and total exposure limits, cash, and
a **liquidity cap**: the size is reduced (binary search on exact quotes) until the entry's own
price impact is below `max_impact_bps`.

## Exits (`risk/position_manager.py`)

Stop loss, a take-profit ladder of partial exits (`SCALE_OUT`, the last level closes), a trailing
stop that activates after `activation_pct`, breakeven after the first take-profit, a maximum
holding time and optional pyramiding (`SCALE_IN` after `add_trigger_pct` with enough
confidence). Every rule works on **liquidation value**, the SOL you would actually receive.
Strategies can override the stop, target and max hold per position (`Signal.exit_overrides`).

## Risk engine and circuit breakers

Entries are refused when a circuit breaker is active, the UTC-day or rolling-hour loss exceeds
`daily_loss_sol` / `hourly_loss_sol`, `max_open_positions` is reached, or the token already had
`max_orders_per_token_per_min` orders in the last minute. Surviving entries are clipped to the
headroom under `max_exposure_sol`, `max_position_per_token_sol`, `max_creator_exposure_sol` and
`max_sector_exposure_sol` (sectors come from keyword classification of name / symbol; pending
orders count as exposure). Exits are never blocked.

| Breaker | Trips when | Effect |
|---|---|---|
| `rpc_latency` | p90 of the last `rpc_latency_window` RPC calls > `rpc_latency_p90_ms` | no entries for `cooldown_s` |
| `congestion` | observed slot time > `congestion_slot_ms` | no entries for `cooldown_s` |
| `slippage` | mean adverse slippage of the last `slippage_window` fills > `slippage_bps_avg` | no entries for `cooldown_s` |
| `failed_swaps` | ≥ `failed_swaps_max` failures in `failed_swaps_window_s` | no entries for `cooldown_s` |
| `drawdown` | equity drawdown from peak > `drawdown_pct` | no entries; with `flatten_on_drawdown`, exit everything |

## Writing a new strategy

```python
from pumpfun_hft.core.types import Action, Signal, hold
from pumpfun_hft.strategies.base import Strategy, StrategyContext, StrategyParams, register


@register
class FreshWalletSurge(Strategy):
    name = "fresh_wallet_surge"

    class Params(StrategyParams):
        min_fresh_pct: float
        max_age_s: float
        max_rug_prob: float

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        if ctx.has_position or not ctx.is_trade or ctx.age_s > self.p.max_age_s:
            return hold()
        if ctx.f.fresh_wallet_pct < self.p.min_fresh_pct or ctx.rug_prob > self.p.max_rug_prob:
            return hold()
        return Signal(Action.BUY, 60.0, f"fresh wallets {ctx.f.fresh_wallet_pct:.0f}%", self.name, expected_return=0.1)
```

Import the module from `strategies/__init__.py`, add `strategy.params.fresh_wallet_surge` to your
config, optionally an `optimizer.spaces.fresh_wallet_surge` block, and run
`backtest --strategy fresh_wallet_surge`.
