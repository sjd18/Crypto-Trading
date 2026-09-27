# Backtest realism

A backtest is only useful if it would have been impossible to do better live. This page lists
every assumption the simulator makes, how look-ahead is ruled out, and how both are tested.

## Replay

* Events are replayed in on-chain order `(slot, seq, ev_idx)`; `backtest.replay_mode` chooses
  when strategies are evaluated:

  | mode | evaluation points |
  |---|---|
  | `event` | every event (default; what the live engine does) |
  | `trade` | swaps only |
  | `tick` | swaps that changed the price |
  | `candle` | the close of each `candle_interval_ms` bar, using only trades inside the bar |

* Internal events (order landing, confirmation, retries, sweeps, equity samples, candle closes)
  live on a heap and are processed before any market event with a later timestamp.
* All randomness comes from one seeded `numpy.random.Generator`, so runs are reproducible
  (`seed` in the report footer).

## The order lifecycle

```
decision t
  └─ + decision_ms + network_ms + rpc_ms                      (lognormal, median / sigma in config)
      submit ── outage window?  → reject or delay               (Poisson outages)
             ── rate limit?     → reject or delay               (token bucket in simulated time)
  └─ + inclusion_ms × (1 − priority_speedup × fee level)      (jito_inclusion_ms for bundles)
     × spike_multiplier with spike_prob, × latency_scale
      land ── drop (never lands; detected at blockhash expiry, no fee)
           ── latency > blockhash_ttl_ms → expired
           ── landed failure (base + priority fee charged, no swap)
           ── otherwise execute against the market *as it stands at landing*
  └─ + confirmation_ms
      confirm → the strategy runtime learns the result; retries are scheduled from here
```

Congestion (market activity above `failures.congestion_events_per_s`) multiplies drop and failure
probabilities. Jito bundles are atomic: a failed bundle costs neither fees nor the tip.

## Fills

* **Exact integer math.** Bonding-curve buys and sells use the same formulas as the program
  (`core/curve.py`, verified against 792 SDK golden cases); PumpSwap uses constant product with
  LP, protocol and creator fees (`core/amm.py`). Fee bps come from the tier for the market cap at
  landing or, with real data and `fees.source: observed_if_available`, from the bps recorded in
  the TradeEvent itself.
* **Persistent own impact.** The historical reserves never contain our trades, so the simulator
  shifts them along the curve's hyperbola by our net token position (`simulation.impact_model:
  persistent`): our buy raises the price for everyone after us — including our own exit — until
  we sell. Without this, a backtest can "buy" a large position for free.
* **Slippage tolerance** mirrors the on-chain checks (`min_tokens_out`, `max_sol_cost`,
  `min_sol_output`): if the price moved beyond the order's tolerance by the time it lands, the
  transaction fails and pays fees, exactly as on chain.
* **Order types.** MARKET; LIMIT (rests up to `limit_ttl_ms`, executes FOK at the limit when it
  lands); IOC (partial fill up to the limit price); FOK (all or nothing at the limit).
* **Partial fills** when the curve sells out on a buy, and **liquidity exhaustion** on sells
  (only what the real SOL reserves can pay). Nothing trades while a completed curve waits for
  migration; after migration, fills move to the PumpSwap pool.
* **Costs**, every one in lamports: protocol / creator / LP fees, router platform fee (Mode A's
  public Metis fee is charged automatically), base fee per signature, priority fee (compute
  units × micro-lamports, raised on retries), Jito tip, token-account rent (refunded when the
  closing sell closes the account).

## Accounting

`risk/portfolio.py` keeps an exact lamport ledger. Equity is marked at **liquidation value**
(what selling the whole position now would return after fees and impact), not at mid price, so a
large position in a thin curve is valued at what it is really worth. The report's cost breakdown,
the trade list and the equity curve reconcile to the lamport; fees of failed transactions that
never opened a position are booked as `unattributed_costs_sol` so nothing disappears.

## Look-ahead guarantees

| Risk | Guard |
|---|---|
| Deciding on data that has not happened | Strategies see `StrategyContext` built from state updated with events up to and including the current one |
| Executing at the decision price | Orders execute at landing, after every event with an earlier timestamp has been applied |
| Knowing a fill before it is confirmed | The runtime is told about fills at confirmation time |
| Labels leaking into scores | Token outcomes (creator history, wallet "rug" involvement) resolve only at `created_ms + resolution_horizon_s` using state up to that moment |
| Wallet scores using the current trade | Wallet flags for a trade are evaluated before the wallet database is updated with that trade |
| Using a model trained on the test period | `TrainedRugModel` raises `LookAheadError` for any event earlier than its training cut-off |
| Tuning on the test set | Test and live-sim splits are sealed; unsealing is logged and only allowed for the final evaluation |
| Feature definitions that peek | Online and vectorised feature pipelines agree exactly; batch features use as-of joins only |

The test suite enforces these with canaries (a strategy that would profit only with look-ahead
must not), latency-causality tests (a trade placed at `t` must not fill at a price from before
`t + latency`), exact reconciliation of equity with the ledger, determinism checks and
online / batch feature parity.

## What is not modelled

* Validator-level ordering inside a slot (we assume our transaction lands behind everything that
  landed earlier in time, which is conservative for entries and neutral on average for exits).
* MEV sandwiches against our own orders beyond what the slippage tolerance allows.
* Changes in Metis / RPC behaviour over time (latency distributions are stationary per run; use
  `montecarlo --paths N` to stress them).

## The synthetic market

`collectors/synthetic.py` generates a reproducible market for tests and demos. It is calibrated
to be *hard*, not realistic in every detail:

| Property | Synthetic default | Why |
|---|---|---|
| Launches | 30 / hour | compute budget (mainnet sees far more) |
| Median peak multiple | ~1.9 × launch price | most launches barely move |
| Reach 3 × / 10 × | ~14 % / ~1.5 % | heavy right tail |
| Graduation | ~0.4–1 % | in line with Pump.fun |
| Early drift (age 10 s → 70 s) | median −9 %, mean −4 % | snipers and copy bots sell into late buyers |
| Snipers | 70 % land in the creation slot | Jito bundles |
| Copy trading | Poisson(2.5) fast bots per smart / whale buy | followers pay their impact |
| Rugs | ~45 % of launches (dev + insider dumps) | serial ruggers with insider bundles |

(24-hour default market, seed 1234.)

It deliberately **plants** structure so the research stack can be shown to find it: informed
("smart") wallets that know a token's latent quality with noise, and serial ruggers. A strategy
that exploits the planted structure (`smart_money`) therefore makes money on synthetic data *by
construction*, while strategies without such an edge lose after costs. Neither result says
anything about the live market.
