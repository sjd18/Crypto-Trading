# Live trading

> Live trading sends real, irreversible transactions from your wallet. This code path is
> covered by tests against mocked RPC / Metis endpoints but has **not** been run against mainnet
> with real funds by its authors. Read the code, paper-trade first, start with tiny sizes and
> limits, and use a dedicated wallet holding only what you can lose.

## Modes of access

| | Mode A — public | Mode B — authenticated |
|---|---|---|
| Config | `network.metis.mode: public` | `network.metis.mode: authenticated` |
| Base URL | `https://public.jupiterapi.com` | `METIS_URL` from `.env` |
| Auth | none | `Authorization: Bearer <JWT>` |
| Pump.fun quotes | computed locally with the exact curve math | `/pump-fun/quote` (or local) |
| Pump.fun swaps | `/pump-fun/swap` (returns a transaction to sign) | `/pump-fun/swap` or `/pump-fun/swap-instructions` |
| After migration | Jupiter `/quote` + `/swap` | Jupiter `/quote` + `/swap` |
| Fees | public platform fee on swaps (charged in backtests automatically) | per your plan |
| Lifetime | QuickNode announced shutdown on **2026-10-14**; the client logs a warning as the date approaches | — |

JWTs come from `PUMPFUN_JWT`, or are minted and refreshed automatically from
`QN_JWT_PRIVATE_KEY_PATH` + `QN_JWT_KID` (RS256 / ES256 with the `kid` registered on the
QuickNode endpoint). No token ever appears in code, config, logs or reports: every secret is a
`SecretStr` registered with the log redactor.

## Pipeline per order

```
QUOTE    local exact curve / pool math on live state (sub-millisecond), or Metis quote in Mode B
BUILD    Metis /pump-fun/swap (sign as returned)  |  /pump-fun/swap-instructions (compose locally:
         compute budget + priority fee + optional in-transaction Jito tip)  |  Jupiter after migration
SIGN     solders keypair from PRIVATE_KEY
SUBMIT   sendTransaction(skipPreflight, maxRetries=0)  |  Jito sendBundle
CONFIRM  one polling loop batches getSignatureStatuses for every in-flight signature;
         the same signed bytes are re-broadcast every rebroadcast_ms until confirmed / failed / expired
RETRY    only once the blockhash is provably expired (isBlockhashValid == false and a final status
         check) — a transaction that can still land is never re-signed, so no double buys
FILL     parsed from the confirmed transaction: SOL and token balance deltas, fees, decoded TradeEvent
```

`exec.quote_to_submit` is measured for every order against `live.quote_to_submit_budget_ms`
(250 ms). Every state transition is written to `MetaStore.live_orders` so a restart knows what
was in flight.

## Infrastructure

* **BlockhashCache** refreshes `getLatestBlockhash` every `blockhash_refresh_ms` and tracks block
  height for expiry decisions.
* **PriorityFeeEstimator** keeps an EWMA of the `dynamic_percentile` of
  `getRecentPrioritizationFees` for the Pump program, applies urgency multipliers (normal / high /
  exit) and bumps on retries; maps urgency to Metis `priorityFeeLevel`.
* **JitoClient** (optional, `jito.enabled`) sends bundles, reads tip accounts and the tip floor.
* **LiveStreamCollector** feeds the trader over an in-process queue; dispatch latency (arrival →
  subscribers fed) is budgeted at 100 ms and reported continuously.

## Safety interlocks

1. `live` refuses to start unless `app.mode: live` is set in the config **and** `--confirm-live`
   is passed on the command line.
2. The risk engine (loss limits, exposure limits, per-token order rate) and all circuit breakers
   run exactly as in the backtest; RPC latency and slot-time breakers are fed from live
   measurements.
3. The drawdown breaker can flatten the book (`flatten_on_drawdown`).
4. `stop(flatten=True)` (the default on exit) closes every position before shutdown.
5. Exits outrank entries in the order queue and are never blocked by risk limits.

## Offline rehearsal: `paper-replay`

`paper-replay` feeds events from the Parquet store into the live engine (`LiveTrader` with the
paper gateway) instead of the WebSocket stream: the same event queue, priority order queue,
worker pool, circuit breakers, snapshots and flatten-on-stop as a real session, with no network
access. It runs on a virtual-time event loop (`core/clock.py`): events are released when market
time reaches their timestamp, every latency, sweep and retry wait is a market-time timer, and the
clock jumps to the next timer whenever all tasks are waiting. A day of market replays in about
20 seconds, deterministically, and the command prints the result next to a backtest of the same
events. On the bundled 24-hour synthetic market with `smart_money` both produce 286 round trips
and the same fill counts, with total returns of +18.78 % and +18.51 %: the live engine draws
simulated latencies and failures in a different order, so the two agree statistically rather
than trade by trade. `--realtime` paces the replay at 1x instead, to watch the Live monitor
update.

## Runbook

```bash
python -m pumpfun_hft.main paper-replay            # offline: recorded events through the live engine vs a backtest
python -m pumpfun_hft.main check-config            # which secrets are present (never their values)
python -m pumpfun_hft.main latency-probe           # your RPC / Metis latency percentiles
python -m pumpfun_hft.main stream --minutes 30     # recorder only: watch dispatch latency, reconnects
python -m pumpfun_hft.main paper --minutes 120     # full stack with simulated fills on live data
python -m pumpfun_hft.main dashboard               # Live monitor page follows the paper / live session

# only after reviewing the code and the paper results:
python -m pumpfun_hft.main --set app.mode=live --set sizing.fixed_sol=0.02 live --confirm-live --minutes 30
```

Operational notes:

* Use a dedicated RPC with WebSocket support close to the validators you care about; public
  endpoints rate-limit and drop `logsSubscribe` messages under load.
* Keep `PRIVATE_KEY` for a hot wallet with a small balance; sweep profits elsewhere.
* Watch `pumpfun_hft/logs/trades.log`, `errors.log` and `latency.log` (JSON lines, rotated).
* After protocol upgrades run `update-idl`, check the fee tiers in the config and re-run the
  test suite before trading.
