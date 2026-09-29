# Module reference

Generated from the source docstrings by `python -m pumpfun_hft.main docs`. Every package docstring states its purpose, architecture, data flow, inputs/outputs and an example.

## `pumpfun_hft.core`

```text
Core protocol layer.

Purpose
    Protocol-exact primitives shared by research, simulation and live trading.

Architecture
    config.py   typed YAML configuration (no hidden defaults) + secrets from .env
    types.py    Event/Order/Fill/Signal model and the canonical Parquet event schema
    curve.py    exact integer Pump bonding-curve math (verified against the official SDK)
    amm.py      constant-product AMM math (PumpSwap canonical pools; template for other DEXs)
    idl.py      IDL-driven Borsh codec (events, accounts, instructions; tolerant of old layouts)
    events.py   log / self-CPI event extraction and normalisation to the flat Event schema
    clock.py    simulated, wall and replay clocks behind one interface; a virtual-time asyncio loop
    idl/        bundled Anchor IDLs (pump, pump_amm, pump_fees) from pump-fun/pump-public-docs

Data flow
    raw tx / logs -> events.EventDecoder -> Event rows -> collectors / replay
    Event reserves -> curve.CurveState -> quotes (BuyQuote / SellQuote) -> simulator & live quotes

Inputs / Outputs
    Inputs: IDL JSON, config YAML, raw transaction JSON or log lines.
    Outputs: Event records, exact integer quotes and fee breakdowns.

Example
    >>> from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
    >>> bc = BondingCurve(FeeSchedule([FeeTier(0, 95, 30)]), 1073000000000000, 30000000000, 793100000000000, 10**15)
    >>> st = bc.new_state()
    >>> bc.buy_tokens_for_sol(st, 1_000_000_000) > 0
    True
```

- **`core/amm.py`** — Constant-product AMM math (PumpSwap canonical pools; template for Raydium/Meteora/Orca).
- **`core/clock.py`** — Clock abstraction so strategy / risk / execution code is identical in simulation, replay and live trading.
- **`core/config.py`** — Typed YAML configuration.
- **`core/curve.py`** — Exact integer math of the Pump bonding curve.
- **`core/events.py`** — Extraction and normalisation of Pump / PumpSwap events from Solana transactions.
- **`core/idl.py`** — IDL-driven Borsh codec for Anchor programs (Anchor >= 0.30 IDL format).
- **`core/pda.py`** — Program-derived addresses used by Pump / PumpSwap / SPL (requires ``solders``).
- **`core/types.py`** — Core domain types shared by research, simulation and live execution.

## `pumpfun_hft.api`

```text
Network API layer.

Purpose
    Resilient async clients for every external service, with uniform rate limiting,
    retries (exponential backoff + full jitter, ``Retry-After`` aware), a per-host circuit
    breaker (API outage handling), latency tracking and redacted JSON logging to ``api.log``.

Architecture
    rate_limit.py  token bucket (async + simulated-time) and AIMD adaptive limiter (429 aware)
    http.py        AsyncHttpClient: httpx pool + limiter + retries + breaker + latency
    auth.py        JWT provider: static ``PUMPFUN_JWT`` or auto-minted RS256/ES256 tokens
    rpc.py         Solana JSON-RPC (single + batch) with typed helpers
    ws.py          Solana WebSocket client: auto-reconnect, resubscribe, arrival timestamps
    metis.py       QuickNode Metis: Mode A (public.jupiterapi.com) / Mode B (JWT + METIS_URL);
                   pump-fun quote/swap/swap-instructions + Jupiter quote/swap for migrated tokens
    pumpfun.py     Pump data facade: bonding-curve state (on-chain), token metadata (Metaplex +
                   URI JSON), holder analytics, profiles (optional pump.fun frontend API)

Data flow
    strategies / execution / collectors -> these clients -> HTTP / WS endpoints
    Every request: limiter.acquire -> breaker check -> request -> latency.record -> retry?

Inputs / Outputs
    Inputs: Settings.network, Secrets (URLs / tokens). Outputs: parsed JSON / typed results.

Example
    rpc = SolanaRpcClient(http, secrets.get("solana_rpc_url"), settings.network.rpc)
    slot = await rpc.get_slot()
```

- **`api/auth.py`** — JWT bearer authentication for Mode B (authenticated Metis / QuickNode endpoints).
- **`api/http.py`** — Resilient async HTTP client.
- **`api/metis.py`** — QuickNode Metis client — Pump.fun and Jupiter swap APIs.
- **`api/pumpfun.py`** — Pump.fun data facade: bonding-curve state, token metadata, holder analytics, profiles.
- **`api/rate_limit.py`** — Rate limiting primitives.
- **`api/rpc.py`** — Solana JSON-RPC client (single and batched requests) with typed helpers.
- **`api/ws.py`** — Solana WebSocket client with automatic reconnection and resubscription.

## `pumpfun_hft.collectors`

```text
Market data collection.

Purpose
    Acquire Pump.fun / PumpSwap market data (historical and live) into a durable, verified,
    day-partitioned Parquet store; provide a synthetic generator for offline research/tests.

Architecture
    storage.py     ParquetEventStore: day partitions, SHA-256 manifest, compaction/de-dup,
                   gap detection, lazy scans and ordered batch iteration (memory-efficient replay)
    historical.py  HistoricalCollector: paged signatures -> batched concurrent getTransaction ->
                   decode -> timestamps -> store; resumable checkpoints; retry queue; gap registry
    live.py        LiveStreamCollector: WebSocket logs/slot subscriptions -> decode -> fan-out
                   queues (<100 ms budget measured continuously) -> buffered persistence
    slotclock.py   millisecond timestamps from (slot, block_time) anchors
    sol_price.py   SOL/USD providers (static, CSV series with as-of lookups)
    synthetic.py   SyntheticMarket: exact-math synthetic launches, rugs, migrations, wallets
    datasets.py    synthetic vs real data-set folders: kind markers, finding event stores on disk,
                   copying real events between stores (synthetic tokens left behind)

Data flow
    RPC / WS  ->  core.events.EventDecoder  ->  Event rows  ->  ParquetEventStore  ->  replay
                                          \->  subscribers (discovery, features, strategies)

Inputs / Outputs
    Inputs: Settings.collector / protocol, RPC & WS clients. Outputs: Parquet files under
    ``paths.data_dir/events``, manifest/checkpoint/gap rows in SQLite, in-memory Event queues.

Example
    store = ParquetEventStore(settings.paths.events_dir, MetaStore(settings.paths.resolve("sqlite_file")))
    SyntheticMarket(settings).write(store, settings.paths.metadata_dir / "tokens.parquet")
    store.verify()   # -> [] when every checksum matches
```

- **`collectors/datasets.py`** — Data sets: which folder holds which kind of events, finding event stores, and moving real events.
- **`collectors/historical.py`** — Historical collector: Pump program history via RPC.
- **`collectors/live.py`** — Live stream collector (Solana WebSockets).
- **`collectors/slotclock.py`** — Millisecond timestamps for historical events.
- **`collectors/sol_price.py`** — SOL/USD price providers (point-in-time lookups for USD-denominated features and reports).
- **`collectors/storage.py`** — Day-partitioned Parquet event store with checksums, de-duplication and gap detection.
- **`collectors/synthetic.py`** — Synthetic Pump.fun market generator.

## `pumpfun_hft.discovery`

```text
Token discovery and creator intelligence.

Purpose
    Detect every new Pump.fun token immediately, enrich it (metadata, socials, launch facts) and
    score its creator statistically from *resolved* past launches only.

Architecture
    scanner.py    TokenDiscoveryEngine -> DiscoveredToken (mint, creator, slot, initial
                  liquidity / SOL deposited / supply, progress, socials, image, creator stats)
    creator.py    CreatorBook: launches, win rate, rugs, migrations, avg ATH multiple, Beta-
                  posterior scores and Wilson bounds -> 0-100 creator score
    lifecycle.py  OutcomeResolver: resolves each token at its horizon (success / rug / neutral)
                  and feeds CreatorBook + WalletIntel

Data flow
    CreateEvent -> MarketState -> TokenDiscoveryEngine.on_create -> strategies (e.g. sniper)
    time passes -> OutcomeResolver.advance -> CreatorBook / WalletIntel updated for *later* launches

Inputs / Outputs
    Inputs: TokenState, Settings.discovery, optional MetadataFetcher. Outputs: DiscoveredToken,
    creator scores, token outcomes (also persisted to DuckDB tables tokens / creators).

Example
    book = CreatorBook(settings.discovery); resolver = OutcomeResolver(settings.discovery, book)
    disc = TokenDiscoveryEngine(settings, market, book)
    tok = disc.on_create(state); tok.creator_score.score
```

- **`discovery/creator.py`** — Point-in-time creator history and statistical creator-quality scoring.
- **`discovery/lifecycle.py`** — Token outcome resolution (feeds creator history and wallet intelligence).
- **`discovery/scanner.py`** — Token discovery engine: track every new token immediately and enrich it.

## `pumpfun_hft.features`

```text
Market state and feature engineering.

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
```

- **`features/batch.py`** — Vectorised (batch) feature computation with Polars — for research and ML datasets.
- **`features/market.py`** — Point-in-time market state (per-token curve / pool state and lifecycle facts).
- **`features/online.py`** — Incremental (online) feature engine — O(1) amortised work per event.
- **`features/registry.py`** — Feature catalogue: names, groups and definitions (single source for docs, ML and dashboards).
- **`features/replay.py`** — Replay helpers: run events through the online state exactly like the backtester does.

## `pumpfun_hft.strategies`

```text
HFT strategy framework and reference strategies.

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
    ml_signal.py            trade a model trained with train-model (fwd_up / migrate)

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
```

- **`strategies/base.py`** — Strategy framework.
- **`strategies/bonding_curve_scalp.py`** — Bonding-curve scalping: buy transient sell-driven dips on otherwise healthy curves.
- **`strategies/liquidity_sweep.py`** — Liquidity sweep detection: identify exhaustion after large buys.
- **`strategies/mean_reversion.py`** — Mean reversion on failed pumps.
- **`strategies/migration.py`** — Migration strategy: trade curves that are about to graduate to PumpSwap.
- **`strategies/ml_signal.py`** — ML signal: trade the probability from a model trained with ``train-model``.
- **`strategies/momentum_ignition.py`** — Momentum ignition: buy young tokens whose launch is turning explosive.
- **`strategies/rug_avoidance.py`** — Rug avoidance overlay: exit before rug signatures complete; veto risky entries.
- **`strategies/runtime.py`** — Strategy runtime: the single decision layer shared by backtests, paper and live trading.
- **`strategies/smart_money.py`** — Smart-money copy trading: enter after several elite wallets buy the same token.
- **`strategies/sniper.py`** — Sniper: enter in the first seconds after launch when the launch profile is clean.
- **`strategies/volume_breakout.py`** — Volume breakout: trade abnormal volume expansion that breaks the recent high.
- **`strategies/whale_follow.py`** — Whale follow: mirror large buys from *statistically profitable* wallets.

## `pumpfun_hft.backtester`

```text
Event-driven backtester with realistic execution.

Purpose
    Replay stored on-chain events in strict (slot, seq, ev_idx) order and run the same strategy
    runtime, sizing, risk engine and position manager that trade live, against a simulated
    execution venue that charges every real cost and never lets a decision see the future.

Architecture
    replay.py         DataSource over a Parquet store or an in-memory frame (batched, bounded by
                      time), token metadata loader, data fingerprint for provenance
    execution_sim.py  ExecutionSimulator: exact curve / AMM fills on the *effective* state
                      (historical reserves shifted by our own persistent impact), protocol /
                      creator / LP / platform fees, priority fees, Jito tips, rent, lognormal
                      latency components, drops, landed failures, congestion, blockhash expiry,
                      outages and rate limits; market / limit / IOC / FOK orders, partial fills
    engine.py         BacktestEngine: event / trade / tick / candle replay modes, an internal
                      heap of order arrivals and confirmations interleaved with market events,
                      periodic sweeps, equity sampling, end-of-data liquidation
    results.py        BacktestResult: metrics, equity, trades, fills, signals, diagnostics; save /
                      load (Parquet + JSON)

Data flow
    DataSource batches -> MarketState / FeatureEngine / WalletIntel / CreatorBook (point-in-time)
    -> StrategyRuntime.evaluate -> Order -> ExecutionSimulator.submit (latency, failures)
    -> heap(arrival) -> ExecutionSimulator.execute on the state at arrival -> Fill -> Portfolio

    Look-ahead guards: a decision only sees events already applied; an order executes against
    the market as it stands when it lands (after later events are applied); fills are learned at
    confirmation; outcome labels used by scorers resolve only after their horizon.

Inputs / Outputs
    Inputs: Settings, a DataSource, strategy names (or instances), optional token metadata.
    Outputs: BacktestResult (metrics, equity curve, trades, fills, signal log, diagnostics).

Example
    src = DataSource(store=ParquetEventStore(settings.paths.events_dir))
    res = BacktestEngine(settings, src, ["momentum_ignition"], metadata=load_metadata(path)).run()
    res.summary()["total_return"]
```

- **`backtester/engine.py`** — Event-driven backtest engine (zero look-ahead by construction).
- **`backtester/execution_sim.py`** — Execution simulator: the path from a decision to an on-chain fill.
- **`backtester/replay.py`** — Replay data sources.
- **`backtester/results.py`** — Backtest result container and persistence (Parquet + JSON, reloadable for reports/dashboard).

## `pumpfun_hft.risk`

```text
Risk management, position management and sizing.

Purpose
    Hard protections that sit between every strategy signal and the market, identical in
    backtests and live trading.

Architecture
    portfolio.py         exact lamport ledger: positions, round-trip TradeRecords, MAE/MFE, equity
    sizing.py            fixed / fixed-risk / Kelly / volatility / confidence / max-exposure sizing,
                         confidence scaling, caps and a liquidity (price-impact) cap
    position_manager.py  stop loss, take-profit ladder (partial exits), trailing stop, breakeven,
                         max hold, pyramiding (scale-in) rules — on liquidation value
    engine.py            RiskEngine: daily/hourly loss, max positions, exposure, per-token,
                         per-creator and per-sector limits, per-token order rate
    breakers.py          circuit breakers: RPC latency, congestion, slippage, failed swaps, drawdown

Data flow
    Signal -> Sizer.size -> RiskEngine.check_entry -> Order ; fills -> Portfolio.apply_fill ->
    RiskEngine.on_fill (breakers) ; equity samples -> RiskEngine.on_equity ; marks ->
    PositionManager.check -> EXIT / SCALE_OUT signals

Inputs / Outputs
    Inputs: signals with confidence, fills, equity samples and mark prices; limits and sizing
    rules from the ``sizing``, ``position`` and ``risk`` config sections.
    Outputs: order sizes in lamports, accept / reject decisions with reasons, exit and scale-out
    signals, breaker state (tripped, reason, trips) and the exact cash / position ledger.

Example
    pf = Portfolio(int(10e9)); risk = RiskEngine(settings.risk, pf, int(0.01e9))
    risk.check_entry(mint, creator, "ai", int(0.2e9), now_ms)
```

- **`risk/breakers.py`** — Circuit breakers: stop opening risk when the environment or the strategy misbehaves.
- **`risk/engine.py`** — Risk engine: hard pre-trade limits + circuit breakers.
- **`risk/portfolio.py`** — Portfolio ledger with exact lamport accounting (shared by backtests, paper and live).
- **`risk/position_manager.py`** — Position management: stops, take-profit ladder (partial exits), trailing exits, time exits and pyramiding rules.
- **`risk/sizing.py`** — Position sizing.

## `pumpfun_hft.optimizer`

```text
Optimisation and validation engine.

Purpose
    Large-scale parameter search that never touches test data, plus statistics that quantify
    how much of a backtest is selection bias.

Architecture
    space.py        ParamSpace: float / int / log-float / categorical; sampling, grids, [0,1]^d encoding
    search.py       grid, random, Bayesian (GP Matérn-5/2 + EI, constant-liar batches), genetic
    objective.py    objective functions with min-trade and max-drawdown constraints
    splits.py       embargoed train / validation / test / live-sim splits (test & live-sim sealed),
                    rolling or anchored walk-forward folds
    overfitting.py  Probabilistic & Deflated Sharpe Ratio, PBO via CSCV
    study.py        OptimizationStudy and WalkForwardAnalysis (process-pool evaluation)

Data flow
    events -> make_splits -> search(train) -> top-k -> validation selection -> DSR/PBO
           -> (explicit unseal) test + live-sim, each evaluated exactly once

Inputs / Outputs
    Inputs: an event frame, token metadata, a strategy name and its search space from
    ``optimizer.spaces``; method, budget, splits and constraints from the ``optimizer`` section.
    Outputs: a trials table (params, train / validation metrics), the selected parameters, DSR,
    PBO, an unseal audit log, and (only when unsealed) one test and one live-sim evaluation.

Example
    res = OptimizationStudy(settings, events, meta_path, "momentum_ignition", method="bayesian").run(final_evaluation=True)
    wf = WalkForwardAnalysis(settings, events, meta_path, "momentum_ignition").run()
```

- **`optimizer/objective.py`** — Optimisation objectives with minimum-trade and drawdown constraints.
- **`optimizer/overfitting.py`** — Backtest-overfitting diagnostics.
- **`optimizer/search.py`** — Search algorithms: grid, random, Bayesian (Gaussian process + expected improvement) and genetic.
- **`optimizer/space.py`** — Parameter spaces: sampling, grids and [0, 1]^d encoding for model-based optimisers.
- **`optimizer/splits.py`** — Chronological data splits with embargo, sealed hold-out sets and walk-forward folds.
- **`optimizer/study.py`** — Optimisation studies and walk-forward analysis on chronological splits.

## `pumpfun_hft.analytics`

```text
Analytics: wallet intelligence, performance metrics, Monte Carlo, charts and reports.

Purpose
    Turn raw activity and backtest results into decisions: which wallets are informed or
    dangerous, how good a run really is (after costs), how fragile it is under resampling and
    execution stress, and how to present all of it.

Architecture
    wallet_intel.py  online, point-in-time wallet profiles: round-trip PnL, posterior "smart"
                     score, sniper / whale / bot / market-maker / insider / rug-wallet labels,
                     union-find insider clusters
    metrics.py       Sharpe, Sortino, Omega, CAGR, Calmar, drawdown statistics, trade metrics
                     (win rate, profit factor, expectancy, Kelly, R multiples, MAE / MFE, streaks),
                     exit-reason categories, monthly / hourly profiles
    montecarlo.py    trade-level Monte Carlo (bootstrap / permutation / block bootstrap with
                     slippage, fee, latency and size perturbations; numba) and path-level
                     re-simulation through the event-driven backtester
    charts.py        shared Plotly theme and figure builders (validated palette, light / dark),
                     escaped sortable HTML tables
    report.py        HTML / PDF / CSV / JSON report generator for a BacktestResult

Data flow
    Events -> WalletIntel.on_trade / on_create / on_token_outcome (point-in-time)
    BacktestResult (equity, trades, fills) -> compute_metrics -> ReportGenerator -> files
    BacktestResult.trades -> trade_monte_carlo -> MonteCarloResult -> report / dashboard

Inputs / Outputs
    Inputs: normalised events, equity / trade / fill frames from the backtester or live session.
    Outputs: metric dicts, wallet tables, Monte Carlo distributions, report files and figures.

Example
    >>> from pumpfun_hft.analytics.metrics import reason_category
    >>> reason_category("take profit L2 +40.0%")
    'take profit'
```

- **`analytics/charts.py`** — Chart theme and figure builders shared by the report generator and the dashboard.
- **`analytics/metrics.py`** — Performance metrics.
- **`analytics/montecarlo.py`** — Monte Carlo validation.
- **`analytics/report.py`** — Performance report generator (runs automatically after every backtest).
- **`analytics/wallet_intel.py`** — Wallet intelligence engine (point-in-time).

## `pumpfun_hft.ml`

```text
Data science / ML module (optional).

Purpose
    Point-in-time prediction models (rug pull, forward return, migration) that plug into the
    strategies without any possibility of learning from the future.

Architecture
    dataset.py    snapshot datasets from the online feature stack + forward labels with windows
    cv.py         PurgedForwardSplit: forward-chaining CV with label purging, embargo, token groups
    models.py     logistic / random forest / XGBoost / LightGBM / CatBoost; CV metrics; native,
                  permutation and SHAP importance; persisted bundles with ``train_end_ms``
    rug_model.py  rug-pull probability (heuristic logistic score or trained model with a
                  look-ahead guard that refuses predictions before the training cut-off)

Data flow
    events -> build_snapshot_dataset -> train_and_evaluate (purged CV) -> bundle.joblib
           -> rug_model.use_trained_model=true -> StrategyRuntime.rug_probability

Inputs / Outputs
    Inputs: an event frame and token metadata (snapshots are built with the same online
    feature engine as live trading); model type, CV and label windows from the ``ml`` section.
    Outputs: per-fold CV metrics (AUC, log loss, Brier, top-decile precision), importance tables (native, permutation,
    SHAP), and a saved model bundle that records the last training timestamp.

Example
    ds = build_snapshot_dataset(settings, events, metadata)
    rep = train_and_evaluate(ds, feature_columns(ds), "rug", settings.ml, seed=7)
    rep.mean["auc"], rep.importance.head()
```

- **`ml/cv.py`** — Purged, embargoed, group-aware forward-chaining cross-validation.
- **`ml/dataset.py`** — Point-in-time ML datasets.
- **`ml/models.py`** — Model training, validation and explainability.
- **`ml/rug_model.py`** — Rug-pull probability model.

## `pumpfun_hft.execution`

```text
Live execution engine.

Purpose
    Turn orders from the shared strategy runtime into confirmed on-chain fills (or paper fills
    on live data) with bounded latency, safe retries and full auditability.

Architecture
    engine.py    LiveTrader: event consumer, priority order queue (EXIT > SELL > BUY), async
                 workers, sweeps, equity/state snapshots for the Live Monitor, breaker feeds
    gateway.py   PaperGateway (backtest execution model on live state) and LiveGateway
                 (QUOTE -> SWAP -> SIGN -> SUBMIT -> CONFIRM -> RETRY), fills parsed from chain
    infra.py     BlockhashCache, PriorityFeeEstimator, ConfirmationTracker, JitoClient
    signer.py    WalletSigner (solders): signing, compute-budget + tip composition
    replay.py    replay_live: recorded events through LiveTrader + PaperGateway on virtual time

Data flow
    LiveStreamCollector -> LiveTrader.on_event -> StrategyRuntime -> queue -> gateway.execute
    -> Fill -> Portfolio / RiskEngine / StrategyRuntime.on_result (retry) ; snapshots -> SQLite

Inputs / Outputs
    Inputs: decoded live events from the stream collector, secrets from ``.env`` (RPC / WS URLs,
    Metis URL and JWT, wallet key for live mode only) and the ``live``, ``priority_fee`` and
    ``network`` config sections.
    Outputs: signed transactions, Fill records reconstructed from confirmed transactions,
    persisted order state (SQLite), trade / latency logs and snapshots for the Live Monitor.

Safety
    * live mode needs ``app.mode: live`` plus ``--confirm-live``; paper mode needs no key
    * retries only after the original blockhash is invalid (no double execution)
    * exits bypass entry halts; ``stop(flatten=True)`` liquidates before shutdown

Example
    trader = LiveTrader(settings, collector.subscribe(), lambda sim: PaperGateway(sim, trader_activity, latency))
    asyncio.run(trader.run())
```

- **`execution/engine.py`** — Live / paper trading engine.
- **`execution/gateway.py`** — Execution gateways: paper (simulated fills on live data) and live (real transactions).
- **`execution/infra.py`** — Live execution infrastructure: blockhash cache, priority-fee estimator, confirmation tracker, Jito.
- **`execution/replay.py`** — Replay recorded events through the live trading engine with the paper gateway.
- **`execution/signer.py`** — Wallet signing (``solders``). The private key is read from ``PRIVATE_KEY`` in ``.env`` only.

## `pumpfun_hft.dashboard`

```text
Interactive analytics dashboard.

Purpose
    Local, offline dashboard over persisted runs, research artefacts and the live session.

Architecture
    app.py    FastAPI app (HTML pages + JSON API + offline Plotly JS) and ``static_export`` (one
              self-contained HTML file with every page as a tab, for sharing)
    views.py  page renderers: Overview, Trades (every fill), Equity curve, Drawdown, Heatmaps
              (weekday x hour PnL), Wallet explorer, Token explorer (price + our fills), Feature
              importance, Strategy comparison, Live monitor (polls the live snapshot)

Data flow
    reports/runs/<run_id>/{result.json, *.parquet, wallets.parquet, tokens.parquet}
    reports/ml/<model_id>/importance.parquet ; SQLite live_state (LiveTrader snapshots)
    events Parquet store (token price paths)

Inputs / Outputs
    Inputs: the files listed under Data flow plus the ``dashboard`` config section (host, port,
    table sizes, live refresh interval).
    Outputs: HTML pages, JSON endpoints (``/api/runs``, ``/api/runs/{id}/metrics``,
    ``/api/runs/{id}/equity``, ``/api/live``; strict JSON, NaN as null) and a single-file export.

Example
    python -m pumpfun_hft.main dashboard            # http://127.0.0.1:8050
    python -m pumpfun_hft.main dashboard-export     # -> reports/dashboard.html
```

- **`dashboard/app.py`** — Local analytics dashboard (FastAPI + Plotly, fully offline).
- **`dashboard/views.py`** — Dashboard page renderers (shared by the FastAPI app and the static single-file export).

## `pumpfun_hft.database`

```text
Storage layer.

Purpose
    DuckDB is the analytical warehouse (derived tables + a view over the Parquet event store);
    SQLite holds small, frequently-updated operational metadata.

Architecture
    schema_duckdb.sql   tokens, token_metadata, creators, wallets, runs, run_trades, run_fills,
                        run_equity, optimizer_trials, ml_models (+ `events` view over Parquet)
    schema_sqlite.sql   collector_checkpoints, file_manifest, gaps, pending_signatures,
                        live_orders, live_state, runs, kv
    warehouse.py        Warehouse (DuckDB): views, upserts, run persistence, SQL -> Polars
    meta.py             MetaStore (SQLite): checkpoints, manifest, gaps, live state

Data flow
    collectors -> Parquet + MetaStore.manifest ; discovery / wallet intel -> Warehouse tables ;
    backtests -> Warehouse run tables ; live engine -> MetaStore.live_* -> dashboard.

Inputs / Outputs
    Inputs: Polars frames (events, runs, trades, fills, wallets, tokens) and operational records
    (checkpoints, file checksums, gaps, pending signatures, order and live state).
    Outputs: SQL query results as Polars frames, idempotent upserts, and the state a restarted
    collector or live session resumes from.

Example
    wh = Warehouse(settings.paths.resolve("duckdb_file"), settings.paths.events_dir)
    df = wh.query("select kind, count(*) n from events group by 1")
```

- **`database/meta.py`** — SQLite metadata store (checkpoints, manifest, gaps, live state, run registry).
- **`database/warehouse.py`** — DuckDB analytical warehouse.

## `pumpfun_hft.utils`

```text
Shared utilities.

Purpose
    Cross-cutting helpers with no dependency on trading logic.

Architecture
    logging.py      JSON, rotating, per-channel logging behind a non-blocking QueueListener,
                    with a redaction filter that scrubs registered secrets.
    latency.py      streaming latency statistics (ring-buffer percentiles, EWMA, budgets).
    retry.py        exponential backoff with full jitter for async callables.
    timeutil.py     wall/monotonic clocks and date helpers (UTC, milliseconds).
    hashing.py      checksums and stable content hashes for reproducibility.
    numba_compat.py optional numba ``njit`` with a pure-Python fallback.
    base58.py       base58 encode/decode (fast path via solders when installed).

Data flow
    Every other package imports from here; nothing here imports trading code. Log records flow
    logger -> redaction -> QueueHandler -> background listener -> per-channel rotating JSON files.

Inputs / Outputs
    Pure functions and small classes; no I/O except logging handlers and file hashing.

Example
    >>> from pumpfun_hft.utils.latency import LatencyTracker
    >>> lt = LatencyTracker(window=128); lt.record("rpc", 12.5); round(lt.percentile("rpc", 50), 1)
    12.5
```

- **`utils/base58.py`** — Base58 (Bitcoin alphabet) encoding used for Solana public keys and signatures.
- **`utils/docgen.py`** — Generate developer documentation (docs/MODULES.md) from package and module docstrings.
- **`utils/hashing.py`** — Checksums and stable hashes used for data integrity and run reproducibility.
- **`utils/jsonutil.py`** — Strict JSON output.
- **`utils/latency.py`** — Streaming latency measurement.
- **`utils/logging.py`** — Structured JSON logging with rotating per-channel files and secret redaction.
- **`utils/numba_compat.py`** — Optional numba acceleration.
- **`utils/retry.py`** — Exponential backoff with full jitter.
- **`utils/timeutil.py`** — Time helpers. All timestamps in the platform are UTC epoch milliseconds (int).
