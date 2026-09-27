"""Event-driven backtester with realistic execution.

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
"""
