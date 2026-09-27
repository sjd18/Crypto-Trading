"""Live execution engine.

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
"""
