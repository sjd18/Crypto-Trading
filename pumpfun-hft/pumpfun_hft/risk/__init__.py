"""Risk management, position management and sizing.

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
"""
