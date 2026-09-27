"""Network API layer.

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
"""
