"""Token discovery and creator intelligence.

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
"""
