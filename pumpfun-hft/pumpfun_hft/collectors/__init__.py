"""Market data collection.

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
                                          \\->  subscribers (discovery, features, strategies)

Inputs / Outputs
    Inputs: Settings.collector / protocol, RPC & WS clients. Outputs: Parquet files under
    ``paths.data_dir/events``, manifest/checkpoint/gap rows in SQLite, in-memory Event queues.

Example
    store = ParquetEventStore(settings.paths.events_dir, MetaStore(settings.paths.resolve("sqlite_file")))
    SyntheticMarket(settings).write(store, settings.paths.metadata_dir / "tokens.parquet")
    store.verify()   # -> [] when every checksum matches
"""
