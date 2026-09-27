"""Solana WebSocket client with automatic reconnection and resubscription.

* Subscriptions are registered declaratively (method + params + async handler) and replayed
  after every reconnect; the client reports reconnect gaps (last slot seen before / first after)
  so collectors can backfill missed data.
* Every notification is stamped with ``recv_ns`` (``perf_counter_ns``) and ``recv_ms`` (wall
  clock) *before* JSON decoding, so downstream latency (arrival -> dispatch) is measured from the
  true arrival time.
* Handlers run inline on the reader task and must be fast (the live collector only decodes and
  enqueues); heavy work belongs on consumer tasks.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import orjson

from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger

log = get_logger("api")

Handler = Callable[[dict[str, Any], int, int], Awaitable[None]]  # (result, recv_ns, recv_ms)


@dataclass(slots=True)
class Subscription:
    key: str
    method: str
    params: list[Any]
    handler: Handler
    unsubscribe_method: str
    server_id: int | None = None


class SolanaWsClient:
    """Reconnecting JSON-RPC-over-WebSocket client for Solana subscriptions.

    Example::

        ws = SolanaWsClient(url, ping_interval_s=20, ...)
        ws.add_subscription("pump_logs", "logsSubscribe", [{"mentions": [PUMP]}, {"commitment": "processed"}], on_logs)
        await ws.run_forever()   # or: task = asyncio.create_task(ws.run_forever())
    """

    def __init__(self, url: str, *, ping_interval_s: float, ping_timeout_s: float, reconnect_base_delay_s: float,
                 reconnect_max_delay_s: float, latency: LatencyTracker | None = None,
                 connect: Callable[..., Any] | None = None) -> None:
        self.url = url
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.base_delay = reconnect_base_delay_s
        self.max_delay = reconnect_max_delay_s
        self.latency = latency or LatencyTracker()
        self._connect = connect
        self._subs: dict[str, Subscription] = {}
        self._by_server_id: dict[int, Subscription] = {}
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._ws: Any = None
        self._stop = asyncio.Event()
        self.connected = asyncio.Event()
        self.reconnects = 0
        self.messages = 0
        self.last_message_ms = 0
        self.gap_callbacks: list[Callable[[float, float], Awaitable[None]]] = []

    def add_subscription(self, key: str, method: str, params: list[Any], handler: Handler) -> None:
        unsub = method.replace("Subscribe", "Unsubscribe")
        self._subs[key] = Subscription(key, method, params, handler, unsub)

    def on_gap(self, cb: Callable[[float, float], Awaitable[None]]) -> None:
        """Register a coroutine called with (disconnected_at_ms, reconnected_at_ms)."""
        self.gap_callbacks.append(cb)

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            await self._ws.close()

    async def _open(self) -> Any:
        if self._connect is not None:
            return await self._connect(self.url)
        from websockets.asyncio.client import connect

        return await connect(self.url, ping_interval=self.ping_interval_s, ping_timeout=self.ping_timeout_s,
                             max_size=2**24, compression=None)

    async def request(self, method: str, params: list[Any]) -> Any:
        """Send a JSON-RPC request over the socket and await its response."""
        if self._ws is None:
            raise ConnectionError("websocket not connected")
        rid = next(self._ids)
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send(orjson.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).decode())
        return await asyncio.wait_for(fut, timeout=10.0)

    async def _subscribe_all(self) -> None:
        self._by_server_id.clear()
        for sub in self._subs.values():
            sid = await self.request(sub.method, sub.params)
            sub.server_id = int(sid)
            self._by_server_id[sub.server_id] = sub
        log.info("ws subscribed", extra={"data": {"subs": list(self._subs)}})

    async def _reader(self) -> None:
        async for raw in self._ws:
            recv_ns = time.perf_counter_ns()
            recv_ms = time.time_ns() // 1_000_000
            self.messages += 1
            self.last_message_ms = recv_ms
            msg = orjson.loads(raw)
            if "id" in msg and msg.get("id") in self._pending:
                fut = self._pending.pop(msg["id"])
                if "error" in msg and msg["error"]:
                    fut.set_exception(ConnectionError(str(msg["error"])))
                else:
                    fut.set_result(msg.get("result"))
                continue
            params = msg.get("params")
            if not params:
                continue
            sub = self._by_server_id.get(params.get("subscription"))
            if sub is None:
                continue
            try:
                await sub.handler(params.get("result") or {}, recv_ns, recv_ms)
            except Exception:  # noqa: BLE001 - a handler bug must not kill the socket
                log.exception("ws handler error", extra={"data": {"sub": sub.key}})
            self.latency.record_ns("ws.handler", recv_ns)

    async def run_forever(self) -> None:
        """Connect, subscribe and read until :meth:`stop`; reconnects with jittered backoff."""
        attempt = 0
        disconnected_at: float | None = None
        while not self._stop.is_set():
            reader: asyncio.Task[None] | None = None
            try:
                self._ws = await self._open()
                self.connected.set()
                reader = asyncio.create_task(self._reader())
                await self._subscribe_all()
                if disconnected_at is not None:
                    now = time.time() * 1000
                    for cb in self.gap_callbacks:
                        await cb(disconnected_at, now)
                    disconnected_at = None
                attempt = 0
                await reader
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("ws disconnected", extra={"data": {"err": repr(exc)[:200], "attempt": attempt}})
            finally:
                self.connected.clear()
                if reader is not None and not reader.done():
                    reader.cancel()
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("disconnected"))
                self._pending.clear()
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:  # noqa: BLE001
                        pass
            if self._stop.is_set():
                break
            if disconnected_at is None:
                disconnected_at = time.time() * 1000
            self.reconnects += 1
            delay = random.uniform(0, min(self.max_delay, self.base_delay * 2**attempt))
            attempt += 1
            await asyncio.sleep(delay)
