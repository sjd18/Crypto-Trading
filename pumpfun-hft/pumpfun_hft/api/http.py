"""Resilient async HTTP client.

Every request passes through: adaptive rate limiter -> circuit breaker -> httpx request ->
latency recording -> retry decision. Retries use exponential backoff with full jitter and honour
``Retry-After``; 429 responses additionally shrink the adaptive rate. After
``failure_threshold`` consecutive failures the breaker *opens* and requests fail fast with
:class:`ServiceUnavailable` until ``reset_timeout_s`` elapses (then one probe is allowed).
This is how API outages are handled without hammering a failing service.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
import orjson

from pumpfun_hft.api.rate_limit import AdaptiveRateLimiter
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger, redact
from pumpfun_hft.utils.retry import RetryPolicy

log = get_logger("api")


class HttpError(Exception):
    """Non-retryable HTTP failure (4xx other than those configured as retryable)."""

    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} for {redact(url)}: {redact(body[:300])}")
        self.status = status
        self.body = body


class ServiceUnavailable(Exception):
    """Raised when the circuit breaker is open or retries are exhausted."""


@dataclass(slots=True)
class BreakerState:
    failures: int = 0
    opened_at: float = 0.0
    open: bool = False


class CircuitBreaker:
    """Consecutive-failure circuit breaker with half-open probing."""

    def __init__(self, failure_threshold: int, reset_timeout_s: float) -> None:
        self.threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.state = BreakerState()

    def allow(self) -> bool:
        if not self.state.open:
            return True
        return time.monotonic() - self.state.opened_at >= self.reset_timeout_s  # half-open probe

    def success(self) -> None:
        self.state = BreakerState()

    def failure(self) -> None:
        self.state.failures += 1
        if self.state.failures >= self.threshold:
            if not self.state.open:
                log.warning("circuit breaker opened", extra={"data": {"failures": self.state.failures}})
            self.state.open = True
            self.state.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return self.state.open and not self.allow()


class AsyncHttpClient:
    """Named HTTP client (one per service) with limits, retries, breaker and latency stats.

    Example::

        client = AsyncHttpClient("metis", base_url, rps=4, burst=4, retry=policy, ...)
        data = await client.get_json("/pump-fun/quote", params={...})
    """

    def __init__(
        self,
        name: str,
        base_url: str = "",
        *,
        rps: float,
        burst: int,
        retry: RetryPolicy,
        retry_statuses: list[int] | tuple[int, ...],
        timeout_s: float,
        breaker_threshold: int,
        breaker_reset_s: float,
        latency: LatencyTracker | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        auth_header: Any = None,
    ) -> None:
        self.name = name
        self.retry = retry
        self.retry_statuses = frozenset(retry_statuses)
        self.limiter = AdaptiveRateLimiter(rps, burst)
        self.breaker = CircuitBreaker(breaker_threshold, breaker_reset_s)
        self.latency = latency or LatencyTracker()
        self._auth_header = auth_header  # callable returning dict of headers (e.g. JWT provider)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
            headers=headers or {},
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=30.0),
            http2=False,
        )
        self._rng = random.Random()
        self.requests = 0
        self.errors = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncHttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def request(self, method: str, url: str, *, params: dict[str, Any] | None = None,
                      json_body: Any = None, headers: dict[str, str] | None = None) -> httpx.Response:
        """Perform a request with rate limiting, breaker, retries and latency tracking."""
        attempt = 0
        last_exc: Exception | None = None
        body = orjson.dumps(json_body) if json_body is not None else None
        while attempt < self.retry.max_attempts:
            if not self.breaker.allow():
                raise ServiceUnavailable(f"{self.name}: circuit open")
            await self.limiter.acquire()
            hdrs = dict(headers or {})
            if body is not None:
                hdrs.setdefault("Content-Type", "application/json")
            if self._auth_header is not None:
                hdrs.update(await self._auth_header())
            t0 = time.perf_counter_ns()
            try:
                resp = await self._client.request(method, url, params=params, content=body, headers=hdrs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.latency.record_ns(f"api.{self.name}", t0)
                self.breaker.failure()
                self.errors += 1
                last_exc = exc
                attempt += 1
                delay = self.retry.delay(attempt - 1, self._rng)
                log.warning("transport error", extra={"data": {"svc": self.name, "err": type(exc).__name__, "attempt": attempt}})
                await asyncio.sleep(delay)
                continue
            ms = self.latency.record_ns(f"api.{self.name}", t0)
            self.requests += 1
            if resp.status_code < 400:
                self.breaker.success()
                self.limiter.on_success()
                if ms > 1000:
                    log.info("slow request", extra={"data": {"svc": self.name, "ms": round(ms, 1), "url": redact(str(resp.url))}})
                return resp
            retry_after = _retry_after(resp)
            if resp.status_code == 429:
                self.limiter.on_throttle(retry_after)
            if resp.status_code in self.retry_statuses:
                self.breaker.failure()
                self.errors += 1
                attempt += 1
                last_exc = HttpError(resp.status_code, resp.text, str(resp.url))
                delay = retry_after if retry_after is not None else self.retry.delay(attempt - 1, self._rng)
                log.warning("retryable status", extra={"data": {"svc": self.name, "status": resp.status_code, "attempt": attempt, "delay_s": round(delay, 3)}})
                await asyncio.sleep(delay)
                continue
            self.errors += 1
            raise HttpError(resp.status_code, resp.text, str(resp.url))
        raise ServiceUnavailable(f"{self.name}: retries exhausted ({last_exc!r})")

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self.request("GET", url, params=params)
        return orjson.loads(resp.content)

    async def post_json(self, url: str, body: Any) -> Any:
        resp = await self.request("POST", url, json_body=body)
        return orjson.loads(resp.content)


def _retry_after(resp: httpx.Response) -> float | None:
    val = resp.headers.get("retry-after")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        return None


def build_client(name: str, base_url: str, rate: Any, network: Any, timeout_s: float,
                 latency: LatencyTracker | None = None, auth_header: Any = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> AsyncHttpClient:
    """Construct a client from config sections (``rate`` is a RateLimitCfg)."""
    return AsyncHttpClient(
        name, base_url,
        rps=rate.rps, burst=rate.burst,
        retry=RetryPolicy(network.retry.max_attempts, network.retry.base_delay_s, network.retry.max_delay_s),
        retry_statuses=network.retry.retry_statuses,
        timeout_s=timeout_s,
        breaker_threshold=network.http_breaker.failure_threshold,
        breaker_reset_s=network.http_breaker.reset_timeout_s,
        latency=latency,
        headers={"User-Agent": network.user_agent},
        transport=transport,
        auth_header=auth_header,
    )
