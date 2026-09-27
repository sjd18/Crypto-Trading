"""Clock abstraction so strategy / risk / execution code is identical in simulation, replay and live trading.

Clocks share one interface (``now_ms()``; the async ones also ``await sleep(ms)``):

* ``SimClock``     advanced explicitly by the backtest engine (no waiting at all);
* ``WallClock``    real UTC time, for live and paper trading;
* ``ReplayClock``  market time for replaying recorded events through the *live* engine: the event
                   loop's clock, offset so that it starts at the first event's timestamp.

``VirtualTimeEventLoop`` is an asyncio event loop whose clock jumps straight to the next timer
whenever every task is waiting, instead of sleeping. Run the live engine on it with a
``ReplayClock`` and a day of recorded market replays in seconds with exact market-time semantics:
an order that "waits 700 ms to land" lets exactly the events of those 700 ms reach the market
state first, as in the backtester, and the run is deterministic. On an ordinary loop the same
``ReplayClock`` replays in real time.
"""

from __future__ import annotations

import asyncio
import selectors
from typing import Any, Protocol

from pumpfun_hft.utils.timeutil import now_ms


class Clock(Protocol):
    def now_ms(self) -> int: ...


class SimClock:
    """Simulation clock advanced explicitly by the replay engine (never goes backwards)."""

    __slots__ = ("_t",)

    def __init__(self, start_ms: int = 0) -> None:
        self._t = int(start_ms)

    def now_ms(self) -> int:
        return self._t

    def advance_to(self, t_ms: int) -> None:
        if t_ms > self._t:
            self._t = int(t_ms)


class WallClock:
    """Real UTC wall clock."""

    __slots__ = ()

    def now_ms(self) -> int:
        return now_ms()

    async def sleep(self, ms: float) -> None:
        await asyncio.sleep(max(0.0, ms) / 1000.0)


class ReplayClock:
    """Market time for a replay: ``start_ms`` plus the running loop's elapsed time (see module docstring).

    Create it inside the coroutine that runs the replay (it binds to the running loop).

    Example::

        clock = ReplayClock(start_ms=int(events["ts_ms"].min()))
        await clock.sleep(700)          # 700 ms of market time (instant on a VirtualTimeEventLoop)
    """

    __slots__ = ("start_ms", "_loop", "_t0")

    def __init__(self, start_ms: int) -> None:
        self.start_ms = int(start_ms)
        self._loop = asyncio.get_running_loop()
        self._t0 = self._loop.time()

    def now_ms(self) -> int:
        return self.start_ms + int(round((self._loop.time() - self._t0) * 1000.0))

    async def sleep(self, ms: float) -> None:
        await asyncio.sleep(max(0.0, ms) / 1000.0)


class _VirtualSelector:
    """Selector wrapper: polls real file descriptors without blocking and turns waiting into a clock jump."""

    def __init__(self, loop: VirtualTimeEventLoop) -> None:
        self._sel = selectors.DefaultSelector()
        self._loop = loop

    def select(self, timeout: float | None = None) -> list[Any]:
        if timeout is None:  # nothing scheduled: only another thread can wake the loop (e.g. executor shutdown)
            return self._sel.select(None)
        ready = self._sel.select(0)
        if not ready and timeout > 0:
            self._loop.advance(timeout)
        return ready

    def __getattr__(self, name: str) -> Any:  # register / unregister / modify / get_map / get_key / close
        return getattr(self._sel, name)


class VirtualTimeEventLoop(asyncio.SelectorEventLoop):
    """Asyncio loop on virtual time: when every task waits, the clock jumps to the next timer.

    Example::

        result = asyncio.run(main(), loop_factory=VirtualTimeEventLoop)
    """

    def __init__(self) -> None:
        self._virtual_now = 0.0
        super().__init__(selector=_VirtualSelector(self))  # type: ignore[arg-type]

    def time(self) -> float:
        return self._virtual_now

    def advance(self, seconds: float) -> None:
        self._virtual_now += seconds
