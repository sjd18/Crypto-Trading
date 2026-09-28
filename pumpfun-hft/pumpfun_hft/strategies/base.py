"""Strategy framework.

A strategy is a small, stateless-by-default object with a validated ``Params`` model and a
single decision method::

    def generate_signal(self, ctx: StrategyContext) -> Signal

returning one of BUY, SELL, HOLD, EXIT, SCALE_IN, SCALE_OUT with a 0-100 confidence (and
optionally an explicit size, order type, limit price, expected return for the cost gate and
per-position exit overrides). Strategies never place orders or touch the portfolio: sizing,
risk limits, cost gating and execution are handled by :class:`~pumpfun_hft.strategies.runtime.StrategyRuntime`,
identically in backtests, paper trading and live trading.

``StrategyContext`` only exposes information available at ``ctx.now_ms``; expensive items
(creator score, rug probability) are computed lazily and cached per context.

Registering a new strategy::

    @register
    class MyStrategy(Strategy):
        name = "my_strategy"
        class Params(StrategyParams):
            threshold: float
        def generate_signal(self, ctx):
            if ctx.f.imbalance_medium > self.p.threshold:
                return Signal(Action.BUY, 70, "flow")
            return hold()

and add ``strategy.params.my_strategy`` to the YAML config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from pumpfun_hft.core.types import Event, EventKind, Fill, Signal

if TYPE_CHECKING:
    from pumpfun_hft.analytics.wallet_intel import WalletIntel
    from pumpfun_hft.discovery.creator import CreatorScore
    from pumpfun_hft.features.market import TokenState
    from pumpfun_hft.features.online import FeatureView
    from pumpfun_hft.risk.portfolio import Position


class StrategyParams(BaseModel):
    """Base for strategy parameter models (all values come from YAML; unknown keys rejected)."""

    model_config = ConfigDict(extra="forbid")


class StrategyContext:
    """Point-in-time view handed to strategies."""

    __slots__ = ("now_ms", "event", "token", "f", "position", "pending", "wallets", "equity_sol", "exposure_sol",
                 "n_positions", "_runtime", "_creator", "_rug", "discovered", "position_strategy")

    def __init__(self, runtime: Any, now_ms: int, event: Event, token: TokenState, f: FeatureView,
                 position: Position | None, pending: bool, wallets: WalletIntel, equity_sol: float,
                 exposure_sol: float, n_positions: int, discovered: Any = None) -> None:
        self._runtime = runtime
        self.now_ms = now_ms
        self.event = event
        self.token = token
        self.f = f
        self.position = position
        self.position_strategy = position.strategy if position is not None else None
        self.pending = pending
        self.wallets = wallets
        self.equity_sol = equity_sol
        self.exposure_sol = exposure_sol
        self.n_positions = n_positions
        self.discovered = discovered
        self._creator: CreatorScore | None = None
        self._rug: float | None = None

    @property
    def creator_score(self) -> CreatorScore:
        if self._creator is None:
            self._creator = self._runtime.creator_score(self.token.creator)
        return self._creator

    @property
    def rug_prob(self) -> float:
        if self._rug is None:
            self._rug = self._runtime.rug_probability(self)
        return self._rug

    @property
    def age_s(self) -> float:
        c = self.token.created_ms
        return (self.now_ms - c) / 1000.0 if c is not None else 0.0

    @property
    def is_trade(self) -> bool:
        return self.event.kind in (EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value)

    @property
    def has_position(self) -> bool:
        return self.position is not None and self.position.tokens > 0

    def position_return(self) -> float:
        """Economic return of the open position at liquidation value (fees & impact included)."""
        return self._runtime.position_return(self.position) if self.position is not None else 0.0

    def round_trip_cost(self, size_sol: float) -> float:
        """Fractional round-trip cost (fees + impact both ways + tx costs) for ``size_sol``."""
        return self._runtime.round_trip_cost(self.token.mint, size_sol)


class Strategy(ABC):
    """Base class for strategies."""

    name: ClassVar[str] = "base"
    Params: ClassVar[type[StrategyParams]] = StrategyParams
    #: event kinds this strategy wants to be evaluated on (entries)
    entry_events: ClassVar[frozenset[str]] = frozenset({EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value})

    def __init__(self, params: dict[str, Any] | StrategyParams) -> None:
        self.p = params if isinstance(params, StrategyParams) else self.Params.model_validate(params)

    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> Signal:
        """Return the strategy's signal for this context (HOLD when nothing to do)."""

    def veto(self, ctx: StrategyContext) -> str | None:
        """Overlays may veto new entries (return a reason string)."""
        return None

    def on_fill(self, fill: Fill, ctx: StrategyContext | None) -> None:  # noqa: B027 - optional hook
        """Notification after a confirmed fill of an order this strategy originated."""

    def params_dict(self) -> dict[str, Any]:
        return self.p.model_dump()


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in STRATEGY_REGISTRY:
        raise ValueError(f"duplicate strategy name {cls.name}")
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def build_strategy(name: str, settings: Any, overrides: dict[str, Any] | None = None) -> Strategy:
    """Instantiate a registered strategy with YAML params (+ optional overrides for optimisation)."""
    from pumpfun_hft import strategies as _pkg  # noqa: F401  (ensures registration)

    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(STRATEGY_REGISTRY)}")
    params = dict(settings.strategy.params.get(name, {}))
    if overrides:
        params.update(overrides)
    strat = STRATEGY_REGISTRY[name](params)
    bind = getattr(strat, "bind_settings", None)
    if bind is not None:  # strategies that need more than their params (e.g. ml_signal loads its model file)
        bind(settings)
    return strat
