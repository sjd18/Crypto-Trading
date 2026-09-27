"""Incremental (online) feature engine — O(1) amortised work per event.

Design
    * One :class:`TokenFeatures` per mint holds sliding time windows (short, 2*short, medium,
      long) with running aggregates, monotonic deques for window highs, time-decayed EWMAs, a
      bar aggregator (ATR, bar-volume statistics), holder balances with a running sum of
      squares (HHI) and as-of buffers for lagged values.
    * Features are *read* through a lazy :class:`FeatureView` (only what a strategy touches is
      computed). Windows are evicted relative to the evaluation time ``now``.
    * Wallet-derived flags (smart / whale / fresh / bot) are evaluated against the wallet
      database *before* that database ingests the current trade, so no trade informs its own
      classification.

The same engine runs in backtests and live trading, which is what makes backtest features
identical to live features. :mod:`pumpfun_hft.features.batch` re-implements a subset with
vectorised Polars for research; ``tests/test_features.py`` checks both agree.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_right
from collections import Counter, deque
from typing import Any

from pumpfun_hft.core.curve import BondingCurve
from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Event, EventKind
from pumpfun_hft.features.market import TokenState
from pumpfun_hft.features.registry import FEATURE_NAMES

_TRADE_KINDS = frozenset({EventKind.TRADE.value, EventKind.AMM_BUY.value, EventKind.AMM_SELL.value})
_LN2 = math.log(2.0)
TWO_PI = 2.0 * math.pi


class TimeSeriesBuffer:
    """(ts, value) buffer with amortised head eviction and as-of lookups."""

    __slots__ = ("ts", "vals", "head", "span")

    def __init__(self, span_ms: int) -> None:
        self.ts: list[int] = []
        self.vals: list[float] = []
        self.head = 0
        self.span = span_ms

    def append(self, t: int, v: float) -> None:
        self.ts.append(t)
        self.vals.append(v)

    def evict(self, now: int) -> None:
        cutoff = now - self.span
        ts, h, n = self.ts, self.head, len(self.ts)
        while h < n - 1 and ts[h + 1] <= cutoff:  # keep one point at/before the cutoff for as-of lookups
            h += 1
        self.head = h
        if h > 2048 and h > n // 2:
            del self.ts[:h]
            del self.vals[:h]
            self.head = 0

    def asof(self, t: int) -> float | None:
        i = bisect_right(self.ts, t, self.head) - 1
        return self.vals[i] if i >= self.head else None


class SlidingMax:
    """Monotonic deque maintaining the max of values within a time span."""

    __slots__ = ("span", "dq")

    def __init__(self, span_ms: int) -> None:
        self.span = span_ms
        self.dq: deque[tuple[int, float]] = deque()

    def push(self, t: int, v: float) -> None:
        dq = self.dq
        while dq and dq[-1][1] <= v:
            dq.pop()
        dq.append((t, v))

    def value(self, now: int) -> float | None:
        dq, cutoff = self.dq, now - self.span
        while dq and dq[0][0] < cutoff:
            dq.popleft()
        return dq[0][1] if dq else None


class TradeWindow:
    """Sliding time window of trades with running aggregates."""

    __slots__ = ("span", "q", "buy", "sell", "n", "agg_buy", "whale", "smart_buy", "pv", "v", "sum_r", "sum_r2", "n_r",
                 "buyers", "fresh", "bots")

    def __init__(self, span_ms: int) -> None:
        self.span = span_ms
        self.q: deque[tuple[Any, ...]] = deque()
        self.buy = self.sell = self.agg_buy = self.whale = self.smart_buy = 0.0
        self.pv = self.v = self.sum_r = self.sum_r2 = 0.0
        self.n = self.n_r = 0
        self.buyers: Counter[str] = Counter()
        self.fresh: Counter[str] = Counter()
        self.bots: Counter[str] = Counter()

    def add(self, rec: tuple[Any, ...]) -> None:
        # rec = (ts, is_buy, sol, price, user, aggressive, whale, smart, fresh, bot, logret, has_ret)
        self.q.append(rec)
        _, is_buy, sol, price, user, aggressive, whale, smart, fresh, bot, r, has_r = rec
        self.n += 1
        self.pv += price * sol
        self.v += sol
        if has_r:
            self.sum_r += r
            self.sum_r2 += r * r
            self.n_r += 1
        if whale:
            self.whale += sol
        if is_buy:
            self.buy += sol
            if aggressive:
                self.agg_buy += sol
            if smart:
                self.smart_buy += sol
            self.buyers[user] += 1
            if fresh:
                self.fresh[user] += 1
            if bot:
                self.bots[user] += 1
        else:
            self.sell += sol

    def evict(self, now: int) -> None:
        q, cutoff = self.q, now - self.span
        while q and q[0][0] < cutoff:
            _, is_buy, sol, price, user, aggressive, whale, smart, fresh, bot, r, has_r = q.popleft()
            self.n -= 1
            self.pv -= price * sol
            self.v -= sol
            if has_r:
                self.sum_r -= r
                self.sum_r2 -= r * r
                self.n_r -= 1
            if whale:
                self.whale -= sol
            if is_buy:
                self.buy -= sol
                if aggressive:
                    self.agg_buy -= sol
                if smart:
                    self.smart_buy -= sol
                _dec(self.buyers, user)
                if fresh:
                    _dec(self.fresh, user)
                if bot:
                    _dec(self.bots, user)
            else:
                self.sell -= sol
        if not q:  # reset float drift
            self.buy = self.sell = self.agg_buy = self.whale = self.smart_buy = 0.0
            self.pv = self.v = self.sum_r = self.sum_r2 = 0.0
            self.n = self.n_r = 0

    @property
    def imbalance(self) -> float:
        tot = self.buy + self.sell
        return (self.buy - self.sell) / tot if tot > 1e-12 else 0.0

    @property
    def vwap(self) -> float | None:
        return self.pv / self.v if self.v > 1e-12 else None


def _dec(c: Counter[str], key: str) -> None:
    n = c[key] - 1
    if n <= 0:
        del c[key]
    else:
        c[key] = n


class TokenFeatures:
    """Incremental per-token feature state."""

    __slots__ = ("mint", "w_short", "w_short2", "w_med", "w_long", "prices", "r_sol_hist", "prog_hist", "hi_short",
                 "hi_med", "ema_fast", "ema_slow", "ema_t", "bar_start", "bar_high", "bar_low", "bar_close", "bar_vol",
                 "prev_close", "atr", "n_bars", "vol_mean", "vol_var", "holders", "sumsq", "circ", "creator", "last_log_ret",
                 "created_ms", "_top_cache", "_top_ver", "_ver")

    def __init__(self, mint: str, cfg: Any, created_ms: int | None) -> None:
        self.mint = mint
        self.w_short = TradeWindow(cfg.short_window_ms)
        self.w_short2 = TradeWindow(2 * cfg.short_window_ms)
        self.w_med = TradeWindow(cfg.medium_window_ms)
        self.w_long = TradeWindow(cfg.long_window_ms)
        span = 2 * max(cfg.long_window_ms, cfg.medium_window_ms)
        self.prices = TimeSeriesBuffer(span)
        self.r_sol_hist = TimeSeriesBuffer(2 * cfg.medium_window_ms)
        self.prog_hist = TimeSeriesBuffer(4 * cfg.short_window_ms)
        self.hi_short = SlidingMax(cfg.short_window_ms)
        self.hi_med = SlidingMax(cfg.medium_window_ms)
        self.ema_fast = self.ema_slow = None
        self.ema_t = 0
        self.bar_start = -1
        self.bar_high = self.bar_low = self.bar_close = 0.0
        self.bar_vol = 0.0
        self.prev_close = None
        self.atr = None
        self.n_bars = 0
        self.vol_mean = 0.0
        self.vol_var = 0.0
        self.holders: dict[str, int] = {}
        self.sumsq = 0
        self.circ = 0
        self.creator: str | None = None
        self.last_log_ret = 0.0
        self.created_ms = created_ms
        self._top_cache = 0
        self._top_ver = -1
        self._ver = 0


class FeatureEngine:
    """Maintains TokenFeatures for all mints plus market-wide activity statistics.

    Example::

        fe = FeatureEngine(settings.features, curve, wallet_intel)
        fe.on_event(ev, token_state)      # in event order
        view = fe.view(mint, now_ms, token_state)
        view.imbalance_medium, view.as_dict()
    """

    def __init__(self, cfg: Any, curve: BondingCurve, wallet_intel: Any | None = None) -> None:
        self.cfg = cfg
        self.curve = curve
        self.wi = wallet_intel
        self.tokens: dict[str, TokenFeatures] = {}
        self.tau_fast = cfg.fast_half_life_ms / _LN2
        self.tau_slow = cfg.slow_half_life_ms / _LN2
        self.tau_tps = cfg.tps_half_life_ms / _LN2
        self.alpha_v = 2.0 / (cfg.atr_bars + 1.0)
        self.tps = 0.0
        self._tps_t = 0
        self.slot_vel = 0.0
        self._slot_t = 0
        self._slot = 0

    def get(self, mint: str) -> TokenFeatures | None:
        return self.tokens.get(mint)

    def _token(self, mint: str, st: TokenState) -> TokenFeatures:
        tf = self.tokens.get(mint)
        if tf is None:
            tf = TokenFeatures(mint, self.cfg, st.created_ms)
            tf.creator = st.creator
            self.tokens[mint] = tf
        return tf

    def on_event(self, ev: Event, st: TokenState | None) -> TokenFeatures | None:
        """Update with one event (market-wide stats for every event, token stats for trades)."""
        t = ev.ts_ms
        # market-wide activity (time-decayed event rate) and slot velocity
        if self._tps_t:
            self.tps *= math.exp(-max(0, t - self._tps_t) / self.tau_tps)
        self.tps += 1000.0 / self.tau_tps
        self._tps_t = t
        if ev.slot > self._slot:
            if self._slot and t > self._slot_t:
                inst = (ev.slot - self._slot) * 1000.0 / (t - self._slot_t)
                self.slot_vel = inst if self.slot_vel == 0 else 0.9 * self.slot_vel + 0.1 * inst
            self._slot, self._slot_t = ev.slot, t
        if st is None or ev.mint is None:
            return None
        if ev.kind == EventKind.CREATE.value:
            tf = self._token(ev.mint, st)
            tf.created_ms = st.created_ms
            tf.creator = st.creator
            tf.prices.append(t, st.price)
            return tf
        if ev.kind not in _TRADE_KINDS:
            return self.tokens.get(ev.mint)
        tf = self._token(ev.mint, st)
        if tf.creator is None:
            tf.creator = st.creator
        self._on_trade(tf, ev, st, t)
        return tf

    def _on_trade(self, tf: TokenFeatures, ev: Event, st: TokenState, t: int) -> None:  # noqa: PLR0915
        cfg = self.cfg
        price = st.price
        prev = st.prev_price
        sol = (ev.sol_amount or 0) / LAMPORTS_PER_SOL
        is_buy = bool(ev.is_buy)
        user = ev.user or ""
        has_r = prev > 0 and price > 0
        r = math.log(price / prev) if has_r else 0.0
        tf.last_log_ret = r
        aggressive = is_buy and has_r and (price / prev - 1.0) * 1e4 >= cfg.aggressive_impact_bps
        wi = self.wi
        whale = sol >= cfg.whale_trade_sol or (wi is not None and wi.is_whale(user))
        smart = False
        fresh = False
        bot = False
        if wi is not None and is_buy:
            sc = wi.score(user)
            smart = sc is not None and sc >= cfg.smart_score_threshold
            fresh = wi.is_fresh(user, t)
            bot = wi.is_bot(user)
        rec = (t, is_buy, sol, price, user, aggressive, whale, smart, fresh, bot, r, has_r)
        for w in (tf.w_short, tf.w_short2, tf.w_med, tf.w_long):
            w.add(rec)
            w.evict(t)
        tf.prices.append(t, price)
        tf.prices.evict(t)
        tf.hi_short.push(t, price)
        tf.hi_med.push(t, price)
        liq = st.pool.quote if (st.migrated and st.pool is not None) else st.curve.r_sol
        tf.r_sol_hist.append(t, liq / LAMPORTS_PER_SOL)
        tf.r_sol_hist.evict(t)
        tf.prog_hist.append(t, self.curve.progress_pct(st.curve))
        tf.prog_hist.evict(t)
        # time-decayed EWMAs of log price
        lp = math.log(price) if price > 0 else 0.0
        if tf.ema_fast is None:
            tf.ema_fast = tf.ema_slow = lp
        else:
            dt = max(0, t - tf.ema_t)
            a_f = 1.0 - math.exp(-dt / self.tau_fast)
            a_s = 1.0 - math.exp(-dt / self.tau_slow)
            tf.ema_fast += a_f * (lp - tf.ema_fast)
            tf.ema_slow += a_s * (lp - tf.ema_slow)
        tf.ema_t = t
        # bars (ATR and bar-volume statistics); empty bars count as zero volume
        bar = t - t % cfg.bar_ms
        if tf.bar_start < 0:
            tf.bar_start, tf.bar_high, tf.bar_low, tf.bar_close, tf.bar_vol = bar, price, price, price, 0.0
        elif bar > tf.bar_start:
            self._close_bar(tf)
            empty = min(50, (bar - tf.bar_start) // cfg.bar_ms - 1)
            for _ in range(int(empty)):
                self._bar_volume(tf, 0.0)
            tf.bar_start, tf.bar_high, tf.bar_low, tf.bar_close, tf.bar_vol = bar, price, price, price, 0.0
        tf.bar_high = max(tf.bar_high, price)
        tf.bar_low = min(tf.bar_low, price)
        tf.bar_close = price
        tf.bar_vol += sol
        # holders (reconstructed from curve / pool trades)
        tok = int(ev.token_amount or 0)
        old = tf.holders.get(user, 0)
        new = old + tok if is_buy else max(0, old - tok)
        if new:
            tf.holders[user] = new
        elif old:
            del tf.holders[user]
        tf.sumsq += new * new - old * old
        tf.circ += new - old
        tf._ver += 1

    def _close_bar(self, tf: TokenFeatures) -> None:
        if tf.prev_close is not None:
            tr = max(tf.bar_high - tf.bar_low, abs(tf.bar_high - tf.prev_close), abs(tf.bar_low - tf.prev_close))
        else:
            tr = tf.bar_high - tf.bar_low
        n = self.cfg.atr_bars
        tf.atr = tr if tf.atr is None else ((n - 1) * tf.atr + tr) / n
        tf.prev_close = tf.bar_close
        self._bar_volume(tf, tf.bar_vol)

    def _bar_volume(self, tf: TokenFeatures, v: float) -> None:
        a = self.alpha_v
        diff = v - tf.vol_mean
        tf.vol_mean += a * diff
        tf.vol_var = (1 - a) * (tf.vol_var + a * diff * diff)
        tf.n_bars += 1

    def view(self, mint: str, now_ms: int, st: TokenState) -> FeatureView | None:
        tf = self.tokens.get(mint)
        if tf is None:
            return None
        return FeatureView(self, tf, st, now_ms)


class FeatureView:
    """Lazy, read-only feature accessor evaluated at ``now`` (windows evicted to ``now``)."""

    __slots__ = ("eng", "tf", "st", "now", "cfg")

    def __init__(self, eng: FeatureEngine, tf: TokenFeatures, st: TokenState, now: int) -> None:
        self.eng, self.tf, self.st, self.now, self.cfg = eng, tf, st, now, eng.cfg
        for w in (tf.w_short, tf.w_short2, tf.w_med, tf.w_long):
            w.evict(now)

    # ---- price
    @property
    def price(self) -> float:
        return self.st.price

    @property
    def log_ret(self) -> float:
        return self.tf.last_log_ret

    def _ret(self, window_ms: int) -> float:
        p0 = self.tf.prices.asof(self.now - window_ms)
        if p0 is None:
            p0 = self.st.launch_price
        return math.log(self.st.price / p0) if p0 and self.st.price > 0 else 0.0

    @property
    def ret_short(self) -> float:
        return self._ret(self.cfg.short_window_ms)

    @property
    def ret_medium(self) -> float:
        return self._ret(self.cfg.medium_window_ms)

    @property
    def ret_long(self) -> float:
        return self._ret(self.cfg.long_window_ms)

    @property
    def momentum(self) -> float:
        tf = self.tf
        if tf.ema_fast is None:
            return 0.0
        # decay EWMAs to `now` with the current price (no new information)
        dt = max(0, self.now - tf.ema_t)
        lp = math.log(self.st.price) if self.st.price > 0 else 0.0
        f = tf.ema_fast + (1.0 - math.exp(-dt / self.eng.tau_fast)) * (lp - tf.ema_fast)
        s = tf.ema_slow + (1.0 - math.exp(-dt / self.eng.tau_slow)) * (lp - tf.ema_slow)
        return f - s

    @property
    def vwap_medium(self) -> float | None:
        return self.tf.w_med.vwap

    @property
    def vwap_dist_medium(self) -> float:
        v = self.tf.w_med.vwap
        return self.st.price / v - 1.0 if v else 0.0

    @property
    def atr_pct(self) -> float:
        return (self.tf.atr / self.st.price) if (self.tf.atr is not None and self.st.price > 0) else 0.0

    @property
    def rv_short(self) -> float:
        return math.sqrt(max(self.tf.w_short.sum_r2, 0.0))

    @property
    def rv_medium(self) -> float:
        return math.sqrt(max(self.tf.w_med.sum_r2, 0.0))

    @property
    def var_medium(self) -> float:
        w = self.tf.w_med
        if w.n_r < 2:
            return 0.0
        mean = w.sum_r / w.n_r
        return max((w.sum_r2 - w.n_r * mean * mean) / (w.n_r - 1), 0.0)

    @property
    def micro_imbalance(self) -> float:
        return self.tf.w_short.imbalance

    @property
    def ath_multiple(self) -> float:
        return self.st.ath_multiple

    @property
    def drawdown_from_ath_pct(self) -> float:
        return 100.0 * (1.0 - self.st.price / self.st.ath_price) if self.st.ath_price > 0 else 0.0

    @property
    def high_medium(self) -> float:
        v = self.tf.hi_med.value(self.now)
        return v if v is not None else self.st.price

    @property
    def high_short(self) -> float:
        v = self.tf.hi_short.value(self.now)
        return v if v is not None else self.st.price

    @property
    def high_medium_dist(self) -> float:
        h = self.high_medium
        return self.st.price / h - 1.0 if h > 0 else 0.0

    # ---- volume / order flow
    @property
    def buy_sol_short(self) -> float:
        return self.tf.w_short.buy

    @property
    def sell_sol_short(self) -> float:
        return self.tf.w_short.sell

    @property
    def buy_sol_medium(self) -> float:
        return self.tf.w_med.buy

    @property
    def sell_sol_medium(self) -> float:
        return self.tf.w_med.sell

    @property
    def delta_sol_medium(self) -> float:
        return self.tf.w_med.buy - self.tf.w_med.sell

    @property
    def volume_sol_short(self) -> float:
        return self.tf.w_short.v

    @property
    def volume_sol_medium(self) -> float:
        return self.tf.w_med.v

    @property
    def volume_sol_long(self) -> float:
        return self.tf.w_long.v

    @property
    def imbalance_short(self) -> float:
        return self.tf.w_short.imbalance

    @property
    def imbalance_medium(self) -> float:
        return self.tf.w_med.imbalance

    @property
    def imbalance_long(self) -> float:
        return self.tf.w_long.imbalance

    @property
    def volume_accel(self) -> float:
        s = self.cfg.short_window_ms / 1000.0
        recent = self.tf.w_short.v
        prior = max(self.tf.w_short2.v - recent, 0.0)
        return (recent / s - prior / s) / s

    @property
    def volume_z(self) -> float:
        tf = self.tf
        if tf.n_bars < 3:
            return 0.0
        scale = self.cfg.short_window_ms / self.cfg.bar_ms
        sd = math.sqrt(max(tf.vol_var, 0.0)) * math.sqrt(scale)
        return (tf.w_short.v - tf.vol_mean * scale) / max(sd, 1e-6)

    @property
    def ofi_medium(self) -> float:
        liq = self.liquidity_sol
        return self.delta_sol_medium / max(liq, 1.0)

    @property
    def aggressive_buy_ratio(self) -> float:
        w = self.tf.w_med
        return w.agg_buy / w.buy if w.buy > 1e-12 else 0.0

    @property
    def n_trades_short(self) -> int:
        return self.tf.w_short.n

    @property
    def n_trades_medium(self) -> int:
        return self.tf.w_med.n

    @property
    def unique_buyers_short(self) -> int:
        return len(self.tf.w_short.buyers)

    @property
    def unique_buyers_medium(self) -> int:
        return len(self.tf.w_med.buyers)

    # ---- wallets
    @property
    def whale_share(self) -> float:
        w = self.tf.w_med
        return w.whale / w.v if w.v > 1e-12 else 0.0

    @property
    def smart_share(self) -> float:
        w = self.tf.w_med
        return w.smart_buy / w.buy if w.buy > 1e-12 else 0.0

    @property
    def smart_buy_sol_medium(self) -> float:
        return self.tf.w_med.smart_buy

    @property
    def fresh_wallet_pct(self) -> float:
        w = self.tf.w_med
        return 100.0 * len(w.fresh) / len(w.buyers) if w.buyers else 0.0

    @property
    def bot_wallet_pct(self) -> float:
        w = self.tf.w_med
        return 100.0 * len(w.bots) / len(w.buyers) if w.buyers else 0.0

    @property
    def hhi(self) -> float:
        circ = self.tf.circ
        return self.tf.sumsq / (circ * circ) if circ > 0 else 0.0

    @property
    def top10_pct(self) -> float:
        tf = self.tf
        if tf._top_ver != tf._ver:
            tf._top_cache = sum(heapq.nlargest(self.cfg.top_holders_k, tf.holders.values())) if tf.holders else 0
            tf._top_ver = tf._ver
        return 100.0 * tf._top_cache / self.st.curve.supply

    @property
    def creator_holding_pct(self) -> float:
        c = self.tf.creator
        return 100.0 * self.tf.holders.get(c, 0) / self.st.curve.supply if c else 0.0

    @property
    def creator_sold_pct(self) -> float:
        return self.st.creator_sold_pct

    @property
    def n_holders(self) -> int:
        return len(self.tf.holders)

    @property
    def bundled_buyers(self) -> int:
        return len(self.st.bundled_buyers)

    @property
    def bundled_unknown(self) -> int:
        """Creation-slot buyers not (yet) recognised as recurring snipers or bots — the likely insider bundle.
        Point-in-time: uses the wallet profiles as they stand now."""
        wi = self.eng.wi
        if wi is None:
            return len(self.st.bundled_buyers)
        return sum(1 for u in self.st.bundled_buyers if not (wi.is_sniper(u) or wi.is_bot(u)))

    @property
    def dev_buy_sol(self) -> float:
        return self.st.dev_buy_lamports / LAMPORTS_PER_SOL

    # ---- bonding curve
    @property
    def progress_pct(self) -> float:
        return 100.0 if self.st.complete else self.eng.curve.progress_pct(self.st.curve)

    @property
    def liquidity_sol(self) -> float:
        st = self.st
        if st.migrated and st.pool is not None:
            return st.pool.quote / LAMPORTS_PER_SOL
        return st.curve.r_sol / LAMPORTS_PER_SOL

    @property
    def liquidity_slope(self) -> float:
        span = self.cfg.medium_window_ms
        past = self.tf.r_sol_hist.asof(self.now - span)
        if past is None:
            past = 0.0
        return (self.liquidity_sol - past) / (span / 1000.0)

    @property
    def liquidity_drop_pct(self) -> float:
        peak = self.st.peak_r_sol / LAMPORTS_PER_SOL
        return 100.0 * (1.0 - self.liquidity_sol / peak) if peak > 0 else 0.0

    @property
    def buy_pressure(self) -> float:
        s = self.cfg.short_window_ms / 1000.0
        return (self.tf.w_short.buy / s) / max(self.liquidity_sol, 1.0)

    @property
    def remaining_tokens_pct(self) -> float:
        return 100.0 * self.st.curve.r_tok / self.eng.curve.initial_r_tok

    @property
    def sol_to_complete(self) -> float:
        if self.st.complete:
            return 0.0
        return self.eng.curve.sol_to_complete(self.st.curve) / LAMPORTS_PER_SOL

    @property
    def curve_accel(self) -> float:
        s = self.cfg.short_window_ms
        h = self.tf.prog_hist
        p0 = self.progress_pct
        p1 = h.asof(self.now - s)
        p2 = h.asof(self.now - 2 * s)
        if p1 is None or p2 is None:
            return 0.0
        sec = s / 1000.0
        return ((p0 - p1) - (p1 - p2)) / (sec * sec)

    @property
    def mcap_sol(self) -> float:
        st = self.st
        if st.migrated and st.pool is not None:
            return st.pool.market_cap_lamports / LAMPORTS_PER_SOL
        return st.curve.market_cap_sol

    # ---- time
    @property
    def age_s(self) -> float:
        c = self.st.created_ms
        return (self.now - c) / 1000.0 if c is not None else 0.0

    @property
    def hour_sin(self) -> float:
        return math.sin(TWO_PI * ((self.now // 1000) % 86_400) / 86_400)

    @property
    def hour_cos(self) -> float:
        return math.cos(TWO_PI * ((self.now // 1000) % 86_400) / 86_400)

    @property
    def slot_velocity(self) -> float:
        return self.eng.slot_vel

    @property
    def tps(self) -> float:
        dt = max(0, self.now - self.eng._tps_t)
        return self.eng.tps * math.exp(-dt / self.eng.tau_tps)

    # ---- export
    def as_dict(self, names: tuple[str, ...] = FEATURE_NAMES) -> dict[str, float]:
        out: dict[str, float] = {}
        for n in names:
            v = getattr(self, n)
            out[n] = float(v) if v is not None else float("nan")
        return out
