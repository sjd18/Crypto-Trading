"""Wallet intelligence engine (point-in-time).

Purpose
    Continuously profile every wallet from its on-chain trades and rank wallets statistically,
    without ever using information from the future: statistics are updated only by events that
    already happened, and "profitability" uses *realised* round trips only.

Statistics per wallet
    trade counts & volumes, tokens traded, average cost positions per mint (for realised PnL),
    closed round trips, wins, sum / sum-of-squares of round-trip returns, holding times, entry
    delay after launch, creation-slot buys, tokens created, rugs created/involved, co-buy links.

Smart score (posterior probability of positive edge)
    Per-trade returns are shrunk toward 0 with prior strength ``k``:
    ``m = sum(r) / (n + k)``, ``s = max(sd(r), score_min_sd)``, ``z = m / (s / sqrt(n + k))`` and
    ``score = Phi(z)``. Wallets need ``min_closed_for_rank`` closed trades to be ranked; the
    "smart" label needs ``score >= smart_label_min_score``.

Labels (recomputed on demand)
    sniper, whale, market_maker, bot, creator, rug_wallet, insider (union-find cluster with a
    creator after >= ``insider_min_cobuys`` creation-slot co-buys), smart.

Example
    wi = WalletIntel(settings.wallet_intel, settings.features)
    wi.on_trade(ev, token_state)            # from the event loop
    wi.score(addr), wi.labels(addr), wi.top(20)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, Event

_SQRT2 = math.sqrt(2.0)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


@dataclass(slots=True)
class OpenPosition:
    """A wallet's current round trip in one mint (average-cost accounting)."""

    tokens: int = 0
    cost_left: int = 0       # remaining cost basis, lamports (incl. fees)
    first_ms: int = 0
    bought: int = 0          # tokens bought during this round trip
    cost_total: int = 0      # total cost of this round trip
    proceeds: int = 0        # net proceeds realised so far


@dataclass(slots=True)
class WalletStats:
    address: str
    first_seen_ms: int
    last_seen_ms: int
    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    buy_lamports: int = 0
    sell_lamports: int = 0
    tokens: set[str] = field(default_factory=set)
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    realized_lamports: int = 0
    closed: int = 0
    wins: int = 0
    sum_ret: float = 0.0
    sum_ret2: float = 0.0
    hold_ms_sum: int = 0
    early_entries: int = 0
    creation_slot_buys: int = 0
    tokens_created: int = 0
    rugs_created: int = 0
    rug_involvements: int = 0

    @property
    def mean_ret(self) -> float:
        return self.sum_ret / self.closed if self.closed else 0.0

    @property
    def sd_ret(self) -> float:
        if self.closed < 2:
            return 0.0
        var = (self.sum_ret2 - self.closed * self.mean_ret**2) / (self.closed - 1)
        return math.sqrt(max(var, 0.0))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        self.parent[x] = p
        return p

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class WalletIntel:
    """Incremental, point-in-time wallet database with statistical ranking."""

    def __init__(self, cfg: Any, feat_cfg: Any | None = None) -> None:
        self.cfg = cfg
        self.fresh_window_ms = feat_cfg.fresh_wallet_window_ms if feat_cfg is not None else 86_400_000
        self.wallets: dict[str, WalletStats] = {}
        self._cobuys: dict[tuple[str, str], int] = defaultdict(int)
        self._uf = _UnionFind()
        self._creators: set[str] = set()
        self._insider_of: dict[str, set[str]] = defaultdict(set)
        self._score_cache: dict[str, tuple[int, float]] = {}

    def __len__(self) -> int:
        return len(self.wallets)

    def get(self, addr: str | None) -> WalletStats | None:
        return self.wallets.get(addr) if addr else None

    def _stats(self, addr: str, ts: int) -> WalletStats:
        w = self.wallets.get(addr)
        if w is None:
            if len(self.wallets) >= self.cfg.max_wallets_tracked:
                # evict the least recently seen wallet (bounded memory)
                oldest = min(self.wallets.values(), key=lambda x: x.last_seen_ms)
                del self.wallets[oldest.address]
            w = WalletStats(addr, ts, ts)
            self.wallets[addr] = w
        return w

    # ------------------------------------------------------------------ updates
    def on_create(self, ev: Event) -> None:
        creator = ev.creator or ev.user
        if creator:
            w = self._stats(creator, ev.ts_ms)
            w.tokens_created += 1
            self._creators.add(creator)

    def on_trade(self, ev: Event, created_ms: int | None = None, created_slot: int | None = None,
                 creator: str | None = None) -> None:
        """Update with one curve or AMM trade (call *after* features used the pre-trade state)."""
        user = ev.user
        if not user or ev.mint is None:
            return
        ts = ev.ts_ms
        w = self._stats(user, ts)
        w.last_seen_ms = ts
        w.n_trades += 1
        w.tokens.add(ev.mint)
        sol = int(ev.sol_amount or 0)
        tok = int(ev.token_amount or 0)
        fees = int(ev.fee or 0) + int(ev.creator_fee or 0) + int(ev.lp_fee or 0)
        pos = w.positions.get(ev.mint)
        if ev.is_buy:
            w.n_buys += 1
            w.buy_lamports += sol
            if pos is None:
                pos = OpenPosition(first_ms=ts)
                w.positions[ev.mint] = pos
                if created_ms is not None and ts - created_ms <= self.cfg.sniper_entry_s * 1000:
                    w.early_entries += 1
            pos.tokens += tok
            pos.cost_left += sol + fees
            pos.bought += tok
            pos.cost_total += sol + fees
            if created_slot is not None and ev.slot == created_slot and creator and user != creator:
                w.creation_slot_buys += 1
                key = (creator, user)
                self._cobuys[key] += 1
                if self._cobuys[key] >= self.cfg.insider_min_cobuys:
                    self._uf.union(creator, user)
                    self._insider_of[user].add(creator)
        else:
            w.n_sells += 1
            w.sell_lamports += sol
            if pos is None or pos.tokens <= 0 or tok <= 0:
                return  # tokens acquired off-curve (e.g. transfer): no cost basis
            sold = min(tok, pos.tokens)
            cost_part = pos.cost_left * sold // pos.tokens
            proceeds = (sol - fees) * sold // tok
            w.realized_lamports += proceeds - cost_part
            pos.tokens -= sold
            pos.cost_left -= cost_part
            pos.proceeds += proceeds
            if pos.tokens <= max(1, pos.bought // 100):  # >= 99 % sold -> round trip closed
                ret = (pos.proceeds / pos.cost_total - 1.0) if pos.cost_total > 0 else 0.0
                ret = max(-1.0, min(ret, 20.0))  # winsorise extreme memecoin multiples
                w.closed += 1
                w.wins += int(ret > 0)
                w.sum_ret += ret
                w.sum_ret2 += ret * ret
                w.hold_ms_sum += ts - pos.first_ms
                del w.positions[ev.mint]
                self._score_cache.pop(user, None)

    def on_token_outcome(self, creator: str | None, rug: bool, insiders: set[str] | None = None) -> None:
        if not rug:
            return
        if creator and creator in self.wallets:
            self.wallets[creator].rugs_created += 1
        for u in insiders or ():
            w = self.wallets.get(u)
            if w is not None:
                w.rug_involvements += 1

    # ------------------------------------------------------------------ queries
    def score(self, addr: str | None) -> float | None:
        """Posterior probability that the wallet's expected round-trip return is positive."""
        w = self.wallets.get(addr) if addr else None
        if w is None or w.closed < self.cfg.min_closed_for_rank:
            return None
        cached = self._score_cache.get(w.address)
        if cached is not None and cached[0] == w.closed:
            return cached[1]
        k = self.cfg.prior_strength
        m = w.sum_ret / (w.closed + k)
        s = max(w.sd_ret, self.cfg.score_min_sd)
        z = m / (s / math.sqrt(w.closed + k))
        val = _phi(z)
        self._score_cache[w.address] = (w.closed, val)
        return val

    def is_fresh(self, addr: str | None, ts_ms: int) -> bool:
        w = self.wallets.get(addr) if addr else None
        return w is None or ts_ms - w.first_seen_ms <= self.fresh_window_ms

    def is_sniper(self, addr: str | None) -> bool:
        """Recurring launch-slot buyer: >= ``sniper_min_tokens`` early entries making up >= half its tokens."""
        w = self.wallets.get(addr) if addr else None
        return (w is not None and w.early_entries >= self.cfg.sniper_min_tokens
                and w.early_entries >= self.cfg.sniper_min_early_frac * len(w.tokens))

    def is_bot(self, addr: str | None) -> bool:
        w = self.wallets.get(addr) if addr else None
        if w is None or w.n_trades < self.cfg.bot_min_trades:
            return False
        hours = max((w.last_seen_ms - w.first_seen_ms) / 3_600_000, 1 / 60)
        return w.n_trades / hours >= self.cfg.bot_min_trades_per_hour

    def is_whale(self, addr: str | None) -> bool:
        w = self.wallets.get(addr) if addr else None
        return bool(w and w.n_buys >= self.cfg.whale_min_buys
                    and w.buy_lamports / w.n_buys >= self.cfg.whale_avg_trade_sol * LAMPORTS_PER_SOL)

    def cluster(self, addr: str) -> str:
        return self._uf.find(addr)

    def labels(self, addr: str | None) -> list[str]:
        w = self.wallets.get(addr) if addr else None
        if w is None:
            return []
        out: list[str] = []
        if self.is_sniper(addr):
            out.append("sniper")
        if self.is_whale(addr):
            out.append("whale")
        if w.closed >= self.cfg.mm_min_closed and w.n_buys and self.cfg.mm_roundtrip_ratio <= w.n_sells / w.n_buys <= 1 / self.cfg.mm_roundtrip_ratio \
                and w.hold_ms_sum / w.closed <= self.cfg.mm_max_hold_s * 1000:
            out.append("market_maker")
        if self.is_bot(addr):
            out.append("bot")
        if w.tokens_created:
            out.append("creator")
        if w.rugs_created >= self.cfg.rug_wallet_min_rugs or w.rug_involvements >= self.cfg.rug_wallet_min_rugs:
            out.append("rug_wallet")
        if self._insider_of.get(w.address):
            out.append("insider")
        sc = self.score(addr)
        if sc is not None and sc >= self.cfg.smart_label_min_score:
            out.append("smart")
        return out

    def top(self, n: int = 50, min_closed: int | None = None) -> list[tuple[str, float, WalletStats]]:
        mc = self.cfg.min_closed_for_rank if min_closed is None else min_closed
        ranked = []
        for addr, w in self.wallets.items():
            if w.closed >= mc:
                sc = self.score(addr)
                if sc is not None:
                    ranked.append((addr, sc, w))
        ranked.sort(key=lambda x: (-x[1], -x[2].realized_lamports))
        return ranked[:n]

    def to_frame(self, min_trades: int = 1, now_ms: int = 0) -> pl.DataFrame:
        rows = []
        for addr, w in self.wallets.items():
            if w.n_trades < min_trades:
                continue
            rows.append({
                "address": addr, "first_seen_ms": w.first_seen_ms, "last_seen_ms": w.last_seen_ms, "n_trades": w.n_trades,
                "n_buys": w.n_buys, "n_sells": w.n_sells, "buy_sol": w.buy_lamports / LAMPORTS_PER_SOL,
                "sell_sol": w.sell_lamports / LAMPORTS_PER_SOL, "tokens_traded": len(w.tokens), "closed": w.closed,
                "wins": w.wins, "realized_pnl_sol": w.realized_lamports / LAMPORTS_PER_SOL, "mean_ret": w.mean_ret,
                "smart_score": self.score(addr), "labels": ",".join(self.labels(addr)),
                "cluster": self._uf.find(addr), "updated_ms": now_ms,  # cluster id = root wallet (deterministic)
            })
        if not rows:
            return pl.DataFrame(schema={"address": pl.Utf8})
        return pl.DataFrame(rows, infer_schema_length=None)
