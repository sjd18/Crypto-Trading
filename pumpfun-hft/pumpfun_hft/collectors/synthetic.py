"""Synthetic Pump.fun market generator.

Purpose
    Produce a realistic, fully reproducible event stream (same schema as real collected data)
    for tests, demos, CI and pipeline validation when no historical data is available. Every
    trade is priced with the exact bonding-curve / AMM math and fee schedule, so reserves,
    fees, completion and migration are internally consistent.

Market model (per token, independently simulated then merged on the time axis)
    * Creators have persistent types (``rugger`` / ``neutral`` / ``legit``) that drive the
      token's latent quality ``q`` and the probability and timing of a creator dump (rug).
      Ruggers bundle insider wallets into the creation slot.
    * Order flow is a self-exciting (Hawkes-type) point process simulated exactly with Ogata
      thinning: baseline intensity decays with token age (faster for low ``q``), each buy excites
      further activity. Flow participants are retail (lognormal sizes) and bots (quick round trips).
    * Informed ``smart`` wallets observe ``q`` with noise, enter early, exit at targets or right
      after a rug; ``snipers`` buy within the first slots; ``whales`` make large buys.
    * Curves that sell out emit CompleteEvent, then a PumpSwap migration and an AMM phase.

    Planted structure (useful for validating research tooling, *not* evidence about real
    markets): smart wallets are profitable by construction; serial ruggers are identifiable from
    their history; flow has short-horizon momentum from self-excitation.

Outputs
    ``events`` (canonical Event schema), ``metadata`` (token URI JSON fields) and ``truth``
    (latent per-token variables — for validation only; never feed it to strategies).

Example
    >>> market = SyntheticMarket(load_settings(overrides={"synthetic.duration_hours": 1}))  # doctest: +SKIP
    >>> data = market.generate()                                                           # doctest: +SKIP
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.core.amm import ConstantProductAmm, PoolState
from pumpfun_hft.core.curve import BondingCurve, FeeSchedule, FeeTier
from pumpfun_hft.core.types import EVENT_COLUMNS, EVENT_SCHEMA, LAMPORTS_PER_SOL, EventKind
from pumpfun_hft.utils.base58 import b58encode, pubkey_to_str
from pumpfun_hft.utils.timeutil import parse_iso_ms

try:  # fast signature encoding
    from solders.signature import Signature as _Sig

    def _sig_str(b: bytes) -> str:
        return str(_Sig.from_bytes(b))
except Exception:  # noqa: BLE001
    def _sig_str(b: bytes) -> str:
        return b58encode(b)

WORDS = ["dog", "cat", "pepe", "frog", "moon", "ai", "gpt", "agent", "trump", "elon", "chad", "wojak", "bonk", "wif",
         "inu", "doge", "based", "meme", "rocket", "sol", "whale", "ape", "banana", "pixel", "cyber", "grok", "hamster",
         "penguin", "shark", "tiger", "degen", "gem", "turbo", "neko", "kek", "zen", "vibe", "lambo", "pump", "sigma"]
ADJ = ["baby", "super", "mega", "tiny", "golden", "dark", "cyber", "based", "happy", "angry", "royal", "wild", "lazy",
       "crazy", "little", "big", "magic", "cosmic", "retro", "lucky"]
SLOT0 = 350_000_000


@dataclass(slots=True)
class SyntheticData:
    events: pl.DataFrame
    metadata: pl.DataFrame
    truth: pl.DataFrame


class _Cols:
    """Column-wise event buffer (faster than a list of dicts)."""

    __slots__ = ("cols", "n", "order")

    def __init__(self) -> None:
        self.cols: dict[str, list[Any]] = {c: [] for c in EVENT_COLUMNS}
        self.cols["_tok"] = []
        self.cols["_n"] = []
        self.n = 0
        self.order = 0

    def emit(self, tok: int, **kw: Any) -> None:
        for c in EVENT_COLUMNS:
            self.cols[c].append(kw.get(c))
        self.cols["_tok"].append(tok)
        self.cols["_n"].append(self.n)
        self.n += 1


class SyntheticMarket:
    """Generate synthetic Pump.fun events from ``settings.synthetic`` (deterministic by seed)."""

    def __init__(self, settings: Any) -> None:
        self.c = settings.synthetic
        self.p = settings.protocol
        self.slot_ms = settings.simulation.slot_ms
        self.curve = BondingCurve.from_config(self.p.curve, self.p.curve_fee_tiers)
        flat = self.p.amm_flat_fees
        self.amm = ConstantProductAmm(FeeSchedule.from_config(self.p.amm_fee_tiers),
                                      FeeTier(0, flat.protocol_bps, flat.creator_bps, flat.lp_bps))
        self.rng = np.random.default_rng(self.c.seed)
        self.start_ms = parse_iso_ms(self.c.start)
        self.end_ms = self.start_ms + int(self.c.duration_hours * 3_600_000)
        self.pool_tokens = self.p.curve.token_total_supply - self.p.curve.initial_real_token_reserves
        self._setup_population()
        self._setup_sol_path()
        self.buf = _Cols()
        self.truth: list[dict[str, Any]] = []
        self.meta: list[dict[str, Any]] = []
        self.recent_names: list[tuple[int, str]] = []

    # ------------------------------------------------------------------ setup
    def _addr(self) -> str:
        return pubkey_to_str(self.rng.bytes(32))

    def _sig(self) -> str:
        return _sig_str(self.rng.bytes(64))

    def _setup_population(self) -> None:
        c, rng = self.c, self.rng
        kinds = list(c.creator_mix)
        probs = np.array([c.creator_mix[k] for k in kinds], dtype=float)
        probs /= probs.sum()
        self.creators = [(self._addr(), kinds[i]) for i in rng.choice(len(kinds), size=c.n_creators, p=probs)]
        weight = {"rugger": 3.0, "neutral": 1.0, "legit": 0.7}
        w = np.array([weight.get(k, 1.0) for _, k in self.creators])
        self.creator_p = w / w.sum()
        wk = list(c.wallet_mix)
        wp = np.array([c.wallet_mix[k] for k in wk], dtype=float)
        wp /= wp.sum()
        counts = np.maximum(1, np.round(wp * c.n_wallets).astype(int))
        self.wallets: dict[str, list[str]] = {k: [self._addr() for _ in range(n)] for k, n in zip(wk, counts, strict=True)}
        self.insiders_of: dict[str, list[str]] = {}
        for addr, kind in self.creators:
            if kind == "rugger" and self.wallets.get("insider"):
                k = int(rng.integers(1, 4))
                self.insiders_of[addr] = list(rng.choice(self.wallets["insider"], size=k, replace=False))

    def _setup_sol_path(self) -> None:
        minutes = int((self.end_ms - self.start_ms) / 60_000) + 2
        sigma = self.c.sol_usd_vol_daily / math.sqrt(1440)
        shocks = self.rng.normal(-0.5 * sigma * sigma, sigma, size=minutes)
        self.sol_path = self.c.sol_usd_start * np.exp(np.cumsum(shocks))

    def _sol_usd(self, ts_ms: int) -> float:
        i = min(max(0, (ts_ms - self.start_ms) // 60_000), len(self.sol_path) - 1)
        return float(self.sol_path[i])

    def _lognormal_sol(self, spec: list[float]) -> int:
        median, sigma = spec
        return max(10_000_000, int(median * math.exp(self.rng.normal(0.0, sigma)) * LAMPORTS_PER_SOL))

    def _name(self, t_ms: int, kind: str) -> tuple[str, str]:
        rng = self.rng
        self.recent_names = [(t, n) for t, n in self.recent_names if t_ms - t < 3_600_000]
        if kind == "rugger" and self.recent_names and rng.random() < 0.4:
            name = self.recent_names[int(rng.integers(len(self.recent_names)))][1]  # copycat launch
        else:
            name = f"{ADJ[int(rng.integers(len(ADJ)))]} {WORDS[int(rng.integers(len(WORDS)))]}"
            if rng.random() < 0.3:
                name = WORDS[int(rng.integers(len(WORDS)))] + WORDS[int(rng.integers(len(WORDS)))]
        self.recent_names.append((t_ms, name))
        symbol = "".join(p[0] for p in name.split()).upper() + name.replace(" ", "")[:4].upper()
        return name.title(), symbol[:8]

    # ------------------------------------------------------------------ generation
    def generate(self) -> SyntheticData:
        rng = self.rng
        n_launch = rng.poisson(self.c.launches_per_hour * self.c.duration_hours)
        launch_times = np.sort(rng.uniform(self.start_ms, self.end_ms - 60_000, size=n_launch))
        for tok, t in enumerate(launch_times):
            t0 = int(t) - (int(t) - self.start_ms) % self.slot_ms + 1  # slot-aligned (+1 ms)
            ci = int(rng.choice(len(self.creators), p=self.creator_p))
            self._simulate_token(tok, t0, *self.creators[ci])
        return self._finalise()

    def _finalise(self) -> SyntheticData:
        cols = self.buf.cols
        df = pl.DataFrame({**{c: cols[c] for c in EVENT_COLUMNS}, "_tok": cols["_tok"], "_n": cols["_n"]},
                          schema={**EVENT_SCHEMA, "_tok": pl.Int64, "_n": pl.Int64}, strict=False)
        df = df.filter(pl.col("ts_ms") < self.end_ms + int(self.c.max_token_life_s * 1000))
        df = (
            df.with_columns(((pl.col("ts_ms") - self.start_ms) // self.slot_ms + SLOT0).alias("slot"))
            .sort(["ts_ms", "_tok", "_n"])
            .with_columns(
                pl.int_range(pl.len()).over("slot").cast(pl.Int32).alias("seq"),
                (pl.col("ts_ms") // 1000).alias("block_time"),
            )
            .drop(["_tok", "_n"])
        )
        return SyntheticData(df.select(list(EVENT_COLUMNS)), pl.DataFrame(self.meta, infer_schema_length=None),
                             pl.DataFrame(self.truth, infer_schema_length=None))

    # ------------------------------------------------------------------ per-token simulation
    def _simulate_token(self, tok: int, t0: int, creator: str, ckind: str) -> None:  # noqa: C901, PLR0915
        c, rng, curve = self.c, self.rng, self.curve
        mint, bonding_curve = self._addr(), self._addr()
        q = float({"legit": rng.beta(4, 2), "neutral": rng.beta(2, 3), "rugger": rng.beta(1.2, 4)}.get(ckind, rng.beta(2, 3)))
        viral = ckind != "rugger" and rng.random() < c.viral_prob  # heavy tail: the rare token that goes viral
        if viral:
            q = max(q, 0.85)
        name, symbol = self._name(t0, ckind)
        socials = rng.random() < c.social_prob.get(ckind, 0.5)
        self.meta.append({
            "mint": mint, "name": name, "symbol": symbol, "uri": f"https://ipfs.io/ipfs/Qm{self._addr()[:40]}",
            "description": f"{name} to the moon", "image": f"https://ipfs.io/ipfs/Qm{self._addr()[:40]}",
            "twitter": f"https://x.com/{symbol.lower()}" if socials else "",
            "telegram": f"https://t.me/{symbol.lower()}" if socials and rng.random() < 0.8 else "",
            "website": f"https://{symbol.lower()}.fun" if socials and rng.random() < 0.5 else "",
        })
        state = curve.new_state(has_creator=True)
        launch_price = state.price
        phase = "curve"
        pool: PoolState | None = None
        pool_addr = self._addr()
        holders: dict[str, int] = {}
        cost: dict[str, float] = {}
        role: dict[str, str] = {creator: "creator"}
        targets: dict[str, float] = {}
        # insertion-ordered dicts (not sets): iteration order must not depend on string hashing, or the
        # generator would not be reproducible across processes
        flow_holders: dict[str, None] = {}
        flippers: dict[str, None] = {}  # snipers / copy-trade bots holding a bag: they also sell into the flow
        heap: list[tuple[int, int, str, str, float]] = []
        hseq = 0
        excite, excite_t = 0.0, t0
        rugged = False
        migrated_ms: int | None = None
        complete_ms: int | None = None
        amm_end = 0
        peak = launch_price
        life_end = t0 + int(c.max_token_life_s * 1000)
        life_scale = c.life_scale_s * (0.3 + 1.4 * q)
        buf = self.buf
        retail, bots = self.wallets.get("retail", []), self.wallets.get("bot", [])

        def push(t: int, action: str, wallet: str = "", x: float = 0.0) -> None:
            nonlocal hseq
            hseq += 1
            heapq.heappush(heap, (t, hseq, action, wallet, x))

        def emit_trade(t: int, user: str, is_buy: bool, sol: int, tokens: int, pf: int, cf: int, fbps: Any) -> None:
            buf.emit(tok, kind=EventKind.TRADE.value, ts_ms=t, signature=self._sig(), mint=mint, user=user, is_buy=is_buy,
                     sol_amount=sol, token_amount=tokens, v_sol=state.v_sol, v_tok=state.v_tok, r_sol=state.r_sol,
                     r_tok=state.r_tok, fee_bps=fbps.protocol, fee=pf, creator_fee_bps=fbps.creator, creator_fee=cf,
                     creator=creator, ix_name="buy_exact_sol_in" if is_buy else "sell", sol_usd=self._sol_usd(t), ev_idx=0)

        def emit_amm(t: int, user: str, is_buy: bool, sol: int, tokens: int, f: Any, fb: Any) -> None:
            assert pool is not None
            buf.emit(tok, kind=(EventKind.AMM_BUY if is_buy else EventKind.AMM_SELL).value, ts_ms=t, signature=self._sig(),
                     mint=mint, user=user, is_buy=is_buy, pool=pool_addr, sol_amount=sol, token_amount=tokens,
                     v_sol=pool.quote_eff, v_tok=pool.base, r_sol=pool.quote, r_tok=pool.base, fee_bps=fb.protocol,
                     fee=f[1], creator_fee_bps=fb.creator, creator_fee=f[2], lp_fee_bps=fb.lp, lp_fee=f[0], creator=creator,
                     ix_name="buy" if is_buy else "sell", sol_usd=self._sol_usd(t), ev_idx=0)

        def price() -> float:
            return pool.price if (phase == "amm" and pool is not None) else state.price

        def buy(t: int, user: str, budget: int) -> bool:
            nonlocal state, pool, excite, excite_t, complete_ms, peak
            if phase == "curve":
                if state.complete:
                    return False
                fb = curve.fees_for(state)
                qt = curve.buy_with_budget(state, budget, fb)
                if qt.tokens <= 0:
                    return False
                state = curve.apply_buy(state, qt)
                emit_trade(t, user, True, qt.sol_curve, qt.tokens, qt.protocol_fee, qt.creator_fee, fb)
                tokens, spent = qt.tokens, qt.total
                if state.complete and complete_ms is None:
                    complete_ms = t
                    buf.emit(tok, kind=EventKind.COMPLETE.value, ts_ms=t, signature=self._sig(), mint=mint, user=user,
                             bonding_curve=bonding_curve, sol_usd=self._sol_usd(t), ev_idx=1)
                    push(t + int(rng.integers(1, 4)) * self.slot_ms, "migrate")
            elif phase == "amm" and pool is not None:
                fb = self.amm.fees_for(pool)
                qa = self.amm.buy_base_for_quote(pool, budget, fb)
                if qa.base_out <= 0:
                    return False
                pool = self.amm.apply_buy(pool, qa)
                emit_amm(t, user, True, qa.quote_in, qa.base_out, (qa.lp_fee, qa.protocol_fee, qa.creator_fee), fb)
                tokens, spent = qa.base_out, qa.total
            else:
                return False
            prev = holders.get(user, 0)
            holders[user] = prev + tokens
            if role.get(user) in ("sniper", "frontrunner"):
                flippers[user] = None
            # competition: fast bots react within ~1 slot and flip quickly. Known smart / whale wallets are
            # copy-traded by several bots (Poisson ``copytrade_mean``); bursts attract a bot with probability
            # ``frontrun_intensity`` scaled by the current excitation. Slower followers pay their impact.
            r = role.get(user)
            if r not in ("frontrunner", "creator", "insider") and bots:
                if r in ("smart", "whale"):
                    n_fr = int(rng.poisson(c.copytrade_mean))
                else:
                    n_fr = 1 if rng.random() < c.frontrun_intensity * min(1.0, excite) else 0
                for _ in range(n_fr):
                    fr = bots[int(rng.integers(len(bots)))]
                    role[fr] = "frontrunner"
                    push(t + int(rng.uniform(40, 350)), "buy", fr, self._lognormal_sol(c.frontrun_trade_sol))
                    push(t + int(rng.uniform(3_000, 15_000)), "sell", fr, 1.0)
            cost[user] = cost.get(user, 0.0) + spent
            # self-excitation (decayed to t, then bumped)
            excite = excite * math.exp(-(t - excite_t) / (c.excitation_decay_s * 1000)) + min(spent / LAMPORTS_PER_SOL, 2.0) * 0.25
            excite_t = t
            peak = max(peak, price())
            return True

        def sell(t: int, user: str, frac: float) -> bool:
            nonlocal state, pool, peak
            held = holders.get(user, 0)
            amount = int(held * min(1.0, frac))
            if amount <= 0:
                return False
            if phase == "curve":
                if state.complete:
                    return False
                fb = curve.fees_for(state)
                qs = curve.sell_proceeds_for_tokens(state, amount, fb)
                if qs.sol_curve <= 0 or qs.sol_curve > state.r_sol:
                    return False
                state = curve.apply_sell(state, qs)
                emit_trade(t, user, False, qs.sol_curve, amount, qs.protocol_fee, qs.creator_fee, fb)
            elif phase == "amm" and pool is not None:
                fb = self.amm.fees_for(pool)
                qa = self.amm.sell_quote_for_base(pool, amount, fb)
                if qa.quote_out <= 0:
                    return False
                pool = self.amm.apply_sell(pool, qa)
                emit_amm(t, user, False, qa.quote_out, amount, (qa.lp_fee, qa.protocol_fee, qa.creator_fee), fb)
            else:
                return False
            holders[user] = held - amount
            cost[user] = cost.get(user, 0.0) * (1 - amount / held)
            if holders[user] <= 0:
                holders.pop(user, None)
                cost.pop(user, None)
                flow_holders.pop(user, None)
                flippers.pop(user, None)
                targets.pop(user, None)
            return True

        # ---- creation slot: create + dev buy + insider bundle
        buf.emit(tok, kind=EventKind.CREATE.value, ts_ms=t0, signature=self._sig(), mint=mint, user=creator, creator=creator,
                 name=name, symbol=symbol, uri=self.meta[-1]["uri"], bonding_curve=bonding_curve, v_sol=state.v_sol,
                 v_tok=state.v_tok, r_sol=0, r_tok=state.r_tok, token_amount=state.supply,
                 token_program="TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb", sol_usd=self._sol_usd(t0), ev_idx=0)
        lo, hi = c.dev_buy_sol
        dev = int(rng.uniform(lo, hi) * (1.6 if ckind == "rugger" else 1.0) * LAMPORTS_PER_SOL)
        buy(t0 + 1, creator, dev)
        insiders = self.insiders_of.get(creator, []) if ckind == "rugger" else []
        for k, ins in enumerate(insiders):
            role[ins] = "insider"
            buy(t0 + 2 + k, ins, self._lognormal_sol(c.sniper_trade_sol))
        will_rug = rng.random() < c.rug_prob.get(ckind, 0.1)
        if will_rug:
            push(t0 + int(rng.uniform(*c.rug_delay_s) * 1000), "rug")
        # ---- snipers
        attract = (1.2 if socials else 0.6) * (1.6 if ckind == "legit" else 1.0)  # reputable devs get sniped hardest
        for w in self.wallets.get("sniper", []):
            if rng.random() < c.sniper_participation * attract:
                role.setdefault(w, "sniper")
                delay = rng.uniform(5, 400) if rng.random() < c.sniper_slot0_frac else rng.uniform(400, 2500)
                push(t0 + int(delay), "buy", w, self._lognormal_sol(c.sniper_trade_sol))
                push(t0 + int(rng.uniform(5_000, 60_000)), "sell", w, 1.0)
        # ---- informed (smart) wallets
        for w in self.wallets.get("smart", []):
            q_obs = q + rng.normal(0, c.smart_signal_noise) - (0.35 if (ckind == "rugger" and rng.random() < 0.7) else 0.0)
            if q_obs > c.smart_entry_threshold:
                role.setdefault(w, "smart")
                push(t0 + int(rng.uniform(3_000, 45_000)), "smart_buy", w, float(self._lognormal_sol(c.smart_trade_sol)))
        # ---- whales
        for w in self.wallets.get("whale", []):
            if rng.random() < c.whale_participation * q:
                role.setdefault(w, "whale")
                push(t0 + int(rng.exponential(120_000)) + 5_000, "whale_buy", w, float(self._lognormal_sol(c.whale_trade_sol)))

        t = t0 + 3

        def intensity(tt: int) -> float:
            age_s = (tt - t0) / 1000.0
            # attention is heavy-tailed: most launches get a handful of trades, good ones get hundreds
            act = (c.activity_floor + c.activity_scale * q ** c.activity_exponent) * math.exp(-age_s / life_scale) \
                * (c.viral_boost if viral else 1.0)
            if rugged:
                act *= 0.08
            if phase == "amm":
                act *= 0.6
            elif state.complete:
                return 0.0
            exc = excite * math.exp(-(tt - excite_t) / (c.excitation_decay_s * 1000))
            return c.base_trade_rate_hz * act + c.excitation * exc

        while True:
            lam = intensity(t)
            t_flow = t + int(rng.exponential(1.0 / lam) * 1000) + 1 if lam > 1e-4 else 1 << 62
            t_sched = heap[0][0] if heap else 1 << 62
            t_next = min(t_flow, t_sched)
            end_t = amm_end if phase == "amm" else life_end
            if t_next >= end_t or t_next >= (1 << 62):
                break
            if t_sched <= t_flow:
                t, _, action, w, x = heapq.heappop(heap)
                if action == "buy":
                    buy(t, w, int(x))
                elif action == "sell":
                    sell(t, w, x)
                elif action == "smart_buy":
                    if not rugged and buy(t, w, int(x)):
                        targets[w] = price() * rng.uniform(1.8, 4.0)
                        push(t + int(life_scale * 1000 * rng.uniform(0.3, 1.0)), "sell", w, 1.0)
                elif action == "whale_buy":
                    if not rugged and buy(t, w, int(x)):
                        targets[w] = price() * rng.uniform(1.3, 2.5)
                        push(t + int(rng.uniform(60_000, 600_000)), "sell", w, 1.0)
                elif action == "rug":
                    if phase == "curve" and not state.complete and holders.get(creator, 0) > 0:
                        rugged = True
                        sell(t, creator, 1.0)
                        for ins in insiders:
                            push(t + int(rng.uniform(50, 900)), "sell", ins, 1.0)
                        for w, r in role.items():
                            if r in ("smart", "sniper") and holders.get(w, 0) > 0:
                                push(t + int(rng.uniform(400, 3_000)), "sell", w, 1.0)
                elif action == "migrate" and phase == "curve":
                    phase = "amm"
                    migrated_ms = t
                    quote = max(state.r_sol - self.p.migration_fee_lamports, 1)
                    pool = PoolState(self.pool_tokens, quote)
                    buf.emit(tok, kind=EventKind.MIGRATE.value, ts_ms=t, signature=self._sig(), mint=mint, user=self._addr(),
                             pool=pool_addr, token_amount=pool.base, sol_amount=pool.quote, v_tok=pool.base, v_sol=pool.quote,
                             r_tok=pool.base, r_sol=pool.quote, fee=self.p.migration_fee_lamports, bonding_curve=bonding_curve,
                             sol_usd=self._sol_usd(t), ev_idx=0)
                    amm_end = t + int(c.amm_life_s * 1000)
                continue
            t = t_flow
            if rng.random() * lam > intensity(t):
                continue  # thinning rejection
            # --- flow event
            px = price()
            for w, target in list(targets.items()):  # informed holders take profit at targets
                if px >= target and holders.get(w, 0) > 0:
                    push(t + int(rng.uniform(100, 1500)), "sell", w, 1.0)
                    targets.pop(w, None)
            gain = 0.0  # unrealised gain of the retail / bot flow holders drives their profit taking
            if flow_holders:
                tot_tok = sum(holders.get(w, 0) for w in flow_holders)
                tot_cost = sum(cost.get(w, 0.0) for w in flow_holders)
                if tot_tok > 0 and tot_cost > 0:
                    gain = (px / ((tot_cost / tot_tok) / 1000.0)) - 1.0  # avg entry in SOL per token
            sellers = {**flow_holders, **flippers}  # early bags (snipers, copy bots) also sell into retail demand
            age_frac = min(1.0, (t - t0) / 1000.0 / life_scale)  # interest fades: holders bleed out over the token's life
            # quality tilts the flow towards buying, but the market prices it in: the tilt decays with age
            q_tilt = 0.2 * (q - 0.5) * math.exp(-(t - t0) / 1000.0 / c.quality_drift_s)
            p_sell = (c.sell_pressure + 0.3 * math.tanh(2.0 * gain) + 0.3 * age_frac + (0.35 if rugged else 0.0)
                      - q_tilt - (0.15 if viral and age_frac < 0.5 else 0.0))
            p_sell = min(0.95, max(0.05, p_sell)) if sellers else 0.0
            if rng.random() < p_sell:
                cand = list(sellers)
                wts = np.array([holders.get(w, 0) for w in cand], dtype=float)
                if wts.sum() > 0:
                    w = cand[int(rng.choice(len(cand), p=wts / wts.sum()))]
                    sell(t, w, 1.0 if role.get(w) == "bot" else rng.uniform(0.3, 1.0))
            else:
                if bots and rng.random() < 0.1:
                    w = bots[int(rng.integers(len(bots)))]
                    role.setdefault(w, "bot")
                    if buy(t, w, max(10_000_000, self._lognormal_sol(c.retail_trade_sol) // 4)):
                        flow_holders[w] = None
                        push(t + int(rng.uniform(2_000, 20_000)), "sell", w, 1.0)
                else:
                    w = retail[int(rng.integers(len(retail)))]
                    role.setdefault(w, "retail")
                    if buy(t, w, self._lognormal_sol(c.retail_trade_sol)):
                        flow_holders[w] = None

        self.truth.append({
            "mint": mint, "creator": creator, "creator_kind": ckind, "quality": q, "rug_planned": will_rug,
            "rugged": rugged, "complete_ms": complete_ms, "migrated_ms": migrated_ms, "peak_multiple": peak / launch_price,
            "launch_ms": t0, "socials": socials, "viral": viral,
        })

    # ------------------------------------------------------------------ persistence helper
    def write(self, store: Any, metadata_path: Any | None = None, truth_path: Any | None = None) -> SyntheticData:
        """Generate and persist to a ParquetEventStore (+ metadata / truth Parquet files)."""
        from pumpfun_hft.collectors.storage import write_metadata

        data = self.generate()
        store.write(data.events)
        store.compact()
        if metadata_path is not None:
            write_metadata(data.metadata, metadata_path)
        if truth_path is not None:
            data.truth.write_parquet(truth_path)
        return data
