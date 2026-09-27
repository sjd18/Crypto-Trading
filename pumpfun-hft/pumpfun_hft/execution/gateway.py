"""Execution gateways: paper (simulated fills on live data) and live (real transactions).

Live pipeline per order (``LiveGateway.execute``)::

    QUOTE   local exact curve math on the live state (or Metis /pump-fun/quote in Mode B)
    SWAP    Metis /pump-fun/swap (signed as returned) or /pump-fun/swap-instructions (composed
            locally with compute-budget + optional in-tx Jito tip); Jupiter /quote + /swap after migration
    SIGN    WalletSigner (solders)
    SUBMIT  sendTransaction(skipPreflight, maxRetries=0) or Jito sendBundle
    CONFIRM batched getSignatureStatuses; the *same signed bytes* are re-broadcast every
            ``rebroadcast_ms`` until confirmed, failed or the blockhash expires
    RETRY   only after the blockhash is provably invalid (``isBlockhashValid`` false and a final
            status check) — never re-sign while the original can still land (no double buys)
    FILL    parsed from the confirmed transaction: SOL / token balance deltas, fee, decoded
            TradeEvent amounts

``quote -> submit`` latency is recorded as ``exec.quote_to_submit`` (budget ``live.quote_to_submit_budget_ms``).
Every state transition is persisted to ``MetaStore.live_orders`` for crash recovery.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from pumpfun_hft.api.metis import MetisClient
from pumpfun_hft.api.rpc import RpcError, SolanaRpcClient
from pumpfun_hft.backtester.execution_sim import ExecutionSimulator
from pumpfun_hft.core.clock import WallClock
from pumpfun_hft.core.types import Fill, Order, OrderStatus, Side, Venue
from pumpfun_hft.utils.latency import LatencyTracker
from pumpfun_hft.utils.logging import get_logger
from pumpfun_hft.utils.timeutil import now_ms

log = get_logger("trades")

# Program errors that mean "the price moved beyond the tolerance" (retried with wider slippage).
SLIPPAGE_ERRORS = {"TooMuchSolRequired", "TooLittleSolReceived", "BuySlippageBelowMinTokensOut",   # pump (bonding curve)
                   "ExceededSlippage", "BuySlippageBelowMinBaseAmountOut",                         # pump_amm (PumpSwap)
                   "SlippageToleranceExceeded"}                                                     # Jupiter v6 router
JUPITER_V6 = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"


class ExecutionGateway(Protocol):
    async def execute(self, order: Order) -> Fill: ...


class PaperGateway:
    """Paper trading: the backtester's execution model applied to *live* market state.

    The order waits the sampled latency in real time while live events keep updating the market,
    then executes against the state at landing — exactly the semantics of the backtest.
    """

    def __init__(self, sim: ExecutionSimulator, activity: Any, latency: LatencyTracker, clock: Any = None) -> None:
        self.sim = sim
        self.activity = activity  # callable -> market events/s (congestion proxy)
        self.latency = latency
        self.clock = clock or WallClock()  # a ReplayClock when recorded events are replayed through the live engine

    async def execute(self, order: Order) -> Fill:
        t = self.clock.now_ms()
        kind, when, payload = self.sim.submit(order, t, self.activity())
        await self.clock.sleep(when - self.clock.now_ms())
        if kind != "land":
            self._log(order, payload)
            return payload
        fill = self.sim.execute(order, self.clock.now_ms(), self.sim.market.last_slot)
        await self.clock.sleep(fill.confirm_ms - fill.land_ms)
        self.latency.record("paper.decision_to_land", fill.latency_ms)
        self._log(order, fill)
        return fill

    @staticmethod
    def _log(order: Order, fill: Fill) -> None:
        """Same trades.log records as the live gateway, marked ``paper``."""
        if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            log.info("paper fill", extra={"data": {"order": order.id, "mint": order.mint, "side": order.side.value,
                                                   "tokens": fill.token_amount, "sol_delta": fill.sol_delta, "price": fill.price,
                                                   "status": fill.status.value, "latency_ms": fill.latency_ms}})
        else:
            log.info("paper order not filled", extra={"data": {"order": order.id, "mint": order.mint,
                                                               "status": fill.status.value, "reason": fill.failure}})


class LiveGateway:
    """Real on-chain execution (see module docstring). Requires ``solders`` and a funded wallet."""

    def __init__(self, settings: Any, rpc: SolanaRpcClient, metis: MetisClient, signer: Any, blockhash: Any, priority: Any,
                 confirm: Any, decoder: Any, quotes: ExecutionSimulator, latency: LatencyTracker, meta: Any = None,
                 jito: Any = None) -> None:
        self.s = settings
        self.rpc = rpc
        self.metis = metis
        self.signer = signer
        self.blockhash = blockhash
        self.priority = priority
        self.confirm = confirm
        self.decoder = decoder
        self.quotes = quotes
        self.latency = latency
        self.meta = meta
        self.jito = jito
        self.latency.set_budget("exec.quote_to_submit", settings.live.quote_to_submit_budget_ms)
        # error codes overlap between programs (6004 is a slippage error in PumpSwap but not on the curve),
        # so errors are resolved per program: the program of the failing top-level instruction
        self._err_maps: dict[str, dict[int, str]] = {
            pid: {e["code"]: e["name"] for e in codec.idl.get("errors", [])} for pid, codec in (decoder.codecs.items() if decoder else [])}
        self._err_maps.setdefault(JUPITER_V6, {6001: "SlippageToleranceExceeded"})

    def _persist(self, order: Order, status: str, signature: str | None = None) -> None:
        if self.meta is not None:
            self.meta.upsert_order(str(order.id), order.mint, order.side.value, status,
                                   {"strategy": order.strategy, "reason": order.reason, "attempt": order.attempt,
                                    "sol_budget": order.sol_budget, "token_amount": order.token_amount}, signature)

    def _fail(self, order: Order, status: OrderStatus, reason: str, t0: int, **kw: Any) -> Fill:
        self._persist(order, status.value)
        t = now_ms()
        slot = int(kw.pop("slot", 0) or 0)
        log.warning("order not filled", extra={"data": {"order": order.id, "mint": order.mint, "status": status.value, "reason": reason}})
        return Fill(order.id, order.mint, order.side, order.action, status, order.strategy, order.reason, order.created_ms,
                    t0, t, t, slot, Venue.CURVE, failure=reason, attempt=order.attempt, decision_price=order.decision_price,
                    latency_ms=float(t - order.created_ms), **kw)

    @staticmethod
    def _programs_of(signed: bytes) -> list[str]:
        """Program id of every top-level instruction (programs are always static keys, even in v0 messages)."""
        try:
            from solders.transaction import VersionedTransaction

            msg = VersionedTransaction.from_bytes(signed).message
            keys = [str(k) for k in msg.account_keys]
            return [keys[ix.program_id_index] for ix in msg.instructions]
        except Exception:  # noqa: BLE001 - legacy or unparsable: fall back to venue error tables
            return []

    def _error_reason(self, err: Any, programs: list[str] | None = None) -> str:
        try:
            idx, detail = err["InstructionError"]
            code = int(detail["Custom"])
        except (KeyError, TypeError, IndexError, ValueError):
            return str(err)[:120]
        programs = programs or []
        program = programs[idx] if 0 <= int(idx) < len(programs) else None
        table = self._err_maps.get(program) if program else None
        if table is not None and code in table:
            name = table[code]
        else:  # unknown top-level program (e.g. a router CPI-ing into the venue): try the venues
            venues = (self.s.protocol.pump_program_id, self.s.protocol.pump_amm_program_id)
            name = next((self._err_maps[v][code] for v in venues if code in self._err_maps.get(v, {})), f"custom_{code}")
        return "slippage" if name in SLIPPAGE_ERRORS else name

    async def _build(self, order: Order, venue: Venue) -> bytes:
        side = "BUY" if order.side is Side.BUY else "SELL"
        amount = order.sol_budget if order.side is Side.BUY else order.token_amount
        level = self.priority.metis_level(order.urgency.value)
        micro = self.priority.micro_lamports(order.urgency.value, order.attempt)
        if venue is Venue.AMM:  # migrated token: route through Jupiter (PumpSwap and other venues)
            sol = self.s.protocol.sol_mint
            q = await self.metis.quote(sol if order.side is Side.BUY else order.mint, order.mint if order.side is Side.BUY else sol,
                                       amount, order.slippage_bps)
            fee_lamports = order.compute_units * micro // 1_000_000
            return self.signer.sign(await self.metis.swap(q, self.signer.pubkey, prioritization_fee_lamports=fee_lamports))
        if self.s.live.swap_route == "metis_swap_instructions":
            ixs = await self.metis.pump_swap_instructions(self.signer.pubkey, side, order.mint, amount, slippage_bps=order.slippage_bps,
                                                          priority_fee_level=level)
            bh, _ = await self.blockhash.get()
            tip = None
            if order.use_jito and self.jito is not None:
                tip = (await self.jito.random_tip_account(), order.jito_tip_lamports)
            return self.signer.compose([self.signer.instruction_from_json(i) for i in ixs], bh, order.compute_units, micro, tip)
        raw = await self.metis.pump_swap(self.signer.pubkey, side, order.mint, amount, slippage_bps=order.slippage_bps,
                                         priority_fee_level=level)
        return self.signer.sign(raw)

    async def _send(self, signed: bytes, order: Order) -> list[bytes]:
        if order.use_jito and self.jito is not None and self.s.live.send_via == "jito":
            bundle = [signed]
            if self.s.live.swap_route != "metis_swap_instructions":  # tip must be a separate tx in the bundle
                bh, _ = await self.blockhash.get()
                bundle.append(self.signer.compose([self.signer.tip_ix(await self.jito.random_tip_account(), order.jito_tip_lamports)], bh))
            await self.jito.send_bundle(bundle)
            return bundle
        await self.rpc.send_transaction(signed, skip_preflight=self.s.live.skip_preflight, max_retries=0)
        return [signed]

    async def execute(self, order: Order) -> Fill:
        t_decision = order.created_ms
        t0 = time.perf_counter_ns()
        st = self.quotes.market.get(order.mint)
        if st is None or st.venue is None:
            return self._fail(order, OrderStatus.REJECTED, "not_tradable", now_ms())
        try:
            signed = await self._build(order, st.venue)
        except Exception as exc:  # noqa: BLE001
            return self._fail(order, OrderStatus.REJECTED, f"build_failed:{type(exc).__name__}", now_ms())
        sig = self.signer.signature_of(signed)
        bh = self.signer.recent_blockhash_of(signed)
        programs = self._programs_of(signed)
        try:
            sent = await self._send(signed, order)
        except RpcError as exc:
            return self._fail(order, OrderStatus.REJECTED, f"send_failed:{exc.message[:80]}", now_ms())
        submit_ms = now_ms()
        q2s = self.latency.record_ns("exec.quote_to_submit", t0)
        if q2s > self.s.live.quote_to_submit_budget_ms:
            log.warning("quote-to-submit over budget", extra={"data": {"ms": round(q2s, 1), "order": order.id}})
        self._persist(order, "submitted", sig)
        fut = self.confirm.track(sig)
        deadline = time.monotonic() + self.s.live.confirm_timeout_ms / 1000.0
        while not fut.done():
            try:
                await asyncio.wait_for(asyncio.shield(fut), timeout=self.s.live.rebroadcast_ms / 1000.0)
            except TimeoutError:
                pass
            if fut.done():
                break
            valid = await self.rpc.call("isBlockhashValid", [bh, {"commitment": "processed"}])
            if (not (valid or {}).get("value", True)) or time.monotonic() > deadline:
                await self.confirm.poll_once()  # final check: it may have landed just before expiry
                if not fut.done():
                    self.confirm.cancel(sig, "expired")
                break
            for tx in sent[:1]:  # re-broadcast the *same* signed bytes (idempotent)
                try:
                    await self.rpc.send_transaction(tx, skip_preflight=True, max_retries=0)
                except RpcError:
                    pass
        res = fut.result()
        self.latency.record("exec.submit_to_confirm", res.elapsed_ms)
        if res.status == "expired":
            return self._fail(order, OrderStatus.EXPIRED, "blockhash_expired", submit_ms, signature=sig)
        tx = None
        for _ in range(5):
            tx = await self.rpc.get_transaction(sig)
            if tx is not None:
                break
            await asyncio.sleep(0.4)
        if res.status == "failed":
            fee = int(((tx or {}).get("meta") or {}).get("fee") or 0)
            return self._fail(order, OrderStatus.FAILED, self._error_reason(res.err, programs), submit_ms, signature=sig,
                              network_fee=fee, sol_delta=-fee, slot=res.slot or 0)
        fill = self._fill_from_tx(order, tx or {}, sig, t_decision, submit_ms, res.slot or 0)
        self._persist(order, fill.status.value, sig)
        log.info("fill", extra={"data": {"order": order.id, "mint": order.mint, "side": order.side.value, "tokens": fill.token_amount,
                                         "sol_delta": fill.sol_delta, "price": fill.price, "sig": sig}})
        return fill

    def _fill_from_tx(self, order: Order, tx: dict[str, Any], sig: str, t_decision: int, submit_ms: int, slot: int) -> Fill:
        """Reconstruct the fill from the confirmed transaction.

        ``sol_delta`` (the wallet's exact lamport change) drives the accounting. The price and the
        cost breakdown come from the decoded TradeEvent / PumpSwap event for our wallet: venue SOL
        leg and fees; the network fee from ``meta.fee``; token-account rent from whether our token
        account appeared (buy) or disappeared (sell); anything left over (e.g. a router platform
        fee transfer) is reported as ``platform_fee``.
        """
        meta = tx.get("meta") or {}
        msg = (tx.get("transaction") or {}).get("message") or {}
        keys = list(msg.get("accountKeys") or [])
        loaded = meta.get("loadedAddresses") or {}
        keys += list(loaded.get("writable") or []) + list(loaded.get("readonly") or [])
        keys = [k if isinstance(k, str) else k.get("pubkey", "") for k in keys]
        wallet = self.signer.pubkey
        sol_delta = 0
        if wallet in keys:
            i = keys.index(wallet)
            sol_delta = int(meta["postBalances"][i]) - int(meta["preBalances"][i])

        def mine(bal: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [b for b in bal if b.get("owner") == wallet and b.get("mint") == order.mint]

        pre, post = mine(meta.get("preTokenBalances") or []), mine(meta.get("postTokenBalances") or [])
        token_delta = sum(int(b["uiTokenAmount"]["amount"]) for b in post) - sum(int(b["uiTokenAmount"]["amount"]) for b in pre)
        rent_cfg = self.s.fees.token_account_rent_lamports
        rent = rent_cfg if (order.side is Side.BUY and post and not pre) else (-rent_cfg if (order.side is Side.SELL and pre and not post) else 0)
        pfee = cfee = lp = 0
        sol_leg = 0
        venue_event = False
        for ev in self.decoder.events_from_transaction(tx, seq=0) if self.decoder else []:
            if ev.mint == order.mint and ev.user == wallet:
                pfee, cfee, lp, sol_leg = int(ev.fee or 0), int(ev.creator_fee or 0), int(ev.lp_fee or 0), int(ev.sol_amount or 0)
                venue_event = True
        tokens = abs(token_delta)
        fee = int(meta.get("fee") or 0)
        base = self.s.fees.base_fee_lamports_per_signature * len((tx.get("transaction") or {}).get("signatures") or [1])
        platform = 0
        if order.side is Side.BUY:
            spent = -sol_delta - fee - rent                      # SOL that went to the swap (venue + router)
            venue_total = sol_leg + pfee + cfee + lp
            if venue_event:
                platform = max(0, spent - venue_total)
            price = spent / tokens / 1000.0 if tokens else 0.0
        else:
            received = sol_delta + fee + rent                    # rent < 0 on a closing sell: remove the refund
            venue_net = sol_leg - pfee - cfee - lp
            if venue_event:
                platform = max(0, venue_net - received)
            price = received / tokens / 1000.0 if tokens else 0.0
        quote_avg = float(order.meta.get("quote_avg") or 0.0)
        if quote_avg > 0 and price > 0:
            slip = (price / quote_avg - 1.0) * 1e4 if order.side is Side.BUY else (1.0 - price / quote_avg) * 1e4
        else:
            slip = 0.0
        land = int((tx.get("blockTime") or submit_ms // 1000) * 1000)
        st = self.quotes.market.get(order.mint)
        return Fill(order.id, order.mint, order.side, order.action, OrderStatus.FILLED if tokens else OrderStatus.FAILED,
                    order.strategy, order.reason, t_decision, submit_ms, max(land, submit_ms), now_ms(), slot,
                    (st.venue if st is not None and st.venue is not None else Venue.CURVE),
                    token_amount=tokens, sol_amount=sol_leg, protocol_fee=pfee, creator_fee=cfee, lp_fee=lp, platform_fee=platform,
                    network_fee=base, priority_fee=max(0, fee - base), rent=rent, sol_delta=sol_delta, price=price, slippage_bps=slip,
                    latency_ms=float(now_ms() - t_decision), attempt=order.attempt, signature=sig,
                    decision_price=order.decision_price, failure="" if tokens else "no_token_delta")
