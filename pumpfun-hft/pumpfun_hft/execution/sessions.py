"""Session archive: every paper / live trading session on the live stream is saved, so it can be
broken down afterwards (dashboard: Recent Session Breakdown) and its strategies tuned on real data.

Layout: ``<reports_dir>/sessions/<session_id>/``
    session.json      who / when / which strategies and parameters, summary metrics, signal outcomes,
                      open positions at the last save, market activity in the session window
    trades.parquet    closed round trips (Portfolio.trades_frame)
    fills.parquet     every fill and failed attempt
    equity.parquet    equity / cash / exposure samples
    signals.parquet   the most recent strategy decisions with their outcome

The recorder rewrites the files every ``every_s`` while the session runs (a crash loses at most that
much) and once more after the session stopped, adding the market statistics of its time window.

Example::

    rec = SessionRecorder(trader, settings.paths.resolve("reports_dir") / "sessions", profile="hfts")
    rec.save()                       # periodically
    rec.save(final=True, market=market_stats(store, rec.started_ms, end_ms))
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from pumpfun_hft.core.types import LAMPORTS_PER_SOL, OrderStatus
from pumpfun_hft.utils import jsonutil

FAILED = (OrderStatus.FAILED, OrderStatus.DROPPED, OrderStatus.EXPIRED, OrderStatus.REJECTED)
#: a "running" session whose files have not been rewritten for this long was interrupted (crash, closed window)
STALE_AFTER_MS = 180_000


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(jsonutil.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


class SessionRecorder:
    """Persists one LiveTrader session (see module docstring)."""

    def __init__(self, trader: Any, root: Path, profile: str = "", tuned: dict[str, Any] | None = None) -> None:
        self.trader = trader
        self.profile = profile or ("paper" if trader.session == "paper" else trader.session)
        self.started_ms = int(time.time() * 1000)
        stamp = datetime.fromtimestamp(self.started_ms / 1000, UTC).strftime("%Y%m%d-%H%M%S")
        self.session_id = f"{self.profile}-{stamp}"
        self.dir = Path(root) / self.session_id
        self.tuned = tuned or {}
        s = trader.s
        names = [st.name for st in trader.runtime.strategies]
        overlay = s.strategy.exit_overlay
        self.params = {n: dict(s.strategy.params.get(n, {})) for n in [*names, *([overlay] if overlay else [])]}

    def summary(self) -> dict[str, Any]:
        t, pf = self.trader, self.trader.portfolio
        trades = pf.trades
        eq = pf.equity_lamports() / LAMPORTS_PER_SOL
        col = t.collector_status() if t.collector_status else {}
        return {"initial_sol": t.initial_sol, "equity_sol": eq, "return": eq / t.initial_sol - 1 if t.initial_sol else None,
                "realized_pnl_sol": sum(x.pnl_sol for x in trades), "closed_trades": len(trades),
                "wins": sum(1 for x in trades if x.pnl_sol > 0),
                "fills": sum(1 for f in t.fills if f.status not in FAILED), "failed_fills": sum(1 for f in t.fills if f.status in FAILED),
                "open_positions": sum(1 for p in pf.positions.values() if p.tokens > 0), "events_processed": t.events_processed,
                "stream": {k: col.get(k) for k in ("events", "reconnects", "truncated", "dropped") if k in col}}

    def save(self, final: bool = False, market: dict[str, Any] | None = None) -> Path:
        t, pf = self.trader, self.trader.portfolio
        self.dir.mkdir(parents=True, exist_ok=True)
        now = int(time.time() * 1000)
        _write_parquet(pf.trades_frame(), self.dir / "trades.parquet")
        _write_parquet(pf.equity_frame(), self.dir / "equity.parquet")
        fills = [f.to_dict() for f in t.fills]
        _write_parquet(pl.DataFrame(fills, infer_schema_length=None) if fills else pl.DataFrame(), self.dir / "fills.parquet")
        sig = list(t.runtime.signal_records)
        _write_parquet(pl.DataFrame(sig, infer_schema_length=None) if sig else pl.DataFrame(), self.dir / "signals.parquet")
        info = {"session_id": self.session_id, "profile": self.profile, "session": t.session,
                "status": "ended" if final else "running", "started_ms": self.started_ms, "updated_ms": now,
                "ended_ms": now if final else None, "strategies": [st.name for st in t.runtime.strategies],
                "exit_overlay": t.s.strategy.exit_overlay, "params": self.params, "tuned": self.tuned,
                "config_fingerprint": t.s.fingerprint(), "summary": self.summary(),
                "signal_counts": dict(t.runtime.counts),
                "positions": (t.last_snapshot or {}).get("positions", []), "market": market}
        _write_json(info, self.dir / "session.json")
        return self.dir


def market_stats(store: Any, start_ms: int, end_ms: int) -> dict[str, Any]:
    """What the market did in a session window, from the events the stream recorded."""
    cols = ["kind", "mint", "user", "sol_amount", "is_buy"]
    lf = store.scan(start_ms, end_ms, columns=cols)
    by_kind = lf.group_by("kind").len().collect()
    swaps = lf.filter(pl.col("kind").is_in(["trade", "amm_buy", "amm_sell"]))
    # Volume comes from bonding-curve trades only: PumpSwap pool events can carry the pool's other
    # asset in sol_amount (pools of unknown tokens, and pools whose base / quote order is reversed),
    # which would inflate the total by orders of magnitude. Pool trades are counted, not summed.
    pools = swaps.filter(pl.col("kind") != "trade").select(pl.len().alias("n"), pl.col("mint").is_null().sum().alias("unknown")).collect()
    curve = swaps.filter(pl.col("kind") == "trade")
    agg = curve.select(pl.len().alias("curve_trades"), pl.col("mint").n_unique().alias("tokens_traded"),
                       pl.col("user").n_unique().alias("wallets"), (pl.col("sol_amount").sum() / LAMPORTS_PER_SOL).alias("volume_sol"),
                       (pl.col("sol_amount").filter(pl.col("is_buy")).sum() / LAMPORTS_PER_SOL).alias("buy_volume_sol")).collect()
    row = agg.row(0, named=True) if agg.height else {}
    row["pool_trades"] = int(pools["n"][0]) if pools.height else 0
    row["unknown_pool_trades"] = int(pools["unknown"][0] or 0) if pools.height else 0
    kinds = {r["kind"]: int(r["len"]) for r in by_kind.iter_rows(named=True)}
    return {"events": sum(kinds.values()), "by_kind": kinds, "tokens_created": kinds.get("create", 0),
            "graduations": kinds.get("complete", 0), **{k: (float(v) if isinstance(v, float) else int(v or 0)) for k, v in row.items()}}


# --------------------------------------------------------------------------- reading the archive
def sessions_root(settings: Any) -> Path:
    """Sessions of the stream live in the real data set (hfts / hftl write there), whatever the dashboard runs on."""
    from pumpfun_hft.dashboard.builder import dataset_root

    return dataset_root("real", settings) / "reports" / "sessions"


def list_sessions(root: Path) -> list[dict[str, Any]]:
    """session.json of every saved session, newest first; a "running" one that stopped saving is "interrupted"."""
    out = []
    now = int(time.time() * 1000)
    for f in Path(root).glob("*/session.json"):
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if info.get("status") == "running" and now - int(info.get("updated_ms") or 0) > STALE_AFTER_MS:
            info["status"] = "interrupted"
        out.append(info)
    return sorted(out, key=lambda i: int(i.get("started_ms") or 0), reverse=True)


def load_frame(root: Path, session_id: str, name: str) -> pl.DataFrame:
    """One table of a session (trades / fills / equity / signals); empty when missing."""
    if not session_id or "/" in session_id or "\\" in session_id or session_id.startswith("."):
        return pl.DataFrame()
    p = Path(root) / session_id / f"{name}.parquet"
    try:
        return pl.read_parquet(p) if p.exists() else pl.DataFrame()
    except Exception:  # noqa: BLE001 - a file being rewritten right now: show nothing rather than fail the page
        return pl.DataFrame()


def strategy_breakdown(info: dict[str, Any], trades: pl.DataFrame) -> pl.DataFrame:
    """Per strategy: trades, wins, PnL, returns, holding time, costs, and what happened to its buy signals."""
    names = list(dict.fromkeys([*(info.get("strategies") or []), *(trades["strategy"].unique().to_list() if trades.height else [])]))
    counts = info.get("signal_counts") or {}
    rows = []
    for n in names:
        t = trades.filter(pl.col("strategy") == n) if trades.height else trades
        wins = t.filter(pl.col("pnl_sol") > 0) if t.height else t
        losses = t.filter(pl.col("pnl_sol") <= 0) if t.height else t
        gross_loss = float(-losses["pnl_sol"].sum()) if losses.height else 0.0
        buys = {k.split(":", 2)[2]: v for k, v in counts.items() if k.startswith(f"{n}:BUY:")}
        rows.append({
            "strategy": n, "trades": t.height, "wins": wins.height, "win_rate": wins.height / t.height if t.height else None,
            "pnl_sol": float(t["pnl_sol"].sum()) if t.height else 0.0,
            "avg_ret": float(t["ret"].mean()) if t.height else None,
            "best_ret": float(t["ret"].max()) if t.height else None, "worst_ret": float(t["ret"].min()) if t.height else None,
            "profit_factor": (float(wins["pnl_sol"].sum()) / gross_loss) if gross_loss > 0 else None,
            "avg_hold_s": float(t["hold_s"].mean()) if t.height else None,
            "costs_sol": float((t["fees_sol"] + t["tx_costs_sol"]).sum()) if t.height else 0.0,
            "buy_signals": sum(buys.values()), "submitted": buys.get("submitted", 0)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def strategy_evidence(root: Path, strategy: str) -> dict[str, Any]:
    """Every saved session's closed trades of one strategy, pooled (the live record behind a tuning decision)."""
    frames, n_sessions = [], 0
    for info in list_sessions(root):
        t = load_frame(root, info["session_id"], "trades")
        if t.height and "strategy" in t.columns:
            t = t.filter(pl.col("strategy") == strategy)
            if t.height:
                frames.append(t.select("pnl_sol", "ret", "hold_s"))
                n_sessions += 1
    if not frames:
        return {"sessions": 0, "trades": 0}
    t = pl.concat(frames)
    return {"sessions": n_sessions, "trades": t.height, "win_rate": float((t["pnl_sol"] > 0).mean()),
            "pnl_sol": float(t["pnl_sol"].sum()), "avg_ret": float(t["ret"].mean()), "avg_hold_s": float(t["hold_s"].mean())}
