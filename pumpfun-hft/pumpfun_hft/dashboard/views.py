"""Dashboard page renderers (shared by the FastAPI app and the static single-file export).

Each ``page_*`` function returns an HTML fragment; charts come from
:mod:`pumpfun_hft.analytics.charts` so the dashboard and the reports share one visual system.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.analytics import charts as C
from pumpfun_hft.analytics.metrics import hourly_profile, monthly_returns
from pumpfun_hft.analytics.report import EXIT_FORMATS, EXIT_HEADERS, TRADE_FORMATS, TRADE_HEADERS, exit_categories
from pumpfun_hft.backtester.results import BacktestResult
from pumpfun_hft.utils.timeutil import ms_to_iso

PAGES = [("builder", "Strategy builder"), ("sessions", "Recent Session Breakdown"), ("params", "Strategy Parameters"),
         ("overview", "Overview"), ("trades", "Trades"), ("equity", "Equity curve"), ("drawdown", "Drawdown"),
         ("heatmaps", "Heatmaps"), ("wallets", "Wallet explorer"), ("tokens", "Token explorer"),
         ("features", "Feature importance"), ("strategies", "Strategy comparison"), ("live", "Live monitor")]


@dataclass
class DashboardData:
    """Loads persisted runs and research artefacts from ``reports_dir`` / ``data_dir``."""

    reports_dir: Path
    events_dir: Path | None = None
    meta: Any = None
    _cache: dict[str, BacktestResult] = field(default_factory=dict)

    def run_dirs(self) -> list[Path]:
        root = self.reports_dir / "runs"
        if not root.exists():
            return []
        return sorted([p for p in root.iterdir() if (p / "result.json").exists()], key=lambda p: p.stat().st_mtime, reverse=True)

    def run_ids(self) -> list[str]:
        return [p.name for p in self.run_dirs()]

    def load(self, run_id: str | None, strict: bool = False) -> BacktestResult | None:
        """The run ``run_id`` (latest run when it is None, or unknown and ``strict`` is False)."""
        dirs = self.run_dirs()
        match = next((p for p in dirs if p.name == run_id), None)
        if match is None and (strict or not dirs):
            return None
        d = match or dirs[0]
        key = f"{d.name}@{(d / 'result.json').stat().st_mtime_ns}"  # a re-saved run is reloaded
        if key not in self._cache:
            self._cache[key] = BacktestResult.load(d)
        return self._cache[key]

    def artefact(self, run_id: str | None, name: str) -> pl.DataFrame:
        dirs = self.run_dirs()
        if not dirs:
            return pl.DataFrame()
        d = next((p for p in dirs if p.name == run_id), dirs[0])
        p = d / name
        return pl.read_parquet(p) if p.exists() else pl.DataFrame()

    def importance(self) -> pl.DataFrame:
        for p in sorted(self.reports_dir.glob("ml/*/importance.parquet"), key=lambda x: x.stat().st_mtime, reverse=True):
            return pl.read_parquet(p)
        return pl.DataFrame()

    def token_prices(self, mint: str) -> pl.DataFrame:
        if self.events_dir is None or not any(self.events_dir.glob("date=*/*.parquet")):
            return pl.DataFrame()
        lf = pl.scan_parquet(str(self.events_dir / "date=*" / "*.parquet"), hive_partitioning=True)
        return (lf.filter((pl.col("mint") == mint) & pl.col("kind").is_in(["trade", "amm_buy", "amm_sell"]))
                .select("ts_ms", (pl.col("v_sol").cast(pl.Float64) / pl.col("v_tok").cast(pl.Float64) / 1000.0).alias("price"),
                        "is_buy", "sol_amount", "user")
                .sort("ts_ms").collect())


STRAT_FORMATS = {"pnl_sol": "sol_s", "win_rate": "pct", "avg_ret": "pct_s", "total_return": "pct_s", "max_drawdown": "pct",
                 "sharpe": "num2", "profit_factor": "num2", "fill_rate": "pct", "n_trades": "int"}
STRAT_HEADERS = {"pnl_sol": "PnL SOL", "win_rate": "win rate", "avg_ret": "avg return", "total_return": "total return",
                 "max_drawdown": "max drawdown", "profit_factor": "profit factor", "fill_rate": "fill rate", "n_trades": "trades"}
FILL_FORMATS = {"mint": "addr", "land_ms": "datetime", "token_amount": "int", "sol_delta": "lamports_sol", "price": "sci",
                "slippage_bps": "bps", "latency_ms": "ms", "reason": "text", "failure": "text"}
FILL_HEADERS = {"order_id": "order", "land_ms": "landed (UTC)", "token_amount": "tokens (raw)", "sol_delta": "SOL delta",
                "price": "price SOL", "slippage_bps": "slippage", "latency_ms": "latency"}
TOKEN_FORMATS = {"mint": "addr", "creator": "addr", "created_ms": "datetime", "ath_multiple": "num2", "max_dd_pct": "num1",
                 "max_liq_dd_pct": "num1",
                 "creator_sold_pct": "num1", "name": "text", "symbol": "text"}
TOKEN_HEADERS = {"created_ms": "created (UTC)", "n_trades": "trades", "ath_multiple": "ATH ×", "max_dd_pct": "max price DD %",
                 "max_liq_dd_pct": "max liquidity DD %", "creator_sold_pct": "creator sold %"}
WALLET_HEADERS = {"first_seen_ms": "first seen", "last_seen_ms": "last seen", "updated_ms": "updated", "n_trades": "trades",
                  "n_buys": "buys", "n_sells": "sells", "buy_sol": "bought SOL", "sell_sol": "sold SOL", "tokens_traded": "tokens",
                  "closed": "round trips", "realized_pnl_sol": "realised PnL SOL", "mean_ret": "mean return", "smart_score": "smart score"}
WALLET_FORMATS = {"address": "addr", "cluster": "addr", "smart_score": "num2", "realized_pnl_sol": "sol_s", "labels": "text",
                  "first_seen_ms": "datetime", "last_seen_ms": "datetime", "updated_ms": "datetime", "buy_sol": "sol",
                  "sell_sol": "sol", "mean_ret": "pct_s"}


def _kpi(label: str, value: str, cls: str = "") -> str:
    return f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v {cls}">{value}</div></div>'


def _pct(v: Any, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    return f"{100 * float(v):+.2f}%" if signed else f"{100 * float(v):.2f}%"


def _num(v: Any, d: int = 2) -> str:
    return C.fmt_num(v, d)


def page_overview(r: BacktestResult | None) -> str:
    if r is None:
        return '<p class="muted">No runs yet. Run <code>python -m pumpfun_hft.main backtest</code> first.</p>'
    m = r.metrics
    pnl = m.get("pnl_sol") or 0.0
    banner = ('<div class="banner"><b>Synthetic data.</b> This run uses the built-in synthetic market; results only '
              'demonstrate the pipeline.</div>') if r.synthetic else ""
    tiles = "".join([
        _kpi("PnL (SOL)", f"{pnl:+.4f}", "pos" if pnl > 0 else "neg" if pnl < 0 else ""),
        _kpi("Total return", _pct(m.get("total_return"), True)), _kpi("Sharpe", _num(m.get("sharpe"))),
        _kpi("Max drawdown", _pct(m.get("max_drawdown"))), _kpi("Trades", _num(m.get("n_trades"), 0)),
        _kpi("Win rate", _pct(m.get("win_rate"))), _kpi("Profit factor", _num(m.get("profit_factor"))),
        _kpi("Fill rate", _pct(m.get("fill_rate"))),
    ])
    t = r.trades
    by_strat = (t.group_by("strategy").agg(pl.len().alias("trades"), pl.col("pnl_sol").sum(), (pl.col("pnl_sol") > 0).mean().alias("win_rate"),
                                           pl.col("ret").mean().alias("avg_ret")) if t.height else pl.DataFrame())
    note = ""
    if t.height:
        gap = pnl - float(t["pnl_sol"].sum())
        note = (f"<p class='muted'>Closed round trips only. Total PnL above also counts fees of failed attempts that never "
                f"opened a position and any position still open at the end ({gap:+.4f} SOL here).</p>")
    return (f"{banner}<div class='kpis'>{tiles}</div>"
            f"<div class='card'><h3>Equity</h3>{C.fig_html(C.equity_drawdown(r.equity, r.initial_capital_sol), 'ov_eq')}</div>"
            f"<div class='card'><h3>By strategy</h3>{C.html_table(by_strat, formats=STRAT_FORMATS, headers=STRAT_HEADERS, scroll=False)}{note}</div>"
            f"<p class='muted'>Run {html.escape(r.run_id)} · {ms_to_iso(r.start_ms)} → {ms_to_iso(r.end_ms)} · config "
            f"{html.escape(str(r.config_hash))} · data {html.escape(str(r.data_hash))}</p>")


def page_trades(r: BacktestResult | None, max_rows: int) -> str:
    if r is None:
        return ""
    f = r.fills
    cols = [c for c in ["order_id", "mint", "side", "action", "status", "strategy", "reason", "land_ms", "token_amount", "sol_delta",
                        "price", "slippage_bps", "latency_ms", "failure"] if c in f.columns]
    fills = f.select(cols).sort("land_ms", descending=True) if f.height else f
    tcols = [c for c in ["trade_id", "mint", "strategy", "entry_ms", "exit_ms", "cost_sol", "pnl_sol", "ret", "r_multiple", "mae", "mfe",
                         "hold_s", "exit_reason", "fees_sol", "tx_costs_sol"] if c in r.trades.columns]
    trades = r.trades.select(tcols).sort("exit_ms", descending=True) if r.trades.height else r.trades
    ex = exit_categories(r.trades)
    return (f"<div class='card'><h3>Closed trades ({r.trades.height:,})</h3>"
            f"{C.html_table(trades, max_rows, formats=TRADE_FORMATS, headers=TRADE_HEADERS, full_data='report/csv/trades.csv')}</div>"
            f"<div class='card'><h3>Exit reasons</h3>{C.html_table(ex, formats=EXIT_FORMATS, headers=EXIT_HEADERS, scroll=False)}</div>"
            f"<div class='card'><h3>Every fill and failed attempt ({f.height:,})</h3>"
            f"{C.html_table(fills, max_rows, formats=FILL_FORMATS, headers=FILL_HEADERS, full_data='report/csv/fills.csv')}</div>")


def page_equity(r: BacktestResult | None) -> str:
    if r is None:
        return ""
    e = r.equity
    return (f"<div class='card'><h3>Equity curve (zoom with drag, double-click to reset)</h3>"
            f"{C.fig_html(C.equity_drawdown(e, r.initial_capital_sol), 'eq_full')}</div>"
            f"<div class='card'><h3>Monthly returns</h3>{C.html_table(_monthly(e), formats={'return': 'pct_s'}, scroll=False)}</div>")


def _monthly(e: pl.DataFrame) -> pl.DataFrame:
    mon = monthly_returns(e)
    if mon.is_empty():
        return mon
    return mon.select(pl.format("{}-{}", pl.col("year"), pl.col("month").cast(pl.String).str.zfill(2)).alias("month"),
                      pl.col("ret").alias("return"))


def page_drawdown(r: BacktestResult | None) -> str:
    if r is None or r.equity.is_empty():
        return ""
    e = r.equity
    eq = e["equity_sol"].to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[r.initial_capital_sol], eq]))[1:]
    dd = eq / peak - 1.0
    # top drawdown episodes
    episodes = []
    in_dd, start, trough, trough_v = False, 0, 0, 0.0
    ts = e["ts_ms"].to_list()
    for i, v in enumerate(dd):
        if v < 0 and not in_dd:
            in_dd, start, trough, trough_v = True, i, i, v
        elif v < 0 and in_dd and v < trough_v:
            trough, trough_v = i, v
        elif v >= 0 and in_dd:
            episodes.append({"start": ms_to_iso(ts[max(0, start - 1)]), "trough": ms_to_iso(ts[trough]), "recovered": ms_to_iso(ts[i]),
                             "depth_pct": 100 * trough_v, "duration_h": (ts[i] - ts[max(0, start - 1)]) / 3.6e6})
            in_dd = False
    if in_dd:
        episodes.append({"start": ms_to_iso(ts[max(0, start - 1)]), "trough": ms_to_iso(ts[trough]), "recovered": "not recovered",
                         "depth_pct": 100 * trough_v, "duration_h": (ts[-1] - ts[max(0, start - 1)]) / 3.6e6})
    ep = pl.DataFrame(episodes).sort("depth_pct").head(10) if episodes else pl.DataFrame()
    m = r.metrics
    tiles = "".join([_kpi("Max drawdown", _pct(m.get("max_drawdown"))), _kpi("Longest under water", f"{_num(m.get('longest_underwater_h'))} h"),
                     _kpi("Time under water", _pct(m.get("pct_time_underwater"))), _kpi("Return / max DD", _num(m.get("recovery_factor")))])
    return (f"<div class='kpis'>{tiles}</div><div class='card'><h3>Drawdown from peak</h3>"
            f"{C.fig_html(C.equity_drawdown(e, r.initial_capital_sol), 'dd_full')}</div>"
            f"<div class='card'><h3>Deepest drawdown episodes</h3>{C.html_table(ep, formats={'depth_pct': 'num2', 'duration_h': 'num2'}, headers={'depth_pct': 'depth %', 'duration_h': 'duration h'}, scroll=False)}</div>")


def page_heatmaps(r: BacktestResult | None) -> str:
    if r is None or r.trades.is_empty():
        return '<p class="muted">No trades.</p>'
    hp = hourly_profile(r.trades)
    z = np.full((7, 24), np.nan)
    for wd, h, p in hp.select("weekday", "hour", "pnl_sol").iter_rows():
        z[int(wd) - 1, int(h)] = p
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_hour = hp.group_by("hour").agg(pl.col("pnl_sol").sum(), pl.col("n").sum()).sort("hour")
    return (f"<div class='card'><h3>PnL by weekday × UTC hour of entry (SOL)</h3>{C.fig_html(C.diverging_heatmap(z, list(range(24)), days, 'PnL (SOL)', '.4f'), 'hm')}</div>"
            f"<div class='card'><h3>PnL by hour</h3>{C.fig_html(C.signed_bars(by_hour['hour'].to_list(), by_hour['pnl_sol'].to_numpy(), 'UTC hour', 'PnL (SOL)'), 'hm_h')}</div>"
            f"<div class='card'><h3>Table view</h3>{C.html_table(hp, formats={'pnl_sol': 'sol_s'})}</div>")


def page_wallets(wallets: pl.DataFrame, max_rows: int) -> str:
    if wallets.is_empty():
        return '<p class="muted">No wallet snapshot for this run.</p>'
    w = wallets.sort("smart_score", descending=True, nulls_last=True)
    import plotly.graph_objects as go

    ranked = w.filter(pl.col("smart_score").is_not_null())
    fig = go.Figure(go.Scatter(x=ranked["closed"], y=ranked["realized_pnl_sol"], mode="markers",
                               marker={"size": 8, "color": ranked["smart_score"], "colorscale": [[0, C.SEQ_BLUE[0]], [1, C.SEQ_BLUE[-1]]],
                                       "colorbar": {"title": {"text": "smart score"}, "thickness": 12}, "line": {"color": C.LIGHT["surface"], "width": 2}},
                               customdata=np.stack([ranked["address"].to_numpy(), ranked["labels"].to_numpy()], axis=1),
                               hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>closed %{x} · PnL %{y:.3f} SOL<extra></extra>"))
    fig.update_xaxes(title_text="closed round trips (log scale)", type="log", tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    fig.update_yaxes(title_text="realised PnL (SOL)")
    labels = (wallets.with_columns(pl.col("labels").str.split(",")).explode("labels").filter(pl.col("labels") != "")
              .group_by("labels").len().sort("len", descending=True))
    return (f"<div class='grid2'><div class='card'><h3>Ranked wallets: activity vs realised PnL</h3>{C.fig_html(C.base_layout(fig, 360, False), 'w_sc')}</div>"
            f"<div class='card'><h3>Wallet classes</h3>{C.html_table(labels, headers={'labels': 'label', 'len': 'wallets'}, scroll=False)}</div></div>"
            f"<div class='card'><h3>Wallet database ({wallets.height:,} wallets; point-in-time at the end of the run)</h3>"
            f"<input id='wq' placeholder='filter by address or label' style='width:100%;padding:6px;margin-bottom:6px'>"
            f"{C.html_table(w, max_rows, formats=WALLET_FORMATS, headers=WALLET_HEADERS, full_data='wallets.parquet in the run folder')}</div>"
            "<script>document.getElementById('wq').addEventListener('input',function(e){const q=e.target.value.toLowerCase();"
            "document.querySelectorAll('#wq ~ .scroll tbody tr').forEach(function(r){r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';});});</script>")


def page_tokens(r: BacktestResult | None, data: DashboardData, tokens: pl.DataFrame, mint: str | None) -> str:
    if r is None:
        return ""
    traded = r.trades.group_by("mint").agg(pl.col("pnl_sol").sum(), pl.len().alias("trades")).sort("pnl_sol") if r.trades.height else pl.DataFrame()
    if mint is None and traded.height:
        mint = traded["mint"][-1]
    chart = ""
    if mint:
        prices = data.token_prices(mint)
        fills = r.fills.filter(pl.col("mint") == mint) if r.fills.height else r.fills
        chart = (f"<div class='card'><h3>{html.escape(mint)} — price with our fills</h3>"
                 f"{C.fig_html(C.price_with_fills(prices, fills), 'tok_px')}</div>")
    tok = tokens if not tokens.is_empty() else pl.DataFrame()
    options = "".join(f"<option value='{html.escape(m, quote=True)}' {'selected' if m == mint else ''}>{html.escape(C.short_addr(m, 6, 6))} ({p:+.4f} SOL)</option>"
                      for m, p in zip(traded["mint"].to_list()[::-1], traded["pnl_sol"].to_list()[::-1], strict=True)) if traded.height else ""
    picker = (f"<form method='get'><input type='hidden' name='run' value='{html.escape(r.run_id)}'><select name='mint' onchange='this.form.submit()'>"
              f"{options}</select></form>") if options else ""
    return (f"{picker}{chart}<div class='card'><h3>Tokens seen (outcomes resolved point-in-time)</h3>"
            f"{C.html_table(tok, 1000, formats=TOKEN_FORMATS, headers=TOKEN_HEADERS, full_data='tokens.parquet in the run folder')}</div>")


def page_features(imp: pl.DataFrame) -> str:
    if imp.is_empty():
        return '<p class="muted">No model trained yet. Run <code>python -m pumpfun_hft.main train-model</code>.</p>'
    top = imp.head(20)
    cards = []
    for col, title in (("native", "Model-native importance"), ("permutation", "Permutation importance (AUC drop, last CV fold)"), ("shap", "Mean |SHAP|")):
        if col in top.columns and top[col].is_not_null().any():
            t = top.drop_nulls(col).sort(col)
            cards.append(f"<div class='card'><h3>{title}</h3>{C.fig_html(C.hbars(t['feature'].to_list(), t[col].to_list(), col), 'fi_' + col)}</div>")
    note = ("<p class='muted'>Model-native importance (split counts or impurity) favours features with many distinct values, "
            "such as time of day. Permutation importance, the AUC lost on held-out data when one feature is shuffled, is the "
            "more reliable ranking; mean |SHAP| is each feature's average contribution to a prediction.</p>")
    return note + "".join(cards) + f"<div class='card'><h3>Table view</h3>{C.html_table(imp, formats={'feature': 'text'})}</div>"


def page_strategies(data: DashboardData) -> str:
    rows, series, colors = [], {}, {}
    for rid in data.run_ids()[:8]:
        r = data.load(rid)
        if r is None:
            continue
        name = f"{'+'.join(r.strategies)} · {rid[-6:]}"
        colors[name] = C.strategy_color(r.strategies[0]) if len(r.strategies) == 1 else C.OTHER
        m = r.metrics
        rows.append({"run": rid, "strategies": ",".join(r.strategies), "synthetic": r.synthetic, "total_return": m.get("total_return"),
                     "sharpe": m.get("sharpe"), "max_drawdown": m.get("max_drawdown"), "n_trades": m.get("n_trades"),
                     "win_rate": m.get("win_rate"), "profit_factor": m.get("profit_factor"), "fill_rate": m.get("fill_rate")})
        if r.equity.height:
            e = r.equity
            series[name] = (((e["ts_ms"] - e["ts_ms"][0]) / 3.6e6).to_list(), (e["equity_sol"] / r.initial_capital_sol).to_list())
    if not rows:
        return '<p class="muted">No runs.</p>'
    fig = C.multi_line(series, "equity / initial", colors)
    fig.update_xaxes(title_text="hours since start")
    return (f"<div class='card'><h3>Equity, indexed to 1.0 at start (first 8 runs)</h3>{C.fig_html(fig, 'cmp')}</div>"
            f"<div class='card'><h3>Metrics</h3>{C.html_table(pl.DataFrame(rows, infer_schema_length=None), formats=STRAT_FORMATS, headers=STRAT_HEADERS, scroll=False)}</div>")


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) or math.isinf(f) else f


def _signed_cls(v: float) -> str:
    return "pos" if v > 0 else "neg" if v < 0 else ""


def page_live(state: dict[str, Any] | None, refresh_ms: int, server: bool) -> str:
    """Live monitor. On the server the whole body is re-fetched from ``/api/live/html`` every ``refresh_ms``."""
    stale_ms = max(5 * refresh_ms, 15_000)  # a few missed snapshots before we call the session gone
    poll = (f"<script>(function(){{"
            f"function paint(){{var d=document.getElementById('lv-data'); if(!d||!d.dataset.updated) return;"
            f"var age=(Date.now()-Number(d.dataset.updated))/1000;"
            f"var st=document.getElementById('lv-status'), ageEl=document.getElementById('lv-age');"
            f"if(ageEl) ageEl.innerText=age.toFixed(0)+' s ago';"
            f"if(!st) return;"
            f"if(d.dataset.stopped==='1'){{st.className='banner';st.innerText='Session ended. This is the final snapshot, taken after positions were closed.';}}"
            f"else if(age*1000>{stale_ms}){{st.className='banner';st.innerText='Not updating — the last snapshot is '+age.toFixed(0)+' s old. The trading session has stopped or is blocked.';}}"
            f"else {{st.className='muted';st.innerText='Live — updating every '+({refresh_ms}/1000).toFixed(0)+' s.';}}}}"
            f"function pull(){{fetch('/api/live/html').then(r=>r.text()).then(function(h){{"
            f"var root=document.getElementById('lv-root'); if(!root) return; var det=root.querySelector('details'), open=det&&det.open;"
            f"root.innerHTML=h; if(open){{var d2=root.querySelector('details'); if(d2) d2.open=true;}} paint();}}).catch(function(){{}});}}"
            f"setInterval(pull, {refresh_ms}); setInterval(paint, 1000); paint();}})();</script>"
            ) if server else ""
    return f"<div id='lv-root'>{live_body(state, server)}</div>{poll}"


#: Open positions columns, in display order (snapshot keys of LiveTrader.snapshot()["positions"])
POSITION_COLUMNS = ("mint", "strategy", "tokens", "value_sol", "cost_sol", "ret", "entry_ms")


def _positions_frame(positions: list[dict[str, Any]]) -> pl.DataFrame:
    """Open positions newest first, with exactly the columns of the table (missing keys of old snapshots -> null)."""
    if not positions:
        return pl.DataFrame()
    from pumpfun_hft.core.types import TOKEN_UNIT

    rows = [{c: p.get(c) for c in POSITION_COLUMNS} for p in positions]
    for r in rows:  # snapshots carry raw base units; show whole tokens (6 decimals)
        r["tokens"] = None if r["tokens"] is None else r["tokens"] / TOKEN_UNIT
    schema = {"mint": pl.String, "strategy": pl.String, "tokens": pl.Float64, "value_sol": pl.Float64, "cost_sol": pl.Float64,
              "ret": pl.Float64, "entry_ms": pl.Int64}
    return pl.DataFrame(rows, schema=schema).sort("entry_ms", descending=True, nulls_last=True)


def _compact(v: float | None) -> str:
    """1,370,026.58 -> 1.37M (token amounts span many magnitudes)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suffix}"
    return f"{v:,.2f}"


def signed_badge(v: float | None, kind: str = "pct") -> str:
    """A return / PnL as a green (gain) or red (loss) badge."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<span class='sig flat'>–</span>"
    cls = "up" if v > 0 else "down" if v < 0 else "flat"
    text = f"{100 * v:+.2f}%" if kind == "pct" else f"{v:+.4f}"
    return f"<span class='sig {cls}'>{text}</span>"


def positions_html(df: pl.DataFrame) -> str:
    """Open positions as a fixed-layout list: it always fits its card (long mints wrap, narrow cards
    switch to one block per position), so there is never a scroll bar."""
    if df.is_empty():
        return "<p class='muted'>No open positions.</p>"
    cols = (("mint", "l", 29), ("strategy", "l", 17), ("tokens", "", 9), ("value (SOL)", "", 11), ("cost (SOL)", "", 11),
            ("return", "", 11), ("entry (UTC)", "", 12))
    colgroup = "".join(f"<col style='width:{w}%'>" for _, _, w in cols)
    head = "".join(f"<th class='{c}'>{n}</th>" for n, c, _ in cols)
    rows = []
    for r in df.iter_rows(named=True):
        mint, strat = html.escape(str(r.get("mint") or "")), html.escape(str(r.get("strategy") or ""))
        rows.append(
            "<tr>"
            f"<td class='l pm' data-l='mint' title='{mint}'>{mint}</td>"
            f"<td class='l ps' data-l='strategy'><span class='pill' title='{strat}'>{strat}</span></td>"
            f"<td data-l='tokens' title='{C.fmt_value(r.get('tokens'), 'num2')}'>{_compact(r.get('tokens'))}</td>"
            f"<td data-l='value (SOL)'>{C.fmt_value(r.get('value_sol'), 'sol')}</td>"
            f"<td data-l='cost (SOL)'>{C.fmt_value(r.get('cost_sol'), 'sol')}</td>"
            f"<td data-l='return'>{signed_badge(r.get('ret'))}</td>"
            f"<td data-l='entry (UTC)'>{C.fmt_value(r.get('entry_ms'), 'time_cs')}</td>"
            "</tr>")
    return (f"<div class='pwrap'><table class='ptbl'><colgroup>{colgroup}</colgroup><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def live_body(state: dict[str, Any] | None, server: bool) -> str:
    """Content of the Live monitor for one snapshot of the trading session (``LiveTrader.snapshot``)."""
    if not state:
        return ('<p class="muted">No live session state found. Start one: <code>hfts trade</code> (paper trading on the live '
                'stream), <code>hftl trade</code> (live) or <code>hftr trade</code> (paper on recorded events). Run the dashboard '
                'with the same command (e.g. <code>hfts dashboard</code>) so it reads that data set; this page follows the '
                'session\'s snapshots.</p>')
    v = state.get("value", {})
    updated = int(_num(state.get("updated_ms")))
    eq, init, pnl = _num(v.get("equity_sol")), _num(v.get("initial_sol")), _num(v.get("realized_pnl_sol"))
    ret = eq / init - 1 if init else None
    closed, wins = int(_num(v.get("closed_trades"))), int(_num(v.get("wins")))
    pos = v.get("positions") or []
    open_sol = sum(_num(p.get("value_sol")) for p in pos)
    col = v.get("collector") or {}
    session = " · ".join(html.escape(str(x)) for x in (v.get("profile"), v.get("session") or "paper") if x)
    tiles = [_kpi("Session", session),
             _kpi("Equity (SOL)", f"{eq:.4f}"),
             _kpi("Return", _pct(ret, signed=True), _signed_cls(ret or 0.0)),
             _kpi("Realised PnL (SOL)", f"{pnl:+.4f}", _signed_cls(pnl)),
             _kpi("Closed trades", f"{closed:,}" + (f" <span class='muted' style='font-size:13px'>{wins / closed:.0%} won</span>" if closed else "")),
             _kpi("Open positions", f"{len(pos)} <span class='muted' style='font-size:13px'>{open_sol:.4f} SOL</span>"),
             _kpi("Fills / failed", f"{int(_num(v.get('fills'))):,} / {int(_num(v.get('failed_fills'))):,}"),
             _kpi("Events processed", f"{int(_num(v.get('events_processed'))):,}")]
    if col:
        tiles.append(_kpi("Stream", f"{int(_num(col.get('events'))):,} <span class='muted' style='font-size:13px'>"
                                    f"{int(_num(col.get('reconnects')))} reconnects</span>"))
    tiles.append(_kpi("Pending / queued", f"{int(_num(v.get('pending_orders')))} / {int(_num(v.get('queue')))}"))
    if v.get("started_ms"):
        tiles.append(_kpi("Started (UTC)", C.fmt_value(v.get("started_ms"), "datetime")))
    tiles.append(_kpi("Last snapshot", f"<span id='lv-age'>{C.fmt_value(updated, 'datetime')}</span>"))
    status = ('<div id="lv-status" class="banner">Session ended. This is the final snapshot, taken after positions were closed.</div>'
              if v.get("stopped") else
              f'<p id="lv-status" class="muted">Snapshot taken {C.fmt_value(updated, "datetime")} UTC.</p>' if not server else
              '<p id="lv-status" class="muted">Live — updating.</p>')
    data = f"<div id='lv-data' hidden data-updated='{updated}' data-stopped='{1 if v.get('stopped') else 0}'></div>"
    tiles_html = "".join(tiles)
    lat = v.get("latency", {})
    lat_df = pl.DataFrame([{"channel": k, **{kk: vv for kk, vv in d.items() if kk in ("count", "p50", "p90", "p99", "max", "budget_ms", "breaches")}}
                           for k, d in lat.items()], infer_schema_length=None) if lat else pl.DataFrame()
    pos_df = _positions_frame(v.get("positions") or [])
    brk = (v.get("risk") or {}).get("breakers") or {}
    brk_df = (pl.DataFrame([{"breaker": k, "active": bool(d.get("active")), "trips": int(d.get("trips") or 0), "reason": str(d.get("reason") or "")}
                            for k, d in brk.items()]) if brk else pl.DataFrame())
    lat_fmt = {**{k: "num1" for k in ("p50", "p90", "p99", "max", "budget_ms")}, "count": "int", "breaches": "int"}
    lat_hdr = {"budget_ms": "budget", "breaches": "over budget"}
    return (f"{data}{status}<div class='kpis'>{tiles_html}</div>"
            f"<div class='card'><h3>Open positions{f' ({pos_df.height})' if pos_df.height else ''}</h3>"
            f"{positions_html(pos_df)}</div>"
            f"<div class='card'><h3>Circuit breakers</h3>{C.html_table(brk_df, scroll=False)}</div>"
            f"<div class='card'><h3>Latency (ms)</h3>{C.html_table(lat_df, formats=lat_fmt, headers=lat_hdr, scroll=False)}</div>"
            f"<details class='card'><summary>Raw snapshot</summary><pre style='white-space:pre-wrap;font-size:11px;max-height:360px;overflow:auto'>"
            f"{html.escape(json.dumps(v, indent=1, default=str)[:20000])}</pre></details>")


# --------------------------------------------------------------------------- strategy builder
BUILDER_CSS = """
main:has(#bt-root){max-width:1480px} #bt-root [hidden]{display:none!important}
.bt-top{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:center;justify-content:space-between;margin:14px 0 6px}
.bt-top .sub{margin:0}
.bt-health{display:flex;gap:10px;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:8px 14px;min-width:260px} .bt-health b{display:block;font-size:13px} .bt-health small{color:var(--ink2);font-size:12px}
.bt-dot{width:14px;height:14px;border-radius:50%;background:var(--muted);flex:none}
.bt-health[data-level=good] .bt-dot{background:var(--good);animation:bt-pulse 2s infinite}
.bt-health[data-level=warn] .bt-dot{background:#d89b00} .bt-health[data-level=bad] .bt-dot{background:var(--bad)}
@keyframes bt-pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.45)}70%{box-shadow:0 0 0 9px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
.bt-grid{display:grid;grid-template-columns:minmax(0,5fr) 64px minmax(0,6fr);gap:12px;align-items:start}
@media (max-width:980px){.bt-grid{grid-template-columns:minmax(0,1fr)}.bt-flow{display:none}}
@media (max-width:560px){.bt-modes{grid-template-columns:1fr}.bt-dur output{min-width:0}.bt-health{min-width:0}}
.bt-cmd code{min-width:0}
.bt-build{padding:16px} .bt-build h3{margin:16px 0 8px} .bt-build h3:first-child{margin-top:0}
.bt-modes{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.bt-mode{position:relative;display:block;border:1px solid var(--border);border-radius:10px;padding:10px 12px;cursor:pointer;background:var(--page)}
.bt-mode input{position:absolute;opacity:0;pointer-events:none}
.bt-mode .k{font-family:ui-monospace,Consolas,monospace;font-weight:700;font-size:15px} .bt-mode .t{display:block;color:var(--ink2);font-size:12px}
.bt-mode:has(input:checked){border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.bt-mode:has(input:focus-visible){outline:2px solid var(--accent);outline-offset:2px}
.bt-mode.real:has(input:checked){border-color:var(--bad);box-shadow:inset 0 0 0 1px var(--bad)} .bt-mode.real .k{color:var(--bad)}
.bt-desc{color:var(--ink2);font-size:13px;margin:8px 0 0}
.bt-strats{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px}
.bt-chip{display:flex;gap:8px;align-items:flex-start;border:1px solid var(--border);border-radius:8px;padding:7px 9px;cursor:pointer}
.bt-chip:has(input:checked){border-color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,transparent)}
.bt-chip input{margin:3px 0 0} .bt-chip b{display:block;font-size:13px} .bt-chip small{color:var(--ink2);font-size:11.5px;line-height:1.3;display:block}
.bt-dur{display:flex;gap:12px;align-items:center} .bt-dur input{flex:1;accent-color:var(--accent)}
.bt-dur output{min-width:150px;text-align:right;font-weight:620;font-variant-numeric:tabular-nums}
.bt-opts{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:13px}
.bt-opts input[type=number]{width:84px;font:inherit;padding:3px 6px;border:1px solid var(--border);border-radius:6px;background:var(--page);color:var(--ink)}
.bt-cmd{display:flex;gap:8px;align-items:stretch;margin-top:6px}
.bt-cmd code{flex:1;display:block;background:#0c0c0c;color:#e8e8e8;border-radius:8px;padding:10px 12px;font:13px/1.4 ui-monospace,Consolas,monospace;
  overflow-x:auto;white-space:pre} .bt-cmd code::before{content:"PS> ";color:#6fb6ff}
.bt-btn{font:inherit;border:1px solid var(--border);background:var(--surface);color:var(--ink);border-radius:8px;padding:7px 14px;cursor:pointer}
.bt-btn:disabled{opacity:.45;cursor:not-allowed} .bt-btn.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.bt-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px} .bt-msg{font-size:13px;margin:8px 0 0;min-height:1.4em}
.bt-msg.err{color:var(--bad)}
.bt-flow{position:sticky;top:90px;padding-top:110px} .bt-flow svg{display:block;width:64px;height:440px}
.bt-flow path{fill:none;stroke:var(--muted);stroke-width:2;stroke-linecap:round;stroke-dasharray:6 10;opacity:.5;animation:bt-flow 2.6s linear infinite}
.bt-flow polygon{fill:var(--muted);opacity:.5}
.bt-running .bt-flow path{stroke:var(--accent);opacity:1;animation-duration:.7s} .bt-running .bt-flow polygon{fill:var(--accent);opacity:1}
@keyframes bt-flow{to{stroke-dashoffset:-32}}
@media (prefers-reduced-motion:reduce){.bt-flow path,.bt-health .bt-dot{animation:none}}
.bt-side .kpis{grid-template-columns:repeat(auto-fill,minmax(130px,1fr))} .bt-side .kpi .v{font-size:18px}
.bt-term:empty::before{content:"Output of a session started with \\"Run in dashboard\\" streams here.";color:#777}
.bt-termhead{display:flex;justify-content:space-between;gap:10px;background:#1f1f1f;color:#cfcfcf;border-radius:10px 10px 0 0;padding:7px 12px;
  font-size:12.5px;margin-top:10px;border:1px solid #333;border-bottom:0} .bt-termhead b{color:#fff}
.bt-term{background:#0c0c0c;color:#d6d6d6;border:1px solid #333;border-top:0;font:12px/1.45 ui-monospace,Consolas,monospace;height:360px;overflow:auto;margin:0;padding:10px 12px;
  border-radius:0 0 10px 10px;white-space:pre-wrap;word-break:break-word}
"""

BUILDER_JS = r"""
(function(){
  const CFG = JSON.parse(document.getElementById('bt-config').textContent);
  const H = {'Content-Type': 'application/json', 'X-Pumpfun-Dashboard': '1'};
  const $ = id => document.getElementById(id);
  const root = $('bt-root'), slider = $('bt-dur'), term = $('bt-term');
  const pick = {stream: CFG.stream.indexOf(60), replay: CFG.replay.indexOf(6)};
  let lastFeed = null, cmd = '', runnable = false, job = null, next = 0, pending = null;

  const mode = () => document.querySelector('input[name=bt-mode]:checked').value;
  const feed = () => CFG.modes[mode()].feed;
  const stops = () => feed() === 'stream' ? CFG.stream : CFG.replay;
  function durText(v){
    if (feed() === 'stream') return !v ? 'until you stop it' : v < 60 ? v + ' min' : (v / 60) + ' h live';
    return !v ? 'all stored events' : v + ' h of market';
  }
  function spec(){
    return {mode: mode(), duration: stops()[+slider.value],
            strategies: Array.from(document.querySelectorAll('.bt-strats input:checked')).map(x => x.value),
            compare: $('bt-compare').checked, flatten: $('bt-flatten').checked,
            status_every: $('bt-status').value, capital_sol: $('bt-capital').value};
  }
  function msg(text, err){ const m = $('bt-msg'); m.textContent = text || ''; m.className = 'bt-msg' + (err ? ' err' : ' muted'); }
  function buttons(){
    const running = !!(job && job.running);
    $('bt-run').disabled = !runnable || running;
    $('bt-stop').disabled = !running || job.stopping;
    root.classList.toggle('bt-running', running);
  }
  function update(){
    const f = feed();
    if (f !== lastFeed){ slider.max = stops().length - 1; slider.value = pick[f]; lastFeed = f; }
    pick[f] = +slider.value;
    $('bt-dur-v').textContent = durText(stops()[+slider.value]);
    $('bt-dur-l').textContent = f === 'stream' ? '3 · How long to trade the live stream' : '3 · How much stored market to replay';
    $('bt-desc').textContent = CFG.modes[mode()].desc;
    document.querySelectorAll('[data-feed]').forEach(el => { el.hidden = el.dataset.feed !== f; });
    $('bt-capital-w').hidden = mode() !== 'hfts';
    const none = !document.querySelector('.bt-strats input:checked');
    clearTimeout(pending);
    pending = setTimeout(function(){
      fetch('/api/builder/command', {method: 'POST', headers: H, body: JSON.stringify(spec())}).then(r => r.json()).then(function(r){
        if (r.error){ cmd = ''; runnable = false; $('bt-cmd').textContent = '…'; msg(none ? 'Pick at least one strategy.' : r.error, true); }
        else {
          cmd = r.powershell; runnable = r.runnable; $('bt-cmd').textContent = cmd;
          msg(r.runnable ? 'Run it here (output below), or copy it into a PowerShell window that loaded hft.ps1.'
                         : 'Real money: the dashboard never starts live trading. Copy this into PowerShell yourself.', !r.runnable);
        }
        buttons();
      }).catch(() => msg('The dashboard server is not reachable.', true));
    }, 120);
  }
  root.addEventListener('input', update);
  root.addEventListener('change', update);

  $('bt-copy').addEventListener('click', function(){
    if (!cmd) return;
    const done = () => { const b = $('bt-copy'); b.textContent = 'Copied'; setTimeout(() => { b.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(cmd).then(done, () => {});
    else { const r = document.createRange(); r.selectNodeContents($('bt-cmd')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }
  });
  $('bt-run').addEventListener('click', function(){
    fetch('/api/jobs', {method: 'POST', headers: H, body: JSON.stringify(spec())}).then(r => r.json()).then(function(r){
      if (r.error){ msg(r.error, true); return; }
      term.textContent = ''; next = 0; poll();
    });
  });
  $('bt-stop').addEventListener('click', function(){
    fetch('/api/jobs/stop', {method: 'POST', headers: H}).then(() => poll());
  });

  function hhmm(ms){ return new Date(ms).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'}); }
  function poll(){
    fetch('/api/jobs/current?since=' + next).then(r => r.json()).then(function(r){
      if (r.job && job && r.job.id !== job.id){ term.textContent = ''; }
      job = r.job;
      if (r.lines && r.lines.length){
        const atEnd = term.scrollHeight - term.scrollTop - term.clientHeight < 40;
        term.textContent += (term.textContent ? '\n' : '') + r.lines.join('\n');
        if (term.textContent.length > 400000) term.textContent = term.textContent.slice(-300000);
        if (atEnd) term.scrollTop = term.scrollHeight;
      }
      next = r.next || 0;
      $('bt-job').textContent = !job ? 'no session started from this page' :
        job.running ? (job.stopping ? 'stopping ' : 'running ') + job.mode + ' since ' + hhmm(job.started_ms) :
        job.mode + ' ended ' + (job.ended_ms ? hhmm(job.ended_ms) : '') + ' · exit code ' + job.exit_code;
      buttons();
    }).catch(() => {});
  }
  function live(){
    fetch('/api/builder/live/html').then(r => r.text()).then(function(h){
      const box = $('bt-live'), det = box.querySelector('details'), open = det && det.open;
      box.innerHTML = h; if (open){ const d2 = box.querySelector('details'); if (d2) d2.open = true; }
    }).catch(() => {});
  }
  const SESSION = {none: 'no trading session yet', starting: 'session starting', running: 'session running',
                   ended: 'last session ended', stale: 'session not updating'};
  function health(){
    fetch('/api/health').then(r => r.json()).then(function(h){
      const el = $('bt-health'), rpc = h.rpc || {};
      el.dataset.level = h.level;
      const who = rpc.provider || 'RPC';
      const head = {ok: 'Connected to ' + who, slow: who + ' is slow', limited: who + ' is rate limiting',
                    down: who + ' is not reachable', missing: 'No RPC configured'}[rpc.state] || who;
      $('bt-health-t').textContent = head + (rpc.ms != null ? ' · ' + Math.round(rpc.ms) + ' ms' : '');
      $('bt-health-d').textContent = [rpc.detail, rpc.ws === false ? 'SOLANA_WS_URL missing' : '', SESSION[h.session] || '']
        .filter(Boolean).join(' · ');
    }).catch(function(){ $('bt-health').dataset.level = 'bad'; $('bt-health-t').textContent = 'Dashboard server not reachable'; });
  }
  update(); poll(); live(); health();
  setInterval(poll, 1000); setInterval(live, CFG.refresh_ms); setInterval(health, 10000);
})();
"""


def _flow_svg() -> str:
    paths, heads = [], []
    # strategy builder (left) feeding the session (right): curves sweep up into the data panel
    for i, (a, b) in enumerate([(70, 40), (140, 95), (210, 150), (280, 205), (350, 260), (410, 330)]):
        paths.append(f"<path d='M3,{a} C40,{a} 22,{b} 54,{b}' style='animation-delay:-{i * 0.43:.2f}s'/>")
        heads.append(f"<polygon points='53,{b - 5} 62,{b} 53,{b + 5}'/>")
    return (f"<div class='bt-flow' aria-hidden='true'><svg viewBox='0 0 64 440'>"
            f"{''.join(paths)}{''.join(heads)}</svg></div>")


def page_builder(catalogue: list[dict[str, Any]], state: dict[str, Any] | None, refresh_ms: int, server: bool) -> str:
    """Strategy builder: compose a ``trade`` command, run paper sessions, watch their data and raw output."""
    if not server:
        return ("<p class='muted'>The strategy builder starts trading sessions, so it needs the dashboard server: run "
                "<code>hfts dashboard</code> and open http://127.0.0.1:8050/builder.</p>")
    from pumpfun_hft.dashboard.builder import MODES, REPLAY_HOURS, STREAM_MINUTES

    modes = "".join(
        f"<label class='bt-mode{' real' if k == 'hftl' else ''}'><input type='radio' name='bt-mode' value='{k}'"
        f"{' checked' if k == 'hfts' else ''}><span class='k'>{k}</span><span class='t'>{html.escape(m['title'])}</span></label>"
        for k, m in MODES.items())
    default = [s["name"] for s in catalogue if s["active"] and not s["needs_model"]][:1]
    if not default and catalogue:
        default = [catalogue[0]["name"]]
    chips = "".join(
        f"<label class='bt-chip'><input type='checkbox' value='{html.escape(s['name'])}'{' checked' if s['name'] in default else ''}>"
        f"<span><b>{html.escape(s['label'])}{' <span class=pill>tuned for hfts</span>' if s.get('tuned') else ''}</b>"
        f"<small>{html.escape(s['summary'])}"
        f"{' · needs a model from train-model' if s['needs_model'] else ''}</small></span></label>"
        for s in catalogue)
    cfg = json.dumps({"modes": {k: {"feed": m["feed"], "desc": m["desc"]} for k, m in MODES.items()},
                      "stream": list(STREAM_MINUTES), "replay": list(REPLAY_HOURS), "refresh_ms": refresh_ms}).replace("</", "<\\/")
    build = (
        "<section class='card bt-build' aria-label='Build a trading command'>"
        "<h3>1 · Mode</h3><div class='bt-modes' role='radiogroup' aria-label='Mode'>" + modes + "</div>"
        "<p class='bt-desc' id='bt-desc'></p>"
        "<h3>2 · Strategies <span class='muted' style='font-weight:400'>(at least one; rug avoidance always runs as the exit overlay)</span></h3>"
        "<div class='bt-strats'>" + chips + "</div>"
        "<h3 id='bt-dur-l'>3 · Duration</h3><div class='bt-dur'><input type='range' id='bt-dur' min='0' max='15' step='1' "
        "aria-labelledby='bt-dur-l'><output id='bt-dur-v' for='bt-dur'></output></div>"
        "<h3>4 · Options</h3>"
        "<div class='bt-opts' data-feed='stream'><label>status line every <input type='number' id='bt-status' value='10' min='0' "
        "max='3600' step='1'> s</label><label><input type='checkbox' id='bt-flatten' checked> close positions when it stops</label>"
        "<label id='bt-capital-w'>paper capital <input type='number' id='bt-capital' value='' placeholder='10' min='0.1' step='0.5'> SOL</label></div>"
        "<div class='bt-opts' data-feed='replay' hidden><label><input type='checkbox' id='bt-compare' checked> "
        "also backtest the same events and compare</label></div>"
        "<h3>5 · Command</h3><div class='bt-cmd'><code id='bt-cmd'>…</code>"
        "<button type='button' class='bt-btn' id='bt-copy'>Copy</button></div>"
        "<div class='bt-actions'><button type='button' class='bt-btn primary' id='bt-run' disabled>Run in dashboard</button>"
        "<button type='button' class='bt-btn' id='bt-stop' disabled>Stop</button></div>"
        "<p class='bt-msg muted' id='bt-msg' role='status'></p></section>")
    side = ("<section class='bt-side'><div class='card'><h3>Session data: open positions, PnL, latency</h3>"
            f"<div id='bt-live'>{live_body(state, server=True)}</div></div>"
            "<div class='bt-termhead'><b>PowerShell output (raw stream)</b><span id='bt-job'></span></div>"
            "<pre class='bt-term' id='bt-term'></pre></section>")
    top = ("<div class='bt-top'><p class='sub'>Pick a mode, strategies and a duration; the command is built for you. "
           "Built around <b>hfts</b>: paper trading on the live stream.</p>"
           "<div class='bt-health' id='bt-health' data-level='unknown' role='status'><span class='bt-dot'></span>"
           "<div><b id='bt-health-t'>Checking connection…</b><small id='bt-health-d'></small></div></div></div>")
    return (f"<style>{BUILDER_CSS}</style><div id='bt-root'>{top}<div class='bt-grid'>{build}{_flow_svg()}{side}</div></div>"
            f"<script type='application/json' id='bt-config'>{cfg}</script><script>{BUILDER_JS}</script>")
