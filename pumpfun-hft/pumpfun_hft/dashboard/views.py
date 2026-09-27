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

PAGES = [("overview", "Overview"), ("trades", "Trades"), ("equity", "Equity curve"), ("drawdown", "Drawdown"),
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


def page_live(state: dict[str, Any] | None, refresh_ms: int, server: bool) -> str:
    poll = (f"<script>setInterval(function(){{fetch('/api/live').then(r=>r.json()).then(function(s){{"
            f"if(!s||!s.value) return; if(!document.getElementById('lv-eq')){{location.reload();return;}} const v=s.value;"
            f"document.getElementById('lv-eq').innerText=Number(v.equity_sol).toFixed(4);"
            f"document.getElementById('lv-pos').innerText=(v.positions||[]).length;"
            f"document.getElementById('lv-ev').innerText=Number(v.events_processed||0).toLocaleString();"
            f"document.getElementById('lv-q').innerText=(v.pending_orders||0)+' / '+(v.queue||0);"
            f"document.getElementById('lv-json').innerText=JSON.stringify(v, null, 1);"
            f"document.getElementById('lv-age').innerText=((Date.now()-s.updated_ms)/1000).toFixed(1)+' s ago';}}).catch(function(){{}});}}, {refresh_ms});</script>"
            ) if server else ""
    if not state:
        return ('<p class="muted">No live session state found. Start <code>python -m pumpfun_hft.main paper</code> (or <code>live</code>); '
                'this page refreshes from its snapshots.</p>') + poll
    v = state.get("value", {})
    tiles = "".join([_kpi("Equity (SOL)", f"<span id='lv-eq'>{float(v.get('equity_sol', 0)):.4f}</span>"),
                     _kpi("Open positions", f"<span id='lv-pos'>{len(v.get('positions', []))}</span>"),
                     _kpi("Events processed", f"<span id='lv-ev'>{int(v.get('events_processed', 0)):,}</span>"),
                     _kpi("Pending / queued", f"<span id='lv-q'>{int(v.get('pending_orders', 0))} / {int(v.get('queue', 0))}</span>"),
                     _kpi("Session", html.escape(str(v.get("session") or "paper"))),
                     _kpi("Snapshot (UTC)", f"<span id='lv-age'>{C.fmt_value(state.get('updated_ms', 0), 'datetime')}</span>")])
    lat = v.get("latency", {})
    lat_df = pl.DataFrame([{"channel": k, **{kk: vv for kk, vv in d.items() if kk in ("count", "p50", "p90", "p99", "max", "budget_ms", "breaches")}}
                           for k, d in lat.items()], infer_schema_length=None) if lat else pl.DataFrame()
    pos_df = pl.DataFrame(v.get("positions") or [], infer_schema_length=None) if v.get("positions") else pl.DataFrame()
    brk = (v.get("risk") or {}).get("breakers") or {}
    brk_df = (pl.DataFrame([{"breaker": k, "active": bool(d.get("active")), "trips": int(d.get("trips") or 0), "reason": str(d.get("reason") or "")}
                            for k, d in brk.items()]) if brk else pl.DataFrame())
    lat_fmt = {**{k: "num1" for k in ("p50", "p90", "p99", "max", "budget_ms")}, "count": "int", "breaches": "int"}
    lat_hdr = {"budget_ms": "budget", "breaches": "over budget"}
    pos_fmt = {"mint": "addr", "value_sol": "sol", "cost_sol": "sol", "ret": "pct_s", "entry_ms": "datetime", "tokens": "int"}
    return (f"<div class='kpis'>{tiles}</div><div class='card'><h3>Latency (ms)</h3>"
            f"{C.html_table(lat_df, formats=lat_fmt, headers=lat_hdr, scroll=False)}</div>"
            f"<div class='grid2'><div class='card'><h3>Circuit breakers</h3>{C.html_table(brk_df, scroll=False)}</div>"
            f"<div class='card'><h3>Open positions</h3>{C.html_table(pos_df, formats=pos_fmt)}</div></div>"
            f"<div class='card'><h3>Raw snapshot</h3><pre id='lv-json' style='white-space:pre-wrap;font-size:11px;max-height:360px;overflow:auto'>"
            f"{html.escape(json.dumps(v, indent=1, default=str)[:20000])}</pre></div>{poll}")
