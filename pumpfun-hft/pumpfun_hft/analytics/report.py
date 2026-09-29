"""Performance report generator (runs automatically after every backtest).

Contents: headline KPIs, equity curve + drawdown, trade table (sortable), trade-return and
R-multiple distributions, MAE/MFE, monthly returns, hourly and weekday x hour profitability,
win/loss streaks, largest winners and losers, cost breakdown, execution quality (latency,
fill / failure breakdown) and, when supplied, Monte Carlo distributions and path fan.

Exports
    HTML  self-contained single file (Plotly JS inlined once, light + dark theme, sortable tables)
    PDF   ReportLab document with Matplotlib figures and summary tables
    CSV   trades, fills, equity, monthly returns, hourly profile, signals
    JSON  metrics, diagnostics, parameters, Monte Carlo summary, provenance hashes
"""

from __future__ import annotations

import html
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import plotly.offline as po
import polars as pl

from pumpfun_hft.analytics import charts as C
from pumpfun_hft.analytics.metrics import hourly_profile, monthly_returns, reason_category, streaks
from pumpfun_hft.backtester.results import BacktestResult
from pumpfun_hft.utils import jsonutil
from pumpfun_hft.utils.timeutil import ms_to_iso

KPI_SPEC = [
    ("Total return", "total_return", "pct_s"), ("PnL (SOL)", "pnl_sol", "sol"), ("Sharpe", "sharpe", "num2"),
    ("Sortino", "sortino", "num2"), ("Max drawdown", "max_drawdown", "pct"), ("Trades", "n_trades", "int"),
    ("Win rate", "win_rate", "pct"), ("Profit factor", "profit_factor", "num2"), ("Expectancy / trade", "expectancy_sol", "sol"),
    ("Avg R multiple", "avg_r_multiple", "num2"), ("Kelly fraction", "kelly_fraction", "num2"), ("Omega", "omega", "num2"),
    ("Return / max DD", "recovery_factor", "num2"), ("Time under water", "pct_time_underwater", "pct"), ("Fill rate", "fill_rate", "pct"),
    ("Avg latency", "avg_latency_ms", "ms"),
]


def _fmt(v: Any, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    if kind == "pct_s":
        return f"{100 * float(v):+.2f}%"
    if kind == "pct":
        return f"{100 * float(v):.2f}%"
    if kind == "sol":
        return f"{float(v):+.4f}"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "ms":
        return f"{float(v):,.0f} ms"
    return f"{float(v):,.2f}"


PCT_METRICS = {"total_return", "cagr", "cagr_raw", "max_drawdown", "pct_time_underwater", "win_rate", "fill_rate", "failed_rate",
               "dropped_rate", "rejected_rate", "avg_mae", "avg_mfe", "expectancy_ret", "median_trade_ret", "vol_annual"}

TRADE_FORMATS = {"mint": "addr", "creator": "addr", "entry_ms": "datetime", "exit_ms": "datetime", "cost_sol": "sol",
                 "proceeds_sol": "sol", "pnl_sol": "sol_s", "ret": "pct_s", "r_multiple": "num2", "mae": "pct_s", "mfe": "pct_s",
                 "hold_s": "num1", "exit_reason": "text", "fees_sol": "sol", "tx_costs_sol": "sol", "slippage_sol": "sol_s"}
TRADE_HEADERS = {"trade_id": "#", "entry_ms": "entry (UTC, MM-DD)", "exit_ms": "exit (UTC)", "cost_sol": "cost SOL", "pnl_sol": "PnL SOL",
                 "ret": "return", "r_multiple": "R", "mae": "MAE", "mfe": "MFE", "hold_s": "hold s", "exit_reason": "exit reason",
                 "fees_sol": "fees SOL", "tx_costs_sol": "tx costs SOL"}

EXIT_FORMATS = {"win_rate": "pct", "pnl_sol": "sol_s", "avg_pnl_sol": "sol_s", "avg_ret": "pct_s", "avg_hold_s": "num1"}
EXIT_HEADERS = {"exit": "exit reason", "win_rate": "win rate", "pnl_sol": "PnL SOL", "avg_pnl_sol": "avg PnL SOL",
                "avg_ret": "avg return", "avg_hold_s": "avg hold s"}


def metric_kind(key: str) -> str | None:
    """Display kind for a metric key (used by the metrics table)."""
    if key in PCT_METRICS or key.endswith("_rate"):
        return "pct"
    if key.endswith("_sol"):
        return "sol"
    if key.endswith("_ms"):
        return "ms"
    if key.endswith("_bps"):
        return "bps"
    if key.startswith("n_") or key.endswith("_streak"):
        return "int"
    return None


def metrics_frame(m: dict[str, Any]) -> pl.DataFrame:
    """Metrics as a two-column table of already-formatted strings.

    On spans shorter than ``MIN_ANNUALISATION_DAYS`` the raw CAGR extrapolation is left out (it is
    kept in the JSON output): compounding one day to a year produces numbers with 20+ digits.
    """
    rows = []
    for k, v in m.items():
        if isinstance(v, (dict, list)) or (k == "cagr_raw" and m.get("cagr_extrapolated")):
            continue
        rows.append({"metric": k, "value": C.fmt_value(v, metric_kind(k)) if not isinstance(v, bool) else str(v)})
    return pl.DataFrame(rows, schema={"metric": pl.String, "value": pl.String})


def exit_categories(trades: pl.DataFrame) -> pl.DataFrame:
    """Trades grouped by normalised exit reason: count, win rate, total / average PnL, average return."""
    if trades.is_empty():
        return pl.DataFrame()
    cats = [reason_category(r) for r in trades["exit_reason"].to_list()]
    return (trades.with_columns(pl.Series("exit", cats))
            .group_by("exit").agg(pl.len().alias("trades"), (pl.col("pnl_sol") > 0).mean().alias("win_rate"),
                                  pl.col("pnl_sol").sum().alias("pnl_sol"), pl.col("pnl_sol").mean().alias("avg_pnl_sol"),
                                  pl.col("ret").mean().alias("avg_ret"), pl.col("hold_s").mean().alias("avg_hold_s"))
            .sort("pnl_sol", descending=True))


class ReportGenerator:
    """Build HTML / PDF / CSV / JSON reports for a :class:`BacktestResult`.

    Example::

        paths = ReportGenerator(result, mc=mc_result).generate("pumpfun_hft/reports/run-123", ["html", "pdf", "csv", "json"])
    """

    def __init__(self, result: BacktestResult, mc: Any = None, title: str | None = None) -> None:
        self.r = result
        self.mc = mc
        self.title = title or f"Backtest report — {', '.join(result.strategies)}"

    # ------------------------------------------------------------------ data helpers
    def _trades(self) -> pl.DataFrame:
        return self.r.trades if self.r.trades.height else pl.DataFrame()

    def largest(self, n: int = 10) -> tuple[pl.DataFrame, pl.DataFrame]:
        t = self._trades()
        if t.is_empty():
            return t, t
        cols = ["trade_id", "mint", "strategy", "pnl_sol", "ret", "r_multiple", "hold_s", "exit_reason"]
        return t.sort("pnl_sol", descending=True).head(n).select(cols), t.sort("pnl_sol").head(n).select(cols)

    def cost_breakdown(self) -> dict[str, float]:
        """Explicit costs paid (SOL), including fees of failed transactions."""
        f = self.r.fills
        if f.is_empty():
            return {}
        s = lambda c: float(f[c].sum()) / 1e9 if c in f.columns else 0.0  # noqa: E731
        return {"Protocol fees": s("protocol_fee"), "Creator fees": s("creator_fee"), "LP fees": s("lp_fee"),
                "Platform fees": s("platform_fee"), "Priority fees": s("priority_fee"), "Base network fees": s("network_fee"),
                "Jito tips": s("jito_tip")}

    def execution_quality(self) -> pl.DataFrame:
        """Slippage vs the decision-time quote and latency, by side (positive slippage = adverse)."""
        f = self.r.fills
        if f.is_empty() or "slippage_bps" not in f.columns:
            return pl.DataFrame()
        filled = f.filter(pl.col("status").is_in(["filled", "partial"]))
        if filled.is_empty():
            return pl.DataFrame()
        notional = pl.col("price") * pl.col("token_amount") / 1e6
        return (filled.group_by("side")
                .agg(pl.len().alias("fills"), pl.col("slippage_bps").mean().alias("avg_slip_bps"),
                     pl.col("slippage_bps").median().alias("median_slip_bps"),
                     pl.col("slippage_bps").quantile(0.9).alias("p90_slip_bps"),
                     (pl.col("slippage_bps") / 1e4 * notional).sum().alias("slippage_sol"),
                     pl.col("latency_ms").median().alias("p50_latency_ms"), pl.col("latency_ms").quantile(0.9).alias("p90_latency_ms"),
                     pl.col("latency_ms").quantile(0.99).alias("p99_latency_ms"))
                .sort("side"))

    def figures(self) -> dict[str, Any]:
        r, t = self.r, self._trades()
        figs: dict[str, Any] = {"equity": C.equity_drawdown(r.equity, r.initial_capital_sol)}
        if t.height:
            figs["returns"] = C.histogram(t["ret"].to_numpy() * 100, "trade return (%)")
            figs["r_mult"] = C.histogram(t["r_multiple"].to_numpy(), "R multiple")
            figs["mae_mfe"] = C.mae_mfe(t)
            hp = hourly_profile(t)
            by_hour = hp.group_by("hour").agg(pl.col("pnl_sol").sum()).sort("hour")
            figs["hourly"] = C.signed_bars(by_hour["hour"].to_list(), by_hour["pnl_sol"].to_numpy(), "UTC hour of entry", "PnL (SOL)")
            z = np.full((7, 24), np.nan)
            for wd, h, p in hp.select("weekday", "hour", "pnl_sol").iter_rows():
                z[int(wd) - 1, int(h)] = p
            figs["heat"] = C.diverging_heatmap(z, list(range(24)), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "PnL (SOL)", ".4f")
            figs["heat"].update_xaxes(title_text="UTC hour", dtick=3)
            signs = list(t.sort("exit_ms")["pnl_sol"].to_numpy() > 0)
            runs: dict[str, dict[int, int]] = {"win": {}, "loss": {}}
            cur, n = None, 0
            for s in signs + [None]:
                if s == cur:
                    n += 1
                    continue
                if cur is not None:
                    k = "win" if cur else "loss"
                    runs[k][n] = runs[k].get(n, 0) + 1
                cur, n = s, 1
            lengths = sorted(set(runs["win"]) | set(runs["loss"]))
            import plotly.graph_objects as go

            sf = go.Figure([go.Bar(x=lengths, y=[runs["win"].get(k, 0) for k in lengths], name="Winning streaks",
                                   marker={"color": C.POS, "cornerradius": 4},
                                   hovertemplate="%{y} winning streaks of %{x}<extra></extra>"),
                            go.Bar(x=lengths, y=[runs["loss"].get(k, 0) for k in lengths], name="Losing streaks",
                                   marker={"color": C.NEG, "cornerradius": 4},
                                   hovertemplate="%{y} losing streaks of %{x}<extra></extra>")])
            sf.update_xaxes(title_text="streak length (trades)")
            sf.update_yaxes(title_text="occurrences")
            sf.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
            figs["streaks"] = C.base_layout(sf, 300, showlegend=True)
            ex = exit_categories(t).head(14).sort("pnl_sol")
            figs["exit_reasons"] = C.hbars([C.clip_label(str(x), 40) for x in ex["exit"].to_list()], ex["pnl_sol"].to_list(), "PnL (SOL)", signed=True,
                                           hover_extra=[f"<br>{n} trades · win rate {100 * w:.0f}%" for n, w in
                                                        zip(ex["trades"].to_list(), ex["win_rate"].to_list(), strict=True)])
        costs = self.cost_breakdown()
        if costs:
            nz = {k: v for k, v in costs.items() if v > 0} or costs
            figs["costs"] = C.hbars(list(nz), list(nz.values()), "SOL")
        f = r.fills
        if f.height:
            filled = f.filter(pl.col("status").is_in(["filled", "partial"]))
            if filled.height:
                figs["latency"] = C.histogram(filled["latency_ms"].to_numpy(), "decision → landing latency (ms)", 50, vline=None)
            st = f.group_by("status").len().sort("len")
            figs["status"] = C.hbars(st["status"].to_list(), [float(x) for x in st["len"].to_list()], "order attempts", fmt=",.0f")
        if self.mc is not None and getattr(self.mc, "distributions", None):
            figs["mc_ret"] = C.histogram(self.mc.distributions["total_return"] * 100, "simulated total return (%)", 60)
            figs["mc_dd"] = C.histogram(self.mc.distributions["max_drawdown"] * 100, "simulated max drawdown (%)", 60, vline=None)
            if self.mc.paths is not None:
                figs["mc_fan"] = C.fan_chart(self.mc.paths, r.initial_capital_sol)
        return figs

    # ------------------------------------------------------------------ HTML
    def html(self, path: str | Path) -> Path:
        r, m = self.r, self.r.metrics
        figs = self.figures()
        kpis = []
        for label, key, kind in KPI_SPEC:
            v = m.get(key)
            cls = ""
            if key in ("total_return", "pnl_sol", "expectancy_sol") and isinstance(v, (int, float)) and not math.isnan(float(v)):
                cls = "pos" if float(v) > 0 else "neg" if float(v) < 0 else ""
            kpis.append(f'<div class="kpi"><div class="l">{label}</div><div class="v {cls}">{_fmt(v, kind)}</div></div>')
        banner = ""
        if r.synthetic:
            banner = ('<div class="banner"><b>Synthetic data.</b> These results come from the built-in synthetic market '
                      'generator, which plants known structure (informed wallets, serial ruggers) to exercise the pipeline. '
                      'They say nothing about live profitability.</div>')
        if not r.trades.height:
            outcomes = ""
            sig = r.signals
            if sig.height and {"action", "outcome", "strategy"} <= set(sig.columns):
                c = sig.filter(pl.col("action") == "BUY").group_by("strategy", "outcome").len().sort("len", descending=True)
                outcomes = " Buy signals and what happened to them: " + html.escape(", ".join(
                    f"{x['strategy']} {x['outcome']} {x['len']:,}" for x in c.iter_rows(named=True))) + "." if c.height else ""
            banner += ('<div class="banner"><b>No trades in this run</b>, so the metrics and charts below are empty (0 or '
                       f'NaN).{outcomes or " No strategy produced a buy signal."} See signals.parquet in the run folder.</div>')
        if m.get("cagr_extrapolated"):
            banner += ('<div class="banner">The backtest spans less than 30 days: CAGR and Calmar are not reported, and the '
                       'annualised Sharpe and Sortino are scaled up from a short sample, so read them as relative scores only.</div>')

        def sec(title: str, key: str, note: str = "") -> str:
            if key not in figs:
                return ""
            n = f'<p class="muted">{note}</p>' if note else ""
            return f'<div class="card"><h3>{title}</h3>{n}{C.fig_html(figs[key], key)}</div>'

        t = self._trades()
        win, lose = self.largest()
        big_fmt = {**TRADE_FORMATS}
        mon = monthly_returns(r.equity)
        if mon.height:
            mon = mon.select(pl.format("{}-{}", pl.col("year"), pl.col("month").cast(pl.String).str.zfill(2)).alias("month"),
                             pl.col("ret").alias("return"))
        trade_cols = [c for c in ["trade_id", "mint", "strategy", "entry_ms", "exit_ms", "cost_sol", "pnl_sol", "ret", "r_multiple",
                                  "mae", "mfe", "hold_s", "exit_reason", "fees_sol", "tx_costs_sol"] if c in t.columns]
        sw, sl = streaks(list(t.sort("exit_ms")["pnl_sol"].to_numpy() > 0)) if t.height else (0, 0)
        ex = exit_categories(t)
        eq_tbl = self.execution_quality()
        diag = r.diagnostics or {}
        sim_counts = pl.DataFrame([{"outcome": k, "count": int(v)} for k, v in sorted((diag.get("sim_counts") or {}).items(),
                                                                                      key=lambda kv: -kv[1])],
                                  schema={"outcome": pl.String, "count": pl.Int64})
        risk = diag.get("risk") or {}
        brk = pl.DataFrame([{"breaker": k, "trips": int(v.get("trips", 0)), "active at end": bool(v.get("active"))}
                            for k, v in (risk.get("breakers") or {}).items()],
                           schema={"breaker": pl.String, "trips": pl.Int64, "active at end": pl.Boolean})
        rej = pl.DataFrame([{"risk rejection": k, "count": int(v)} for k, v in (risk.get("rejections") or {}).items()],
                           schema={"risk rejection": pl.String, "count": pl.Int64})
        mc_block = ""
        if self.mc is not None:
            ms = self.mc.summary()
            qrows = []
            kinds = {"total_return": "pct_s", "cagr": "pct_s", "max_drawdown": "pct", "longest_dd_trades": "int"}
            for k, v in (ms["quantiles"] or {}).items():
                vals = [v.get(q) for q in ("q05", "q25", "q50", "q75", "q95")]
                if all(x is None or (isinstance(x, float) and not math.isfinite(x)) for x in vals):
                    continue  # e.g. CAGR on a span shorter than 30 days
                qrows.append({"metric": k.replace("_", " "), **{q: C.fmt_value(x, kinds.get(k)) for q, x in
                                                               zip(("5 %", "25 %", "median", "75 %", "95 %"), vals, strict=True)}})
            qdf = pl.DataFrame(qrows) if qrows else pl.DataFrame()
            mc_block = (f'<h2>Monte Carlo ({ms["n_sims"]:,} simulations, {html.escape(str(ms["method"]))})</h2>'
                        '<p class="muted">Closed trades are resampled and each one\'s slippage, fees, size and latency are perturbed '
                        '(log-normal, adverse on average), so this is a stress test around the realised trades rather than a forecast.</p>'
                        f'<div class="kpis"><div class="kpi"><div class="l">Probability of ruin</div><div class="v">{_fmt(ms["prob_ruin"], "pct")}</div></div>'
                        f'<div class="kpi"><div class="l">Probability of loss</div><div class="v">{_fmt(ms["prob_loss"], "pct")}</div></div>'
                        f'<div class="kpi"><div class="l">Worst case return</div><div class="v">{_fmt(ms["worst"].get("total_return"), "pct_s")}</div></div>'
                        f'<div class="kpi"><div class="l">Best case return</div><div class="v">{_fmt(ms["best"].get("total_return"), "pct_s")}</div></div></div>'
                        f'<div class="grid2">{sec("Total return distribution", "mc_ret")}{sec("Max drawdown distribution", "mc_dd")}</div>'
                        f'{sec("Equity path fan (resampled trades)", "mc_fan")}'
                        f'<div class="card"><h3>Quantiles</h3>{C.html_table(qdf, scroll=False)}</div>')
        prov = (f"run <code>{html.escape(r.run_id)}</code> · config <code>{html.escape(str(r.config_hash))}</code> · data "
                f"<code>{html.escape(str(r.data_hash))}</code> · seed {r.seed} · {r.n_events:,} events in {r.elapsed_s:.1f}s "
                f"({r.events_per_second:,.0f} events/s) · {ms_to_iso(r.start_ms)} → {ms_to_iso(r.end_ms)}")
        eq_fmt = {"avg_slip_bps": "bps", "median_slip_bps": "bps", "p90_slip_bps": "bps", "slippage_sol": "sol_s",
                  "p50_latency_ms": "ms", "p90_latency_ms": "ms", "p99_latency_ms": "ms"}
        eq_hdr = {"avg_slip_bps": "avg slippage", "median_slip_bps": "median slippage", "p90_slip_bps": "p90 slippage",
                  "slippage_sol": "slippage SOL", "p50_latency_ms": "p50 latency", "p90_latency_ms": "p90 latency",
                  "p99_latency_ms": "p99 latency"}
        body = f"""
<main>
<h1>{html.escape(self.title)}</h1>
<p class="sub">Initial capital {r.initial_capital_sol:g} SOL · strategies: {html.escape(', '.join(r.strategies))} ·
{r.metrics.get('span_days', 0):.2f} days simulated · {r.metrics.get('n_trades', 0):,} round trips</p>
{banner}
<div class="kpis">{''.join(kpis)}</div>
<h2>Equity and drawdown</h2>{sec("Equity (SOL, marked at liquidation value) and drawdown from peak", "equity")}
<h2>Trade distribution</h2>
<div class="grid2">{sec("Trade returns", "returns")}{sec("R multiples (PnL / initial risk)", "r_mult")}</div>
<div class="grid2">{sec("MAE vs MFE per trade", "mae_mfe")}{sec("Win / loss streaks", "streaks", f"Longest winning streak {sw}, losing streak {sl}.")}</div>
<h2>Timing</h2>
<div class="grid2">{sec("PnL by hour of entry (UTC)", "hourly")}{sec("PnL heatmap: weekday × hour (UTC)", "heat")}</div>
<div class="card"><h3>Monthly returns</h3>{C.html_table(mon, formats={"return": "pct_s"}, scroll=False)}</div>
<h2>Exits</h2>
{sec("PnL by exit reason", "exit_reasons")}
<div class="card"><h3>Exit reasons</h3>{C.html_table(ex, formats=EXIT_FORMATS, headers=EXIT_HEADERS, scroll=False)}</div>
<h2>Largest winners and losers</h2>
<div class="card"><h3>Largest winners</h3>{C.html_table(win, formats=big_fmt, headers=TRADE_HEADERS, scroll=False)}</div>
<div class="card"><h3>Largest losers</h3>{C.html_table(lose, formats=big_fmt, headers=TRADE_HEADERS, scroll=False)}</div>
<h2>Costs and execution quality</h2>
<div class="grid2">{sec("Explicit costs paid (incl. failed transactions)", "costs")}{sec("Order outcomes", "status")}</div>
<div class="card"><h3>Slippage vs decision-time quote and latency, by side</h3>
<p class="muted">Positive slippage is adverse (paid more on buys, received less on sells). Latency runs from the decision to the landed slot.</p>
{C.html_table(eq_tbl, formats=eq_fmt, headers=eq_hdr, scroll=False)}</div>
<div class="grid2">{sec("Latency", "latency")}<div class="card"><h3>Simulator outcomes</h3>{C.html_table(sim_counts, scroll=False)}</div></div>
<div class="grid2"><div class="card"><h3>Circuit breakers</h3>{C.html_table(brk, scroll=False)}</div>
<div class="card"><h3>Risk rejections</h3>{C.html_table(rej, scroll=False) if rej.height else '<p class="muted">No orders were rejected by the risk engine.</p>'}</div></div>
{mc_block}
<h2>All trades</h2><div class="card">{C.html_table(t.select(trade_cols) if t.height else t, 2000, formats=TRADE_FORMATS, headers=TRADE_HEADERS, full_data="csv/trades.csv")}</div>
<h2>All metrics</h2><div class="card">{C.html_table(metrics_frame(m), 200)}</div>
<footer>{prov}</footer>
</main>"""
        doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f"<title>Backtest Report</title><style>{C.PAGE_CSS}</style><script>{po.get_plotlyjs()}</script></head><body>{body}"
               f"<script>{C.TABLE_JS}{C.THEME_JS}</script></body></html>")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
        return p

    # ------------------------------------------------------------------ CSV / JSON
    def csv(self, directory: str | Path) -> list[Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for name, df in (("trades", self.r.trades), ("fills", self.r.fills), ("equity", self.r.equity),
                         ("monthly_returns", monthly_returns(self.r.equity)), ("hourly_profile", hourly_profile(self.r.trades)),
                         ("signals", self.r.signals)):
            if df is not None and df.height:
                p = d / f"{name}.csv"
                df.write_csv(p)
                out.append(p)
        return out

    def json(self, path: str | Path) -> Path:
        r = self.r
        payload = {"run_id": r.run_id, "strategies": r.strategies, "params": r.params, "config_hash": r.config_hash,
                   "data_hash": r.data_hash, "seed": r.seed, "synthetic": r.synthetic, "start": ms_to_iso(r.start_ms),
                   "end": ms_to_iso(r.end_ms), "metrics": r.metrics, "diagnostics": r.diagnostics,
                   "monte_carlo": self.mc.summary() if self.mc is not None else None}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(jsonutil.dumps(payload))  # strict JSON: non-finite metrics are written as null
        return p

    # ------------------------------------------------------------------ PDF
    def pdf(self, path: str | Path) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        r, m, t = self.r, self.r.metrics, self._trades()
        styles = getSampleStyleSheet()
        story: list[Any] = [Paragraph(html.escape(self.title), styles["Title"]),
                            Paragraph(f"Run {html.escape(r.run_id)} · {ms_to_iso(r.start_ms)} → {ms_to_iso(r.end_ms)} · "
                                      f"initial capital {r.initial_capital_sol:g} SOL", styles["Normal"])]
        if r.synthetic:
            story.append(Paragraph("<b>Synthetic data</b> — results exercise the pipeline and say nothing about live profitability.",
                                   styles["Normal"]))
        story.append(Spacer(1, 0.4 * cm))
        rows = [["Metric", "Value", "Metric", "Value"]]
        half = (len(KPI_SPEC) + 1) // 2
        for i in range(half):
            a = KPI_SPEC[i]
            b = KPI_SPEC[i + half] if i + half < len(KPI_SPEC) else None
            rows.append([a[0], _fmt(m.get(a[1]), a[2]), b[0] if b else "", _fmt(m.get(b[1]), b[2]) if b else ""])
        tbl = Table(rows, hAlign="LEFT")
        tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold"), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                                 ("FONTSIZE", (0, 0), (-1, -1), 9), ("ALIGN", (1, 0), (1, -1), "RIGHT"), ("ALIGN", (3, 0), (3, -1), "RIGHT")]))
        story += [tbl, Spacer(1, 0.4 * cm)]

        def add_plot(fn: Any, title: str) -> None:
            fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=150)
            fn(ax)
            ax.set_title(title, fontsize=10, loc="left")
            ax.grid(True, color="#e1e0d9", linewidth=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            story.append(Image(buf, width=17 * cm, height=6.6 * cm))

        if r.equity.height:
            ts = np.array(r.equity["ts_ms"].to_list(), dtype="datetime64[ms]")
            eq = r.equity["equity_sol"].to_numpy()
            add_plot(lambda ax: ax.plot(ts, eq, color=C.SERIES[0], lw=1.6), "Equity (SOL)")
            peak = np.maximum.accumulate(np.concatenate([[r.initial_capital_sol], eq]))[1:]
            dd = (eq / peak - 1) * 100
            add_plot(lambda ax: ax.fill_between(ts, dd, 0, color=C.SERIES[7], alpha=0.25, lw=0), "Drawdown (%)")
        if t.height:
            add_plot(lambda ax: ax.hist(t["ret"].to_numpy() * 100, bins=40, color=C.SERIES[0]), "Trade returns (%)")
            hp = hourly_profile(t).group_by("hour").agg(pl.col("pnl_sol").sum()).sort("hour")
            add_plot(lambda ax: ax.bar(hp["hour"].to_list(), hp["pnl_sol"].to_list(),
                                       color=[C.SERIES[0] if v >= 0 else C.SERIES[7] for v in hp["pnl_sol"].to_list()]),
                     "PnL by UTC hour of entry (SOL)")
            story.append(PageBreak())
            win, lose = self.largest(8)
            for title, df in (("Largest winners", win), ("Largest losers", lose)):
                story.append(Paragraph(title, styles["Heading3"]))
                data = [["mint", "strategy", "PnL SOL", "return", "R", "hold s", "exit"]]
                for row in df.iter_rows(named=True):
                    data.append([row["mint"][:10] + "…", row["strategy"], f"{row['pnl_sol']:+.4f}", f"{100 * row['ret']:+.1f}%",
                                 f"{row['r_multiple']:+.2f}", f"{row['hold_s']:.0f}", C.clip_label(str(row["exit_reason"]), 30)])
                tt = Table(data, hAlign="LEFT")
                tt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey)]))
                story += [tt, Spacer(1, 0.3 * cm)]
        if self.mc is not None and getattr(self.mc, "distributions", None):
            story.append(Paragraph(f"Monte Carlo — probability of ruin {100 * self.mc.prob_ruin:.2f}%, probability of loss "
                                   f"{100 * self.mc.prob_loss:.2f}%", styles["Heading3"]))
            add_plot(lambda ax: ax.hist(self.mc.distributions["total_return"] * 100, bins=60, color=C.SERIES[0]),
                     "Simulated total return (%)")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(str(p), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                          title=self.title).build(story)
        return p

    def generate(self, out_dir: str | Path, formats: list[str] | tuple[str, ...] = ("html", "pdf", "csv", "json")) -> dict[str, Any]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = {}
        if "html" in formats:
            out["html"] = self.html(d / "report.html")
        if "pdf" in formats:
            out["pdf"] = self.pdf(d / "report.pdf")
        if "csv" in formats:
            out["csv"] = self.csv(d / "csv")
        if "json" in formats:
            out["json"] = self.json(d / "report.json")
        return out
