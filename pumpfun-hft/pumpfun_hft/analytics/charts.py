"""Chart theme and figure builders shared by the report generator and the dashboard.

Design rules (validated palette, see docs/DASHBOARD.md):
    * categorical hues in fixed order (blue, orange, aqua, yellow, ...), never cycled
    * sequential magnitude = one hue light -> dark; polarity (PnL) = diverging blue <-> red around
      a neutral gray midpoint; status colours only where the colour *means* good / bad
    * one y-axis per plot (equity and drawdown are separate stacked panels, never dual-axis)
    * 2 px lines, bars <= 24 px with rounded ends, hairline solid grids, legend for >= 2 series
    * every chart has a hover layer; tables accompany charts so no value is colour- or hover-only
    * light and dark themes: figures are built for light and re-themed client-side (see ``THEME_JS``)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
         "grid": "#e1e0d9", "axis": "#c3c2b7"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#b3261e"], [0.25, "#e34948"], [0.5, "#f0efec"], [0.75, "#3987e5"], [1.0, "#184f95"]]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
POS, NEG = SERIES[0], SERIES[7]  # polarity (gain / loss) = the diverging poles, never status green / red
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(fig: go.Figure, height: int = 320, showlegend: bool | None = None) -> go.Figure:
    t = LIGHT
    fig.update_layout(
        height=height, margin={"l": 56, "r": 16, "t": 16, "b": 44}, paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font={"family": FONT, "color": t["ink2"], "size": 12}, hoverlabel={"font": {"family": FONT}},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"color": t["ink2"]}},
        hovermode="closest",
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], zeroline=False, tickfont={"color": t["muted"]})
    fig.update_yaxes(showgrid=True, gridcolor=t["grid"], gridwidth=1, linecolor=t["axis"], zeroline=False, tickfont={"color": t["muted"]})
    return fig


def _dt(ms: Any) -> Any:
    return pl.Series(ms).cast(pl.Int64).cast(pl.Datetime("ms")).to_list()


def equity_drawdown(equity: pl.DataFrame, initial: float) -> go.Figure:
    """Equity (top) and drawdown (bottom) as two stacked panels sharing time (no dual axis)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    if equity.height:
        x = _dt(equity["ts_ms"])
        eq = equity["equity_sol"].to_numpy()
        peak = np.maximum.accumulate(np.concatenate([[initial], eq]))[1:]
        dd = np.where(peak > 0, eq / peak - 1.0, 0.0) * 100
        fig.add_trace(go.Scatter(x=x, y=eq, mode="lines", line={"width": 2, "color": SERIES[0]}, name="Equity",
                                 hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.4f} SOL<extra></extra>"), row=1, col=1)
        fig.add_hline(y=initial, line={"width": 1, "color": LIGHT["axis"]}, row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", line={"width": 2, "color": SERIES[7]}, fill="tozeroy",
                                 fillcolor="rgba(227,73,72,0.10)", name="Drawdown",
                                 hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="SOL", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    return base_layout(fig, 420, showlegend=False)


def histogram(values: np.ndarray, title_x: str, bins: int = 40, color: str = SERIES[0], vline: float | None = 0.0) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=bins, marker={"color": color, "line": {"color": LIGHT["surface"], "width": 2}},
                                 hovertemplate=f"{title_x} %{{x}}<br>%{{y}} trades<extra></extra>"))
    if vline is not None:
        fig.add_vline(x=vline, line={"width": 1, "color": LIGHT["ink2"]})
    fig.update_xaxes(title_text=title_x)
    fig.update_yaxes(title_text="count")
    fig.update_layout(bargap=0.02)
    return base_layout(fig, 300, showlegend=False)


def mae_mfe(trades: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.height:
        for label, cond, color in (("Winners", trades["pnl_sol"] > 0, POS), ("Losers", trades["pnl_sol"] <= 0, NEG)):
            t = trades.filter(cond)
            fig.add_trace(go.Scatter(x=t["mae"] * 100, y=t["mfe"] * 100, mode="markers", name=label,
                                     marker={"size": 8, "color": color, "line": {"color": LIGHT["surface"], "width": 2}},
                                     customdata=np.stack([t["mint"].to_numpy(), t["ret"].to_numpy() * 100], axis=1),
                                     hovertemplate="MAE %{x:.1f}% · MFE %{y:.1f}%<br>return %{customdata[1]:.1f}%<br>%{customdata[0]}<extra></extra>"))
    fig.update_xaxes(title_text="MAE (worst unrealised return, %)")
    fig.update_yaxes(title_text="MFE (best unrealised return, %)")
    return base_layout(fig, 340, showlegend=True)


def diverging_heatmap(z: np.ndarray, x: list[Any], y: list[Any], title: str, fmt: str = ".3f", y_reversed: bool = True) -> go.Figure:
    finite = z[np.isfinite(z)]
    m = float(np.max(np.abs(finite))) if finite.size else 1.0
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=DIVERGING, zmin=-m, zmax=m, xgap=2, ygap=2,
                               colorbar={"title": {"text": title}, "thickness": 12},
                               hovertemplate=f"%{{y}} · %{{x}}<br>{title} %{{z:{fmt}}}<extra></extra>"))
    fig = base_layout(fig, 300, showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed" if y_reversed else True)
    return fig


def signed_bars(x: list[Any], y: np.ndarray, title_x: str, title_y: str) -> go.Figure:
    colors = [POS if v >= 0 else NEG for v in y]
    fig = go.Figure(go.Bar(x=x, y=y, marker={"color": colors, "cornerradius": 4, "line": {"color": LIGHT["surface"], "width": 2}},
                           hovertemplate=f"{title_x} %{{x}}<br>{title_y} %{{y:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text=title_x)
    fig.update_yaxes(title_text=title_y)
    fig.add_hline(y=0, line={"width": 1, "color": LIGHT["axis"]})
    fig.update_layout(bargap=0.35)
    return base_layout(fig, 300, showlegend=False)


def clip_label(text: str, n: int) -> str:
    """``text`` shortened to ``n`` characters with an ellipsis (category labels on chart axes)."""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def hbars(labels: list[str], values: list[float], title_x: str, color: str | list[str] = SERIES[0], fmt: str = ".4f",
          signed: bool = False, hover_extra: list[str] | None = None) -> go.Figure:
    """Horizontal bars with direct value labels. ``signed=True`` colours gains / losses with the diverging poles."""
    colors: str | list[str] = [POS if v >= 0 else NEG for v in values] if signed else color
    custom = hover_extra or [""] * len(values)
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker={"color": colors, "cornerradius": 4},
                           text=[f"{v:{fmt}}" for v in values], textposition="outside", cliponaxis=False, customdata=custom,
                           hovertemplate=f"%{{y}}<br>{title_x} %{{x:{fmt}}}%{{customdata}}<extra></extra>"))
    fig.update_xaxes(title_text=title_x)
    if values:  # leave room for the outside value labels so they never run into the category labels
        lo, hi = min(0.0, min(values)), max(0.0, max(values))
        pad = 0.28 * ((hi - lo) or abs(hi) or 1.0)
        fig.update_xaxes(range=[lo - (pad if lo < 0 else 0.0), hi + (pad if hi > 0 else 0.0)])
    if signed:
        fig.add_vline(x=0, line={"width": 1, "color": LIGHT["axis"]})
    fig.update_layout(bargap=0.45)
    fig = base_layout(fig, max(220, 34 * len(labels) + 80), showlegend=False)
    fig.update_layout(margin={"l": 170, "r": 72, "t": 16, "b": 44})
    fig.update_yaxes(automargin=True)  # long category labels widen the margin instead of being cut off
    return fig


def fan_chart(paths: np.ndarray, initial: float) -> go.Figure:
    """Monte Carlo equity-path fan: 5-95 % and 25-75 % bands plus the median (sequential blue)."""
    fig = go.Figure()
    if paths is not None and paths.size:
        x = np.arange(1, paths.shape[1] + 1)
        q = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        fig.add_trace(go.Scatter(x=x, y=q[4], line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=q[0], line={"width": 0}, fill="tonexty", fillcolor="rgba(57,135,229,0.12)", name="5-95 %",
                                 hovertemplate="trade %{x}<br>5 %: %{y:.3f} SOL<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=q[3], line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=q[1], line={"width": 0}, fill="tonexty", fillcolor="rgba(57,135,229,0.25)", name="25-75 %",
                                 hovertemplate="trade %{x}<br>25 %: %{y:.3f} SOL<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=q[2], line={"width": 2, "color": SERIES[0]}, name="Median",
                                 hovertemplate="trade %{x}<br>median %{y:.3f} SOL<extra></extra>"))
        fig.add_hline(y=initial, line={"width": 1, "color": LIGHT["axis"]})
    fig.update_xaxes(title_text="trade #")
    fig.update_yaxes(title_text="equity (SOL)")
    return base_layout(fig, 320, showlegend=True)


# Fixed colour per strategy (colour follows the entity, never its rank in a list). Eight hues for the
# eight directional strategies; anything else (combinations, migration, custom) is drawn in the
# neutral "other" colour and told apart by the legend and line dash.
STRATEGY_ORDER = ["momentum_ignition", "smart_money", "sniper", "volume_breakout", "mean_reversion", "whale_follow",
                  "bonding_curve_scalp", "liquidity_sweep"]
OTHER = "#898781"


def strategy_color(name: str) -> str:
    return SERIES[STRATEGY_ORDER.index(name)] if name in STRATEGY_ORDER else OTHER


def multi_line(series: dict[str, tuple[Any, Any]], y_title: str, colors: dict[str, str] | None = None) -> go.Figure:
    """Up to 8 named series. ``colors`` maps series name -> colour (entity-stable); default is the
    fixed categorical order, which is only appropriate when the series set itself is fixed."""
    fig = go.Figure()
    seen: dict[str, int] = {}
    for i, (name, (x, y)) in enumerate(list(series.items())[:8]):
        color = (colors or {}).get(name, SERIES[i])
        k = seen.get(color, 0)
        seen[color] = k + 1
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                 line={"width": 2, "color": color, "dash": ["solid", "dash", "dot", "dashdot"][k % 4]},
                                 hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>"))
    fig.update_yaxes(title_text=y_title)
    return base_layout(fig, 360, showlegend=len(series) > 1)


def price_with_fills(prices: pl.DataFrame, fills: pl.DataFrame) -> go.Figure:
    """Token price path with our buys / sells overlaid (two identities -> legend)."""
    fig = go.Figure()
    if prices.height:
        fig.add_trace(go.Scatter(x=_dt(prices["ts_ms"]), y=prices["price"], mode="lines", name="Price",
                                 line={"width": 2, "color": LIGHT["muted"]}, hovertemplate="%{x|%H:%M:%S}<br>%{y:.3e} SOL<extra></extra>"))
    for side, color, symbol in (("buy", SERIES[0], "triangle-up"), ("sell", SERIES[1], "triangle-down")):
        f = fills.filter((pl.col("side") == side) & pl.col("status").is_in(["filled", "partial"])) if fills.height else fills
        if f.height:
            fig.add_trace(go.Scatter(x=_dt(f["land_ms"]), y=f["price"], mode="markers", name=f"Our {side}s",
                                     marker={"size": 11, "color": color, "symbol": symbol, "line": {"color": LIGHT["surface"], "width": 2}},
                                     hovertemplate=f"{side} %{{x|%H:%M:%S}}<br>%{{y:.3e}} SOL<extra></extra>"))
    fig.update_yaxes(title_text="SOL per token", type="log")
    vals = [v for df, c in ((prices, "price"), (fills, "price")) if df.height and c in df.columns
            for v in (df[c].min(), df[c].max()) if v is not None and v > 0]
    if vals:  # log axis labelled at 1-2-5 steps in plain scientific notation (Plotly's default mixes "100n" and bare digits)
        ticks = log_ticks(min(vals), max(vals))
        fig.update_yaxes(tickvals=ticks, ticktext=[sci_label(t) for t in ticks])
    return base_layout(fig, 360, showlegend=True)


def log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 tick values covering ``[lo, hi]`` (for log axes)."""
    out: list[float] = []
    for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo / 1.5 <= v <= hi * 1.5:
                out.append(v)
    return out


def sci_label(v: float) -> str:
    """``2e-8`` style label (no zero-padded exponent)."""
    m, e = f"{v:.0e}".split("e")
    return f"{m}e{int(e)}"


def fig_html(fig: go.Figure, div_id: str) -> str:
    h = int(fig.layout.height or 320)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, default_height=f"{h}px",
                       config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


# Re-themes every Plotly figure on the page for dark mode (OS preference or data-theme toggle):
# surfaces, grid and ink, plus every trace colour mapped to its validated dark-mode step
# (SERIES -> SERIES_DARK; the light palette fails the lightness band on the dark surface).
_L2D = {**{a.lower(): b.lower() for a, b in zip(SERIES, SERIES_DARK, strict=True)}, LIGHT["surface"]: DARK["surface"],
        LIGHT["muted"]: DARK["muted"], LIGHT["axis"]: DARK["axis"]}
THEME_JS = """
(function(){
  const LIGHT = %s, DARK = %s, L2D = %s;
  const D2L = {}; Object.keys(L2D).forEach(function(k){ D2L[L2D[k]] = k; });
  function isDark(){ const r=document.documentElement.getAttribute('data-theme');
    if(r==='dark') return true; if(r==='light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches; }
  function mapc(c, m){ if(typeof c==='string'){ return m[c.toLowerCase()] || c; }
    if(Array.isArray(c)){ return c.map(function(x){ return (typeof x==='string') ? (m[x.toLowerCase()] || x) : x; }); } return c; }
  function apply(){ const dark = isDark(), t = dark?DARK:LIGHT, m = dark?L2D:D2L;
    document.querySelectorAll('.plotly-graph-div').forEach(function(gd){
      if(!window.Plotly||!gd.layout) return;
      const upd={'paper_bgcolor':t.surface,'plot_bgcolor':t.surface,'font.color':t.ink2,'legend.font.color':t.ink2};
      Object.keys(gd.layout).forEach(function(k){ if(k.startsWith('xaxis')||k.startsWith('yaxis')){
        upd[k+'.gridcolor']=t.grid; upd[k+'.linecolor']=t.axis; upd[k+'.tickfont.color']=t.muted; }});
      (gd.layout.shapes||[]).forEach(function(sh, j){ if(sh.line && sh.line.color){ upd['shapes['+j+'].line.color'] = mapc(sh.line.color, m); } });
      try{ Plotly.relayout(gd, upd); }catch(e){}
      (gd.data||[]).forEach(function(tr, i){ const u = {};
        if(tr.line && tr.line.color) u['line.color'] = [mapc(tr.line.color, m)];
        if(tr.marker && tr.marker.color !== undefined) u['marker.color'] = [mapc(tr.marker.color, m)];
        if(tr.marker && tr.marker.line && tr.marker.line.color) u['marker.line.color'] = [mapc(tr.marker.line.color, m)];
        if(Object.keys(u).length){ try{ Plotly.restyle(gd, u, [i]); }catch(e){} } }); }); }
  window.__applyChartTheme = apply;
  window.addEventListener('load', apply);
  if(window.matchMedia){ window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply); }
})();
""" % (str(LIGHT).replace("'", '"'), str(DARK).replace("'", '"'), str(_L2D).replace("'", '"'))

PAGE_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
  --border:rgba(11,11,11,0.10);--accent:#2a78d6;--good:#006300;--bad:#d03b3b;--warn-bg:#fff4dc;--warn-ink:#6b4a00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;
  --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,0.10);--accent:#3987e5;--good:#0ca30c;--bad:#e66767;
  --warn-bg:#3a2d0c;--warn-ink:#fad27a}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--border:rgba(255,255,255,0.10);--accent:#3987e5;--good:#0ca30c;--bad:#e66767;--warn-bg:#3a2d0c;--warn-ink:#fad27a}
*{box-sizing:border-box} body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:14px;line-height:1.45} main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px;font-weight:650} h2{font-size:17px;margin:28px 0 10px;font-weight:620}
.sub{color:var(--ink2);margin:0 0 16px} .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 14px 6px;margin:10px 0}
.card h3{margin:0 0 6px;font-size:13px;font-weight:600;color:var(--ink2)}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px}
.kpi .l{color:var(--ink2);font-size:12px} .kpi .v{font-size:22px;font-weight:620;margin-top:2px;overflow-wrap:anywhere;
  font-variant-numeric:tabular-nums}
.pos{color:var(--good)} .neg{color:var(--bad)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px} @media (max-width:820px){.grid2{grid-template-columns:1fr}}
.grid2>*,.grid3>*{min-width:0} .card{overflow-x:auto}
.banner{background:var(--warn-bg);color:var(--warn-ink);border-radius:10px;padding:10px 14px;margin:8px 0 14px;font-size:13px}
.tbl{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left} .tbl th{color:var(--ink2);font-weight:600;cursor:pointer;position:sticky;top:0;background:var(--surface)}
.scroll{max-height:420px;overflow:auto} .muted{color:var(--muted)} code{font-size:12px}
.tbl td.addr{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px} .tbl td.txt{text-align:left}
.tbl th.txt{text-align:left} .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
@media (max-width:980px){.grid3{grid-template-columns:1fr}}
nav.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 4px} nav.tabs button{border:1px solid var(--border);background:var(--surface);color:var(--ink);
  border-radius:8px;padding:6px 10px;font:inherit;cursor:pointer} nav.tabs button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
.tab{display:none} .tab.on{display:block} footer{color:var(--muted);font-size:12px;margin-top:28px}
"""

TABLE_JS = """
document.querySelectorAll('table.tbl').forEach(function(t){
  t.querySelectorAll('th').forEach(function(th, i){ th.addEventListener('click', function(){
    const rows = Array.from(t.tBodies[0].rows); const asc = th.dataset.asc !== '1'; th.dataset.asc = asc ? '1' : '0';
    rows.sort(function(a,b){ const x=a.cells[i].dataset.v ?? a.cells[i].innerText, y=b.cells[i].dataset.v ?? b.cells[i].innerText;
      const nx=parseFloat(x), ny=parseFloat(y); const c = (!isNaN(nx)&&!isNaN(ny)) ? nx-ny : String(x).localeCompare(String(y));
      return asc ? c : -c; }); rows.forEach(function(r){ t.tBodies[0].appendChild(r); }); }); });
});
"""


def fmt_num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "–"
    if math.isinf(f):
        return "∞" if f > 0 else "−∞"
    if abs(f) >= 1e6:
        return f"{f:,.0f}"
    return f"{f:,.{digits}f}"


def short_addr(a: str, head: int = 4, tail: int = 4) -> str:
    """``GypFxa6r…8SEz37`` style shortening for base58 addresses."""
    return a if len(a) <= head + tail + 1 else f"{a[:head]}…{a[-tail:]}"


def fmt_value(v: Any, kind: str | None, digits: int = 4) -> str:
    """Format one cell. Kinds: pct, pct_s, sol, sol_s, int, num1, num2, bps, ms, time, datetime, addr, sci,
    lamports_sol, text."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    try:
        if kind == "pct":
            return f"{100 * float(v):.2f}%"
        if kind == "pct_s":
            return f"{100 * float(v):+.2f}%"
        if kind == "sol":
            return f"{float(v):,.4f}"
        if kind == "sol_s":
            return f"{float(v):+,.4f}"
        if kind == "int":
            return f"{int(v):,}"
        if kind == "num1":
            return f"{float(v):,.1f}"
        if kind == "num2":
            return f"{float(v):,.2f}"
        if kind == "bps":
            return f"{float(v):+,.1f} bps"
        if kind == "ms":
            return f"{float(v):,.0f} ms"
        if kind in ("time", "datetime"):
            from datetime import UTC, datetime

            d = datetime.fromtimestamp(int(v) / 1000, UTC)
            return d.strftime("%H:%M:%S") if kind == "time" else d.strftime("%m-%d %H:%M:%S")
        if kind == "addr":
            return short_addr(str(v))
        if kind == "sci":
            return f"{float(v):.3e}"
        if kind == "lamports_sol":
            return f"{int(v) / 1e9:+,.6f}"
    except (TypeError, ValueError, OverflowError):
        return str(v)
    if kind == "text" or isinstance(v, str):
        return str(v)
    return fmt_num(v, digits)


_TEXT_KINDS = {"text", "addr"}


def html_table(df: pl.DataFrame, max_rows: int = 500, digits: int = 4, formats: dict[str, str] | None = None,
               headers: dict[str, str] | None = None, scroll: bool = True, full_data: str = "") -> str:
    """Sortable HTML table. Every value is HTML-escaped (token names and symbols are attacker-controlled);
    ``data-v`` keeps the raw value so sorting stays numeric and full addresses remain available on hover."""
    import html as _html

    if df.is_empty():
        return '<p class="muted">No rows.</p>'
    formats = formats or {}
    headers = headers or {}
    cols = df.columns
    kinds = [formats.get(c) or ("text" if df.schema[c] == pl.String else None) for c in cols]
    head = "".join(f'<th class="{"txt" if k in _TEXT_KINDS else ""}">{_html.escape(headers.get(c, c))}</th>' for c, k in zip(cols, kinds, strict=True))
    body = []
    for row in df.head(max_rows).iter_rows():
        cells = []
        for v, k in zip(row, kinds, strict=True):
            raw = "" if v is None else str(v)
            shown = _html.escape(fmt_value(v, k, digits))
            cls = "addr" if k == "addr" else "txt" if k == "text" else ""
            title = f' title="{_html.escape(raw, quote=True)}"' if k == "addr" else ""
            cells.append(f'<td class="{cls}" data-v="{_html.escape(raw, quote=True)}"{title}>{shown}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    where = f" (full table: {_html.escape(full_data)})" if full_data else ""
    more = (f'<p class="muted">Showing the first {min(max_rows, df.height):,} of {df.height:,} rows{where}.</p>'
            if df.height > max_rows else "")
    table = f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    return (f'<div class="scroll">{table}</div>' if scroll else table) + more
