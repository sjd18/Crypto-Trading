"""Dashboard pages over the session archive and strategy tuning.

Recent Session Breakdown
    pick a saved hfts / hftl session (execution/sessions.py) and see what happened: return, PnL,
    trades, win rate and costs per strategy, equity and drawdown, cumulative PnL per strategy, exit
    reasons, what became of each strategy's buy signals, the market activity the stream recorded in
    the session window, the parameters each strategy ran with, and every closed trade (CSV download).

Strategy Parameters
    tune one strategy on the recorded real events with the optimiser (Bayesian optimisation by
    default: a Gaussian-process model of score vs parameters picks each next trial), see every study
    of that strategy next to the current parameters on the same train / validation periods, with
    plain-language verdicts and overfitting statistics, plus the strategy's pooled live paper record
    from the saved sessions; **Save for hfts** writes the chosen parameters to tuned_params.json.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

import numpy as np
import polars as pl

from pumpfun_hft.analytics import charts as C
from pumpfun_hft.dashboard.views import _kpi, _num, _signed_cls, positions_html, signed_badge
from pumpfun_hft.optimizer.objective import PENALTY
from pumpfun_hft.utils.timeutil import ms_to_dt

_E = html.escape


# --------------------------------------------------------------------------- shared helpers
def _when(ms: Any) -> str:
    return ms_to_dt(int(ms)).strftime("%m-%d %H:%M UTC") if ms else "–"


def _dur(ms: float | None) -> str:
    if ms is None or ms < 0:
        return "–"
    s = int(ms / 1000)
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s % 60:02d} s" if m else f"{s} s"


def _pct(v: Any) -> str:
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{100 * float(v):.1f}%"


def _status_badge(status: str) -> str:
    color = {"running": "var(--accent)", "ended": "var(--good)", "interrupted": "#d89b00"}.get(status, "var(--muted)")
    return (f"<span class='pill' style='color:{color};background:color-mix(in srgb,{color} 14%,transparent)'>"
            f"{_E(status)}</span>")


TRADING_CSS = """
.tv-head{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:6px 0 4px}
.tv-head select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--ink);
  max-width:100%}
.tv-meta{color:var(--ink2);font-size:13px;display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin:4px 0 12px}
.tv-kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;font-size:13px} .tv-kv dt{color:var(--ink2)} .tv-kv dd{margin:0;
  font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.tv-params{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.tv-params h4{margin:0 0 6px;font-size:13px} .tv-dl a{margin-right:14px}
.tv-verdict{border-radius:8px;padding:9px 12px;font-size:13px;margin:4px 0 10px}
.tv-verdict.good{background:color-mix(in srgb,var(--good) 13%,transparent);color:var(--good)}
.tv-verdict.warn{background:var(--warn-bg);color:var(--warn-ink)} .tv-verdict.bad{background:color-mix(in srgb,var(--bad) 13%,transparent);color:var(--bad)}
.tv-study{padding:14px} .tv-study h3{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center}
.tv-grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:12px;align-items:start}
@media (max-width:980px){.tv-grid{grid-template-columns:minmax(0,1fr)}}
.tv-form label{display:block;font-size:12px;color:var(--ink2);margin:10px 0 4px}
.tv-form select,.tv-form input[type=number]{font:inherit;width:100%;padding:6px 8px;border-radius:8px;border:1px solid var(--border);
  background:var(--page);color:var(--ink)} .tv-form .row{display:grid;grid-template-columns:1fr 1fr;gap:0 12px}
.tv-form .check{display:flex;gap:8px;align-items:center;margin-top:12px;font-size:13px;color:var(--ink)}
.tv-note{color:var(--ink2);font-size:12.5px;margin:6px 0 0}
"""


# --------------------------------------------------------------------------- Recent Session Breakdown
def session_label(info: dict[str, Any]) -> str:
    sm = info.get("summary") or {}
    end = info.get("ended_ms") or info.get("updated_ms")
    span = _dur((end or 0) - (info.get("started_ms") or 0)) if end else "–"
    return (f"{info.get('profile', '?')} · {_when(info.get('started_ms'))} · {span} · {int(sm.get('closed_trades') or 0)} trades · "
            f"{_num(sm.get('realized_pnl_sol')):+.4f} SOL · {info.get('status', '?')}")


TRADE_FMT = {"mint": "mono", "strategy": "text", "entry_ms": "time_cs", "exit_ms": "time_cs", "hold_s": "num1", "cost_sol": "sol",
             "proceeds_sol": "sol", "pnl_sol": "sol_c", "ret": "pct_c", "exit_reason": "text"}
TRADE_HDR = {"entry_ms": "entry (UTC)", "exit_ms": "exit (UTC)", "hold_s": "hold (s)", "cost_sol": "cost", "proceeds_sol": "proceeds",
             "pnl_sol": "PnL (SOL)", "ret": "return", "exit_reason": "exit"}
BREAKDOWN_FMT = {"strategy": "text", "trades": "int", "wins": "int", "win_rate": "pct", "pnl_sol": "sol_c", "avg_ret": "pct_c",
                 "best_ret": "pct_c", "worst_ret": "pct_c", "profit_factor": "num2", "avg_hold_s": "num1", "costs_sol": "sol",
                 "buy_signals": "int", "submitted": "int"}
BREAKDOWN_HDR = {"win_rate": "win rate", "pnl_sol": "PnL (SOL)", "avg_ret": "avg return", "best_ret": "best", "worst_ret": "worst",
                 "profit_factor": "profit factor", "avg_hold_s": "avg hold (s)", "costs_sol": "fees + tx (SOL)",
                 "buy_signals": "buy signals", "submitted": "orders"}


def page_sessions(sessions: list[dict[str, Any]], info: dict[str, Any] | None, frames: dict[str, pl.DataFrame],
                  breakdown: pl.DataFrame, server: bool) -> str:
    """Recent Session Breakdown (see module docstring)."""
    css = f"<style>{TRADING_CSS}</style>"
    if not sessions or info is None:
        return css + ("<p class='muted'>No saved sessions yet. Every paper (<code>hfts trade</code>) or live (<code>hftl trade</code>) "
                      "session on the live stream is saved when it runs; start one from the Strategy builder tab.</p>")
    sid = info["session_id"]
    opts = "".join(f"<option value='{_E(s['session_id'])}'{' selected' if s['session_id'] == sid else ''}>{_E(session_label(s))}</option>"
                   for s in sessions)
    picker = (f"<form method='get' class='tv-head'><label class='muted' for='tv-s'>Session</label>"
              f"<select id='tv-s' name='session' onchange='this.form.submit()'>{opts}</select></form>")
    sm = info.get("summary") or {}
    trades, equity = frames.get("trades", pl.DataFrame()), frames.get("equity", pl.DataFrame())
    end = info.get("ended_ms") or info.get("updated_ms") or 0
    meta = (f"<div class='tv-meta'>{_status_badge(info.get('status', '?'))}<span><b>{_E(str(info.get('profile', '')))}</b> · "
            f"{_E(str(info.get('session', '')))}</span><span>{_when(info.get('started_ms'))} → {_when(end)} "
            f"({_dur(end - (info.get('started_ms') or 0))})</span>"
            + "".join(f"<span class='pill'>{_E(n)}</span>" for n in info.get("strategies") or [])
            + (f"<span class='muted'>exit overlay: {_E(str(info.get('exit_overlay')))}</span>" if info.get("exit_overlay") else "")
            + "</div>")
    ret, pnl = sm.get("return"), _num(sm.get("realized_pnl_sol"))
    n, wins = int(sm.get("closed_trades") or 0), int(sm.get("wins") or 0)
    gross_win = float(trades.filter(pl.col("pnl_sol") > 0)["pnl_sol"].sum()) if trades.height else 0.0
    gross_loss = float(-trades.filter(pl.col("pnl_sol") <= 0)["pnl_sol"].sum()) if trades.height else 0.0
    mk = info.get("market") or {}
    stream = sm.get("stream") or {}
    tiles = [_kpi("Return", _pct(ret) if ret is None else f"{100 * ret:+.2f}%", _signed_cls(ret or 0.0)),
             _kpi("Realised PnL (SOL)", f"{pnl:+.4f}", _signed_cls(pnl)),
             _kpi("Closed trades", f"{n:,}"), _kpi("Win rate", _pct(wins / n) if n else "–"),
             _kpi("Profit factor", f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "–"),
             _kpi("Avg hold", f"{float(trades['hold_s'].mean()):.0f} s" if trades.height else "–"),
             _kpi("Fills / failed", f"{int(sm.get('fills') or 0):,} / {int(sm.get('failed_fills') or 0):,}"),
             _kpi("Events streamed", f"{int(stream.get('events') or sm.get('events_processed') or 0):,}")]
    if mk:
        tiles += [_kpi("Tokens created", f"{int(mk.get('tokens_created') or 0):,}"),
                  _kpi("Curve volume", f"{float(mk.get('volume_sol') or 0):,.0f} <span class='muted' style='font-size:13px'>SOL</span>")]
    out = [css, picker, meta, f"<div class='kpis'>{''.join(tiles)}</div>"]
    if info.get("status") == "running":
        out.append("<p class='muted'>This session is still running; the archive is rewritten every 30 s and this page reloads "
                   "every 30 s.</p><script>setTimeout(function(){location.reload();},30000);</script>")
    elif info.get("status") == "interrupted":
        out.append("<div class='banner'>This session stopped without its final save (window closed or crash); the figures "
                   "are from its last save.</div>")
    # per strategy
    out.append(f"<div class='card'><h3>By strategy</h3>{C.html_table(breakdown, formats=BREAKDOWN_FMT, headers=BREAKDOWN_HDR, scroll=False)}"
               "<p class='tv-note'>buy signals: how often the strategy wanted to buy; orders: how many of those passed the cost "
               "gate, sizing and risk checks and were sent.</p></div>")
    charts = []
    if breakdown.height:
        b = breakdown.sort("pnl_sol")
        fig = C.hbars(b["strategy"].to_list(), [float(x) for x in b["pnl_sol"].to_list()], "realised PnL (SOL)", signed=True)
        charts.append(f"<div class='card'><h3>PnL by strategy (SOL)</h3>{C.fig_html(fig, 'tv-pnl')}</div>")
    if trades.height:
        series = {}
        for name in trades["strategy"].unique().sort().to_list():
            t = trades.filter(pl.col("strategy") == name).sort("exit_ms")
            x = [int(info.get("started_ms") or t["exit_ms"][0]), *t["exit_ms"].to_list()]
            series[name] = (C._dt(x), [0.0, *np.cumsum(t["pnl_sol"].to_numpy()).tolist()])
        fig = C.multi_line(series, "cumulative PnL (SOL)", {k: C.strategy_color(k) for k in series})
        fig.update_traces(mode="lines+markers", line_shape="hv", marker={"size": 7})
        fig.add_hline(y=0, line={"width": 1, "color": C.LIGHT["axis"]})
        charts.append(f"<div class='card'><h3>Cumulative PnL per strategy</h3>{C.fig_html(fig, 'tv-cum')}</div>")
    if charts:
        out.append(f"<div class='grid2'>{''.join(charts)}</div>")
    if equity.height > 1:
        out.append(f"<div class='card'><h3>Equity and drawdown</h3>"
                   f"{C.fig_html(C.equity_drawdown(equity, float(sm.get('initial_sol') or equity['equity_sol'][0])), 'tv-eq')}</div>")
    # exits, signals
    side = []
    if trades.height:
        ex = (trades.group_by("strategy", "exit_reason").agg(pl.len().alias("trades"), pl.col("pnl_sol").sum().alias("pnl_sol"),
                                                             pl.col("ret").mean().alias("avg_ret"))
              .sort(["strategy", "trades"], descending=[False, True]))
        side.append(f"<div class='card'><h3>Exit reasons</h3>{C.html_table(ex, formats={'pnl_sol': 'sol_c', 'avg_ret': 'pct_c', 'exit_reason': 'text'}, headers={'exit_reason': 'exit', 'pnl_sol': 'PnL (SOL)', 'avg_ret': 'avg return'}, scroll=False)}</div>")
    counts = info.get("signal_counts") or {}
    rows = [{"strategy": k.split(":")[0], "outcome": k.split(":", 2)[2], "signals": v} for k, v in counts.items()
            if k.count(":") >= 2 and k.split(":")[1] == "BUY"]
    if rows:
        sig = pl.DataFrame(rows).sort(["strategy", "signals"], descending=[False, True])
        side.append(f"<div class='card'><h3>What happened to the buy signals</h3>{C.html_table(sig, scroll=False)}"
                    "<p class='tv-note'>submitted = order sent · cost_gate = expected return below the round-trip cost · "
                    "low_confidence = below sizing.min_confidence · vetoed = rug avoidance · risk:… = a risk limit.</p></div>")
    if side:
        out.append(f"<div class='grid2'>{''.join(side)}</div>")
    # market + params
    lower = []
    if mk:
        kinds = mk.get("by_kind") or {}
        dl = "".join(f"<dt>{_E(k)}</dt><dd>{v:,}</dd>" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        lower.append("<div class='card'><h3>Market activity recorded during the session</h3><dl class='tv-kv'>"
                     f"<dt>events</dt><dd>{int(mk.get('events') or 0):,}</dd><dt>tokens created</dt><dd>{int(mk.get('tokens_created') or 0):,}</dd>"
                     f"<dt>tokens traded</dt><dd>{int(mk.get('tokens_traded') or 0):,}</dd><dt>wallets</dt><dd>{int(mk.get('wallets') or 0):,}</dd>"
                     f"<dt>bonding-curve trades</dt><dd>{int(mk.get('curve_trades') or 0):,}</dd><dt>bonding-curve volume</dt><dd>{float(mk.get('volume_sol') or 0):,.2f} SOL "
                     f"({float(mk.get('buy_volume_sol') or 0):,.2f} buys)</dd><dt>graduations</dt><dd>{int(mk.get('graduations') or 0):,}</dd>"
                     + f"<dt>PumpSwap pool trades</dt><dd>{int(mk.get('pool_trades') or 0):,} ({int(mk.get('unknown_pool_trades') or 0):,} of "
                       "tokens the stream never saw created; not in the volume)</dd>"
                     + f"{dl}</dl><p class='tv-note'>These events are in the real data set, so <code>hftr backtest --start … --end …</code> "
                     "replays exactly this window.</p></div>")
    params = info.get("params") or {}
    tuned = info.get("tuned") or {}
    if params:
        blocks = "".join(
            f"<div><h4>{_E(n)}{' <span class=pill>tuned · ' + _E(str(tuned[n])) + '</span>' if n in tuned else ''}</h4><dl class='tv-kv'>"
            + "".join(f"<dt>{_E(str(k))}</dt><dd>{_E(json.dumps(v) if not isinstance(v, str) else v)}</dd>" for k, v in p.items())
            + "</dl></div>" for n, p in params.items())
        lower.append(f"<div class='card'><h3>Parameters each strategy ran with</h3><div class='tv-params'>{blocks}</div></div>")
    if lower:
        out.append(f"<div class='grid2'>{''.join(lower)}</div>")
    if info.get("positions"):
        out.append(f"<div class='card'><h3>Open at the last save</h3>{positions_html(pl.DataFrame(info['positions'], infer_schema_length=None))}</div>")
    # trades
    if trades.height:
        t = trades.sort("exit_ms", descending=True).select([c for c in TRADE_FMT if c in trades.columns])
        out.append(f"<div class='card'><h3>Closed trades ({trades.height:,}, newest first)</h3>"
                   f"{C.html_table(t, max_rows=300, formats=TRADE_FMT, headers=TRADE_HDR, scroll=False, full_data='download trades.csv below')}</div>")
    if server:
        q = _E(sid)
        out.append("<p class='tv-dl'>Download: " + "".join(
            f"<a href='/api/sessions/{q}/{n}.csv'>{n}.csv</a>" for n in ("trades", "fills", "equity", "signals")) + "</p>")
    return "".join(out)


# --------------------------------------------------------------------------- Strategy Parameters
def score_text(score: Any, min_trades: int) -> str:
    if score is None:
        return "–"
    v = float(score)
    if v < PENALTY + min_trades:
        return f"too few trades ({int(round(v - PENALTY))})"
    if v <= PENALTY / 2:
        return "drawdown over limit"
    if v == PENALTY / 4:
        return "no score"
    return f"{v:.3f}"


def study_verdict(st: dict[str, Any], min_trades: int) -> tuple[str, str]:
    tv, bv = st.get("selected_val_score"), st.get("baseline_val_score")
    n_val = (st.get("selected_val_metrics") or {}).get("n_trades")
    if tv is None:
        return "warn", "This study has no validation scores to compare; see the metrics below."
    if tv < PENALTY + min_trades:
        return "bad", (f"Not enough trades in the validation period to judge ({n_val or 0} < {min_trades}). Tune on more data "
                       "(a longer window), or this strategy trades too rarely on it.")
    if tv <= PENALTY / 2:
        return "bad", "The tuned parameters break the drawdown limit on the validation period."
    if bv is not None and bv > PENALTY / 4 and tv <= bv:
        return "warn", "Did not beat the current parameters on unseen (validation) data: keeping the current ones is the safer choice."
    ret = (st.get("selected_val_metrics") or {}).get("total_return")
    if ret is not None and ret <= 0:
        return "warn", (f"Less bad than the current parameters on unseen (validation) data, but still losing money there "
                        f"({100 * ret:+.2f}%). Saving it makes hfts lose less, not win; this strategy has no edge on this data yet.")
    pbo = (st.get("pbo") or {}).get("pbo")
    if pbo is not None and math.isfinite(pbo) and pbo > 0.5:
        return "warn", (f"Beat the current parameters on validation data, but the probability that the search overfit is high "
                        f"(PBO {pbo:.0%}). Paper trade it with hfts before trusting it.")
    return "good", "Beat the current parameters on unseen (validation) data."


_METRIC_ROWS = (("n_trades", "trades", "int"), ("total_return", "total return", "pct_s"), ("win_rate", "win rate", "pct"),
                ("sharpe", "Sharpe", "num2"), ("profit_factor", "profit factor", "num2"), ("max_drawdown", "max drawdown", "pct"))


def _metrics_table(st: dict[str, Any], min_trades: int) -> str:
    cols = [("current · train", st.get("baseline_train_metrics"), st.get("baseline_train_score")),
            ("tuned · train", st.get("selected_train_metrics"), st.get("selected_train_score")),
            ("current · validation", st.get("baseline_val_metrics"), st.get("baseline_val_score")),
            ("tuned · validation", st.get("selected_val_metrics"), st.get("selected_val_score"))]
    if st.get("test_metrics"):
        cols.append(("tuned · test (sealed)", st["test_metrics"], None))
    cols = [c for c in cols if c[1]]
    head = "<th>metric</th>" + "".join(f"<th>{_E(c[0])}</th>" for c in cols)
    rows = [f"<tr><td>score ({_E(str(st.get('objective', '')))})</td>" + "".join(f"<td>{_E(score_text(c[2], min_trades))}</td>" for c in cols) + "</tr>"]
    for key, label, kind in _METRIC_ROWS:
        cells = []
        for _, m, _s in cols:
            v = (m or {}).get(key)
            cls = (" class='pos'" if v > 0 else " class='neg'" if v < 0 else "") if key == "total_return" and isinstance(v, (int, float)) else ""
            cells.append(f"<td{cls}>{_E(C.fmt_value(v, kind))}</td>")
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _params_table(st: dict[str, Any]) -> str:
    sel, base = st.get("selected_params") or {}, st.get("baseline_params") or {}
    rows = []
    for k, v in sel.items():
        b = base.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(b, (int, float)) and not isinstance(b, bool) and b:
            ch = (v - b) / abs(b)
            change = f"<span class='{'pos' if ch > 0 else 'neg' if ch < 0 else ''}'>{100 * ch:+.0f}%</span>"
        else:
            change = "" if v == b else "changed"
        fv = f"{v:.4g}" if isinstance(v, float) else str(v)
        fb = "–" if b is None else f"{b:.4g}" if isinstance(b, float) else str(b)
        rows.append(f"<tr><td>{_E(k)}</td><td>{_E(fb)}</td><td><b>{_E(fv)}</b></td><td>{change}</td></tr>")
    return ("<table class='tbl'><thead><tr><th>parameter</th><th>current</th><th>tuned</th><th>change</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _split_note(st: dict[str, Any]) -> str:
    ev = st.get("split_events") or {}
    if not ev:
        return ("<p class='tv-note'>Split by clock time. If the recording has long gaps a split can be nearly empty; "
                "studies run now split by event count when that is the case.</p>") if st.get("split_mode") is None else ""
    how = "event count (the recording has gaps)" if st.get("split_mode") == "events" else "time"
    return ("<p class='tv-note'>Split by " + how + ": " + " · ".join(f"{_E(k)} {int(v):,} events" for k, v in ev.items())
            + "</p>")


def _study_card(st: dict[str, Any], tuned_entry: dict[str, Any] | None, min_trades: int, server: bool) -> str:
    level, text = study_verdict(st, min_trades)
    sid = _E(str(st.get("study_id", "")))
    active = tuned_entry is not None and tuned_entry.get("study_id") == st.get("study_id")
    dsr, pbo = (st.get("dsr") or {}).get("dsr"), (st.get("pbo") or {}).get("pbo")
    stats = (f"Deflated Sharpe {('%.2f' % dsr) if isinstance(dsr, (int, float)) and math.isfinite(dsr) else '–'} "
             "(probability the edge is real after trying this many settings; higher is better) · "
             f"PBO {_pct(pbo) if isinstance(pbo, (int, float)) and math.isfinite(pbo) else '–'} "
             "(probability the selection is overfit; lower is better)")
    if not server:
        action = ""
    elif active:
        action = (f"<span class='pill' style='color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)'>active for hfts</span>"
                  f" <button type='button' class='bt-btn' data-remove='{_E(str(st.get('strategy', '')))}'>Remove (back to configured)</button>")
    elif level == "bad":
        action = ("<button type='button' class='bt-btn' disabled>Save for hfts</button> <span class='muted'>not enough evidence "
                  "on the validation period to use these parameters</span>")
    else:
        action = f"<button type='button' class='bt-btn primary' data-save='{sid}'>Save for hfts</button>"
    window = (f"{int(st.get('n_events') or 0):,} events, {_when(st.get('data_start_ms'))} → {_when(st.get('data_end_ms'))}"
              if st.get("n_events") else "")
    return (f"<div class='card tv-study'><h3><span>{_when(st.get('created_ms'))}</span><span class='pill'>{_E(str(st.get('method', '')))}</span>"
            f"<span class='muted' style='font-weight:400'>{int(st.get('n_trials') or len(st.get('trials') or []))} trials · objective "
            f"{_E(str(st.get('objective', '–')))} · {window} · {sid}</span></h3>"
            f"<div class='tv-verdict {level}'>{_E(text)}</div>{_split_note(st)}"
            f"<div class='grid2'><div>{_metrics_table(st, min_trades)}</div><div>{_params_table(st)}</div></div>"
            f"<p class='tv-note'>{stats}</p><div class='bt-actions'>{action}</div></div>")


PARAMS_JS = r"""
(function(){
  const H = {'Content-Type': 'application/json', 'X-Pumpfun-Dashboard': '1'};
  const $ = id => document.getElementById(id), term = $('bt-term');
  let job = null, next = 0, wasRunning = false, pending = null;
  function spec(){ return {strategy: $('op-strategy').value, method: $('op-method').value, trials: $('op-trials').value,
    last_hours: $('op-hours').value, objective: $('op-objective').value, workers: $('op-workers').value, final: $('op-final').checked}; }
  function msg(t, err){ const m = $('bt-msg'); m.textContent = t || ''; m.className = 'bt-msg' + (err ? ' err' : ' muted'); }
  function buttons(){ const r = !!(job && job.running); $('bt-run').disabled = r; $('bt-stop').disabled = !r || job.stopping; }
  const EST = JSON.parse($('op-est').textContent);
  function estimate(){
    const n = EST.events[$('op-hours').value], w = +$('op-workers').value, t = +$('op-trials').value;
    if (n == null){ $('op-estimate').textContent = ''; return; }
    if (!n){ $('op-estimate').textContent = 'No recorded events in this window.'; return; }
    const min = (n * EST.train * Math.ceil(t / w) + n * EST.validation * Math.ceil((EST.top_k + 1) / w)) / EST.rate + 1;
    $('op-estimate').textContent = n.toLocaleString() + ' recorded events · roughly ' +
      (min < 90 ? Math.round(min) + ' min' : (min / 60).toFixed(1) + ' h') + ' with these settings (each trial is a full backtest).';
  }
  function preview(){
    estimate();
    clearTimeout(pending);
    pending = setTimeout(function(){
      fetch('/api/optimize/command', {method: 'POST', headers: H, body: JSON.stringify(spec())}).then(r => r.json()).then(function(r){
        if (r.error){ $('bt-cmd').textContent = '…'; msg(r.error, true); } else { $('bt-cmd').textContent = r.powershell; msg(''); }
      });
    }, 120);
  }
  $('op-strategy').addEventListener('change', function(){ location.href = '/params?strategy=' + encodeURIComponent(this.value); });
  document.querySelectorAll('.tv-form select, .tv-form input').forEach(el => el.addEventListener('change', preview));
  $('bt-run').addEventListener('click', function(){
    fetch('/api/optimize', {method: 'POST', headers: H, body: JSON.stringify(spec())}).then(r => r.json()).then(function(r){
      if (r.error){ msg(r.error, true); return; } term.textContent = ''; next = 0; poll();
    });
  });
  $('bt-stop').addEventListener('click', () => fetch('/api/optimize/stop', {method: 'POST', headers: H}).then(poll));
  document.querySelectorAll('[data-save]').forEach(b => b.addEventListener('click', function(){
    b.disabled = true;
    fetch('/api/tuned', {method: 'POST', headers: H, body: JSON.stringify({study_id: b.dataset.save})}).then(r => r.json()).then(function(r){
      if (r.error){ b.disabled = false; alert(r.error); } else location.reload();
    });
  }));
  document.querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', function(){
    fetch('/api/tuned/remove', {method: 'POST', headers: H, body: JSON.stringify({strategy: b.dataset.remove})}).then(() => location.reload());
  }));
  function poll(){
    fetch('/api/optimize/current?since=' + next).then(r => r.json()).then(function(r){
      if (r.job && job && r.job.id !== job.id) term.textContent = '';
      job = r.job;
      if (r.lines && r.lines.length){
        const atEnd = term.scrollHeight - term.scrollTop - term.clientHeight < 40;
        term.textContent += (term.textContent ? '\n' : '') + r.lines.join('\n');
        if (atEnd) term.scrollTop = term.scrollHeight;
      }
      next = r.next || 0;
      $('bt-job').textContent = !job ? 'no optimisation started from this page' : job.running
        ? (job.stopping ? 'stopping ' : 'tuning ') + (job.strategy || '') : (job.strategy || '') + ' finished · exit code ' + job.exit_code;
      if (wasRunning && job && !job.running && job.exit_code === 0 && job.strategy === $('op-strategy').value) setTimeout(() => location.reload(), 1500);
      wasRunning = !!(job && job.running);
      buttons();
    }).catch(() => {});
  }
  preview(); poll(); setInterval(poll, 1000);
})();
"""


#: measured on the real data set: one worker backtests about this many events per minute (rough)
EVENTS_PER_WORKER_MINUTE = 450_000


def page_params(catalogue: list[dict[str, Any]], strategy: str, studies: list[dict[str, Any]], tuned: dict[str, Any],
                evidence: dict[str, Any], configured: dict[str, Any], opt: Any, cpus: int, server: bool,
                window_events: dict[int, int] | None = None) -> str:
    """Strategy Parameters (see module docstring). ``opt`` is settings.optimizer; ``window_events``:
    events in each data window of the form (hours -> count, 0 = all) for the run-time estimate."""
    from pumpfun_hft.dashboard.views import BUILDER_CSS

    css = f"<style>{BUILDER_CSS}{TRADING_CSS}</style>"
    if not server:
        return css + "<p class='muted'>Tuning runs on the dashboard server: run <code>hfts dashboard</code> and open /params.</p>"
    entry = next((c for c in catalogue if c["name"] == strategy), None)
    opts = "".join(f"<option value='{_E(c['name'])}'{' selected' if c['name'] == strategy else ''}>{_E(c['label'])}"
                   f"{' · tuned' if c.get('tuned') else ''}{' · auto search space' if c.get('space') == 'derived' else ''}</option>"
                   for c in catalogue)
    hours = [(6, "last 6 h"), (12, "last 12 h"), (24, "last 24 h"), (48, "last 48 h"), (168, "last 7 days"), (0, "all recorded data")]
    methods = [("bayesian", "Bayesian optimisation (ML: Gaussian process) · recommended"), ("genetic", "Genetic algorithm"),
               ("random", "Random search"), ("grid", "Grid search")]
    objectives = [("sharpe", "Sharpe ratio"), ("sortino", "Sortino ratio"), ("profit_factor", "Profit factor"),
                  ("expectancy", "Expectancy per trade"), ("calmar", "Return / max drawdown"), ("total_return", "Total return"),
                  ("composite", "Composite (Sharpe x trades x drawdown)")]

    def select(id_: str, items: list[tuple[Any, str]], chosen: Any) -> str:
        return f"<select id='{id_}'>" + "".join(f"<option value='{_E(str(v))}'{' selected' if v == chosen else ''}>{_E(t)}</option>"
                                                for v, t in items) + "</select>"

    form = (
        "<section class='card bt-build tv-form' aria-label='Tune a strategy'>"
        f"<h3>1 · Strategy</h3><select id='op-strategy'>{opts}</select>"
        + (f"<p class='tv-note'>{_E(entry['summary'])}. "
           + ("Searches the configured optimizer.spaces ranges." if entry and entry.get("space") == "configured" else
              "No configured search space: each numeric parameter is searched from half to one and a half times its current value.")
           + "</p>" if entry else "")
        + "<h3>2 · Data and method</h3>"
        f"<label for='op-hours'>Recorded real events to tune on</label>{select('op-hours', hours, 0)}"
        f"<label for='op-method'>Search method</label>{select('op-method', methods, 'bayesian')}"
        f"<div class='row'><div><label for='op-objective'>Optimise for</label>{select('op-objective', objectives, opt.objective)}</div>"
        f"<div><label for='op-trials'>Trials</label>{select('op-trials', [(16, '16'), (32, '32'), (64, '64'), (100, '100'), (200, '200')], 32)}</div></div>"
        f"<div class='row'><div><label for='op-workers'>Parallel workers</label>"
        f"{select('op-workers', [(i, str(i)) for i in range(1, cpus + 1)], min(4, cpus))}</div><div></div></div>"
        "<p class='tv-note' id='op-estimate'></p>"
        "<label class='check'><input type='checkbox' id='op-final'> also score the sealed test period (do this once, for the final choice)</label>"
        "<p class='tv-note'>The data is split in time order: parameters are searched on the first half (train), the best "
        "candidates are compared on the next 20 % (validation) next to the current parameters, and the last 30 % stays "
        "sealed. When the recording has gaps the split follows the event count, so no part is empty. Each trial is a full "
        "backtest with every cost charged.</p>"
        "<h3>3 · Command</h3><div class='bt-cmd'><code id='bt-cmd'>…</code></div>"
        "<div class='bt-actions'><button type='button' class='bt-btn primary' id='bt-run'>Run optimisation</button>"
        "<button type='button' class='bt-btn' id='bt-stop' disabled>Stop</button></div>"
        "<p class='bt-msg muted' id='bt-msg' role='status'></p>"
        "<div class='bt-termhead'><b>Optimiser output</b><span id='bt-job'></span></div><pre class='bt-term' id='bt-term' style='height:260px'></pre>"
        "</section>")
    te = tuned.get(strategy)
    space_names = list((studies[0].get("selected_params") or {}).keys()) if studies else []
    keys = list(dict.fromkeys([*space_names, *configured.keys()]))
    rows = "".join(
        f"<tr><td>{_E(k)}</td><td>{_E(_fmt_param(configured.get(k)))}</td>"
        f"<td>{'<b>' + _E(_fmt_param(te['params'][k])) + '</b>' if te and k in te['params'] else '–'}</td></tr>" for k in keys)
    status = (f"<div class='tv-verdict good'>hfts uses the tuned parameters (study {_E(str(te.get('study_id')))}, saved "
              f"{_when(te.get('saved_ms'))}).</div>" if te else
              "<div class='tv-verdict warn'>hfts uses the configured parameters (no tuned set saved for this strategy).</div>")
    current = (f"<div class='card'><h3>Parameters for hfts</h3>{status}<table class='tbl'><thead><tr><th>parameter</th><th>configured</th>"
               f"<th>tuned (active)</th></tr></thead><tbody>{rows}</tbody></table></div>")
    if evidence.get("trades"):
        ev = (f"<div class='kpis'>{_kpi('Sessions', str(evidence['sessions']))}{_kpi('Trades', f'{evidence['trades']:,}')}"
              f"{_kpi('Win rate', _pct(evidence.get('win_rate')))}"
              f"{_kpi('PnL (SOL)', f'{evidence['pnl_sol']:+.4f}', _signed_cls(evidence['pnl_sol']))}"
              f"{_kpi('Avg return', signed_badge(evidence.get('avg_ret')))}{_kpi('Avg hold', f'{evidence['avg_hold_s']:.0f} s')}</div>")
    else:
        ev = "<p class='muted'>No closed trades of this strategy in the saved sessions yet.</p>"
    record = (f"<div class='card'><h3>Live paper record (all saved sessions)</h3>{ev}<p class='tv-note'>What this strategy did on "
              "the live stream so far. The optimiser itself tunes on the recorded events (which include every streamed session), "
              "so its verdicts rest on far more data than these trades.</p></div>")
    min_trades = int(opt.min_trades)
    cards = [_study_card(st, te, min_trades, server) for st in studies[:5]]
    older = [_study_card(st, te, min_trades, server) for st in studies[5:]]
    studies_html = ("".join(cards) + (f"<details class='card'><summary>{len(older)} older studies</summary>{''.join(older)}</details>" if older else "")
                    if cards else "<p class='muted'>No studies of this strategy yet: run one above.</p>")
    est = json.dumps({"events": {str(k): v for k, v in (window_events or {}).items()}, "rate": EVENTS_PER_WORKER_MINUTE,
                      "top_k": int(opt.top_k), "train": float(opt.splits.train), "validation": float(opt.splits.validation)})
    return (css + "<div id='bt-root' class='tv-grid'>" + form + f"<div>{current}{record}</div></div>"
            f"<h2>Studies of {_E(strategy)}</h2>{studies_html}"
            f"<script type='application/json' id='op-est'>{est}</script><script>{PARAMS_JS}</script>")


def _fmt_param(v: Any) -> str:
    if v is None:
        return "–"
    return f"{v:.4g}" if isinstance(v, float) else str(v)
