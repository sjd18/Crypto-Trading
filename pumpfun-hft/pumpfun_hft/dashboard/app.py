"""Local analytics dashboard (FastAPI + Plotly, fully offline).

Run with ``python -m pumpfun_hft.main dashboard`` then open http://127.0.0.1:8050.
Pages: Overview · Trades · Equity curve · Drawdown · Heatmaps · Wallet explorer · Token
explorer · Feature importance · Strategy comparison · Live monitor. JSON endpoints under
``/api`` expose the same data (``/api/runs``, ``/api/runs/{id}/metrics``, ``/api/live``).
Plotly's JavaScript is served from the installed Python package, so nothing loads from the web.
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import plotly.offline as po
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

from pumpfun_hft.analytics import charts as C
from pumpfun_hft.dashboard import views as V
from pumpfun_hft.utils.jsonutil import jsonable

PAGE_KEYS = {k for k, _ in V.PAGES}


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    return po.get_plotlyjs()


def shell(title: str, body: str, active: str, run_ids: list[str], run: str | None) -> str:
    q = html.escape(quote(run or "", safe=""))  # run ids are directory names: URL-encode, then HTML-escape
    nav = "".join(f"<a href='/{k}?run={q}' style='text-decoration:none'><button class='{'on' if k == active else ''}'>{label}</button></a>"
                  for k, label in V.PAGES)
    runs = "".join(f"<option value='{html.escape(r)}' {'selected' if r == run else ''}>{html.escape(r)}</option>" for r in run_ids)
    picker = (f"<form method='get' style='margin:6px 0'><label class='muted'>Run </label><select name='run' onchange='this.form.submit()'>{runs}</select></form>"
              if run_ids else "")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Pump.fun HFT Dashboard</title><style>{C.PAGE_CSS}</style><script src='/static/plotly.min.js'></script></head>"
            f"<body><main><h1>{html.escape(title)}</h1>{picker}<nav class='tabs'>{nav}</nav>{body}</main>"
            f"<script>{C.TABLE_JS}{C.THEME_JS}</script></body></html>")


def create_app(settings: Any, meta: Any = None) -> FastAPI:
    """Build the FastAPI application bound to ``settings.paths``."""
    data = V.DashboardData(settings.paths.resolve("reports_dir"), settings.paths.events_dir, meta)
    app = FastAPI(title="pumpfun-hft dashboard", docs_url="/api/docs")
    cfg = settings.dashboard

    @app.get("/static/plotly.min.js")
    def plotly_js() -> Response:
        return Response(_plotly_js(), media_type="application/javascript", headers={"Cache-Control": "max-age=86400"})

    def render(page: str, run: str | None, mint: str | None = None) -> str:
        r = data.load(run)
        rid = r.run_id if r else None
        if page == "overview":
            body = V.page_overview(r)
        elif page == "trades":
            body = V.page_trades(r, cfg.max_table_rows)
        elif page == "equity":
            body = V.page_equity(r)
        elif page == "drawdown":
            body = V.page_drawdown(r)
        elif page == "heatmaps":
            body = V.page_heatmaps(r)
        elif page == "wallets":
            body = V.page_wallets(data.artefact(rid, "wallets.parquet"), cfg.max_table_rows)
        elif page == "tokens":
            body = V.page_tokens(r, data, data.artefact(rid, "tokens.parquet"), mint)
        elif page == "features":
            body = V.page_features(data.importance())
        elif page == "strategies":
            body = V.page_strategies(data)
        elif page == "live":
            state = meta.all_state().get("live") if meta is not None else None
            body = V.page_live(state, cfg.live_refresh_ms, server=True)
        else:
            body = f"<p>Unknown page <code>{html.escape(page)}</code>.</p>"
        return shell(dict(V.PAGES).get(page, page), body, page, data.run_ids(), rid)

    @app.get("/", response_class=HTMLResponse)
    def index(run: str | None = None) -> str:
        return render("overview", run)

    @app.get("/{page}", response_class=HTMLResponse)
    def page(page: str, run: str | None = None, mint: str | None = Query(default=None)) -> HTMLResponse:
        return HTMLResponse(render(page, run or None, mint), status_code=200 if page in PAGE_KEYS else 404)

    @app.get("/api/runs")
    def api_runs() -> JSONResponse:
        out = []
        for rid in data.run_ids():
            r = data.load(rid)
            if r is not None:
                out.append({"run_id": rid, "strategies": r.strategies, "synthetic": r.synthetic, **r.summary()})
        return JSONResponse(jsonable(out))

    @app.get("/api/runs/{run_id}/metrics")
    def api_metrics(run_id: str) -> JSONResponse:
        r = data.load(run_id, strict=True)
        return JSONResponse(jsonable(r.metrics) if r else {"error": "unknown run"}, status_code=200 if r else 404)

    @app.get("/api/runs/{run_id}/equity")
    def api_equity(run_id: str) -> JSONResponse:
        r = data.load(run_id, strict=True)
        return JSONResponse(jsonable(r.equity.to_dicts()) if r else {"error": "unknown run"}, status_code=200 if r else 404)

    @app.get("/api/live")
    def api_live() -> JSONResponse:
        state = meta.all_state().get("live") if meta is not None else None
        return JSONResponse(jsonable(state or {}))

    return app


def serve(settings: Any, meta: Any = None) -> None:
    import uvicorn

    uvicorn.run(create_app(settings, meta), host=settings.dashboard.host, port=settings.dashboard.port, log_level="warning")


def static_export(settings: Any, out_path: str | Path, run_id: str | None = None, meta: Any = None) -> Path:
    """Single self-contained HTML file with every page as a tab (no server needed)."""
    data = V.DashboardData(settings.paths.resolve("reports_dir"), settings.paths.events_dir, meta)
    r = data.load(run_id)
    rid = r.run_id if r else None
    cfg = settings.dashboard
    state = meta.all_state().get("live") if meta is not None else None
    bodies = {
        "overview": V.page_overview(r), "trades": V.page_trades(r, min(cfg.max_table_rows, 1500)), "equity": V.page_equity(r),
        "drawdown": V.page_drawdown(r), "heatmaps": V.page_heatmaps(r),
        "wallets": V.page_wallets(data.artefact(rid, "wallets.parquet"), 1500),
        "tokens": V.page_tokens(r, data, data.artefact(rid, "tokens.parquet"), None).split("</form>", 1)[-1],
        "features": V.page_features(data.importance()), "strategies": V.page_strategies(data),
        "live": V.page_live(state, cfg.live_refresh_ms, server=False),
    }
    nav = "".join(f"<button data-t='{k}' class='{'on' if i == 0 else ''}'>{label}</button>" for i, (k, label) in enumerate(V.PAGES))
    tabs = "".join(f"<section class='tab {'on' if i == 0 else ''}' id='t-{k}'>{bodies[k]}</section>" for i, (k, _) in enumerate(V.PAGES))
    js = ("document.querySelectorAll('nav.tabs button').forEach(function(b){b.addEventListener('click',function(){"
          "document.querySelectorAll('nav.tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');"
          "document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));document.getElementById('t-'+b.dataset.t).classList.add('on');"
          "window.dispatchEvent(new Event('resize'));});});")
    title = f"Pump.fun HFT Dashboard — {', '.join(r.strategies) if r else 'no runs'}"
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Pump.fun HFT Dashboard</title><style>{C.PAGE_CSS}</style><script>{_plotly_js()}</script></head><body><main>"
           f"<h1>{html.escape(title)}</h1><p class='sub'>Static export of the local dashboard · run {html.escape(rid or '-')}</p>"
           f"<nav class='tabs'>{nav}</nav>{tabs}</main><script>{js}{C.TABLE_JS}{C.THEME_JS}</script></body></html>")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p
