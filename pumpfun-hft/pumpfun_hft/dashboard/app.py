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
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from pumpfun_hft.analytics import charts as C
from pumpfun_hft.core.config import PROJECT_ROOT
from pumpfun_hft.dashboard import builder as B
from pumpfun_hft.dashboard import trading_views as TV
from pumpfun_hft.dashboard import views as V
from pumpfun_hft.execution import sessions as SS
from pumpfun_hft.optimizer import tuned as TU
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


def create_app(settings: Any, meta: Any = None, config: str | Path | None = None, runner: B.JobRunner | None = None,
               health: B.HealthProbe | None = None, opt_runner: B.JobRunner | None = None) -> FastAPI:
    """Build the FastAPI application bound to ``settings.paths``. ``config`` is the --config file the
    strategy builder passes on to the sessions it starts."""
    data = V.DashboardData(settings.paths.resolve("reports_dir"), settings.paths.events_dir, meta)
    app = FastAPI(title="pumpfun-hft dashboard", docs_url="/api/docs")
    cfg = settings.dashboard
    runner = runner or B.JobRunner(settings, PROJECT_ROOT, config)
    opt_runner = opt_runner or B.JobRunner(settings, PROJECT_ROOT, config)
    real_root = B.dataset_root("real", settings)
    sessions_dir = SS.sessions_root(settings)
    studies_dir = real_root / "reports" / "optimize"
    health = health or B.HealthProbe()
    allowed_hosts = {f"{h}:{cfg.port}" for h in (cfg.host, "127.0.0.1", "localhost")}
    stores: dict[str, Any] = {}

    def guard(request: Request) -> None:
        """Endpoints that start or stop sessions answer only this dashboard's own page: the custom header
        makes a cross-site request need a CORS preflight (which is never granted), and the Host check
        stops DNS-rebinding pages that pretend to be 127.0.0.1."""
        if request.headers.get("x-pumpfun-dashboard") != "1" or request.headers.get("host") not in allowed_hosts:
            raise HTTPException(403, "forbidden")
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1] not in allowed_hosts:
            raise HTTPException(403, "forbidden")

    def session_state() -> Any:
        """Live snapshot of the builder's newest session (its data set), else this dashboard's data set."""
        p = runner.meta_path()
        if p is not None and p.exists():
            from pumpfun_hft.database.meta import MetaStore

            key = str(p)
            if key not in stores:
                stores[key] = MetaStore(p)
            st = stores[key].all_state().get("live")
            if st:
                return st
        return meta.all_state().get("live") if meta is not None else None

    @app.get("/static/plotly.min.js")
    def plotly_js() -> Response:
        return Response(_plotly_js(), media_type="application/javascript", headers={"Cache-Control": "max-age=86400"})

    def render(page: str, run: str | None, mint: str | None = None, session: str | None = None, strategy: str | None = None) -> str:
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
        elif page == "builder":
            body = V.page_builder(B.strategy_catalogue(settings), session_state(), cfg.live_refresh_ms, server=True)
        elif page == "sessions":
            body = sessions_page(session, server=True)
        elif page == "params":
            body = params_page(strategy)
        else:
            body = f"<p>Unknown page <code>{html.escape(page)}</code>.</p>"
        return shell(dict(V.PAGES).get(page, page), body, page, data.run_ids(), rid)

    @app.get("/", response_class=HTMLResponse)
    def index(run: str | None = None) -> str:
        return render("overview", run)

    @app.get("/{page}", response_class=HTMLResponse)
    def page(page: str, run: str | None = None, mint: str | None = Query(default=None), session: str | None = None,
             strategy: str | None = None) -> HTMLResponse:
        return HTMLResponse(render(page, run or None, mint, session or None, strategy or None), status_code=200 if page in PAGE_KEYS else 404)

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

    @app.get("/api/live/html", response_class=HTMLResponse)
    def api_live_html() -> str:
        """The Live monitor body for the newest snapshot (the page swaps it in on every refresh)."""
        return V.live_body(meta.all_state().get("live") if meta is not None else None, server=True)

    # ---------------------------------------------------------------- strategy builder
    @app.post("/api/builder/command")
    async def api_builder_command(request: Request) -> JSONResponse:
        """The PowerShell line for a builder form (nothing is run)."""
        guard(request)
        try:
            spec = B.parse_spec(await request.json(), B.strategy_names())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)})
        return JSONResponse({"powershell": B.powershell_line(spec), "runnable": spec.mode != "hftl"})

    @app.post("/api/jobs")
    async def api_job_start(request: Request) -> JSONResponse:
        guard(request)
        try:
            spec = B.parse_spec(await request.json(), B.strategy_names())
            runner.start(spec)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(jsonable(runner.status()))

    @app.post("/api/jobs/stop")
    def api_job_stop(request: Request) -> JSONResponse:
        guard(request)
        return JSONResponse({"stopping": runner.stop()})

    @app.get("/api/jobs/current")
    def api_job_current(since: int = 0) -> JSONResponse:
        return JSONResponse(jsonable(runner.status(since)))

    @app.get("/api/builder/live/html", response_class=HTMLResponse)
    def api_builder_live() -> str:
        return V.live_body(session_state(), server=True)

    # ---------------------------------------------------------------- session archive
    def sessions_page(session: str | None, server: bool) -> str:
        sessions = SS.list_sessions(sessions_dir)
        info = next((x for x in sessions if x["session_id"] == session), sessions[0] if sessions else None)
        if info is None:
            return TV.page_sessions(sessions, None, {}, pl.DataFrame(), server)
        frames = {n: SS.load_frame(sessions_dir, info["session_id"], n) for n in ("trades", "equity", "signals")}
        return TV.page_sessions(sessions, info, frames, SS.strategy_breakdown(info, frames["trades"]), server)

    @app.get("/api/sessions/{session_id}/{name}.csv")
    def api_session_csv(session_id: str, name: str) -> Response:
        if name not in ("trades", "fills", "equity", "signals") or session_id not in {x["session_id"] for x in SS.list_sessions(sessions_dir)}:
            raise HTTPException(404, "unknown session or table")
        df = SS.load_frame(sessions_dir, session_id, name)
        return Response(df.write_csv() if df.width else "", media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{session_id}-{name}.csv"'})

    # ---------------------------------------------------------------- strategy parameters
    def load_studies(strategy: str | None = None) -> list[dict[str, Any]]:
        import json as _json

        out = []
        for f in studies_dir.glob("*/study.json") if studies_dir.is_dir() else []:
            try:
                st = _json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if strategy is None or st.get("strategy") == strategy:
                st.setdefault("created_ms", int(f.stat().st_mtime * 1000))
                out.append(st)
        return sorted(out, key=lambda x: int(x.get("created_ms") or 0), reverse=True)

    def params_page(strategy: str | None) -> str:
        import os as _os

        cat = B.strategy_catalogue(settings)
        names = [c["name"] for c in cat]
        if strategy not in names:
            active = [n for n in settings.strategy.active if n in names]
            strategy = active[0] if active else names[0]
        return TV.page_params(cat, strategy, load_studies(strategy), TU.load_tuned(real_root),
                              SS.strategy_evidence(sessions_dir, strategy), dict(settings.strategy.params.get(strategy, {})),
                              settings.optimizer, max(1, (_os.cpu_count() or 2) - 1), server=True, window_events=window_events())

    window_cache: dict[str, Any] = {}

    def window_events() -> dict[int, int]:
        """Recorded real events in each tuning window (cached for a minute: a scan of the timestamps only)."""
        import time as _time

        if window_cache.get("t", 0) > _time.monotonic() - 60:
            return window_cache["v"]
        out: dict[int, int] = {}
        files = list((real_root / "events").glob("date=*/*.parquet"))
        if files:
            ts = pl.scan_parquet([str(f) for f in files]).select("ts_ms").collect()["ts_ms"]
            hi = int(ts.max()) if ts.len() else 0
            for h in (6, 12, 24, 48, 168):
                out[h] = int((ts >= hi - h * 3_600_000).sum())
            out[0] = ts.len()
        window_cache.update(t=_time.monotonic(), v=out)
        return out

    @app.post("/api/optimize/command")
    async def api_optimize_command(request: Request) -> JSONResponse:
        guard(request)
        try:
            spec = B.parse_optimize(await request.json(), B.strategy_names())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)})
        return JSONResponse({"powershell": B.optimize_line(spec)})

    @app.post("/api/optimize")
    async def api_optimize_start(request: Request) -> JSONResponse:
        guard(request)
        try:
            opt_runner.start_optimize(B.parse_optimize(await request.json(), B.strategy_names()))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(jsonable(opt_runner.status()))

    @app.post("/api/optimize/stop")
    def api_optimize_stop(request: Request) -> JSONResponse:
        guard(request)
        return JSONResponse({"stopping": opt_runner.stop()})

    @app.get("/api/optimize/current")
    def api_optimize_current(since: int = 0) -> JSONResponse:
        return JSONResponse(jsonable(opt_runner.status(since)))

    @app.post("/api/tuned")
    async def api_tuned_save(request: Request) -> JSONResponse:
        """Save a study's selected parameters for hfts. Only studies on the real data set qualify."""
        guard(request)
        body = await request.json()
        st = next((x for x in load_studies() if x.get("study_id") == str(body.get("study_id") or "")), None)
        if st is None:
            return JSONResponse({"error": "unknown study"}, status_code=404)
        if st.get("dataset") not in ("real", None):
            return JSONResponse({"error": "only studies on the real (recorded) data can be used for hfts"}, status_code=400)
        known = B.strategy_names()
        if st.get("strategy") not in known or not st.get("selected_params"):
            return JSONResponse({"error": "the study has no usable parameters"}, status_code=400)
        entry = TU.save_tuned(real_root, st["strategy"], {
            "params": st["selected_params"], "study_id": st["study_id"], "objective": st.get("objective"),
            "method": st.get("method"), "val_metrics": st.get("selected_val_metrics"),
            "baseline_val_metrics": st.get("baseline_val_metrics"), "data_end_ms": st.get("data_end_ms")})
        return JSONResponse(jsonable({"saved": st["strategy"], **entry}))

    @app.post("/api/tuned/remove")
    async def api_tuned_remove(request: Request) -> JSONResponse:
        guard(request)
        body = await request.json()
        return JSONResponse({"removed": TU.remove_tuned(real_root, str(body.get("strategy") or ""))})

    @app.get("/api/health")
    def api_health() -> JSONResponse:
        job = runner.status()["job"]
        return JSONResponse(jsonable(health.check(session_state(), job, max(5 * cfg.live_refresh_ms, 15_000))))

    return app


def serve(settings: Any, meta: Any = None, config: str | Path | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(settings, meta, config), host=settings.dashboard.host, port=settings.dashboard.port, log_level="warning")


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
        "builder": V.page_builder(B.strategy_catalogue(settings), state, cfg.live_refresh_ms, server=False),
        "sessions": "<p class='muted'>The session breakdown reads the session archive: open it on the dashboard server "
                    "(<code>hfts dashboard</code>).</p>",
        "params": TV.page_params([], "", [], {}, {}, {}, settings.optimizer, 1, server=False),
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
