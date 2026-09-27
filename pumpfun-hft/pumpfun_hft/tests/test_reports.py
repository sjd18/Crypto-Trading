"""Reports and dashboard: every output format, strict JSON, HTML escaping of untrusted strings
(token names and wallet labels come from the chain), all ten dashboard pages, the JSON API and
the single-file static export."""

from __future__ import annotations

import dataclasses
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from pumpfun_hft.analytics.montecarlo import trade_monte_carlo
from pumpfun_hft.analytics.report import ReportGenerator, exit_categories, metrics_frame
from pumpfun_hft.backtester.results import BacktestResult
from pumpfun_hft.dashboard.app import create_app, static_export
from pumpfun_hft.dashboard.views import PAGES, DashboardData
from pumpfun_hft.database.meta import MetaStore

# A token name / label an attacker could put on chain: a script element and an event handler.
EVIL = "<script>alert('pwn')</script><img src=x onerror=alert(1)>"


class _Audit(HTMLParser):
    """Collects what a browser would execute: script bodies and inline event-handler attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self.handlers: list[tuple[str, str, str]] = []
        self.external: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._in_script = True
            self.scripts.append("")
        for k, v in attrs:
            if k.startswith("on"):
                self.handlers.append((tag, k, v or ""))
            if k in ("src", "href") and v and v.startswith(("http://", "https://", "//")):
                self.external.append(v)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.scripts[-1] += data


def audit(doc: str) -> _Audit:
    a = _Audit()
    a.feed(doc)
    a.close()
    return a


def assert_safe(doc: str) -> None:
    """No injected script element or event handler, and nothing loaded from the network."""
    a = audit(doc)
    assert not any("alert(" in s and len(s.strip()) < 40 for s in a.scripts), "an injected <script> element was parsed"
    assert not any("alert" in v for _, _, v in a.handlers), a.handlers
    assert not a.external, a.external  # fully offline: Plotly is inlined or served locally


def strict_json(text: str) -> Any:
    def refuse(c: str) -> Any:
        raise ValueError(f"non-standard JSON constant {c}")

    return json.loads(text, parse_constant=refuse)


def with_evil_exit_reasons(r: BacktestResult) -> BacktestResult:
    t = r.trades.with_columns(pl.when(pl.int_range(pl.len()) < 5).then(pl.lit(EVIL)).otherwise(pl.col("exit_reason")).alias("exit_reason"))
    return dataclasses.replace(r, trades=t)


# --------------------------------------------------------------------------- report generator
@pytest.fixture(scope="module")
def mc(backtest_result: BacktestResult, settings: Any) -> Any:
    r = backtest_result
    return trade_monte_carlo(r.trades, r.initial_capital_sol, r.metrics["span_days"], settings.montecarlo, 500.0, n_sims=300)


def test_report_writes_every_format(tmp_path: Path, backtest_result: BacktestResult, mc: Any) -> None:
    r = backtest_result
    out = ReportGenerator(r, mc).generate(tmp_path / "rep")
    assert set(out) == {"html", "pdf", "csv", "json"}
    doc = out["html"].read_text(encoding="utf-8")
    for heading in ("Equity and drawdown", "Trade distribution", "Timing", "Exits", "Largest winners", "Costs and execution quality",
                    "Monte Carlo", "All trades", "All metrics"):
        assert heading in doc, heading
    assert "Synthetic data." in doc  # the synthetic-data banner is always shown for synthetic runs
    assert "backtest spans less than 30 days" in doc  # short-sample warning replaces a meaningless CAGR
    assert_safe(doc)
    assert out["pdf"].read_bytes()[:5] == b"%PDF-" and out["pdf"].stat().st_size > 20_000
    csvs = {p.stem: p for p in out["csv"]}
    assert {"trades", "fills", "equity", "signals"} <= set(csvs)
    assert pl.read_csv(csvs["trades"]).height == r.trades.height
    payload = strict_json(out["json"].read_text())  # NaN metrics (e.g. CAGR on 3 hours) are written as null
    assert payload["run_id"] == r.run_id and payload["metrics"]["n_trades"] == r.metrics["n_trades"]
    assert payload["metrics"]["cagr"] is None and payload["monte_carlo"]["n_sims"] == 300


def test_report_escapes_untrusted_text(tmp_path: Path, backtest_result: BacktestResult) -> None:
    r = with_evil_exit_reasons(backtest_result)
    doc = ReportGenerator(r, title=EVIL).html(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in doc  # shown as text in the tables and the title
    assert_safe(doc)


def test_report_without_trades(tmp_path: Path, backtest_result: BacktestResult) -> None:
    r = dataclasses.replace(backtest_result, trades=backtest_result.trades.clear(), fills=backtest_result.fills.clear(),
                            signals=pl.DataFrame())
    out = ReportGenerator(r).generate(tmp_path / "empty", ["html", "json", "csv", "pdf"])
    assert out["html"].exists() and out["pdf"].exists()
    assert strict_json(out["json"].read_text())["monte_carlo"] is None
    assert {p.stem for p in out["csv"]} == {"equity", "monthly_returns"}  # empty tables are skipped, not written as header-only


def test_metrics_table_and_exit_categories(backtest_result: BacktestResult) -> None:
    r = backtest_result
    mf = metrics_frame(r.metrics)
    scalars = [k for k, v in r.metrics.items() if not isinstance(v, (dict, list))]
    assert r.metrics["cagr_extrapolated"] and "cagr_raw" not in mf["metric"].to_list()  # 3 hours compounded to a year is noise
    assert mf.height == len(scalars) - 1
    ex = exit_categories(r.trades)
    assert int(ex["trades"].sum()) == r.trades.height
    assert float(ex["pnl_sol"].sum()) == pytest.approx(float(r.trades["pnl_sol"].sum()), abs=1e-9)
    assert ex["exit"].n_unique() < 30  # free-text reasons collapse into a handful of categories


# --------------------------------------------------------------------------- dashboard
def _save_run(root: Path, r: BacktestResult, name: str | None = None) -> Path:
    d = r.save(root / "runs" / (name or r.run_id))
    eng = r.engine  # type: ignore[attr-defined]
    w = eng.wallets.to_frame(min_trades=2, now_ms=r.end_ms)
    w = w.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.lit(EVIL)).otherwise(pl.col("labels")).alias("labels"))
    w.write_parquet(d / "wallets.parquet")
    toks = [{"mint": m, "name": EVIL if i == 0 else st.name, "symbol": EVIL if i == 0 else st.symbol, "creator": st.creator,
             "created_ms": st.created_ms, "n_trades": st.n_trades, "ath_multiple": st.ath_multiple, "max_dd_pct": st.max_dd_pct,
             "max_liq_dd_pct": st.max_liq_dd_pct, "migrated": st.migrated, "outcome": "unresolved"}
            for i, (m, st) in enumerate(eng.market.tokens.items())]
    pl.DataFrame(toks, infer_schema_length=None).write_parquet(d / "tokens.parquet")
    return d


@pytest.fixture()
def dash(tmp_settings: Any, backtest_result: BacktestResult, events: pl.DataFrame) -> Any:
    s = tmp_settings
    reports = s.paths.resolve("reports_dir")
    _save_run(reports, backtest_result)
    ml = reports / "ml" / "lightgbm-rug-test"
    ml.mkdir(parents=True)
    pl.DataFrame({"feature": ["sell_pressure", EVIL], "native": [0.6, 0.4], "permutation": [0.05, 0.01],
                  "shap": [None, None]}, schema={"feature": pl.String, "native": pl.Float64, "permutation": pl.Float64,
                                                 "shap": pl.Float64}).write_parquet(ml / "importance.parquet")
    ev_dir = s.paths.events_dir / "date=2026-01-01"
    ev_dir.mkdir(parents=True)
    events.head(20_000).write_parquet(ev_dir / "part-0.parquet")
    meta = MetaStore(s.paths.resolve("sqlite_file"))
    meta.set_state("live", {"equity_sol": 10.25, "events_processed": 1234, "pending_orders": 1, "queue": 0,
                            "positions": [{"mint": "M" * 44, "tokens": 5, "strategy": EVIL}],
                            "latency": {"live.event_to_dispatch": {"count": 10, "p50": 0.4, "p90": 0.8, "p99": 1.2, "max": 2.0,
                                                                   "budget_ms": 100.0, "breaches": 0}},
                            "risk": {"breakers": {"daily_loss": {"active": False, "trips": 0, "reason": ""}}},
                            "unrealised": float("nan")})
    client = TestClient(create_app(s, meta))
    return client, s, meta, backtest_result


def test_every_dashboard_page_renders(dash: Any) -> None:
    client, _, _, r = dash
    assert len(PAGES) == 10
    for key, label in PAGES:
        resp = client.get(f"/{key}", params={"run": r.run_id})
        assert resp.status_code == 200, key
        doc = resp.text
        assert label in doc and "No runs yet" not in doc, key
        assert_safe(doc)
    assert client.get("/").status_code == 200
    assert "&lt;script&gt;" in client.get("/tokens").text  # the malicious token name is displayed as text
    assert client.get("/no-such-page").status_code == 404
    js = client.get("/static/plotly.min.js")
    assert js.status_code == 200 and len(js.content) > 1_000_000  # Plotly is served locally, never from a CDN


def test_token_explorer_with_a_selected_mint(dash: Any) -> None:
    client, _, _, r = dash
    mint = r.trades["mint"][0]
    doc = client.get("/tokens", params={"run": r.run_id, "mint": mint}).text
    assert f"{mint} — price with our fills" in doc
    assert_safe(client.get("/tokens", params={"mint": EVIL}).text)  # query parameters are echoed escaped


def test_dashboard_api(dash: Any) -> None:
    client, _, _, r = dash
    runs = strict_json(client.get("/api/runs").text)
    assert [x["run_id"] for x in runs] == [r.run_id] and runs[0]["n_trades"] == r.metrics["n_trades"]
    resp = client.get(f"/api/runs/{r.run_id}/metrics")
    m = strict_json(resp.text)  # strict JSON even though CAGR / Calmar are NaN on a 3-hour run
    assert resp.status_code == 200 and m["cagr"] is None and m["n_trades"] == r.metrics["n_trades"]
    eq = strict_json(client.get(f"/api/runs/{r.run_id}/equity").text)
    assert len(eq) == r.equity.height
    assert client.get("/api/runs/does-not-exist/metrics").status_code == 404  # never silently the latest run
    assert client.get("/api/runs/does-not-exist/equity").status_code == 404
    live = strict_json(client.get("/api/live").text)
    assert live["value"]["events_processed"] == 1234 and live["value"]["unrealised"] is None


def test_run_ids_are_escaped(dash: Any) -> None:
    client, s, _, r = dash
    hostile = "x' autofocus onfocus='alert(1)"  # a directory name is just another untrusted string
    shutil.copytree(s.paths.resolve("reports_dir") / "runs" / r.run_id, s.paths.resolve("reports_dir") / "runs" / hostile)
    for page in ("overview", "trades", "tokens"):
        assert_safe(client.get(f"/{page}").text)


def test_dashboard_data_reloads_a_resaved_run(tmp_settings: Any, backtest_result: BacktestResult) -> None:
    reports = tmp_settings.paths.resolve("reports_dir")
    d = _save_run(reports, backtest_result)
    data = DashboardData(reports)
    assert data.load(None).metrics["n_trades"] == backtest_result.metrics["n_trades"]
    changed = dataclasses.replace(backtest_result, metrics={**backtest_result.metrics, "n_trades": -1})
    import os
    import time

    changed.save(d)
    t = time.time() + 5
    os.utime(d / "result.json", (t, t))
    assert data.load(backtest_result.run_id).metrics["n_trades"] == -1
    assert data.load("unknown", strict=True) is None and data.load("unknown") is not None


def test_static_export(dash: Any, tmp_path: Path) -> None:
    _, s, meta, r = dash
    p = static_export(s, tmp_path / "dashboard.html", None, meta)
    doc = p.read_text(encoding="utf-8")
    for key, _ in PAGES:
        assert f"id='t-{key}'" in doc, key
    assert r.run_id in doc and "Plotly" in doc
    assert_safe(doc)
