"""Command-line interface (Typer + Rich).

Usage: ``python -m pumpfun_hft.main [--config my.yaml] [--set key=value ...] COMMAND [options]``

Research:   synth · collect-history · verify-data · backtest · optimize · walkforward · montecarlo ·
            report · train-model · wallets · query
Live:       stream · paper · paper-replay · live (requires app.mode=live AND --confirm-live) · latency-probe
Tooling:    init · check-config · dashboard · dashboard-export · update-idl · docs
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml
from rich.console import Console
from rich.table import Table

from pumpfun_hft.core.config import PROJECT_ROOT, Secrets, Settings, load_settings
from pumpfun_hft.utils import jsonutil
from pumpfun_hft.utils.logging import get_logger, setup_logging
from pumpfun_hft.utils.timeutil import ms_to_iso, parse_iso_ms

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pump.fun quantitative research & execution platform")
console = Console()
log = get_logger("system")
STATE: dict[str, Any] = {}


def _parse_sets(sets: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in sets or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


@app.callback()
def main(config: Path = typer.Option(None, "--config", "-c", help="User YAML merged over configs/default.yaml"),
         set_: list[str] = typer.Option(None, "--set", "-s", help="Dotted override, e.g. backtest.initial_capital_sol=5")) -> None:
    overrides = _parse_sets(set_)
    settings = load_settings(config, overrides)
    STATE.update(settings=settings, config_path=config, overrides=overrides)
    lg = settings.logging
    setup_logging(lg.level, settings.paths.resolve("logs_dir"), lg.max_bytes, lg.backup_count, lg.console, lg.channels)


def S() -> Settings:
    return STATE["settings"]


def _meta() -> Any:
    from pumpfun_hft.database.meta import MetaStore

    return MetaStore(S().paths.resolve("sqlite_file"))


def _store() -> Any:
    from pumpfun_hft.collectors.storage import ParquetEventStore

    return ParquetEventStore(S().paths.events_dir, _meta())


def _metadata_path() -> Path:
    return S().paths.metadata_dir / "tokens.parquet"


def _load_events(start: str | None, end: str | None) -> pl.DataFrame:
    store = _store()
    df = store.read(parse_iso_ms(start) if start else None, parse_iso_ms(end) if end else None)
    if df.is_empty():
        console.print("[yellow]The event store is empty. Generate data with `synth` or collect with `collect-history`.[/]")
        raise typer.Exit(1)
    return df


def _is_synthetic() -> bool:
    return (S().paths.metadata_dir / "synthetic_truth.parquet").exists()


def _print_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    t = Table(title=title, show_header=True, header_style="bold")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k in ("total_return", "pnl_sol", "sharpe", "sortino", "omega", "calmar", "recovery_factor", "max_drawdown", "pct_time_underwater", "n_trades",
              "win_rate", "profit_factor", "expectancy_sol", "avg_r_multiple", "kelly_fraction", "median_trade_sol", "avg_mae",
              "avg_mfe", "fill_rate", "failed_rate", "avg_latency_ms", "avg_slippage_bps"):
        v = metrics.get(k)
        t.add_row(k, "–" if v is None else (f"{v:,.4f}" if isinstance(v, float) else str(v)))
    console.print(t)


# ============================================================================ tooling
@app.command()
def init() -> None:
    """Create data/log/report directories and a .env from .env.example (if missing)."""
    s = S()
    for attr in ("data_dir", "logs_dir", "reports_dir", "models_dir"):
        s.paths.resolve(attr).mkdir(parents=True, exist_ok=True)
    s.paths.events_dir.mkdir(parents=True, exist_ok=True)
    s.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    env, example = PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        console.print(f"Created {env} from .env.example — fill in your endpoints and keys.")
    console.print("[green]Initialised.[/]")


@app.command("check-config")
def check_config() -> None:
    """Validate configuration and show which secrets are present (values are never printed)."""
    s = S()
    secrets = Secrets.load()
    console.print(f"config fingerprint [bold]{s.fingerprint()}[/] · mode [bold]{s.app.mode}[/] · metis mode [bold]{s.network.metis.mode}[/]")
    t = Table(title="Secrets (.env)")
    t.add_column("name")
    t.add_column("present")
    for k, v in secrets.summary().items():
        t.add_row(k, "yes" if v else "no")
    console.print(t)
    if s.network.metis.mode == "public":
        console.print(f"[yellow]Mode A uses {s.network.metis.public_url}, which QuickNode shuts down on "
                      f"{s.network.metis.public_sunset_date}. It only serves /pump-fun/swap and charges a platform fee.[/]")
    console.print(f"active strategies: {s.strategy.active} · overlay: {s.strategy.exit_overlay}")


@app.command("update-idl")
def update_idl() -> None:
    """Refresh bundled IDLs from github.com/pump-fun/pump-public-docs."""
    import httpx

    base = "https://raw.githubusercontent.com/pump-fun/pump-public-docs/main/idl/"
    out = S().paths.resolve("idl_dir")
    for name in ("pump.json", "pump_amm.json", "pump_fees.json"):
        r = httpx.get(base + name, timeout=30)
        r.raise_for_status()
        json.loads(r.text)
        (out / name).write_text(r.text)
        console.print(f"updated {name} ({len(r.text):,} bytes)")


@app.command()
def docs() -> None:
    """Regenerate docs/MODULES.md from every package docstring."""
    from pumpfun_hft.utils.docgen import generate_module_docs

    p = generate_module_docs(PROJECT_ROOT / "docs" / "MODULES.md")
    console.print(f"wrote {p}")


# ============================================================================ data
@app.command()
def synth(hours: float = typer.Option(None, help="Override synthetic.duration_hours"),
          seed: int = typer.Option(None, help="Override synthetic.seed"),
          clear: bool = typer.Option(True, help="Remove existing events first")) -> None:
    """Generate a synthetic market into the event store (for demos / tests; results are not real)."""
    from pumpfun_hft.collectors.storage import write_metadata
    from pumpfun_hft.collectors.synthetic import SyntheticMarket

    ov = {}
    if hours is not None:
        ov["synthetic.duration_hours"] = hours
    if seed is not None:
        ov["synthetic.seed"] = seed
    s = load_settings(STATE.get("config_path"), {**_settings_overrides(), **ov}) if ov else S()
    if clear and s.paths.events_dir.exists():
        shutil.rmtree(s.paths.events_dir)
    t0 = time.perf_counter()
    data = SyntheticMarket(s).generate()
    store = _store()
    store.write(data.events)
    store.compact()
    s.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    if (s.paths.metadata_dir / "tokens.parquet").exists() and clear:
        (s.paths.metadata_dir / "tokens.parquet").unlink()
    write_metadata(data.metadata, s.paths.metadata_dir / "tokens.parquet")
    data.truth.write_parquet(s.paths.metadata_dir / "synthetic_truth.parquet")
    console.print(f"[green]synthetic market:[/] {data.events.height:,} events, {data.truth.height} tokens, "
                  f"{int(data.truth['complete_ms'].is_not_null().sum())} graduations in {time.perf_counter() - t0:.1f}s")


def _settings_overrides() -> dict[str, Any]:
    """Global ``--set`` overrides given on the command line (re-applied when a command reloads settings)."""
    return dict(STATE.get("overrides") or {})


@app.command("collect-history")
def collect_history(max_signatures: int = typer.Option(None), mode: str = typer.Option("backfill", help="backfill | catchup"),
                    retry_pending: bool = typer.Option(True)) -> None:
    """Download Pump program history via RPC into the Parquet store (resumable)."""
    from pumpfun_hft.api.http import build_client
    from pumpfun_hft.api.rpc import SolanaRpcClient
    from pumpfun_hft.collectors.historical import HistoricalCollector
    from pumpfun_hft.collectors.sol_price import provider_from_settings
    from pumpfun_hft.core.events import EventDecoder

    s, secrets = S(), Secrets.load()
    url = secrets.get("solana_rpc_url")
    if not url:
        console.print("[red]SOLANA_RPC_URL is not set in .env[/]")
        raise typer.Exit(1)

    async def run() -> None:
        http = build_client("rpc", "", s.network.rpc.rate_limit, s.network, s.network.rpc.timeout_s)
        rpc = SolanaRpcClient(http, url, s.network.rpc.commitment)
        coll = HistoricalCollector(rpc, EventDecoder.from_settings(s), _store(), _meta(), s.collector, s.protocol.pump_program_id,
                                   provider_from_settings(s), s.simulation.slot_ms)
        stats = await coll.run(max_signatures, mode)
        if retry_pending:
            stats_r = await coll.retry_pending()
            console.print(f"recovered {stats_r} pending transactions")
        await http.aclose()
        console.print(stats)

    asyncio.run(run())
    _store().compact()


@app.command("verify-data")
def verify_data(gap_slots: int = typer.Option(None, help="Report slot gaps wider than this")) -> None:
    """Verify Parquet checksums against the manifest and report slot gaps."""
    store = _store()
    bad = store.verify()
    console.print(f"files: {store.stats()} · checksum mismatches: {len(bad)}")
    for b in bad[:20]:
        console.print(f"  [red]{b}[/]")
    gaps = store.detect_gaps(gap_slots or S().collector.gap_slot_threshold)
    console.print(f"slot gaps > threshold: {gaps.height}")
    if gaps.height:
        console.print(gaps.head(20))


# ============================================================================ research
def _run_backtest(strategies: list[str], start: str | None, end: str | None, seed: int | None, run_id: str | None = None) -> Any:
    from pumpfun_hft.backtester.engine import BacktestEngine
    from pumpfun_hft.backtester.replay import DataSource, load_metadata

    s = S()
    events = _load_events(start, end)
    meta = load_metadata(_metadata_path())
    eng = BacktestEngine(s, DataSource(frame=events), strategies or None, metadata=meta, seed=seed, run_id=run_id,
                         synthetic=_is_synthetic())
    res = eng.run()
    return eng, res


def _save_run(eng: Any, res: Any) -> Path:
    s = S()
    d = s.paths.resolve("reports_dir") / "runs" / res.run_id
    res.save(d)
    wallets = eng.wallets.to_frame(min_trades=2, now_ms=res.end_ms)
    wallets.write_parquet(d / "wallets.parquet")
    toks = []
    outcomes = eng.resolver.outcomes
    for mint, st in eng.market.tokens.items():
        o = outcomes.get(mint)
        toks.append({"mint": mint, "name": st.name, "symbol": st.symbol, "creator": st.creator, "created_ms": st.created_ms,
                     "sector": st.sector, "n_trades": st.n_trades, "ath_multiple": st.ath_multiple, "max_dd_pct": st.max_dd_pct,
                     "max_liq_dd_pct": st.max_liq_dd_pct,
                     "migrated": st.migrated, "creator_sold_pct": st.creator_sold_pct, "outcome": o.label if o else "unresolved"})
    tokens = pl.DataFrame(toks, infer_schema_length=None)
    tokens.write_parquet(d / "tokens.parquet")
    try:
        _meta().register_run(res.run_id, "backtest", ",".join(res.strategies), res.config_hash, res.data_hash, str(d), res.summary())
    except Exception:  # noqa: BLE001 - registry is best effort
        pass
    _warehouse_save(res, wallets, tokens)
    return d


def _warehouse_save(res: Any, wallets: pl.DataFrame, tokens: pl.DataFrame) -> None:
    """Persist the run, its trades / fills / equity and the wallet and token tables to DuckDB."""
    from pumpfun_hft.database.warehouse import Warehouse

    try:
        wh = Warehouse(S().paths.resolve("duckdb_file"), S().paths.events_dir)
    except Exception as exc:  # noqa: BLE001 - e.g. the database is locked by another process
        log.warning("warehouse unavailable", extra={"data": {"error": repr(exc)[:200]}})
        return
    try:
        wh.save_run(res.run_id, "backtest", ",".join(res.strategies), res.config_hash, res.data_hash, res.summary(),
                    res.trades, res.fills, res.equity)
        if not wallets.is_empty():
            wh.upsert("wallets", wallets, key=["address"])
        if not tokens.is_empty():
            t = tokens.rename({"max_dd_pct": "max_drawdown_pct", "max_liq_dd_pct": "max_liq_drawdown_pct"})
            cols = [c[0] for c in wh.con.execute("DESCRIBE tokens").fetchall()]
            wh.upsert("tokens", t.select([c for c in cols if c in t.columns]), key=["mint"])
    finally:
        wh.close()


@app.command()
def backtest(strategy: list[str] = typer.Option(None, "--strategy", help="Strategy name (repeatable); default strategy.active"),
             start: str = typer.Option(None, help="ISO start (UTC)"), end: str = typer.Option(None, help="ISO end (UTC)"),
             seed: int = typer.Option(None), report: bool = typer.Option(True, help="Generate HTML/PDF/CSV/JSON report"),
             montecarlo: bool = typer.Option(True, help="Include trade-level Monte Carlo in the report")) -> None:
    """Run an event-driven backtest on the stored events; saves results and the report."""
    eng, res = _run_backtest(list(strategy or []), start, end, seed)
    d = _save_run(eng, res)
    _print_metrics(res.metrics, f"{res.run_id} ({', '.join(res.strategies)})")
    console.print(f"{res.n_events:,} events in {res.elapsed_s:.1f}s ({res.events_per_second:,.0f}/s) · saved to {d}")
    if res.synthetic:
        console.print("[yellow]Synthetic data: these numbers exercise the pipeline and say nothing about live profitability.[/]")
    if report:
        mc = None
        if montecarlo and res.trades.height:
            from pumpfun_hft.analytics.montecarlo import trade_monte_carlo

            mc = trade_monte_carlo(res.trades, res.initial_capital_sol, res.metrics.get("span_days", 1.0), S().montecarlo,
                                   res.metrics.get("avg_latency_ms") or 500.0)
        from pumpfun_hft.analytics.report import ReportGenerator

        out = ReportGenerator(res, mc).generate(d / "report")
        console.print(f"report: {out.get('html')}")


@app.command()
def optimize(strategy: str = typer.Option(..., help="Strategy with an optimizer.spaces entry"),
             method: str = typer.Option(None, help="grid | random | bayesian | genetic"), trials: int = typer.Option(None),
             workers: int = typer.Option(None), final: bool = typer.Option(False, help="Unseal test + live-sim for the final estimate")) -> None:
    """Optimise on train, select on validation; DSR / PBO; optional sealed test evaluation."""
    from pumpfun_hft.optimizer.study import OptimizationStudy

    events = _load_events(None, None)
    res = OptimizationStudy(S(), events, str(_metadata_path()), strategy, method, trials, None, workers).run(final_evaluation=final)
    out = S().paths.resolve("reports_dir") / "optimize" / res.study_id
    out.mkdir(parents=True, exist_ok=True)
    res.trials_frame().write_parquet(out / "trials.parquet")
    summary = {k: getattr(res, k) for k in ("study_id", "strategy", "method", "splits", "selected_params", "selected_train_metrics",
                                            "selected_val_metrics", "dsr", "pbo", "test_metrics", "live_sim_metrics", "unseal_log")}
    (out / "study.json").write_text(jsonutil.dumps(summary))
    console.print_json(json.dumps({k: summary[k] for k in ("selected_params", "dsr", "pbo", "test_metrics")}, default=str))
    console.print(f"saved {out}")


@app.command()
def walkforward(strategy: str = typer.Option(...), method: str = typer.Option(None), trials: int = typer.Option(None),
                workers: int = typer.Option(None)) -> None:
    """Walk-forward optimisation with stitched out-of-sample results."""
    from pumpfun_hft.optimizer.study import WalkForwardAnalysis

    events = _load_events(None, None)
    res = WalkForwardAnalysis(S(), events, str(_metadata_path()), strategy, method, trials, None, workers).run()
    out = S().paths.resolve("reports_dir") / "walkforward" / f"{strategy}-{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    res.frame().write_parquet(out / "folds.parquet")
    console.print(res.frame())
    console.print(f"walk-forward efficiency {res.wfe:.3f} · stitched OOS return {res.oos_total_return:+.4f} · saved {out}")


@app.command()
def montecarlo(run: str = typer.Option(None, help="Run id (default: latest)"), sims: int = typer.Option(None),
               paths: int = typer.Option(0, help="Also run N full event-driven re-simulations")) -> None:
    """Monte Carlo validation of a saved run (trade-level; optional path-level)."""
    from pumpfun_hft.analytics.montecarlo import path_monte_carlo, trade_monte_carlo
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.dashboard.views import DashboardData

    s = S()
    r = DashboardData(s.paths.resolve("reports_dir")).load(run)
    if r is None:
        console.print("[red]no saved runs[/]")
        raise typer.Exit(1)
    mc = trade_monte_carlo(r.trades, r.initial_capital_sol, r.metrics.get("span_days", 1.0), s.montecarlo,
                           r.metrics.get("avg_latency_ms") or 500.0, sims)
    if paths:
        mc.path_level = path_monte_carlo(s, _load_events(None, None), r.strategies, load_metadata(_metadata_path()), paths)
    console.print_json(json.dumps(mc.summary(), default=str))
    for p in mc.path_level:
        console.print(p)


@app.command()
def report(run: str = typer.Option(None), formats: str = typer.Option("html,pdf,csv,json")) -> None:
    """(Re)generate the report for a saved run."""
    from pumpfun_hft.analytics.montecarlo import trade_monte_carlo
    from pumpfun_hft.analytics.report import ReportGenerator
    from pumpfun_hft.dashboard.views import DashboardData

    s = S()
    data = DashboardData(s.paths.resolve("reports_dir"))
    r = data.load(run)
    if r is None:
        console.print("[red]no saved runs[/]")
        raise typer.Exit(1)
    mc = trade_monte_carlo(r.trades, r.initial_capital_sol, r.metrics.get("span_days", 1.0), s.montecarlo) if r.trades.height else None
    out = ReportGenerator(r, mc).generate(data.reports_dir / "runs" / r.run_id / "report", formats.split(","))
    console.print(out)


@app.command("train-model")
def train_model(model: str = typer.Option(None, help="logistic | random_forest | xgboost | lightgbm | catboost"),
                target: str = typer.Option(None, help="rug | fwd_up | migrate"), save: bool = typer.Option(True)) -> None:
    """Build a point-in-time dataset, run purged CV, report importance/SHAP and save the model."""
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.ml.dataset import build_snapshot_dataset, feature_columns
    from pumpfun_hft.ml.models import train_and_evaluate

    s = S()
    ml = s.ml.model_copy(update={"model": model}) if model else s.ml
    tgt = target or ("fwd_up" if s.ml.target == "fwd_return" else s.ml.target)
    ds = build_snapshot_dataset(s, _load_events(None, None), load_metadata(_metadata_path()))
    rep = train_and_evaluate(ds, feature_columns(ds), tgt, ml, s.app.seed)
    console.print(f"{rep.kind} → {tgt}: rows {rep.n_rows:,}, base rate {rep.base_rate:.3f}, CV mean {json.dumps(rep.mean)}")
    console.print(rep.importance.head(15))
    if save:
        mid = f"{rep.kind}-{tgt}-{int(time.time())}"
        out = s.paths.resolve("reports_dir") / "ml" / mid
        out.mkdir(parents=True, exist_ok=True)
        rep.importance.write_parquet(out / "importance.parquet")
        (out / "cv.json").write_text(jsonutil.dumps({"folds": rep.folds, "mean": rep.mean, "train_end_ms": rep.train_end_ms}))
        path = rep.save(s.paths.resolve("models_dir") / f"{mid}.joblib")
        console.print(f"saved model {path} (trained through {ms_to_iso(rep.train_end_ms)}; set rug_model.model_path to use it)")


@app.command()
def wallets(run: str = typer.Option(None), top: int = typer.Option(25)) -> None:
    """Show the top-ranked wallets (point-in-time at the end of a saved run)."""
    from pumpfun_hft.dashboard.views import DashboardData

    w = DashboardData(S().paths.resolve("reports_dir")).artefact(run, "wallets.parquet")
    if w.is_empty():
        console.print("no wallet snapshot")
        raise typer.Exit(1)
    console.print(w.sort("smart_score", descending=True, nulls_last=True).head(top).select(
        "address", "smart_score", "closed", "wins", "realized_pnl_sol", "labels"))


@app.command()
def query(sql: str = typer.Argument(..., help="SQL over the DuckDB warehouse (tables + the `events` Parquet view)"),
          limit: int = typer.Option(50, help="Rows to print")) -> None:
    """Run SQL against the warehouse, e.g. `query "select kind, count(*) from events group by 1"`."""
    from pumpfun_hft.database.warehouse import Warehouse

    wh = Warehouse(S().paths.resolve("duckdb_file"), S().paths.events_dir, read_only=S().paths.resolve("duckdb_file").exists())
    try:
        with pl.Config(tbl_rows=limit, tbl_cols=30, fmt_str_lengths=60):
            console.print(wh.query(sql).head(limit))
    finally:
        wh.close()


# ============================================================================ dashboard
@app.command()
def dashboard() -> None:
    """Serve the local dashboard (FastAPI)."""
    from pumpfun_hft.dashboard.app import serve

    console.print(f"dashboard on http://{S().dashboard.host}:{S().dashboard.port}")
    serve(S(), _meta())


@app.command("dashboard-export")
def dashboard_export(run: str = typer.Option(None), out: Path = typer.Option(None)) -> None:
    """Write a single self-contained HTML dashboard (all pages as tabs)."""
    from pumpfun_hft.dashboard.app import static_export

    p = static_export(S(), out or S().paths.resolve("reports_dir") / "dashboard.html", run, _meta())
    console.print(f"wrote {p}")


# ============================================================================ live
def _live_components(paper: bool) -> tuple[Any, ...]:
    from pumpfun_hft.api.http import build_client
    from pumpfun_hft.api.rpc import SolanaRpcClient
    from pumpfun_hft.api.ws import SolanaWsClient
    from pumpfun_hft.collectors.live import LiveStreamCollector
    from pumpfun_hft.collectors.sol_price import provider_from_settings
    from pumpfun_hft.core.events import EventDecoder
    from pumpfun_hft.utils.latency import LatencyTracker

    s, secrets = S(), Secrets.load()
    ws_url, rpc_url = secrets.get("solana_ws_url"), secrets.get("solana_rpc_url")
    if not ws_url or not rpc_url:
        console.print("[red]SOLANA_WS_URL and SOLANA_RPC_URL must be set in .env[/]")
        raise typer.Exit(1)
    latency = LatencyTracker()
    w = s.network.ws
    ws = SolanaWsClient(ws_url, ping_interval_s=w.ping_interval_s, ping_timeout_s=w.ping_timeout_s,
                        reconnect_base_delay_s=w.reconnect_base_delay_s, reconnect_max_delay_s=w.reconnect_max_delay_s, latency=latency)
    http = build_client("rpc", "", s.network.rpc.rate_limit, s.network, s.network.rpc.timeout_s, latency)
    rpc = SolanaRpcClient(http, rpc_url, s.network.rpc.commitment)
    decoder = EventDecoder.from_settings(s)
    meta = _meta()
    from pumpfun_hft.collectors.historical import HistoricalCollector

    store = _store()
    historical = HistoricalCollector(rpc, decoder, store, meta, s.collector, s.protocol.pump_program_id, provider_from_settings(s),
                                     s.simulation.slot_ms)
    collector = LiveStreamCollector(ws, decoder, store, meta, s.collector, s.protocol, latency, provider_from_settings(s),
                                    commitment=w.commitment, historical=historical, queue_size=w.max_queue)
    return s, secrets, latency, rpc, decoder, meta, collector


@app.command()
def stream(minutes: float = typer.Option(0, help="Stop after N minutes (0 = run until Ctrl-C)")) -> None:
    """Run the live stream collector only (records events, prints latency status)."""
    s, _, latency, _, _, _, collector = _live_components(True)

    async def run() -> None:
        task = asyncio.create_task(collector.run())
        t_end = time.time() + minutes * 60 if minutes else None
        while t_end is None or time.time() < t_end:
            await asyncio.sleep(5)
            st = collector.status()
            lat = st["latency"].get("live.event_to_dispatch", {})
            console.print(f"events {st['events']:,} · slot {st['slot']} · reconnects {st['reconnects']} · dispatch p50 "
                          f"{lat.get('p50', float('nan')):.2f} ms p99 {lat.get('p99', float('nan')):.2f} ms (budget {s.collector.live_latency_budget_ms} ms)")
        await collector.stop()
        task.cancel()

    asyncio.run(run())


def _run_trader(paper: bool, strategies: list[str], minutes: float, flatten_on_exit: bool) -> None:
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.execution.engine import LiveTrader
    from pumpfun_hft.execution.gateway import LiveGateway, PaperGateway

    s, secrets, latency, rpc, decoder, meta, collector = _live_components(paper)
    queue = collector.subscribe()
    holder: dict[str, Any] = {}

    def gateway_factory(sim: Any) -> Any:
        if paper:
            return PaperGateway(sim, lambda: holder["trader"].features.tps, latency)
        from pumpfun_hft.api.metis import MetisClient
        from pumpfun_hft.execution.infra import BlockhashCache, ConfirmationTracker, PriorityFeeEstimator
        from pumpfun_hft.execution.signer import WalletSigner

        signer = WalletSigner.from_secret(secrets.get("private_key"))
        metis = MetisClient.from_settings(s, secrets, latency)
        bh = BlockhashCache(rpc, s.live.blockhash_refresh_ms, s.live.confirm_commitment)
        prio = PriorityFeeEstimator(rpc, s.priority_fee, s.network.metis.priority_fee_levels, [s.protocol.pump_program_id])
        conf = ConfirmationTracker(rpc, s.live.confirm_poll_ms, s.live.confirm_commitment, latency)
        holder.update(bh=bh, prio=prio)
        return LiveGateway(s, rpc, metis, signer, bh, prio, conf, decoder, sim, latency, meta)

    trader = LiveTrader(s, queue, gateway_factory, strategies=strategies or None, meta=meta, latency=latency,
                        metadata=load_metadata(_metadata_path()), collector_status=collector.status)
    holder["trader"] = trader

    async def run() -> None:
        tasks = [asyncio.create_task(collector.run()), asyncio.create_task(trader.run())]
        if "bh" in holder:
            holder["bh"].start()
            holder["prio"].start()
            trader.runtime.priority_override = lambda urgency: holder["prio"].micro_lamports(urgency.value)
        try:
            if minutes:
                await asyncio.sleep(minutes * 60)
            else:
                await asyncio.Event().wait()
        finally:
            await trader.stop(flatten=flatten_on_exit)
            await collector.stop()
            for t in tasks:
                t.cancel()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("stopped")


@app.command()
def paper(strategy: list[str] = typer.Option(None, "--strategy"), minutes: float = typer.Option(0),
          flatten_on_exit: bool = typer.Option(True)) -> None:
    """Paper trading: live data, simulated execution (no wallet needed)."""
    _run_trader(True, list(strategy or []), minutes, flatten_on_exit)


@app.command()
def live(strategy: list[str] = typer.Option(None, "--strategy"), minutes: float = typer.Option(0),
         confirm_live: bool = typer.Option(False, "--confirm-live", help="Required acknowledgement for real trading"),
         flatten_on_exit: bool = typer.Option(True)) -> None:
    """LIVE trading with real funds. Requires app.mode=live in config AND --confirm-live."""
    if S().app.mode != "live" or not confirm_live:
        console.print("[red]Refusing to trade live: set app.mode: live in your config and pass --confirm-live.[/]")
        raise typer.Exit(2)
    console.print("[bold red]LIVE TRADING — real transactions will be signed and sent.[/]")
    _run_trader(False, list(strategy or []), minutes, flatten_on_exit)


@app.command("paper-replay")
def paper_replay(strategy: list[str] = typer.Option(None, "--strategy"),
                 start: str = typer.Option(None, help="ISO start (UTC)"), end: str = typer.Option(None, help="ISO end (UTC)"),
                 hours: float = typer.Option(0.0, help="Replay only the first N hours of the selection (0 = all)"),
                 realtime: bool = typer.Option(False, help="Pace the replay at 1x instead of virtual time"),
                 compare: bool = typer.Option(True, help="Also backtest the same events and print both side by side")) -> None:
    """Replay stored events through the LIVE engine with the paper gateway: the whole live stack, offline."""
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.execution.replay import replay_live

    s = S()
    events = _load_events(start, end)
    if hours:
        events = events.filter(pl.col("ts_ms") < int(events["ts_ms"].min()) + int(hours * 3_600_000))
    span_h = (int(events["ts_ms"].max()) - int(events["ts_ms"].min())) / 3_600_000
    console.print(f"replaying {events.height:,} events ({span_h:.2f} h of market time) through the live engine"
                  f"{' in real time' if realtime else ' on virtual time'}")
    metadata = load_metadata(_metadata_path())
    marks = iter(range(10, 101, 10))
    nxt = [next(marks)]

    def progress(done: int, total: int) -> None:  # one line per 10 %
        while nxt[0] <= 100 and 100 * done / total >= nxt[0]:
            console.print(f"  {nxt[0]:3d} %  {done:,} / {total:,} events")
            nxt[0] = next(marks, 101)

    names = list(strategy or []) or None
    rep = replay_live(s, events, metadata, names, meta=_meta(), realtime=realtime, progress=progress)
    t = Table(title=f"Live engine replay: {rep.events:,} events in {rep.wall_s:.1f} s")
    t.add_column("metric")
    t.add_column("live engine (paper)", justify="right")
    bt = None
    if compare:
        from pumpfun_hft.backtester.engine import BacktestEngine
        from pumpfun_hft.backtester.replay import DataSource

        bt = BacktestEngine(s, DataSource(frame=events), names, metadata=metadata, synthetic=_is_synthetic()).run()
        t.add_column("backtest", justify="right")

    def win_rate(tr: pl.DataFrame) -> str:
        return f"{100 * float((tr['pnl_sol'] > 0).mean()):.1f}%" if tr.height else "–"

    def fill_counts(f: pl.DataFrame) -> str:
        if not f.height:
            return "0 / 0"
        return f"{int(f['status'].is_in(['filled', 'partial']).sum()):,} / {int(f['status'].is_in(['failed', 'dropped', 'expired']).sum()):,}"

    rows = [("round trips", f"{rep.round_trips:,}", bt and f"{bt.trades.height:,}"),
            ("total return", f"{100 * rep.total_return:+.2f}%", bt and f"{100 * bt.metrics['total_return']:+.2f}%"),
            ("realised PnL (SOL)", f"{rep.realized_pnl_sol:+.4f}", bt and f"{float(bt.trades['pnl_sol'].sum()) if bt.trades.height else 0.0:+.4f}"),
            ("win rate", win_rate(rep.trades), bt and win_rate(bt.trades)),
            ("fills / failed or dropped", fill_counts(rep.fills_frame), bt and fill_counts(bt.fills)),
            ("open positions at end", str(rep.open_positions), bt and "0")]
    for name, a, b in rows:
        t.add_row(name, a, *([b] if compare else []))
    console.print(t)
    lat = rep.latency.get("live.event_to_decision")
    if lat:
        console.print(f"event -> decision (processing): p50 {lat['p50']:.3f} ms · p99 {lat['p99']:.3f} ms · max {lat['max']:.2f} ms")
    console.print("[dim]The live engine is asynchronous and draws latencies / failures in a different order than the "
                  "backtester, so the two agree statistically, not trade by trade. The final snapshot is in the "
                  "dashboard's Live monitor.[/]")


@app.command("latency-probe")
def latency_probe(n: int = typer.Option(20, help="Requests per method")) -> None:
    """Measure RPC / Metis latency from this machine (p50/p90/p99)."""
    from pumpfun_hft.api.http import build_client
    from pumpfun_hft.api.metis import MetisClient
    from pumpfun_hft.api.rpc import SolanaRpcClient
    from pumpfun_hft.utils.latency import LatencyTracker

    s, secrets = S(), Secrets.load()
    lat = LatencyTracker()

    async def run() -> None:
        url = secrets.get("solana_rpc_url")
        if url:
            http = build_client("rpc", "", s.network.rpc.rate_limit, s.network, s.network.rpc.timeout_s, lat)
            rpc = SolanaRpcClient(http, url)
            for _ in range(n):
                with lat.measure("rpc.getSlot"):
                    await rpc.get_slot()
                with lat.measure("rpc.getLatestBlockhash"):
                    await rpc.get_latest_blockhash()
            await http.aclose()
        m = MetisClient.from_settings(s, secrets, lat)
        sol, usdc = "So11111111111111111111111111111111111111112", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        for _ in range(max(1, n // 4)):
            try:
                with lat.measure("metis.quote"):
                    await m.quote(sol, usdc, 10_000_000, 50)
            except Exception as exc:  # noqa: BLE001 - report and keep probing the rest
                console.print(f"[yellow]metis quote failed: {exc}[/]")
                break
        await m.aclose()

    asyncio.run(run())
    t = Table(title="Latency (ms)")
    for c in ("channel", "p50", "p90", "p99", "max"):
        t.add_column(c)
    for k, v in lat.snapshot().items():
        t.add_row(k, f"{v['p50']:.1f}", f"{v['p90']:.1f}", f"{v['p99']:.1f}", f"{v['max']:.1f}")
    console.print(t)


if __name__ == "__main__":
    app()
