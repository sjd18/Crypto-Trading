"""Command-line interface (Typer + Rich).

Usage: ``python -m pumpfun_hft.main [--config my.yaml] [--set key=value ...] COMMAND [options]``

Data sets:  --dataset synthetic | real (hft.ps1: hft / hftr) · data-info · find-data · import-data
Research:   synth · collect-history · verify-data · backtest · optimize · walkforward · montecarlo ·
            report · train-model · wallets · query
Live:       stream · paper · paper-replay · live (requires app.mode=live AND --confirm-live) · latency-probe
Tooling:    init · check-config · dashboard · dashboard-export · update-idl · docs
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from pumpfun_hft.core.config import DATASET_KINDS, PROJECT_ROOT, Secrets, Settings, dataset_path_overrides, load_settings
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


#: commands that do not read or write a data set (no data-set check, no banner)
NO_DATASET_COMMANDS = frozenset({"check-config", "docs", "update-idl", "latency-probe", "find-data", "data-info", "import-data"})


@app.callback()
def main(ctx: typer.Context,
         config: Path = typer.Option(None, "--config", "-c", help="User YAML merged over configs/default.yaml"),
         set_: list[str] = typer.Option(None, "--set", "-s", help="Dotted override, e.g. backtest.initial_capital_sol=5"),
         dataset: str = typer.Option(None, "--dataset", "-d",
                                     help="synthetic | real: run on that data set, whose folder holds its events, models and "
                                          "reports (hft.ps1: hft = synthetic, hftr = real)"),
         data_root: Path = typer.Option(None, "--data-root", help="Folder of the --dataset (default: datasets.<name> in the config)")) -> None:
    overrides = _parse_sets(set_)
    name = dataset or load_settings(config, overrides).datasets.active  # a config file may also set datasets.active
    if name and name not in DATASET_KINDS:
        raise typer.BadParameter(f"--dataset must be one of {', '.join(DATASET_KINDS)}, got {name!r}")
    if data_root is not None and not name:
        raise typer.BadParameter("--data-root needs --dataset synthetic|real")
    if name:
        root = data_root.expanduser().resolve() if data_root is not None else load_settings(config, overrides).datasets.root(name)
        # every path of the data set lives in its folder; explicit --set values still win
        overrides = {**dataset_path_overrides(root), **overrides, "datasets.active": name}
    settings = load_settings(config, overrides)
    STATE.update(settings=settings, config_path=config, overrides=overrides)
    lg = settings.logging
    setup_logging(lg.level, settings.paths.resolve("logs_dir"), lg.max_bytes, lg.backup_count, lg.console, lg.channels)
    if name and ctx.invoked_subcommand not in NO_DATASET_COMMANDS:
        _check_dataset()


def S() -> Settings:
    return STATE["settings"]


def _cli(kind: str | None = None) -> str:
    """How the user runs commands on a data set (the hft.ps1 function names)."""
    kind = kind if kind is not None else S().datasets.active
    return {"synthetic": "hft", "real": "hftr"}.get(kind, "python -m pumpfun_hft.main")


def _data_root() -> Path:
    return S().paths.resolve("data_dir")


def _kind_on_disk() -> str:
    from pumpfun_hft.collectors.datasets import detect_kind

    p = S().paths
    return detect_kind(_data_root(), p.events_subdir, p.metadata_subdir)


def _check_dataset() -> None:
    """The folder of ``--dataset NAME`` must hold that kind of data (or nothing yet); marks it on first use."""
    from pumpfun_hft.collectors.datasets import read_marker, write_marker

    name, root = S().datasets.active, _data_root()
    on_disk = _kind_on_disk()
    console.print(f"[dim]data set: {name} · {escape(str(root))}[/]")
    if on_disk == "empty":
        return
    if on_disk != name:
        console.print(f"[red]{escape(str(root))} holds {on_disk} data, but `{_cli(name)}` runs on the {name} data set. "
                      "Stopped so the two never mix.[/]")
        if name == "real":
            console.print("Point HFT_REAL_DIR in hft.ps1 (or --data-root) at the folder with your recorded events; "
                          "`hftr find-data` lists every event store on this computer. If that folder also contains a "
                          "synthetic market, copy just the recorded events into an empty real folder with "
                          f"`hftr import-data --from {escape(str(root))}`.")
        else:
            console.print("Point HFT_SYNTH_DIR in hft.ps1 (or --data-root) at a different folder for the synthetic market.")
        raise typer.Exit(2)
    if read_marker(root) is None:
        write_marker(root, name)


def _require_real_store(command: str) -> None:
    """Commands that record live Pump.fun events must write into the real data set, never the synthetic one."""
    from pumpfun_hft.collectors.datasets import write_marker

    if S().datasets.active == "synthetic":
        console.print(f"[red]`{command}` records real Pump.fun events, so it runs on the real data set: use "
                      f"`hftr {command} ...` (not `hft`).[/]")
        raise typer.Exit(2)
    if _kind_on_disk() == "synthetic":
        console.print(f"[red]{escape(str(_data_root()))} holds a synthetic market; recording real events into it would mix the two. "
                      f"Use `hftr {command} ...` (or --dataset real).[/]")
        raise typer.Exit(2)
    if S().datasets.active == "real":
        write_marker(_data_root(), "real")


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
        where = escape(str(S().paths.events_dir))
        if (start or end) and not store.scan(columns=["ts_ms"]).head(1).collect().is_empty():
            console.print(f"[yellow]No events between --start {start or '(beginning)'} and --end {end or '(end)'} in {where}. "
                          f"`{_cli()} data-info` shows the time range of the data.[/]")
        elif S().datasets.active == "real":
            console.print(f"[yellow]The event store is empty: no recorded events in {where}. If you recorded them into another "
                          "folder, `hftr find-data` lists every event store on this computer; then either point HFT_REAL_DIR "
                          "in hft.ps1 at it or copy them here with `hftr import-data --from <folder>`. New data: "
                          "`hftr stream --minutes 60`.[/]")
        elif S().datasets.active == "synthetic":
            console.print(f"[yellow]The event store is empty ({where}). Generate a synthetic market with `hft synth --hours 24`.[/]")
        else:
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
    from pumpfun_hft.collectors.datasets import real_event_count, write_marker

    if S().datasets.active == "real":
        console.print("[red]`synth` writes a fake market, so it runs on the synthetic data set: use `hft synth ...` (not `hftr`).[/]")
        raise typer.Exit(2)
    s = load_settings(STATE.get("config_path"), {**_settings_overrides(), **ov}) if ov else S()
    n_real = real_event_count(s.paths.resolve("data_dir"), s.paths.events_subdir, s.paths.metadata_subdir)
    if n_real:
        console.print(f"[red]{escape(str(s.paths.events_dir))} holds {n_real:,} recorded (real) events; `synth` would "
                      f"{'delete' if clear else 'mix synthetic tokens into'} them. Run synth on the synthetic data set "
                      "(`hft synth`) or point paths.data_dir at another folder.[/]")
        raise typer.Exit(2)
    if clear and s.paths.events_dir.exists():
        shutil.rmtree(s.paths.events_dir)
        meta = _meta()
        for rec in meta.files():  # the manifest must not keep checksums of the files just deleted
            meta.remove_file(rec["path"])
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
    write_marker(s.paths.resolve("data_dir"), "synthetic")
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

    _require_real_store("collect-history")
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
def verify_data(gap_slots: int = typer.Option(None, help="Report slot gaps wider than this"),
                adopt: bool = typer.Option(False, help="Checksum files the manifest does not know yet into it")) -> None:
    """Verify Parquet checksums against the manifest and report slot gaps."""
    store = _store()
    bad = store.verify()
    console.print(f"files: {store.stats()} · checksum mismatches: {len(bad)}")
    for b in bad[:20]:
        console.print(f"  [red]{b}[/]")
    new = store.unregistered()
    if new and adopt:
        console.print(f"added {store.adopt(new)} file(s) to the manifest")
    elif new:
        console.print(f"[yellow]{len(new)} file(s) are not in this data set's manifest yet (written before it had its own "
                      f"meta.sqlite, or copied in), so they were not checked: `{_cli()} verify-data --adopt` records their "
                      "checksums now.[/]")
    try:
        gaps = store.detect_gaps(gap_slots or S().collector.gap_slot_threshold)
    except Exception as exc:  # noqa: BLE001 - e.g. a corrupt file: report it, the checksums above say which
        console.print(f"[red]could not read the events for the gap check: {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    console.print(f"slot gaps > threshold: {gaps.height}")
    if gaps.height:
        console.print(gaps.head(20))
    if bad:
        raise typer.Exit(1)


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def _iso_minute(ms: int) -> str:
    from pumpfun_hft.utils.timeutil import ms_to_dt

    return ms_to_dt(ms - ms % 60_000).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.command("data-info")
def data_info(train_frac: float = typer.Option(0.6, help="Share of the events to train on when suggesting --end")) -> None:
    """Where the data set lives and what it holds: events, time range, models, runs, and a train / test split."""
    import joblib

    from pumpfun_hft.collectors.datasets import describe, time_quantile

    s = S()
    p = s.paths
    active = s.datasets.active
    info = describe(p.events_dir, p.metadata_subdir)
    rows: list[tuple[str, str]] = []
    rows.append(("data set", f"{active} (`{_cli()}`)" if active else "not selected (paths as configured; hft.ps1 selects one)"))
    rows.append(("folder", escape(str(_data_root()))))
    rows.append(("holds", f"{info.contents} events" + (f" (marked {info.marker})" if info.marker else "")))
    if info.events:
        rows.append(("events", f"{info.events:,} in {len(info.days)} day(s), {info.files:,} file(s), {_fmt_bytes(info.bytes)}"))
        rows.append(("time range", info.span()))
        if info.contents == "mixed":
            rows.append(("", f"[yellow]{info.real_events:,} recorded + {info.synthetic_events:,} synthetic events[/]"))
    models_dir = p.resolve("models_dir")
    models = sorted(models_dir.glob("*.joblib"), key=lambda f: f.stat().st_mtime, reverse=True) if models_dir.is_dir() else []
    rows.append(("models", f"{len(models)} in {escape(str(models_dir))}"))
    for f in models[:8]:
        try:
            b = joblib.load(f)
            cut = ms_to_iso(int(b["train_end_ms"])) if b.get("train_end_ms") is not None else "?"
            rows.append(("", f"{escape(f.name)} · target {b.get('target', '?')} · cut-off {cut} · "
                             f"trained on {b.get('dataset', 'unrecorded')} data"))
        except Exception as exc:  # noqa: BLE001 - list what can be read
            rows.append(("", f"{escape(f.name)} · unreadable ({type(exc).__name__})"))
    runs_dir = p.resolve("reports_dir") / "runs"
    n_runs = sum(1 for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.is_dir() else 0
    rows.append(("backtests", f"{n_runs} saved in {escape(str(runs_dir))}"))
    for label, value in rows:  # plain lines, never truncated: the paths are what people copy
        console.print(f"[bold]{label:<11}[/]{value}", soft_wrap=True)
    if active and info.events and info.contents not in (active, "mixed"):
        console.print(f"[red]This folder holds {info.contents} data but is used as the {active} data set.[/]")
    if not info.events:
        console.print("No events yet." + (" `hftr find-data` lists every event store on this computer." if active != "synthetic" else
                                          " Create a synthetic market: `hft synth --hours 24`."))
        return
    if not 0.0 < train_frac < 1.0:
        raise typer.BadParameter("--train-frac must be between 0 and 1")
    q = time_quantile(p.events_dir, train_frac)
    if q is not None and info.start_ms is not None and info.start_ms < q:
        end_iso = _iso_minute(q)
        console.print(f"\nTrain on the first {train_frac:.0%} of the events and test on the rest:")
        console.print(f"  {_cli()} train-model --model lightgbm --target fwd_up --end {end_iso}", soft_wrap=True)
        console.print(f"  {_cli()} backtest --strategy ml_signal --start <the training cut-off that train-model prints>",
                      soft_wrap=True)
        console.print("[dim]The cut-off is at or before --end: rows whose label window the data does not fully cover are "
                      "left out of training.[/]")


def _search_roots(extra: list[Path]) -> list[Path]:
    s = S()
    roots = [*extra, s.datasets.root("synthetic"), s.datasets.root("real"), _data_root(), PROJECT_ROOT, PROJECT_ROOT.parent]
    roots += [r.parent for r in (s.datasets.root("synthetic"), s.datasets.root("real"), _data_root())]
    if os.name == "nt":
        roots += [Path("C:/pumpfun"), Path(PROJECT_ROOT.anchor) / "pumpfun"]
    roots.append(Path.home())
    out: list[Path] = []
    for r in roots:
        if r not in out:
            out.append(r)
    return out


@app.command("find-data")
def find_data(path: list[Path] = typer.Option(None, "--path", help="Also search this folder (repeatable)"),
              depth: int = typer.Option(7, help="How many folder levels below each search root to look")) -> None:
    """Find every event store on this computer and say what each holds (real / synthetic / mixed)."""
    from pumpfun_hft.collectors.datasets import describe, find_event_dirs

    roots = _search_roots(list(path or []))
    console.print("searching " + ", ".join(escape(str(r)) for r in roots if r.is_dir()) + " ...")
    found = find_event_dirs(roots, depth)
    if not found:
        console.print("[yellow]No event stores found. Add a folder to search with --path.[/]")
        raise typer.Exit(1)
    current = os.path.normcase(str(S().paths.events_dir.resolve()))
    console.print(f"\n[bold]{len(found)} event store(s)[/] (data-set folder, then what it holds; times in UTC)")
    for ev in found:
        info = describe(ev, S().paths.metadata_subdir)
        here = "  [green](this data set)[/]" if os.path.normcase(str(ev)) == current else ""
        sub = "" if ev.name == S().paths.events_subdir else f"  (events in the '{escape(ev.name)}' subfolder)"
        console.print(f"\n{escape(str(info.root))}{sub}{here}", soft_wrap=True)
        console.print(f"    {info.contents} · {info.events:,} events, {info.real_events:,} of them recorded (real) · "
                      f"{_fmt_bytes(info.bytes)}\n    {info.span()}", soft_wrap=True)
    console.print()
    console.print("Use recorded data with hftr: set HFT_REAL_DIR in hft.ps1 to its data-set folder (when it holds only real "
                  "events), or copy the recorded events into the current real folder: `hftr import-data --from <folder>`.")


@app.command("import-data")
def import_data(from_: Path = typer.Option(..., "--from", help="A data-set folder (or its events folder) holding recorded events"),
                ) -> None:
    """Copy the recorded (real) events of another folder into this real data set; synthetic tokens are left behind.

    The source is never modified. Events already in this data set are de-duplicated."""
    from pumpfun_hft.collectors.datasets import describe, detect_kind, import_events, resolve_events_dir, write_marker

    s = S()
    p = s.paths
    if s.datasets.active != "real":
        console.print("[red]import-data copies recorded events into the real data set: run it as `hftr import-data --from ...`.[/]")
        raise typer.Exit(2)
    on_disk = detect_kind(_data_root(), p.events_subdir, p.metadata_subdir)
    if on_disk == "synthetic":
        console.print(f"[red]{escape(str(_data_root()))} holds a synthetic market; point HFT_REAL_DIR at an empty folder "
                      "(or one with only recorded events) first.[/]")
        raise typer.Exit(2)
    try:
        src = resolve_events_dir(from_, p.events_subdir)
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    if os.path.normcase(str(src)) == os.path.normcase(str(p.events_dir.resolve())):
        console.print("[yellow]That is this data set's own event store; nothing to import.[/]")
        raise typer.Exit(1)
    info = describe(src, p.metadata_subdir)
    console.print(f"source {escape(str(src))}: {info.events:,} events ({info.contents}; {info.real_events:,} recorded), {info.span()}")
    if info.real_events == 0:
        console.print("[yellow]The source holds no recorded events (only a synthetic market): nothing to import.[/]")
        raise typer.Exit(1)
    write_marker(_data_root(), "real")
    rep = import_events(src, _store(), _metadata_path(), p.metadata_subdir,
                        progress=lambda day, n: console.print(f"  {day}: {n:,} events"))
    console.print(f"[green]imported[/] {rep.written:,} events from {rep.days} day(s) · left behind {rep.skipped_synthetic:,} "
                  f"synthetic · removed {rep.duplicates_removed:,} duplicates · token metadata rows {rep.tokens_metadata:,}")
    console.print(f"this data set now: {describe(p.events_dir, p.metadata_subdir).span()}")
    console.print("The source folder was not changed; delete it yourself once you have checked the result "
                  "(`hftr data-info`, `hftr verify-data`).")


# ============================================================================ research
def _preflight_models(names: list[str]) -> Any:
    """Load the trained models a run will use before it starts, so a missing file or a model of the other data
    set is a one-line error instead of a traceback. Returns the ml_signal strategy (or None)."""
    from pumpfun_hft.ml.rug_model import build_rug_scorer
    from pumpfun_hft.strategies.base import build_strategy

    s = S()
    try:
        if s.rug_model.use_trained_model:
            build_rug_scorer(s.rug_model, s.paths.resolve("models_dir"), s.datasets.active)
        strat: Any = build_strategy("ml_signal", s) if "ml_signal" in names else None
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    if strat is not None:
        console.print(f"ml_signal model: {escape(strat.path.name)} (cut-off {ms_to_iso(strat.train_end_ms)})")
        for line in strat.describe():
            console.print(f"  {'[yellow]' + line + '[/]' if line.startswith('warning') else line}", soft_wrap=True)
    return strat


def _ml_cutoff_notice(names: list[str], events: pl.DataFrame) -> None:
    """Check the models; ml_signal never trades before its model's training cut-off: say so when the data starts earlier."""
    strat = _preflight_models(names)
    if strat is None:
        return
    t0, t1 = int(events["ts_ms"].min()), int(events["ts_ms"].max())
    if strat.train_end_ms >= t1:
        console.print("[yellow]All of the selected data is before the model's training cut-off: ml_signal will not trade. "
                      f"Train on older data (train-model --end ...; `{_cli()} data-info` suggests one) and test on the rest.[/]")
    elif strat.train_end_ms > t0:
        console.print(f"[yellow]ml_signal trades only after {ms_to_iso(strat.train_end_ms)}; "
                      f"use --start {ms_to_iso(strat.train_end_ms)} for a clean out-of-sample window.[/]")


def _ml_funnel(runtime: Any) -> None:
    strat = getattr(runtime, "by_name", {}).get("ml_signal")
    if strat is not None:
        console.print(f"ml_signal: {strat.funnel(getattr(runtime, 'signal_records', None))}")


def _explain_no_trades(res: Any) -> None:
    """With no round trips every metric is 0 / NaN: say so, and where the entries went."""
    if res.trades.height:
        return
    console.print("[yellow]No trades, so the metrics above (and the report) are empty: 0 or NaN.[/]")
    sig = res.signals.filter(pl.col("action") == "BUY") if res.signals.height else res.signals
    if not sig.height:
        console.print("[yellow]No strategy produced a buy signal on this data (for ml_signal the funnel above shows why).[/]")
        return
    counts = sig.group_by("strategy", "outcome").len().sort("len", descending=True)
    console.print("[yellow]Buy signals and what happened to them:[/] " + " · ".join(
        f"{r['strategy']} {r['outcome']} {r['len']:,}" for r in counts.iter_rows(named=True)), soft_wrap=True)
    hints = {"cost_gate": "expected return below strategy.cost_gate_multiple x round-trip cost",
             "low_confidence": "confidence below sizing.min_confidence", "vetoed": "vetoed by the exit overlay (rug_avoidance)"}
    for k in counts["outcome"].unique().to_list():
        if k in hints:
            console.print(f"  {k}: {hints[k]}")


def _run_backtest(strategies: list[str], start: str | None, end: str | None, seed: int | None, run_id: str | None = None) -> Any:
    from pumpfun_hft.backtester.engine import BacktestEngine
    from pumpfun_hft.backtester.replay import DataSource, load_metadata

    s = S()
    events = _load_events(start, end)
    _ml_cutoff_notice(strategies or list(s.strategy.active), events)
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
    _ml_funnel(eng.runtime)
    _explain_no_trades(res)
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
    _ml_cutoff_notice([strategy], events)
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
    _ml_cutoff_notice([strategy], events)
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
                target: str = typer.Option(None, help="rug | fwd_up | migrate"),
                start: str = typer.Option(None, help="ISO start (UTC) of the training data"),
                end: str = typer.Option(None, help="ISO end (UTC) of the training data; test on data after it"),
                save: bool = typer.Option(True)) -> None:
    """Build a point-in-time dataset, run purged CV, report importance/SHAP and save the model."""
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.ml.dataset import build_snapshot_dataset, feature_columns
    from pumpfun_hft.ml.models import NotEnoughTrainingData, train_and_evaluate

    s = S()
    ml = s.ml.model_copy(update={"model": model}) if model else s.ml
    tgt = target or ("fwd_up" if s.ml.target == "fwd_return" else s.ml.target)
    events = _load_events(start, end)
    # synthetic markets have no recording gaps (their quiet stretches are real lulls); real recordings do
    ds = build_snapshot_dataset(s, events, load_metadata(_metadata_path()),
                                max_gap_s=None if _is_synthetic() else s.ml.label_max_data_gap_s)
    if ds.is_empty():
        console.print("[yellow]No snapshots: the selected data has no tokens old enough to snapshot.[/]")
        raise typer.Exit(1)
    try:
        rep = train_and_evaluate(ds, feature_columns(ds), tgt, ml, s.app.seed)
    except NotEnoughTrainingData as exc:
        console.print(f"[yellow]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    flag = "fwd_complete" if tgt in ("fwd_up", "fwd_return") else "label_complete"
    dropped = int((~ds[flag]).sum())
    if dropped:
        console.print(f"left out {dropped:,} of {ds.height:,} snapshots whose label window the data does not fully cover "
                      "(end of the data, or a recording gap)")
    console.print(f"{rep.kind} → {tgt}: rows {rep.n_rows:,}, base rate {rep.base_rate:.3f}, CV mean {json.dumps(rep.mean)}")
    console.print(rep.importance.head(15))
    if save:
        mid = f"{rep.kind}-{tgt}-{int(time.time())}"
        out = s.paths.resolve("reports_dir") / "ml" / mid
        out.mkdir(parents=True, exist_ok=True)
        rep.importance.write_parquet(out / "importance.parquet")
        (out / "cv.json").write_text(jsonutil.dumps({"folds": rep.folds, "mean": rep.mean, "train_end_ms": rep.train_end_ms}))
        # the label definition travels with the model, so ml_signal scores and exits on the same terms
        rep.bundle.update({"snapshot_delays_s": list(s.rug_model.snapshot_delays_s), "label_horizon_s": s.rug_model.label_horizon_s,
                           "fwd_return_horizon_s": s.ml.fwd_return_horizon_s, "fwd_return_threshold": s.ml.fwd_return_threshold,
                           "data_start_ms": int(events["ts_ms"].min()), "data_end_ms": int(events["ts_ms"].max()),
                           "dataset": S().datasets.active or _kind_on_disk()})
        path = rep.save(s.paths.resolve("models_dir") / f"{mid}.joblib")
        cut = ms_to_iso(rep.train_end_ms)
        console.print(f"saved model {escape(str(path))}", soft_wrap=True)
        console.print(f"training cut-off {cut} (the model will not trade any event before it)", soft_wrap=True)
        if tgt == "rug":
            console.print(f"use it: set rug_model.use_trained_model: true and rug_model.model_path: {path.name} "
                          "(looked up in this data set's models folder)")
        else:
            console.print(f"use it: {_cli()} backtest --strategy ml_signal --start {cut}", soft_wrap=True)
            console.print("(ml_signal loads the newest fwd_up model of this data set)")
            from pumpfun_hft.strategies.ml_signal import MlSignal

            preview = MlSignal(dict(s.strategy.params.get("ml_signal", {})))
            preview.path = path
            preview.load_bundle(rep.bundle, s)
            for line in preview.describe():
                console.print(f"  ml_signal {'[yellow]' + line + '[/]' if line.startswith('warning') else line}", soft_wrap=True)
            table = MlSignal.threshold_table(rep.bundle)
            if table and tgt == "fwd_up":
                t = Table(title=f"Out-of-sample: what followed the top-scored snapshots ({s.ml.fwd_return_horizon_s / 60:g} min)")
                for c in ("top_frac", "score >=", "snapshots", "hit rate", "mean return"):
                    t.add_column(c, justify="right")
                for r in table:
                    t.add_row(f"{r['top_frac']:.0%}", f"{r['threshold']:.3f}", f"{int(r['n']):,}",
                              "–" if r["hit_rate"] != r["hit_rate"] else f"{r['hit_rate']:.1%}",
                              "–" if r["mean_return"] != r["mean_return"] else f"{r['mean_return']:+.1%}")
                console.print(t)
                console.print(f"[dim]ml_signal buys the top {preview.p.top_frac:.0%} "
                              "(strategy.params.ml_signal.top_frac). An entry also needs its mean return to beat "
                              f"{s.strategy.cost_gate_multiple:g} x the round-trip cost (fees, slippage, impact), or the "
                              "cost gate skips it. Choosing top_frac from this table uses training data only.[/]")
            store_end = _store().scan(columns=["ts_ms"]).select(pl.col("ts_ms").max()).collect().item() if end \
                else int(events["ts_ms"].max())
            t0 = int(events["ts_ms"].min())
            if rep.train_end_ms >= t0 + 0.9 * (int(store_end) - t0):
                console.print("[yellow]Almost none of your stored data is after the cut-off, so there is nothing left to "
                              "test it on. Retrain on the older part with --end (e.g. the first 60 %), backtest the rest "
                              f"with --start (`{_cli()} data-info` suggests a split), or paper trade it on new live data.[/]")


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
    except Exception as exc:  # noqa: BLE001 - SQL errors are the user's to fix: show them without a traceback
        console.print(f"[red]{escape(str(exc))}[/]")
        if "events" in sql.lower() and not any(S().paths.events_dir.glob("date=*/*.parquet")):
            console.print(f"[yellow]This data set has no events yet ({escape(str(S().paths.events_dir))}).[/]")
        raise typer.Exit(1) from exc
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
    _require_real_store("stream")
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
    store = _store()
    reports = store.compact()  # merge the part files and drop re-fetched duplicates
    dupes = sum(r.duplicates for r in reports)
    st = collector.status()
    console.print(f"[green]recorded[/] {st['events']:,} events · {st['reconnects']} reconnects · {st['truncated']} truncated logs · "
                  f"removed {dupes:,} duplicate rows · store now {store.stats()}")
    gaps = store.detect_gaps(S().collector.gap_slot_threshold)
    if gaps.height:
        console.print(f"[yellow]{gaps.height} slot gaps wider than the threshold — run `verify-data` to inspect them.[/]")


def _run_trader(paper: bool, strategies: list[str], minutes: float, flatten_on_exit: bool) -> None:
    from pumpfun_hft.backtester.replay import load_metadata
    from pumpfun_hft.execution.engine import LiveTrader
    from pumpfun_hft.execution.gateway import LiveGateway, PaperGateway

    _preflight_models(strategies or list(S().strategy.active))
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
    _ml_funnel(trader.runtime)


@app.command()
def paper(strategy: list[str] = typer.Option(None, "--strategy"), minutes: float = typer.Option(0),
          flatten_on_exit: bool = typer.Option(True)) -> None:
    """Paper trading: live data, simulated execution (no wallet needed)."""
    _require_real_store("paper")
    _run_trader(True, list(strategy or []), minutes, flatten_on_exit)


@app.command()
def live(strategy: list[str] = typer.Option(None, "--strategy"), minutes: float = typer.Option(0),
         confirm_live: bool = typer.Option(False, "--confirm-live", help="Required acknowledgement for real trading"),
         flatten_on_exit: bool = typer.Option(True)) -> None:
    """LIVE trading with real funds. Requires app.mode=live in config AND --confirm-live."""
    if S().app.mode != "live" or not confirm_live:
        console.print("[red]Refusing to trade live: set app.mode: live in your config and pass --confirm-live.[/]")
        raise typer.Exit(2)
    _require_real_store("live")
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
    _ml_cutoff_notice(names or list(s.strategy.active), events)
    rep = replay_live(s, events, metadata, names, meta=_meta(), realtime=realtime, progress=progress)
    if rep.ml_funnel:
        console.print(f"ml_signal (live engine): {rep.ml_funnel}")
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
