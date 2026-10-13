"""Strategy builder: compose a ``trade`` command (profile + strategies + duration + options), show it
as the PowerShell line to type, and run paper sessions from the dashboard with their raw output.

Profiles are the hft.ps1 functions: ``hft`` / ``hftr`` replay stored events (synthetic / recorded),
``hfts`` paper trades the live stream, ``hftl`` trades it with real money. The dashboard only ever
*shows* hftl commands: real-money sessions are started by hand in PowerShell, never from a browser.

Example::

    spec = parse_spec({"mode": "hfts", "strategies": ["smart_money"], "duration": 60}, strategy_names())
    powershell_line(spec)   # 'hfts trade --strategy smart_money --minutes 60'
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: profile -> what `trade` does there (shown in the builder)
MODES: dict[str, dict[str, str]] = {
    "hft": {"dataset": "synthetic", "feed": "replay", "title": "Paper · synthetic market",
            "desc": "Replays the stored synthetic market through the live engine with simulated fills, next to a "
                    "backtest of the same events. Exercises the pipeline; says nothing about real profitability."},
    "hftr": {"dataset": "real", "feed": "replay", "title": "Paper · recorded real data",
             "desc": "Replays your recorded (downloaded) Pump.fun events through the live engine with simulated fills, "
                     "next to a backtest of the same events."},
    "hfts": {"dataset": "real", "feed": "stream", "title": "Paper · live stream",
             "desc": "Trades the live Pump.fun stream with simulated fills: real market data, no wallet, no money at "
                     "risk. The stream is recorded into the real data set, so hftr can backtest it later."},
    "hftl": {"dataset": "real", "feed": "stream", "title": "LIVE · real money",
             "desc": "Signs and sends real transactions from your wallet. The dashboard only builds this command: "
                     "copy it into PowerShell after reviewing docs/LIVE_TRADING.md and your paper results."},
}
#: duration slider stops: minutes of live stream (0 = until stopped), hours of stored events (0 = all)
STREAM_MINUTES = (1, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440, 0)
REPLAY_HOURS = (0.5, 1, 2, 3, 6, 12, 24, 48, 72, 0)
STATUS_EVERY_DEFAULT = 10.0


def strategy_catalogue(settings: Any = None) -> list[dict[str, Any]]:
    """Every entry strategy in the strategies package (the exit overlay is always on, so it is not offered)."""
    from pumpfun_hft.strategies import STRATEGY_REGISTRY

    overlay = getattr(getattr(settings, "strategy", None), "exit_overlay", None) or "rug_avoidance"
    active = set(getattr(getattr(settings, "strategy", None), "active", None) or [])
    spaces = set(getattr(getattr(settings, "optimizer", None), "spaces", None) or [])
    tuned: dict[str, Any] = {}
    if settings is not None:
        from pumpfun_hft.optimizer.tuned import load_tuned

        tuned = load_tuned(dataset_root("real", settings))
    out = []
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        if name == overlay:
            continue
        doc = (sys.modules[cls.__module__].__doc__ or "").strip().splitlines()
        first = doc[0] if doc else ""
        summary = (first.split(": ", 1)[1] if ": " in first else first).replace("``", "").rstrip(".")
        label = name.replace("_", " ").capitalize().replace("Ml ", "ML ")
        if summary.lower().startswith(label.lower()):  # "Mean reversion on failed pumps" -> "on failed pumps"
            summary = summary[len(label):].strip()
        out.append({"name": name, "label": label, "summary": summary,
                    "needs_model": name == "ml_signal", "active": name in active, "space": "configured" if name in spaces else "derived",
                    "tuned": name in tuned})
    return out


def strategy_names() -> set[str]:
    return {s["name"] for s in strategy_catalogue()}


@dataclass
class TradeSpec:
    """One ``trade`` invocation. ``duration``: minutes of live stream (hfts / hftl) or hours of stored
    events (hft / hftr); 0 = until stopped / all events."""

    mode: str
    strategies: list[str]
    duration: float = 0.0
    compare: bool = True
    status_every: float = STATUS_EVERY_DEFAULT
    flatten: bool = True
    capital_sol: float | None = None

    @property
    def stream(self) -> bool:
        return MODES[self.mode]["feed"] == "stream"


def _num(d: dict[str, Any], key: str, default: float | None, lo: float, hi: float) -> float | None:
    v = d.get(key, default)
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not lo <= f <= hi:
        raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
    return f


def parse_spec(d: dict[str, Any], known: set[str]) -> TradeSpec:
    """Validate a builder form (untrusted JSON) into a TradeSpec; ValueError says what is wrong."""
    mode = str(d.get("mode") or "")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    raw = d.get("strategies") or []
    if not isinstance(raw, list):
        raise ValueError("strategies must be a list")
    strategies = list(dict.fromkeys(str(s) for s in raw))
    if not strategies:
        raise ValueError("pick at least one strategy")
    unknown = [s for s in strategies if s not in known]
    if unknown:
        raise ValueError(f"unknown strategy: {', '.join(unknown)}")
    stream = MODES[mode]["feed"] == "stream"
    duration = _num(d, "duration", 0.0, 0.0, 10_080.0 if stream else 8_760.0) or 0.0
    capital = _num(d, "capital_sol", None, 0.1, 100_000.0) if mode == "hfts" else None
    return TradeSpec(mode, strategies, duration, bool(d.get("compare", True)),
                     _num(d, "status_every", STATUS_EVERY_DEFAULT, 0.0, 3600.0) or 0.0, bool(d.get("flatten", True)), capital)


OPT_METHODS = ("bayesian", "genetic", "random", "grid")
OPT_OBJECTIVES = ("sharpe", "sortino", "profit_factor", "expectancy", "calmar", "total_return", "composite")


@dataclass
class OptimizeSpec:
    """One ``hftr optimize`` run: tune ``strategy`` on the recorded real events (``last_hours`` = 0: all)."""

    strategy: str
    method: str = "bayesian"
    trials: int = 32
    last_hours: float = 0.0
    objective: str = "sharpe"
    workers: int = 4
    final: bool = False


def parse_optimize(d: dict[str, Any], known: set[str]) -> OptimizeSpec:
    strategy = str(d.get("strategy") or "")
    if strategy not in known:
        raise ValueError("pick a strategy")
    method, objective = str(d.get("method") or "bayesian"), str(d.get("objective") or "sharpe")
    if method not in OPT_METHODS:
        raise ValueError(f"method must be one of {', '.join(OPT_METHODS)}")
    if objective not in OPT_OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OPT_OBJECTIVES)}")
    cpus = max(1, (os.cpu_count() or 2) - 1)
    return OptimizeSpec(strategy, method, int(_num(d, "trials", 32, 4, 400) or 32), _num(d, "last_hours", 0.0, 0.0, 100_000.0) or 0.0,
                        objective, int(_num(d, "workers", min(4, cpus), 1, cpus) or 1), bool(d.get("final", False)))


def optimize_args(spec: OptimizeSpec) -> list[str]:
    args = ["optimize", "--strategy", spec.strategy, "--method", spec.method, "--trials", str(spec.trials),
            "--objective", spec.objective, "--workers", str(spec.workers)]
    if spec.last_hours:
        args += ["--last-hours", _g(spec.last_hours)]
    if spec.final:
        args.append("--final")
    return args


def optimize_line(spec: OptimizeSpec) -> str:
    return " ".join(["hftr", *optimize_args(spec)])


def _g(v: float) -> str:
    return f"{v:g}"


def trade_args(spec: TradeSpec) -> tuple[list[str], list[str]]:
    """(global options placed before ``trade``, options of ``trade``) for this spec."""
    pre: list[str] = []
    if spec.mode == "hftl":
        pre += ["--set", "app.mode=live"]
    if spec.capital_sol is not None:
        pre += ["--set", f"live.paper_initial_capital_sol={_g(spec.capital_sol)}"]
    args: list[str] = []
    for s in spec.strategies:
        args += ["--strategy", s]
    if spec.stream:
        if spec.duration:
            args += ["--minutes", _g(spec.duration)]
        if spec.status_every != STATUS_EVERY_DEFAULT:
            args += ["--status-every", _g(spec.status_every)]
        if not spec.flatten:
            args.append("--no-flatten-on-exit")
        if spec.mode == "hftl":
            args.append("--confirm-live")
    else:
        if spec.duration:
            args += ["--hours", _g(spec.duration)]
        if not spec.compare:
            args.append("--no-compare")
    return pre, args


def powershell_line(spec: TradeSpec) -> str:
    """The command to type in a PowerShell window that loaded hft.ps1 (names and numbers only: no quoting needed)."""
    pre, args = trade_args(spec)
    return " ".join([spec.mode, *pre, "trade", *args])


def dataset_root(kind: str, settings: Any) -> Path:
    """Folder of a data set as hft.ps1 sets it (HFT_SYNTH_DIR / HFT_REAL_DIR), else the config's datasets.<kind>."""
    env = os.environ.get("HFT_SYNTH_DIR" if kind == "synthetic" else "HFT_REAL_DIR")
    return Path(env) if env else settings.datasets.root(kind)


def _profile_argv(profile: str, settings: Any, config: str | Path | None, python: str | None) -> list[str]:
    cfg = config or os.environ.get("HFT_CONFIG")
    cfg_args = ["--config", str(cfg)] if cfg and Path(cfg).exists() else []
    kind = MODES[profile]["dataset"]
    return [python or sys.executable, "-m", "pumpfun_hft.main", *cfg_args, "--profile", profile, "--dataset", kind,
            "--data-root", str(dataset_root(kind, settings))]


def cli_argv(spec: TradeSpec, settings: Any, config: str | Path | None = None, python: str | None = None) -> list[str]:
    """argv that runs the spec exactly like the hft.ps1 function would (no shell involved)."""
    pre, args = trade_args(spec)
    return [*_profile_argv(spec.mode, settings, config, python), *pre, "trade", *args]


def optimize_argv(spec: OptimizeSpec, settings: Any, config: str | Path | None = None, python: str | None = None) -> list[str]:
    return [*_profile_argv("hftr", settings, config, python), *optimize_args(spec)]


# --------------------------------------------------------------------------- running sessions
@dataclass
class Job:
    id: int
    kind: str               # "trade" | "optimize"
    mode: str               # profile the command runs under
    stream: bool            # a live-stream session (graceful stop through the stop file)
    command: str
    data_root: Path
    stop_file: Path
    proc: subprocess.Popen[bytes]
    started_ms: int
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=4000))
    total: int = 0          # lines ever read; index of the next line
    ended_ms: int | None = None
    stopping: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class JobRunner:
    """At most one job (a paper session, or an optimisation) per runner; its stdout + stderr are kept for the page.

    Stopping a live-stream session writes the stop file the CLI polls (``PUMPFUN_STOP_FILE``), so it ends
    the normal way: positions closed, final snapshot, events compacted. A replay is simply terminated."""

    def __init__(self, settings: Any, project_root: Path, config: str | Path | None = None, python: str | None = None,
                 kill_after_s: float = 90.0) -> None:
        self.settings = settings
        self.project_root = project_root
        self.config = config
        self.python = python
        self.kill_after_s = kill_after_s
        self.job: Job | None = None
        self._ids = 0
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    def running(self) -> bool:
        return self.job is not None and self.job.proc.poll() is None

    def start(self, spec: TradeSpec) -> Job:
        if spec.mode == "hftl":
            raise ValueError("The dashboard never starts live trading. Copy the command into PowerShell instead.")
        return self._launch(cli_argv(spec, self.settings, self.config, self.python), "trade", spec.mode, spec.stream,
                            powershell_line(spec), "A session is already running; stop it first.")

    def start_optimize(self, spec: OptimizeSpec) -> Job:
        return self._launch(optimize_argv(spec, self.settings, self.config, self.python), "optimize", "hftr", False,
                            optimize_line(spec), "An optimisation is already running; stop it first.", strategy=spec.strategy)

    def _launch(self, argv: list[str], kind: str, mode: str, stream: bool, command: str, busy: str, **extra: Any) -> Job:
        with self._lock:
            if self.running():
                raise ValueError(busy)
            root = dataset_root(MODES[mode]["dataset"], self.settings)
            root.mkdir(parents=True, exist_ok=True)
            stop_file = root / f"dashboard-{kind}.stop"
            stop_file.unlink(missing_ok=True)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "TERM": "dumb",
                   "COLUMNS": "180", "PUMPFUN_STOP_FILE": str(stop_file)}
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(argv, cwd=self.project_root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, env=env, creationflags=flags)
            self._ids += 1
            job = Job(self._ids, kind, mode, stream, command, root, stop_file, proc, int(time.time() * 1000))
            job.extra = dict(extra)
            job.lines.append(f"PS> {job.command}")
            job.total = 1
            self.job = job
            threading.Thread(target=self._pump, args=(job,), daemon=True).start()
            return job

    def _pump(self, job: Job) -> None:
        assert job.proc.stdout is not None
        for raw in iter(job.proc.stdout.readline, b""):
            job.lines.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            job.total += 1
        job.proc.wait()
        job.stop_file.unlink(missing_ok=True)
        job.lines.append(f"[{'session' if job.kind == 'trade' else job.kind} ended, exit code {job.proc.returncode}]")
        job.total += 1
        job.ended_ms = int(time.time() * 1000)  # last: readers treat it as "everything above is final"

    def stop(self) -> bool:
        job = self.job
        if job is None or job.proc.poll() is not None:
            return False
        job.stopping = True
        if job.stream:
            job.stop_file.touch()
            job.lines.append("[stop requested: closing positions and ending the session ...]")
            job.total += 1
            threading.Timer(self.kill_after_s, lambda: job.proc.poll() is None and job.proc.kill()).start()
        else:
            job.proc.terminate()
        return True

    def status(self, since: int = 0) -> dict[str, Any]:
        job = self.job
        if job is None:
            return {"job": None, "lines": [], "next": 0}
        first = job.total - len(job.lines)  # index of the oldest line still kept
        start = max(since, first)
        lines = list(job.lines)[start - first:] if start < job.total else []
        return {"job": {"id": job.id, "kind": job.kind, "mode": job.mode, "command": job.command, "running": job.proc.poll() is None,
                        "stopping": job.stopping, "started_ms": job.started_ms, "ended_ms": job.ended_ms,
                        "exit_code": job.proc.returncode, "dataset": MODES[job.mode]["dataset"], **job.extra},
                "lines": lines, "next": job.total, "skipped": max(0, first - since) if since else 0}

    def meta_path(self) -> Path | None:
        """SQLite store of the newest session's data set (where its Live monitor snapshots go)."""
        return self.job.data_root / "meta.sqlite" if self.job is not None else None

    def shutdown(self) -> None:
        job = self.job
        if job is None or job.proc.poll() is not None:
            return
        self.stop()
        try:
            job.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            job.proc.kill()


# --------------------------------------------------------------------------- status bar
class HealthProbe:
    """RPC reachability (getSlot round trip, cached) + trading-session freshness for the status bar.
    Only the provider's host name is ever returned: RPC URLs carry API keys."""

    def __init__(self, ttl_s: float = 15.0, timeout_s: float = 4.0, secrets: Any = None) -> None:
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self._secrets = secrets
        self._rpc: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._secrets is None:
            from pumpfun_hft.core.config import Secrets

            return Secrets.load()
        return self._secrets

    def rpc(self) -> dict[str, Any]:
        with self._lock:
            if self._rpc is not None and time.monotonic() - self._rpc[0] < self.ttl_s:
                return self._rpc[1]
            res = self._probe()
            self._rpc = (time.monotonic(), res)
            return res

    def _probe(self) -> dict[str, Any]:
        import httpx

        secrets = self._load()
        url = secrets.get("solana_rpc_url")
        ws = bool(secrets.get("solana_ws_url"))
        if not url:
            return {"state": "missing", "provider": None, "ws": ws, "detail": "SOLANA_RPC_URL is not set in .env"}
        host = urlparse(url).hostname or "?"
        provider = "Helius" if "helius" in host else host
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getSlot"}, timeout=self.timeout_s)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 429:
                return {"state": "limited", "provider": provider, "ws": ws, "ms": ms, "detail": "rate limited (HTTP 429)"}
            r.raise_for_status()
            slot = r.json().get("result")
            return {"state": "slow" if ms > 1500 else "ok", "provider": provider, "ws": ws, "ms": ms, "slot": slot,
                    "detail": f"slot {slot:,}" if isinstance(slot, int) else ""}
        except Exception as exc:  # noqa: BLE001 - any failure means "down"; the message never contains the URL
            return {"state": "down", "provider": provider, "ws": ws, "detail": type(exc).__name__}

    def check(self, live_state: dict[str, Any] | None, job: dict[str, Any] | None, stale_ms: int = 15_000) -> dict[str, Any]:
        rpc = self.rpc()
        session = "none"
        if live_state and live_state.get("value"):
            age = int(time.time() * 1000) - int(live_state.get("updated_ms") or 0)
            session = "ended" if live_state["value"].get("stopped") else "running" if age <= stale_ms else "stale"
        if job and job.get("running"):
            session = "starting" if session in ("none", "ended", "stale") else session
        level = ("bad" if rpc["state"] in ("missing", "down") else
                 "warn" if rpc["state"] in ("limited", "slow") or not rpc.get("ws") or session == "stale" else "good")
        return {"level": level, "rpc": rpc, "session": session}
