"""Command-line workflows end to end on an isolated data directory (Typer ``CliRunner``):
synthetic data → verify → backtest → SQL → report → Monte Carlo → wallets → dashboard export,
plus the guard rails (secrets never printed, strict overrides, live trading needs explicit consent)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner, Result

from pumpfun_hft.main import app
from pumpfun_hft.utils.docgen import PACKAGES, generate_module_docs
from pumpfun_hft.utils.logging import shutdown_logging

runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def _restore_logging() -> Any:
    """Each CLI invocation configures file logging under the temp dir; put the process back afterwards."""
    root = logging.getLogger("pumpfun_hft")
    saved = (list(root.handlers), root.propagate, root.level)
    yield
    shutdown_logging()
    root.handlers[:] = saved[0]
    root.propagate = saved[1]
    root.setLevel(saved[2])


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cli")


def cli_args(home: Path, *cmd: str) -> list[str]:
    sets = {"paths.data_dir": home / "data", "paths.duckdb_file": home / "data" / "warehouse.duckdb",
            "paths.sqlite_file": home / "data" / "meta.sqlite", "paths.logs_dir": home / "logs",
            "paths.reports_dir": home / "reports", "paths.models_dir": home / "models", "logging.console": False}
    out: list[str] = []
    for k, v in sets.items():
        out += ["--set", f"{k}={v}"]
    return out + list(cmd)


def invoke(home: Path, *cmd: str, code: int = 0) -> Result:
    res = runner.invoke(app, cli_args(home, *cmd))
    assert res.exit_code == code, (res.output, repr(res.exception))
    return res


@pytest.fixture(scope="module")
def workflow(home: Path) -> Path:
    """Synthetic market + one saved backtest, shared by the workflow tests below."""
    out = invoke(home, "synth", "--hours", "0.5", "--seed", "11").output
    assert "synthetic market:" in out
    out = invoke(home, "backtest", "--no-report", "--seed", "3").output
    assert "saved to" in out and "Synthetic data" in out
    runs = list((home / "reports" / "runs").iterdir())
    assert len(runs) == 1 and (runs[0] / "result.json").exists()
    return runs[0]


# --------------------------------------------------------------------------- guard rails
def test_check_config_never_prints_secrets(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jwt = "eyJ" + "hbGciOiJFUzI1NiJ9.eyJzdWIiOiJjbGkifQ.c2VjcmV0LXNpZ25hdHVyZQ"
    key = "5" + "Kd3NBUAdUnhyzenEwVLy9pBKxSwXvE9FMPyR4UKZvpe6E3AgLr"
    monkeypatch.setenv("PUMPFUN_JWT", jwt)
    monkeypatch.setenv("PRIVATE_KEY", key)
    out = invoke(home, "check-config").output
    assert "config fingerprint" in out and "pumpfun_jwt" in out and "yes" in out
    assert jwt not in out and key not in out
    for f in (home / "logs").glob("*.log"):
        text = f.read_text()
        assert jwt not in text and key not in text


def test_overrides_are_strict(home: Path) -> None:
    assert runner.invoke(app, cli_args(home, "--set", "no-equals-sign", "check-config")).exit_code == 2
    res = runner.invoke(app, cli_args(home, "--set", "backtest.initial_captial_sol=5", "check-config"))  # typo in the key
    assert res.exit_code != 0 and isinstance(res.exception, ValidationError)


def test_live_trading_requires_config_and_flag(home: Path) -> None:
    for extra in ([], ["--confirm-live"]):  # app.mode is "paper" by default
        assert "Refusing to trade live" in invoke(home, "live", *extra, code=2).output
    res = runner.invoke(app, cli_args(home, "--set", "app.mode=live", "live"))  # config alone is not enough either
    assert res.exit_code == 2 and "Refusing to trade live" in res.output


# --------------------------------------------------------------------------- research workflow
def test_verify_data(home: Path, workflow: Path) -> None:
    out = invoke(home, "verify-data").output
    assert "checksum mismatches: 0" in out


def test_query_warehouse(home: Path, workflow: Path) -> None:
    out = invoke(home, "query", "select kind, count(*) as n from events group by kind order by n desc").output
    assert "trade" in out and "create" in out
    out = invoke(home, "query", "select count(*) as runs from runs").output
    assert "1" in out


def test_report_and_montecarlo(home: Path, workflow: Path) -> None:
    invoke(home, "report", "--formats", "html,json")
    rep = workflow / "report"
    assert (rep / "report.html").stat().st_size > 1_000_000  # Plotly is inlined: the file opens offline
    payload = json.loads((rep / "report.json").read_text())
    assert payload["run_id"] == workflow.name and payload["synthetic"] is True
    out = invoke(home, "montecarlo", "--sims", "200").output
    assert "prob_ruin" in out


def test_wallets_and_dashboard_export(home: Path, workflow: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, cli_args(home, "wallets", "--top", "5"))
    assert res.exit_code in (0, 1)  # 1 = no wallet closed two round trips in a 30-minute market
    if res.exit_code == 0:
        assert "smart_score" in res.output
    out = tmp_path / "dash.html"
    invoke(home, "dashboard-export", "--out", str(out))
    assert out.exists() and workflow.name in out.read_text(encoding="utf-8")


def test_paper_replay_runs_the_live_engine_offline(home: Path, workflow: Path) -> None:
    out = invoke(home, "paper-replay", "--strategy", "momentum_ignition").output
    assert "Live engine replay" in out and "live engine (paper)" in out and "backtest" in out
    assert "open positions at end" in out


def test_empty_store_is_reported(tmp_path: Path) -> None:
    res = runner.invoke(app, cli_args(tmp_path, "backtest", "--no-report"))
    assert res.exit_code == 1 and "event store is empty" in res.output


# --------------------------------------------------------------------------- documentation
def test_every_module_is_documented(tmp_path: Path) -> None:
    doc = generate_module_docs(tmp_path / "MODULES.md").read_text(encoding="utf-8")
    from pumpfun_hft.core.config import PACKAGE_DIR

    undocumented = []
    for pkg in PACKAGES:
        assert f"## `pumpfun_hft.{pkg}`" in doc
        init = (PACKAGE_DIR / pkg / "__init__.py").read_text(encoding="utf-8")
        for section in ("Purpose", "Architecture", "Data flow", "Inputs", "Outputs", "Example"):
            assert section in init, f"{pkg}: package docstring lacks '{section}'"
        for f in (PACKAGE_DIR / pkg).glob("*.py"):
            src = f.read_text(encoding="utf-8").lstrip()
            if not src.startswith(('"""', 'r"""')):
                undocumented.append(str(f.relative_to(PACKAGE_DIR)))
    assert not undocumented, undocumented
