"""Strategy builder: command composition, validation, the guarded run / stop endpoints, the session
runner (output capture, graceful stop file) and the status-bar health probe."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pumpfun_hft.dashboard import builder as B
from pumpfun_hft.dashboard.app import create_app
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.tests.conftest import make_settings

KNOWN = B.strategy_names()
OK = {"x-pumpfun-dashboard": "1"}


def spec(**kw: Any) -> B.TradeSpec:
    return B.parse_spec({"mode": "hfts", "strategies": ["smart_money"], "duration": 60, **kw}, KNOWN)


# --------------------------------------------------------------------------- composing commands
def test_catalogue_is_the_strategies_folder_without_the_exit_overlay() -> None:
    names = [s["name"] for s in B.strategy_catalogue()]
    assert "smart_money" in names and "momentum_ignition" in names and "rug_avoidance" not in names
    assert all(s["summary"] for s in B.strategy_catalogue())


@pytest.mark.parametrize("form, line", [
    ({}, "hfts trade --strategy smart_money --minutes 60"),
    ({"strategies": ["smart_money", "sniper"], "duration": 0, "status_every": 30, "flatten": False},
     "hfts trade --strategy smart_money --strategy sniper --status-every 30 --no-flatten-on-exit"),
    ({"capital_sol": 25}, "hfts --set live.paper_initial_capital_sol=25 trade --strategy smart_money --minutes 60"),
    ({"mode": "hftr", "duration": 6, "compare": False}, "hftr trade --strategy smart_money --hours 6 --no-compare"),
    ({"mode": "hft", "duration": 0, "capital_sol": 25}, "hft trade --strategy smart_money"),
    ({"mode": "hftl", "duration": 30}, "hftl --set app.mode=live trade --strategy smart_money --minutes 30 --confirm-live"),
])
def test_powershell_line(form: dict[str, Any], line: str) -> None:
    assert B.powershell_line(spec(**form)) == line


@pytest.mark.parametrize("form, error", [
    ({"strategies": []}, "at least one strategy"),
    ({"strategies": ["rm -rf"]}, "unknown strategy"),
    ({"mode": "HFTS"}, "mode must be one of"),
    ({"duration": -5}, "duration must be between"),
    ({"duration": "soon"}, "duration must be a number"),
])
def test_invalid_forms_are_refused(form: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        spec(**form)


def test_argv_runs_the_same_command_as_hft_ps1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HFT_REAL_DIR", str(tmp_path / "real"))
    monkeypatch.delenv("HFT_CONFIG", raising=False)
    argv = B.cli_argv(spec(), make_settings(), python="py")
    assert argv == ["py", "-m", "pumpfun_hft.main", "--profile", "hfts", "--dataset", "real", "--data-root",
                    str(tmp_path / "real"), "trade", "--strategy", "smart_money", "--minutes", "60"]


# --------------------------------------------------------------------------- endpoints
@pytest.fixture()
def client(tmp_path: Path) -> Any:
    s = make_settings(**{"paths.data_dir": str(tmp_path / "data"), "paths.reports_dir": str(tmp_path / "reports"),
                         "paths.sqlite_file": str(tmp_path / "data" / "meta.sqlite")})

    class Secrets:
        def get(self, name: str) -> str | None:
            return None

    app = create_app(s, MetaStore(tmp_path / "meta.sqlite"), health=B.HealthProbe(secrets=Secrets()))
    return TestClient(app, base_url=f"http://127.0.0.1:{s.dashboard.port}")


def test_builder_page_and_command_endpoint(client: TestClient) -> None:
    page = client.get("/builder").text
    assert "Strategy builder" in page and "bt-term" in page and "value='hfts' checked" in page and "PowerShell output" in page
    r = client.post("/api/builder/command", json={"mode": "hfts", "strategies": ["sniper"], "duration": 15}, headers=OK).json()
    assert r == {"powershell": "hfts trade --strategy sniper --minutes 15", "runnable": True}
    r = client.post("/api/builder/command", json={"mode": "hftl", "strategies": ["sniper"]}, headers=OK).json()
    assert r["runnable"] is False
    assert "error" in client.post("/api/builder/command", json={"mode": "hfts", "strategies": []}, headers=OK).json()


def test_session_endpoints_refuse_other_sites_and_live_trading(client: TestClient) -> None:
    body = {"mode": "hfts", "strategies": ["sniper"]}
    assert client.post("/api/jobs", json=body).status_code == 403                                   # no custom header
    assert client.post("/api/jobs", json=body, headers={**OK, "origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/jobs", json=body, headers={**OK, "host": "evil.example:8050"}).status_code == 403  # rebinding
    assert client.post("/api/jobs/stop").status_code == 403
    r = client.post("/api/jobs", json={"mode": "hftl", "strategies": ["sniper"]}, headers=OK)
    assert r.status_code == 400 and "never starts live trading" in r.json()["error"]
    assert client.get("/api/jobs/current").json() == {"job": None, "lines": [], "next": 0}


def test_health_without_rpc_is_red_and_never_shows_urls(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["level"] == "bad" and h["rpc"]["state"] == "missing" and h["session"] == "none"


def test_health_hides_the_rpc_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class Secrets:
        def get(self, name: str) -> str | None:
            return {"solana_rpc_url": "https://mainnet.helius-rpc.com/?api-key=SECRET", "solana_ws_url": "wss://x"}.get(name)

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, json={"result": 123}, request=httpx.Request("POST", a[0])))
    h = B.HealthProbe(secrets=Secrets()).check({"updated_ms": int(time.time() * 1000), "value": {"stopped": False}}, None)
    assert h["level"] == "good" and h["rpc"]["provider"] == "Helius" and h["session"] == "running"
    assert "SECRET" not in repr(h)


# --------------------------------------------------------------------------- running sessions
def test_runner_captures_output_and_stops_through_the_stop_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in child that behaves like `trade`: prints, polls PUMPFUN_STOP_FILE, ends cleanly."""
    child = tmp_path / "child.py"
    child.write_text("import os, time\nprint('hfts trading · started', flush=True)\n"
                     "while not os.path.exists(os.environ['PUMPFUN_STOP_FILE']):\n    time.sleep(0.05)\n"
                     "print('stop requested from the dashboard: closing positions', flush=True)\n", encoding="utf-8")
    monkeypatch.setenv("HFT_REAL_DIR", str(tmp_path / "real"))
    monkeypatch.setattr(B, "cli_argv", lambda *a, **k: [sys.executable, str(child)])
    runner = B.JobRunner(make_settings(), tmp_path)
    job = runner.start(spec())
    with pytest.raises(ValueError, match="already running"):
        runner.start(spec())
    for _ in range(200):
        if any("started" in ln for ln in job.lines):
            break
        time.sleep(0.05)
    assert runner.stop() and job.stop_file.exists()
    job.proc.wait(timeout=20)
    for _ in range(100):
        if job.ended_ms:
            break
        time.sleep(0.05)
    st = runner.status()
    lines = st["lines"]
    assert lines[0] == "PS> hfts trade --strategy smart_money --minutes 60" and "hfts trading · started" in lines
    assert "stop requested from the dashboard: closing positions" in lines and lines[-1] == "[session ended, exit code 0]"
    assert st["job"]["running"] is False and st["job"]["exit_code"] == 0 and not job.stop_file.exists()
    assert runner.status(since=st["next"])["lines"] == []
    assert runner.meta_path() == tmp_path / "real" / "meta.sqlite"
