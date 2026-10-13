"""Session archive, Recent Session Breakdown, tuned parameters and the Strategy Parameters page."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

from pumpfun_hft.dashboard import builder as B
from pumpfun_hft.dashboard.app import create_app
from pumpfun_hft.dashboard.trading_views import score_text, study_verdict
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.execution import sessions as SS
from pumpfun_hft.optimizer import tuned as TU
from pumpfun_hft.optimizer.space import ParamSpace, derive_space
from pumpfun_hft.risk.portfolio import Portfolio, TradeRecord
from pumpfun_hft.tests.conftest import make_settings

OK = {"x-pumpfun-dashboard": "1"}


def trade(i: int, strategy: str, pnl: float, ret: float, exit_ms: int, reason: str = "take_profit") -> TradeRecord:
    return TradeRecord(i, f"M{i:043d}", strategy, None, "other", exit_ms - 30_000, exit_ms, 0.1, 0.1 + pnl, pnl, ret, 0.0, 0.0, 0.0,
                       30.0, reason, 2, 0.002, 0.001, 0.0, 1e-8, 1e-8)


def fake_trader(settings: Any, trades: list[TradeRecord]) -> Any:
    pf = Portfolio(10 * 10**9)
    pf.trades.extend(trades)
    pf.equity_curve += [(1_788_000_000_000, 10.0, 10.0, 0.0, 0), (1_788_000_060_000, 10.0 + sum(t.pnl_sol for t in trades), 10.0, 0.0, 0)]
    runtime = SimpleNamespace(strategies=[SimpleNamespace(name="sniper"), SimpleNamespace(name="smart_money")],
                              signal_records=[{"ts_ms": 1, "mint": "m", "strategy": "sniper", "action": "BUY", "confidence": 70.0,
                                               "reason": "r", "outcome": "submitted"}],
                              counts={"sniper:BUY:submitted": 3, "sniper:BUY:cost_gate": 5, "smart_money:BUY:low_confidence": 2})
    return SimpleNamespace(s=settings, session="paper", runtime=runtime, portfolio=pf, fills=[], initial_sol=10.0, events_processed=1234,
                           collector_status=lambda: {"events": 1300, "reconnects": 1}, last_snapshot={"positions": []})


@pytest.fixture()
def archive(tmp_path: Path) -> Any:
    s = make_settings()
    trades = [trade(1, "sniper", 0.05, 0.5, 1_788_000_010_000), trade(2, "sniper", -0.02, -0.2, 1_788_000_020_000, "stop_loss"),
              trade(3, "smart_money", 0.01, 0.1, 1_788_000_030_000)]
    rec = SS.SessionRecorder(fake_trader(s, trades), tmp_path / "sessions", "hfts", {"sniper": "study-abc"})
    return s, rec, tmp_path / "sessions"


# --------------------------------------------------------------------------- archive
def test_recorder_writes_a_running_then_final_session(archive: Any) -> None:
    _, rec, root = archive
    rec.save()
    [info] = SS.list_sessions(root)
    assert info["status"] == "running" and info["profile"] == "hfts" and info["strategies"] == ["sniper", "smart_money"]
    assert info["tuned"] == {"sniper": "study-abc"} and "sniper" in info["params"] and "rug_avoidance" in info["params"]
    assert info["summary"]["closed_trades"] == 3 and info["summary"]["wins"] == 2 and info["summary"]["stream"]["events"] == 1300
    rec.save(final=True, market={"events": 10})
    [info] = SS.list_sessions(root)
    assert info["status"] == "ended" and info["ended_ms"] and info["market"] == {"events": 10}
    assert SS.load_frame(root, rec.session_id, "trades").height == 3 and SS.load_frame(root, rec.session_id, "signals").height == 1
    assert SS.load_frame(root, "../escape", "trades").is_empty()


def test_a_session_that_stopped_saving_is_interrupted(archive: Any) -> None:
    _, rec, root = archive
    d = rec.save()
    info = json.loads((d / "session.json").read_text())
    info["updated_ms"] = int(time.time() * 1000) - SS.STALE_AFTER_MS - 1
    (d / "session.json").write_text(json.dumps(info))
    assert SS.list_sessions(root)[0]["status"] == "interrupted"


def test_strategy_breakdown_and_pooled_evidence(archive: Any) -> None:
    _, rec, root = archive
    rec.save(final=True)
    info = SS.list_sessions(root)[0]
    b = {r["strategy"]: r for r in SS.strategy_breakdown(info, SS.load_frame(root, rec.session_id, "trades")).iter_rows(named=True)}
    assert b["sniper"]["trades"] == 2 and b["sniper"]["wins"] == 1 and b["sniper"]["pnl_sol"] == pytest.approx(0.03)
    assert b["sniper"]["profit_factor"] == pytest.approx(2.5) and b["sniper"]["buy_signals"] == 8 and b["sniper"]["submitted"] == 3
    assert b["smart_money"]["trades"] == 1 and b["smart_money"]["profit_factor"] is None
    ev = SS.strategy_evidence(root, "sniper")
    assert ev["sessions"] == 1 and ev["trades"] == 2 and ev["win_rate"] == 0.5
    assert SS.strategy_evidence(root, "whale_follow") == {"sessions": 0, "trades": 0}


def test_market_stats_of_a_session_window(tmp_path: Path, events: pl.DataFrame) -> None:
    from pumpfun_hft.collectors.storage import ParquetEventStore

    store = ParquetEventStore(tmp_path / "events", MetaStore(tmp_path / "meta.sqlite"))
    ev = events.head(5000)
    store.write(ev)
    t0, t1 = int(ev["ts_ms"].min()), int(ev["ts_ms"].max()) + 1
    m = SS.market_stats(store, t0, t1)
    assert m["events"] == 5000 and m["tokens_created"] == int((ev["kind"] == "create").sum())
    curve = ev.filter(pl.col("kind") == "trade")
    pools = ev.filter(pl.col("kind").is_in(["amm_buy", "amm_sell"]))
    assert m["curve_trades"] == curve.height and m["volume_sol"] == pytest.approx(curve["sol_amount"].sum() / 1e9)
    assert m["pool_trades"] == pools.height and m["unknown_pool_trades"] == int(pools["mint"].is_null().sum())


def test_pool_events_stay_out_of_the_volume(tmp_path: Path, events: pl.DataFrame) -> None:
    from pumpfun_hft.collectors.storage import ParquetEventStore

    store = ParquetEventStore(tmp_path / "events", MetaStore(tmp_path / "meta.sqlite"))
    ev = events.filter(pl.col("kind") == "trade").head(100)
    alien = ev.head(3).with_columns(pl.lit("amm_buy").alias("kind"), pl.lit(None, dtype=pl.Utf8).alias("mint"),
                                    pl.lit(80_000 * 10**9).alias("sol_amount"), (pl.col("seq") + 10_000).alias("seq"))
    store.write(pl.concat([ev, alien]))
    m = SS.market_stats(store, int(ev["ts_ms"].min()), int(ev["ts_ms"].max()) + 1)
    assert m["unknown_pool_trades"] == 3 and m["pool_trades"] == 3 and m["curve_trades"] == 100
    assert m["volume_sol"] == pytest.approx(ev["sol_amount"].sum() / 1e9)


# --------------------------------------------------------------------------- tuned parameters
def test_tuned_file_round_trip(tmp_path: Path) -> None:
    TU.save_tuned(tmp_path, "sniper", {"params": {"min_creator_score": 40.0}, "study_id": "s1"})
    TU.save_tuned(tmp_path, "smart_money", {"params": {"min_smart_buyers": 3}, "study_id": "s2"})
    t = TU.load_tuned(tmp_path)
    assert set(t) == {"sniper", "smart_money"} and t["sniper"]["saved_ms"]
    assert TU.tuned_overrides(t, ["sniper"]) == {"strategy.params.sniper.min_creator_score": 40.0}
    assert TU.remove_tuned(tmp_path, "sniper") and not TU.remove_tuned(tmp_path, "sniper") and set(TU.load_tuned(tmp_path)) == {"smart_money"}


def test_only_hfts_loads_the_tuned_parameters(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from pumpfun_hft.main import STATE, app

    TU.save_tuned(tmp_path, "smart_money", {"params": {"min_smart_buyers": 4}, "study_id": "s9"})
    runner = CliRunner()
    for profile, expected in (("hfts", 4), ("hftr", make_settings().strategy.params["smart_money"]["min_smart_buyers"])):
        res = runner.invoke(app, ["--set", "logging.console=false", "--profile", profile, "--data-root", str(tmp_path), "check-config"])
        assert res.exit_code == 0, res.output
        assert STATE["settings"].strategy.params["smart_money"]["min_smart_buyers"] == expected, profile
    res = runner.invoke(app, ["--set", "logging.console=false", "--set", "strategy.params.smart_money.min_smart_buyers=7",
                              "--profile", "hfts", "--data-root", str(tmp_path), "check-config"])
    assert STATE["settings"].strategy.params["smart_money"]["min_smart_buyers"] == 7  # an explicit --set still wins


def test_derived_space_brackets_the_current_values() -> None:
    params = make_settings().strategy.params["whale_follow"]
    space = derive_space(dict(params))
    assert set(space) >= {"min_whale_buy_sol", "min_wallet_score", "max_rug_prob", "min_wallet_closed"}
    for k, spec in space.items():
        v = params[k]
        if spec.type in ("float", "int"):
            assert spec.low <= v <= spec.high, k
        if "prob" in k or "score" in k:
            assert 0.0 <= spec.low and spec.high <= 1.0
    import numpy as np

    ParamSpace(space).sample(np.random.default_rng(0))  # a valid space
    assert derive_space({"model_path": "latest", "zero": 0.0, "flag": True}) .keys() == {"flag"}


def test_study_verdicts_in_plain_language() -> None:
    base = {"selected_val_score": 0.8, "baseline_val_score": 0.3, "pbo": {"pbo": 0.2},
            "selected_val_metrics": {"n_trades": 40, "total_return": 0.03}}
    losing = {**base, "selected_val_score": -70.7, "baseline_val_score": -93.6,
              "selected_val_metrics": {"n_trades": 40, "total_return": -0.021}}
    assert study_verdict(losing, 15) == ("warn", study_verdict(losing, 15)[1]) and "still losing money there (-2.10%)" in         study_verdict(losing, 15)[1]
    assert study_verdict(base, 15)[0] == "good"
    assert study_verdict({**base, "selected_val_score": 0.2}, 15)[0] == "warn"
    assert study_verdict({**base, "pbo": {"pbo": 0.8}}, 15)[1].startswith("Beat the current parameters")
    level, text = study_verdict({**base, "selected_val_score": -1000 + 6, "selected_val_metrics": {"n_trades": 6}}, 15)
    assert level == "bad" and "6 < 15" in text
    assert score_text(-994.0, 15) == "too few trades (6)" and score_text(1.2345, 15) == "1.234"


# --------------------------------------------------------------------------- pages and endpoints
@pytest.fixture()
def client(tmp_path: Path, archive: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    s, rec, _ = archive
    real = tmp_path / "real"
    monkeypatch.setenv("HFT_REAL_DIR", str(real))
    rec.dir = real / "reports" / "sessions" / rec.session_id
    rec.save(final=True, market={"events": 999, "tokens_created": 12, "by_kind": {"trade": 900, "create": 12}, "volume_sol": 55.5})
    for sid, dataset in (("study-real0001", "real"), ("study-syn00001", "synthetic")):
        d = real / "reports" / "optimize" / sid
        d.mkdir(parents=True)
        (d / "study.json").write_text(json.dumps({
            "study_id": sid, "strategy": "sniper", "method": "bayesian", "objective": "sharpe", "dataset": dataset,
            "created_ms": 1_788_000_000_000, "n_events": 5000, "n_trials": 32, "data_start_ms": 1, "data_end_ms": 2,
            "selected_params": {"min_creator_score": 40.0, "max_bundled_buyers": 3}, "baseline_params": {"min_creator_score": 25.0,
            "max_bundled_buyers": 2}, "selected_val_score": 1.1, "baseline_val_score": 0.4, "selected_train_score": 1.5,
            "baseline_train_score": 0.6, "selected_val_metrics": {"n_trades": 30, "total_return": 0.05},
            "baseline_val_metrics": {"n_trades": 25, "total_return": 0.01}, "selected_train_metrics": {"n_trades": 60},
            "baseline_train_metrics": {"n_trades": 50}, "dsr": {"dsr": 0.9}, "pbo": {"pbo": 0.2}}))

    class Secrets:
        def get(self, name: str) -> str | None:
            return None

    app = create_app(s, MetaStore(tmp_path / "meta.sqlite"), health=B.HealthProbe(secrets=Secrets()))
    return TestClient(app, base_url=f"http://127.0.0.1:{s.dashboard.port}"), rec, real


def test_session_breakdown_page_and_downloads(client: Any) -> None:
    c, rec, _ = client
    page = c.get("/sessions").text
    for part in ("Recent Session Breakdown", "By strategy", "PnL by strategy", "Cumulative PnL per strategy", "Exit reasons",
                 "What happened to the buy signals", "Market activity recorded", "Parameters each strategy ran with",
                 "Closed trades (3", "tuned · study-abc", f"value='{rec.session_id}' selected"):
        assert part in page, part
    csv = c.get(f"/api/sessions/{rec.session_id}/trades.csv")
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("trade_id,mint,strategy") and len(csv.text.splitlines()) == 4
    assert c.get(f"/api/sessions/{rec.session_id}/secrets.csv").status_code == 404
    assert c.get("/api/sessions/nope/trades.csv").status_code == 404


def test_strategy_parameters_page_and_saving_for_hfts(client: Any) -> None:
    c, _, real = client
    page = c.get("/params?strategy=sniper").text
    assert "Strategy Parameters" in page and "Studies of sniper" in page and "Save for hfts" in page
    assert "Beat the current parameters on unseen (validation) data." in page and "Live paper record" in page
    r = c.post("/api/optimize/command", json={"strategy": "sniper", "trials": 16, "last_hours": 24}, headers=OK).json()
    assert r["powershell"] == "hftr optimize --strategy sniper --method bayesian --trials 16 --objective sharpe --workers " + \
        r["powershell"].split("--workers ")[1].split()[0] + " --last-hours 24"
    assert c.post("/api/tuned", json={"study_id": "study-real0001"}).status_code == 403          # guarded
    assert c.post("/api/tuned", json={"study_id": "study-syn00001"}, headers=OK).status_code == 400  # synthetic data
    assert c.post("/api/tuned", json={"study_id": "study-real0001"}, headers=OK).json()["saved"] == "sniper"
    assert TU.load_tuned(real)["sniper"]["params"]["min_creator_score"] == 40.0
    page = c.get("/params?strategy=sniper").text
    assert "active for hfts" in page and "hfts uses the tuned parameters" in page
    assert c.get("/builder").text.count("tuned for hfts") == 1                       # the builder shows which ones are tuned
    assert c.post("/api/tuned/remove", json={"strategy": "sniper"}, headers=OK).json() == {"removed": True}
    assert "sniper" not in TU.load_tuned(real)
