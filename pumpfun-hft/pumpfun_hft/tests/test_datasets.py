"""Synthetic vs real data sets (``--dataset`` / hft.ps1's hft and hftr): each lives in its own folder, the
CLI refuses to mix them, recorded events can be found and copied out of a mixed folder, and ML labels
are only trained on where the data covers the whole label window."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner, Result

from pumpfun_hft.collectors.datasets import describe, detect_kind, find_event_dirs, import_events, read_marker, write_marker
from pumpfun_hft.collectors.storage import ParquetEventStore
from pumpfun_hft.core.config import dataset_path_overrides, load_settings
from pumpfun_hft.main import app
from pumpfun_hft.ml.dataset import build_snapshot_dataset, feature_columns, window_complete
from pumpfun_hft.ml.models import train_and_evaluate
from pumpfun_hft.ml.rug_model import TrainedRugModel, build_rug_scorer
from pumpfun_hft.utils.logging import shutdown_logging

runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def _restore_logging() -> Any:
    root = logging.getLogger("pumpfun_hft")
    saved = (list(root.handlers), root.propagate, root.level)
    yield
    shutdown_logging()
    root.handlers[:] = saved[0]
    root.propagate = saved[1]
    root.setLevel(saved[2])


def cli(tmp: Path, dataset: str | None, root: Path | None, *cmd: str, code: int = 0) -> Result:
    args = ["--set", "logging.console=false", "--set", f"paths.logs_dir={tmp / 'logs'}"]
    if dataset:
        args += ["--dataset", dataset, "--data-root", str(root)]
    res = runner.invoke(app, args + list(cmd))
    assert res.exit_code == code, (res.output, repr(res.exception))
    return res


def flat(res: Result) -> str:
    """Output with line wrapping undone (Rich wraps long lines at the terminal width)."""
    return " ".join(res.output.split())


def as_recorded(events: pl.DataFrame, root: Path, shift_ms: int = 0, prefix: str = "") -> Path:
    """Write events into ``root/events`` as a recorder would (no synthetic truth file): a 'real' store."""
    ev = events.with_columns((pl.col("ts_ms") + shift_ms).alias("ts_ms"), (pl.lit(prefix) + pl.col("mint")).alias("mint"))
    ParquetEventStore(root / "events").write(ev)
    return root


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("ds")
    out = flat(cli(tmp, "synthetic", tmp / "syn", "synth", "--hours", "2", "--seed", "3"))
    assert "synthetic market:" in out
    return tmp / "syn"


# --------------------------------------------------------------------------- layout
def test_a_data_set_keeps_every_file_in_its_folder(tmp_path: Path) -> None:
    s = load_settings(None, {**dataset_path_overrides(tmp_path / "real"), "datasets.active": "real"})
    p = s.paths
    for path in (p.events_dir, p.metadata_dir, p.resolve("sqlite_file"), p.resolve("duckdb_file"), p.resolve("models_dir"),
                 p.resolve("reports_dir")):
        assert Path(path).is_relative_to(tmp_path / "real"), path
    assert s.datasets.root("real").is_absolute()


def test_kind_detection_and_marker(tmp_path: Path, events: pl.DataFrame) -> None:
    assert detect_kind(tmp_path / "nothing") == "empty"
    as_recorded(events.head(500), tmp_path / "rec")
    assert detect_kind(tmp_path / "rec") == "real" and read_marker(tmp_path / "rec") is None
    write_marker(tmp_path / "rec", "synthetic")  # the marker wins over inference
    assert detect_kind(tmp_path / "rec") == "synthetic"
    with pytest.raises(ValueError):
        write_marker(tmp_path / "rec", "fake")


# --------------------------------------------------------------------------- guards
def test_synth_marks_its_folder_and_the_real_data_set_refuses_it(synth_root: Path, tmp_path: Path) -> None:
    assert read_marker(synth_root) == "synthetic" and (synth_root / "meta.sqlite").exists()
    out = flat(cli(tmp_path, "real", synth_root, "backtest", code=2))
    assert "holds synthetic data" in out and "find-data" in out
    assert "`synth` writes a fake market" in flat(cli(tmp_path, "real", tmp_path / "r", "synth", code=2))
    for cmd in ("stream", "paper", "collect-history"):
        assert "hftr" in flat(cli(tmp_path, "synthetic", synth_root, cmd, code=2))


def test_synth_never_deletes_recorded_events(tmp_path: Path, events: pl.DataFrame) -> None:
    root = as_recorded(events.head(2_000), tmp_path / "rec")
    res = runner.invoke(app, ["--set", "logging.console=false", "--set", f"paths.logs_dir={tmp_path / 'logs'}",
                              "--set", f"paths.data_dir={root}", "--set", f"paths.sqlite_file={root / 'meta.sqlite'}",
                              "synth", "--hours", "0.1"])
    assert res.exit_code == 2 and "2,000 recorded" in flat(res)
    assert describe(root / "events").events == 2_000  # untouched
    assert "holds real data" in flat(cli(tmp_path, "synthetic", root, "backtest", code=2))


def test_recording_refuses_a_synthetic_folder_without_a_data_set(synth_root: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, ["--set", "logging.console=false", "--set", f"paths.logs_dir={tmp_path / 'logs'}",
                              "--set", f"paths.data_dir={synth_root}", "stream"])
    assert res.exit_code == 2 and "holds a synthetic market" in flat(res)


def test_empty_real_data_set_points_to_find_data(tmp_path: Path) -> None:
    out = flat(cli(tmp_path, "real", tmp_path / "empty", "backtest", code=1))
    assert "find-data" in out and "import-data" in out


# --------------------------------------------------------------------------- finding and importing
def test_find_and_import_split_a_mixed_folder(synth_root: Path, tmp_path: Path, events: pl.DataFrame) -> None:
    mixed = tmp_path / "mixed"
    shutil.copytree(synth_root, mixed)
    (mixed / "dataset.json").unlink()
    recorded = events.head(3_000)
    as_recorded(recorded, mixed, shift_ms=86_400_000, prefix="R")
    info = describe(mixed / "events")
    assert info.contents == "mixed" and info.real_events == 3_000
    assert mixed.resolve() in {p.parent for p in find_event_dirs([tmp_path], 4)}

    out = flat(cli(tmp_path, None, None, "find-data", "--path", str(tmp_path), "--depth", "3"))
    assert "mixed" in out and "3,000 of them recorded" in out

    real = tmp_path / "real"
    out = flat(cli(tmp_path, "real", real, "import-data", "--from", str(mixed)))
    assert "imported 3,000 events" in out
    got = ParquetEventStore(real / "events").read()
    assert got.height == 3_000 and got["mint"].str.starts_with("R").all()  # no synthetic token came along
    assert read_marker(real) == "real"
    cli(tmp_path, "real", real, "import-data", "--from", str(mixed))  # again: duplicates are removed
    assert describe(real / "events").events == 3_000
    assert describe(mixed / "events").events == info.events  # the source is never modified
    cli(tmp_path, "synthetic", synth_root, "import-data", "--from", str(mixed), code=2)


def test_import_refuses_its_own_store(tmp_path: Path, events: pl.DataFrame) -> None:
    root = as_recorded(events.head(100), tmp_path / "rec")
    with pytest.raises(ValueError):
        import_events(root / "events", ParquetEventStore(root / "events"), root / "metadata" / "tokens.parquet")


def test_data_info_suggests_a_train_test_split(tmp_path: Path, events: pl.DataFrame) -> None:
    root = as_recorded(events, tmp_path / "rec")
    res = cli(tmp_path, "real", root, "data-info")
    assert "real events" in flat(res) and f"{events.height:,}" in flat(res)
    line = next(ln for ln in res.output.splitlines() if "--end" in ln)  # printed unwrapped, ready to copy
    assert line.strip().startswith("hftr train-model")
    end = line.split("--end ")[1].strip()
    from pumpfun_hft.utils.timeutil import parse_iso_ms

    t = parse_iso_ms(end)
    ts = events["ts_ms"]
    assert int(ts.min()) < t < int(ts.max())
    assert 0.5 < float((ts < t).mean()) < 0.7


# --------------------------------------------------------------------------- models belong to their data set
def test_models_of_the_other_data_set_are_refused(synth_root: Path, tmp_path: Path, events: pl.DataFrame) -> None:
    out = flat(cli(tmp_path, "synthetic", synth_root, "train-model", "--model", "logistic", "--target", "fwd_up"))
    assert "hft backtest --strategy ml_signal" in out
    model = next((synth_root / "models").glob("*-fwd_up-*.joblib"))
    assert joblib.load(model)["dataset"] == "synthetic"
    real = as_recorded(events, tmp_path / "rec")
    assert "no fwd_up model" in flat(cli(tmp_path, "real", real, "backtest", "--strategy", "ml_signal", "--no-report", code=1))
    out = flat(cli(tmp_path, "real", real, "--set", f"strategy.params.ml_signal.model_path={model}", "backtest", "--strategy",
                   "ml_signal", "--no-report", code=1))
    assert "trained on the synthetic data set" in out


def test_rug_model_path_resolution(tmp_path: Path, settings: Any) -> None:
    cfg = settings.rug_model.model_copy(update={"use_trained_model": True, "model_path": "lightgbm-rug-1.joblib"})
    with pytest.raises(FileNotFoundError, match="use_trained_model"):
        build_rug_scorer(cfg, tmp_path)  # configured but missing: an error, not a silent heuristic
    bundle = {"model": None, "features": ["creator_score"], "train_end_ms": 0, "target": "rug", "dataset": "synthetic"}
    joblib.dump(bundle, tmp_path / "lightgbm-rug-1.joblib")
    assert isinstance(build_rug_scorer(cfg, tmp_path, "synthetic"), TrainedRugModel)
    with pytest.raises(ValueError, match="synthetic data set"):
        build_rug_scorer(cfg, tmp_path, "real")
    off = settings.rug_model.model_copy(update={"use_trained_model": False})
    assert not isinstance(build_rug_scorer(off, tmp_path), TrainedRugModel)


# --------------------------------------------------------------------------- label windows
def test_window_complete() -> None:
    ts = np.array([0, 10, 20, 30, 200, 210, 220], dtype=np.int64)  # silence from 30 to 200
    start = np.array([0, 5, 25, 200, 210, 0], dtype=np.int64)
    end = np.array([20, 25, 205, 220, 230, 220], dtype=np.int64)
    assert window_complete(start, end, ts, None).tolist() == [True, True, True, True, False, True]
    assert window_complete(start, end, ts, 50).tolist() == [True, True, False, True, False, False]


def test_training_skips_labels_the_data_does_not_cover(settings: Any, events: pl.DataFrame, metadata: dict[str, Any]) -> None:
    """Snapshots near the end of the data (or of --end) have truncated label windows: never trained on."""
    cut = int(events["ts_ms"].min()) + int(0.6 * (int(events["ts_ms"].max()) - int(events["ts_ms"].min())))
    part = events.filter(pl.col("ts_ms") < cut)
    ds = build_snapshot_dataset(settings, part, metadata)
    last = int(part["ts_ms"].max())
    assert (~ds["fwd_complete"]).sum() > 0 and (~ds["label_complete"]).sum() > (~ds["fwd_complete"]).sum()
    assert (ds.filter(pl.col("fwd_complete"))["fwd_end_ms"] <= last).all()
    ml = settings.ml.model_copy(update={"model": "logistic", "cv_folds": 3})
    fwd = train_and_evaluate(ds, feature_columns(ds), "fwd_up", ml, 1)
    rug = train_and_evaluate(ds, feature_columns(ds), "rug", ml, 1)
    assert fwd.train_end_ms <= last and rug.train_end_ms <= last  # the cut-off never lies past the data
    assert fwd.n_rows == ds.filter(pl.col("fwd_complete")).drop_nulls("fwd_up").height
    assert rug.n_rows == ds.filter(pl.col("label_complete")).drop_nulls("rug").height
    gappy = pl.concat([part.filter(pl.col("ts_ms") < cut - 3_600_000), part.filter(pl.col("ts_ms") >= cut - 1_800_000)])
    with_gap = build_snapshot_dataset(settings, gappy, metadata, max_gap_s=60.0)
    without = build_snapshot_dataset(settings, gappy, metadata)
    assert int(with_gap["fwd_complete"].sum()) < int(without["fwd_complete"].sum())  # the recording gap is respected


def test_verify_data_adopts_files_written_without_this_manifest(tmp_path: Path, events: pl.DataFrame) -> None:
    root = as_recorded(events.head(1_000), tmp_path / "rec")  # e.g. recorded while the manifest lived elsewhere
    out = flat(cli(tmp_path, "real", root, "verify-data"))
    assert "checksum mismatches: 0" in out and "not in this data set's manifest" in out and "hftr verify-data --adopt" in out
    assert "added 1 file(s)" in flat(cli(tmp_path, "real", root, "verify-data", "--adopt"))
    out = flat(cli(tmp_path, "real", root, "verify-data"))
    assert "checksum mismatches: 0" in out and "manifest" not in out
    (next((root / "events").glob("date=*/*.parquet"))).write_bytes(b"corrupt")
    assert "checksum mismatches: 1" in flat(cli(tmp_path, "real", root, "verify-data", code=1))


def test_a_run_without_trades_says_why(synth_root: Path, tmp_path: Path) -> None:
    res = cli(tmp_path, "synthetic", synth_root, "--set", "sizing.min_confidence=101", "backtest", "--strategy",
              "momentum_ignition", "--no-report")
    out = flat(res)
    assert "No trades" in out and "low_confidence" in out and "sizing.min_confidence" in out
    cli(tmp_path, "synthetic", synth_root, "report", "--formats", "html")
    run = max((synth_root / "reports" / "runs").iterdir(), key=lambda d: d.stat().st_mtime)
    page = (run / "report" / "report.html").read_text(encoding="utf-8")
    assert "No trades in this run" in page and "low_confidence" in page


def test_events_without_a_price_never_become_labels(settings: Any, events: pl.DataFrame, metadata: dict[str, Any],
                                                    tmp_path: Path) -> None:
    """Trades with zero reserves give a NaN / inf price; polars orders NaN above every number, so before the fix
    such a snapshot was labelled fwd_up = 1 and a model learned the artifact (100 % 'hit rate' at score 0.999)."""
    mints = events.filter(pl.col("kind") == "create")["mint"].head(40).to_list()
    broken = events.with_columns(
        [pl.when(pl.col("mint").is_in(mints) & (pl.col("kind") == "trade")).then(0).otherwise(pl.col(c)).alias(c)
         for c in ("v_sol", "v_tok")])
    ds = build_snapshot_dataset(settings, broken, metadata)
    bad = ds.filter(pl.col("mint").is_in(mints))
    assert bad.height and bad["fwd_up"].is_null().all() and bad["fwd_return"].is_null().all()
    good = ds.filter(~pl.col("mint").is_in(mints) & pl.col("fwd_up").is_not_null())
    assert good["fwd_return"].is_finite().all()
    assert ds.filter(pl.col("fwd_up").is_not_null())["fwd_return"].is_finite().all()
    as_recorded(broken, tmp_path / "rec")
    from pumpfun_hft.collectors.datasets import price_quality

    q = price_quality(tmp_path / "rec" / "events")
    assert q["trades_without_price"] == broken.filter(pl.col("mint").is_in(mints) & (pl.col("kind") == "trade")).height
    assert "without a valid price" in flat(cli(tmp_path, "real", tmp_path / "rec", "data-info"))
