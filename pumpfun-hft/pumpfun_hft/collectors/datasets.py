"""Data sets: which folder holds which kind of events, finding event stores, and moving real events.

There are two data sets, each a folder that holds everything made from it (see
``core.config.dataset_path_overrides``):

* ``synthetic`` - the fake market written by ``synth`` (``hft`` in ``hft.ps1``)
* ``real``      - Pump.fun events recorded by ``stream`` / ``paper`` / ``collect-history`` (``hftr``)

A folder's kind comes from its marker file ``dataset.json`` (written the first time a command
uses the folder). Without a marker it is inferred: ``metadata/synthetic_truth.parquet`` (written
only by ``synth``) means synthetic, any other events mean real. The CLI refuses to run a
synthetic-only command on a real folder and vice versa, so the two never mix.

``find_event_dirs`` locates event stores on disk (any folder of ``date=YYYY-MM-DD`` partitions),
``describe`` summarises one (events, time range, and how many events belong to synthetic tokens),
and ``import_events`` copies the real events of one store into another, leaving synthetic tokens
behind and de-duplicating on the way.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from pumpfun_hft.collectors.storage import ParquetEventStore, normalise_frame, write_metadata
from pumpfun_hft.utils.timeutil import ms_to_iso, now_ms

MARKER = "dataset.json"
TRUTH_FILE = "synthetic_truth.parquet"
KINDS = ("synthetic", "real")
_DAY_DIR = re.compile(r"^date=\d{4}-\d{2}-\d{2}$")
#: folder names never searched for event stores (lower case)
_SKIP = frozenset({"appdata", "node_modules", "site-packages", "__pycache__", "venv", "env", "windows", "program files",
                   "program files (x86)", "programdata", "library", "system volume information", "anaconda3", "miniconda3",
                   "pictures", "music", "videos", "movies", "applications", "proc", "sys", "dev", "usr", "bin", "lib", "etc"})


def events_present(events_dir: Path) -> bool:
    return events_dir.is_dir() and any(events_dir.glob("date=*/*.parquet"))


def read_marker(root: str | Path) -> str | None:
    p = Path(root) / MARKER
    try:
        kind = json.loads(p.read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError, AttributeError):
        return None
    return kind if kind in KINDS else None


def write_marker(root: str | Path, kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown data set kind {kind!r}")
    r = Path(root)
    r.mkdir(parents=True, exist_ok=True)
    (r / MARKER).write_text(json.dumps({"kind": kind, "marked_utc": ms_to_iso(now_ms())}, indent=2) + "\n", encoding="utf-8")


def detect_kind(root: str | Path, events_subdir: str = "events", metadata_subdir: str = "metadata") -> str:
    """``synthetic`` | ``real`` | ``empty`` for the data-set folder ``root`` (the marker wins)."""
    r = Path(root)
    marker = read_marker(r)
    if marker:
        return marker
    if (r / metadata_subdir / TRUTH_FILE).exists():
        return "synthetic"
    return "real" if events_present(r / events_subdir) else "empty"


@dataclass
class StoreInfo:
    """Summary of one event store (``root`` is the data-set folder, the parent of ``events_dir``)."""

    events_dir: Path
    root: Path
    kind: str                       # detect_kind(root)
    marker: str | None
    days: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    events: int = 0
    start_ms: int | None = None
    end_ms: int | None = None
    synthetic_events: int | None = None  # events of tokens in the synthetic truth file (None = no truth file)

    @property
    def real_events(self) -> int:
        return self.events - (self.synthetic_events or 0)

    @property
    def contents(self) -> str:
        """What the events actually are: ``synthetic`` | ``real`` | ``mixed`` | ``empty``."""
        if not self.events:
            return "empty"
        if self.synthetic_events is None:
            return "synthetic" if self.kind == "synthetic" else "real"
        if self.synthetic_events == 0:
            return "real"
        return "synthetic" if self.real_events == 0 else "mixed"

    def span(self) -> str:
        if self.start_ms is None or self.end_ms is None:
            return "-"
        hours = (self.end_ms - self.start_ms) / 3_600_000
        return f"{ms_to_iso(self.start_ms)} -> {ms_to_iso(self.end_ms)} ({hours:,.1f} h)"


def synthetic_mints(root: str | Path, metadata_subdir: str = "metadata") -> list[str] | None:
    p = Path(root) / metadata_subdir / TRUTH_FILE
    if not p.exists():
        return None
    return pl.read_parquet(p, columns=["mint"], memory_map=False)["mint"].drop_nulls().unique().to_list()


def describe(events_dir: str | Path, metadata_subdir: str = "metadata", count_synthetic: bool = True) -> StoreInfo:
    """Count the events of one store, their time range, and how many belong to synthetic tokens."""
    ev = Path(events_dir)
    root = ev.parent
    info = StoreInfo(ev, root, detect_kind(root, ev.name, metadata_subdir), read_marker(root))
    files = sorted(ev.glob("date=*/*.parquet"))
    if not files:
        return info
    info.days = sorted({f.parent.name.split("=", 1)[1] for f in files})
    info.files = len(files)
    info.bytes = sum(f.stat().st_size for f in files)
    lf = ParquetEventStore(ev).scan(columns=["ts_ms", "mint"])
    agg = lf.select(pl.len().alias("n"), pl.col("ts_ms").min().alias("t0"), pl.col("ts_ms").max().alias("t1")).collect()
    info.events = int(agg["n"][0])
    info.start_ms = None if agg["t0"][0] is None else int(agg["t0"][0])
    info.end_ms = None if agg["t1"][0] is None else int(agg["t1"][0])
    mints = synthetic_mints(root, metadata_subdir) if count_synthetic else None
    if mints is not None:
        info.synthetic_events = int(lf.filter(pl.col("mint").is_in(mints)).select(pl.len()).collect().item())
    return info


def real_event_count(root: str | Path, events_subdir: str = "events", metadata_subdir: str = "metadata") -> int:
    """Events in the folder that do not belong to a synthetic token (what ``synth --clear`` would destroy)."""
    ev = Path(root) / events_subdir
    if not events_present(ev):
        return 0
    return describe(ev, metadata_subdir).real_events


def time_quantile(events_dir: str | Path, q: float) -> int | None:
    """Timestamp (ms) before which a fraction ``q`` of the store's events lie."""
    ev = Path(events_dir)
    if not events_present(ev):
        return None
    v = ParquetEventStore(ev).scan(columns=["ts_ms"]).select(pl.col("ts_ms").quantile(q, "nearest")).collect().item()
    return None if v is None else int(v)


# ---------------------------------------------------------------------------- finding stores
def _is_events_dir(path: Path, subdirs: Iterable[str]) -> bool:
    for d in subdirs:
        if _DAY_DIR.match(d) and any((path / d).glob("*.parquet")):
            return True
    return False


def find_event_dirs(search_roots: Iterable[str | Path], max_depth: int = 7) -> list[Path]:
    """Every event store (a folder of ``date=YYYY-MM-DD`` partitions holding Parquet) under the roots."""
    found: list[Path] = []
    seen: set[str] = set()
    for base in search_roots:
        b = Path(base)
        if not b.is_dir():
            continue
        base_depth = len(b.parts)
        for dirpath, dirnames, _ in os.walk(b, onerror=lambda _e: None):
            p = Path(dirpath)
            if _is_events_dir(p, dirnames):
                key = os.path.normcase(str(p.resolve()))
                if key not in seen:
                    seen.add(key)
                    found.append(p.resolve())
                dirnames[:] = []
                continue
            if len(p.parts) - base_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "$")) and d.lower() not in _SKIP]
    return found


# ---------------------------------------------------------------------------- importing
@dataclass
class ImportReport:
    source: Path
    days: int = 0
    read: int = 0
    skipped_synthetic: int = 0
    written: int = 0
    duplicates_removed: int = 0
    tokens_metadata: int = 0


def resolve_events_dir(path: str | Path, events_subdir: str = "events") -> Path:
    """Accept a data-set folder or its events folder; return the folder holding the ``date=`` partitions."""
    p = Path(path).expanduser()
    if events_present(p):
        return p.resolve()
    if events_present(p / events_subdir):
        return (p / events_subdir).resolve()
    raise FileNotFoundError(f"no events under {p} (expected date=YYYY-MM-DD folders of .parquet files, "
                            f"either directly or in {p / events_subdir})")


def import_events(source_events_dir: str | Path, dest: ParquetEventStore, dest_metadata: Path,
                  metadata_subdir: str = "metadata", progress: Callable[[str, int], None] | None = None) -> ImportReport:
    """Copy the real events of another store into ``dest`` (never modifies the source).

    Events of tokens listed in the source's synthetic truth file are left behind, so a store in
    which a synthetic market and recorded events got mixed can be split. Duplicates (events the
    destination already has) are removed by the compaction at the end."""
    src = Path(source_events_dir).resolve()
    if os.path.normcase(str(src)) == os.path.normcase(str(Path(dest.root).resolve())):
        raise ValueError("the source is the destination's own event store")
    rep = ImportReport(src)
    fake = set(synthetic_mints(src.parent, metadata_subdir) or [])
    for day_dir in sorted(d for d in src.glob("date=*") if d.is_dir() and _DAY_DIR.match(d.name)):
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            continue
        df = pl.concat([normalise_frame(pl.read_parquet(f, memory_map=False)) for f in files], how="vertical_relaxed")
        rep.read += df.height
        if fake:
            keep = df.filter(pl.col("mint").is_null() | ~pl.col("mint").is_in(list(fake)))
            rep.skipped_synthetic += df.height - keep.height
            df = keep
        if df.height:
            dest.write(df)
            rep.written += df.height
        rep.days += 1
        if progress is not None:
            progress(day_dir.name.split("=", 1)[1], df.height)
    rep.duplicates_removed = sum(r.duplicates for r in dest.compact())
    meta_src = src.parent / metadata_subdir / "tokens.parquet"
    if meta_src.exists():
        tokens = pl.read_parquet(meta_src, memory_map=False)
        if fake and "mint" in tokens.columns:
            tokens = tokens.filter(~pl.col("mint").is_in(list(fake)))
        if tokens.height:
            write_metadata(tokens, dest_metadata)
            rep.tokens_metadata = tokens.height
    return rep
