"""Day-partitioned Parquet event store with checksums, de-duplication and gap detection.

Layout::

    <root>/date=YYYY-MM-DD/part-<ms>-<rand>.parquet   (append-only writes)
    <root>/date=YYYY-MM-DD/data.parquet              (after compaction: sorted, de-duplicated)

* Every file written is SHA-256 checksummed into the SQLite manifest; :meth:`verify` re-hashes.
* :meth:`compact` merges a day's parts, drops duplicates (key: signature + ev_idx + kind; rows
  without a signature fall back to full-row identity), sorts by (slot, seq, ev_idx) and writes
  atomically (tmp file + rename).
* :meth:`detect_gaps` reports slot ranges with no events wider than a threshold — useful
  after interrupted downloads or WebSocket disconnects.
* :meth:`iter_batches` streams events in on-chain order one day at a time for memory-efficient
  replay of arbitrarily large histories.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pumpfun_hft.core.types import EVENT_COLUMNS, EVENT_SCHEMA, SORT_KEYS
from pumpfun_hft.database.meta import MetaStore
from pumpfun_hft.utils.hashing import sha256_file
from pumpfun_hft.utils.logging import get_logger
from pumpfun_hft.utils.timeutil import now_ms

log = get_logger("system")
DEDUP_KEYS = ("signature", "ev_idx", "kind")


@dataclass(slots=True)
class CompactionReport:
    day: str
    files_in: int
    rows_in: int
    rows_out: int
    duplicates: int
    path: str


def normalise_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce any event-like frame to the canonical schema and column order."""
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=EVENT_SCHEMA[c]).alias(c) for c in missing])
    return df.select([pl.col(c).cast(EVENT_SCHEMA[c], strict=False) for c in EVENT_COLUMNS])


class ParquetEventStore:
    """Append/compact/scan events stored as hive-partitioned Parquet.

    Example::

        store = ParquetEventStore(settings.paths.events_dir, MetaStore(settings.paths.resolve("sqlite_file")))
        store.write(df); store.compact(); lf = store.scan(start_ms=t0, kinds=["trade"])
    """

    def __init__(self, root: str | Path, meta: MetaStore | None = None, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.compression = compression

    # ------------------------------------------------------------------ write
    def _day_dir(self, day: str) -> Path:
        return self.root / f"date={day}"

    def _register(self, path: Path, day: str, df: pl.DataFrame, compacted: bool) -> None:
        if self.meta is None:
            return
        self.meta.add_file(
            str(path.relative_to(self.root)), day, sha256_file(path), df.height,
            int(df["slot"].min()) if df.height else None, int(df["slot"].max()) if df.height else None,
            path.stat().st_size, compacted,
        )

    def write(self, df: pl.DataFrame) -> list[Path]:
        """Append events (any order) as new part files, one per UTC day."""
        if df.is_empty():
            return []
        df = normalise_frame(df)
        df = df.with_columns(pl.from_epoch(pl.col("ts_ms"), time_unit="ms").dt.strftime("%Y-%m-%d").alias("_day"))
        written: list[Path] = []
        for (day,), part in df.partition_by("_day", as_dict=True).items():
            part = part.drop("_day").sort(list(SORT_KEYS))
            ddir = self._day_dir(str(day))
            ddir.mkdir(parents=True, exist_ok=True)
            path = ddir / f"part-{now_ms()}-{secrets.token_hex(4)}.parquet"
            tmp = path.with_suffix(".tmp")
            part.write_parquet(tmp, compression=self.compression, statistics=True)
            os.replace(tmp, path)
            self._register(path, str(day), part, False)
            written.append(path)
        return written

    # ------------------------------------------------------------------ maintenance
    def days(self) -> list[str]:
        return sorted(p.name.split("=", 1)[1] for p in self.root.glob("date=*") if p.is_dir())

    def compact(self, day: str | None = None) -> list[CompactionReport]:
        """Merge, de-duplicate and sort part files (all days when ``day`` is None)."""
        reports = []
        for d in [day] if day else self.days():
            ddir = self._day_dir(d)
            files = sorted(ddir.glob("*.parquet"))
            if not files:
                continue
            df = pl.concat([normalise_frame(pl.read_parquet(f, memory_map=False)) for f in files], how="vertical_relaxed")
            rows_in = df.height
            with_sig = df.filter(pl.col("signature").is_not_null()).unique(subset=list(DEDUP_KEYS), keep="first", maintain_order=True)
            no_sig = df.filter(pl.col("signature").is_null()).unique(keep="first", maintain_order=True)
            out = pl.concat([with_sig, no_sig]).sort(list(SORT_KEYS))
            target = ddir / "data.parquet"
            tmp = ddir / "data.parquet.tmp"
            out.write_parquet(tmp, compression=self.compression, statistics=True)
            os.replace(tmp, target)
            for f in files:
                if f != target:
                    f.unlink(missing_ok=True)
                    if self.meta is not None:
                        self.meta.remove_file(str(f.relative_to(self.root)))
            self._register(target, d, out, True)
            reports.append(CompactionReport(d, len(files), rows_in, out.height, rows_in - out.height, str(target)))
            log.info("compacted", extra={"data": {"day": d, "rows_in": rows_in, "rows_out": out.height}})
        return reports

    def verify(self) -> list[str]:
        """Return relative paths whose checksum does not match the manifest (or that vanished)."""
        if self.meta is None:
            return []
        bad = []
        for rec in self.meta.files():
            p = self.root / rec["path"]
            if not p.exists() or sha256_file(p) != rec["sha256"]:
                bad.append(rec["path"])
        return bad

    def detect_gaps(self, max_slot_gap: int, start_ms: int | None = None, end_ms: int | None = None) -> pl.DataFrame:
        """Slot ranges without any event that are wider than ``max_slot_gap`` slots."""
        lf = self.scan(start_ms=start_ms, end_ms=end_ms, columns=["slot", "ts_ms"])
        slots = lf.group_by("slot").agg(pl.col("ts_ms").min()).sort("slot").collect()
        if slots.height < 2:
            return pl.DataFrame(schema={"gap_start_slot": pl.Int64, "gap_end_slot": pl.Int64, "gap_slots": pl.Int64,
                                        "start_ms": pl.Int64, "end_ms": pl.Int64})
        return (
            slots.with_columns(pl.col("slot").shift(1).alias("prev_slot"), pl.col("ts_ms").shift(1).alias("prev_ms"))
            .with_columns((pl.col("slot") - pl.col("prev_slot")).alias("gap_slots"))
            .filter(pl.col("gap_slots") > max_slot_gap)
            .select(pl.col("prev_slot").alias("gap_start_slot"), pl.col("slot").alias("gap_end_slot"), "gap_slots",
                    pl.col("prev_ms").alias("start_ms"), pl.col("ts_ms").alias("end_ms"))
        )

    # ------------------------------------------------------------------ read
    def _glob(self) -> str:
        return str(self.root / "date=*" / "*.parquet")

    def scan(self, start_ms: int | None = None, end_ms: int | None = None, kinds: list[str] | None = None,
             mints: list[str] | None = None, columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazy scan with predicate push-down (hive ``date`` column is dropped)."""
        if not any(self.root.glob("date=*/*.parquet")):
            empty = pl.DataFrame(schema=EVENT_SCHEMA).lazy()
            return empty.select(columns) if columns else empty
        lf = pl.scan_parquet(self._glob(), hive_partitioning=True)
        if "date" in lf.collect_schema().names():
            lf = lf.drop("date")
        if start_ms is not None:
            lf = lf.filter(pl.col("ts_ms") >= start_ms)
        if end_ms is not None:
            lf = lf.filter(pl.col("ts_ms") < end_ms)
        if kinds:
            lf = lf.filter(pl.col("kind").is_in(kinds))
        if mints:
            lf = lf.filter(pl.col("mint").is_in(mints))
        if columns:
            lf = lf.select(columns)
        return lf

    def read(self, start_ms: int | None = None, end_ms: int | None = None, kinds: list[str] | None = None,
             mints: list[str] | None = None) -> pl.DataFrame:
        df = self.scan(start_ms, end_ms, kinds, mints).collect()
        return normalise_frame(df).sort(list(SORT_KEYS))

    def iter_batches(self, start_ms: int | None = None, end_ms: int | None = None, kinds: list[str] | None = None,
                     batch_rows: int = 250_000) -> Iterator[pl.DataFrame]:
        """Yield events in on-chain order, one day (sliced to ``batch_rows``) at a time."""
        for d in self.days():
            files = sorted(self._day_dir(d).glob("*.parquet"))
            if not files:
                continue
            lf = pl.scan_parquet([str(f) for f in files])
            if start_ms is not None:
                lf = lf.filter(pl.col("ts_ms") >= start_ms)
            if end_ms is not None:
                lf = lf.filter(pl.col("ts_ms") < end_ms)
            if kinds:
                lf = lf.filter(pl.col("kind").is_in(kinds))
            day_df = normalise_frame(lf.collect()).sort(list(SORT_KEYS))
            for off in range(0, day_df.height, batch_rows):
                yield day_df.slice(off, batch_rows)

    def stats(self) -> dict[str, object]:
        files = list(self.root.glob("date=*/*.parquet"))
        return {"days": len(self.days()), "files": len(files), "bytes": sum(f.stat().st_size for f in files)}


def write_metadata(df: pl.DataFrame, path: str | Path) -> Path:
    """Persist token metadata (one row per mint) as Parquet, merging with any existing file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        df = pl.concat([pl.read_parquet(p), df], how="diagonal_relaxed").unique(subset=["mint"], keep="last")
    tmp = p.with_suffix(".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, p)
    return p
