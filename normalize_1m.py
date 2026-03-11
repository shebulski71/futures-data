#!/usr/bin/env python3
"""
normalize_1m.py

Normalize raw Databento 1-minute parquet files into a clean, partitioned,
contract-level 1m lake.

Input:
  /data/lake/raw/databento/GLBX.MDP3/ohlcv-1m/**/*.parquet

Output:
  /data/lake/curated_normalized/ohlcv_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

What this script does:
- reads all raw 1m parquet files for each root/year/month partition
- preserves ts_event
- derives:
    - root
    - trade_date_utc
    - year
    - month
- keeps a stable schema
- de-duplicates on (symbol, ts_event)
- writes one normalized parquet per root/year/month
- supports incremental/resumable processing via a progress file
- supports --overwrite for rebuilding existing partitions

Notes:
- This is a contract-level normalized layer, not a continuous series
- Multiple raw files can overlap inside a month; dedup handles that
- ts_event is assumed to be UTC
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

RAW_BASE = Path("/data/lake/raw/databento/GLBX.MDP3/ohlcv-1m")
OUT_BASE = Path("/data/lake/curated_normalized/ohlcv_1m")
PROGRESS_PATH = Path("/data/lake/state/normalize_1m_progress.json")
REPORT_PATH = Path("/data/lake/state/normalize_1m_report.json")

REQUIRED_COLUMNS = [
    "ts_event",
    "rtype",
    "publisher_id",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
]

OUTPUT_COLUMNS = [
    "ts_event",
    "trade_date_utc",
    "root",
    "symbol",
    "instrument_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "publisher_id",
    "rtype",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_progress() -> Dict:
    if PROGRESS_PATH.exists():
        try:
            data = json.loads(PROGRESS_PATH.read_text())
            data.setdefault("done", {})
            return data
        except Exception:
            pass
    return {"done": {}}


def save_progress_atomic(data: Dict) -> None:
    ensure_dir(PROGRESS_PATH.parent)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(PROGRESS_PATH)


def is_done(progress: Dict, root: str, year: str, month: str) -> bool:
    return bool(progress.get("done", {}).get(root, {}).get(year, {}).get(month, False))


def mark_done(progress: Dict, root: str, year: str, month: str) -> None:
    progress.setdefault("done", {}).setdefault(root, {}).setdefault(year, {})
    progress["done"][root][year][month] = True


def out_path(root: str, year: str, month: str) -> Path:
    return OUT_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def discover_raw_partitions() -> Dict[Tuple[str, str, str], List[Path]]:
    """
    Returns:
      {(root, year, month): [parquet_files...]}
    """
    buckets: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)

    for f in RAW_BASE.rglob("*.parquet"):
        parts = f.parts
        root = next((p.split("=", 1)[1] for p in parts if p.startswith("root=")), None)
        year = next((p.split("=", 1)[1] for p in parts if p.startswith("year=")), None)
        month = next((p.split("=", 1)[1] for p in parts if p.startswith("month=")), None)

        if not root or not year or not month:
            continue

        buckets[(root, year, month)].append(f)

    for key in buckets:
        buckets[key].sort()

    return dict(sorted(buckets.items()))


def normalize_partition(root: str, year: str, month: str, files: List[Path]) -> pl.DataFrame:
    frames: List[pl.DataFrame] = []

    for f in files:
        df = pl.read_parquet(f)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns: {missing}")

        # Keep only required raw columns, then derive normalized columns
        df = (
            df.select(REQUIRED_COLUMNS)
            .with_columns(
                pl.lit(root).alias("root"),
                pl.col("ts_event").dt.date().alias("trade_date_utc"),
            )
            .select(OUTPUT_COLUMNS)
        )

        frames.append(df)

    if not frames:
        return pl.DataFrame(schema={c: pl.String for c in OUTPUT_COLUMNS})

    df_all = pl.concat(frames, how="vertical_relaxed")

    # Stable sort before dedup so output is deterministic
    # Dedup key: contract bar identity
    df_all = (
        df_all.sort(["symbol", "ts_event", "instrument_id"])
        .unique(subset=["symbol", "ts_event"], keep="first", maintain_order=True)
        .sort(["ts_event", "symbol"])
    )

    # Final schema cleanup
    df_all = df_all.with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
        pl.col("trade_date_utc").cast(pl.Date),
        pl.col("root").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("instrument_id").cast(pl.UInt32),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.UInt64),
        pl.col("publisher_id").cast(pl.UInt16),
        pl.col("rtype").cast(pl.UInt8),
    )

    return df_all.select(OUTPUT_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize raw Databento 1m parquet into a curated contract-level 1m lake.")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild existing normalized partitions.")
    ap.add_argument("--root", default=None, help="Optional single-root run, e.g. ES")
    args = ap.parse_args()

    if not RAW_BASE.exists():
        raise SystemExit(f"RAW_BASE does not exist: {RAW_BASE}")

    ensure_dir(OUT_BASE.parent)
    ensure_dir(PROGRESS_PATH.parent)

    progress = load_progress()
    raw_partitions = discover_raw_partitions()

    if args.root:
        raw_partitions = {k: v for k, v in raw_partitions.items() if k[0] == args.root}

    if not raw_partitions:
        raise SystemExit("No raw 1m parquet partitions found.")

    print(f"Discovered raw partitions: {len(raw_partitions)}")
    print(f"Raw base:  {RAW_BASE}")
    print(f"Out base:  {OUT_BASE}")
    print(f"Progress:  {PROGRESS_PATH}")
    print(f"Overwrite: {args.overwrite}")
    if args.root:
        print(f"Root:      {args.root}")

    wrote = 0
    skipped = 0
    errors = 0
    total_rows_written = 0
    error_details: List[Dict] = []

    for (root, year, month), files in raw_partitions.items():
        op = out_path(root, year, month)

        if (
            not args.overwrite
            and (is_done(progress, root, year, month) or (op.exists() and op.stat().st_size > 0))
        ):
            skipped += 1
            continue

        try:
            df_out = normalize_partition(root, year, month, files)

            if df_out.is_empty():
                print(f"[skip-empty] {root} {year}-{month}")
                skipped += 1
                continue

            ensure_dir(op.parent)
            df_out.write_parquet(op, compression="zstd")

            mark_done(progress, root, year, month)
            save_progress_atomic(progress)

            wrote += 1
            total_rows_written += df_out.height
            print(f"[wrote] {root} {year}-{month}: files={len(files)} rows={df_out.height}")

        except Exception as e:
            errors += 1
            detail = {
                "root": root,
                "year": year,
                "month": month,
                "files": [str(x) for x in files],
                "error": f"{type(e).__name__}: {e}",
            }
            error_details.append(detail)
            print(f"[ERROR] {root} {year}-{month}: {type(e).__name__}: {e}")

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "raw_base": str(RAW_BASE),
        "out_base": str(OUT_BASE),
        "overwrite": args.overwrite,
        "root_filter": args.root,
        "partitions_discovered": len(raw_partitions),
        "partitions_written": wrote,
        "partitions_skipped": skipped,
        "partitions_errored": errors,
        "total_rows_written": total_rows_written,
        "errors": error_details,
        "ok": errors == 0,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))

    print()
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Summary: wrote={wrote} skipped={skipped} errors={errors} rows={total_rows_written}")

    raise SystemExit(0 if errors == 0 else 2)


if __name__ == "__main__":
    main()