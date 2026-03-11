#!/usr/bin/env python3
"""
build_resampled_bars.py

Build resampled intraday bars from continuous 1-minute futures bars.

Input:
  /data/lake/research/continuous_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Outputs:
  /data/lake/research/resampled_bars/<FREQ>/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Supported frequencies:
  5m, 15m, 30m, 1h

Resampling logic:
- open  = first open in bucket
- high  = max high in bucket
- low   = min low in bucket
- close = last close in bucket
- volume = sum volume in bucket
- symbol/instrument_id/root/trade_date_utc = last non-null in bucket

Notes:
- Uses dynamic_group_by on ts_event
- Assumes ts_event is UTC
- Builds bars independently within each monthly partition
- Buckets are calendar-time UTC buckets
- Does not currently stitch prior-month buffers; month boundaries are handled per file

Examples:
  python /home/marketdata/futures-data/build_resampled_bars.py --freq 5m
  python /home/marketdata/futures-data/build_resampled_bars.py --freq 15m --root ES --overwrite
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

SRC_BASE = Path("/data/lake/research/continuous_1m")
OUT_BASE = Path("/data/lake/research/resampled_bars")
STATE_DIR = Path("/data/lake/state")

SUPPORTED_FREQS = {"2m", "5m", "10m", "15m", "30m", "1h"}

REQUIRED_COLUMNS = [
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
]

OUTPUT_COLUMNS = REQUIRED_COLUMNS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def progress_path(freq: str) -> Path:
    return STATE_DIR / f"resampled_bars_{freq}_progress.json"


def report_path(freq: str) -> Path:
    return STATE_DIR / f"build_resampled_bars_{freq}_report.json"


def load_progress(freq: str) -> Dict:
    p = progress_path(freq)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            data.setdefault("done", {})
            return data
        except Exception:
            pass
    return {"done": {}}


def save_progress_atomic(freq: str, data: Dict) -> None:
    p = progress_path(freq)
    ensure_dir(p.parent)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(p)


def is_done(progress: Dict, root: str, year: str, month: str) -> bool:
    return bool(progress.get("done", {}).get(root, {}).get(year, {}).get(month, False))


def mark_done(progress: Dict, root: str, year: str, month: str) -> None:
    progress.setdefault("done", {}).setdefault(root, {}).setdefault(year, {})
    progress["done"][root][year][month] = True


def read_roots(path: Path) -> List[str]:
    roots: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def src_path(root: str, year: str, month: str) -> Path:
    return SRC_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def out_path(freq: str, root: str, year: str, month: str) -> Path:
    return OUT_BASE / freq / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def discover_months_for_root(root: str) -> List[Tuple[str, str]]:
    base = SRC_BASE / f"root={root}"
    if not base.exists():
        return []

    out: List[Tuple[str, str]] = []
    for year_dir in sorted(base.glob("year=*")):
        year = year_dir.name.split("=", 1)[1]
        for month_dir in sorted(year_dir.glob("month=*")):
            month = month_dir.name.split("=", 1)[1]
            out.append((year, month))
    return out


def load_continuous_1m(root: str, year: str, month: str) -> Optional[pl.DataFrame]:
    p = src_path(root, year, month)
    if not p.exists():
        return None

    df = pl.read_parquet(p)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} missing required columns: {missing}")

    return (
        df.select(REQUIRED_COLUMNS)
        .with_columns(
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
        )
        .sort("ts_event")
    )


def polars_every(freq: str) -> str:
    mapping = {
        "2m": "2m",
        "5m": "5m",
        "10m": "10m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
    }
    return mapping[freq]


def resample_bars(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if df.is_empty():
        return df

    every = polars_every(freq)

    out = (
        df.sort("ts_event")
        .group_by_dynamic(
            index_column="ts_event",
            every=every,
            period=every,
            closed="right",
            label="right",
            start_by="window",
        )
        .agg(
            [
                pl.col("trade_date_utc").last().alias("trade_date_utc"),
                pl.col("root").last().alias("root"),
                pl.col("symbol").last().alias("symbol"),
                pl.col("instrument_id").last().alias("instrument_id"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(subset=["open", "high", "low", "close"])
        .select(OUTPUT_COLUMNS)
        .sort("ts_event")
    )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build resampled bars from continuous_1m.")
    ap.add_argument("--freq", required=True, choices=sorted(SUPPORTED_FREQS))
    ap.add_argument("--roots-file", default="/home/marketdata/futures-data/roots.txt")
    ap.add_argument("--root", default=None, help="Optional single-root run, e.g. ES")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    freq = args.freq
    roots_file = Path(args.roots_file).expanduser().resolve()
    if not roots_file.exists():
        raise SystemExit(f"roots file not found: {roots_file}")

    roots = read_roots(roots_file)
    if args.root:
        roots = [r for r in roots if r == args.root]
    if not roots:
        raise SystemExit("No roots selected.")

    progress = load_progress(freq)

    print(f"Frequency:  {freq}")
    print(f"Roots file: {roots_file}")
    print(f"Roots:      {len(roots)}")
    print(f"SRC_BASE:   {SRC_BASE}")
    print(f"OUT_BASE:   {OUT_BASE / freq}")
    print(f"Progress:   {progress_path(freq)}")
    print(f"Overwrite:  {args.overwrite}")
    if args.root:
        print(f"Root:       {args.root}")
    print()

    wrote = 0
    skipped = 0
    empty = 0
    errors = 0
    total_rows_written = 0
    error_details: List[Dict] = []

    for root in roots:
        months = discover_months_for_root(root)
        if not months:
            print(f"[skip-root] {root}: no continuous_1m partitions found")
            continue

        print(f"=== ROOT {root} months={len(months)} ===")

        for year, month in months:
            op = out_path(freq, root, year, month)

            if (
                not args.overwrite
                and (is_done(progress, root, year, month) or (op.exists() and op.stat().st_size > 0))
            ):
                skipped += 1
                continue

            try:
                df = load_continuous_1m(root, year, month)
                if df is None or df.is_empty():
                    empty += 1
                    continue

                out = resample_bars(df, freq)
                if out.is_empty():
                    empty += 1
                    continue

                ensure_dir(op.parent)
                out.write_parquet(op, compression="zstd")

                mark_done(progress, root, year, month)
                save_progress_atomic(freq, progress)

                wrote += 1
                total_rows_written += out.height
                print(f"[wrote] {root} {year}-{month}: rows={out.height}")

            except Exception as e:
                errors += 1
                detail = {
                    "root": root,
                    "year": year,
                    "month": month,
                    "error": f"{type(e).__name__}: {e}",
                }
                error_details.append(detail)
                print(f"[ERROR] {root} {year}-{month}: {type(e).__name__}: {e}")

        save_progress_atomic(freq, progress)
        print()

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "frequency": freq,
        "roots_file": str(roots_file),
        "root_filter": args.root,
        "overwrite": args.overwrite,
        "src_base": str(SRC_BASE),
        "out_base": str(OUT_BASE / freq),
        "partitions_written": wrote,
        "partitions_skipped": skipped,
        "partitions_empty": empty,
        "partitions_errored": errors,
        "total_rows_written": total_rows_written,
        "errors": error_details,
        "ok": errors == 0,
    }

    rp = report_path(freq)
    ensure_dir(rp.parent)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Wrote report: {rp}")
    print(
        f"Summary: wrote={wrote} skipped={skipped} empty={empty} "
        f"errors={errors} rows={total_rows_written}"
    )

    raise SystemExit(0 if errors == 0 else 2)


if __name__ == "__main__":
    main()