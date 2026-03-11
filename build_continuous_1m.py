#!/usr/bin/env python3
"""
build_continuous_1m.py

Build continuous 1-minute futures series by applying the daily continuous
contract selection to normalized 1m bars.

Inputs:
  Normalized 1m contract bars:
    /data/lake/curated_normalized/ohlcv_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

  Continuous daily series:
    /data/lake/research/continuous_daily/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Outputs:
  Continuous 1m bars:
    /data/lake/research/continuous_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

What this does:
- For each root/month:
    - reads normalized 1m bars
    - reads continuous daily selected symbol per trade_date_utc
    - joins on (trade_date_utc, symbol)
    - keeps only 1m bars belonging to the daily-selected contract
- Writes one continuous 1m parquet per root/year/month
- Supports resume/incremental mode with a progress file
- Supports --overwrite

Important:
- This first version uses the DAILY selected symbol as the source of truth
  for intraday contract selection.
- That means the continuous 1m series changes contract on the trade_date_utc
  chosen by your daily continuous builder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

NORM_BASE = Path("/data/lake/curated_normalized/ohlcv_1m")
DAILY_CONT_BASE = Path("/data/lake/research/continuous_daily")
OUT_BASE = Path("/data/lake/research/continuous_1m")
PROGRESS_PATH = Path("/data/lake/state/continuous_1m_progress.json")
REPORT_PATH = Path("/data/lake/state/build_continuous_1m_report.json")


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


def read_roots(path: Path) -> List[str]:
    roots: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def out_path(root: str, year: str, month: str) -> Path:
    return OUT_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def norm_path(root: str, year: str, month: str) -> Path:
    return NORM_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def daily_cont_path(root: str, year: str, month: str) -> Path:
    return DAILY_CONT_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def discover_months_for_root(root: str) -> List[Tuple[str, str]]:
    """
    Discover (year, month) partitions from normalized 1m for a root.
    """
    base = NORM_BASE / f"root={root}"
    if not base.exists():
        return []

    out: List[Tuple[str, str]] = []
    for year_dir in sorted(base.glob("year=*")):
        year = year_dir.name.split("=", 1)[1]
        for month_dir in sorted(year_dir.glob("month=*")):
            month = month_dir.name.split("=", 1)[1]
            out.append((year, month))
    return out


def load_daily_selected_symbols(root: str, year: str, month: str) -> Optional[pl.DataFrame]:
    """
    Load daily continuous selected symbols for a root/month.
    Returns columns: trade_date_utc, symbol
    """
    p = daily_cont_path(root, year, month)
    if not p.exists():
        return None

    df = pl.read_parquet(p)

    required = {"trade_date_utc", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing required columns: {sorted(missing)}")

    return (
        df.select(["trade_date_utc", "symbol"])
        .with_columns(
            pl.col("trade_date_utc").cast(pl.Date),
            pl.col("symbol").cast(pl.Utf8),
        )
        .unique(subset=["trade_date_utc"], keep="first")
        .sort("trade_date_utc")
    )


def load_normalized_1m(root: str, year: str, month: str) -> Optional[pl.DataFrame]:
    p = norm_path(root, year, month)
    if not p.exists():
        return None

    df = pl.read_parquet(p)

    required = {
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
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing required columns: {sorted(missing)}")

    return (
        df.select(OUTPUT_COLUMNS)
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
    )


def build_continuous_month(root: str, year: str, month: str) -> Optional[pl.DataFrame]:
    df_1m = load_normalized_1m(root, year, month)
    if df_1m is None or df_1m.is_empty():
        return None

    df_daily = load_daily_selected_symbols(root, year, month)
    if df_daily is None or df_daily.is_empty():
        return None

    # Join daily selected contract onto 1m bars using (trade_date_utc, symbol)
    df_out = (
        df_1m.join(df_daily, on=["trade_date_utc", "symbol"], how="inner")
        .sort(["ts_event", "symbol"])
        .select(OUTPUT_COLUMNS)
    )

    if df_out.is_empty():
        return None

    # Defensive dedup in case normalized layer ever contains unexpected repeats
    df_out = (
        df_out.unique(subset=["root", "symbol", "ts_event"], keep="first", maintain_order=True)
        .sort(["ts_event", "symbol"])
        .select(OUTPUT_COLUMNS)
    )

    return df_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build continuous 1m futures series from normalized 1m + continuous daily contract selection.")
    ap.add_argument("--roots-file", default="/home/marketdata/futures-data/roots.txt")
    ap.add_argument("--root", default=None, help="Optional single-root run, e.g. ES")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild existing continuous 1m partitions.")
    args = ap.parse_args()

    roots_file = Path(args.roots_file).expanduser().resolve()
    if not roots_file.exists():
        raise SystemExit(f"roots file not found: {roots_file}")

    roots = read_roots(roots_file)
    if args.root:
        roots = [r for r in roots if r == args.root]
    if not roots:
        raise SystemExit("No roots selected.")

    progress = load_progress()

    print(f"Roots file: {roots_file}")
    print(f"Roots:      {len(roots)}")
    print(f"NORM_BASE:  {NORM_BASE}")
    print(f"DAILY_BASE: {DAILY_CONT_BASE}")
    print(f"OUT_BASE:   {OUT_BASE}")
    print(f"Progress:   {PROGRESS_PATH}")
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
            print(f"[skip-root] {root}: no normalized 1m partitions found")
            continue

        print(f"=== ROOT {root} months={len(months)} ===")

        for year, month in months:
            op = out_path(root, year, month)

            if (
                not args.overwrite
                and (is_done(progress, root, year, month) or (op.exists() and op.stat().st_size > 0))
            ):
                skipped += 1
                continue

            try:
                df_out = build_continuous_month(root, year, month)

                if df_out is None or df_out.is_empty():
                    empty += 1
                    continue

                ensure_dir(op.parent)
                df_out.write_parquet(op, compression="zstd")

                mark_done(progress, root, year, month)
                save_progress_atomic(progress)

                wrote += 1
                total_rows_written += df_out.height
                print(f"[wrote] {root} {year}-{month}: rows={df_out.height}")

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

        save_progress_atomic(progress)
        print()

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "roots_file": str(roots_file),
        "root_filter": args.root,
        "overwrite": args.overwrite,
        "normalized_base": str(NORM_BASE),
        "continuous_daily_base": str(DAILY_CONT_BASE),
        "out_base": str(OUT_BASE),
        "partitions_written": wrote,
        "partitions_skipped": skipped,
        "partitions_empty": empty,
        "partitions_errored": errors,
        "total_rows_written": total_rows_written,
        "errors": error_details,
        "ok": errors == 0,
    }

    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Wrote report: {REPORT_PATH}")
    print(
        f"Summary: wrote={wrote} skipped={skipped} empty={empty} "
        f"errors={errors} rows={total_rows_written}"
    )

    raise SystemExit(0 if errors == 0 else 2)


if __name__ == "__main__":
    main()