#!/usr/bin/env python3
"""
build_features_1m.py

Build research features from continuous 1-minute futures bars.

Input:
  /data/lake/research/continuous_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Output:
  /data/lake/research/features_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Features included:
- ret_1m: 1-minute close-to-close return
- logret_1m: 1-minute log return
- ret_5m: 5-minute close-to-close return
- logret_5m: 5-minute log return
- hl_spread: high-low spread as fraction of close
- oc_spread: close-open spread as fraction of open
- rolling_vol_30m: rolling std of 1m returns over 30 bars
- rolling_vol_60m: rolling std of 1m returns over 60 bars
- rolling_vol_1d: rolling std of 1m returns over 1440 bars
- volume_ma_30m: rolling mean volume over 30 bars
- volume_ma_60m: rolling mean volume over 60 bars
- dollar_volume: close * volume
- session_vwap: cumulative session VWAP by trade_date_utc
- session_ret: return from first close of trade_date_utc
- session_hod: running session high
- session_lod: running session low
- dist_hod: distance from running session high
- dist_lod: distance from running session low

Notes:
- Works per root/month.
- Uses only data inside each monthly file for rolling windows, so very first rows
  of each month will have nulls for longer lookbacks. This is normal.
- If you later want seamless month-boundary rolling windows, that can be added
  by carrying a tail buffer from prior month.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl

SRC_BASE = Path("/data/lake/research/continuous_1m")
OUT_BASE = Path("/data/lake/research/features_1m")
PROGRESS_PATH = Path("/data/lake/state/features_1m_progress.json")
REPORT_PATH = Path("/data/lake/state/build_features_1m_report.json")

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

FEATURE_COLUMNS = [
    "ret_1m",
    "logret_1m",
    "ret_5m",
    "logret_5m",
    "hl_spread",
    "oc_spread",
    "rolling_vol_30m",
    "rolling_vol_60m",
    "rolling_vol_1d",
    "volume_ma_30m",
    "volume_ma_60m",
    "dollar_volume",
    "session_vwap",
    "session_ret",
    "session_hod",
    "session_lod",
    "dist_hod",
    "dist_lod",
]

BASE_OUTPUT_COLUMNS = REQUIRED_COLUMNS + FEATURE_COLUMNS


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


def src_path(root: str, year: str, month: str) -> Path:
    return SRC_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


def out_path(root: str, year: str, month: str) -> Path:
    return OUT_BASE / f"root={root}" / f"year={year}" / f"month={month}" / "part-0000.parquet"


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
            pl.col("volume").cast(pl.Float64),  # float for rolling/stat math
        )
        .sort("ts_event")
    )


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    close = pl.col("close")
    open_ = pl.col("open")
    high = pl.col("high")
    low = pl.col("low")
    volume = pl.col("volume")

    # returns
    ret_1m = (close / close.shift(1) - 1.0).alias("ret_1m")
    logret_1m = close.log().sub(close.shift(1).log()).alias("logret_1m")

    ret_5m = (close / close.shift(5) - 1.0).alias("ret_5m")
    logret_5m = close.log().sub(close.shift(5).log()).alias("logret_5m")

    # spreads
    hl_spread = ((high - low) / close).alias("hl_spread")
    oc_spread = ((close - open_) / open_).alias("oc_spread")

    # rolling vol
    rolling_vol_30m = pl.col("ret_1m").rolling_std(window_size=30).alias("rolling_vol_30m")
    rolling_vol_60m = pl.col("ret_1m").rolling_std(window_size=60).alias("rolling_vol_60m")
    rolling_vol_1d = pl.col("ret_1m").rolling_std(window_size=1440).alias("rolling_vol_1d")

    # volume features
    volume_ma_30m = volume.rolling_mean(window_size=30).alias("volume_ma_30m")
    volume_ma_60m = volume.rolling_mean(window_size=60).alias("volume_ma_60m")
    dollar_volume = (close * volume).alias("dollar_volume")

    # session features by trade_date_utc
    session_vwap = (
        ((close * volume).cum_sum().over("trade_date_utc")) /
        (volume.cum_sum().over("trade_date_utc"))
    ).alias("session_vwap")

    first_close = close.first().over("trade_date_utc")
    session_ret = (close / first_close - 1.0).alias("session_ret")

    session_hod = high.cum_max().over("trade_date_utc").alias("session_hod")
    session_lod = low.cum_min().over("trade_date_utc").alias("session_lod")

    dist_hod = (close / pl.col("session_hod") - 1.0).alias("dist_hod")
    dist_lod = (close / pl.col("session_lod") - 1.0).alias("dist_lod")

    out = (
        df.with_columns([
            ret_1m,
            logret_1m,
            ret_5m,
            logret_5m,
            hl_spread,
            oc_spread,
        ])
        .with_columns([
            rolling_vol_30m,
            rolling_vol_60m,
            rolling_vol_1d,
            volume_ma_30m,
            volume_ma_60m,
            dollar_volume,
            session_vwap,
            session_ret,
            session_hod,
            session_lod,
        ])
        .with_columns([
            dist_hod,
            dist_lod,
        ])
        .with_columns([
            pl.col("volume").cast(pl.UInt64)
        ])
        .select(BASE_OUTPUT_COLUMNS)
        .sort("ts_event")
    )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 1-minute research features from continuous_1m.")
    ap.add_argument("--roots-file", default="/home/marketdata/futures-data/roots.txt")
    ap.add_argument("--root", default=None, help="Optional single-root run, e.g. ES")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
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
    print(f"SRC_BASE:   {SRC_BASE}")
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
            print(f"[skip-root] {root}: no continuous_1m partitions found")
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
                df = load_continuous_1m(root, year, month)
                if df is None or df.is_empty():
                    empty += 1
                    continue

                out = build_features(df)
                if out.is_empty():
                    empty += 1
                    continue

                ensure_dir(op.parent)
                out.write_parquet(op, compression="zstd")

                mark_done(progress, root, year, month)
                save_progress_atomic(progress)

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

        save_progress_atomic(progress)
        print()

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "roots_file": str(roots_file),
        "root_filter": args.root,
        "overwrite": args.overwrite,
        "src_base": str(SRC_BASE),
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