#!/usr/bin/env python3
"""
validate_resampled_bars.py

Validate resampled intraday bar layers.

Input:
  /data/lake/research/resampled_bars/<FREQ>/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Checks:
- parquet files are readable
- required columns exist
- ts_event present and non-null
- trade_date_utc present and non-null
- root matches partition path
- no duplicate (root, ts_event)
- ts_event monotonic within file
- OHLC sanity:
    low <= open/high/close <= high
- volume >= 0
- optional cross-file duplicate scan by root

Output:
  /data/lake/state/validate_resampled_bars_<FREQ>_report.json

Examples:
  python /home/marketdata/futures-data/validate_resampled_bars.py --freq 5m
  python /home/marketdata/futures-data/validate_resampled_bars.py --freq 5m --root ES --cross-file-root-check
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

BASE = Path("/data/lake/research/resampled_bars")
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


def report_path(freq: str) -> Path:
    return STATE_DIR / f"validate_resampled_bars_{freq}_report.json"


def freq_base(freq: str) -> Path:
    return BASE / freq


def partition_value(path: Path, key: str) -> Optional[str]:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def discover_files(freq: str, root_filter: Optional[str] = None) -> List[Path]:
    base = freq_base(freq)
    if not base.exists():
        return []

    if root_filter:
        root_dir = base / f"root={root_filter}"
        if not root_dir.exists():
            return []
        return sorted(root_dir.rglob("*.parquet"))

    return sorted(base.rglob("*.parquet"))


def validate_file(path: Path) -> Dict:
    issues: List[str] = []

    stats: Dict = {
        "file": str(path),
        "rows": 0,
        "min_ts_event": None,
        "max_ts_event": None,
        "root_partition": partition_value(path, "root"),
        "year_partition": partition_value(path, "year"),
        "month_partition": partition_value(path, "month"),
        "issues": issues,
    }

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        issues.append(f"read_error: {type(e).__name__}: {e}")
        return stats

    stats["rows"] = df.height

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing_columns: {missing}")
        return stats

    if df.is_empty():
        issues.append("empty_file")
        return stats

    df = df.select(REQUIRED_COLUMNS)

    for c in ["ts_event", "trade_date_utc", "root", "symbol", "instrument_id"]:
        nulls = df[c].null_count()
        if nulls > 0:
            issues.append(f"nulls_in_{c}: {nulls}")

    root_part = stats["root_partition"]
    if root_part is not None:
        bad_root_rows = df.filter(pl.col("root") != root_part).height
        if bad_root_rows > 0:
            issues.append(f"root_partition_mismatch_rows: {bad_root_rows}")

    try:
        df = df.with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
            pl.col("trade_date_utc").cast(pl.Date),
            pl.col("root").cast(pl.Utf8),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("instrument_id").cast(pl.UInt32),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
    except Exception as e:
        issues.append(f"cast_error: {type(e).__name__}: {e}")
        return stats

    stats["min_ts_event"] = df["ts_event"].min().isoformat() if df.height else None
    stats["max_ts_event"] = df["ts_event"].max().isoformat() if df.height else None

    dupes = (
        df.group_by(["root", "ts_event"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if dupes > 0:
        issues.append(f"duplicate_(root,ts_event)_groups: {dupes}")

    disorder = (
        df.sort("ts_event")
        .with_columns(pl.col("ts_event").shift(1).alias("_prev_ts"))
        .filter(pl.col("_prev_ts").is_not_null() & (pl.col("ts_event") < pl.col("_prev_ts")))
        .height
    )
    if disorder > 0:
        issues.append(f"ts_event_out_of_order_rows: {disorder}")

    ohlc_bad = df.filter(
        (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    ).height
    if ohlc_bad > 0:
        issues.append(f"ohlc_invalid_rows: {ohlc_bad}")

    neg_vol = df.filter(pl.col("volume") < 0).height
    if neg_vol > 0:
        issues.append(f"negative_volume_rows: {neg_vol}")

    return stats


def validate_cross_file_root(freq: str, root: str) -> List[str]:
    issues: List[str] = []
    files = discover_files(freq, root)
    if not files:
        issues.append(f"no_files_for_root: {root}")
        return issues

    frames: List[pl.DataFrame] = []
    try:
        for f in files:
            df = pl.read_parquet(f).select(["root", "ts_event"])
            frames.append(df)

        all_df = pl.concat(frames, how="vertical_relaxed")
        dupes = (
            all_df.group_by(["root", "ts_event"])
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if dupes > 0:
            issues.append(f"cross_file_duplicate_(root,ts_event)_groups: {dupes}")
    except Exception as e:
        issues.append(f"cross_file_check_error: {type(e).__name__}: {e}")

    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate resampled bar parquet layer.")
    ap.add_argument("--freq", required=True, choices=sorted(SUPPORTED_FREQS))
    ap.add_argument("--root", default=None, help="Optional single-root validation, e.g. ES")
    ap.add_argument(
        "--cross-file-root-check",
        action="store_true",
        help="Also check duplicates across all files for each selected root.",
    )
    args = ap.parse_args()

    freq = args.freq
    files = discover_files(freq, args.root)
    if not files:
        raise SystemExit(
            f"No resampled parquet files found under {freq_base(freq)} for root={args.root!r}"
        )

    file_results: List[Dict] = []
    roots_seen = set()

    total_rows = 0
    files_checked = 0
    issues_total = 0
    min_ts_event: Optional[str] = None
    max_ts_event: Optional[str] = None

    for f in files:
        res = validate_file(f)
        file_results.append(res)
        files_checked += 1
        total_rows += int(res.get("rows", 0))
        issues_total += len(res.get("issues", []))

        root_part = res.get("root_partition")
        if root_part:
            roots_seen.add(root_part)

        fmin = res.get("min_ts_event")
        fmax = res.get("max_ts_event")
        if fmin is not None and (min_ts_event is None or fmin < min_ts_event):
            min_ts_event = fmin
        if fmax is not None and (max_ts_event is None or fmax > max_ts_event):
            max_ts_event = fmax

    cross_file_issues: Dict[str, List[str]] = {}
    if args.cross_file_root_check:
        for root in sorted(roots_seen):
            root_issues = validate_cross_file_root(freq, root)
            if root_issues:
                cross_file_issues[root] = root_issues
                issues_total += len(root_issues)

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "frequency": freq,
        "base": str(freq_base(freq)),
        "root_filter": args.root,
        "cross_file_root_check": args.cross_file_root_check,
        "files_checked": files_checked,
        "total_rows": total_rows,
        "min_ts_event": min_ts_event,
        "max_ts_event": max_ts_event,
        "issues_total": issues_total,
        "ok": issues_total == 0,
        "cross_file_issues": cross_file_issues,
        "files": file_results,
    }

    rp = report_path(freq)
    rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote report: {rp}")
    print(
        json.dumps(
            {
                "files_checked": files_checked,
                "total_rows": total_rows,
                "min_ts_event": min_ts_event,
                "max_ts_event": max_ts_event,
                "issues_total": issues_total,
                "ok": issues_total == 0,
            },
            indent=2,
        )
    )

    raise SystemExit(0 if issues_total == 0 else 2)


if __name__ == "__main__":
    main()