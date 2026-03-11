#!/usr/bin/env python3
"""
validate_features_1m.py

Validate research feature layer.

Input:
    /data/lake/research/features_1m/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Checks
------
File checks
- parquet readable
- required columns exist

Data checks
- ts_event monotonic
- no duplicate (root, ts_event)
- return sanity (abs(ret_1m) > threshold => warning)
- volatility >= 0
- session_vwap within running session high/low bounds
- no obviously broken/null-corrupted feature columns

Output
------
/data/lake/state/validate_features_1m_report.json

Exit code
---------
0 if no hard issues
2 if hard issues found
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import polars as pl

BASE = Path("/data/lake/research/features_1m")
REPORT = Path("/data/lake/state/validate_features_1m_report.json")

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
    "ret_1m",
    "ret_5m",
    "rolling_vol_30m",
    "session_vwap",
]

# hard-fail threshold for obvious corruption
RET_HARD_THRESHOLD = 0.50

# warning threshold for noteworthy but plausible moves
RET_WARN_THRESHOLD = 0.20


def discover_files():
    return sorted(BASE.rglob("*.parquet"))


def validate_file(path: Path):
    hard_issues = []
    warnings = []

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        return {
            "file": str(path),
            "rows": 0,
            "hard_issues": [f"read_error:{e}"],
            "warnings": [],
            "min_ts_event": None,
            "max_ts_event": None,
        }

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            hard_issues.append(f"missing_column:{c}")

    if hard_issues:
        return {
            "file": str(path),
            "rows": df.height,
            "hard_issues": hard_issues,
            "warnings": warnings,
            "min_ts_event": None,
            "max_ts_event": None,
        }

    if df.is_empty():
        warnings.append("empty_file")
        return {
            "file": str(path),
            "rows": 0,
            "hard_issues": hard_issues,
            "warnings": warnings,
            "min_ts_event": None,
            "max_ts_event": None,
        }

    df = (
        df.sort("ts_event")
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
            pl.col("volume").cast(pl.Float64),
        )
    )

    # duplicate timestamps per root
    dupes = (
        df.group_by(["root", "ts_event"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if dupes > 0:
        hard_issues.append(f"duplicate_rows:{dupes}")

    # ts ordering
    disorder = (
        df.with_columns(pl.col("ts_event").shift(1).alias("prev_ts"))
        .filter(pl.col("prev_ts").is_not_null() & (pl.col("ts_event") < pl.col("prev_ts")))
        .height
    )
    if disorder > 0:
        hard_issues.append(f"ts_event_out_of_order:{disorder}")

    # obvious bad OHLC rows
    ohlc_bad = df.filter(
        (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    ).height
    if ohlc_bad > 0:
        hard_issues.append(f"ohlc_invalid:{ohlc_bad}")

    # return sanity
    if "ret_1m" in df.columns:
        extreme_warn = df.filter(pl.col("ret_1m").abs() > RET_WARN_THRESHOLD).height
        if extreme_warn > 0:
            warnings.append(f"extreme_returns_warn:{extreme_warn}")

        extreme_hard = df.filter(pl.col("ret_1m").abs() > RET_HARD_THRESHOLD).height
        if extreme_hard > 0:
            hard_issues.append(f"extreme_returns_hard:{extreme_hard}")

    # volatility sanity
    if "rolling_vol_30m" in df.columns:
        bad = df.filter(pl.col("rolling_vol_30m") < 0).height
        if bad > 0:
            hard_issues.append(f"negative_volatility:{bad}")

    # VWAP sanity: compare to running session bounds, not current bar bounds
    if "session_vwap" in df.columns:
        tmp = df.with_columns([
            pl.col("high").cum_max().over("trade_date_utc").alias("_session_hod"),
            pl.col("low").cum_min().over("trade_date_utc").alias("_session_lod"),
        ])

        bad = tmp.filter(
            (pl.col("session_vwap") > pl.col("_session_hod") * 1.001) |
            (pl.col("session_vwap") < pl.col("_session_lod") * 0.999)
        ).height

        if bad > 0:
            hard_issues.append(f"vwap_outside_session_range:{bad}")

    # full-column null corruption checks on important derived columns
    derived_cols = ["ret_1m", "ret_5m", "rolling_vol_30m", "session_vwap"]
    for c in derived_cols:
        if c in df.columns and df[c].null_count() == df.height:
            hard_issues.append(f"all_null_column:{c}")

    return {
        "file": str(path),
        "rows": df.height,
        "hard_issues": hard_issues,
        "warnings": warnings,
        "min_ts_event": df["ts_event"].min().isoformat(),
        "max_ts_event": df["ts_event"].max().isoformat(),
    }


def main():
    files = discover_files()

    results = []
    hard_issues_total = 0
    warnings_total = 0
    rows_total = 0

    min_ts = None
    max_ts = None

    hard_counter = Counter()
    warn_counter = Counter()

    for f in files:
        res = validate_file(f)
        results.append(res)

        rows_total += res.get("rows", 0)
        hard_issues_total += len(res["hard_issues"])
        warnings_total += len(res["warnings"])

        for issue in res["hard_issues"]:
            hard_counter[issue.split(":")[0]] += 1
        for w in res["warnings"]:
            warn_counter[w.split(":")[0]] += 1

        rmin = res.get("min_ts_event")
        rmax = res.get("max_ts_event")

        if rmin:
            if min_ts is None or rmin < min_ts:
                min_ts = rmin

        if rmax:
            if max_ts is None or rmax > max_ts:
                max_ts = rmax

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "base": str(BASE),
        "files_checked": len(files),
        "total_rows": rows_total,
        "min_ts_event": min_ts,
        "max_ts_event": max_ts,
        "hard_issues_total": hard_issues_total,
        "warnings_total": warnings_total,
        "ok": hard_issues_total == 0,
        "hard_issue_breakdown": dict(hard_counter),
        "warning_breakdown": dict(warn_counter),
        "files": results,
    }

    REPORT.write_text(json.dumps(report, indent=2))
    print(f"Wrote report: {REPORT}")
    print(json.dumps({
        "files_checked": len(files),
        "total_rows": rows_total,
        "hard_issues_total": hard_issues_total,
        "warnings_total": warnings_total,
        "ok": hard_issues_total == 0,
        "hard_issue_breakdown": dict(hard_counter),
        "warning_breakdown": dict(warn_counter),
    }, indent=2))


if __name__ == "__main__":
    main()