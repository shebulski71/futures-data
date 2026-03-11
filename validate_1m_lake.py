"""
validate_1m_lake.py

Sanity validation for raw Databento 1-minute futures parquet lake.

Checks:
- DBN count vs parquet count
- unreadable parquet files
- missing required columns
- null ts_event / symbol / volume
- duplicate (symbol, ts_event) within file
- bad OHLC logic:
    * high < low
    * open outside [low, high]
    * close outside [low, high]
    * negative volume
- partition coverage summary by root/year/month
- global min/max ts_event
- row counts by root

Outputs:
- JSON report under /data/lake/state/validate_1m_lake_report.json

Exit codes:
- 0: validation passed
- 2: validation completed but failed checks
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl


BASE = Path("/data/lake/raw/databento/GLBX.MDP3/ohlcv-1m")
REPORT_PATH = Path("/data/lake/state/validate_1m_lake_report.json")

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


@dataclass
class FileIssue:
    file: str
    issue_type: str
    detail: str | int | dict


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_partition_info(path: Path) -> dict[str, Optional[str]]:
    parts = path.parts
    root = next((p.split("=", 1)[1] for p in parts if p.startswith("root=")), None)
    year = next((p.split("=", 1)[1] for p in parts if p.startswith("year=")), None)
    month = next((p.split("=", 1)[1] for p in parts if p.startswith("month=")), None)
    return {"root": root, "year": year, "month": month}


def safe_iso(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if hasattr(x, "isoformat"):
            return x.isoformat()
        return str(x)
    except Exception:
        return str(x)


def main() -> None:
    if not BASE.exists():
        print(f"Base path does not exist: {BASE}")
        sys.exit(2)

    dbn_files = sorted(BASE.rglob("*.dbn"))
    parquet_files = sorted(BASE.rglob("*.parquet"))

    issues: list[FileIssue] = []

    partitions_seen: set[tuple[str, str, str]] = set()
    root_file_counts: dict[str, int] = defaultdict(int)
    root_row_counts: dict[str, int] = defaultdict(int)

    total_rows = 0
    global_min_ts = None
    global_max_ts = None

    files_checked = 0

    for pq in parquet_files:
        files_checked += 1
        pinfo = parse_partition_info(pq)
        root = pinfo["root"] or "UNKNOWN"
        year = pinfo["year"] or "UNKNOWN"
        month = pinfo["month"] or "UNKNOWN"

        partitions_seen.add((root, year, month))
        root_file_counts[root] += 1

        try:
            df = pl.read_parquet(pq)
        except Exception as e:
            issues.append(FileIssue(str(pq), "read_error", f"{type(e).__name__}: {e}"))
            continue

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            issues.append(FileIssue(str(pq), "missing_columns", {"missing": missing_cols}))
            continue

        rows = df.height
        total_rows += rows
        root_row_counts[root] += rows

        # null checks
        null_ts = df["ts_event"].null_count()
        null_symbol = df["symbol"].null_count()
        null_volume = df["volume"].null_count()

        if null_ts:
            issues.append(FileIssue(str(pq), "null_ts_event", null_ts))
        if null_symbol:
            issues.append(FileIssue(str(pq), "null_symbol", null_symbol))
        if null_volume:
            issues.append(FileIssue(str(pq), "null_volume", null_volume))

        # duplicate (symbol, ts_event)
        try:
            dupes = (
                df.group_by(["symbol", "ts_event"])
                .len()
                .filter(pl.col("len") > 1)
                .height
            )
            if dupes > 0:
                issues.append(FileIssue(str(pq), "duplicate_symbol_ts_event", dupes))
        except Exception as e:
            issues.append(FileIssue(str(pq), "dupe_check_error", f"{type(e).__name__}: {e}"))

        # OHLCV logic
        try:
            bad_hilo = df.filter(pl.col("high") < pl.col("low")).height
            if bad_hilo > 0:
                issues.append(FileIssue(str(pq), "high_lt_low", bad_hilo))

            bad_open = df.filter(
                (pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))
            ).height
            if bad_open > 0:
                issues.append(FileIssue(str(pq), "open_outside_range", bad_open))

            bad_close = df.filter(
                (pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))
            ).height
            if bad_close > 0:
                issues.append(FileIssue(str(pq), "close_outside_range", bad_close))

            neg_vol = df.filter(pl.col("volume") < 0).height
            if neg_vol > 0:
                issues.append(FileIssue(str(pq), "negative_volume", neg_vol))
        except Exception as e:
            issues.append(FileIssue(str(pq), "ohlcv_check_error", f"{type(e).__name__}: {e}"))

        # ts range
        try:
            file_min_ts = df["ts_event"].min()
            file_max_ts = df["ts_event"].max()

            if global_min_ts is None or (file_min_ts is not None and file_min_ts < global_min_ts):
                global_min_ts = file_min_ts
            if global_max_ts is None or (file_max_ts is not None and file_max_ts > global_max_ts):
                global_max_ts = file_max_ts
        except Exception as e:
            issues.append(FileIssue(str(pq), "ts_range_error", f"{type(e).__name__}: {e}"))

    # dbn <-> parquet parity
    missing_parquet_for_dbn = []
    for dbn in dbn_files:
        pq = dbn.with_suffix(".parquet")
        if not pq.exists() or pq.stat().st_size == 0:
            missing_parquet_for_dbn.append(str(dbn))

    orphan_parquet = []
    dbn_set = {str(x.with_suffix(".dbn")) for x in parquet_files}
    actual_dbn_set = {str(x) for x in dbn_files}
    for pq in parquet_files:
        candidate_dbn = str(pq.with_suffix(".dbn"))
        if candidate_dbn not in actual_dbn_set:
            orphan_parquet.append(str(pq))

    issue_counts: dict[str, int] = defaultdict(int)
    for issue in issues:
        issue_counts[issue.issue_type] += 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_path": str(BASE),
        "counts": {
            "dbn_files": len(dbn_files),
            "parquet_files": len(parquet_files),
            "files_checked": files_checked,
            "total_rows": total_rows,
            "root_month_partitions": len(partitions_seen),
            "missing_parquet_for_dbn": len(missing_parquet_for_dbn),
            "orphan_parquet_without_dbn": len(orphan_parquet),
            "issues_total": len(issues),
        },
        "global_time_range": {
            "min_ts_event": safe_iso(global_min_ts),
            "max_ts_event": safe_iso(global_max_ts),
        },
        "root_summary": {
            root: {
                "file_count": root_file_counts[root],
                "row_count": root_row_counts[root],
            }
            for root in sorted(set(root_file_counts) | set(root_row_counts))
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "missing_parquet_for_dbn_examples": missing_parquet_for_dbn[:100],
        "orphan_parquet_examples": orphan_parquet[:100],
        "issues_examples": [asdict(x) for x in issues[:500]],
        "ok": (
            len(missing_parquet_for_dbn) == 0
            and len(orphan_parquet) == 0
            and len(issues) == 0
            and len(dbn_files) == len(parquet_files)
        ),
    }

    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Wrote report: {REPORT_PATH}")
    print("Summary:")
    print(json.dumps({
        "dbn_files": report["counts"]["dbn_files"],
        "parquet_files": report["counts"]["parquet_files"],
        "files_checked": report["counts"]["files_checked"],
        "total_rows": report["counts"]["total_rows"],
        "min_ts_event": report["global_time_range"]["min_ts_event"],
        "max_ts_event": report["global_time_range"]["max_ts_event"],
        "issues_total": report["counts"]["issues_total"],
        "ok": report["ok"],
    }, indent=2))

    sys.exit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()