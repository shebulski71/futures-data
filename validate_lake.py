from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl


NORMALIZED_DAILY = Path("/data/lake/curated_normalized/daily_joined")
CATALOG_DIR = Path("/data/lake/state/catalog_daily")


@dataclass
class RootSummary:
    root: str
    years_present: List[int]
    months_present: int
    sample_checks: List[str]
    problems: List[str]


def read_roots(path: Path) -> List[str]:
    roots = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def list_month_dirs(root: str) -> List[Path]:
    # /data/lake/curated_normalized/daily_joined/root=ES/year=2015/month=01/part-0000.parquet
    base = NORMALIZED_DAILY / f"root={root}"
    if not base.exists():
        return []
    out = []
    for ydir in sorted(base.glob("year=*")):
        for mdir in sorted(ydir.glob("month=*")):
            out.append(mdir)
    return out


def sample_months(month_dirs: List[Path], n: int, seed: int) -> List[Path]:
    if not month_dirs:
        return []
    rng = random.Random(seed)
    if len(month_dirs) <= n:
        return month_dirs
    return rng.sample(month_dirs, n)


def parquet_files_in_dir(mdir: Path) -> List[Path]:
    return sorted(mdir.glob("*.parquet"))


def validate_daily_file(pq: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns (problems, metrics)
    """
    problems = []
    metrics: Dict[str, int] = {}

    if not pq.exists() or pq.stat().st_size == 0:
        return [f"missing_or_empty_file: {pq}"], metrics

    # Lazy scan for speed
    lf = pl.scan_parquet(str(pq))

    # Required columns
    required = {
        "instrument_id", "symbol", "trade_date_utc",
        "open", "high", "low", "close", "volume"
    }
    cols = set(lf.collect_schema().names())
    missing = sorted(list(required - cols))
    if missing:
        problems.append(f"missing_columns={missing}")
        return problems, metrics

    # Basic row count
    n = lf.select(pl.len()).collect().item()
    metrics["rows"] = int(n)
    if n == 0:
        problems.append("zero_rows")
        return problems, metrics

    # Duplicate key check: (instrument_id, trade_date_utc)
    dup = (
        lf.group_by(["instrument_id", "trade_date_utc"])
          .agg(pl.len().alias("n"))
          .filter(pl.col("n") > 1)
          .select(pl.len())
          .collect()
          .item()
    )
    metrics["duplicate_keys"] = int(dup)
    if dup > 0:
        problems.append(f"duplicate_keys={dup}")

    # OHLC sanity checks
    bad_ohlc = (
        lf.filter(
            (pl.col("high") < pl.max_horizontal(["open", "close"])) |
            (pl.col("low") > pl.min_horizontal(["open", "close"])) |
            (pl.col("high") < pl.col("low")) |
            (pl.col("volume") < 0)
        )
        .select(pl.len())
        .collect()
        .item()
    )
    metrics["bad_ohlc_rows"] = int(bad_ohlc)
    if bad_ohlc > 0:
        problems.append(f"bad_ohlc_rows={bad_ohlc}")

    # Date within partition sanity (optional but useful)
    # infer year/month from path: .../year=YYYY/month=MM/...
    try:
        year = int(pq.parent.parent.name.split("=")[1])
        month = int(pq.parent.name.split("=")[1])
        min_d, max_d = (
            lf.select([
                pl.col("trade_date_utc").min().alias("min_d"),
                pl.col("trade_date_utc").max().alias("max_d"),
            ]).collect().row(0)
        )
        metrics["min_date"] = int(min_d.year) * 10000 + int(min_d.month) * 100 + int(min_d.day)
        metrics["max_date"] = int(max_d.year) * 10000 + int(max_d.month) * 100 + int(max_d.day)
        if min_d.year != year and max_d.year != year:
            # allow some edge weirdness but flag if fully outside
            problems.append(f"dates_outside_year_partition min={min_d} max={max_d} partition={year}")
        if min_d.month != month and max_d.month != month:
            problems.append(f"dates_outside_month_partition min={min_d} max={max_d} partition={month:02d}")
    except Exception:
        pass

    return problems, metrics


def main():
    ap = argparse.ArgumentParser(description="Validate daily lake outputs (quick sanity + duplicates + OHLC checks).")
    ap.add_argument("--roots-file", default="roots.txt")
    ap.add_argument("--samples-per-root", type=int, default=6, help="How many months to sample per root")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/data/lake/state/validation_daily_report.json")
    args = ap.parse_args()

    roots = read_roots(Path(args.roots_file))
    report = {
        "generated_at": str(date.today()),
        "normalized_daily_base": str(NORMALIZED_DAILY),
        "roots": {},
        "summary": {
            "roots_total": len(roots),
            "roots_missing_all": 0,
            "roots_with_problems": 0,
            "files_checked": 0,
            "files_with_problems": 0,
        },
    }

    roots_missing_all = 0
    roots_with_problems = 0
    files_checked = 0
    files_with_problems = 0

    for root in roots:
        month_dirs = list_month_dirs(root)
        years = sorted({int(p.parent.name.split("=")[1]) for p in month_dirs}) if month_dirs else []

        entry = {
            "years_present": years,
            "months_present": len(month_dirs),
            "checks": [],
            "problems": [],
        }

        if not month_dirs:
            roots_missing_all += 1
            entry["problems"].append("no_daily_joined_partitions_found")
            report["roots"][root] = entry
            continue

        samples = sample_months(month_dirs, args.samples_per_root, args.seed + hash(root) % 10000)

        for mdir in samples:
            pqs = parquet_files_in_dir(mdir)
            if not pqs:
                entry["problems"].append(f"no_parquet_in {mdir}")
                continue

            # we expect part-0000.parquet but accept any parquet
            for pq in pqs:
                files_checked += 1
                probs, metrics = validate_daily_file(pq)
                if probs:
                    files_with_problems += 1
                    entry["problems"].append(
                        f"{pq}: " + "; ".join(probs) + f" metrics={metrics}"
                    )
                else:
                    entry["checks"].append(f"{pq}: ok metrics={metrics}")
                # only check one file per month dir (keep it fast)
                break

        if entry["problems"]:
            roots_with_problems += 1

        report["roots"][root] = entry

    report["summary"]["roots_missing_all"] = roots_missing_all
    report["summary"]["roots_with_problems"] = roots_with_problems
    report["summary"]["files_checked"] = files_checked
    report["summary"]["files_with_problems"] = files_with_problems

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report: {out_path}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()