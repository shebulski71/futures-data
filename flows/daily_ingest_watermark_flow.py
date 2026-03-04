from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
from prefect import flow, task, get_run_logger

REPO = Path("/home/marketdata/futures-data").resolve()
DUCKDB_PATH = "/data/lake/duckdb/futures.duckdb"

DEFAULT_RULES = "CL=4:10,RB=4:10,HO=4:10,NG=3:7,default=3:0"
DEFAULT_DATASET = "GLBX.MDP3"

# You can set this to 2014-01-01 if you want "roots with no data" to backfill.
DEFAULT_FIRST_DATE = date(2014, 1, 1)


def _run(cmd: List[str], cwd: Path = REPO) -> None:
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _today_ct() -> date:
    return datetime.now(ZoneInfo("America/Chicago")).date()


def _last_complete_trade_date_utc() -> date:
    """
    For daily bars, we ingest *completed* trade_date_utc.
    Running early morning CT, yesterday UTC-date is generally safe.
    We'll use (today CT - 1 day) as the target trade date.
    """
    return _today_ct() - timedelta(days=1)


def _load_roots(roots_file: Path) -> List[str]:
    roots = []
    for line in roots_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return sorted(set(roots))


def _watermarks_from_duckdb(roots: List[str]) -> Dict[str, date]:
    """
    Returns max(trade_date_utc) per root from daily_joined (normalized daily).
    """
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        # daily_joined has trade_date_utc + root partitions (hive)
        q = f"""
        SELECT root, MAX(trade_date_utc) AS max_date
        FROM daily_joined
        WHERE root IN ({",".join([f"'{r}'" for r in roots])})
        GROUP BY root
        """
        rows = con.sql(q).fetchall()
        wm = {}
        for root, max_date in rows:
            if max_date is None:
                continue
            # duckdb returns python date already for DATE columns
            wm[root] = max_date
        return wm
    finally:
        con.close()


def _group_roots_by_start(missing: Dict[str, date]) -> Dict[date, List[str]]:
    buckets: Dict[date, List[str]] = {}
    for root, start_dt in missing.items():
        buckets.setdefault(start_dt, []).append(root)
    for k in buckets:
        buckets[k].sort()
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


@task
def ingest_missing_daily(
    roots_file: str,
    dataset: str = DEFAULT_DATASET,
    target_trade_date: str | None = None,
) -> Tuple[str, str, int]:
    """
    Computes per-root watermarks and ingests only missing days up to target_trade_date (inclusive).
    Returns: (start_iso, end_iso, groups_run)
    """
    logger = get_run_logger()
    roots_path = Path(roots_file).expanduser().resolve()
    roots = _load_roots(roots_path)

    target = date.fromisoformat(target_trade_date) if target_trade_date else _last_complete_trade_date_utc()
    end_exclusive = target + timedelta(days=1)

    wm = _watermarks_from_duckdb(roots)

    # Determine missing start per root:
    missing: Dict[str, date] = {}
    for r in roots:
        last = wm.get(r)
        if last is None:
            # root not present in daily_joined yet
            start_dt = DEFAULT_FIRST_DATE
        else:
            start_dt = last + timedelta(days=1)

        if start_dt < end_exclusive:
            missing[r] = start_dt

    if not missing:
        logger.info("No missing daily data. Everything up to %s is already ingested.", target)
        return (end_exclusive.isoformat(), end_exclusive.isoformat(), 0)

    buckets = _group_roots_by_start(missing)
    logger.info("Need ingest for %d roots, grouped into %d start-date buckets, target=%s",
                len(missing), len(buckets), target)

    groups_run = 0
    min_start = min(buckets.keys())

    for start_dt, roots_subset in buckets.items():
        # Write a temp roots file for this subset
        with tempfile.NamedTemporaryFile(mode="w", delete=False, prefix="roots_", suffix=".txt") as tmp:
            for r in roots_subset:
                tmp.write(r + "\n")
            tmp_path = tmp.name

        try:
            logger.info("Bucket start=%s roots=%d", start_dt.isoformat(), len(roots_subset))
            _run([
                "python", str(REPO / "ingest_daily_all_roots.py"),
                "--dataset", dataset,
                "--start", start_dt.isoformat(),
                "--end", end_exclusive.isoformat(),
                "--roots-file", tmp_path,
            ])
            groups_run += 1
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return (min_start.isoformat(), end_exclusive.isoformat(), groups_run)


@task
def transcode() -> None:
    _run(["python", str(REPO / "transcode_to_parquet.py")])


@task
def normalize_daily() -> None:
    _run(["python", str(REPO / "normalize_daily.py")])


@task
def rebuild_continuous(start: str, end: str, roots_file: str, rules: str) -> None:
    # Rebuild only for the window that changed (start..end)
    _run([
        "python", str(REPO / "build_continuous_daily_all_roots.py"),
        "--start", start,
        "--end", end,
        "--roots-file", roots_file,
        "--write-rolls",
        "--rules", rules,
    ])
    
@task
def postrun_validate(roots_file: str, target_trade_date_utc: str | None = None) -> None:
    cmd = [
        "python", str(REPO / "validate_postrun_daily.py"),
        "--roots-file", roots_file,
        "--strict",
    ]
    if target_trade_date_utc:
        cmd += ["--target-trade-date-utc", target_trade_date_utc]
    _run(cmd)


@flow(name="daily_ingest_futures_watermark")
def daily_ingest_futures_watermark(
    roots_file: str = str(REPO / "roots.txt"),
    dataset: str = DEFAULT_DATASET,
    rules: str = DEFAULT_RULES,
    # Optional override if you want to force a target day
    target_trade_date_utc: str | None = None,
) -> None:
    """
    Watermark ingestion:
      - figures out last ingested trade_date_utc per root from DuckDB daily_joined
      - ingests only missing days up to target_trade_date_utc (inclusive)
      - transcodes + normalizes
      - rebuilds continuous daily for the impacted date window
    """
    logger = get_run_logger()
    logger.info("dataset=%s roots_file=%s target_trade_date_utc=%s", dataset, roots_file, target_trade_date_utc)

    start, end, groups = ingest_missing_daily(roots_file, dataset, target_trade_date_utc)
    if groups == 0:
        logger.info("Nothing ingested; skipping transcode/normalize/rebuild.")
        return

    transcode()
    normalize_daily()
    rebuild_continuous(start, end, roots_file, rules)
    postrun_validate(roots_file, target_trade_date_utc)