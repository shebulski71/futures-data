#!/usr/bin/env python3
"""
daily_ingest_1m_watermark_flow.py

Incremental daily 1m futures ingestion and downstream rebuild pipeline.

Pipeline
--------
1) Determine latest normalized 1m trade date already present
2) Compute missing date window up to target_trade_date_utc
3) Ingest missing 1m raw data
4) Transcode new DBNs -> parquet
5) Normalize affected 1m partitions
6) Rebuild continuous_1m for affected window
7) Rebuild features_1m for affected window
8) Rebuild resampled bars for affected window
9) Validate outputs

Notes
-----
- Dates are ISO only: YYYY-MM-DD
- End date is treated as exclusive for the ingest/rebuild scripts
- Default target is yesterday UTC to avoid requesting partially available data
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import duckdb
from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner


REPO = Path("/home/marketdata/futures-data")

DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_ROOTS_FILE = str(REPO / "roots.txt")


# ============================================================
# HELPERS
# ============================================================

@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], cwd: Optional[Path] = None) -> RunResult:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )

    stdout = p.stdout or ""
    stderr = p.stderr or ""

    if p.returncode != 0:
        if stdout.strip():
            print("stdout:\n", stdout)
        if stderr.strip():
            print("stderr:\n", stderr)
        raise subprocess.CalledProcessError(
            p.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )

    return RunResult(p.returncode, stdout, stderr)


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _default_target_trade_date_utc() -> str:
    return (_utc_today() - timedelta(days=1)).isoformat()


def _normalize_iso_date(s: str) -> str:
    return date.fromisoformat(str(s).strip()).isoformat()


# ============================================================
# TASKS
# ============================================================

@task
def determine_missing_window(
    target_trade_date_utc: Optional[str],
) -> Tuple[str, str, int]:
    """
    Determine missing normalized 1m date window.

    Returns
    -------
    (start_iso, end_iso, days_missing)

    where end_iso is exclusive.
    """
    logger = get_run_logger()

    target = _normalize_iso_date(target_trade_date_utc or _default_target_trade_date_utc())
    target_dt = date.fromisoformat(target)

    # Read max normalized trade date from normalized 1m lake
    max_date: Optional[str] = None
    glob_path = "/data/lake/curated_normalized/ohlcv_1m/**/*.parquet"

    try:
        con = duckdb.connect(database=":memory:")
        q = f"""
        SELECT MAX(trade_date_utc)::VARCHAR AS max_date
        FROM read_parquet('{glob_path}')
        """
        max_date = con.execute(q).fetchone()[0]
    except Exception as e:
        logger.warning("Could not query normalized 1m max_date (%s). Assuming empty lake.", e)

    if max_date:
        start_dt = date.fromisoformat(max_date) + timedelta(days=1)
    else:
        start_dt = target_dt

    if start_dt > target_dt:
        logger.info("No missing 1m data. max normalized date already >= target=%s", target)
        return target, (target_dt + timedelta(days=1)).isoformat(), 0

    start = start_dt.isoformat()
    end = (target_dt + timedelta(days=1)).isoformat()
    days_missing = (target_dt - start_dt).days + 1

    logger.info("Missing 1m window: start=%s end=%s days=%s target=%s", start, end, days_missing, target)
    return start, end, days_missing


@task
def ingest_1m_window(
    roots_file: str,
    dataset: str,
    start: str,
    end: str,
    batch_size: int = 100,
    force_window: bool = False,
) -> None:
    logger = get_run_logger()

    cmd = [
        "python",
        str(REPO / "ingest_1m_all_roots.py"),
        "--dataset",
        dataset,
        "--start",
        _normalize_iso_date(start),
        "--end",
        _normalize_iso_date(end),
        "--roots-file",
        roots_file,
        "--batch-size",
        str(batch_size),
    ]
    if force_window:
        cmd.append("--force-window")

    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def transcode_1m(
    dataset: str,
    jobs: int = 8,
    force: bool = False,
) -> None:
    logger = get_run_logger()

    cmd = [
        "python",
        str(REPO / "transcode_to_parquet.py"),
    ]

    env = {
        "TRANSCODE_DATASET": dataset,
        "TRANSCODE_JOBS": str(jobs),
        "TRANSCODE_FORCE": "1" if force else "0",
    }

    logger.info("Running: %s with env=%s", " ".join(cmd), env)

    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )

    if p.returncode != 0:
        raise subprocess.CalledProcessError(
            p.returncode,
            cmd,
            output=p.stdout,
            stderr=p.stderr,
        )


@task
def normalize_1m(root_filter: Optional[str] = None, overwrite: bool = False) -> None:
    logger = get_run_logger()

    cmd = [
        "python",
        str(REPO / "normalize_1m.py"),
    ]
    if root_filter:
        cmd.extend(["--root", root_filter])
    if overwrite:
        cmd.append("--overwrite")

    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def build_continuous_1m(
    roots_file: str,
    root_filter: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_run_logger()

    cmd = [
        "python",
        str(REPO / "build_continuous_1m.py"),
        "--roots-file",
        roots_file,
    ]

    if root_filter:
        cmd.extend(["--root", root_filter])

    if overwrite:
        cmd.append("--overwrite")

    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def build_features_1m(
    roots_file: str,
    root_filter: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_run_logger()

    cmd = [
        "python",
        str(REPO / "build_features_1m.py"),
        "--roots-file",
        roots_file,
    ]

    if root_filter:
        cmd.extend(["--root", root_filter])

    if overwrite:
        cmd.append("--overwrite")

    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def build_resampled_bars(
    roots_file: str,
    root_filter: str | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_run_logger()

    freqs = ["2m", "5m", "10m", "15m", "30m", "1h"]

    for freq in freqs:
        cmd = [
            "python",
            str(REPO / "build_resampled_bars.py"),
            "--freq",
            freq,
            "--roots-file",
            roots_file,
        ]

        if root_filter:
            cmd.extend(["--root", root_filter])

        if overwrite:
            cmd.append("--overwrite")

        logger.info("Running: %s", " ".join(cmd))
        _run(cmd, cwd=REPO)


@task
def validate_all(root_filter: str | None = None) -> None:
    logger = get_run_logger()

    # 1) continuous
    cmd = ["python", str(REPO / "validate_continuous_1m.py")]
    if root_filter:
        cmd.extend(["--root", root_filter])
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)

    # 2) features
    cmd = ["python", str(REPO / "validate_features_1m.py")]
    if root_filter:
        cmd.extend(["--root", root_filter])
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)

    # 3) resampled bars: one validation per frequency
    freqs = ["2m", "5m", "10m", "15m", "30m", "1h"]

    for freq in freqs:
        cmd = [
            "python",
            str(REPO / "validate_resampled_bars.py"),
            "--freq",
            freq,
        ]
        if root_filter:
            cmd.extend(["--root", root_filter])

        logger.info("Running: %s", " ".join(cmd))
        _run(cmd, cwd=REPO)

# ============================================================
# FLOW
# ============================================================

@flow(name="daily_ingest_futures_1m_watermark", task_runner=ThreadPoolTaskRunner(max_workers=1))
def daily_ingest_futures_1m_watermark(
    roots_file: str = DEFAULT_ROOTS_FILE,
    dataset: str = DEFAULT_DATASET,
    target_trade_date_utc: str | None = None,
    batch_size: int = 100,
    force_window: bool = False,
    transcode_jobs: int = 8,
    transcode_force: bool = False,
    overwrite_downstream: bool = False,
) -> None:
    """
    Daily incremental 1m pipeline.

    Default target is yesterday UTC.
    """
    logger = get_run_logger()
    logger.info(
        "dataset=%s roots_file=%s target_trade_date_utc=%s batch_size=%s",
        dataset,
        roots_file,
        target_trade_date_utc,
        batch_size,
    )

    start, end, days_missing = determine_missing_window(target_trade_date_utc)

    if days_missing == 0:
        logger.info("No new 1m days missing. Running validations only.")
        validate_all()
        return

    ingest_1m_window(
        roots_file=roots_file,
        dataset=dataset,
        start=start,
        end=end,
        batch_size=batch_size,
        force_window=force_window,
    )

    transcode_1m(dataset=dataset, jobs=transcode_jobs, force=transcode_force)
    normalize_1m(overwrite=overwrite_downstream)
    build_continuous_1m(roots_file=roots_file,overwrite=overwrite_downstream)
    build_features_1m(roots_file=roots_file,overwrite=overwrite_downstream,)
    build_resampled_bars(roots_file=roots_file,overwrite=overwrite_downstream,)
    validate_all()