#!/usr/bin/env python3
"""
backfill_1m_flow.py

Manual / ad-hoc Prefect flow for historical 1-minute futures backfills.

What it does:
- runs ingest_1m_all_roots.py for a requested date window
- optionally runs transcode_to_parquet.py afterward
- writes subprocess stdout/stderr into Prefect logs
- is intended for manual deployment runs, not a schedule

Typical use:
    prefect deployment run "backfill_futures_1m/backfill-1m" \
      --param start="2015-04-01" \
      --param end="2016-01-01"

Assumptions:
- repo lives at /home/marketdata/futures-data
- your existing worker is already attached to work pool: futures-data-pool
- ingest_1m_all_roots.py already exists and works from CLI
- transcode_to_parquet.py already exists and preserves ts_event
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

REPO = Path("/home/marketdata/futures-data").resolve()

DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_ROOTS_FILE = str(REPO / "roots.txt")


def _run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> None:
    logger = get_run_logger()
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    logger.info("Running: %s", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd or REPO),
        env=merged_env,
        text=True,
        capture_output=True,
    )

    if p.stdout:
        logger.info("stdout:\n%s", p.stdout[-12000:])
    if p.stderr:
        logger.warning("stderr:\n%s", p.stderr[-12000:])

    if p.returncode != 0:
        raise subprocess.CalledProcessError(
            p.returncode,
            cmd,
            output=p.stdout,
            stderr=p.stderr,
        )


@task
def ingest_1m_window(
    start: str,
    end: str,
    roots_file: str = DEFAULT_ROOTS_FILE,
    dataset: str = DEFAULT_DATASET,
    batch_size: int = 100,
    strict: bool = False,
    force_window: bool = False,
) -> None:
    cmd = [
        "python",
        str(REPO / "ingest_1m_all_roots.py"),
        "--dataset",
        dataset,
        "--start",
        start,
        "--end",
        end,
        "--roots-file",
        roots_file,
        "--batch-size",
        str(batch_size),
    ]
    if strict:
        cmd.append("--strict")
    if force_window:
        cmd.append("--force-window")

    _run(cmd, cwd=REPO)


@task
def transcode_all_dbn_for_dataset(
    dataset: str = DEFAULT_DATASET,
    jobs: int = 8,
    force: bool = False,
) -> None:
    env = {
        "TRANSCODE_DATASET": dataset,
        "TRANSCODE_JOBS": str(jobs),
        "TRANSCODE_FORCE": "1" if force else "0",
    }

    cmd = [
        "python",
        str(REPO / "transcode_to_parquet.py"),
    ]
    _run(cmd, cwd=REPO, env=env)


@flow(
    name="backfill_futures_1m",
    task_runner=ThreadPoolTaskRunner(max_workers=1),
)
def backfill_futures_1m(
    start: str,
    end: str,
    roots_file: str = DEFAULT_ROOTS_FILE,
    dataset: str = DEFAULT_DATASET,
    batch_size: int = 100,
    strict: bool = False,
    force_window: bool = False,
    transcode_after: bool = True,
    transcode_jobs: int = 8,
    transcode_force: bool = False,
) -> None:
    """
    Backfill 1-minute futures data for a fixed historical window.

    Parameters:
    - start: YYYY-MM-DD inclusive
    - end:   YYYY-MM-DD exclusive-ish
    - roots_file: roots list
    - dataset: Databento dataset
    - batch_size: symbols per request batch
    - strict: fail flow if any month fails inside the ingest script
    - force_window: ignore progress-based skips in the ingest script
    - transcode_after: whether to run transcode_to_parquet.py after ingest
    - transcode_jobs: process count for transcode_to_parquet.py
    - transcode_force: overwrite existing parquet outputs
    """
    logger = get_run_logger()
    logger.info(
        "start=%s end=%s dataset=%s roots_file=%s batch_size=%s strict=%s force_window=%s transcode_after=%s transcode_jobs=%s transcode_force=%s",
        start,
        end,
        dataset,
        roots_file,
        batch_size,
        strict,
        force_window,
        transcode_after,
        transcode_jobs,
        transcode_force,
    )

    ingest_1m_window(
        start=start,
        end=end,
        roots_file=roots_file,
        dataset=dataset,
        batch_size=batch_size,
        strict=strict,
        force_window=force_window,
    )

    if transcode_after:
        transcode_all_dbn_for_dataset(
            dataset=dataset,
            jobs=transcode_jobs,
            force=transcode_force,
        )

    logger.info("1m backfill flow complete.")