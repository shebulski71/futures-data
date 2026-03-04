from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from prefect import flow, task, get_run_logger

REPO = Path("/home/marketdata/futures-data").resolve()

def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or REPO), check=True)

@task
def ingest_daily(start: str, end: str, roots_file: str, dataset: str = "GLBX.MDP3") -> None:
    _run([
        "python", str(REPO / "ingest_daily_all_roots.py"),
        "--dataset", dataset,
        "--start", start,
        "--end", end,
        "--roots-file", roots_file,
    ])

@task
def transcode() -> None:
    _run(["python", str(REPO / "transcode_to_parquet.py")])

@task
def normalize_daily() -> None:
    _run(["python", str(REPO / "normalize_daily.py")])

@task
def rebuild_continuous(start: str, end: str, roots_file: str, rules: str) -> None:
    _run([
        "python", str(REPO / "build_continuous_daily_all_roots.py"),
        "--start", start,
        "--end", end,
        "--roots-file", roots_file,
        "--write-rolls",
        "--rules", rules,
    ])

@flow(name="daily_ingest_futures")
def daily_ingest_futures(
    start: str,
    end: str,
    roots_file: str = str(REPO / "roots.txt"),
    dataset: str = "GLBX.MDP3",
    rules: str = "CL=4:10,RB=4:10,HO=4:10,NG=3:7,default=3:0",
) -> None:
    logger = get_run_logger()
    logger.info("start=%s end=%s dataset=%s", start, end, dataset)

    ingest_daily(start, end, roots_file, dataset)
    transcode()
    normalize_daily()
    rebuild_continuous(start, end, roots_file, rules)