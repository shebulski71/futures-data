#!/usr/bin/env python3
"""
daily_ingest_watermark_flow.py

Prefect flow for daily futures watermark ingestion.

Key behaviors:
- Finds missing daily windows per root and ingests only what's missing
- Transcodes DBN -> parquet
- Normalizes
- Rebuilds continuous daily for impacted window
- Validates postrun invariants (daily_joined and continuous_daily watermark)

Improvements:
1) Databento "available_end" clamp:
   If ingest requests end after available_end, parse available_end from error
   and retry once with a clamped --end.
2) Sequential task execution:
   Use ThreadPoolTaskRunner(max_workers=1) to avoid overlap and keep ordering strict.
3) More robust DuckDB parquet glob + dynamic roots count.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import duckdb
from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

REPO = Path("/home/marketdata/futures-data")
STATE_DIR = Path("/data/lake/state")

DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_RULES = "CL=4:10,RB=4:10,HO=4:10,NG=3:7,default=3:0"

# DuckDB-friendly partition glob (avoid **)
DAILY_JOINED_GLOB = "/data/lake/curated_normalized/daily_joined/root=*/year=*/month=*/part-*.parquet"

_AVAILABLE_END_RE = re.compile(r"data available up to\s+'(?P<ts>[^']+)'", re.IGNORECASE)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _default_target_trade_date_utc() -> str:
    # safest default for daily bars: yesterday in UTC
    return (_utc_today() - timedelta(days=1)).isoformat()


def _read_roots(path: str) -> List[str]:
    p = Path(path)
    roots: List[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def _parse_available_end(stdout: str, stderr: str) -> Optional[datetime]:
    """
    Parse Databento available_end timestamp from error text like:
      The dataset GLBX.MDP3 has data available up to '2026-03-05 00:00:00+00:00'.
    Returns aware datetime in UTC if found.
    """
    txt = "\n".join([stdout or "", stderr or ""])
    m = _AVAILABLE_END_RE.search(txt)
    if not m:
        return None

    raw = m.group("ts").strip()
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _replace_cli_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out):
            out[i + 1] = value
            return out
    out.extend([flag, value])
    return out


def _get_cli_arg(cmd: list[str], flag: str) -> Optional[str]:
    if flag in cmd:
        i = cmd.index(flag)
        if i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _run(cmd: list[str], cwd: Optional[Path] = None) -> RunResult:
    """
    Run a subprocess command and return stdout/stderr, raising CalledProcessError on failure.
    Includes ONE retry for Databento available_end clamp when a 422 occurs.
    """
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    stdout = p.stdout or ""
    stderr = p.stderr or ""

    if p.returncode == 0:
        return RunResult(p.returncode, stdout, stderr)

    combined = (stdout + "\n" + stderr).lower()

    # ---- Databento available_end clamp retry ----
    if "data_end_after_available_end" in combined:
        avail_end = _parse_available_end(stdout, stderr)
        end_str = _get_cli_arg(cmd, "--end")

        if avail_end and end_str:
            # If available_end is e.g. 2026-03-05 00:00 UTC,
            # the latest safe exclusive end date is 2026-03-05.
            clamped_end = avail_end.date().isoformat()
            if clamped_end != end_str:
                cmd2 = _replace_cli_arg(cmd, "--end", clamped_end)
                p2 = subprocess.run(
                    cmd2,
                    cwd=str(cwd) if cwd else None,
                    capture_output=True,
                    text=True,
                )
                stdout2 = p2.stdout or ""
                stderr2 = p2.stderr or ""
                if p2.returncode == 0:
                    return RunResult(p2.returncode, stdout2, stderr2)

                raise subprocess.CalledProcessError(
                    p2.returncode, cmd2, output=stdout2, stderr=stderr2
                )

    raise subprocess.CalledProcessError(p.returncode, cmd, output=stdout, stderr=stderr)


@task
def ingest_missing_daily(
    roots_file: str,
    dataset: str,
    target_trade_date_utc: Optional[str],
) -> Tuple[str, str, int]:
    """
    Determine missing window by reading daily_joined max date; if missing, ingest.
    Returns (start_date_iso, end_date_iso, groups_count)
    where end_date_iso is exclusive.
    """
    logger = get_run_logger()

    roots = _read_roots(roots_file)
    roots_n = len(roots)

    target = target_trade_date_utc or _default_target_trade_date_utc()
    target_dt = date.fromisoformat(target)

    max_date: Optional[str] = None
    try:
        con = duckdb.connect(database=":memory:")
        q = f"""
        SELECT MAX(trade_date_utc)::VARCHAR AS max_date
        FROM read_parquet('{DAILY_JOINED_GLOB}')
        """
        max_date = con.execute(q).fetchone()[0]
    except Exception as e:
        logger.warning("Could not query daily_joined max_date via DuckDB (%s). Assuming empty.", e)

    if max_date:
        start_dt = date.fromisoformat(max_date) + timedelta(days=1)
    else:
        start_dt = target_dt

    if start_dt > target_dt:
        logger.info("No missing daily data. Everything up to %s is already ingested.", target)
        # return a window consistent with downstream rebuild/validate if forced later
        return target, (target_dt + timedelta(days=1)).isoformat(), 0

    start = start_dt.isoformat()
    end = (target_dt + timedelta(days=1)).isoformat()  # exclusive end

    logger.info(
        "Need ingest for %d roots, grouped into 1 start-date buckets, target=%s",
        roots_n,
        target,
    )
    logger.info("Bucket start=%s roots=%d", start, roots_n)

    # Write a temp roots file (matches your prior behavior and supports bucketing later)
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="roots_", suffix=".txt") as tf:
        tf.write("\n".join(roots) + "\n")
        tmp_roots = tf.name

    cmd = [
        "python",
        str(REPO / "ingest_daily_all_roots.py"),
        "--dataset",
        dataset,
        "--start",
        start,
        "--end",
        end,
        "--roots-file",
        tmp_roots,
        "--force-window",
    ]
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)

    return start, end, 1


@task
def transcode() -> None:
    logger = get_run_logger()
    cmd = ["python", str(REPO / "transcode_to_parquet.py")]
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def normalize_daily() -> None:
    logger = get_run_logger()
    cmd = ["python", str(REPO / "normalize_daily.py")]
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def rebuild_continuous(start: str, end: str, roots_file: str, rules: str, overwrite: bool = False) -> None:
    logger = get_run_logger()
    cmd = [
        "python",
        str(REPO / "build_continuous_daily_all_roots.py"),
        "--start",
        start,
        "--end",
        end,
        "--roots-file",
        roots_file,
        "--write-rolls",
        "--rules",
        rules,
    ]
    if overwrite:
        cmd.append("--overwrite")

    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@task
def postrun_validate(roots_file: str, target_trade_date_utc: Optional[str]) -> None:
    logger = get_run_logger()
    target = target_trade_date_utc or _default_target_trade_date_utc()

    cmd = [
        "python",
        str(REPO / "validate_postrun_daily.py"),
        "--roots-file",
        roots_file,
        "--strict",
        "--target-trade-date-utc",
        target,
    ]
    logger.info("Running: %s", " ".join(cmd))
    _run(cmd, cwd=REPO)


@flow(name="daily_ingest_futures_watermark", task_runner=ThreadPoolTaskRunner(max_workers=1))
def daily_ingest_futures_watermark(
    roots_file: str = str(REPO / "roots.txt"),
    dataset: str = DEFAULT_DATASET,
    rules: str = DEFAULT_RULES,
    target_trade_date_utc: str | None = None,
) -> None:
    logger = get_run_logger()
    logger.info("dataset=%s roots_file=%s target_trade_date_utc=%s", dataset, roots_file, target_trade_date_utc)

    start, end, groups = ingest_missing_daily(roots_file, dataset, target_trade_date_utc)

    if groups == 0:
        logger.info("Nothing ingested.")
        if target_trade_date_utc:
            logger.info(
                "Nothing ingested, but target_trade_date_utc=%s provided; rebuilding continuous + validating anyway.",
                target_trade_date_utc,
            )
            rebuild_continuous(start, end, roots_file, rules, overwrite=True)
            postrun_validate(roots_file, target_trade_date_utc)
        else:
            logger.info("Nothing ingested; skipping transcode/normalize/rebuild/validate.")
        return

    # Strict sequential ordering (max_workers=1)
    transcode()
    normalize_daily()
    rebuild_continuous(start, end, roots_file, rules, overwrite=False)
    postrun_validate(roots_file, target_trade_date_utc)