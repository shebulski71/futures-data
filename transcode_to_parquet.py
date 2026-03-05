#!/usr/bin/env python3
"""
transcode_to_parquet.py

Parallel, incremental DBN -> parquet transcoding (IN-PLACE).

- Finds DBN files under /data/lake/raw/databento/<DATASET>/
- Writes parquet next to each DBN with same basename:
    .../foo.dbn -> .../foo.parquet
- Skips files that already have a non-empty parquet output
- Uses a process pool to speed up transcoding

Env vars:
  TRANSCODE_DATASET   (default: GLBX.MDP3)
  TRANSCODE_JOBS      (default: max(1, os.cpu_count()//2))
  TRANSCODE_FORCE     (default: 0)  # if 1, overwrite existing parquet
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

RAW_BASE = Path("/data/lake/raw/databento")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _out_path_inplace(dbn_path: Path) -> Path:
    return dbn_path.with_suffix(".parquet")


def _already_done(out_path: Path) -> bool:
    try:
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _transcode_one(args: Tuple[str, str, bool]) -> Tuple[str, bool, Optional[str]]:
    """
    Worker: read DBN -> write parquet
    Returns (dbn_path, ok, err)
    """
    dataset, dbn_str, force = args
    dbn_path = Path(dbn_str)
    out_path = _out_path_inplace(dbn_path)

    try:
        if (not force) and _already_done(out_path):
            return (dbn_str, True, None)

        # Import inside worker for faster parent startup
        import databento as db

        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()

        # Normalize to Polars
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        # Write parquet (zstd is a good tradeoff)
        df.write_parquet(out_path, compression="zstd")
        return (dbn_str, True, None)

    except Exception as e:
        return (dbn_str, False, f"{type(e).__name__}: {e}")


def main() -> None:
    dataset = (os.environ.get("TRANSCODE_DATASET", "GLBX.MDP3") or "").strip()
    jobs = _env_int("TRANSCODE_JOBS", max(