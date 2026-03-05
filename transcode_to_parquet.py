#!/usr/bin/env python3
"""
transcode_to_parquet.py

Parallel, incremental DBN -> parquet transcoding.

- Finds DBN files under /data/lake/raw/databento/<DATASET>/
- Writes parquet under /data/lake/raw/databento_parquet/<DATASET>/ with same relative layout
- Skips files that already have a non-empty parquet output
- Uses a process pool to speed up transcoding

Env vars:
  TRANSCODE_DATASET   (default: GLBX.MDP3)
  TRANSCODE_JOBS      (default: os.cpu_count()//2)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

RAW_BASE = Path("/data/lake/raw/databento")
OUT_BASE = Path("/data/lake/raw/databento_parquet")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def _out_path(dataset: str, dbn_path: Path) -> Path:
    rel = dbn_path.relative_to(RAW_BASE / dataset)
    return (OUT_BASE / dataset / rel).with_suffix(".parquet")


def _already_done(out_path: Path) -> bool:
    try:
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def _transcode_one(args: Tuple[str, str]) -> Tuple[str, bool, Optional[str]]:
    """
    Worker: read DBN -> write parquet
    Returns (dbn_path, ok, err)
    """
    dataset, dbn_str = args
    dbn_path = Path(dbn_str)
    out_path = _out_path(dataset, dbn_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if _already_done(out_path):
            return (dbn_str, True, None)

        import databento as db  # local import for multiproc startup

        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()

        # Normalize to Polars
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        # Write parquet (zstd is a good tradeoff; you can tweak compression_level if desired)
        df.write_parquet(out_path, compression="zstd")
        return (dbn_str, True, None)

    except Exception as e:
        return (dbn_str, False, f"{type(e).__name__}: {e}")


def main() -> None:
    dataset = os.environ.get("TRANSCODE_DATASET", "GLBX.MDP3").strip()
    jobs = _env_int("TRANSCODE_JOBS", max(1, (os.cpu_count() or 4) // 2))

    raw_dir = RAW_BASE / dataset
    if not raw_dir.exists():
        raise SystemExit(f"raw dataset dir not found: {raw_dir}")

    dbn_files = sorted(raw_dir.rglob("*.dbn"))
    print(f"Found {len(dbn_files)} DBN files under {raw_dir}")
    print(f"Output base: {OUT_BASE/dataset}")
    print(f"Jobs: {jobs}")

    # Prepare work list with skip filtering (cheap)
    work = []
    skipped = 0
    for p in dbn_files:
        outp = _out_path(dataset, p)
        if _already_done(outp):
            skipped += 1
        else:
            work.append((dataset, str(p)))

    print(f"To transcode: {len(work)} (skipping {skipped} already done)")

    if not work:
        print("Nothing to do.")
        return

    ok = 0
    bad = 0

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_transcode_one, item) for item in work]
        for fut in as_completed(futs):
            path, success, err = fut.result()
            if success:
                ok += 1
            else:
                bad += 1
                print(f"[ERROR] {path}: {err}")

    print(f"Done. ok={ok} bad={bad} skipped={skipped}")
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()