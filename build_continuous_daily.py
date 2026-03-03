from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
from dateutil.relativedelta import relativedelta


SRC_BASE = Path("/data/lake/curated_normalized/daily_joined")
OUT_BASE = Path("/data/lake/research/continuous_daily")


MONTH_CODE_TO_NUM = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def month_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    cur = date(start.year, start.month, 1)
    out = []
    while cur < end:
        nxt = cur + relativedelta(months=1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def parse_month_code(symbol: str) -> Optional[int]:
    # Best effort: last 2 chars = month code + year digit, e.g. ESH5, CLZ4, MCLZ1
    # Some symbols may have two-digit years; we only need month code for ordering fallback.
    if not symbol or len(symbol) < 2:
        return None
    mc = symbol[-2]
    return MONTH_CODE_TO_NUM.get(mc)


def build_continuous_for_month(root: str, month_start: date) -> Optional[pl.DataFrame]:
    y = month_start.year
    m = month_start.month
    src_dir = SRC_BASE / f"root={root}" / f"year={y:04d}" / f"month={m:02d}"

    if not src_dir.exists():
        return None

    files = list(src_dir.glob("*.parquet"))
    if not files:
        return None

    df = pl.read_parquet(str(src_dir / "*.parquet"))

    if df.is_empty():
        return None

    # Add month_num for fallback ordering
    # (Polars UDF via map_elements is fine here; monthly partitions are small-ish)
    df = df.with_columns(
        pl.col("symbol")
        .map_elements(parse_month_code, return_dtype=pl.Int32)
        .alias("month_num")
    )

    # Selection rule:
    # 1) Prefer rows with non-null open_interest (oi_present desc)
    # 2) Highest open_interest
    # 3) Highest volume
    # 4) Lowest month_num (nearest contract) as a fallback
    # 5) Stable tie-breaker: symbol
    df = df.with_columns(
        pl.col("open_interest").is_not_null().cast(pl.Int8).alias("oi_present")
    )

    # One "best" row per trade_date_utc
    picked = (
        df.sort(
            by=["trade_date_utc", "oi_present", "open_interest", "volume", "month_num", "symbol"],
            descending=[False, True, True, True, False, False],
        )
        .group_by("trade_date_utc")
        .agg(pl.all().first())
        .sort("trade_date_utc")
    )

    # Flatten struct columns if any (group_by+agg can create structs depending on polars version)
    # This keeps it robust across versions.
    if any(dt.startswith("struct") for dt in [str(t) for t in picked.dtypes]):
        picked = picked.unnest(picked.columns)

    # Keep a clean continuous schema
    out = picked.select(
        [
            "trade_date_utc",
            "symbol",
            "instrument_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "settlement_price",
            "open_interest",
            "cleared_volume",
        ]
    )

    return out


def write_month(root: str, month_start: date, df: pl.DataFrame, overwrite: bool = True) -> Path:
    out_dir = OUT_BASE / f"root={root}" / f"year={month_start.year:04d}" / f"month={month_start.month:02d}"
    ensure_dir(out_dir)
    out_path = out_dir / "part-0000.parquet"

    if out_path.exists() and not overwrite:
        return out_path

    df.write_parquet(out_path, compression="zstd")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build continuous daily series from daily_joined using OI->Volume selection.")
    ap.add_argument("--root", required=True, help="Root symbol, e.g. ES")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive-ish)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquet")
    args = ap.parse_args()

    root = args.root
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    wrote = 0
    skipped = 0

    for m_start, _m_end in month_ranges(start, end):
        out_path = OUT_BASE / f"root={root}" / f"year={m_start.year:04d}" / f"month={m_start.month:02d}" / "part-0000.parquet"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        df = build_continuous_for_month(root, m_start)
        if df is None or df.is_empty():
            skipped += 1
            continue

        p = write_month(root, m_start, df, overwrite=args.overwrite)
        wrote += 1
        print(f"Wrote {root} {m_start:%Y-%m}: {p} rows={df.height}")

    print(f"Done. wrote={wrote} skipped={skipped}")


if __name__ == "__main__":
    main()