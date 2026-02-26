from __future__ import annotations

from pathlib import Path
import polars as pl

ROOT = "ES"
YEAR = "2015"
MONTH = "01"

BARS_IN = f"/data/lake/curated/futures_ohlcv_1d/root={ROOT}/year={YEAR}/month={MONTH}/*.parquet"
STATS_IN = f"/data/lake/curated/futures_statistics/root={ROOT}/year={YEAR}/month={MONTH}/*.parquet"

OUT_ROOT = Path("/data/lake/curated_normalized")
BARS_OUT = OUT_ROOT / "daily_bars_clean" / f"root={ROOT}" / f"year={YEAR}" / f"month={MONTH}" / "part-0000.parquet"
STATS_OUT = OUT_ROOT / "daily_stats_pivoted" / f"root={ROOT}" / f"year={YEAR}" / f"month={MONTH}" / "part-0000.parquet"
JOIN_OUT = OUT_ROOT / "daily_joined" / f"root={ROOT}" / f"year={YEAR}" / f"month={MONTH}" / "part-0000.parquet"

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    # --- Bars: sort + dedupe deterministically ---
    bars = (
        pl.scan_parquet(BARS_IN)
        .sort(["instrument_id", "ts_event"])
        .with_columns(pl.col("ts_event").dt.date().alias("trade_date_utc"))
        .unique(subset=["instrument_id", "ts_event"], keep="last")
        .select([
            "instrument_id", "symbol", "trade_date_utc", "ts_event",
            "open", "high", "low", "close", "volume",
        ])
        .collect()
    )

    ensure_parent(BARS_OUT)
    bars.write_parquet(BARS_OUT, compression="zstd")
    print("Wrote:", str(BARS_OUT), "rows=", bars.height)

    # --- Stats: filter + keep latest stat per day/type ---
    stats = (
        pl.scan_parquet(STATS_IN)
        .filter(pl.col("stat_type").is_in([3, 6, 9]))  # settlement, cleared volume, open interest
        .with_columns(pl.col("ts_event").dt.date().alias("trade_date_utc"))
        .sort(["instrument_id", "trade_date_utc", "stat_type", "ts_event"])
        .unique(subset=["instrument_id", "trade_date_utc", "stat_type"], keep="last")
        .with_columns([
            pl.when(pl.col("stat_type") == 3).then(pl.col("price")).otherwise(None).alias("settlement_price"),
            pl.when(pl.col("stat_type") == 9).then(pl.col("quantity")).otherwise(None).alias("open_interest"),
            pl.when(pl.col("stat_type") == 6).then(pl.col("quantity")).otherwise(None).alias("cleared_volume"),
        ])
        .group_by(["instrument_id", "symbol", "trade_date_utc"])
        .agg([
            pl.max("settlement_price").alias("settlement_price"),
            pl.max("open_interest").alias("open_interest"),
            pl.max("cleared_volume").alias("cleared_volume"),
        ])
        .collect()
    )

    ensure_parent(STATS_OUT)
    stats.write_parquet(STATS_OUT, compression="zstd")
    print("Wrote:", str(STATS_OUT), "rows=", stats.height)

    # --- Join ---
    joined = bars.join(stats, on=["instrument_id", "symbol", "trade_date_utc"], how="left")
    ensure_parent(JOIN_OUT)
    joined.write_parquet(JOIN_OUT, compression="zstd")
    print("Wrote:", str(JOIN_OUT), "rows=", joined.height)

if __name__ == "__main__":
    main()