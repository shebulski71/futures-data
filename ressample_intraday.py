from __future__ import annotations

from pathlib import Path
import polars as pl

ROOT = "ES"
YEAR = "2015"
MONTH = "01"

IN_1M = f"/data/lake/curated_normalized_intraday/bars_1m_clean/root={ROOT}/year={YEAR}/month={MONTH}/part-0000.parquet"

OUT_BASE = Path("/data/lake/curated_normalized_intraday")

TIMEFRAMES = [
    ("2m", "2m"),
    ("5m", "5m"),
    ("15m", "15m"),
    ("30m", "30m"),
    ("1h", "1h"),
]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ohlcv_agg():
    return [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

def main():
    df = pl.scan_parquet(IN_1M)

    # IMPORTANT:
    # We resample using Chicago timestamps so windows align to ETH clock.
    # Partition by trade_date_eth so we don't "cross" the 17:00 boundary.
    for name, every in TIMEFRAMES:
        out_path = OUT_BASE / f"bars_{name}" / f"root={ROOT}" / f"year={YEAR}" / f"month={MONTH}" / "part-0000.parquet"
        ensure_dir(out_path.parent)

        resampled = (
            df
            .sort(["instrument_id", "ts_event_ct"])
            .group_by_dynamic(
                index_column="ts_event_ct",
                every=every,
                period=every,
                group_by=["instrument_id", "symbol", "trade_date_eth"],
                closed="left",
                label="left",
            )
            .agg(ohlcv_agg())
            # rename the bucket start to a consistent timestamp column
            .rename({"ts_event_ct": "bar_start_ct"})
            .with_columns([
                pl.col("bar_start_ct").dt.convert_time_zone("UTC").alias("bar_start_utc")
            ])
            .select([
                "instrument_id", "symbol", "trade_date_eth",
                "bar_start_ct", "bar_start_utc",
                "open", "high", "low", "close", "volume"
            ])
            .collect()
        )

        resampled.write_parquet(out_path, compression="zstd")
        print(f"Wrote {name}: {out_path} rows={resampled.height}")

if __name__ == "__main__":
    main()