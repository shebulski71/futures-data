from pathlib import Path
import polars as pl

ROOT = "ES"
YEAR = "2015"
MONTH = "01"

IN_PATH = f"/data/lake/curated/futures_ohlcv_1m/root={ROOT}/year={YEAR}/month={MONTH}/*.parquet"

OUT_BASE = Path("/data/lake/curated_normalized_intraday")

OUT_1M = OUT_BASE / "bars_1m_clean" / f"root={ROOT}" / f"year={YEAR}" / f"month={MONTH}" / "part-0000.parquet"

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():

    df = (
        pl.scan_parquet(IN_PATH)

        # deterministic ordering
        .sort(["instrument_id", "ts_event"])

        # dedupe
        .unique(subset=["instrument_id", "ts_event"], keep="last")

        # timezone conversion
        .with_columns([
            pl.col("ts_event")
            .dt.convert_time_zone("America/Chicago")
            .alias("ts_event_ct")
        ])

        # CME ETH trading day:
        # if time >= 17:00 CT → next trading day
        .with_columns([
            pl.when(pl.col("ts_event_ct").dt.hour() >= 17)
            .then((pl.col("ts_event_ct") + pl.duration(days=1)).dt.date())
            .otherwise(pl.col("ts_event_ct").dt.date())
            .alias("trade_date_eth")
        ])

        .select([
            "instrument_id",
            "symbol",
            "ts_event",
            "ts_event_ct",
            "trade_date_eth",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ])

        .collect()
    )

    ensure_parent(OUT_1M)
    df.write_parquet(OUT_1M, compression="zstd")

    print("Wrote:", OUT_1M, "rows=", df.height)

if __name__ == "__main__":
    main()