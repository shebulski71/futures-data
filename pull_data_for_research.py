from pathlib import Path
import polars as pl
import pandas as pd

base = Path("/data/lake/research/continuous_1m/root=ES")

# Pull only the monthly partitions you need
files = [
    base / "year=2026" / "month=01" / "part-0000.parquet",
    base / "year=2026" / "month=02" / "part-0000.parquet",
]
files = [f for f in files if f.exists()]

df = (
    pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    .filter(
        (pl.col("ts_event") >= pl.lit(pd.Timestamp("2026-01-01", tz="UTC"))) &
        (pl.col("ts_event") <  pl.lit(pd.Timestamp("2026-02-01", tz="UTC")))
    )
    .sort("ts_event")
)

pdf = df.to_pandas()
pdf = pdf.set_index("ts_event")

bars_5m = (
    pdf.resample("5min", label="right", closed="right")
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "symbol": "last",
        "instrument_id": "last",
        "root": "last",
        "trade_date_utc": "last",
    })
    .dropna(subset=["open", "high", "low", "close"])
)
print(bars_5m)
print(bars_5m.head())
print(bars_5m.tail())