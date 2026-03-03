import polars as pl

bars_path = "/data/lake/curated/futures_ohlcv_1d/root=ES/year=2015/month=01/*.parquet"
stats_path = "/data/lake/curated/futures_statistics/root=ES/year=2015/month=01/*.parquet"

bars = pl.scan_parquet(bars_path)
stats = pl.scan_parquet(stats_path)

print("\n=== BARS columns ===")
print(bars.columns)

print("\n=== BARS schema ===")
print(bars.schema)

print("\n=== BARS sample (sorted) ===")
bars_sample = (
    bars
    .select(pl.all())
    .sort([c for c in ["symbol", "ts_event", "date"] if c in bars.columns])
    .limit(12)
    .collect()
)
print(bars_sample)

print("\n=== STATS columns ===")
print(stats.columns)

print("\n=== STATS schema ===")
print(stats.schema)

print("\n=== STATS sample (sorted) ===")
stats_sample = (
    stats
    .select(pl.all())
    .sort([c for c in ["symbol", "ts_event", "date"] if c in stats.columns])
    .limit(12)
    .collect()
)
print(stats_sample)