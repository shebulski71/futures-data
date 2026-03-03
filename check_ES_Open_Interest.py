import polars as pl

p = "/data/lake/curated_normalized/daily_joined/root=ES/year=2015/month=01/*.parquet"

df = pl.read_parquet(p)

print(
    df.select([
        pl.count().alias("rows"),
        pl.col("open_interest").null_count().alias("oi_nulls"),
        pl.col("open_interest").min().alias("oi_min"),
        pl.col("open_interest").max().alias("oi_max"),
    ])
)