#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import glob
import duckdb

DB_PATH = Path("/data/lake/duckdb/futures.duckdb")

VIEWS = {
    # research outputs
    "continuous_daily": "/data/lake/research/continuous_daily/root=*/year=*/month=*/part-*.parquet",
    "continuous_daily_rolls": "/data/lake/research/continuous_daily_rolls/root=*/year=*/month=*/part-*.parquet",

    # normalized daily (curated_normalized)
    "daily_joined": "/data/lake/curated_normalized/daily_joined/root=*/year=*/month=*/part-*.parquet",
    "daily_bars_clean": "/data/lake/curated_normalized/daily_bars_clean/root=*/year=*/month=*/part-*.parquet",
    "daily_stats_pivoted": "/data/lake/curated_normalized/daily_stats_pivoted/root=*/year=*/month=*/part-*.parquet",

    # intraday normalized (if present)
    "bars_1m_clean": "/data/lake/curated_normalized_intraday/bars_1m_clean/root=*/year=*/month=*/part-*.parquet",
    "bars_2m": "/data/lake/curated_normalized_intraday/bars_2m/root=*/year=*/month=*/part-*.parquet",
    "bars_5m": "/data/lake/curated_normalized_intraday/bars_5m/root=*/year=*/month=*/part-*.parquet",
    "bars_15m": "/data/lake/curated_normalized_intraday/bars_15m/root=*/year=*/month=*/part-*.parquet",
    "bars_30m": "/data/lake/curated_normalized_intraday/bars_30m/root=*/year=*/month=*/part-*.parquet",
    "bars_1h": "/data/lake/curated_normalized_intraday/bars_1h/root=*/year=*/month=*/part-*.parquet",
}

DERIVED_VIEW_NAME = "continuous_daily_with_roll_flag"
DERIVED_VIEW_SQL = """
CREATE OR REPLACE VIEW continuous_daily_with_roll_flag AS
SELECT
  d.*,
  CASE WHEN r.roll_date IS NULL THEN FALSE ELSE TRUE END AS is_roll_day
FROM continuous_daily d
LEFT JOIN continuous_daily_rolls r
  ON d.root = r.root AND d.trade_date_utc = r.roll_date
"""


def parquet_exists(pattern: str) -> bool:
    # quick existence check: does the glob match at least one parquet file?
    return len(glob.glob(pattern)) > 0


def view_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    try:
        con.execute("INSTALL parquet;")
        con.execute("LOAD parquet;")

        created = 0
        skipped = 0

        for name, pattern in VIEWS.items():
            if not parquet_exists(pattern):
                print(f"[skip] {name}: no parquet files match: {pattern}")
                skipped += 1
                continue

            sql = f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT *
            FROM read_parquet('{pattern}', hive_partitioning=1)
            """
            con.execute(sql)
            print(f"[ok]   view: {name}")
            created += 1

        # Create derived view only if its inputs exist
        if view_exists(con, "continuous_daily") and view_exists(con, "continuous_daily_rolls"):
            con.execute(DERIVED_VIEW_SQL.strip())
            print(f"[ok]   view: {DERIVED_VIEW_NAME}")
        else:
            print(f"[skip] {DERIVED_VIEW_NAME}: requires continuous_daily + continuous_daily_rolls")

        print(f"\nDone. db={DB_PATH} created={created} skipped={skipped}")

    finally:
        con.close()


if __name__ == "__main__":
    main()