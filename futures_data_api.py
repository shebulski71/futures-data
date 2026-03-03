#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import duckdb
import polars as pl

Timeframe = Literal["1m", "2m", "5m", "15m", "30m", "1h"]


def _quote_list(vals: Iterable[str]) -> str:
    return ", ".join([f"'{v}'" for v in vals])


@dataclass(frozen=True)
class FuturesLake:
    """Small DuckDB-backed data access layer returning Polars DataFrames."""
    db_path: str = "/data/lake/duckdb/futures.duckdb"
    read_only: bool = True

    def _con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path, read_only=self.read_only)

    def query(self, sql: str) -> pl.DataFrame:
        con = self._con()
        try:
            rel = con.sql(sql)
            return pl.from_arrow(rel.arrow())
        finally:
            con.close()

    # ---------- Continuous Daily ----------

    def get_continuous_daily(
        self,
        root: str,
        start: str,
        end: str,
        cols: Optional[list[str]] = None,
        with_roll_flag: bool = True,
    ) -> pl.DataFrame:
        view = "continuous_daily_with_roll_flag" if with_roll_flag else "continuous_daily"
        if cols is None:
            cols = [
                "trade_date_utc", "root", "symbol", "instrument_id",
                "open", "high", "low", "close", "volume",
                "settlement_price", "open_interest", "cleared_volume",
            ] + (["is_roll_day"] if with_roll_flag else [])

        sel = ", ".join(cols)
        sql = f"""
        SELECT {sel}
        FROM {view}
        WHERE root = '{root}'
          AND trade_date_utc >= '{start}'
          AND trade_date_utc < '{end}'
        ORDER BY trade_date_utc
        """
        return self.query(sql)

    def get_continuous_daily_many(
        self,
        roots: list[str],
        start: str,
        end: str,
        with_roll_flag: bool = True,
    ) -> pl.DataFrame:
        view = "continuous_daily_with_roll_flag" if with_roll_flag else "continuous_daily"
        roots_in = _quote_list(roots)
        sql = f"""
        SELECT *
        FROM {view}
        WHERE root IN ({roots_in})
          AND trade_date_utc >= '{start}'
          AND trade_date_utc < '{end}'
        ORDER BY root, trade_date_utc
        """
        return self.query(sql)

    def get_rolls(
        self,
        root: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        sql = f"""
        SELECT *
        FROM continuous_daily_rolls
        WHERE root = '{root}'
          AND roll_date >= '{start}'
          AND roll_date < '{end}'
        ORDER BY roll_date
        """
        return self.query(sql)

    def get_close_series_wide(
        self,
        roots: list[str],
        start: str,
        end: str,
        col: str = "close",
    ) -> pl.DataFrame:
        """
        Returns wide dataframe: trade_date_utc + one column per root.
        Perfect input for vectorbt.
        """
        roots_in = _quote_list(roots)
        sql = f"""
        SELECT trade_date_utc, root, {col}
        FROM continuous_daily
        WHERE root IN ({roots_in})
          AND trade_date_utc >= '{start}'
          AND trade_date_utc < '{end}'
        """
        df = self.query(sql)
        return (
            df.pivot(
                index="trade_date_utc",
                columns="root",
                values=col,
                aggregate_function="first",
            )
            .sort("trade_date_utc")
        )

    # ---------- Intraday ----------

    def get_intraday(
        self,
        root: str,
        start_ts_ct: str,
        end_ts_ct: str,
        timeframe: Timeframe = "1m",
    ) -> pl.DataFrame:
        view_map = {
            "1m": "bars_1m_clean",
            "2m": "bars_2m",
            "5m": "bars_5m",
            "15m": "bars_15m",
            "30m": "bars_30m",
            "1h": "bars_1h",
        }
        view = view_map[timeframe]
        sql = f"""
        SELECT *
        FROM {view}
        WHERE root = '{root}'
          AND ts_event_ct >= '{start_ts_ct}'
          AND ts_event_ct < '{end_ts_ct}'
        ORDER BY ts_event_ct
        """
        return self.query(sql)