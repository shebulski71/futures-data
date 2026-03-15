#!/usr/bin/env python3
"""
query_futures_data.py (V3)

Convenience loader for the futures research lake.

Supported datasets
------------------
continuous : continuous 1m futures bars
features   : 1m feature dataset
resampled  : higher timeframe bars (5m, 15m, 30m, 1h)

Main functions
--------------
load_bars(...)
load_close_matrix(...)
load_bars_with_features(...)

Examples
--------
from query_futures_data import load_bars, load_close_matrix, load_bars_with_features

df = load_bars(
    roots=["ES", "NQ"],
    freq="5m",
    start="2026-01-01",
    end="2026-02-01",
    dataset="resampled",
    session="rth",
)

joined = load_bars_with_features(
    roots="MES",
    start="2026-01-01",
    end="2026-02-01",
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, Union
import datetime as dt

import polars as pl


CONTINUOUS_BASE = Path("/data/lake/research/continuous_1m")
FEATURES_BASE = Path("/data/lake/research/features_1m")
RESAMPLED_BASE = Path("/data/lake/research/resampled_bars")

SUPPORTED_RESAMPLED_FREQS = {"5m", "15m", "30m", "1h"}
SUPPORTED_DATASETS = {"continuous", "features", "resampled"}
SUPPORTED_SESSIONS = {"all", "rth", "eth"}


def _to_utc_timestamp(value: Union[str, dt.date, dt.datetime]) -> dt.datetime:
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.timezone.utc)

    s = str(value).strip()

    if len(s) <= 10:
        return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)

    parsed = dt.datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _month_iter(start: dt.datetime, end: dt.datetime) -> Iterable[tuple[int, int]]:
    cur = dt.date(start.year, start.month, 1)
    end_month = dt.date(end.year, end.month, 1)

    while cur <= end_month:
        yield cur.year, cur.month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)


def _dataset_base(dataset: str, freq: str) -> Path:
    dataset = dataset.lower().strip()

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(SUPPORTED_DATASETS)}")

    if dataset == "continuous":
        if freq != "1m":
            raise ValueError("dataset='continuous' only supports freq='1m'")
        return CONTINUOUS_BASE

    if dataset == "features":
        if freq != "1m":
            raise ValueError("dataset='features' only supports freq='1m'")
        return FEATURES_BASE

    if dataset == "resampled":
        if freq not in SUPPORTED_RESAMPLED_FREQS:
            raise ValueError(
                f"dataset='resampled' supports only {sorted(SUPPORTED_RESAMPLED_FREQS)}"
            )
        return RESAMPLED_BASE / freq

    raise RuntimeError("unexpected dataset selection")


def _normalize_roots(roots: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(roots, str):
        roots = [roots]

    out: list[str] = []
    for r in roots:
        rr = str(r).strip()
        if rr:
            out.append(rr)

    if not out:
        raise ValueError("at least one root is required")

    return sorted(set(out))


def _candidate_files(
    base: Path,
    roots: Sequence[str],
    start: dt.datetime,
    end: dt.datetime,
) -> list[Path]:
    files: list[Path] = []

    for root in roots:
        for year, month in _month_iter(start, end):
            p = (
                base
                / f"root={root}"
                / f"year={year:04d}"
                / f"month={month:02d}"
                / "part-0000.parquet"
            )
            if p.exists():
                files.append(p)

    return sorted(files)


def available_roots(dataset: str = "resampled", freq: str = "5m") -> list[str]:
    base = _dataset_base(dataset, freq)
    if not base.exists():
        return []

    out = []
    for p in sorted(base.glob("root=*")):
        out.append(p.name.split("=", 1)[1])
    return out


def _apply_session_filter(lf: pl.LazyFrame, session: str) -> pl.LazyFrame:
    """
    Session filter in UTC.
    RTH approximation for CME equity-style session:
      14:30 <= time < 21:00 UTC
    ETH = everything else

    This is intentionally simple. For asset-specific session calendars, build a
    separate calendar layer later.
    """
    session = session.lower().strip()
    if session not in SUPPORTED_SESSIONS:
        raise ValueError(f"session must be one of {sorted(SUPPORTED_SESSIONS)}")

    if session == "all":
        return lf

    minute_of_day = (
        pl.col("ts_event").dt.hour() * 60
        + pl.col("ts_event").dt.minute()
    )

    rth_expr = (minute_of_day >= 14 * 60 + 30) & (minute_of_day < 21 * 60)

    if session == "rth":
        return lf.filter(rth_expr)

    if session == "eth":
        return lf.filter(~rth_expr)

    return lf


def load_bars(
    roots: Union[str, Sequence[str]],
    freq: str,
    start: Union[str, dt.date, dt.datetime],
    end: Union[str, dt.date, dt.datetime],
    dataset: str = "resampled",
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[pl.Expr]] = None,
    session: str = "all",
    as_pandas: bool = False,
) -> Union[pl.DataFrame, "pandas.DataFrame"]:
    roots_list = _normalize_roots(roots)
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)

    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    base = _dataset_base(dataset, freq)
    files = _candidate_files(base, roots_list, start_ts, end_ts)

    if not files:
        raise FileNotFoundError(
            f"No files found for roots={roots_list}, dataset={dataset}, freq={freq}, "
            f"range=[{start_ts.isoformat()}, {end_ts.isoformat()})"
        )

    scans = [pl.scan_parquet(str(f)) for f in files]
    lf = pl.concat(scans, how="vertical_relaxed")

    lf = lf.filter(
        (pl.col("ts_event") >= pl.lit(start_ts))
        & (pl.col("ts_event") < pl.lit(end_ts))
    )

    lf = _apply_session_filter(lf, session=session)

    if filters:
        for expr in filters:
            lf = lf.filter(expr)

    schema_names = lf.collect_schema().names()

    sort_cols = ["ts_event"]
    if "root" in schema_names:
        sort_cols = ["root", "ts_event"]

    lf = lf.sort(sort_cols)

    if columns:
        missing = [c for c in columns if c not in schema_names]
        if missing:
            raise ValueError(f"Requested columns not present: {missing}")
        lf = lf.select(list(columns))

    df = lf.collect()

    if as_pandas:
        pdf = df.to_pandas()
        if "ts_event" in pdf.columns:
            pdf = pdf.set_index("ts_event")
        return pdf

    return df


def load_close_matrix(
    roots: Union[str, Sequence[str]],
    freq: str,
    start: Union[str, dt.date, dt.datetime],
    end: Union[str, dt.date, dt.datetime],
    dataset: str = "resampled",
    session: str = "all",
    as_pandas: bool = False,
) -> Union[pl.DataFrame, "pandas.DataFrame"]:
    df = load_bars(
        roots=roots,
        freq=freq,
        start=start,
        end=end,
        dataset=dataset,
        columns=["ts_event", "root", "close"],
        session=session,
        as_pandas=False,
    )

    wide = (
        df.pivot(index="ts_event", on="root", values="close", aggregate_function="last")
        .sort("ts_event")
    )

    if as_pandas:
        pdf = wide.to_pandas().set_index("ts_event")
        return pdf

    return wide


def load_bars_with_features(
    roots: Union[str, Sequence[str]],
    start: Union[str, dt.date, dt.datetime],
    end: Union[str, dt.date, dt.datetime],
    feature_columns: Optional[Sequence[str]] = None,
    session: str = "all",
    as_pandas: bool = False,
) -> Union[pl.DataFrame, "pandas.DataFrame"]:
    """
    Join continuous 1m bars with 1m features on [root, ts_event].
    """
    bars = load_bars(
        roots=roots,
        freq="1m",
        start=start,
        end=end,
        dataset="continuous",
        session=session,
        as_pandas=False,
    )

    feats = load_bars(
        roots=roots,
        freq="1m",
        start=start,
        end=end,
        dataset="features",
        session=session,
        as_pandas=False,
    )

    if feature_columns:
        keep = ["root", "ts_event"] + list(feature_columns)
        missing = [c for c in keep if c not in feats.columns]
        if missing:
            raise ValueError(f"Requested feature columns not present: {missing}")
        feats = feats.select(keep)

    out = bars.join(feats, on=["root", "ts_event"], how="left").sort(["root", "ts_event"])

    if as_pandas:
        pdf = out.to_pandas()
        if "ts_event" in pdf.columns:
            pdf = pdf.set_index("ts_event")
        return pdf

    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Query futures research parquet layers.")
    ap.add_argument("--roots", required=True, help="Comma-separated roots, e.g. ES,NQ,CL")
    ap.add_argument("--freq", required=False, default="5m")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument(
        "--dataset",
        default="resampled",
        choices=["continuous", "features", "resampled"],
    )
    ap.add_argument(
        "--session",
        default="all",
        choices=["all", "rth", "eth"],
    )
    ap.add_argument(
        "--close-matrix",
        action="store_true",
        help="Return wide close matrix instead of long-form bars",
    )
    args = ap.parse_args()

    roots = [x.strip() for x in args.roots.split(",") if x.strip()]

    if args.close_matrix:
        df = load_close_matrix(
            roots=roots,
            freq=args.freq,
            start=args.start,
            end=args.end,
            dataset=args.dataset,
            session=args.session,
            as_pandas=False,
        )
    else:
        df = load_bars(
            roots=roots,
            freq=args.freq,
            start=args.start,
            end=args.end,
            dataset=args.dataset,
            session=args.session,
            as_pandas=False,
        )

    print(df.head(10))
    print(f"rows={df.height}")