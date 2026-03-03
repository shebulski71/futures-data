#!/usr/bin/env python3
"""
build_continuous_daily_all_roots.py

Build continuous daily series for futures roots from:
  /data/lake/curated_normalized/daily_joined

Algorithm per root/month:
  1) Read full month partition (all contracts for that root)
  2) For each trade_date_utc, pick a best *candidate* contract row using:
       - prefer non-null open_interest
       - highest open_interest
       - highest volume
       - nearest month code (fallback)
       - symbol (stable tie-break)
  3) Stabilize (debounce) the candidate symbol sequence:
       - only switch to a new candidate if it remains candidate for N consecutive trading days (confirm_days)
       - optional minimum hold days after a switch (min_hold_days)
  4) Re-select the OHLCV row for the stabilized symbol on each date (true continuous series)

Outputs:
  Continuous daily:
    /data/lake/research/continuous_daily/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

  Roll logs (optional, partitioned by month):
    /data/lake/research/continuous_daily_rolls/root=<ROOT>/year=YYYY/month=MM/part-0000.parquet

Progress (resume):
  /data/lake/state/continuous_daily_progress.json

Per-root parameterization:
  --rules "CL=4:10,NG=3:7,default=3:0"
Meaning:
  confirm_days:min_hold_days
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from dateutil.relativedelta import relativedelta

SRC_BASE = Path("/data/lake/curated_normalized/daily_joined")
OUT_BASE = Path("/data/lake/research/continuous_daily")
ROLL_BASE = Path("/data/lake/research/continuous_daily_rolls")
PROGRESS_PATH = Path("/data/lake/state/continuous_daily_progress.json")

MONTH_CODE_TO_NUM = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_roots(path: Path) -> List[str]:
    roots: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        roots.append(s)
    return roots


def month_ranges(start: date, end: date) -> List[Tuple[date, date]]:
    cur = date(start.year, start.month, 1)
    out: List[Tuple[date, date]] = []
    while cur < end:
        nxt = cur + relativedelta(months=1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def load_progress() -> Dict:
    if PROGRESS_PATH.exists():
        try:
            data = json.loads(PROGRESS_PATH.read_text())
            data.setdefault("done", {})
            return data
        except Exception:
            pass
    return {"done": {}}


def save_progress_atomic(data: Dict) -> None:
    ensure_dir(PROGRESS_PATH.parent)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(PROGRESS_PATH)


def is_done(progress: Dict, root: str, month: str) -> bool:
    return bool(progress.get("done", {}).get(root, {}).get(month, False))


def mark_done(progress: Dict, root: str, month: str) -> None:
    progress.setdefault("done", {}).setdefault(root, {})
    progress["done"][root][month] = True


def parse_month_code(symbol: str) -> Optional[int]:
    if not symbol or len(symbol) < 2:
        return None
    return MONTH_CODE_TO_NUM.get(symbol[-2])


def out_path(root: str, month_start: date) -> Path:
    return (
        OUT_BASE / f"root={root}"
        / f"year={month_start.year:04d}"
        / f"month={month_start.month:02d}"
        / "part-0000.parquet"
    )


def roll_path(root: str, month_start: date) -> Path:
    return (
        ROLL_BASE / f"root={root}"
        / f"year={month_start.year:04d}"
        / f"month={month_start.month:02d}"
        / "part-0000.parquet"
    )


def parse_rules(rules_str: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse --rules string like:
      "CL=4:10,NG=3:7,default=3:0"

    Returns dict root->(confirm_days, min_hold_days) including optional key "default".
    """
    rules: Dict[str, Tuple[int, int]] = {}
    if not rules_str:
        return rules

    parts = [p.strip() for p in rules_str.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad rules entry (missing '='): {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if ":" not in v:
            raise ValueError(f"Bad rules value (expected confirm:minhold): {p}")
        a, b = v.split(":", 1)
        confirm = int(a)
        hold = int(b)
        if confirm < 1:
            raise ValueError(f"confirm_days must be >=1 in: {p}")
        if hold < 0:
            raise ValueError(f"min_hold_days must be >=0 in: {p}")
        rules[k] = (confirm, hold)
    return rules


def get_rule_for_root(rules: Dict[str, Tuple[int, int]], root: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if root in rules:
        return rules[root]
    if "default" in rules:
        return rules["default"]
    return fallback


def stabilize_symbols(
    trade_dates: List[date],
    candidate_symbols: List[str],
    candidate_instrument_ids: List[int],
    prev_symbol: Optional[str],
    prev_instrument_id: Optional[int],
    confirm_days: int,
    min_hold_days: int,
) -> Tuple[List[str], List[int]]:
    n = len(trade_dates)
    if n == 0:
        return [], []

    cur_sym = prev_symbol if prev_symbol is not None else candidate_symbols[0]
    cur_iid = int(prev_instrument_id) if prev_instrument_id is not None else int(candidate_instrument_ids[0])

    pending_sym: Optional[str] = None
    pending_iid: Optional[int] = None
    pending_count = 0

    hold_count = 0  # days since last switch

    stable_sym: List[str] = []
    stable_iid: List[int] = []

    for i in range(n):
        cand_sym = candidate_symbols[i]
        cand_iid = int(candidate_instrument_ids[i])

        if cand_sym == cur_sym:
            pending_sym = None
            pending_iid = None
            pending_count = 0
            hold_count += 1
            stable_sym.append(cur_sym)
            stable_iid.append(cur_iid)
            continue

        if min_hold_days and hold_count < min_hold_days:
            hold_count += 1
            stable_sym.append(cur_sym)
            stable_iid.append(cur_iid)
            continue

        if pending_sym == cand_sym:
            pending_count += 1
        else:
            pending_sym = cand_sym
            pending_iid = cand_iid
            pending_count = 1

        if pending_count >= confirm_days:
            cur_sym = pending_sym
            cur_iid = int(pending_iid) if pending_iid is not None else cur_iid
            pending_sym = None
            pending_iid = None
            pending_count = 0
            hold_count = 1
        else:
            hold_count += 1

        stable_sym.append(cur_sym)
        stable_iid.append(cur_iid)

    return stable_sym, stable_iid


def build_continuous_for_month(
    root: str,
    month_start: date,
    prev_symbol: Optional[str],
    prev_instrument_id: Optional[int],
    confirm_days: int,
    min_hold_days: int,
) -> Tuple[Optional[pl.DataFrame], Optional[str], Optional[int]]:
    y, m = month_start.year, month_start.month
    src_dir = SRC_BASE / f"root={root}" / f"year={y:04d}" / f"month={m:02d}"
    if not src_dir.exists():
        return None, prev_symbol, prev_instrument_id

    files = list(src_dir.glob("*.parquet"))
    if not files:
        return None, prev_symbol, prev_instrument_id

    df_all = pl.read_parquet(str(src_dir / "*.parquet"))
    if df_all.is_empty():
        return None, prev_symbol, prev_instrument_id

    df_all = df_all.with_columns(
        pl.col("symbol").map_elements(parse_month_code, return_dtype=pl.Int32).alias("month_num"),
        pl.col("open_interest").is_not_null().cast(pl.Int8).alias("oi_present"),
    )

    cand = (
        df_all.sort(
            by=["trade_date_utc", "oi_present", "open_interest", "volume", "month_num", "symbol"],
            descending=[False, True, True, True, False, False],
        )
        .group_by("trade_date_utc")
        .agg(pl.all().first())
        .sort("trade_date_utc")
        .select(["trade_date_utc", "symbol", "instrument_id"])
    )

    if cand.is_empty():
        return None, prev_symbol, prev_instrument_id

    trade_dates = cand["trade_date_utc"].to_list()
    cand_syms = cand["symbol"].to_list()
    cand_iids = [int(x) for x in cand["instrument_id"].to_list()]

    stable_syms, stable_iids = stabilize_symbols(
        trade_dates=trade_dates,
        candidate_symbols=cand_syms,
        candidate_instrument_ids=cand_iids,
        prev_symbol=prev_symbol,
        prev_instrument_id=prev_instrument_id,
        confirm_days=confirm_days,
        min_hold_days=min_hold_days,
    )

    stable_keys = pl.DataFrame({"trade_date_utc": trade_dates, "symbol": stable_syms})

    chosen = stable_keys.join(df_all, on=["trade_date_utc", "symbol"], how="left")

    # Fallback to candidate row if stabilized row not found for some date
    if "instrument_id" in chosen.columns and chosen["instrument_id"].null_count() > 0:
        cand_full = cand.rename({"symbol": "cand_symbol", "instrument_id": "cand_instrument_id"})
        tmp = chosen.join(cand_full, on=["trade_date_utc"], how="left")

        # Build alt keys (use candidate symbol when stabilized join misses)
        miss = tmp["instrument_id"].is_null().to_list()
        alt_sym = [tmp["cand_symbol"][i] if miss[i] else tmp["symbol"][i] for i in range(len(miss))]
        alt_keys = pl.DataFrame({"trade_date_utc": tmp["trade_date_utc"], "symbol": alt_sym})
        chosen = alt_keys.join(df_all, on=["trade_date_utc", "symbol"], how="left")

    df_out = (
        chosen.select(
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
        .sort("trade_date_utc")
    )

    last_symbol = df_out["symbol"][-1]
    last_iid = int(df_out["instrument_id"][-1])
    return df_out, last_symbol, last_iid


def compute_roll_events(
    root: str,
    df_out: pl.DataFrame,
    prev_symbol: Optional[str],
    prev_instrument_id: Optional[int],
) -> pl.DataFrame:
    if df_out.is_empty():
        return pl.DataFrame()

    df = df_out.sort("trade_date_utc")

    prev_sym = pl.col("symbol").shift(1)
    prev_iid = pl.col("instrument_id").shift(1)
    prev_close = pl.col("close").shift(1)

    if prev_symbol is not None:
        prev_sym = prev_sym.fill_null(prev_symbol)
    if prev_instrument_id is not None:
        prev_iid = prev_iid.fill_null(prev_instrument_id)

    tmp = df.with_columns(
        prev_sym.alias("from_symbol"),
        prev_iid.alias("from_instrument_id"),
        prev_close.alias("from_close"),
        pl.col("symbol").alias("to_symbol"),
        pl.col("instrument_id").alias("to_instrument_id"),
        pl.col("open").alias("to_open"),
        pl.col("trade_date_utc").alias("roll_date"),
    )

    rolls = (
        tmp.filter(
            pl.col("from_symbol").is_not_null()
            & (pl.col("to_symbol") != pl.col("from_symbol"))
        )
        .select(
            [
                pl.lit(root).alias("root"),
                "roll_date",
                "from_symbol",
                "to_symbol",
                "from_instrument_id",
                "to_instrument_id",
                "from_close",
                "to_open",
            ]
        )
    )
    return rolls


def main() -> None:
    ap = argparse.ArgumentParser(description="Build continuous daily series for all roots (OI->Volume) with roll logging and per-root rules.")
    ap.add_argument("--roots-file", default="roots.txt")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD exclusive-ish")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--root", default=None, help="Optional single-root run")
    ap.add_argument("--write-rolls", action="store_true", help="Write roll logs to parquet partitions")

    # Global fallback if no --rules
    ap.add_argument("--confirm-days", type=int, default=3, help="Global fallback confirm days (used if --rules missing)")
    ap.add_argument("--min-hold-days", type=int, default=0, help="Global fallback min hold days (used if --rules missing)")

    # Per-root rules
    ap.add_argument("--rules", default="", help='Per-root rules: "CL=4:10,NG=3:7,default=3:0"')

    args = ap.parse_args()

    roots_file = Path(args.roots_file).expanduser().resolve()
    if not roots_file.exists():
        raise SystemExit(f"roots file not found: {roots_file}")

    roots = read_roots(roots_file)
    if args.root:
        roots = [r for r in roots if r == args.root]
    if not roots:
        raise SystemExit(f"No roots loaded from {roots_file} (or filtered out by --root).")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    months = month_ranges(start, end)
    if not months:
        raise SystemExit("No months to process (check --start/--end).")

    rules = parse_rules(args.rules)
    global_fallback = (args.confirm_days, args.min_hold_days)

    print(f"Roots loaded: {len(roots)} from {roots_file}")
    print(f"Months: {len(months)} from {start} -> {end}")
    print(f"Output base: {OUT_BASE}")
    if args.write_rolls:
        print(f"Roll base:   {ROLL_BASE}")
    print(f"Progress:    {PROGRESS_PATH}")
    if rules:
        print(f"Rules:       {args.rules}")
    else:
        print(f"Rules:       (none) using global confirm={args.confirm_days} hold={args.min_hold_days}")
    print()

    progress = load_progress()

    wrote = 0
    skipped = 0
    empty = 0
    roll_partitions_written = 0

    for root in roots:
        confirm_days, min_hold_days = get_rule_for_root(rules, root, global_fallback)
        print(f"\n=== ROOT {root} (confirm={confirm_days} hold={min_hold_days}) ===")

        prev_symbol: Optional[str] = None
        prev_instrument_id: Optional[int] = None

        for m_start, _m_end in months:
            mk = month_key(m_start)
            op = out_path(root, m_start)
            rp = roll_path(root, m_start)

            if not args.overwrite and (is_done(progress, root, mk) or (op.exists() and op.stat().st_size > 0)):
                if op.exists():
                    try:
                        existing = pl.read_parquet(op).sort("trade_date_utc")
                        if existing.height > 0:
                            prev_symbol = existing["symbol"][-1]
                            prev_instrument_id = int(existing["instrument_id"][-1])
                    except Exception:
                        pass
                skipped += 1
                continue

            prev_symbol_before = prev_symbol
            prev_iid_before = prev_instrument_id

            df_out, last_sym, last_iid = build_continuous_for_month(
                root=root,
                month_start=m_start,
                prev_symbol=prev_symbol,
                prev_instrument_id=prev_instrument_id,
                confirm_days=confirm_days,
                min_hold_days=min_hold_days,
            )

            if df_out is None or df_out.is_empty():
                empty += 1
                continue

            ensure_dir(op.parent)
            df_out.write_parquet(op, compression="zstd")
            mark_done(progress, root, mk)
            save_progress_atomic(progress)
            wrote += 1
            print(f"Wrote {root} {mk}: rows={df_out.height}")

            if args.write_rolls:
                rolls_df = compute_roll_events(root, df_out, prev_symbol_before, prev_iid_before)
                if rolls_df.height > 0:
                    ensure_dir(rp.parent)
                    rolls_df.write_parquet(rp, compression="zstd")
                    roll_partitions_written += 1
                else:
                    if args.overwrite and rp.exists():
                        rp.unlink(missing_ok=True)

            prev_symbol = last_sym
            prev_instrument_id = last_iid

        save_progress_atomic(progress)

    print(f"\nDone. wrote={wrote} skipped={skipped} empty={empty} roll_partitions_written={roll_partitions_written}")
    print(f"Progress file: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()