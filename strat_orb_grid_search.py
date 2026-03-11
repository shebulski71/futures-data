#!/usr/bin/env python3
"""
strat_orb_gridsearch.py

Grid search for Opening Range Breakout on futures data from the research lake.

Features
--------
- loads bars from query_futures_data.py
- applies futures session-date logic internally
- runs many ORB parameter combinations
- outputs ranked results table
- writes CSV results

Notes
-----
- Uses session="all" from the loader, then filters RTH in-strategy
- Conservative intrabar assumption:
    if both stop and target are touched in one bar, stop is assumed first
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import math
import time
import datetime as dt

import numpy as np
import pandas as pd

from query_futures_data import load_bars


# ============================================================
# CONFIG
# ============================================================

ROOT = "MES"
FREQ = "5m"
START = "2026-01-01"
END = "2026-02-01"
DATASET = "resampled"

INITIAL_CAPITAL = 500.0

# Trading costs
COMMISSION_PER_CONTRACT_SIDE = 0.62
SLIPPAGE_POINTS = 0.25

# Contract economics
POINT_VALUE = 5.0  # MES = $5/point, ES = $50/point

# Position sizing
USE_RISK_BASED_SIZING = False
FIXED_CONTRACTS = 1
RISK_PCT_PER_TRADE = 0.05
MAX_CONTRACTS = 2

# RTH window for CME equity index futures approximation
RTH_START_UTC = (14, 30)
RTH_END_UTC = (21, 0)

# Session boundary: bars >= 23:00 UTC belong to next futures session
SESSION_BOUNDARY_HOUR_UTC = 23

# Grid parameters
OPENING_RANGE_MINUTES_GRID = [5, 15, 30]
STOP_LOSS_POINTS_GRID = [2.0, 3.0, 4.0]
TARGET_MULTIPLIER_GRID = [1.0, 1.5, 2.0]
SIDE_MODE_GRID = ["long", "short", "both"]
MAX_HOLD_BARS_GRID = [None]   # e.g. [None, 12, 24]

ONE_TRADE_PER_DAY = True

# Output
OUTPUT_DIR = Path(".")
RUN_TAG = f"{ROOT.lower()}_{START}_{END}"
RESULTS_CSV = OUTPUT_DIR / f"orb_gridsearch_{RUN_TAG}.csv"
BEST_TRADES_CSV = OUTPUT_DIR / f"orb_gridsearch_best_trades_{RUN_TAG}.csv"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    session_date: object
    root: str
    side: str
    contracts: int
    entry_time: object
    entry_price: float
    exit_time: object
    exit_price: float
    reason: str
    or_high: float
    or_low: float
    stop_price: float
    target_price: float
    gross_pnl_usd: float
    commission_usd: float
    net_pnl_usd: float
    start_equity: float
    end_equity: float
    hold_bars: int


# ============================================================
# HELPERS
# ============================================================

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def futures_session_date(ts: pd.Timestamp) -> dt.date:
    ts = _ensure_utc(ts)
    if ts.hour >= SESSION_BOUNDARY_HOUR_UTC:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


def in_rth(ts: pd.Timestamp) -> bool:
    ts = _ensure_utc(ts)
    mod = ts.hour * 60 + ts.minute
    start_mod = RTH_START_UTC[0] * 60 + RTH_START_UTC[1]
    end_mod = RTH_END_UTC[0] * 60 + RTH_END_UTC[1]
    return start_mod <= mod < end_mod


def entry_slippage_adjust(price: float, side: str) -> float:
    return price + SLIPPAGE_POINTS if side == "long" else price - SLIPPAGE_POINTS


def exit_slippage_adjust(price: float, side: str) -> float:
    return price - SLIPPAGE_POINTS if side == "long" else price + SLIPPAGE_POINTS


def commission_for_round_trip(contracts: int) -> float:
    return 2.0 * COMMISSION_PER_CONTRACT_SIDE * contracts


def calc_contracts(equity: float, stop_loss_points: float) -> int:
    if not USE_RISK_BASED_SIZING:
        return FIXED_CONTRACTS

    risk_dollars_per_contract = stop_loss_points * POINT_VALUE
    if risk_dollars_per_contract <= 0:
        return 0

    target_risk = equity * RISK_PCT_PER_TRADE
    contracts = math.floor(target_risk / risk_dollars_per_contract)
    contracts = max(0, min(contracts, MAX_CONTRACTS))
    return contracts


def summarize_trades(trades_df: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty:
        return {
            "start_equity": initial_capital,
            "end_equity": initial_capital,
            "total_net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_commission": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = trades_df.loc[trades_df["net_pnl_usd"] > 0, "net_pnl_usd"]
    losses = trades_df.loc[trades_df["net_pnl_usd"] < 0, "net_pnl_usd"]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss_abs = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.nan

    eq = trades_df["end_equity"].astype(float)
    peak = eq.cummax()
    dd = eq - peak
    dd_pct = dd / peak.replace(0, np.nan)

    return {
        "start_equity": initial_capital,
        "end_equity": float(eq.iloc[-1]),
        "total_net_pnl": float(trades_df["net_pnl_usd"].sum()),
        "gross_pnl": float(trades_df["gross_pnl_usd"].sum()),
        "total_commission": float(trades_df["commission_usd"].sum()),
        "num_trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_pnl_usd"] > 0).mean()) if len(trades_df) else 0.0,
        "avg_trade": float(trades_df["net_pnl_usd"].mean()) if len(trades_df) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "max_drawdown_usd": float(dd.min()) if len(dd) else 0.0,
        "max_drawdown_pct": float(dd_pct.min()) if len(dd_pct) else 0.0,
    }


# ============================================================
# STRATEGY ENGINE
# ============================================================

def run_orb_day(
    day_df: pd.DataFrame,
    session_date: dt.date,
    start_equity: float,
    opening_range_minutes: int,
    stop_loss_points: float,
    target_multiplier: float,
    side_mode: str,
    max_hold_bars: int | None,
) -> Trade | None:
    day_df = day_df.sort_values("ts_event").reset_index(drop=True)

    if day_df.empty:
        return None

    orb_bars = opening_range_minutes // 5
    if len(day_df) <= orb_bars:
        return None

    opening = day_df.iloc[:orb_bars].copy()
    trade_window = day_df.iloc[orb_bars:].copy()
    if trade_window.empty:
        return None

    target_points = stop_loss_points * target_multiplier

    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    contracts = calc_contracts(start_equity, stop_loss_points)
    if contracts <= 0:
        return None

    side = None
    entry_time = None
    entry_price = None
    stop_price = None
    target_price = None
    hold_bars = 0

    for _, row in trade_window.iterrows():
        if side is None:
            long_allowed = side_mode in ("long", "both")
            short_allowed = side_mode in ("short", "both")

            long_trigger = long_allowed and (float(row["high"]) > or_high)
            short_trigger = short_allowed and (float(row["low"]) < or_low)

            if long_trigger and short_trigger:
                dist_up = abs(or_high - float(row["open"]))
                dist_dn = abs(float(row["open"]) - or_low)
                if dist_up <= dist_dn:
                    long_trigger, short_trigger = True, False
                else:
                    long_trigger, short_trigger = False, True

            if long_trigger:
                side = "long"
                entry_time = row["ts_event"]
                raw_entry = max(or_high, float(row["open"]))
                entry_price = entry_slippage_adjust(raw_entry, side)
                stop_price = entry_price - stop_loss_points
                target_price = entry_price + target_points
                hold_bars = 0
                continue

            if short_trigger:
                side = "short"
                entry_time = row["ts_event"]
                raw_entry = min(or_low, float(row["open"]))
                entry_price = entry_slippage_adjust(raw_entry, side)
                stop_price = entry_price + stop_loss_points
                target_price = entry_price - target_points
                hold_bars = 0
                continue

        else:
            hold_bars += 1
            high_ = float(row["high"])
            low_ = float(row["low"])

            if side == "long":
                stop_hit = low_ <= stop_price
                target_hit = high_ >= target_price

                if stop_hit:
                    exit_time = row["ts_event"]
                    exit_price = exit_slippage_adjust(stop_price, side)
                    reason = "stop"
                    break

                if target_hit:
                    exit_time = row["ts_event"]
                    exit_price = exit_slippage_adjust(target_price, side)
                    reason = "target"
                    break

            else:
                stop_hit = high_ >= stop_price
                target_hit = low_ <= target_price

                if stop_hit:
                    exit_time = row["ts_event"]
                    exit_price = exit_slippage_adjust(stop_price, side)
                    reason = "stop"
                    break

                if target_hit:
                    exit_time = row["ts_event"]
                    exit_price = exit_slippage_adjust(target_price, side)
                    reason = "target"
                    break

            if max_hold_bars is not None and hold_bars >= max_hold_bars:
                exit_time = row["ts_event"]
                exit_price = exit_slippage_adjust(float(row["close"]), side)
                reason = "max_hold"
                break

    else:
        if side is None:
            return None
        last_row = day_df.iloc[-1]
        exit_time = last_row["ts_event"]
        exit_price = exit_slippage_adjust(float(last_row["close"]), side)
        reason = "eod"

    pnl_points = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
    gross_pnl_usd = pnl_points * POINT_VALUE * contracts
    commission_usd = commission_for_round_trip(contracts)
    net_pnl_usd = gross_pnl_usd - commission_usd
    end_equity = start_equity + net_pnl_usd

    return Trade(
        session_date=session_date,
        root=ROOT,
        side=side,
        contracts=contracts,
        entry_time=entry_time,
        entry_price=float(entry_price),
        exit_time=exit_time,
        exit_price=float(exit_price),
        reason=reason,
        or_high=float(or_high),
        or_low=float(or_low),
        stop_price=float(stop_price),
        target_price=float(target_price),
        gross_pnl_usd=float(gross_pnl_usd),
        commission_usd=float(commission_usd),
        net_pnl_usd=float(net_pnl_usd),
        start_equity=float(start_equity),
        end_equity=float(end_equity),
        hold_bars=int(hold_bars),
    )


def run_parameter_set(
    pdf: pd.DataFrame,
    opening_range_minutes: int,
    stop_loss_points: float,
    target_multiplier: float,
    side_mode: str,
    max_hold_bars: int | None,
) -> tuple[dict, pd.DataFrame]:
    trades: list[Trade] = []
    equity = INITIAL_CAPITAL

    for sdate, day_df in pdf.groupby("session_date"):
        trade = run_orb_day(
            day_df=day_df,
            session_date=sdate,
            start_equity=equity,
            opening_range_minutes=opening_range_minutes,
            stop_loss_points=stop_loss_points,
            target_multiplier=target_multiplier,
            side_mode=side_mode,
            max_hold_bars=max_hold_bars,
        )
        if trade is not None:
            trades.append(trade)
            equity = trade.end_equity
            if ONE_TRADE_PER_DAY:
                continue

    if len(trades) == 0:
        trades_df = pd.DataFrame()
        stats = summarize_trades(trades_df, INITIAL_CAPITAL)
    else:
        trades_df = pd.DataFrame([asdict(t) for t in trades]).sort_values("session_date").reset_index(drop=True)
        trades_df["equity_peak"] = trades_df["end_equity"].cummax()
        trades_df["drawdown_usd"] = trades_df["end_equity"] - trades_df["equity_peak"]
        trades_df["drawdown_pct"] = trades_df["drawdown_usd"] / trades_df["equity_peak"].replace(0, np.nan)
        stats = summarize_trades(trades_df, INITIAL_CAPITAL)

    result = {
        "root": ROOT,
        "freq": FREQ,
        "start": START,
        "end": END,
        "opening_range_minutes": opening_range_minutes,
        "stop_loss_points": stop_loss_points,
        "target_multiplier": target_multiplier,
        "target_points": stop_loss_points * target_multiplier,
        "side_mode": side_mode,
        "max_hold_bars": max_hold_bars,
        **stats,
    }
    return result, trades_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    t0 = time.time()

    print("Loading bars from lake...")
    df = load_bars(
        roots=ROOT,
        freq=FREQ,
        start=START,
        end=END,
        dataset=DATASET,
        session="all",
        as_pandas=False,
    )

    pdf = df.to_pandas()
    if pdf.empty:
        print("No bars loaded from lake. Check root/frequency/date range.")
        raise SystemExit(0)

    pdf["ts_event"] = pd.to_datetime(pdf["ts_event"], utc=True)
    pdf = pdf.sort_values("ts_event").reset_index(drop=True)

    pdf["session_date"] = pdf["ts_event"].map(futures_session_date)
    pdf = pdf[pdf["ts_event"].map(in_rth)].copy()

    print("Loaded ALL bars from lake:", len(df))
    print("Bars after RTH filter:", len(pdf))
    print("Session dates:", pdf["session_date"].nunique())

    if pdf.empty:
        print("No RTH bars available after strategy-side filtering.")
        raise SystemExit(0)

    grid = list(product(
        OPENING_RANGE_MINUTES_GRID,
        STOP_LOSS_POINTS_GRID,
        TARGET_MULTIPLIER_GRID,
        SIDE_MODE_GRID,
        MAX_HOLD_BARS_GRID,
    ))

    print(f"Running {len(grid)} parameter combinations...")

    results = []
    best_trades_df = None
    best_score = -np.inf

    for i, (or_minutes, stop_pts, target_mult, side_mode, max_hold) in enumerate(grid, start=1):
        result, trades_df = run_parameter_set(
            pdf=pdf,
            opening_range_minutes=or_minutes,
            stop_loss_points=stop_pts,
            target_multiplier=target_mult,
            side_mode=side_mode,
            max_hold_bars=max_hold,
        )

        results.append(result)

        # ranking metric: end equity, then lower drawdown as tie-break
        score = result["end_equity"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_trades_df = trades_df.copy() if trades_df is not None else None

        if i % 10 == 0 or i == len(grid):
            print(f"  completed {i}/{len(grid)}")

    results_df = pd.DataFrame(results)

    # Sort best-first
    results_df = results_df.sort_values(
        by=["end_equity", "total_net_pnl", "profit_factor", "max_drawdown_usd"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_CSV, index=False)

    if best_trades_df is not None and not best_trades_df.empty:
        best_trades_df.to_csv(BEST_TRADES_CSV, index=False)

    elapsed = time.time() - t0

    print("\nTop 20 parameter sets")
    print("---------------------")
    show_cols = [
        "opening_range_minutes",
        "stop_loss_points",
        "target_multiplier",
        "target_points",
        "side_mode",
        "num_trades",
        "win_rate",
        "total_net_pnl",
        "end_equity",
        "profit_factor",
        "max_drawdown_usd",
        "max_drawdown_pct",
    ]
    print(results_df[show_cols].head(20).to_string(index=False))

    print(f"\nWrote results:     {RESULTS_CSV}")
    if best_trades_df is not None and not best_trades_df.empty:
        print(f"Wrote best trades: {BEST_TRADES_CSV}")
    print(f"Elapsed seconds:   {elapsed:.2f}")


if __name__ == "__main__":
    main()