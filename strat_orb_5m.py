#!/usr/bin/env python3
"""
strat_orb_5m_v3.py

Research-grade Opening Range Breakout backtester for futures using the parquet lake.

Key fixes in V3
---------------
- Does NOT rely on query_futures_data.py session="rth"
- Loads session="all" and applies RTH logic inside the strategy
- Uses a CME-style futures session date with 23:00 UTC boundary
- Safely handles empty datasets and no-trade periods

Assumptions
-----------
- Bars come from /data/lake/research/resampled_bars/5m
- End time is exclusive
- Bar-based simulator
- If both stop and target are touched in one bar, stop is assumed first
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
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
# MES = $5/point, ES = $50/point
POINT_VALUE = 5.0

# Position sizing
USE_RISK_BASED_SIZING = True
FIXED_CONTRACTS = 1
RISK_PCT_PER_TRADE = 0.05
MAX_CONTRACTS = 2

# Strategy logic
SIDE_MODE = "both"               # "long", "short", "both"
OPENING_RANGE_MINUTES = 30       # first 30 min of RTH
STOP_LOSS_POINTS = 4.0
TAKE_PROFIT_POINTS = 8.0
MAX_HOLD_BARS = None             # e.g. 12 = 1 hour on 5m bars
ONE_TRADE_PER_DAY = True

# RTH session in UTC (CME equity futures approximation)
RTH_START_UTC = (14, 30)
RTH_END_UTC = (21, 0)

# Futures session boundary in UTC
# Bars from 23:00 UTC onward belong to the NEXT session date
SESSION_BOUNDARY_HOUR_UTC = 23

# Output
OUTPUT_DIR = Path(".")
TRADE_LOG_CSV = OUTPUT_DIR / f"orb_{ROOT.lower()}_{START}_{END}_trades.csv"
SUMMARY_CSV = OUTPUT_DIR / f"orb_{ROOT.lower()}_{START}_{END}_summary.csv"


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
# SESSION HELPERS
# ============================================================

def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def futures_session_date(ts: pd.Timestamp) -> dt.date:
    """
    CME-style session date:
    if timestamp is >= 23:00 UTC, assign it to next date.
    """
    ts = _ensure_utc(ts)
    if ts.hour >= SESSION_BOUNDARY_HOUR_UTC:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


def in_rth(ts: pd.Timestamp) -> bool:
    """
    Approximate RTH window for CME equity index futures:
    14:30 UTC <= ts < 21:00 UTC
    """
    ts = _ensure_utc(ts)
    mod = ts.hour * 60 + ts.minute
    start_mod = RTH_START_UTC[0] * 60 + RTH_START_UTC[1]
    end_mod = RTH_END_UTC[0] * 60 + RTH_END_UTC[1]
    return start_mod <= mod < end_mod


# ============================================================
# TRADING HELPERS
# ============================================================

def entry_slippage_adjust(price: float, side: str) -> float:
    if side == "long":
        return price + SLIPPAGE_POINTS
    return price - SLIPPAGE_POINTS


def exit_slippage_adjust(price: float, side: str) -> float:
    if side == "long":
        return price - SLIPPAGE_POINTS
    return price + SLIPPAGE_POINTS


def commission_for_round_trip(contracts: int) -> float:
    return 2.0 * COMMISSION_PER_CONTRACT_SIDE * contracts


def calc_contracts(equity: float) -> int:
    if not USE_RISK_BASED_SIZING:
        return FIXED_CONTRACTS

    risk_dollars_per_contract = STOP_LOSS_POINTS * POINT_VALUE
    if risk_dollars_per_contract <= 0:
        return 0

    target_risk = equity * RISK_PCT_PER_TRADE
    contracts = math.floor(target_risk / risk_dollars_per_contract)
    contracts = max(0, min(contracts, MAX_CONTRACTS))
    return contracts


def safe_summary(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame([{
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
        }])

    wins = trades_df.loc[trades_df["net_pnl_usd"] > 0, "net_pnl_usd"]
    losses = trades_df.loc[trades_df["net_pnl_usd"] < 0, "net_pnl_usd"]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss_abs = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.nan

    eq = trades_df["end_equity"].astype(float)
    peak = eq.cummax()
    dd = eq - peak
    dd_pct = dd / peak.replace(0, np.nan)

    return pd.DataFrame([{
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
    }])


# ============================================================
# STRATEGY ENGINE
# ============================================================

def run_orb_day(day_df: pd.DataFrame, session_date: dt.date, start_equity: float) -> Trade | None:
    """
    day_df should already be filtered to ONE futures session date and to RTH only.
    """
    day_df = day_df.sort_values("ts_event").reset_index(drop=True)

    if day_df.empty:
        return None

    orb_bars = OPENING_RANGE_MINUTES // 5
    if len(day_df) <= orb_bars:
        return None

    opening = day_df.iloc[:orb_bars].copy()
    trade_window = day_df.iloc[orb_bars:].copy()

    if trade_window.empty:
        return None

    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())

    contracts = calc_contracts(start_equity)
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
            long_allowed = SIDE_MODE in ("long", "both")
            short_allowed = SIDE_MODE in ("short", "both")

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
                stop_price = entry_price - STOP_LOSS_POINTS
                target_price = entry_price + TAKE_PROFIT_POINTS
                hold_bars = 0
                continue

            if short_trigger:
                side = "short"
                entry_time = row["ts_event"]
                raw_entry = min(or_low, float(row["open"]))
                entry_price = entry_slippage_adjust(raw_entry, side)
                stop_price = entry_price + STOP_LOSS_POINTS
                target_price = entry_price - TAKE_PROFIT_POINTS
                hold_bars = 0
                continue

        else:
            hold_bars += 1
            high_ = float(row["high"])
            low_ = float(row["low"])

            # conservative intrabar assumption: stop first if both hit
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

            if MAX_HOLD_BARS is not None and hold_bars >= MAX_HOLD_BARS:
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

    if side == "long":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Load ALL bars; session slicing is done inside strategy
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

    # Futures session date assignment
    pdf["session_date"] = pdf["ts_event"].map(futures_session_date)

    # Strategy-side RTH filtering
    pdf = pdf[pdf["ts_event"].map(in_rth)].copy()

    print("Loaded ALL bars from lake:", len(df))
    print("Bars after RTH filter:", len(pdf))

    if pdf.empty:
        print("No RTH bars available after strategy-side filtering.")
        raise SystemExit(0)

    print("Session dates:", pdf["session_date"].nunique())
    print(pdf.head(5))

    trades: list[Trade] = []
    equity = INITIAL_CAPITAL

    for sdate, day_df in pdf.groupby("session_date"):
        trade = run_orb_day(day_df, session_date=sdate, start_equity=equity)
        if trade is not None:
            trades.append(trade)
            equity = trade.end_equity
            if ONE_TRADE_PER_DAY:
                continue

    if len(trades) == 0:
        print("\nNo trades generated in this period.")
        summary_df = safe_summary(pd.DataFrame(), INITIAL_CAPITAL)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"\nWrote summary: {SUMMARY_CSV}")
        raise SystemExit(0)

    trades_df = pd.DataFrame([asdict(t) for t in trades]).sort_values("session_date").reset_index(drop=True)

    trades_df["equity_peak"] = trades_df["end_equity"].cummax()
    trades_df["drawdown_usd"] = trades_df["end_equity"] - trades_df["equity_peak"]
    trades_df["drawdown_pct"] = trades_df["drawdown_usd"] / trades_df["equity_peak"].replace(0, np.nan)

    summary_df = safe_summary(trades_df, INITIAL_CAPITAL)

    print("\nORB Summary")
    print("-----------")
    print(summary_df.to_string(index=False))

    print("\nSample Trades")
    show_cols = [
        "session_date",
        "side",
        "contracts",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "reason",
        "net_pnl_usd",
        "end_equity",
    ]
    print(trades_df[show_cols].head(15).to_string(index=False))

    trades_df.to_csv(TRADE_LOG_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print(f"\nWrote trade log: {TRADE_LOG_CSV}")
    print(f"Wrote summary:   {SUMMARY_CSV}")


if __name__ == "__main__":
    main()