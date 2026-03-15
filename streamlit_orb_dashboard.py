import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from query_futures_data import load_bars

try:
    import polars as pl
except Exception:
    pl = None

try:
    import pyarrow as pa
except Exception:
    pa = None

try:
    from strat_orb_5m_v3 import run_orb_backtest
    HAS_STRAT = True
except Exception:
    HAS_STRAT = False


# ===============================
# Theme
# ===============================

BG = "#07080D"
PANEL = "#11131C"
PANEL_2 = "#161927"
BORDER = "#23263A"
TEXT = "#F3F4F8"
MUTED = "#9AA3B2"
ACCENT = "#8B5CF6"
ACCENT_2 = "#6D5EF8"
CYAN = "#7DD3FC"
GREEN = "#22C55E"
RED = "#F43F5E"
GRID = "rgba(154,163,178,0.12)"

st.set_page_config(page_title="ORB Research Dashboard", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(139,92,246,0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(125,211,252,0.08), transparent 22%),
            {BG};
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1500px;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0A0B12 0%, #0D0F17 100%);
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div,
    .stNumberInput > div > div {{
        background: {PANEL_2};
        border: 1px solid {BORDER};
        border-radius: 14px;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 8px 24px rgba(139,92,246,0.28);
    }}

    .stButton > button:hover {{
        filter: brightness(1.06);
    }}

    .orb-card {{
        background: linear-gradient(180deg, rgba(22,25,39,0.96) 0%, rgba(17,19,28,0.96) 100%);
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.28);
    }}

    .orb-card-title {{
        font-size: 0.92rem;
        color: {MUTED};
        margin-bottom: 0.35rem;
    }}

    .orb-metric {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.1;
    }}

    .orb-sub {{
        color: {MUTED};
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }}

    .orb-header {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 0.1rem;
    }}

    .orb-caption {{
        color: {MUTED};
        margin-bottom: 1.0rem;
    }}

    .stDataFrame {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        overflow: hidden;
    }}

    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===============================
# Helpers
# ===============================

def card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="orb-card">
            <div class="orb-card-title">{title}</div>
            <div class="orb-metric">{value}</div>
            <div class="orb-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def infer_pandas(df):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if pa is not None and isinstance(df, pa.Table):
        return df.to_pandas()
    return df


@st.cache_data(show_spinner=False)
def get_bars(root, start, end, timeframe, dataset, session_mode):
    bars = load_bars(
        roots=root,
        freq=timeframe,
        start=str(start),
        end=str(end),
        dataset=dataset,
        session=session_mode,
    )

    bars = infer_pandas(bars)

    if not isinstance(bars, pd.DataFrame):
        bars = pd.DataFrame(bars)

    if "ts_event" not in bars.columns:
        raise ValueError("Expected 'ts_event' column in bars data.")

    bars["ts_event"] = pd.to_datetime(bars["ts_event"], utc=False, errors="coerce")
    bars = bars.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    needed = ["open", "high", "low", "close"]
    for c in needed:
        if c not in bars.columns:
            raise ValueError(f"Expected '{c}' column in bars data.")

    return bars


def compute_session_date(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    return (ts + pd.to_timedelta((ts.dt.hour >= 23).astype(int), unit="D")).dt.date


def fallback_orb(bars, orb_minutes=30, stop_multiple=1.0, target_multiple=2.0):
    bars = bars.copy()
    bars["session_date"] = compute_session_date(bars["ts_event"])

    trades = []
    equity_points = []
    running_pnl = 0.0

    bar_minutes = int((bars["ts_event"].diff().dropna().median() / pd.Timedelta(minutes=1))) if len(bars) > 1 else 5
    orb_bars = max(1, int(orb_minutes / max(bar_minutes, 1)))

    for session_date, day in bars.groupby("session_date"):
        day = day.sort_values("ts_event").reset_index(drop=True)
        if len(day) <= orb_bars + 1:
            continue

        orb = day.iloc[:orb_bars]
        orb_high = float(orb["high"].max())
        orb_low = float(orb["low"].min())
        orb_range = max(orb_high - orb_low, 1e-9)

        entry = None

        for _, row in day.iloc[orb_bars:].iterrows():
            if entry is None:
                if row["high"] > orb_high:
                    entry = {
                        "side": "long",
                        "entry_time": row["ts_event"],
                        "entry_price": orb_high,
                        "stop_price": orb_high - stop_multiple * orb_range,
                        "target_price": orb_high + target_multiple * orb_range,
                        "session_date": session_date,
                    }
                elif row["low"] < orb_low:
                    entry = {
                        "side": "short",
                        "entry_time": row["ts_event"],
                        "entry_price": orb_low,
                        "stop_price": orb_low + stop_multiple * orb_range,
                        "target_price": orb_low - target_multiple * orb_range,
                        "session_date": session_date,
                    }
                continue

            exit_reason = None
            exit_price = None

            if entry["side"] == "long":
                if row["low"] <= entry["stop_price"]:
                    exit_price = entry["stop_price"]
                    exit_reason = "stop"
                elif row["high"] >= entry["target_price"]:
                    exit_price = entry["target_price"]
                    exit_reason = "target"
            else:
                if row["high"] >= entry["stop_price"]:
                    exit_price = entry["stop_price"]
                    exit_reason = "stop"
                elif row["low"] <= entry["target_price"]:
                    exit_price = entry["target_price"]
                    exit_reason = "target"

            if exit_price is None:
                is_last_bar = row["ts_event"] == day.iloc[-1]["ts_event"]
                if is_last_bar:
                    exit_price = float(row["close"])
                    exit_reason = "eod"

            if exit_price is not None:
                pnl = (
                    exit_price - entry["entry_price"]
                    if entry["side"] == "long"
                    else entry["entry_price"] - exit_price
                )
                running_pnl += pnl

                trades.append({
                    **entry,
                    "exit_time": row["ts_event"],
                    "exit_price": float(exit_price),
                    "exit_reason": exit_reason,
                    "pnl": float(pnl),
                })
                equity_points.append({
                    "exit_time": row["ts_event"],
                    "equity": running_pnl,
                })
                break

    trades = pd.DataFrame(trades)
    equity = pd.DataFrame(equity_points)

    if not equity.empty:
        equity["drawdown"] = equity["equity"] - equity["equity"].cummax()
    else:
        equity = pd.DataFrame(columns=["exit_time", "equity", "drawdown"])

    return trades, equity


def stats_from_trades(trades: pd.DataFrame):
    if trades.empty:
        return {
            "net": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "pf": 0.0,
        }

    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = trades.loc[trades["pnl"] < 0, "pnl"]

    gross_profit = wins.sum() if not wins.empty else 0.0
    gross_loss = abs(losses.sum()) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0

    return {
        "net": float(trades["pnl"].sum()),
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean() * 100.0),
        "pf": float(pf),
    }


def base_layout(title=""):
    return dict(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        margin=dict(l=16, r=16, t=42, b=16),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(color=MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            linecolor=BORDER,
            tickfont=dict(color=MUTED),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
    )


# ===============================
# Sidebar
# ===============================

st.sidebar.markdown("## ORB Controls")

root = st.sidebar.selectbox("Root", ["MES", "ES", "NQ", "MNQ", "CL", "GC"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h"], index=1)
dataset = st.sidebar.selectbox("Dataset", ["resampled", "continuous", "features"], index=0)
session_mode = st.sidebar.selectbox("Loader Session Mode", ["all", "rth"], index=0)

start = st.sidebar.date_input("Start", date(2026, 1, 1))
end = st.sidebar.date_input("End", date(2026, 2, 1))

st.sidebar.markdown("---")
orb_minutes = st.sidebar.number_input("ORB Window (minutes)", 5, 120, 30, step=5)
stop_multiple = st.sidebar.number_input("Stop multiple", 0.25, 10.0, 1.0, step=0.25)
target_multiple = st.sidebar.number_input("Target multiple", 0.25, 20.0, 2.0, step=0.25)

run_bt = st.sidebar.button("Run Backtest", use_container_width=True)


# ===============================
# Header
# ===============================

st.markdown('<div class="orb-header">ORB Research Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="orb-caption">Local parquet lake • Streamlit UI • Futures session-aware backtesting</div>',
    unsafe_allow_html=True,
)

bars = get_bars(root, start, end, timeframe, dataset, session_mode)

# ===============================
# Top stats
# ===============================

c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Root", root, f"{dataset} • {timeframe}")
with c2:
    card("Bars Loaded", f"{len(bars):,}", f"{bars['ts_event'].min()} → {bars['ts_event'].max()}")
with c3:
    card("Session Mode", session_mode, "Keep loader on all for futures strategy work")
with c4:
    bar_span = f"{start} → {end}"
    card("Date Range", bar_span, "CME session-date handling stays in strategy layer")

# ===============================
# Preview chart
# ===============================

preview = go.Figure()
preview.add_trace(
    go.Candlestick(
        x=bars["ts_event"],
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        name="Price",
        increasing_line_color=ACCENT,
        increasing_fillcolor=ACCENT,
        decreasing_line_color=CYAN,
        decreasing_fillcolor=CYAN,
    )
)
preview.update_layout(**base_layout("Price Preview"))
preview.update_xaxes(rangeslider_visible=False)

st.plotly_chart(preview, use_container_width=True)

# ===============================
# Backtest
# ===============================

if run_bt:
    if HAS_STRAT:
        try:
            trades, equity = run_orb_backtest(
                bars,
                orb_minutes=orb_minutes,
                stop_multiple=stop_multiple,
                target_multiple=target_multiple,
            )
            trades = infer_pandas(trades)
            equity = infer_pandas(equity)
            if not isinstance(trades, pd.DataFrame):
                trades = pd.DataFrame(trades)
            if not isinstance(equity, pd.DataFrame):
                equity = pd.DataFrame(equity)
        except Exception as e:
            st.warning(f"Falling back to built-in ORB engine because strat_orb_5m_v3 failed: {e}")
            trades, equity = fallback_orb(
                bars,
                orb_minutes=orb_minutes,
                stop_multiple=stop_multiple,
                target_multiple=target_multiple,
            )
    else:
        trades, equity = fallback_orb(
            bars,
            orb_minutes=orb_minutes,
            stop_multiple=stop_multiple,
            target_multiple=target_multiple,
        )

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    stats = stats_from_trades(trades)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        card("Net PnL", f"{stats['net']:,.2f}", "Aggregate trade PnL")
    with m2:
        card("Trades", f"{stats['trades']:,}", "Completed trades")
    with m3:
        card("Win Rate", f"{stats['win_rate']:.1f}%", "Winning trades ratio")
    with m4:
        pf_display = "∞" if np.isinf(stats["pf"]) else f"{stats['pf']:.2f}"
        card("Profit Factor", pf_display, "Gross profit / gross loss")

    left, right = st.columns([2.1, 1.1])

    with left:
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=bars["ts_event"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
                name="Price",
                increasing_line_color=ACCENT,
                increasing_fillcolor=ACCENT,
                decreasing_line_color=CYAN,
                decreasing_fillcolor=CYAN,
            )
        )

        if not trades.empty:
            longs = trades[trades["side"] == "long"]
            shorts = trades[trades["side"] == "short"]

            if not longs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=longs["entry_time"],
                        y=longs["entry_price"],
                        mode="markers",
                        name="Long Entries",
                        marker=dict(size=11, symbol="triangle-up", color=GREEN, line=dict(width=1, color=TEXT)),
                    )
                )

            if not shorts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=shorts["entry_time"],
                        y=shorts["entry_price"],
                        mode="markers",
                        name="Short Entries",
                        marker=dict(size=11, symbol="triangle-down", color=RED, line=dict(width=1, color=TEXT)),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=trades["exit_time"],
                    y=trades["exit_price"],
                    mode="markers",
                    name="Exits",
                    marker=dict(size=9, symbol="x", color=TEXT),
                )
            )

        fig.update_layout(**base_layout("Candles + Trade Markers"))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if equity.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(**base_layout("No backtest results"))
            st.plotly_chart(empty_fig, use_container_width=True)
        else:
            eq_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.64, 0.36])

            eq_fig.add_trace(
                go.Scatter(
                    x=equity["exit_time"],
                    y=equity["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color=ACCENT, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(139,92,246,0.12)",
                ),
                row=1,
                col=1,
            )

            eq_fig.add_trace(
                go.Scatter(
                    x=equity["exit_time"],
                    y=equity["drawdown"],
                    mode="lines",
                    name="Drawdown",
                    line=dict(color=CYAN, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(125,211,252,0.12)",
                ),
                row=2,
                col=1,
            )

            eq_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=PANEL,
                font=dict(color=TEXT),
                margin=dict(l=16, r=16, t=42, b=16),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED),
                ),
            )

            eq_fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
            eq_fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=BORDER, tickfont=dict(color=MUTED))

            st.plotly_chart(eq_fig, use_container_width=True)

    st.markdown("### Trade Log")
    if trades.empty:
        st.info("No trades generated for the selected configuration.")
    else:
        cols = [c for c in [
            "session_date", "side", "entry_time", "entry_price",
            "exit_time", "exit_price", "exit_reason", "pnl"
        ] if c in trades.columns]

        display = trades[cols].copy()
        if "pnl" in display.columns:
            display["pnl"] = display["pnl"].round(2)

        st.dataframe(display, use_container_width=True, hide_index=True)