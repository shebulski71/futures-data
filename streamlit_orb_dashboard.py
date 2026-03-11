import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

# your research imports
from query_futures_data import load_bars

# optional real backtest
try:
    from strat_orb_5m_v3 import run_orb_backtest
    HAS_STRAT = True
except:
    HAS_STRAT = False


st.set_page_config(
    page_title="ORB Research Dashboard",
    layout="wide"
)


# ===============================
# Sidebar Controls
# ===============================

st.sidebar.title("ORB Controls")

root = st.sidebar.selectbox(
    "Root",
    ["MES","ES","NQ","MNQ","CL","GC"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m","5m","15m","30m","1h"],
    index=1
)

dataset = st.sidebar.selectbox(
    "Dataset",
    ["resampled","continuous","features"],
    index=0
)

session_mode = st.sidebar.selectbox(
    "Loader Session Mode",
    ["all","rth"],
    index=0
)

start = st.sidebar.date_input("Start", date(2026,1,1))
end   = st.sidebar.date_input("End", date(2026,2,1))


st.sidebar.markdown("---")

orb_minutes = st.sidebar.number_input(
    "ORB Window (minutes)",
    5,
    60,
    30
)

stop_multiple = st.sidebar.number_input(
    "Stop multiple",
    0.5,
    10.0,
    1.0
)

target_multiple = st.sidebar.number_input(
    "Target multiple",
    0.5,
    20.0,
    2.0
)


# ===============================
# Cached Data Loader
# ===============================

@st.cache_data(show_spinner=False)
def get_bars(root, start, end, timeframe, dataset, session_mode):

    bars = load_bars(
        roots=root,
        freq=timeframe,
        start=str(start),
        end=str(end),
        dataset=dataset,
        session=session_mode
    )

    bars = bars.sort_values("ts_event")
    bars = bars.reset_index(drop=True)

    return bars


# ===============================
# Simple fallback ORB engine
# ===============================

def fallback_orb(bars):

    trades = []
    equity = []
    pnl = 0

    bars["session_date"] = bars["ts_event"].dt.date

    for d, day in bars.groupby("session_date"):

        if len(day) < 10:
            continue

        orb = day.iloc[:6]

        orb_high = orb.high.max()
        orb_low = orb.low.min()

        trade = None

        for i,row in day.iterrows():

            if trade is None:

                if row.high > orb_high:

                    trade = {
                        "entry_time":row.ts_event,
                        "entry_price":orb_high,
                        "side":"long"
                    }

                elif row.low < orb_low:

                    trade = {
                        "entry_time":row.ts_event,
                        "entry_price":orb_low,
                        "side":"short"
                    }

            else:

                exit_price = row.close

                pnl_trade = (
                    exit_price - trade["entry_price"]
                    if trade["side"]=="long"
                    else trade["entry_price"] - exit_price
                )

                pnl += pnl_trade

                trades.append({
                    **trade,
                    "exit_time":row.ts_event,
                    "exit_price":exit_price,
                    "pnl":pnl_trade
                })

                break

        equity.append(pnl)

    trades = pd.DataFrame(trades)

    equity = pd.Series(equity)

    return trades, equity


# ===============================
# Load Data
# ===============================

bars = get_bars(root,start,end,timeframe,dataset,session_mode)

st.title("Futures ORB Research Dashboard")

st.write(f"{len(bars)} bars loaded")


# ===============================
# Run Backtest
# ===============================

if st.button("Run Backtest"):

    if HAS_STRAT:

        trades, equity = run_orb_backtest(
            bars,
            orb_minutes=orb_minutes,
            stop_multiple=stop_multiple,
            target_multiple=target_multiple
        )

    else:

        trades, equity = fallback_orb(bars)


    # ===============================
    # Candlestick Chart
    # ===============================

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=bars.ts_event,
        open=bars.open,
        high=bars.high,
        low=bars.low,
        close=bars.close,
        name="Price"
    ))


    if len(trades):

        fig.add_trace(go.Scatter(
            x=trades.entry_time,
            y=trades.entry_price,
            mode="markers",
            name="Entries",
            marker=dict(size=10,symbol="triangle-up")
        ))

        fig.add_trace(go.Scatter(
            x=trades.exit_time,
            y=trades.exit_price,
            mode="markers",
            name="Exits",
            marker=dict(size=10,symbol="x")
        ))


    st.plotly_chart(fig,use_container_width=True)


    # ===============================
    # Equity Curve
    # ===============================

    st.subheader("Equity Curve")

    eq_fig = go.Figure()

    eq_fig.add_trace(go.Scatter(
        y=equity,
        mode="lines",
        name="Equity"
    ))

    st.plotly_chart(eq_fig,use_container_width=True)


    # ===============================
    # Drawdown
    # ===============================

    st.subheader("Drawdown")

    dd = equity - equity.cummax()

    dd_fig = go.Figure()

    dd_fig.add_trace(go.Scatter(
        y=dd,
        mode="lines",
        name="Drawdown"
    ))

    st.plotly_chart(dd_fig,use_container_width=True)


    # ===============================
    # Trade Table
    # ===============================

    st.subheader("Trades")

    st.dataframe(trades)