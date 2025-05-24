# streamlit_app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler

from backtest import run_backtest
from forecast_and_trade import forecast_and_trade, PAIRS
from data.fetch_candles import fetch_ohlcv
from features import compute_features, compute_raw_features
from model.utils import load_params, load_scaler
from model.qml import qnode

# --- Page Config ---
st.set_page_config(page_title="Quantum FX Forecast & Trading", layout="centered")
st.title("🚀 Quantum FX Forecast & Trading")

# --- Backtest & Trade Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Run Backtest"):
        st.info("Running backtest, please wait…")
        run_backtest()
        st.success("Backtest complete. See logs for details.")
with col2:
    if st.button("Run Forecast & Trade"):
        st.info("Executing forecast & trade stub…")
        try:
            forecast_and_trade()
            st.success("Forecast executed. Check logs for stub output.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.header("🔮 Price Prediction Chart")

if st.checkbox("Show Price Prediction Chart"):
    # 1) UI controls
    pair     = st.selectbox( "Select FX Pair", PAIRS )
    days     = st.slider(      "Days of data",   7, 90, 30 )
    interval = st.selectbox(   "Interval",       ["1h","4h","1d"], index=0 )
    lookback = st.number_input("Lookback periods", 1, 168, 24)
    risk     = st.slider(      "Risk multiplier", 0.0, 1.0, 0.8)

    # 2) Fetch data
    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
        st.stop()

    st.subheader(f"Actual Close Price for {pair}")
    st.line_chart(df["close"])

    # 3) Load or initialize model params
    params = load_params()

    # 4) Attempt to load scaler; if missing, build one on the fly
    try:
        scaler = load_scaler()
    except FileNotFoundError:
        st.info("No saved scaler found — fitting scaler on the fly for your chart.")
        # Build feature matrix over all windows
        X_raw = []
        for t in range(lookback, len(df)-1):
            window = df.iloc[t-lookback : t+1]
            X_raw.append(compute_raw_features(window))
        X_raw = np.vstack(X_raw)
        scaler = StandardScaler().fit(X_raw)

    # 5) Generate actual vs predicted series
    times, actuals, preds = [], [], []
    for t in range(lookback, len(df)-1):
        window = df.iloc[t-lookback : t+1]
        # Compute (and scale) features
        x = compute_features(window, scaler)
        q_out = float(qnode(params, x))
        last_price = float(window["close"].iloc[-1])
        sigma = window["close"].pct_change().std()
        r_hat = q_out * sigma * risk
        pred_price = last_price * (1 + r_hat)

        times.append(df.index[t+1])
        actuals.append(float(df["close"].iloc[t+1]))
        preds.append(pred_price)

    # 6) Create tidy DataFrame for Altair
    chart_df = pd.DataFrame({
        "time": times,
        "Actual": actuals,
        "Predicted": preds
    })
    melted = chart_df.melt(id_vars=["time"], var_name="Series", value_name="Price")

    # 7) Plot with Altair
    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("Price:Q", title="Price"),
            color=alt.Color("Series:N")
        )
        .properties(
            width=700, height=400,
            title=f"{pair}: Actual vs QML-Predicted Next-Close"
        )
    )
    st.altair_chart(chart, use_container_width=True)
