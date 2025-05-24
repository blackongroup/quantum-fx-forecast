# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math

from backtest import run_backtest
from forecast_and_trade import forecast_and_trade, PAIRS
from data.fetch_candles import fetch_ohlcv
from features import compute_raw_features
from model.utils import load_params
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
    # UI controls
    pair = st.selectbox("Select FX Pair", PAIRS)
    days = st.slider("Days of data", 7, 90, 30)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    lookback = st.number_input("Lookback periods", min_value=1, max_value=168, value=24)
    risk = st.slider("Risk multiplier", min_value=0.0, max_value=1.0, value=0.8)

    # Fetch data
    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
        st.stop()

    st.subheader(f"Actual Close Price for {pair}")
    st.line_chart(df["close"])

    # Load model params
    params = load_params()

    # Prediction loop
    times, actuals, preds = [], [], []
    for t in range(lookback, len(df) - 1):
        window = df.iloc[t - lookback : t + 1]
        x = compute_raw_features(window)
        q_out = float(qnode(params, x))
        last_price = float(window["close"].iloc[-1])

        # Robust volatility
        try:
            sigma_val = window["close"].pct_change().std()
            sigma = float(sigma_val)
            if math.isnan(sigma):
                sigma = 0.0
        except Exception:
            sigma = 0.0

        r_hat = q_out * sigma * risk
        pred_price = last_price * (1 + r_hat)

        times.append(df.index[t + 1])
        actuals.append(float(df["close"].iloc[t + 1]))
        preds.append(pred_price)

            # Build DataFrame for chart
    chart_df2 = pd.DataFrame({"Actual": actuals, "Predicted": preds}, index=times)
    st.subheader("Actual vs. QML-Predicted Next-Close")
    st.line_chart(chart_df2)
