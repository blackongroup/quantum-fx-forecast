# streamlit_app.py

import os
import streamlit as st
import pandas as pd
from backtest import run_backtest
from forecast_and_trade import forecast_and_trade, PAIRS
from data.fetch_candles import fetch_ohlcv
from features import compute_features
from model.utils import load_params
from model.qml import qnode

# --- Page Config ---
st.set_page_config(page_title="Quantum FX Forecast & Trading", layout="centered")

# --- Title ---
st.title("🚀 Quantum FX Forecast & Trading")

# --- Backtest & Execution Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Run Backtest"):
        st.info("Running backtest, please wait…")
        run_backtest()
        st.success("Backtest complete! Check logs for details.")
with col2:
    if st.button("Run Forecast & Trade"):
        st.info("Executing forecast and trade stub…")
        try:
            forecast_and_trade()
            st.success("Forecast executed. Check logs for stub output.")
        except Exception as e:
            st.error(f"Error during execution: {e}")

# --- Price Prediction Chart ---
st.markdown("---")
st.header("🔮 Price Prediction Chart")
show_chart = st.checkbox("Show Price Prediction Chart")
if show_chart:
    pair = st.selectbox("Select FX Pair", PAIRS)
    days = st.slider("Days of data", 7, 90, 30)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    lookback = st.number_input("Lookback periods", min_value=1, max_value=168, value=24)

    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
    else:
        st.subheader(f"Actual Close Price for {pair}")
        st.line_chart(df["close"])

        params = load_params()
        times, actuals, preds = [], [], []
        for t in range(lookback, len(df) - 1):
            window = df.iloc[t - lookback : t + 1]
            x = compute_features(window)
            # Ensure qnode output is a Python float
            q_out = float(qnode(params, x))
            actual_price = float(df["close"].iloc[t + 1])
            predicted_price = float(df["close"].iloc[t] * (1 + q_out))
            times.append(df.index[t + 1])
            actuals.append(actual_price)
            preds.append(predicted_price)

        # Prepare DataFrame for charting
        chart_df = pd.DataFrame({"Actual": actuals, "Predicted": preds}, index=times)
        st.subheader("Actual vs. QML-Predicted Next-Close")
        st.line_chart(chart_df)
