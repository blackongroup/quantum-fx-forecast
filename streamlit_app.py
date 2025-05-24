# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np

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
    days = st.slider("Days of data", min_value=7, max_value=365, value=365)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    lookback = st.number_input("Lookback periods", min_value=1, max_value=168, value=24)
    risk = st.slider("Risk multiplier", min_value=0.0, max_value=1.0, value=0.8)

    # Fetch data
    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
        st.stop()

    # 1) Plot raw close price
    st.subheader(f"Actual Close Price for {pair} ({days} days)")
    st.line_chart(df["close"])

    # 2) Load model parameters
    params = load_params()

    # 3) Compute actual vs predicted Next-Close
    times, actuals, preds = [], [], []
    for t in range(lookback, len(df) - 1):
        window = df.iloc[t - lookback : t + 1]
        x = compute_raw_features(window)
        q_out = float(qnode(params, x))
        last_price = float(window["close"].iloc[-1])

        # robust volatility calculation
        try:
            sigma = float(window["close"].pct_change().dropna().std())
        except Exception:
            sigma = 0.0

        r_hat = q_out * sigma * risk
        pred_price = last_price * (1 + r_hat)

        times.append(df.index[t + 1])
        actuals.append(float(df["close"].iloc[t + 1]))
        preds.append(pred_price)

    chart_df = pd.DataFrame({"Actual": actuals, "Predicted": preds}, index=times)

    # 4) Compute one-step future forecast
    last_window = df.iloc[-lookback:]
    x_last = compute_raw_features(last_window)
    q_last = float(qnode(params, x_last))
    last_price = float(last_window["close"].iloc[-1])
    try:
        sigma_last = float(last_window["close"].pct_change().dropna().std())
    except Exception:
        sigma_last = 0.0
    r_future = q_last * sigma_last * risk
    future_price = last_price * (1 + r_future)

    # infer future timestamp
    freq = pd.infer_freq(df.index) or interval
    future_time = df.index[-1] + pd.tseries.frequencies.to_offset(freq)

    # append future point
    future_row = pd.DataFrame(
        {"Actual": [np.nan], "Predicted": [future_price]},
        index=[future_time]
    )
    chart_ext = pd.concat([chart_df, future_row])

    # 5) Plot extended series
    st.subheader("Actual vs. QML-Predicted Next-Close (with Future Forecast)")
    st.line_chart(chart_ext)
