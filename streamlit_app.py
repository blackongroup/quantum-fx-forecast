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

# --- Page Configuration ---
st.set_page_config(page_title="Quantum FX Forecast & Trading", layout="centered")
st.title("🚀 Quantum FX Forecast & Trading")

# --- Backtest & Trade Controls ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Run Backtest"):
        st.info("Running backtest, please wait…")
        run_backtest()
        st.success("Backtest complete. Check logs for details.")
with col2:
    if st.button("Run Forecast & Trade"):
        st.info("Executing forecast & trade stub…")
        try:
            forecast_and_trade()
            st.success("Forecast executed. Check logs for stub output.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.header("🔮 Price Prediction & Multi-Day Forecast")

if st.checkbox("Show Price Prediction Chart"):
    # 1) User Controls
    pair     = st.selectbox("Select FX Pair", PAIRS)
    days     = st.slider("Days of data to load", min_value=7, max_value=365, value=365)
    interval = st.selectbox("Data Interval", ["1h","4h","1d"], index=0)
    lookback = st.number_input("Lookback periods (bars)", min_value=1, max_value=168, value=24)
    risk     = st.slider("Risk multiplier", min_value=0.0, max_value=1.0, value=0.8)
    N_steps  = st.number_input("Forecast horizon (# steps ahead)", min_value=1, max_value=30, value=7)

    # 2) Fetch Data
    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
        st.stop()

    # 3) Historical Close Price Chart
    st.subheader(f"Historical Close Price ({pair}) - Last {days} days")
    st.line_chart(df["close"])

    # 4) Load QML Model Parameters
    params = load_params()

    # 5) One-Step Next-Close Predictions
    times, actuals, preds = [], [], []
    for t in range(lookback, len(df) - 1):
        window = df.iloc[t - lookback : t + 1]
        x = compute_raw_features(window)
        q_out = float(qnode(params, x))
        last_price = float(window["close"].iloc[-1])

        # robust volatility calculation
        vol_series = window["close"].pct_change().dropna()
        try:
            sigma = float(vol_series.std())
        except Exception:
            sigma = 0.0

        r_hat = q_out * sigma * risk
        pred_price = last_price * (1 + r_hat)

        times.append(df.index[t + 1])
        actuals.append(float(df["close"].iloc[t + 1]))
        preds.append(pred_price)

    # Base DataFrame of one-step predictions
    chart_df = pd.DataFrame({"Actual": actuals, "Predicted": preds}, index=times)

    # 6) Multi-Step Walk-Forward Forecast
    ext_df = chart_df.copy()
    current_prices = df[["close"]].copy()
    freq = pd.infer_freq(df.index) or interval

    for _ in range(N_steps):
        window = current_prices.iloc[-lookback:]
        x = compute_raw_features(window)
        q_out = float(qnode(params, x))
        last_p = float(window["close"].iloc[-1])

        vol_series = window["close"].pct_change().dropna()
        try:
            sigma = float(vol_series.std())
        except Exception:
            sigma = 0.0

        r_hat = q_out * sigma * risk
        next_p = last_p * (1 + r_hat)

        next_t = current_prices.index[-1] + pd.tseries.frequencies.to_offset(freq)
        ext_df.loc[next_t] = [np.nan, next_p]
        current_prices.loc[next_t] = next_p

    # 7) Plot Predictions & Forecast
    st.subheader(f"Actual vs. QML-Predicted Next-Close + {N_steps}-Step Forecast")
    st.line_chart(ext_df)
