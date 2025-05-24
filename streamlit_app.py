import streamlit as st
from backtest import run_backtest
from forecast_and_trade import forecast_and_trade

st.set_page_config(page_title="Quantum FX Forecast", layout="centered")
st.title("🚀 Quantum FX Forecast & Trading")

if st.button("Run Backtest"):
    st.info("Running backtest, please wait…")
    run_backtest()
    st.success("Backtest complete! Check your console or logs for results.")

if st.button("Run Forecast & Trade"):
    st.info("Executing forecast and trades…")
    try:
        forecast_and_trade()
        st.success("Forecast executed and trades placed.")
    except Exception as e:
        st.error(f"Error during execution: {e}")
