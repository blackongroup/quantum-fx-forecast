# streamlit_app.py
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
    # User selects pair and parameters
    pair = st.selectbox("Select FX Pair", PAIRS)
    days = st.slider("Days of data", 7, 90, 30)
    interval = st.selectbox("Interval", ["1h","4h","1d"], index=0)
    lookback = st.number_input("Lookback periods", min_value=1, max_value=168, value=24)

    # Fetch data
    df = fetch_ohlcv(pair, period=f"{days}d", interval=interval)
    if df.empty:
        st.error(f"No data for {pair}.")
    else:
        st.subheader(f"Actual Close Price for {pair}")
        st.line_chart(df["close"])

        # Generate predictions
        params = load_params()
        times, actuals, preds = [], [], []
        for t in range(lookback, len(df)-1):
            window = df.iloc[t-lookback:t+1]
            x = compute_features(window)
            q_out = qnode(params, x)
            actual_price = df["close"].iloc[t+1]
            predicted_price = df["close"].iloc[t] * (1 + q_out)
            times.append(df.index[t+1])
            actuals.append(actual_price)
            preds.append(predicted_price)

        # Plot comparison
        chart_df = pd.DataFrame({"Actual": actuals, "Predicted": preds}, index=times)
        st.subheader("Actual vs. QML-Predicted Next-Close")
        st.line_chart(chart_df)
