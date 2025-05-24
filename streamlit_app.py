    # 4) Compute actual vs predicted using raw features
    import math
    times, actuals, preds = [], [], []
    for t in range(lookback, len(df) - 1):
        window = df.iloc[t - lookback : t + 1]
        x = compute_raw_features(window)
        q_out = float(qnode(params, x))
        last_price = float(window["close"].iloc[-1])

        # Robust volatility calculation
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

    # 5) Build DataFrame
    chart_df = pd.DataFrame({
        "time": times,
        "Actual": actuals,
        "Predicted": preds
    })
    melted = chart_df.melt(id_vars=["time"], var_name="Series", value_name="Price")
