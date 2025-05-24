# forecast_and_trade.py

import os
import numpy as np
from data.fetch_candles import fetch_ohlcv
from features import compute_features
from model.utils import load_params, load_scaler
from model.qml import qnode
from executor_fx import execute

# Your 10 FX pairs
PAIRS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "NZD_USD", "USD_CAD", "EUR_GBP",
    "EUR_JPY", "GBP_JPY"
]

def forecast_and_trade(risk: float = 0.8,
                       period: str = "7d",
                       interval: str = "1h"):
    """
    Generates QML-driven return forecasts, converts them
    into allocations, and executes (stub) trades.
    """
    # 1) Load model & scaler
    params = load_params()
    scaler = load_scaler()

    # 2) Gather signals
    raw_signals = {}
    prices = {}
    for p in PAIRS:
        df = fetch_ohlcv(p, period=period, interval=interval)
        if df.empty or "close" not in df.columns:
            continue

        # 3) Compute normalized feature vector
        x = compute_features(df, scaler)  # shape (F,)

        # 4) QML prediction [-1,1]
        q_out = float(qnode(params, x))

        # 5) Calibrate by realized volatility
        sigma = df["close"].pct_change().std()
        r_hat = q_out * sigma * risk

        # 6) Use positive forecasts only (no shorts)
        raw_signals[p] = max(r_hat, 0.0)

        # 7) Store last price for order sizing
        prices[p] = float(df["close"].iloc[-1])

    # 8) Normalize into weights summing to 1
    total = sum(raw_signals.values()) or 1.0
    allocations = {p: raw_signals[p] / total for p in raw_signals}

    # 9) Execute the stub (prints out orders)
    execute(allocations, prices)

if __name__ == "__main__":
    forecast_and_trade()
