# backtest.py

import numpy as np
import pandas as pd
from data.fetch_candles import fetch_ohlcv
from features import compute_features
from model.qml import qnode
from model.utils import load_params


def run_backtest(pair: str = "EUR_USD"):
    df = fetch_ohlcv(pair, period="200d", interval="1h")
    params = load_params()
    pnl = []

    # Simulate predictions over the last half of data
    for t in range(1000, len(df)-1):
        window = df.iloc[t-1000:t+1]
        x = compute_features(window)
        pred = qnode(params, x)  # in [-1,1]
        signal = 1 if pred < 0 else -1
        ret = (df["close"].iloc[t+1] - df["close"].iloc[t]) / df["close"].iloc[t]
        pnl.append(signal * ret)

    cum = np.cumsum(pnl)
    ann = (1 + cum[-1]/1000)**(365*24) - 1
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()
    print(f"Annualized Return: {ann:.2%}")
    print(f"Max Drawdown: {dd:.2%}")

if __name__ == "__main__":
    run_backtest()
