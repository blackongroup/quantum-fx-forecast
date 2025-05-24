# forecast_and_trade.py

import os
import numpy as np
from data.fetch_candles import fetch_ohlcv
from features import compute_features
from model.utils import load_params
from model.qml import qnode
from executor_fx import execute

PAIRS = [
    "EUR_USD","USD_JPY","GBP_USD","USD_CHF",
    "AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY"
]

params = load_params()


def forecast_and_trade():
    allocs = {}
    prices = {}

    for p in PAIRS:
        df = fetch_ohlcv(p, period="7d", interval="1h")
        x = compute_features(df)
        pred = qnode(params, x)  # in [-1,1]
        weight = (1 - pred) / 2    # map to [0,1]
        allocs[p] = float(weight)
        prices[p] = df["close"].iloc[-1]

    execute(allocs, prices)


if __name__ == "__main__":
    forecast_and_trade()
