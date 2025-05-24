# data/fetch_candles.py

import pandas as pd
import yfinance as yf

def fetch_ohlcv(pair: str, period: str = "365d", interval: str = "1h") -> pd.DataFrame:
    """
    Fetches historical FX data from Yahoo Finance.
    Expects pair like 'EUR_USD'; constructs ticker 'EURUSD=X'.
    Returns a DataFrame indexed by timestamp with a 'close' column.
    """
    # Yahoo uses e.g. 'EURUSD=X' for EUR/USD
    yf_ticker = pair.replace("_", "") + "=X"
    df = yf.download(yf_ticker, period=period, interval=interval)
    if "Close" not in df.columns:
        raise ValueError(f"No data for {pair} via yfinance")
    df = df.rename(columns={"Close": "close"})
    return df[["close"]]

