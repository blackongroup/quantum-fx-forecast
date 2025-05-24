# features.py

import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Input: df with columns ['open','high','low','close'] indexed by time.
    Output: feature array x of shape (F,).
    """
    close = df["close"]
    # Example technical features
    ret_1h  = close.pct_change(1).iloc[-1]
    ret_24h = close.pct_change(24).iloc[-1]
    sma_24  = close.rolling(24).mean().iloc[-1]
    std_24  = close.rolling(24).std().iloc[-1]
    # Normalize features roughly
    x = np.array([
        ret_1h,
        ret_24h,
        (close.iloc[-1] - sma_24) / sma_24,
        std_24 / close.iloc[-1]
    ])
    return x
