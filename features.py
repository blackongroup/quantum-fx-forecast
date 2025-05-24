# features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model.utils import save_scaler


def compute_raw_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute raw technical features for QML model, safely handling NaNs.
    """
    close = df["close"]
    # Compute features
    f1 = close.pct_change(1).iloc[-1]
    f2 = close.pct_change(24).iloc[-1]
    sma_24 = close.rolling(24).mean().iloc[-1]
    f3 = (close.iloc[-1] - sma_24) / sma_24
    f4 = close.rolling(24).std().iloc[-1] / close.iloc[-1]

    # Assemble and sanitize
    raw = np.array([f1, f2, f3, f4], dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def fit_and_save_scaler(dfs: list[pd.DataFrame]) -> StandardScaler:
    """
    Fit a StandardScaler over raw features from multiple DataFrames.
    """
    raw = np.vstack([compute_raw_features(df) for df in dfs])
    scaler = StandardScaler().fit(raw)
    save_scaler(scaler)
    return scaler


def compute_features(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Compute and scale features for inference.
    """
    # Compute and sanitize raw features
    raw = compute_raw_features(df)
    # Ensure shape (1, F)
    raw_2d = raw.reshape(1, -1)
    scaled = scaler.transform(raw_2d)
    return scaled.flatten()
