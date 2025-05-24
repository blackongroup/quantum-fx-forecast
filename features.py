import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model.utils import save_scaler

def compute_raw_features(df: pd.DataFrame) -> np.ndarray:
    """
    Computes raw technical features for the QML model.
    """
    close = df["close"]
    f1 = close.pct_change(1).iloc[-1]
    f2 = close.pct_change(24).iloc[-1]
    f3 = (close.iloc[-1] - close.rolling(24).mean().iloc[-1]) / close.rolling(24).mean().iloc[-1]
    f4 = close.rolling(24).std().iloc[-1] / close.iloc[-1]
    return np.array([f1, f2, f3, f4], dtype=float)

def fit_and_save_scaler(df_list: list[pd.DataFrame]):
    """
    Fit a StandardScaler on a list of DataFrames, then save it to disk.
    Call this *once* after you have your training set prepared.
    """
    raw = np.vstack([compute_raw_features(df) for df in df_list])
    scaler = StandardScaler().fit(raw)
    save_scaler(scaler)
    return scaler

def compute_features(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Compute & scale features for inference.
    """
    raw = compute_raw_features(df).reshape(1, -1)
    return scaler.transform(raw)[0]

