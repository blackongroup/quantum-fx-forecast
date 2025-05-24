import json
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

PARAMS_PATH = "model/params.json"
SCALER_PATH = "model/scaler.pkl"

def save_params(params: np.ndarray, filename: str = PARAMS_PATH):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(params.tolist(), f)

def load_params(filename: str = PARAMS_PATH) -> np.ndarray:
    # if missing, create random defaults
    if not os.path.exists(filename):
        from model.qml import n_qubits
        p = np.random.randn(n_qubits)
        save_params(p, filename)
        return p
    with open(filename) as f:
        return np.array(json.load(f))

def save_scaler(scaler: StandardScaler, filename: str = SCALER_PATH):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(scaler, filename)

def load_scaler(filename: str = SCALER_PATH) -> StandardScaler:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No scaler at {filename}; run training first.")
    return joblib.load(filename)

