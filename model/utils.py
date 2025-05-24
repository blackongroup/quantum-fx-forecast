# model/utils.py

import json
import numpy as np
import os

PARAMS_FILE_DEFAULT = "model/params.json"


def save_params(params: np.ndarray, filename: str = PARAMS_FILE_DEFAULT):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(params.tolist(), f)


def load_params(filename: str = PARAMS_FILE_DEFAULT) -> np.ndarray:
    """
    Load model parameters from JSON file. If file not found,
    generate random params, save them, and return.
    """
    if not os.path.exists(filename):
        # Generate default random parameters
        from model.qml import n_qubits
        params = np.random.randn(n_qubits)
        save_params(params, filename)
        return params
    # Load existing parameters
    with open(filename, "r") as f:
        data = json.load(f)
    return np.array(data)
