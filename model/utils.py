# model/utils.py

import json
import numpy as np

def save_params(params: np.ndarray, filename: str = "model/params.json"):
    with open(filename, "w") as f:
        json.dump(params.tolist(), f)


def load_params(filename: str = "model/params.json") -> np.ndarray:
    with open(filename, "r") as f:
        data = json.load(f)
    return np.array(data)
