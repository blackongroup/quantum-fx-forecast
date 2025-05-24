import numpy as np
import pandas as pd
from data.fetch_candles import fetch_ohlcv
from features import compute_raw_features, fit_and_save_scaler
from model.qml import circuit, n_qubits
from model.utils import save_params, load_scaler
from sklearn.model_selection import train_test_split

# 1) Build your dataset of (X, y)
pairs = ["EUR_USD","USD_JPY","GBP_USD"]  # or all 10
dfs = [fetch_ohlcv(p, period="90d", interval="1h") for p in pairs]

# Prepare feature matrix & target vector
X_raw, y = [], []
for df in dfs:
    for t in range(24, len(df)-1):
        window = df.iloc[t-24:t+1]
        raw = compute_raw_features(window)
        X_raw.append(raw)
        # target = next-close return
        ny = (df["close"].iloc[t+1] - df["close"].iloc[t]) / df["close"].iloc[t]
        y.append(ny)
X_raw = np.vstack(X_raw)
y = np.array(y)

# 2) Fit scaler & save
scaler = fit_and_save_scaler(dfs)

# 3) Scale features & split
X = scaler.transform(X_raw)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4) QML training loop
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer

dev = qml.device("default.qubit", wires=n_qubits)
opt = NesterovMomentumOptimizer(stepsize=0.1)
params = np.random.randn(n_qubits)

@qml.qnode(dev)
def qnode(p, x=None):
    # same as model/qml.py
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(p[i], wires=i)
    return qml.expval(qml.PauliZ(0))

# 5) Training
epochs = 50
for epoch in range(epochs):
    def cost(p):
        preds = [qnode(p, x) for x in X_train]
        return np.mean((np.array(preds) - y_train)**2)
    params, loss = opt.step_and_cost(cost, params)
    if epoch % 10 == 0:
        val_preds = [qnode(params, x) for x in X_val]
        val_loss = np.mean((np.array(val_preds) - y_val)**2)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

# 6) Save final parameters
save_params(params)
print("Training complete; params saved.")
