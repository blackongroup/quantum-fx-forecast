# model/qml.py

import pennylane as qml
import numpy as np

# Number of features = number of qubits
n_qubits = 4

# Use default simulator; swap to 'qiskit.ibmq' backend when ready
dev = qml.device("default.qubit", wires=n_qubits)

# Feature map: angle encoding
@qml.qnode(dev)
def circuit(params, x=None):
    # Encode features
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    # Entangle
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    # Variational layer
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    # Measurement
    return qml.expval(qml.PauliZ(0))


def qnode(params, x):
    return circuit(params, x)
