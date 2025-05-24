# executor_fx.py

import os

# Stub executor: prints allocations instead of placing real trades
# Once OANDA is approved and oandapyV20 is available, replace this stub with real API calls.

def execute(allocs: dict, prices: dict):
    """
    Stub execution function. Prints allocations and prices.
    Replace with broker API logic when ready.
    """
    print("--- Executing Trades Stub ---")
    for instrument, weight in allocs.items():
        price = prices.get(instrument)
        print(f"Instrument: {instrument}, Weight: {weight:.2%}, Price: {price}")
