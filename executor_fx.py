# executor_fx.py

import os
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate

CLIENT = API(access_token=os.getenv("OANDA_TOKEN"))
ACCOUNT = os.getenv("OANDA_ACCOUNT")

def execute(allocs: dict, prices: dict):
    total = sum(prices.values())
    for p, w in allocs.items():
        notional = total * w
        units = int(notional / prices[p])
        order = {
            "order": {
                "instrument": p,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "REDUCE_ONLY"
            }
        }
        CLIENT.request(OrderCreate(ACCOUNT, data=order))
