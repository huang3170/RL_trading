from __future__ import annotations

from typing import List

import numpy as np

from market_data import MarketEvent


def make_synthetic_events(n: int = 50_000, depth: int = 5) -> List[MarketEvent]:
    mid = 100.0
    events = []
    ts = 0.0
    for _ in range(n):
        ts += np.random.exponential(scale=0.01)  # irregular event time
        mid += np.random.normal(0.0, 0.005)
        spread = max(0.01, abs(np.random.normal(0.02, 0.005)))
        bid = mid - spread / 2
        ask = mid + spread / 2
        bid_vols = np.random.randint(100, 5000, size=depth).astype(np.float32)
        ask_vols = np.random.randint(100, 5000, size=depth).astype(np.float32)
        signed_vol = np.random.normal(0.0, 1000.0)
        events.append(MarketEvent(ts, bid, ask, bid_vols, ask_vols, signed_vol))
    return events
