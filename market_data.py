from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketEvent:
    """
    A normalized event record.

    You can map your real LOB data into this format.
    Expected minimal fields:
    - timestamp: event timestamp; irregular spacing is allowed
    - best_bid / best_ask
    - bid_volumes / ask_volumes: top-k queue volume arrays
    - signed_trade_volume: buy-initiated positive, sell-initiated negative
    """
    timestamp: float
    best_bid: float
    best_ask: float
    bid_volumes: np.ndarray
    ask_volumes: np.ndarray
    signed_trade_volume: float = 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


class EventStream:
    """Sequential replay of historical LOB events."""

    def __init__(self, events: List[MarketEvent]):
        if len(events) < 2:
            raise ValueError("Need at least two events.")
        self.events = events
        self.idx = 0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        depth: int = 5,
        time_col: str = "timestamp",
        best_bid_col: str = "best_bid",
        best_ask_col: str = "best_ask",
        signed_volume_col: str = "signed_trade_volume",
    ) -> "EventStream":
        events: List[MarketEvent] = []
        for _, row in df.iterrows():
            bid_vols = np.array([row.get(f"bid_vol_{i+1}", 0.0) for i in range(depth)], dtype=np.float32)
            ask_vols = np.array([row.get(f"ask_vol_{i+1}", 0.0) for i in range(depth)], dtype=np.float32)
            events.append(
                MarketEvent(
                    timestamp=float(row[time_col]),
                    best_bid=float(row[best_bid_col]),
                    best_ask=float(row[best_ask_col]),
                    bid_volumes=bid_vols,
                    ask_volumes=ask_vols,
                    signed_trade_volume=float(row.get(signed_volume_col, 0.0)),
                )
            )
        return cls(events)

    def reset(self) -> MarketEvent:
        self.idx = 0
        return self.events[self.idx]

    def current(self) -> MarketEvent:
        return self.events[self.idx]

    def next(self) -> Tuple[MarketEvent, bool]:
        self.idx += 1
        done = self.idx >= len(self.events) - 1
        return self.events[self.idx], done
