from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from advanced import AdvancedEventDrivenMarketMakingEnv, EventType
from config import EnvConfig
from market_data import EventStream, MarketEvent


@dataclass
class BarRecord:
    """
    15-minute OHLCV bar.

    Required columns in your dataframe:
        timestamp, open, high, low, close, volume

    Optional columns:
        bid, ask, spread
    """
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None


class BarToPseudoLOBReconstructor:
    """
    Converts low-frequency OHLCV bars into a synthetic event stream.

    This is NOT true LOB reconstruction. It is a controlled approximation for cases
    where only 15-minute data is available.

    Main assumptions:
    1. Within each bar, price follows a synthetic path through O/H/L/C.
    2. Bar volume is split across synthetic events.
    3. Bid/ask spread is estimated from provided spread, bid/ask, or volatility proxy.
    4. Top-5 depth is generated from volume and spread proxies.
    5. Event types are inferred from price and volume changes.

    Output:
        List[MarketEvent]
    which can be passed directly to EventStream(events).
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        depth: int = 5,
        events_per_bar: int = 8,
        base_depth_fraction: float = 0.08,
        seed: int = 42,
    ):
        self.tick_size = tick_size
        self.depth = depth
        self.events_per_bar = max(events_per_bar, 4)
        self.base_depth_fraction = base_depth_fraction
        self.rng = np.random.default_rng(seed)

    def from_dataframe(self, df: pd.DataFrame) -> EventStream:
        bars = []
        for _, row in df.iterrows():
            bars.append(
                BarRecord(
                    timestamp=float(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    bid=float(row["bid"]) if "bid" in row and not pd.isna(row["bid"]) else None,
                    ask=float(row["ask"]) if "ask" in row and not pd.isna(row["ask"]) else None,
                    spread=float(row["spread"]) if "spread" in row and not pd.isna(row["spread"]) else None,
                )
            )
        return EventStream(self.reconstruct(bars))

    def reconstruct(self, bars: List[BarRecord]) -> List[MarketEvent]:
        events: List[MarketEvent] = []
        prev_mid: Optional[float] = None

        for i, bar in enumerate(bars):
            mids = self._intrabar_price_path(bar, previous_close=bars[i - 1].close if i > 0 else None)
            n = len(mids)
            timestamps = np.linspace(bar.timestamp, bar.timestamp + 15 * 60, n, endpoint=False)
            volume_slices = self._split_volume(bar.volume, n)

            for j, mid in enumerate(mids):
                spread = self._estimate_spread(bar, mid)
                best_bid = self._round_to_tick(mid - spread / 2)
                best_ask = self._round_to_tick(mid + spread / 2)
                if best_ask <= best_bid:
                    best_ask = best_bid + self.tick_size

                signed_volume = self._signed_volume(mid, prev_mid, volume_slices[j])
                event_type = self._infer_event_type(mid, prev_mid, signed_volume, j)
                bid_depth, ask_depth = self._synthetic_depth(bar, spread, signed_volume)

                events.append(
                    MarketEvent(
                        timestamp=float(timestamps[j]),
                        best_bid=float(best_bid),
                        best_ask=float(best_ask),
                        bid_volumes=bid_depth,
                        ask_volumes=ask_depth,
                        signed_trade_volume=float(signed_volume),
                    )
                )
                # Attach event_type dynamically for compatibility with existing MarketEvent dataclass.
                setattr(events[-1], "event_type", int(event_type))
                prev_mid = mid

        if len(events) < 2:
            raise ValueError("Reconstruction produced fewer than two events.")
        return events

    def _intrabar_price_path(self, bar: BarRecord, previous_close: Optional[float]) -> np.ndarray:
        """
        Creates a plausible O-H-L-C or O-L-H-C path.
        Direction is randomized but biased by close-open.
        """
        if bar.close >= bar.open:
            anchors = [bar.open, bar.low, bar.high, bar.close] if self.rng.random() < 0.35 else [bar.open, bar.high, bar.low, bar.close]
        else:
            anchors = [bar.open, bar.high, bar.low, bar.close] if self.rng.random() < 0.35 else [bar.open, bar.low, bar.high, bar.close]

        if previous_close is not None:
            anchors[0] = bar.open

        segments = len(anchors) - 1
        points_per_segment = max(2, self.events_per_bar // segments)
        path = []
        for a, b in zip(anchors[:-1], anchors[1:]):
            seg = np.linspace(a, b, points_per_segment, endpoint=False)
            path.extend(seg.tolist())
        path.append(bar.close)

        path = np.array(path, dtype=np.float32)
        if len(path) > self.events_per_bar:
            idx = np.linspace(0, len(path) - 1, self.events_per_bar).astype(int)
            path = path[idx]
        elif len(path) < self.events_per_bar:
            pad = np.full(self.events_per_bar - len(path), bar.close, dtype=np.float32)
            path = np.concatenate([path, pad])

        # Add tiny bridge noise but keep inside high-low range.
        noise_scale = max((bar.high - bar.low) * 0.02, self.tick_size)
        path += self.rng.normal(0.0, noise_scale, size=path.shape)
        path = np.clip(path, bar.low, bar.high)
        return path

    def _split_volume(self, volume: float, n: int) -> np.ndarray:
        weights = self.rng.dirichlet(np.ones(n))
        return volume * weights

    def _estimate_spread(self, bar: BarRecord, mid: float) -> float:
        if bar.spread is not None and bar.spread > 0:
            return max(self.tick_size, self._round_to_tick(bar.spread))
        if bar.bid is not None and bar.ask is not None and bar.ask > bar.bid:
            return max(self.tick_size, self._round_to_tick(bar.ask - bar.bid))

        # Volatility/range proxy. Wider bars imply wider synthetic spread.
        range_spread = 0.02 * max(bar.high - bar.low, self.tick_size)
        price_spread = 0.0001 * max(abs(mid), 1.0)
        return max(self.tick_size, self._round_to_tick(max(range_spread, price_spread)))

    def _signed_volume(self, mid: float, prev_mid: Optional[float], volume: float) -> float:
        if prev_mid is None:
            return 0.0
        if mid > prev_mid:
            return abs(volume)
        if mid < prev_mid:
            return -abs(volume)
        return float(self.rng.choice([-1.0, 1.0]) * abs(volume) * 0.2)

    def _infer_event_type(self, mid: float, prev_mid: Optional[float], signed_volume: float, idx_in_bar: int) -> int:
        if prev_mid is not None and abs(mid - prev_mid) >= self.tick_size:
            return EventType.PRICE_MOVE
        if abs(signed_volume) > 0:
            return EventType.TRADE
        return self.rng.choice([EventType.LIMIT_ADD, EventType.CANCEL])

    def _synthetic_depth(self, bar: BarRecord, spread: float, signed_volume: float) -> Tuple[np.ndarray, np.ndarray]:
        base = max(bar.volume * self.base_depth_fraction, 1.0)
        decay = np.exp(-np.arange(self.depth) / 1.5)
        raw = base * decay

        # Order-flow imbalance: buy pressure reduces ask depth and increases bid depth.
        pressure = np.tanh(signed_volume / max(bar.volume, 1.0))
        bid_scale = 1.0 + 0.5 * pressure
        ask_scale = 1.0 - 0.5 * pressure

        bid = raw * bid_scale * self.rng.lognormal(mean=0.0, sigma=0.25, size=self.depth)
        ask = raw * ask_scale * self.rng.lognormal(mean=0.0, sigma=0.25, size=self.depth)
        return bid.astype(np.float32), ask.astype(np.float32)

    def _round_to_tick(self, x: float) -> float:
        return round(x / self.tick_size) * self.tick_size


class FifteenMinuteBarMarketMakingEnv(AdvancedEventDrivenMarketMakingEnv):
    """
    Convenience wrapper:
    - Input: 15-minute OHLCV dataframe
    - Internally reconstructs pseudo-LOB event stream
    - Then uses the same advanced market-making env

    This lets you keep the existing RL code while using 15-minute historical data.
    """

    def __init__(
        self,
        bar_df: pd.DataFrame,
        cfg: EnvConfig,
        events_per_bar: int = 8,
        depth: int = 5,
    ):
        recon = BarToPseudoLOBReconstructor(
            tick_size=cfg.tick_size,
            depth=depth,
            events_per_bar=events_per_bar,
        )
        stream = recon.from_dataframe(bar_df)
        super().__init__(stream, cfg)
