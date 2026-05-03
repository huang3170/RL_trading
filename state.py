from __future__ import annotations

from collections import deque

import numpy as np

from config import EnvConfig
from market_data import MarketEvent
from trading_types import StateRepresentation


class StateBuilder:
    """
    Builds three paper-aligned state representations.

    agent-state:
        inventory, normalized active ask distance, normalized active bid distance

    market-state:
        spread, mid-price move, book imbalance, signed volume, volatility, RSI

    full-state:
        concat(agent-state, market-state)

    lctc:
        returns a dict of {agent, market, full}; neural networks later encode each
        branch independently and combine them, mirroring the paper's LCTC idea.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.mid_history: deque[float] = deque(maxlen=128)
        self.return_history: deque[float] = deque(maxlen=128)
        self.signed_volume_history: deque[float] = deque(maxlen=128)

    @property
    def agent_dim(self) -> int:
        return 3

    @property
    def market_dim(self) -> int:
        return 6

    @property
    def full_dim(self) -> int:
        return self.agent_dim + self.market_dim

    def reset(self, event: MarketEvent):
        self.mid_history.clear()
        self.return_history.clear()
        self.signed_volume_history.clear()
        self.mid_history.append(event.mid)
        self.signed_volume_history.append(event.signed_trade_volume)

    def update_market_memory(self, event: MarketEvent):
        if self.mid_history:
            ret = event.mid - self.mid_history[-1]
            self.return_history.append(ret)
        self.mid_history.append(event.mid)
        self.signed_volume_history.append(event.signed_trade_volume)

    def build_agent_state(self, inventory: int, theta_ask: float, theta_bid: float) -> np.ndarray:
        inv_scale = max(abs(self.cfg.min_inventory), abs(self.cfg.max_inventory), 1)
        return np.array(
            [
                inventory / inv_scale,
                theta_ask / max(self.cfg.max_theta, 1e-8),
                theta_bid / max(self.cfg.max_theta, 1e-8),
            ],
            dtype=np.float32,
        )

    def build_market_state(self, event: MarketEvent) -> np.ndarray:
        spread = event.spread
        delta_mid = 0.0 if len(self.mid_history) == 0 else event.mid - self.mid_history[-1]

        bid_depth = float(np.sum(event.bid_volumes))
        ask_depth = float(np.sum(event.ask_volumes))
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-8)

        signed_volume = event.signed_trade_volume
        volatility = float(np.std(self.return_history)) if len(self.return_history) >= 2 else 0.0
        rsi = self._compute_rsi()

        return np.array([spread, delta_mid, imbalance, signed_volume, volatility, rsi], dtype=np.float32)

    def build(self, event: MarketEvent, inventory: int, theta_ask: float, theta_bid: float):
        agent = self.build_agent_state(inventory, theta_ask, theta_bid)
        market = self.build_market_state(event)
        full = np.concatenate([agent, market], axis=0).astype(np.float32)

        if self.cfg.state_representation == "agent":
            return agent
        if self.cfg.state_representation == "full":
            return full
        if self.cfg.state_representation == "lctc":
            return {"agent": agent, "market": market, "full": full}
        raise ValueError(f"Unknown representation: {self.cfg.state_representation}")

    def _compute_rsi(self, lookback: int = 14) -> float:
        if len(self.return_history) < 2:
            return 0.5
        rets = np.array(list(self.return_history)[-lookback:], dtype=np.float32)
        gains = np.maximum(rets, 0.0).mean()
        losses = np.maximum(-rets, 0.0).mean()
        if losses <= 1e-12:
            return 1.0
        rs = gains / losses
        return float(1.0 - 1.0 / (1.0 + rs))  # normalized to [0,1]


class ObservationStacker:
    """Maintains rolling event-window observations for MLP/Transformer."""

    def __init__(self, window_size: int, representation: StateRepresentation):
        self.window_size = window_size
        self.representation = representation
        self.buffer = deque(maxlen=window_size)

    def reset(self):
        self.buffer.clear()

    def push(self, obs):
        self.buffer.append(obs)
        return self.output()

    def output(self):
        if self.representation == "lctc":
            while len(self.buffer) < self.window_size:
                self.buffer.appendleft(self.buffer[0])
            keys = ["agent", "market", "full"]
            return {k: np.stack([b[k] for b in self.buffer], axis=0).astype(np.float32) for k in keys}
        else:
            while len(self.buffer) < self.window_size:
                self.buffer.appendleft(self.buffer[0])
            return np.stack(list(self.buffer), axis=0).astype(np.float32)
