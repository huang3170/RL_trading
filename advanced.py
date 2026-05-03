from __future__ import annotations

from typing import Tuple
import math

import numpy as np
import torch

from config import EnvConfig, TD3Config
from envs import EventDrivenMarketMakingEnv
from market_data import EventStream, MarketEvent
from replay import ReplayBuffer
from rewards import compute_reward
from state import StateBuilder


class EventType:
    """Event categories to include in state."""
    TRADE = 0       # market order / execution update
    LIMIT_ADD = 1   # new displayed liquidity added
    CANCEL = 2      # displayed liquidity removed without trade
    PRICE_MOVE = 3  # best bid/ask changed


def event_type_onehot(event_type: int) -> np.ndarray:
    x = np.zeros(4, dtype=np.float32)
    x[int(np.clip(event_type, 0, 3))] = 1.0
    return x


class AdvancedStateBuilder(StateBuilder):
    """
    Drop-in replacement for StateBuilder that adds:
    - dt between irregular events
    - one-hot event_type

    New market-state dimension:
        [spread, delta_mid, imbalance, signed_volume, volatility, RSI, dt,
         is_trade, is_limit_add, is_cancel, is_price_move]
    """

    @property
    def market_dim(self) -> int:
        return 11

    @property
    def full_dim(self) -> int:
        return self.agent_dim + self.market_dim

    def reset(self, event: MarketEvent):
        super().reset(event)
        self.last_timestamp = event.timestamp

    def build_market_state(self, event: MarketEvent) -> np.ndarray:
        base = super().build_market_state(event)  # original 6 features
        dt = max(0.0, float(event.timestamp - getattr(self, "last_timestamp", event.timestamp)))
        self.last_timestamp = event.timestamp
        etype = getattr(event, "event_type", EventType.TRADE)
        return np.concatenate([base, np.array([dt], dtype=np.float32), event_type_onehot(etype)]).astype(np.float32)


class QueuePositionTracker:
    """
    Industrial-style queue approximation for the agent's own limit orders.

    Main idea:
    - When we post at a price, volume already at that price is ahead of us.
    - TRADE consumes queue ahead first; after queue_ahead <= 0, our order fills.
    - CANCEL reduces queue ahead by a uniform-cancellation approximation,
      matching the paper's assumption that cancellations are uniformly distributed
      through the queue.
    """

    def __init__(self, order_size: int, tick_size: float):
        self.order_size = order_size
        self.tick_size = tick_size
        self.ask_price = None
        self.bid_price = None
        self.ask_ahead = 0.0
        self.bid_ahead = 0.0

    def reset(self):
        self.ask_price = None
        self.bid_price = None
        self.ask_ahead = 0.0
        self.bid_ahead = 0.0

    def post_quotes(self, event: MarketEvent, ask_price: float, bid_price: float):
        self.ask_price = ask_price
        self.bid_price = bid_price
        self.ask_ahead = self._volume_ahead(event, ask_price, side="ask")
        self.bid_ahead = self._volume_ahead(event, bid_price, side="bid")

    def update_and_fill(self, prev_event: MarketEvent, event: MarketEvent) -> Tuple[int, int]:
        event_type = getattr(event, "event_type", EventType.TRADE)
        matched_ask = 0
        matched_bid = 0

        if event_type == EventType.TRADE:
            vol = abs(float(event.signed_trade_volume))
            if event.signed_trade_volume > 0 and self.ask_price is not None:
                self.ask_ahead -= vol
                if self.ask_ahead <= 0:
                    matched_ask = self.order_size
            elif event.signed_trade_volume < 0 and self.bid_price is not None:
                self.bid_ahead -= vol
                if self.bid_ahead <= 0:
                    matched_bid = self.order_size

        elif event_type == EventType.CANCEL:
            prev_ask_depth = max(float(np.sum(prev_event.ask_volumes)), 1e-8)
            prev_bid_depth = max(float(np.sum(prev_event.bid_volumes)), 1e-8)
            ask_cancel = max(0.0, float(np.sum(prev_event.ask_volumes) - np.sum(event.ask_volumes)))
            bid_cancel = max(0.0, float(np.sum(prev_event.bid_volumes) - np.sum(event.bid_volumes)))
            self.ask_ahead -= ask_cancel * (self.ask_ahead / prev_ask_depth)
            self.bid_ahead -= bid_cancel * (self.bid_ahead / prev_bid_depth)

        elif event_type == EventType.PRICE_MOVE:
            if self.ask_price is not None and event.best_bid >= self.ask_price:
                matched_ask = self.order_size
            if self.bid_price is not None and event.best_ask <= self.bid_price:
                matched_bid = self.order_size

        elif event_type == EventType.LIMIT_ADD:
            # New displayed volume is usually behind our existing order if same price.
            pass

        self.ask_ahead = max(0.0, self.ask_ahead)
        self.bid_ahead = max(0.0, self.bid_ahead)
        return matched_ask, matched_bid

    def _volume_ahead(self, event: MarketEvent, price: float, side: str) -> float:
        # Synthetic price ladder from top-of-book. Replace with real level prices if available.
        levels = np.arange(len(event.ask_volumes), dtype=np.float32) * self.tick_size
        if side == "ask":
            ask_prices = event.best_ask + levels
            return float(np.sum(event.ask_volumes[ask_prices <= price]))
        bid_prices = event.best_bid - levels
        return float(np.sum(event.bid_volumes[bid_prices >= price]))


class SemiMDPReplayBuffer(ReplayBuffer):
    """
    Replay buffer with per-transition discount gamma_dt.

    In normal MDP RL, every step uses fixed gamma.
    In semi-MDP / continuous-time event-driven RL, event gaps are irregular, so use:
        gamma_dt = exp(-rho * delta_t)
    or equivalently:
        gamma_dt = gamma_base ** delta_t
    """

    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        super().__init__(env_cfg, rl_cfg)
        self.gamma_dt = np.ones((self.max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done, gamma_dt: float = 1.0):
        super().add(obs, action, reward, next_obs, done)
        self.gamma_dt[(self.ptr - 1) % self.max_size] = gamma_dt

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, size=batch_size)
        if self.is_lctc:
            obs = {k: torch.as_tensor(v[idx], device=device) for k, v in self.obs.items()}
            next_obs = {k: torch.as_tensor(v[idx], device=device) for k, v in self.next_obs.items()}
        else:
            obs = torch.as_tensor(self.obs[idx], device=device)
            next_obs = torch.as_tensor(self.next_obs[idx], device=device)
        return (
            obs,
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            next_obs,
            torch.as_tensor(self.dones[idx], device=device),
            torch.as_tensor(self.gamma_dt[idx], device=device),
        )


def semi_mdp_target(reward: torch.Tensor, done: torch.Tensor, gamma_dt: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """TD target for irregular event-time transitions."""
    return reward + (1.0 - done) * gamma_dt * target_q


class AdvancedEventDrivenMarketMakingEnv(EventDrivenMarketMakingEnv):
    """
    Advanced env combining all three upgrades:
    1. event_type in state
    2. semi-MDP delta_t / gamma_dt in info
    3. queue-position fill simulation
    """

    def __init__(self, stream: EventStream, cfg: EnvConfig):
        super().__init__(stream, cfg)
        self.state_builder = AdvancedStateBuilder(cfg)
        self.queue_tracker = QueuePositionTracker(cfg.order_size, cfg.tick_size)

    def reset(self):
        obs = super().reset()
        self.queue_tracker.reset()
        return obs

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        prev_event = self.stream.current()
        prev_mid = prev_event.mid
        self.theta_ask, self.theta_bid, mo_frac = self._decode_action(action)
        self.active_ask_price, self.active_bid_price = self._quotes_from_action(prev_event)
        self.queue_tracker.post_quotes(prev_event, self.active_ask_price, self.active_bid_price)

        market_order_size = self._market_order_size(mo_frac)
        self.inventory += market_order_size

        event, self.done = self.stream.next()
        matched_ask, matched_bid = self.queue_tracker.update_and_fill(prev_event, event)

        self.inventory -= int(matched_ask)
        self.inventory += int(matched_bid)

        delta_mid = event.mid - prev_mid
        delta_t = max(0.0, float(event.timestamp - prev_event.timestamp))
        gamma_dt = math.pow(0.97, delta_t)

        reward_dict = compute_reward(
            reward_type=self.cfg.reward_type,
            matched_ask=matched_ask,
            matched_bid=matched_bid,
            ask_price=self.active_ask_price,
            bid_price=self.active_bid_price,
            mid_price=event.mid,
            inventory=self.inventory,
            delta_mid=delta_mid,
            eta=self.cfg.eta,
            inventory_penalty=self.cfg.inventory_penalty,
        )

        obs_raw = self.state_builder.build(event, self.inventory, self.theta_ask, self.theta_bid)
        obs = self.stacker.push(obs_raw)
        self.state_builder.update_market_memory(event)

        info = {
            **reward_dict,
            "timestamp": event.timestamp,
            "delta_t": delta_t,
            "gamma_dt": gamma_dt,
            "event_type": getattr(event, "event_type", EventType.TRADE),
            "inventory": self.inventory,
            "ask_queue_ahead": self.queue_tracker.ask_ahead,
            "bid_queue_ahead": self.queue_tracker.bid_ahead,
            "matched_ask": matched_ask,
            "matched_bid": matched_bid,
        }
        return obs, float(reward_dict["reward"]), self.done, info
