"""
Event-driven PyTorch RL framework for Spooner et al. (2018),
"Market Making via Reinforcement Learning".

This framework keeps the paper's three state representations:
1. agent-state
2. full-state
3. LCTC-style state decomposition

Modernization:
- original paper: tile coding + TD learning / SARSA
- this framework: PyTorch neural encoders + TD3 for continuous quoting

Important:
- The environment is event-driven: env.step(action) consumes exactly ONE market event.
- Market time advances according to event timestamps, not fixed time intervals.
- Replace the CSV schema / fill simulation with your actual LOB replay data as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
from collections import deque
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


StateRepresentation = Literal["agent", "full", "lctc"]
RewardType = Literal["pnl", "symmetric", "asymmetric"]
EncoderType = Literal["mlp", "transformer"]


# ============================================================
# 1. Config
# ============================================================

@dataclass
class EnvConfig:
    tick_size: float = 0.01
    lot_size: int = 1
    order_size: int = 1000
    min_inventory: int = -10_000
    max_inventory: int = 10_000
    max_theta: float = 5.0
    max_market_order_fraction: float = 1.0
    reward_type: RewardType = "asymmetric"
    eta: float = 0.6
    inventory_penalty: float = 0.0
    window_size: int = 32
    state_representation: StateRepresentation = "lctc"


@dataclass
class TD3Config:
    action_dim: int = 3  # theta_ask, theta_bid, market-order-fraction
    gamma: float = 0.97
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    replay_size: int = 1_000_000
    warmup_steps: int = 10_000
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1
    hidden_dim: int = 256
    encoder_type: EncoderType = "mlp"
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Market Event + Lightweight LOB
# ============================================================

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


# ============================================================
# 3. State Builder: agent / full / lctc
# ============================================================

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


# ============================================================
# 4. Event-driven Market-Making Env
# ============================================================

class EventDrivenMarketMakingEnv:
    """
    Event-driven env.

    One call to step(action) does:
    1. Decode action into ask/bid quotes and optional market-order fraction.
    2. Consume exactly one historical market event.
    3. Simulate fills caused by the event.
    4. Update inventory.
    5. Compute event-to-event reward.
    6. Return next event-state.
    """

    def __init__(self, stream: EventStream, cfg: EnvConfig):
        self.stream = stream
        self.cfg = cfg
        self.state_builder = StateBuilder(cfg)
        self.stacker = ObservationStacker(cfg.window_size, cfg.state_representation)
        self.inventory = 0
        self.theta_ask = 1.0
        self.theta_bid = 1.0
        self.active_ask_price: Optional[float] = None
        self.active_bid_price: Optional[float] = None
        self.done = False

    def reset(self):
        event = self.stream.reset()
        self.state_builder.reset(event)
        self.stacker.reset()
        self.inventory = 0
        self.theta_ask = 1.0
        self.theta_bid = 1.0
        self.active_ask_price = None
        self.active_bid_price = None
        self.done = False
        obs = self.state_builder.build(event, self.inventory, self.theta_ask, self.theta_bid)
        return self.stacker.push(obs)

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        prev_event = self.stream.current()
        prev_mid = prev_event.mid

        self.theta_ask, self.theta_bid, mo_frac = self._decode_action(action)
        self.active_ask_price, self.active_bid_price = self._quotes_from_action(prev_event)

        # Optional inventory-clearing market order.
        market_order_size = self._market_order_size(mo_frac)
        self.inventory += market_order_size

        # Consume exactly ONE market event: this is the event-driven transition.
        event, self.done = self.stream.next()

        matched_ask, matched_bid = self._simulate_fills(prev_event, event)

        # ask fill means we sell -> inventory decreases
        # bid fill means we buy -> inventory increases
        self.inventory -= int(matched_ask)
        self.inventory += int(matched_bid)

        delta_mid = event.mid - prev_mid
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

        # Enforce inventory constraints by penalizing and clipping.
        if self.inventory < self.cfg.min_inventory or self.inventory > self.cfg.max_inventory:
            reward_dict["reward"] -= abs(self.inventory) * 1e-3
            self.inventory = int(np.clip(self.inventory, self.cfg.min_inventory, self.cfg.max_inventory))

        obs_raw = self.state_builder.build(event, self.inventory, self.theta_ask, self.theta_bid)
        obs = self.stacker.push(obs_raw)
        self.state_builder.update_market_memory(event)

        info = {
            **reward_dict,
            "timestamp": event.timestamp,
            "mid": event.mid,
            "spread": event.spread,
            "inventory": self.inventory,
            "theta_ask": self.theta_ask,
            "theta_bid": self.theta_bid,
            "ask_price": self.active_ask_price,
            "bid_price": self.active_bid_price,
            "matched_ask": matched_ask,
            "matched_bid": matched_bid,
            "market_order_size": market_order_size,
        }
        return obs, float(reward_dict["reward"]), self.done, info

    def _decode_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        theta_ask = (action[0] + 1.0) * 0.5 * self.cfg.max_theta
        theta_bid = (action[1] + 1.0) * 0.5 * self.cfg.max_theta
        mo_frac = (action[2] + 1.0) * 0.5 * self.cfg.max_market_order_fraction
        return float(theta_ask), float(theta_bid), float(mo_frac)

    def _quotes_from_action(self, event: MarketEvent) -> Tuple[float, float]:
        spread_scale = max(self.cfg.tick_size, event.spread / 2.0)
        ask = event.mid + self.theta_ask * spread_scale
        bid = event.mid - self.theta_bid * spread_scale
        return self._round_to_tick(ask), self._round_to_tick(bid)

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.cfg.tick_size) * self.cfg.tick_size

    def _market_order_size(self, mo_frac: float) -> int:
        if self.inventory == 0:
            return 0
        raw = -mo_frac * self.inventory
        size = int(round(raw / self.cfg.lot_size) * self.cfg.lot_size)
        return size

    def _simulate_fills(self, prev_event: MarketEvent, event: MarketEvent) -> Tuple[int, int]:
        """
        Simplified event-driven fill logic.

        Replace this with queue-position tracking if your data has enough fields.
        Logic:
        - ask quote fills if market best bid moves up through our ask, or buy pressure is large
        - bid quote fills if market best ask moves down through our bid, or sell pressure is large
        """
        matched_ask = 0
        matched_bid = 0

        if self.active_ask_price is not None:
            if event.best_bid >= self.active_ask_price or event.signed_trade_volume > 0:
                # Fill probability decreases when quote is deeper from mid.
                dist = max(self.active_ask_price - prev_event.mid, 0.0)
                p = math.exp(-dist / max(prev_event.spread, self.cfg.tick_size))
                if random.random() < min(1.0, p):
                    matched_ask = self.cfg.order_size

        if self.active_bid_price is not None:
            if event.best_ask <= self.active_bid_price or event.signed_trade_volume < 0:
                dist = max(prev_event.mid - self.active_bid_price, 0.0)
                p = math.exp(-dist / max(prev_event.spread, self.cfg.tick_size))
                if random.random() < min(1.0, p):
                    matched_bid = self.cfg.order_size

        return matched_ask, matched_bid


# ============================================================
# 5. Paper reward functions
# ============================================================

def compute_reward(
    reward_type: RewardType,
    matched_ask: float,
    matched_bid: float,
    ask_price: float,
    bid_price: float,
    mid_price: float,
    inventory: float,
    delta_mid: float,
    eta: float,
    inventory_penalty: float = 0.0,
) -> Dict[str, float]:
    psi_a = matched_ask * (ask_price - mid_price)
    psi_b = matched_bid * (mid_price - bid_price)
    spread_capture = psi_a + psi_b
    inventory_pnl = inventory * delta_mid
    raw_pnl = spread_capture + inventory_pnl

    if reward_type == "pnl":
        reward = raw_pnl
    elif reward_type == "symmetric":
        reward = raw_pnl - eta * inventory_pnl
    elif reward_type == "asymmetric":
        reward = raw_pnl - max(0.0, eta * inventory_pnl)
    else:
        raise ValueError(reward_type)

    reward -= inventory_penalty * inventory * inventory

    return {
        "reward": float(reward),
        "raw_pnl": float(raw_pnl),
        "spread_capture": float(spread_capture),
        "inventory_pnl": float(inventory_pnl),
        "psi_a": float(psi_a),
        "psi_b": float(psi_b),
    }


# ============================================================
# 6. Neural State Encoders
# ============================================================

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] or [B,D]
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, heads: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        t = x.size(1)
        z = self.proj(x) + self.pos[:, :t]
        z = self.encoder(z)
        return self.norm(z[:, -1])


def make_encoder(input_dim: int, rl_cfg: TD3Config):
    if rl_cfg.encoder_type == "mlp":
        return MLPEncoder(input_dim, rl_cfg.hidden_dim)
    return TransformerEncoder(
        input_dim=input_dim,
        hidden_dim=rl_cfg.hidden_dim,
        layers=rl_cfg.transformer_layers,
        heads=rl_cfg.transformer_heads,
        dropout=rl_cfg.dropout,
    )


class StateEncoder(nn.Module):
    """Supports agent, full, and neural LCTC representations."""

    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        super().__init__()
        self.representation = env_cfg.state_representation
        self.agent_dim = 3
        self.market_dim = 6
        self.full_dim = 9

        if self.representation == "agent":
            self.encoder = make_encoder(self.agent_dim, rl_cfg)
            self.output_dim = self.encoder.output_dim
        elif self.representation == "full":
            self.encoder = make_encoder(self.full_dim, rl_cfg)
            self.output_dim = self.encoder.output_dim
        elif self.representation == "lctc":
            self.agent_encoder = make_encoder(self.agent_dim, rl_cfg)
            self.market_encoder = make_encoder(self.market_dim, rl_cfg)
            self.full_encoder = make_encoder(self.full_dim, rl_cfg)
            # Paper uses fixed lambda = (0.6, 0.1, 0.3). Here we concatenate encodings,
            # and also keep learnable mixing weights.
            self.mix_logits = nn.Parameter(torch.tensor([0.6, 0.1, 0.3], dtype=torch.float32))
            self.output_dim = self.agent_encoder.output_dim
        else:
            raise ValueError(self.representation)

    def forward(self, obs) -> torch.Tensor:
        if self.representation in {"agent", "full"}:
            return self.encoder(obs)

        agent_z = self.agent_encoder(obs["agent"])
        market_z = self.market_encoder(obs["market"])
        full_z = self.full_encoder(obs["full"])
        weights = torch.softmax(self.mix_logits, dim=0)
        return weights[0] * agent_z + weights[1] * market_z + weights[2] * full_z


# ============================================================
# 7. TD3 Actor-Critic
# ============================================================

class Actor(nn.Module):
    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        super().__init__()
        self.encoder = StateEncoder(env_cfg, rl_cfg)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, rl_cfg.hidden_dim), nn.ReLU(),
            nn.Linear(rl_cfg.hidden_dim, rl_cfg.action_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.head(self.encoder(obs))


class Critic(nn.Module):
    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        super().__init__()
        self.encoder = StateEncoder(env_cfg, rl_cfg)
        self.q = nn.Sequential(
            nn.Linear(self.encoder.output_dim + rl_cfg.action_dim, rl_cfg.hidden_dim), nn.ReLU(),
            nn.Linear(rl_cfg.hidden_dim, rl_cfg.hidden_dim), nn.ReLU(),
            nn.Linear(rl_cfg.hidden_dim, 1),
        )

    def forward(self, obs, action):
        z = self.encoder(obs)
        return self.q(torch.cat([z, action], dim=-1))


# ============================================================
# 8. Replay Buffer supporting dict observations
# ============================================================

class ReplayBuffer:
    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        self.env_cfg = env_cfg
        self.rl_cfg = rl_cfg
        self.max_size = rl_cfg.replay_size
        self.ptr = 0
        self.size = 0
        self.is_lctc = env_cfg.state_representation == "lctc"
        t = env_cfg.window_size

        if self.is_lctc:
            self.obs = {
                "agent": np.zeros((self.max_size, t, 3), dtype=np.float32),
                "market": np.zeros((self.max_size, t, 6), dtype=np.float32),
                "full": np.zeros((self.max_size, t, 9), dtype=np.float32),
            }
            self.next_obs = {k: np.zeros_like(v) for k, v in self.obs.items()}
        else:
            d = 3 if env_cfg.state_representation == "agent" else 9
            self.obs = np.zeros((self.max_size, t, d), dtype=np.float32)
            self.next_obs = np.zeros((self.max_size, t, d), dtype=np.float32)

        self.actions = np.zeros((self.max_size, rl_cfg.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        if self.is_lctc:
            for k in self.obs:
                self.obs[k][self.ptr] = obs[k]
                self.next_obs[k][self.ptr] = next_obs[k]
        else:
            self.obs[self.ptr] = obs
            self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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
        )


class TD3Agent:
    def __init__(self, env_cfg: EnvConfig, rl_cfg: TD3Config):
        self.env_cfg = env_cfg
        self.rl_cfg = rl_cfg
        self.device = rl_cfg.device
        self.actor = Actor(env_cfg, rl_cfg).to(self.device)
        self.actor_target = Actor(env_cfg, rl_cfg).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(env_cfg, rl_cfg).to(self.device)
        self.critic2 = Critic(env_cfg, rl_cfg).to(self.device)
        self.critic1_target = Critic(env_cfg, rl_cfg).to(self.device)
        self.critic2_target = Critic(env_cfg, rl_cfg).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=rl_cfg.actor_lr)
        self.critic1_opt = torch.optim.AdamW(self.critic1.parameters(), lr=rl_cfg.critic_lr)
        self.critic2_opt = torch.optim.AdamW(self.critic2.parameters(), lr=rl_cfg.critic_lr)
        self.it = 0

    def _to_tensor_obs(self, obs):
        if isinstance(obs, dict):
            return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0) for k, v in obs.items()}
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def act(self, obs, noise_std: float = 0.0) -> np.ndarray:
        action = self.actor(self._to_tensor_obs(obs)).cpu().numpy()[0]
        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train_step(self, replay: ReplayBuffer) -> Dict[str, float]:
        self.it += 1
        cfg = self.rl_cfg
        obs, action, reward, next_obs, done = replay.sample(cfg.batch_size, cfg.device)

        with torch.no_grad():
            noise = torch.randn_like(action) * cfg.policy_noise
            noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(
                self.critic1_target(next_obs, next_action),
                self.critic2_target(next_obs, next_action),
            )
            target = reward + (1.0 - done) * cfg.gamma * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        loss1 = F.mse_loss(q1, target)
        loss2 = F.mse_loss(q2, target)

        self.critic1_opt.zero_grad(set_to_none=True)
        loss1.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        loss2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic2_opt.step()

        actor_loss_value = 0.0
        if self.it % cfg.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        return {"critic1_loss": float(loss1.item()), "critic2_loss": float(loss2.item()), "actor_loss": actor_loss_value}

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - self.rl_cfg.tau).add_(self.rl_cfg.tau * sp.data)


# ============================================================
# 9. Training loop
# ============================================================

def train(env: EventDrivenMarketMakingEnv, agent: TD3Agent, env_cfg: EnvConfig, rl_cfg: TD3Config, steps: int):
    replay = ReplayBuffer(env_cfg, rl_cfg)
    obs = env.reset()
    logs = []

    for step in range(1, steps + 1):
        if step < rl_cfg.warmup_steps:
            action = np.random.uniform(-1.0, 1.0, size=(rl_cfg.action_dim,)).astype(np.float32)
        else:
            action = agent.act(obs, noise_std=rl_cfg.exploration_noise)

        next_obs, reward, done, info = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if replay.size >= rl_cfg.batch_size and step >= rl_cfg.warmup_steps:
            metrics = agent.train_step(replay)
        else:
            metrics = {}

        if done:
            obs = env.reset()

        if step % 1000 == 0:
            row = {"step": step, "reward": reward, **info, **metrics}
            logs.append(row)
            print({k: row[k] for k in ["step", "reward", "raw_pnl", "inventory", "spread_capture"] if k in row})

    return pd.DataFrame(logs)


# ============================================================
# 10. Minimal synthetic example
# ============================================================

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


if __name__ == "__main__":
    env_cfg = EnvConfig(
        state_representation="lctc",  # choose: "agent", "full", "lctc"
        reward_type="asymmetric",
        eta=0.6,
        window_size=32,
    )
    rl_cfg = TD3Config(
        encoder_type="transformer",  # choose: "mlp" or "transformer"
        warmup_steps=5000,
        batch_size=256,
    )
    stream = EventStream(make_synthetic_events())
    env = EventDrivenMarketMakingEnv(stream, env_cfg)
    agent = TD3Agent(env_cfg, rl_cfg)
    train(env, agent, env_cfg, rl_cfg, steps=20_000)


# ============================================================
# 11. Advanced extensions: event type, semi-MDP, queue position
# ============================================================

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
