from __future__ import annotations

from dataclasses import dataclass

import torch

from trading_types import EncoderType, RewardType, StateRepresentation


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
