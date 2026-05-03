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

from agent import TD3Agent
from advanced import (
    AdvancedEventDrivenMarketMakingEnv,
    AdvancedStateBuilder,
    EventType,
    QueuePositionTracker,
    SemiMDPReplayBuffer,
    event_type_onehot,
    semi_mdp_target,
)
from bars import BarRecord, BarToPseudoLOBReconstructor, FifteenMinuteBarMarketMakingEnv
from config import EnvConfig, TD3Config
from envs import EventDrivenMarketMakingEnv
from market_data import EventStream, MarketEvent
from models import MLPEncoder, StateEncoder, TransformerEncoder, make_encoder
from networks import Actor, Critic
from replay import ReplayBuffer
from rewards import compute_reward
from state import ObservationStacker, StateBuilder
from synthetic import make_synthetic_events
from trading_types import EncoderType, RewardType, StateRepresentation
from training import train


__all__ = [
    "Actor",
    "AdvancedEventDrivenMarketMakingEnv",
    "AdvancedStateBuilder",
    "BarRecord",
    "BarToPseudoLOBReconstructor",
    "Critic",
    "EncoderType",
    "EnvConfig",
    "EventDrivenMarketMakingEnv",
    "EventStream",
    "EventType",
    "FifteenMinuteBarMarketMakingEnv",
    "MLPEncoder",
    "MarketEvent",
    "ObservationStacker",
    "QueuePositionTracker",
    "ReplayBuffer",
    "RewardType",
    "SemiMDPReplayBuffer",
    "StateBuilder",
    "StateEncoder",
    "StateRepresentation",
    "TD3Agent",
    "TD3Config",
    "TransformerEncoder",
    "compute_reward",
    "event_type_onehot",
    "make_encoder",
    "make_synthetic_events",
    "semi_mdp_target",
    "train",
]


# Example usage:
# df = pd.read_csv("your_15min_data.csv")
# env_cfg = EnvConfig(state_representation="lctc", reward_type="asymmetric", window_size=32)
# env = FifteenMinuteBarMarketMakingEnv(df, env_cfg, events_per_bar=8)
# agent = TD3Agent(env_cfg, TD3Config(encoder_type="transformer"))
# train(env, agent, env_cfg, TD3Config(), steps=100_000)

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
    logs = train(env, agent, env_cfg, rl_cfg, steps=20_000)
    logs.to_csv("train_logs.csv")
