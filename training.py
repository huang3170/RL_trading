from __future__ import annotations

import numpy as np
import pandas as pd

from agent import TD3Agent
from config import EnvConfig, TD3Config
from envs import EventDrivenMarketMakingEnv
from replay import ReplayBuffer

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

        if step % 50 == 0:
            row = {"step": step, "reward": reward, **info, **metrics}
            logs.append(row)
            print({k: row[k] for k in ["step", "reward", "raw_pnl", "inventory", "spread_capture"] if k in row})

    return pd.DataFrame(logs)
