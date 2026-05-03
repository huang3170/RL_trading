from __future__ import annotations

import numpy as np
import torch

from config import EnvConfig, TD3Config


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
