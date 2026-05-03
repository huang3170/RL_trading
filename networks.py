from __future__ import annotations

import torch
import torch.nn as nn

from config import EnvConfig, TD3Config
from models import StateEncoder


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
