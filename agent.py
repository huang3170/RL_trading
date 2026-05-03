from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EnvConfig, TD3Config
from networks import Actor, Critic
from replay import ReplayBuffer


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
