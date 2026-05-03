from __future__ import annotations

import torch
import torch.nn as nn

from config import EnvConfig, TD3Config


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
