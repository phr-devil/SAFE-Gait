from __future__ import annotations

import torch
from torch import nn

QUANTILES = torch.tensor([0.1, 0.5, 0.9])


class DistributionalValueModel(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64, action_dim: int = 16) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, action_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, cost_weight: torch.Tensor) -> torch.Tensor:
        if cost_weight.ndim == 1:
            cost_weight = cost_weight.unsqueeze(-1)
        raw = self.net(torch.cat([state, self.action_emb(action), cost_weight], dim=-1))
        q50 = raw[:, 1:2]
        low_gap = torch.nn.functional.softplus(raw[:, 0:1])
        high_gap = torch.nn.functional.softplus(raw[:, 2:3])
        return torch.cat([q50 - low_gap, q50, q50 + high_gap], dim=-1)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = QUANTILES.to(pred.device, pred.dtype).view(1, -1)
    err = target.view(-1, 1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()
