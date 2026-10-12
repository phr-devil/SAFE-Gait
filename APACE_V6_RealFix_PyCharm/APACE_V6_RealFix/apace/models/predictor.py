from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProbabilisticPredictor(nn.Module):
    def __init__(self, in_dim: int, target_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, 64)
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.mean = nn.Linear(hidden_dim, target_dim)
        self.raw_scale = nn.Linear(hidden_dim, target_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean(h)
        scale = F.softplus(self.raw_scale(h)) + 1e-4
        return mean, scale


def gaussian_nll(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = scale.square().clamp_min(1e-8)
    return (0.5 * ((target - mean).square() / var + torch.log(var))).mean()
