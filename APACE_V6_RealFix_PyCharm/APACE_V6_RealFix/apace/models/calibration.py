from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CalibrationBeliefEncoder(nn.Module):
    """DeepSets encoder for variable-length calibration contexts."""

    def __init__(self, embedding_dim: int, target_dim: int, n_actions: int, belief_dim: int = 64, action_dim: int = 16) -> None:
        super().__init__()
        self.belief_dim = belief_dim
        self.action_emb = nn.Embedding(n_actions, action_dim)
        in_dim = embedding_dim + target_dim + action_dim
        self.item = nn.Sequential(nn.Linear(in_dim, belief_dim), nn.GELU(), nn.Linear(belief_dim, belief_dim), nn.GELU())
        self.post = nn.Sequential(nn.Linear(belief_dim, belief_dim), nn.LayerNorm(belief_dim), nn.GELU())

    def empty(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.belief_dim, device=device)

    def forward(
        self,
        embeddings: torch.Tensor,
        residuals: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must be [batch, context, dim]")
        a = self.action_emb(actions)
        item = self.item(torch.cat([embeddings, residuals, a], dim=-1))
        m = mask.unsqueeze(-1).to(item.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (item * m).sum(dim=1) / denom
        out = self.post(pooled)
        empty_rows = (mask.sum(dim=1) == 0).unsqueeze(-1)
        return torch.where(empty_rows, torch.zeros_like(out), out)


class PersonalizedPredictor(nn.Module):
    def __init__(self, embedding_dim: int, belief_dim: int, target_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.film = nn.Linear(belief_dim, embedding_dim * 2)
        self.head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, target_dim * 2))

    def forward(self, embedding: torch.Tensor, belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a residual correction distribution."""
        gamma, beta = self.film(belief).chunk(2, dim=-1)
        h = embedding * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        out = self.head(h)
        correction, raw_scale = out.chunk(2, dim=-1)
        return correction, F.softplus(raw_scale) + 1e-4

    def predict(
        self, embedding: torch.Tensor, belief: torch.Tensor, base_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction, scale = self.forward(embedding, belief)
        return base_mean + correction, scale
