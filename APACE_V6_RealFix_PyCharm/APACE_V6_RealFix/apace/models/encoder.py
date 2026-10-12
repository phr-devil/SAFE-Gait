"""Phase-aware causal-free temporal encoder with valid-length masking.

This is an offline pass-level regression model: noncausal convolutions are appropriate
for completed walks; it is not claimed to make instantaneous online predictions.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class PressureEncoder(nn.Module):
    """Separable strided TCN, 4 ordered gait-phase bins, coordinate/ID sensor pooling.

    pressure: B x S x T; valid: B x T True for nonpadding; duration_s: B.
    `coords` may be fixed sensor-identity features (WearGait) rather than physical XY.
    """
    def __init__(self, n_sensors: int, hidden_dim: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError('hidden_dim must be >= 8')
        self.n_sensors = n_sensors
        width = hidden_dim // 2
        self.stem = nn.Conv1d(1, width, 7, stride=2, padding=3)
        self.dw1 = nn.Conv1d(width, width, 5, stride=2, padding=2, groups=width)
        self.pw1 = nn.Conv1d(width, hidden_dim, 1)
        self.dw2 = nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2, groups=hidden_dim)
        self.pw2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.coord_mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.sensor_gate = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.out = nn.Sequential(nn.Linear(hidden_dim * 4 + 1, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, pressure: torch.Tensor, coords: torch.Tensor,
                valid: torch.Tensor | None = None,
                duration_s: torch.Tensor | None = None) -> torch.Tensor:
        if pressure.ndim != 3:
            raise ValueError('pressure must be [batch,sensors,time]')
        b, s, t = pressure.shape
        if s != self.n_sensors or t < 8:
            raise ValueError('sensor count mismatch or time too short')
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(b, -1, -1)
        if coords.shape != (b, s, 2):
            raise ValueError('coords must be [S,2] or [B,S,2]')
        if valid is None:
            valid = torch.ones(b, t, device=pressure.device, dtype=torch.bool)
        if valid.shape != (b, t) or not torch.all(valid[:, 0]):
            raise ValueError('valid mask must be [B,T] and start with True')
        if not torch.equal(valid, torch.arange(t, device=pressure.device).unsqueeze(0) < valid.sum(1, keepdim=True)):
            raise ValueError('valid mask must describe right padding only')
        if duration_s is None:
            duration_s = (valid.sum(dim=1).float() - 1) / 100.0
        if duration_s.shape != (b,) or not torch.all(torch.isfinite(duration_s) & (duration_s > 0)):
            raise ValueError('duration_s must be positive [B]')
        x = (pressure * valid[:, None, :]).reshape(b * s, 1, t)
        orig_len = valid.sum(dim=1)
        def mask_features(h, stage_len):
            positions = torch.arange(h.shape[-1], device=h.device).unsqueeze(0)
            stage_mask = positions < stage_len[:, None]
            stage_mask = stage_mask[:, None, :].expand(-1, s, -1).reshape(b * s, 1, -1)
            return h * stage_mask
        x = F.gelu(self.stem(x))
        x = mask_features(x, (orig_len + 1) // 2)
        x = F.gelu(self.pw1(self.dw1(x)))
        x = mask_features(x, (orig_len + 3) // 4)
        x = F.gelu(self.pw2(self.dw2(x)))
        x = mask_features(x, (orig_len + 7) // 8)
        # Three stride-2 convolutions: output length ceil(n / 8).
        length = (valid.sum(dim=1) + 7) // 8
        tt = x.shape[-1]
        bins = []
        for phase in range(4):
            starts = torch.div(length * phase, 4, rounding_mode='floor')
            ends = torch.div(length * (phase + 1) + 3, 4, rounding_mode='floor')
            idx = torch.arange(tt, device=x.device).unsqueeze(0)
            mask = ((idx >= starts[:, None]) & (idx < ends[:, None]) & (idx < length[:, None]))
            mask = mask[:, None, :].expand(-1, s, -1).reshape(b * s, 1, tt)
            pooled = (x * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            bins.append(pooled.reshape(b, s, -1))
        phase_tensor = torch.stack(bins, dim=2)  # B,S,4,H
        phase_tensor = phase_tensor + self.coord_mlp(coords).unsqueeze(2)
        h = phase_tensor.reshape(b, s, -1)
        weights = torch.softmax(self.sensor_gate(h).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        return self.out(torch.cat([pooled, torch.log1p(duration_s).unsqueeze(-1)], dim=-1))
