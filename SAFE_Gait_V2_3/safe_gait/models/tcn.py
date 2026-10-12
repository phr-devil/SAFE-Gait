from __future__ import annotations
import torch
from torch import nn


def _validate_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2 or mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[-1]:
        raise ValueError("mask must have shape [batch, time]")
    raw = mask.to(device=x.device)
    if raw.dtype != torch.bool:
        if not torch.isfinite(raw).all() or not torch.all((raw == 0) | (raw == 1)):
            raise ValueError("mask must be boolean or binary 0/1")
    mask = raw.to(dtype=torch.bool)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each sample must contain at least one valid time point")
    # This mask is a variable-length padding mask, not a generic missing-data mask.
    # Valid samples must therefore form a contiguous prefix: True...True, False...False.
    if mask.shape[1] > 1 and torch.any((~mask[:, :-1]) & mask[:, 1:]):
        raise ValueError("mask must be right-padded (valid True values must form a contiguous prefix)")
    return mask


def _mask_features(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    m = mask.unsqueeze(1)
    # torch.where is intentional: NaN * 0 remains NaN, while where removes
    # non-finite garbage that exists only in padded positions.
    return torch.where(m, x, torch.zeros((), dtype=x.dtype, device=x.device))


class TemporalBlock(nn.Module):
    """Mask-safe residual temporal block.

    Padded time points are re-zeroed after every convolutional stage so activations
    created inside padding cannot leak back into valid positions in deeper layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = _mask_features(x, mask)
        y = self.conv1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = _mask_features(y, mask)
        y = self.conv2(y)
        y = self.relu(y)
        y = _mask_features(y, mask)
        skip = _mask_features(self.skip(x), mask)
        return _mask_features(y + skip, mask)


class PopulationTCN(nn.Module):
    """Small mask-safe population backbone. Input: [batch, channels, time]."""

    def __init__(
        self,
        in_channels: int = 32,
        n_targets: int = 5,
        hidden: int = 64,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        if in_channels < 1 or n_targets < 1 or hidden < 1:
            raise ValueError("in_channels, n_targets, and hidden must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(in_channels, hidden, kernel_size=kernel_size, dilation=1, dropout=dropout),
                TemporalBlock(hidden, hidden, kernel_size=kernel_size, dilation=2, dropout=dropout),
                TemporalBlock(hidden, hidden, kernel_size=kernel_size, dilation=4, dropout=dropout),
            ]
        )
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, channels, time]")
        if x.shape[0] < 1 or x.shape[-1] < 1:
            raise ValueError("batch and time dimensions must be non-empty")
        mask = _validate_mask(x, mask)
        if mask is None:
            if not torch.isfinite(x).all():
                raise ValueError("x contains NaN/Inf")
        else:
            valid = mask.unsqueeze(1).expand_as(x)
            if not torch.isfinite(x[valid]).all():
                raise ValueError("valid (unmasked) x values contain NaN/Inf")
        h = _mask_features(x, mask)
        for block in self.blocks:
            h = block(h, mask)
        if mask is None:
            pooled = h.mean(dim=-1)
        else:
            m = mask.to(h.dtype).unsqueeze(1)
            denom = m.sum(dim=-1)
            pooled = (h * m).sum(dim=-1) / denom
        return self.head(pooled)
