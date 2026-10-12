"""Leakage-free, variable-length, physically timed plantar pass batching."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from apace.data.types import Trial


def collate_passes(trials, device='cpu', max_points: int = 384):
    """Return pressure [B,S,T], sensor layouts, target, valid sample mask and duration.

    Short sequences are right padded, with their true lengths recorded in `valid`.
    Very long sequences are *anti-aliased averaged* down to max_points to cap VRAM;
    original physical duration is preserved as a separate numeric feature. Padding
    cannot contaminate the final temporal pooling. No target/reference is input.
    """
    if not trials or max_points < 16:
        raise ValueError('nonempty trials and max_points >=16 required')
    n_sensors = trials[0].pressure.shape[0]
    if any(t.pressure.ndim != 2 or t.pressure.shape[0] != n_sensors or t.pressure.shape[1] < 8 for t in trials):
        raise ValueError('incompatible sensor count or invalid pass')
    prepared, lengths, seconds = [], [], []
    for t in trials:
        x = torch.as_tensor(t.pressure, dtype=torch.float32)
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite pressure: run data QC before batching')
        original_n = x.shape[-1]
        if original_n > max_points:
            x = F.adaptive_avg_pool1d(x.unsqueeze(0), max_points).squeeze(0)
        prepared.append(x)
        lengths.append(x.shape[-1])
        seconds.append((original_n - 1) * float(t.sample_interval_s))
    padded_len = max(lengths)
    data = torch.zeros(len(trials), n_sensors, padded_len, dtype=torch.float32)
    valid = torch.zeros(len(trials), padded_len, dtype=torch.bool)
    for i, x in enumerate(prepared):
        data[i, :, :x.shape[-1]] = x
        valid[i, :x.shape[-1]] = True
    coords = np.stack([t.coords for t in trials]).astype(np.float32)
    targets = np.stack([t.target for t in trials]).astype(np.float32)
    return (data.to(device), torch.from_numpy(coords).to(device),
            torch.from_numpy(targets).to(device), valid.to(device),
            torch.tensor(seconds, dtype=torch.float32, device=device))
