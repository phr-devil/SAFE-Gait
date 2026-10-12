from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Trial:
    """One independent walking/reference trial.

    pressure: [n_sensors, n_time]
    coords: [n_sensors, 2]
    target: [n_targets]
    """
    subject_id: str
    trial_id: str
    condition: str
    pressure: np.ndarray
    coords: np.ndarray
    target: np.ndarray
    site: Optional[str] = None
    sample_interval_s: float = 0.01

    def __post_init__(self) -> None:
        if self.pressure.ndim != 2:
            raise ValueError("pressure must have shape [n_sensors, n_time]")
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape [n_sensors, 2]")
        if self.pressure.shape[0] != self.coords.shape[0]:
            raise ValueError("pressure sensor count must match coords")
        if not np.isfinite(self.sample_interval_s) or self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        if self.target.ndim != 1:
            raise ValueError("target must be a 1D vector")
