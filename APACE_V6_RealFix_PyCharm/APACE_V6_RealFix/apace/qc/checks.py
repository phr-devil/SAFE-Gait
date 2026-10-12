from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from apace.data.types import Trial


@dataclass(frozen=True)
class QCResult:
    ok: bool
    reasons: dict[str, float | str]


class DataQualityGuard:
    def __init__(
        self,
        max_nan_fraction: float = 0.0,
        max_flat_fraction: float = 0.25,
        min_timepoints: int = 8,
        flat_std_threshold: float = 1e-8,
    ) -> None:
        self.max_nan_fraction = max_nan_fraction
        self.max_flat_fraction = max_flat_fraction
        self.min_timepoints = min_timepoints
        self.flat_std_threshold = flat_std_threshold

    def check_trial(self, trial: Trial) -> QCResult:
        reasons: dict[str, float | str] = {}
        x = np.asarray(trial.pressure)
        nan_fraction = 1.0 - float(np.isfinite(x).mean())
        if nan_fraction > self.max_nan_fraction:
            reasons["nan_fraction"] = nan_fraction
        if x.shape[1] < self.min_timepoints:
            reasons["timepoints"] = float(x.shape[1])
        safe = np.nan_to_num(x, copy=True)
        flat_fraction = float((np.std(safe, axis=1) <= self.flat_std_threshold).mean())
        if flat_fraction > self.max_flat_fraction:
            reasons["flat_fraction"] = flat_fraction
        if not np.all(np.isfinite(trial.target)):
            reasons["target_nonfinite"] = "target contains NaN/Inf"
        if not np.all(np.isfinite(trial.coords)):
            reasons["coords_nonfinite"] = "coords contains NaN/Inf"
        return QCResult(ok=not reasons, reasons=reasons)
