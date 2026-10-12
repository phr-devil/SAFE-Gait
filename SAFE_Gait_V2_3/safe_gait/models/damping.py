from __future__ import annotations
import numpy as np


class GlobalDampingCalibrator:
    """Validation-only condition-pair × target damping.

    Fits gamma in [0,1] to minimize squared centered residual error:
        argmin_gamma sum_i (actual_query_z - gamma * predicted_correction_z)^2

    Inputs must already be in the SAME centered/standardized residual space.
    This calibrator is deliberately not subject-specific and must never be fit
    on final test subjects.
    """

    def __init__(self):
        self.gamma_: dict[tuple[str, str], np.ndarray] = {}

    def fit_pair(
        self,
        support_condition: str,
        query_condition: str,
        predicted_correction_z: np.ndarray,
        actual_query_z: np.ndarray,
    ) -> np.ndarray:
        p = np.asarray(predicted_correction_z, dtype=float)
        y = np.asarray(actual_query_z, dtype=float)
        if p.shape != y.shape or p.ndim != 2:
            raise ValueError("predicted and actual arrays must be same shape [n_subjects, n_targets]")
        if p.shape[0] < 2 or p.shape[1] < 1:
            raise ValueError("need at least two validation subjects and one target")
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("damping inputs contain NaN/Inf")
        denom = np.sum(p * p, axis=0)
        numer = np.sum(p * y, axis=0)
        gamma = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
        gamma = np.clip(gamma, 0.0, 1.0)
        self.gamma_[(str(support_condition), str(query_condition))] = gamma
        return gamma.copy()

    def get(self, support_condition: str, query_condition: str, n_targets: int) -> np.ndarray:
        key = (str(support_condition), str(query_condition))
        if key not in self.gamma_:
            raise KeyError(f"no validation damping fitted for pair {key}")
        g = self.gamma_[key]
        if len(g) != n_targets:
            raise ValueError("requested n_targets does not match fitted damping")
        return g.copy()

    @staticmethod
    def identity(n_targets: int) -> np.ndarray:
        if n_targets < 1:
            raise ValueError("n_targets must be positive")
        return np.ones(n_targets, dtype=float)
