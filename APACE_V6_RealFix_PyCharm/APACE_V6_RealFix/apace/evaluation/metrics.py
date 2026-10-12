from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(a * a)))


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    if values.ndim != 1:
        raise ValueError('values must be a one-dimensional subject-level vector')
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite_research_effects: NaN/Inf cannot enter bootstrap or research gates')
    rng = np.random.default_rng(seed)
    if np.all(values == values[0]):
        v = float(values[0])
        return v, v, v
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    result = (float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite_bootstrap_result')
    return result


def auec(reference_counts, errors) -> float:
    x = np.asarray(reference_counts, dtype=float)
    y = np.asarray(errors, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("reference_counts and errors must have equal length >= 2")
    return float(np.trapezoid(y, x))


def normalized_regret(policy_cost: float, oracle_cost: float, no_calibration_cost: float) -> float:
    denom = no_calibration_cost - oracle_cost
    if denom <= 0:
        return 0.0 if policy_cost <= oracle_cost else float("inf")
    return float((policy_cost - oracle_cost) / denom)
