from __future__ import annotations
import numpy as np


def subject_bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
):
    """Percentile CI by resampling subjects, never cycles/passes."""
    x = np.asarray(values, dtype=float)
    if x.ndim < 1 or x.shape[0] < 2:
        raise ValueError("need at least two subjects")
    if not np.isfinite(x).all():
        raise ValueError("bootstrap values contain NaN/Inf")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")
    rng = np.random.default_rng(seed)
    stats = []
    n = x.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stat = statistic(x[idx], axis=0)
        except TypeError:
            stat = statistic(x[idx])
        stats.append(stat)
    stats = np.asarray(stats)
    a = (1.0 - confidence) / 2.0
    return np.quantile(stats, a, axis=0), np.quantile(stats, 1.0 - a, axis=0)
