from __future__ import annotations
import numpy as np


def _finite_2d(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D [subjects/samples, targets]")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one target")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return a


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = _finite_2d(y_true, "y_true")
    p = _finite_2d(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    err = p - y
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    denom = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    sse = np.sum(err**2, axis=0)
    # R² is undefined for a constant target; report NaN instead of an enormous artifact.
    r2 = np.full(y.shape[1], np.nan, dtype=float)
    ok = denom > 1e-12
    r2[ok] = 1.0 - sse[ok] / denom[ok]
    return {"mae": mae, "rmse": rmse, "r2": r2}


def subject_personalization_metrics(
    baseline_abs_error: np.ndarray,
    personalized_abs_error: np.ndarray,
    repeatability_threshold: np.ndarray | None = None,
) -> dict:
    """Inputs are subject × target absolute errors (already subject-aggregated)."""
    b = _finite_2d(baseline_abs_error, "baseline_abs_error")
    p = _finite_2d(personalized_abs_error, "personalized_abs_error")
    if b.shape != p.shape:
        raise ValueError("baseline and personalized errors must have same shape")
    if np.any(b < 0) or np.any(p < 0):
        raise ValueError("absolute-error inputs must be nonnegative")
    gain = b - p
    out = {
        "mean_gain": gain.mean(axis=0),
        "median_gain": np.median(gain, axis=0),
        "subject_benefit_rate": np.mean(gain > 0, axis=0),
        "raw_negative_transfer_rate": np.mean(gain < 0, axis=0),
    }
    if repeatability_threshold is not None:
        t = np.asarray(repeatability_threshold, dtype=float).reshape(1, -1)
        if t.shape[1] != b.shape[1] or not np.isfinite(t).all() or np.any(t < 0):
            raise ValueError("repeatability threshold must be finite, nonnegative, one per target")
        harm = (p - b) > t
        out["repeatability_aware_harm_rate"] = np.mean(harm, axis=0)
    return out


def estimate_repeatability_threshold(
    error_a: np.ndarray,
    error_b: np.ndarray,
    quantile: float = 0.95,
) -> np.ndarray:
    """Training/validation-only threshold from repeated-pass error variation.

    `error_a` and `error_b` should be paired subject-level absolute-error vectors
    from independent repeated passes under the same condition/model. This measures
    repeatability variation, not pure sensor measurement noise.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be strictly between 0 and 1")
    a = _finite_2d(error_a, "error_a")
    b = _finite_2d(error_b, "error_b")
    if a.shape != b.shape:
        raise ValueError("repeatability error arrays must have same shape")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("repeatability inputs must be absolute errors and therefore nonnegative")
    return np.quantile(np.abs(a - b), quantile, axis=0)


def episode_absolute_errors(
    query_residuals: np.ndarray,
    population_bias: np.ndarray,
    personalization_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-target MAE across independent query passes for one subject episode.

    Crucially, this computes mean(|r_p - correction|), NOT |mean(r_p) - correction|.
    The latter allows positive/negative query residuals to cancel and biases MAE downward.
    """
    q = _finite_2d(query_residuals, "query_residuals")
    mu = np.asarray(population_bias, dtype=float).reshape(-1)
    corr = np.asarray(personalization_correction, dtype=float).reshape(-1)
    if len(mu) != q.shape[1] or len(corr) != q.shape[1]:
        raise ValueError("population_bias/correction dimension mismatch")
    if not np.isfinite(mu).all() or not np.isfinite(corr).all():
        raise ValueError("population_bias/correction contains NaN/Inf")
    raw = np.mean(np.abs(q), axis=0)
    calibrated = np.mean(np.abs(q - mu.reshape(1, -1)), axis=0)
    personalized = np.mean(np.abs(q - mu.reshape(1, -1) - corr.reshape(1, -1)), axis=0)
    return raw, calibrated, personalized
