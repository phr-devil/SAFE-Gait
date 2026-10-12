from __future__ import annotations
import numpy as np


def symmetrize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    return 0.5 * (a + a.T)


def project_psd(a: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the PSD cone with an eigenvalue floor."""
    if min_eig < 0:
        raise ValueError("min_eig must be >= 0")
    a = symmetrize(a)
    if not np.isfinite(a).all():
        raise ValueError("matrix contains NaN/Inf")
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, min_eig)
    return symmetrize((vecs * vals) @ vecs.T)


def regularize_cov(a: np.ndarray, shrinkage: float = 0.15, min_eig: float = 1e-6) -> np.ndarray:
    """Shrink covariance toward its diagonal, then enforce PSD/PD."""
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")
    if min_eig < 0:
        raise ValueError("min_eig must be >= 0")
    a = symmetrize(np.asarray(a, dtype=float))
    if not np.isfinite(a).all():
        raise ValueError("covariance contains NaN/Inf")
    d = np.diag(np.maximum(np.diag(a), min_eig))
    out = (1.0 - shrinkage) * a + shrinkage * d
    return project_psd(out, min_eig=min_eig)


def solve_psd(a: np.ndarray, b: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Stable solve for symmetric PSD/PD systems."""
    if jitter <= 0:
        raise ValueError("jitter must be > 0")
    a = symmetrize(np.asarray(a, dtype=float))
    b = np.asarray(b, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("solve inputs contain NaN/Inf")
    if b.shape[0] != a.shape[0]:
        raise ValueError("incompatible matrix dimensions")
    eye = np.eye(a.shape[0])
    for k in range(6):
        try:
            return np.linalg.solve(a + (10**k) * jitter * eye, b)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(a) @ b


def max_psd_scale(shared: np.ndarray, marginals: list[np.ndarray], tol: float = 1e-10) -> float:
    """Largest alpha in [0,1] such that each marginal - alpha*shared is PSD.

    This keeps a shared variance component from exceeding either condition's
    estimable latent marginal covariance. A single global scale preserves the
    candidate shared covariance structure rather than clipping each entry.
    """
    shared = project_psd(shared, min_eig=0.0)
    marginals = [project_psd(m, min_eig=0.0) for m in marginals]
    if np.allclose(shared, 0.0):
        return 0.0

    def feasible(alpha: float) -> bool:
        for m in marginals:
            vals = np.linalg.eigvalsh(symmetrize(m - alpha * shared))
            if vals.min(initial=0.0) < -tol:
                return False
        return True

    if feasible(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def targetwise_cap_shared(
    shared: np.ndarray,
    marginals: list[np.ndarray],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-wise variance cap for a PSD shared covariance candidate.

    A single global alpha can let one low-variance target suppress transfer for all
    targets. This congruence scaling first caps each shared marginal variance by
    the minimum available latent variance across conditions while preserving the
    candidate correlation structure and PSD. A final max_psd_scale can then enforce
    the full Loewner-order constraints including off-diagonal directions.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    shared = project_psd(shared, min_eig=0.0)
    marginals = [project_psd(m, min_eig=0.0) for m in marginals]
    if not marginals:
        raise ValueError("at least one marginal covariance is required")
    n = shared.shape[0]
    if any(m.shape != (n, n) for m in marginals):
        raise ValueError("shared and marginal covariances must have matching shapes")
    diag_shared = np.maximum(np.diag(shared), 0.0)
    caps = np.min(np.vstack([np.maximum(np.diag(m), 0.0) for m in marginals]), axis=0)
    factors = np.ones(n, dtype=float)
    active = diag_shared > eps
    factors[active] = np.minimum(1.0, np.sqrt(np.maximum(caps[active], 0.0) / diag_shared[active]))
    factors[~active] = 0.0
    d = np.diag(factors)
    return symmetrize(d @ shared @ d), factors
