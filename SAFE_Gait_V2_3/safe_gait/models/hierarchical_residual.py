from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
from statistics import NormalDist
import numpy as np
import pandas as pd

from safe_gait.data.guards import assert_unique_pass_rows
from safe_gait.utils.linalg import (
    max_psd_scale,
    project_psd,
    regularize_cov,
    solve_psd,
    symmetrize,
    targetwise_cap_shared,
)


@dataclass
class PosteriorResult:
    support_condition: str
    query_condition: str
    centered_support_z: np.ndarray
    posterior_mean_z: np.ndarray
    correction_z: np.ndarray
    correction_raw: np.ndarray
    total_correction_raw: np.ndarray
    predictive_cov_z: np.ndarray
    predictive_sd_raw: np.ndarray
    corrected_error_second_moment_z: np.ndarray
    expected_corrected_rmse_raw: np.ndarray


class HierarchicalResidualModel:
    """SAFE-Gait V2.3 hierarchical one-shot residual model.

    Standardized residual model:
        z_{s,c,p} = u_s + v_{s,c} + eps_{s,c,p}

    u_s      ~ N(0, Sigma_u)      transferable subject component
    v_{s,c}  ~ N(0, Sigma_v[c])   condition-specific subject component
    eps      ~ N(0, Sigma_eps[c]) pass-to-pass component

    Condition-specific population residual bias mu[c] is estimated separately
    from TRAINING subjects and is not counted as personalization.
    """

    def __init__(
        self,
        targets: Iterable[str],
        conditions: Iterable[str] = ("SP", "HP"),
        covariance_shrinkage: float = 0.20,
        min_eig: float = 1e-5,
        min_transfer_correlation: float = 0.0,
        min_repeated_subjects: int | None = None,
    ):
        self.targets = list(targets)
        self.conditions = tuple(str(c) for c in conditions)
        if not self.targets:
            raise ValueError("at least one target is required")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("target names must be unique")
        if len(self.conditions) != 2 or len(set(self.conditions)) != 2:
            raise ValueError("V2.3 requires exactly two distinct conditions")
        if not (0.0 <= covariance_shrinkage <= 1.0):
            raise ValueError("covariance_shrinkage must be in [0,1]")
        if min_eig <= 0:
            raise ValueError("min_eig must be > 0")
        if not (0.0 <= min_transfer_correlation <= 1.0):
            raise ValueError("min_transfer_correlation must be in [0,1]; the core shared component is same-sign")
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.min_eig = float(min_eig)
        self.min_transfer_correlation = float(min_transfer_correlation)
        if min_repeated_subjects is None:
            min_repeated_subjects = max(3, self.n_targets + 1)
        if not isinstance(min_repeated_subjects, int) or isinstance(min_repeated_subjects, bool) or min_repeated_subjects < 2:
            raise ValueError("min_repeated_subjects must be an integer >= 2")
        self.min_repeated_subjects = int(min_repeated_subjects)
        self.fitted_ = False

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    def _validate_df(
        self,
        df: pd.DataFrame,
        subject_col: str,
        condition_col: str,
        pass_col: str,
    ) -> None:
        if not isinstance(pass_col, str) or not pass_col:
            raise ValueError("pass_col is required; pass-level identity cannot be disabled")
        needed = {subject_col, condition_col, pass_col, *self.targets}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        x = df[self.targets].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("residual target columns contain NaN/Inf")
        observed = set(df[condition_col].astype(str).unique())
        missing_c = set(self.conditions) - observed
        unexpected_c = observed - set(self.conditions)
        if missing_c:
            raise ValueError(f"training data missing conditions: {sorted(missing_c)}")
        if unexpected_c:
            raise ValueError(f"unexpected conditions in residual table: {sorted(unexpected_c)}")
        assert_unique_pass_rows(df, subject_col, condition_col, pass_col)

    @staticmethod
    def _sample_cov(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("covariance input must be 2D")
        j = x.shape[1]
        if len(x) < 2:
            return np.zeros((j, j), dtype=float)
        c = np.cov(x, rowvar=False, ddof=1)
        if j == 1:
            c = np.array([[float(c)]])
        return symmetrize(c)

    def fit(
        self,
        residual_df: pd.DataFrame,
        subject_col: str = "subject_id",
        condition_col: str = "condition",
        pass_col: str = "pass_id",
    ) -> "HierarchicalResidualModel":
        """Fit variance components from TRAINING cross-fitted pass-level residuals.

        V2.3 retains the V2.2 coherent estimator and adds identifiability guards in two important ways:
        1) the shared covariance is exactly zero for transfer-disabled targets;
        2) Sigma_u is constrained so that it cannot exceed either condition's
           finite-pass-corrected latent marginal covariance. This preserves a
           coherent decomposition Sigma_latent[c] = Sigma_u + Sigma_v[c].
        """
        self._validate_df(residual_df, subject_col, condition_col, pass_col)
        df = residual_df.copy()
        df[subject_col] = df[subject_col].astype(str)
        df[condition_col] = df[condition_col].astype(str)
        df[pass_col] = df[pass_col].astype(str)
        j = self.n_targets

        # 1) Subject-balanced population bias.
        self.mu_: Dict[str, np.ndarray] = {}
        subject_means_raw: Dict[str, pd.DataFrame] = {}
        for c in self.conditions:
            cdf = df[df[condition_col] == c]
            sm = cdf.groupby(subject_col)[self.targets].mean()
            if len(sm) < 3:
                raise ValueError(f"too few training subjects for condition {c}: {len(sm)}")
            subject_means_raw[c] = sm
            self.mu_[c] = sm.mean(axis=0).to_numpy(float)

        # 2) Common target scale across conditions; each subject-condition group has equal weight.
        weighted_ss = np.zeros(j, dtype=float)
        total_weight = 0.0
        for c in self.conditions:
            cdf = df[df[condition_col] == c]
            for _, sdf in cdf.groupby(subject_col, sort=False):
                x = sdf[self.targets].to_numpy(float) - self.mu_[c]
                w = 1.0 / len(x)
                weighted_ss += w * np.sum(x * x, axis=0)
                total_weight += w * len(x)
        scale = np.sqrt(weighted_ss / max(total_weight, 1.0))
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        self.scale_ = scale

        zcols = [f"__z_{t}" for t in self.targets]
        for c in self.conditions:
            mask = df[condition_col] == c
            df.loc[mask, zcols] = (
                (df.loc[mask, self.targets].to_numpy(float) - self.mu_[c]) / self.scale_
            )

        # 3) Within-subject covariance and subject means.
        self.sigma_eps_: Dict[str, np.ndarray] = {}
        self.repeated_subjects_: Dict[str, int] = {}
        self.within_dof_: Dict[str, int] = {}
        z_subject_means: Dict[str, pd.DataFrame] = {}
        inv_n_mean: Dict[str, float] = {}
        mean_cov: Dict[str, np.ndarray] = {}
        for c in self.conditions:
            cdf = df[df[condition_col] == c]
            scat = np.zeros((j, j), dtype=float)
            dof = 0
            counts: list[int] = []
            means: list[np.ndarray] = []
            ids: list[object] = []
            repeated_subjects = 0
            for sid, sdf in cdf.groupby(subject_col, sort=True):
                x = sdf[zcols].to_numpy(float)
                n = len(x)
                counts.append(n)
                ids.append(sid)
                m = x.mean(axis=0)
                means.append(m)
                if n >= 2:
                    repeated_subjects += 1
                    d = x - m
                    scat += d.T @ d
                    dof += n - 1
            # Sigma_eps is a within-subject variance component and is not identifiable
            # from one pass per subject. V2.2 silently substituted total across-subject
            # variance in that case, which mixes subject heterogeneity into trial noise.
            if repeated_subjects < self.min_repeated_subjects or dof < self.n_targets:
                raise ValueError(
                    f"condition {c} has insufficient repeated-pass information to estimate "
                    f"within-subject covariance: repeated_subjects={repeated_subjects}, dof={dof}, "
                    f"required_repeated_subjects>={self.min_repeated_subjects}, dof>={self.n_targets}"
                )
            cov_eps = scat / dof
            cov_eps = regularize_cov(cov_eps, self.covariance_shrinkage, self.min_eig)
            self.sigma_eps_[c] = cov_eps
            self.repeated_subjects_[c] = int(repeated_subjects)
            self.within_dof_[c] = int(dof)
            z_subject_means[c] = pd.DataFrame(means, index=ids, columns=self.targets)
            inv_n_mean[c] = float(np.mean([1.0 / n for n in counts]))
            mean_cov[c] = self._sample_cov(np.asarray(means, dtype=float))

        # 4) Finite-pass-corrected latent marginal covariance per condition.
        self.sigma_latent_: Dict[str, np.ndarray] = {}
        for c in self.conditions:
            raw_latent = symmetrize(mean_cov[c] - inv_n_mean[c] * self.sigma_eps_[c])
            # Sample noise can make raw_latent slightly indefinite. Regularize the marginal
            # itself first; the subsequent shared/condition-specific split must sum back to it.
            self.sigma_latent_[c] = regularize_cov(
                project_psd(raw_latent, min_eig=0.0),
                self.covariance_shrinkage,
                min_eig=0.0,
            )

        # 5) Candidate shared covariance from paired subject means.
        c0, c1 = self.conditions
        paired = z_subject_means[c0].join(
            z_subject_means[c1], how="inner", lsuffix="_0", rsuffix="_1"
        )
        if len(paired) < max(5, j + 1):
            raise ValueError(
                f"too few paired training subjects to estimate cross-condition transfer: {len(paired)}"
            )
        x0 = paired[[f"{t}_0" for t in self.targets]].to_numpy(float)
        x1 = paired[[f"{t}_1" for t in self.targets]].to_numpy(float)
        x0 = x0 - x0.mean(axis=0)
        x1 = x1 - x1.mean(axis=0)
        cross = (x0.T @ x1) / max(len(paired) - 1, 1)
        sigma_u_raw = symmetrize(cross)

        paired_corr = []
        paired_corr_defined = []
        for k in range(j):
            a, b = x0[:, k], x1[:, k]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                paired_corr.append(0.0)
                paired_corr_defined.append(False)
            else:
                paired_corr.append(float(np.corrcoef(a, b)[0, 1]))
                paired_corr_defined.append(True)
        self.paired_correlation_ = np.asarray(paired_corr)
        self.paired_correlation_defined_ = np.asarray(paired_corr_defined, dtype=bool)
        self.transfer_enabled_ = self.paired_correlation_ > self.min_transfer_correlation

        # EXACT zero rows/columns for disabled targets. V2.1 incorrectly put min_eig on the
        # whole diagonal, leaving tiny non-zero cross-condition corrections.
        sigma_u_candidate = np.zeros((j, j), dtype=float)
        idx = np.where(self.transfer_enabled_)[0]
        if len(idx):
            block = sigma_u_raw[np.ix_(idx, idx)]
            block = regularize_cov(block, self.covariance_shrinkage, min_eig=0.0)
            sigma_u_candidate[np.ix_(idx, idx)] = block

        # First cap each target's shared marginal variance separately. V2.2 used only
        # one global scale, so one weak target could suppress otherwise valid transfer
        # for every other target. Congruence scaling preserves PSD/correlation structure.
        sigma_u_candidate, self.shared_target_scale_ = targetwise_cap_shared(
            sigma_u_candidate,
            [self.sigma_latent_[c0], self.sigma_latent_[c1]],
        )
        # Then apply only the remaining global scale required by full matrix PSD order.
        self.shared_scale_ = max_psd_scale(
            sigma_u_candidate,
            [self.sigma_latent_[c0], self.sigma_latent_[c1]],
        )
        self.sigma_u_ = symmetrize(self.shared_scale_ * sigma_u_candidate)

        # 6) Coherent condition-specific remainder. Keep PSD numerical cleanup at zero floor;
        # observation matrices remain invertible because Sigma_eps has a positive floor.
        self.sigma_v_: Dict[str, np.ndarray] = {}
        self.decomposition_error_: Dict[str, float] = {}
        for c in self.conditions:
            remainder = symmetrize(self.sigma_latent_[c] - self.sigma_u_)
            sigma_v = project_psd(remainder, min_eig=0.0)
            self.sigma_v_[c] = sigma_v
            recon = self.sigma_u_ + self.sigma_v_[c]
            self.decomposition_error_[c] = float(
                np.linalg.norm(recon - self.sigma_latent_[c], ord="fro")
            )

        self.n_training_subjects_ = int(df[subject_col].nunique())
        self.n_paired_subjects_ = int(len(paired))
        self.paired_fraction_ = float(
            self.n_paired_subjects_ / max(self.n_training_subjects_, 1)
        )
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")

    def population_bias(self, condition: str) -> np.ndarray:
        self._check_fitted()
        if condition not in self.conditions:
            raise ValueError("unknown condition")
        return self.mu_[condition].copy()

    def paired_correlation_ci(self, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Approximate Fisher-z CI for paired-subject SP/HP residual correlation.

        This is a diagnostic for transfer stability, not a formal individualized
        guarantee. Constant targets return NaN bounds.
        """
        self._check_fitted()
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0,1)")
        lo = np.full(self.n_targets, np.nan, dtype=float)
        hi = np.full(self.n_targets, np.nan, dtype=float)
        if self.n_paired_subjects_ <= 3:
            return lo, hi
        zcrit = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        se = 1.0 / np.sqrt(self.n_paired_subjects_ - 3.0)
        for k, r in enumerate(self.paired_correlation_):
            if not self.paired_correlation_defined_[k]:
                continue
            rc = float(np.clip(r, -0.999999, 0.999999))
            z = np.arctanh(rc)
            lo[k] = np.tanh(z - zcrit * se)
            hi[k] = np.tanh(z + zcrit * se)
        return lo, hi

    def transferable_variance_ratio(self, condition: str) -> np.ndarray:
        self._check_fitted()
        if condition not in self.conditions:
            raise ValueError("unknown condition")
        u = np.diag(self.sigma_u_)
        v = np.diag(self.sigma_v_[condition])
        ratio = np.divide(u, u + v, out=np.zeros_like(u), where=(u + v) > 1e-12)
        ratio = np.where(self.transfer_enabled_, ratio, 0.0)
        return np.clip(ratio, 0.0, 1.0)

    def posterior(
        self,
        support_residual_raw: np.ndarray,
        support_condition: str,
        query_condition: str,
        support_n: int = 1,
        query_n: int = 1,
        damping: np.ndarray | None = None,
    ) -> PosteriorResult:
        """Infer a query residual from one supervised support residual.

        `support_residual_raw` must be the mean residual of `support_n` independent
        support passes. `query_n` describes whether evaluation/prediction targets a
        single future pass (`1`) or the mean of multiple independent query passes.
        The posterior mean is unchanged by query_n, but predictive variance is not.
        """
        self._check_fitted()
        if support_condition not in self.conditions or query_condition not in self.conditions:
            raise ValueError("unknown condition")
        for name, value in (("support_n", support_n), ("query_n", query_n)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        support_n = int(support_n)
        query_n = int(query_n)
        r = np.asarray(support_residual_raw, dtype=float).reshape(-1)
        if len(r) != self.n_targets:
            raise ValueError("support residual dimension mismatch")
        if not np.isfinite(r).all():
            raise ValueError("support residual contains NaN/Inf")
        x = (r - self.mu_[support_condition]) / self.scale_

        su = self.sigma_u_
        sv_s = self.sigma_v_[support_condition]
        se_s = self.sigma_eps_[support_condition] / float(support_n)
        obs_cov = su + sv_s + se_s

        if query_condition == support_condition:
            latent_cov = su + sv_s
            gain = latent_cov @ solve_psd(obs_cov, np.eye(self.n_targets))
            posterior_mean_z = gain @ x
            post_latent = latent_cov - latent_cov @ solve_psd(obs_cov, latent_cov)
        else:
            gain = su @ solve_psd(obs_cov, np.eye(self.n_targets))
            posterior_mean_z = gain @ x
            post_u = su - su @ solve_psd(obs_cov, su)
            post_latent = post_u + self.sigma_v_[query_condition]

        correction_z = posterior_mean_z.copy()
        if damping is not None:
            g = np.asarray(damping, dtype=float).reshape(-1)
            if len(g) != self.n_targets or np.any(~np.isfinite(g)) or np.any((g < 0) | (g > 1)):
                raise ValueError("damping must have one finite value in [0,1] per target")
            correction_z = g * correction_z

        # Predictive covariance for the future residual mean around the posterior mean.
        pred_cov = project_psd(
            post_latent + self.sigma_eps_[query_condition] / float(query_n),
            self.min_eig,
        )
        # If damping changes the Bayes posterior mean, expected squared corrected error
        # includes the remaining conditional mean bias as well as predictive variance.
        bias_after_correction = posterior_mean_z - correction_z
        corrected_second = symmetrize(
            pred_cov + np.outer(bias_after_correction, bias_after_correction)
        )

        corr_raw = self.scale_ * correction_z
        total = self.mu_[query_condition] + corr_raw
        sd_raw = self.scale_ * np.sqrt(np.maximum(np.diag(pred_cov), 0.0))
        expected_rmse_raw = self.scale_ * np.sqrt(
            np.maximum(np.diag(corrected_second), 0.0)
        )
        return PosteriorResult(
            support_condition=support_condition,
            query_condition=query_condition,
            centered_support_z=x,
            posterior_mean_z=posterior_mean_z,
            correction_z=correction_z,
            correction_raw=corr_raw,
            total_correction_raw=total,
            predictive_cov_z=pred_cov,
            predictive_sd_raw=sd_raw,
            corrected_error_second_moment_z=corrected_second,
            expected_corrected_rmse_raw=expected_rmse_raw,
        )

    def personalize_prediction(
        self,
        zero_shot_query: np.ndarray,
        support_residual_raw: np.ndarray,
        support_condition: str,
        query_condition: str,
        support_n: int = 1,
        query_n: int = 1,
        damping: np.ndarray | None = None,
    ) -> np.ndarray:
        zq = np.asarray(zero_shot_query, dtype=float).reshape(-1)
        if len(zq) != self.n_targets or not np.isfinite(zq).all():
            raise ValueError("zero_shot_query must be a finite vector with one value per target")
        post = self.posterior(
            support_residual_raw=support_residual_raw,
            support_condition=support_condition,
            query_condition=query_condition,
            support_n=support_n,
            query_n=query_n,
            damping=damping,
        )
        return zq + post.total_correction_raw

    def summary(self) -> dict:
        self._check_fitted()
        corr_lo, corr_hi = self.paired_correlation_ci(0.95)
        return {
            "targets": self.targets,
            "conditions": list(self.conditions),
            "n_training_subjects": self.n_training_subjects_,
            "n_paired_subjects": self.n_paired_subjects_,
            "paired_fraction": self.paired_fraction_,
            "repeated_subjects": dict(self.repeated_subjects_),
            "within_subject_dof": dict(self.within_dof_),
            "min_repeated_subjects": self.min_repeated_subjects,
            "population_bias": {k: v.tolist() for k, v in self.mu_.items()},
            "scale": self.scale_.tolist(),
            "paired_condition_correlation": self.paired_correlation_.tolist(),
            "paired_condition_correlation_95ci": [corr_lo.tolist(), corr_hi.tolist()],
            "transfer_enabled": self.transfer_enabled_.astype(bool).tolist(),
            "shared_target_scale": self.shared_target_scale_.tolist(),
            "shared_covariance_scale": self.shared_scale_,
            "decomposition_error_fro": self.decomposition_error_,
            "tvr": {c: self.transferable_variance_ratio(c).tolist() for c in self.conditions},
        }
