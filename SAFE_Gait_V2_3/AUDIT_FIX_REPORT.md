# SAFE-Gait V2.2 — Second Audit and Fix Report

This audit reviewed executable code, not only the algorithm description.

## Result-changing bugs fixed

1. **Incoherent variance decomposition (high severity).** V2.1 projected `Sigma_u` and `Sigma_v` independently. Sampling noise could make projected shared variance exceed a condition's estimable latent marginal variance, after which projecting `Sigma_v` silently inflated the total. V2.2 first estimates each finite-pass-corrected latent marginal covariance, then scales the shared covariance so `Sigma_latent[c] - Sigma_u` is PSD for both conditions. Therefore `Sigma_latent[c] ≈ Sigma_u + Sigma_v[c]` is preserved.

2. **Disabled transfer was not exactly disabled (high severity).** V2.1 initialized `Sigma_u` with `min_eig * I`, so a target flagged as non-transferable still received a tiny nonzero cross-condition correction. V2.2 uses exact zero rows/columns for disabled targets.

3. **Query MAE was computed as `abs(mean residual)` (high severity).** This permits positive/negative query residuals to cancel. V2.2 retains every independent query-pass residual and computes `mean(abs(pass residual - correction))` before subject aggregation.

4. **TCN padding-mask contamination (high severity).** V2.1 applied the mask only at pooling. Convolutional activations created in padded time points could propagate back into valid positions in deeper layers. V2.2 re-masks after every convolutional stage and residual branch. A regression test verifies that a short sequence produces the same prediction alone or when padded inside a longer batch.

5. **NaN padding was not actually masked (high severity).** Multiplying `NaN` padding by zero still yields `NaN`. V2.2 uses `torch.where(mask, x, 0)` and rejects nonfinite values only when they occur at valid time points.

6. **Cross-condition pass-ID false leakage (medium severity).** A pass called `1` in SP and a different pass called `1` in HP was incorrectly treated as the same pass. V2.2 scopes pass overlap checks by condition.

7. **Duplicate pass pseudo-replication was not rejected (high severity).** V2.2 rejects duplicate `(subject, condition, pass)` rows so cycle-level or duplicate exports cannot silently enter the variance-component estimator as independent passes.

8. **Random support selection was the default (medium severity).** That changed the calibration protocol with seed. V2.2 uses deterministic `first` support by default; random selection is explicit sensitivity analysis. An `all` mode enumerates all supports but aggregates within subject before headline metrics.

9. **Predictive variance ignored query averaging (medium severity).** V2.2 adds `query_n` for cases where the inferential target truly is the mean of multiple independent future passes. Headline per-pass prediction evaluation uses `query_n=1`.

10. **Damping changed the prediction mean but not reported expected corrected-error risk (medium severity).** V2.2 separates the undamped posterior mean from the applied correction and reports a corrected-error second-moment matrix when damping is used.

11. **Silent missing damping pair (medium severity).** V2.1 silently returned an all-ones vector if a pair had not been fitted. V2.2 raises an error; identity damping must be requested explicitly.

12. **Source package quick-start/import failure (medium severity).** Direct `pytest -q`/script execution could fail unless `PYTHONPATH` or an editable install had already been set. V2.2 configures pytest source paths, makes direct scripts source-tree runnable, and documents editable installation.

13. **Incoherent negative transfer threshold (medium severity).** The core shared component has an identity/same-sign loading across SP and HP, but V2.1 allowed a negative `min_transfer_correlation`, which could admit negatively associated targets into a model that cannot represent sign reversal. V2.2 restricts this threshold to `[0,1]`; genuine sign-reversing transfer requires a different loading model.

## Additional robustness fixes

- Reject NaN/Inf residuals and invalid identifiers.
- Reject unexpected conditions instead of silently ignoring task-mismatched rows.
- Normalize subject/pass identifiers as strings to avoid mixed-type splitting.
- Validate covariance and damping hyperparameter ranges.
- Return `NaN` for R² when the target is constant instead of a numerically meaningless huge negative number.
- Harden subject bootstrap input validation.
- Stop swallowing arbitrary PyTorch reproducibility exceptions.
- Report raw population, population-intercept-calibrated, and incremental personalization gains separately.
- Add subject-level bootstrap CIs to the headroom CLI.

## Remaining scientific limitations (not silently 'fixed')

- The variance-component estimator is method-of-moments, not REML.
- The model assumes an identity-loading shared subject component across SP/HP after standardization.
- Positive sample cross-condition correlation can still be unstable in small paired cohorts; bootstrap/stability diagnostics should accompany final claims.
- The residual prior can be miscalibrated if the population predictor used to generate cross-fitted training residuals is materially different from the predictor used at validation/test time. See `docs/RESIDUAL_GENERATION_CONTRACT.md`.
- No full 173-subject WearGait experiment is included here because the complete local raw dataset/training pipeline was not present in this chat.
