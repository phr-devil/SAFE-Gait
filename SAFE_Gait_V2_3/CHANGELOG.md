# Changelog

## V2.3.0

- Added repeated-pass identifiability guards; removed invalid within-noise fallback.
- Made pass identity mandatory in the hierarchical fit API.
- Added target-wise PSD-preserving shared-variance capping before final global coherence scaling.
- Enforced integer support/query pass counts.
- Added explicit episode `order_col` / CLI `--order-col`.
- Rejected absent requested conditions, empty metric arrays, invalid padding masks, and whitespace ID aliases.
- Added paired-correlation 95% CI and repeated-pass diagnostics to model summaries.
- Added median personalization gain and negative-transfer rate + CI to the headroom CLI.
- Clarified that cross-condition transfer is not a temporal future claim without verified chronology.

## V2.2.0

Second executable-code audit of SAFE-Gait.

### Statistical/evaluation corrections

- Reworked variance-component estimation so the shared covariance cannot exceed either condition's finite-pass-corrected latent marginal covariance.
- Enforced exact zero shared covariance for transfer-disabled targets.
- Added decomposition-coherence diagnostics and paired-subject fraction reporting.
- Corrected multi-pass query MAE from `abs(mean residual)` to pass-wise `mean(abs residual))` before subject aggregation.
- Added `query_n` only for inferential targets that truly average independent future passes; per-pass evaluation uses `query_n=1`.
- Separated posterior mean, applied damped correction, predictive covariance, and expected corrected-error second moment.
- Added subject-level bootstrap CIs to the headroom CLI.
- Headroom report now separates raw population, population-intercept calibration, and incremental personalization gain.

### Data/leakage corrections

- Reject duplicate `(subject, condition, pass)` rows.
- Reject unexpected task/condition rows instead of silently ignoring them.
- Scope pass-overlap checks by condition so identical bare pass numbers in SP and HP are not false leakage.
- Normalize identifiers to strings and reject missing identifiers.
- Primary support selection is deterministic `first`; random/all-support modes are explicit sensitivity analyses.
- Added residual-generation contract describing cross-fit/evaluation predictor distribution alignment.

### TCN/numerical corrections

- Reintroduced mask-safe convolution: invalid padded positions are zeroed after each convolutional stage and residual branch.
- Replaced multiplication masking with `torch.where` so NaN padding cannot contaminate valid computation.
- Reject nonfinite values in valid time points and all-padding samples.
- Require odd temporal kernels to preserve exact sequence length.
- Hardened covariance, damping, metric, bootstrap, and reproducibility validation.
- Constant-target R² is reported as undefined (`NaN`) instead of a huge numerical artifact.

### Packaging corrections

- Version bumped to 2.2.0.
- Direct source-tree `pytest` and script invocation fixed.
- Added editable-install/dev metadata.
- Added executable synthetic CLI smoke test.

## V2.1.0

- Replaced black-box 10D condition covariance with hierarchical residual decomposition.
- Added finite-pass correction, common target standardization, population-bias separation, validation-only damping, TVR, repeatability-aware harm, and initial leakage guards.
