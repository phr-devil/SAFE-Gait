# Fold 0 sanity check implementation plan

> Execute inline, with leakage/sign tests and final independent review. User authorized this run in the current chat. No core edits, no other outer folds.

**Goal:** Freeze the existing data/splits, validate reconstructed references, generate true training OOF residuals and report four correction methods on validation/test separately.

**Spec:** Current user request plus `reports/weargait_manifest_audit/README.md` and V2.3 residual contract.

**Architecture:** Standalone experiment scripts and frozen artifacts under `reports/fold0_sanity_20260929`. Import existing WearGait data helpers, PopulationTCN, HierarchicalResidualModel, episode builder and subject bootstrap without editing them.

**Constraints:** Preserve frozen 60/16/20 identities. WPD021 remains assigned to validation but all its passes are excluded from primary analysis (60/15/20 used). Three targets velocity (cm/s), step_length (cm), step_time (s). Only complete original pressure arrays. No global standardization, query-label updates, outer refit, test tuning, damping, other outer folds or replacement subjects.

## Task 1: Freeze and reference gate

- [x] Copy original manifest/arrays, audit data and fold0 candidate into a new frozen snapshot; hash all inputs and original core sources. Save original and primary analysis manifests and split lists (line-based IDs, preserving spaces).
- [x] Compare exact subject-condition PKMAS matches against means of all valid reconstructed walkway passes, and against strict-pressure selected pass means; exclude WPD021. The all-reference-pass view checks reconstruction without selection confounding; incomplete pressure passes are never used for modeling.
- [x] Report three-target signed bias, relative error quantiles and outliers, overall and by condition/split. Practical software sanity gate (not clinical certification): median absolute relative error <=5%, p90 <=15%, abs(mean signed relative error) <=5%, fraction >25% relative error <=5%, for each primary target in all-reference-pass matched data. Gate fixed before fresh analysis. Inspect outliers; do not remove subjects based on test results.
- [x] Only permit training if gate passes. Freeze configuration before training.

## Task 2: OOF population predictions

- [x] Write tests before implementation for training-only pressure/target scalers, held-out extreme-value invariance, residual sign, model routing excluding every predicted train subject, and equal ensemble size.
- [x] Use unchanged SAFE PopulationTCN(32,3,hidden=32,dropout=.1). Apply fixed factor-4 mean pooling (100Hz ->25Hz, no time warping), log1p pressure, inner-train-only subject-balanced channel standardization and target standardization. Right padding with valid masks.
- [x] Fixed 100 epochs, AdamW lr=.001 weight_decay=.001, batches16, subject-balanced MSE, gradient clipping5; no early stopping or parameter selection on held-out labels. Deterministic seed per model, record device and logs.
- [x] Three repeated inner five-group partitions (seeds7,17,27), 48 fitting subjects /12 OOF subjects per model, 15 fits total within outer training only. First inner partition reproduces existing candidate groups. Each training subject predicted once per repeat by an excluding model; ensemble average of three predictions.
- [x] For validation/test, route each subject to one model per repeat via fixed SHA256(subject,repeat) modulo5; average three. All models trained on48 subjects, same ensemble size as OOF. Do not use a stronger outer refit or average all15 for evaluation.
- [x] Save model state, preprocessing parameters/fit IDs, membership, individual predictions, ensemble predictions and residual CSV with residual=reference-prediction. Validate provenance and finiteness before correction models.

## Task 3: Correction comparison

- [x] Test bias sign, same-condition distinct passes, pass-wise MAE before subject aggregation, diagonal shrinkage and subject-bootstrap inputs.
- [x] Fit unchanged SAFE on training OOF residuals only, default covariance_shrinkage=.2, no damping. Use its training-only condition intercept for all methods.
- [x] Calibrated population correction=mu_query. Simple bias correction=mu_query+(r_support-mu_support). Diagonal shrinkage is SAFE's diagonal-only covariance ablation: same latent variance/(latent+noise) within condition, shared variance/(support latent+noise) across conditions. SAFE uses full covariance. Clearly label diagonal definition.
- [x] Earliest eligible support; independent query passes. Four SP/HP pairs, validation/test separately; never headline in-sample prior-fit training results.
- [x] Per target and method: subject-balanced MAE, gain vs calibrated population, benefit and negative-transfer rates, 2000 subject bootstrap percentile95% CIs for every metric, fixed seed7. Resample paired subject rows consistently across methods. Warn that 2-4 test subjects/pair makes CIs unstable and conditional on fixed trained models.
- [x] Save query-pass predictions, subject metrics, aggregate CSV/JSON and readable Markdown report; do not pool units across targets.

## Review focus

1. IDs with spaces must not be split by whitespace.
2. WPD021 exclusions must not silently change frozen split identities.
3. OOF preprocessing and prediction membership both exclude the subject.
4. Equal OOF/evaluation predictor ensemble size and training-set size.
5. Reference-vs-prediction sign and intercept counted exactly once.

## Verification/ledger

- Source workspace is not a Git repository; isolation is a new output directory, with core SHA256 checked before/after rather than branch commits.
- Unit tests, output provenance assertions, source hashes and independent review must pass before completion.
