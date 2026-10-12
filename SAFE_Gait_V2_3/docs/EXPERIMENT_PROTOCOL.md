# SAFE-Gait V2.3 Experiment Protocol

## Outer design

Use 5 subject-level folds × seeds 7/17/27. Final test subjects remain untouched until target selection, covariance shrinkage, transfer threshold and optional damping are frozen.

## Residual generation

Generate out-of-subject residuals within each outer-training fold. Use one row per physical pass and retain explicit pass identity. Confirm every condition has enough repeated subjects to estimate within-subject covariance; V2.3 will refuse an unidentifiable fit.

## Primary support policy

Use one globally predeclared calibration protocol. For deterministic first-pass calibration, supply an explicit `order_col` where possible. Random/all-support modes are sensitivity analyses.

## Four pairs

Report SP→SP, HP→HP, SP→HP and HP→SP separately. Same-condition support/query use different physical passes.

Cross-condition means held-out condition transfer, not automatically future-in-time prediction. Temporal claims require a global chronology field from the source protocol.

## Required comparisons

1. Raw population predictor.
2. Population + training-only condition intercept.
3. Simple centered support-residual correction.
4. Scalar/diagonal shrinkage.
5. Same-condition multivariate shrinkage.
6. SAFE-Gait V2.3 hierarchical model.
7. V2.3 + validation-only damping only if justified.
8. Optional tiny residual adapter only after reproducible statistical headroom.

## Metrics

Report MAE/RMSE/R² as appropriate, plus incremental personalization gain beyond the calibrated population baseline, median gain, subject benefit rate, raw negative-transfer rate, and repeatability-aware harm when its threshold is defined at a matched aggregation level.

Mechanism diagnostics should include TVR, paired SP/HP correlation + CI, paired-subject fraction, repeated-subject counts, within-subject degrees of freedom, target-wise shared scale, final shared coherence scale, and decomposition error.

## Bootstrap

Bootstrap subjects, never passes or cycles. When all possible support passes are enumerated, aggregate support choices within subject before headline metrics or bootstrap.
