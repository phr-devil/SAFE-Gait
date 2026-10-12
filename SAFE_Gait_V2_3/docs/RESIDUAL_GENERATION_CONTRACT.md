# Residual Generation Contract (V2.3)

SAFE-Gait is valid only if residuals used to estimate the prior represent unseen-subject population-model error.

## 1. Out-of-subject prediction

For every training-subject residual row

\[
r_{s,c,p}=y_{s,c,p}-\hat y^{(0)}_{s,c,p},
\]

the population predictor producing `yhat0` must not have trained on subject `s`. Use subject-level cross-fitting, not ordinary in-sample residuals.

## 2. Predictor-distribution alignment

Prefer a matched repeated-cross-fit ensemble protocol. If a materially stronger outer refit is used at evaluation, quantify the residual-distribution shift on validation subjects before claiming calibrated posterior uncertainty.

## 3. Pass-level identity is mandatory

One row must be one physical walking pass. `subject_id`, `condition`, and `pass_id` are mandatory. The statistical model no longer allows pass identity to be disabled.

Do not export gait cycles as independent passes. Duplicate `(subject, condition, pass)` rows are rejected, but unique cycle IDs cannot be detected automatically if they are falsely presented as pass IDs; this remains an upstream data responsibility.

## 4. Repeated-pass identifiability

`Sigma_epsilon[c]` is a within-subject variance component and cannot be learned from one pass per subject. V2.3 requires a minimum number of repeated subjects and within-subject degrees of freedom in each condition. If this requirement fails, simplify the model or obtain more repeated measurements; do not substitute total between-subject variance.

## 5. Ordering / chronology

The primary experiment can use an explicit `order_col` to define the first support pass reproducibly. Without such a column, input row order is only a file-order convention.

SP→HP means held-out cross-condition transfer. It may be described as temporal “future HP” only if a global timestamp/session order establishes that HP occurred after the support SP pass.
