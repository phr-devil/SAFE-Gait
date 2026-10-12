# SAFE-Gait V2.3 Algorithm Design

## Objective

Given one supervised calibration pass from an unseen subject, infer the stable component of that subject's population-model residual that can safely personalize held-out gait predictions.

## Residual model

\[
r_{s,c,p}=\mu_c+s\odot z_{s,c,p},\qquad
z_{s,c,p}=u_s+v_{s,c}+\epsilon_{s,c,p}.
\]

`mu_c` is a training-only condition intercept and is not counted as personalization. `u_s` is transferable across SP/HP, `v_s,c` is stable condition-specific bias, and `epsilon` is pass-to-pass variation.

## Identifiability

Within-subject covariance `Sigma_epsilon[c]` is estimated only from repeated passes. V2.3 refuses to fit when the number of repeated subjects / within-subject degrees of freedom is insufficient. This replaces the invalid V2.2 fallback that could mix between-subject heterogeneity into within-subject noise.

## Finite-pass correction

For subject means:

\[
Var(\bar z_{s,c})=\Sigma_u+\Sigma_{v,c}+E(1/n_{s,c})\Sigma_{\epsilon,c}.
\]

Thus

\[
\Sigma_{latent,c}\approx Var(\bar z_{s,c})-E(1/n_{s,c})\Sigma_{\epsilon,c}.
\]

## Shared covariance

Paired SP/HP subject means estimate a candidate shared covariance under the working assumption that condition-specific components are independent across conditions after `u_s` is accounted for.

V2.3 applies two coherence steps:

1. **Target-wise variance cap.** A PSD congruence scaling limits each shared target variance to the minimum latent variance available in both conditions, preventing one weak target from globally suppressing all transfer.
2. **Final matrix coherence scale.** A remaining global scale is applied only as needed so
   \[
   \Sigma_{latent,c}-\Sigma_u\succeq0
   \]
   for both conditions.

Then

\[
\Sigma_{v,c}=\Sigma_{latent,c}-\Sigma_u.
\]

## Transfer eligibility

Targets whose paired SP/HP residual correlation does not exceed the predeclared training/validation threshold have exact-zero rows/columns in `Sigma_u`. The package reports the point correlation and an approximate Fisher-z 95% CI as a stability diagnostic.

## One-shot posterior

For a support residual mean from integer `n_s >= 1` independent passes:

\[
x_c=(r_c-\mu_c)/s=u+v_c+\bar\epsilon_c,
\]

\[
S_c=\Sigma_u+\Sigma_{v,c}+\Sigma_{\epsilon,c}/n_s.
\]

Same-condition:

\[
E[u+v_c\mid x_c]=(\Sigma_u+\Sigma_{v,c})S_c^{-1}x_c.
\]

Cross-condition:

\[
E[u\mid x_c]=\Sigma_u S_c^{-1}x_c.
\]

Final prediction:

\[
\hat y_q=\hat y_q^{(0)}+\mu_q+s\odot\hat z_q.
\]

## Evaluation

For multiple independent query passes, compute pass-wise error first, then aggregate within subject. Never replace MAE with `abs(mean residual)`.

Primary cross-condition language is “held-out cross-condition transfer.” Temporal “future” wording requires verified global chronology.
