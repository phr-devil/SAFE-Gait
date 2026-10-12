# SAFE-Gait V2.3

**Reliable Cross-Condition One-Shot Personalization via Audited Hierarchical Residual Decomposition**

SAFE-Gait V2.3 models pass-level, out-of-subject population-model residuals as

\[
r_{s,c,p}=\mu_c+u_s+v_{s,c}+\epsilon_{s,c,p},
\]

where `mu_c` is training-only condition-level population bias, `u_s` is the transferable subject error signature, `v_s,c` is stable condition-specific bias, and `epsilon` is within-subject pass-to-pass variation.

For same-condition personalization, the posterior correction uses `u+v_c`. For cross-condition personalization, only the shared `u` component transfers.

## What is new in V2.3

V2.3 is a third executable audit of V2.2. Important new corrections are:

- refuse fitting when repeated passes are insufficient to identify within-subject covariance;
- make pass identity mandatory so pseudo-replication checks cannot be bypassed;
- replace one-global-scale shared-covariance clipping with target-wise PSD-preserving capping plus a final coherence scale;
- require integer `support_n` / `query_n`;
- support an explicit pass-order column for deterministic `first` calibration;
- reject absent requested conditions instead of returning silent empty experiments;
- require binary, right-padded TCN masks;
- reject empty metric arrays and identifier whitespace aliases;
- report correlation CIs, repeated-subject diagnostics, median gain and negative-transfer rate.

See `THIRD_AUDIT_FIX_REPORT.md` for details. The previous V2.2 audit is retained in `AUDIT_FIX_REPORT.md` as history.

## Required residual table

The statistical core consumes **pass-level, out-of-subject cross-fitted residuals**:

```text
subject_id,condition,pass_id,pass_order,velocity,step_length,step_time
S001,SP,SP_0,0,...
S001,SP,SP_1,1,...
S001,HP,HP_0,0,...
```

Each target column is `reference - raw population prediction`.

Do not use in-sample residuals. Do not export gait cycles as independent pass rows. At least enough subjects must have repeated passes in each condition to identify within-subject covariance; the default minimum is `max(3, n_targets+1)`.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_synthetic_demo.py
```

Real residual headroom analysis:

```bash
python scripts/run_headroom_from_csv.py \
  --csv residuals.csv \
  --targets velocity,step_length,step_time \
  --train-subjects splits/fold0_train.txt \
  --eval-subjects splits/fold0_val.txt \
  --order-col pass_order \
  --support-mode first \
  --out reports/fold0_headroom.json
```

`--order-col` is recommended whenever “first pass” has a scientific meaning. `--support-mode all` is a sensitivity analysis; support choices are aggregated within subject before headline statistics.

## Experiment order

1. Train the population predictor using strict subject-level splits.
2. Generate matched out-of-subject cross-fitted pass-level residuals.
3. Separate raw population error from the training-only condition-intercept correction.
4. Establish residual headroom for SP→SP, HP→HP, SP→HP and HP→SP.
5. Compare simple residual correction, diagonal shrinkage, same-condition multivariate shrinkage and the hierarchical model.
6. Use validation-only damping only if over-correction remains.
7. Add a small neural adapter only if reproducible structure remains after the statistical model.
8. Final evaluation: 5 subject-level folds × 3 seeds with subject-bootstrap confidence intervals.

## Non-negotiable safeguards

- train/validation/test subjects are disjoint;
- query labels never enter personalization state;
- `mu`, scales and variance components are training-only;
- pass identity is mandatory;
- same-condition support/query use distinct physical passes;
- confidence intervals bootstrap subjects, not cycles/passes;
- the residual-generating population predictor must be distributionally aligned with the evaluation predictor;
- cross-condition transfer is **not automatically a temporal future-prediction experiment** unless global chronology is verified.

## Status

V2.3 is software/statistics-audited on synthetic data. It does **not** claim that the full WearGait-PD real-data experiment has already demonstrated scientific benefit.
