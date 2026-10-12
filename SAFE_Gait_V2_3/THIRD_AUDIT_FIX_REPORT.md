# SAFE-Gait V2.3 — Third Audit and Fix Report

This audit started from the clean V2.2 ZIP, re-ran the package, injected edge cases, and modified executable code rather than only editing the design text.

## Newly fixed result/statistics issues

1. **Unidentifiable within-subject covariance was silently fabricated (high severity).**
   V2.2 fell back to total across-subject variance when a condition had no repeated passes. That mixes subject heterogeneity with pass-to-pass noise and makes the variance decomposition scientifically invalid. V2.3 now requires enough repeated subjects / within-subject degrees of freedom and raises a clear error when `Sigma_epsilon` is not identifiable.

2. **Pass-identity protection could be bypassed (high severity).**
   `fit(..., pass_col=None)` disabled duplicate-pass checks. V2.3 requires a pass identifier; pass-level identity can no longer be turned off through the public fit API.

3. **One weak target could suppress transfer for every target (medium-high severity).**
   V2.2 used one global scalar to force the shared covariance below both condition-specific latent covariances. A single low-variance target could therefore shrink all otherwise valid shared effects. V2.3 first performs PSD-preserving target-wise variance capping, then applies only the remaining global matrix-scale correction needed for Loewner-order coherence.

4. **`support_n` / `query_n` accepted non-integer values (medium severity).**
   Passing values such as `1.5` changed covariance scaling even though the variables represent numbers of independent passes. V2.3 requires positive integers.

5. **Primary `first` support still depended on row order (medium severity).**
   V2.3 adds an optional explicit `order_col` to episode builders and the headroom CLI. This lets the first support pass be defined from a predeclared pass order even after CSV concatenation/shuffling.

6. **Absent requested conditions could silently produce empty episode tables (medium severity).**
   V2.3 now errors if a requested support/query condition is absent from the residual table.

7. **Padding masks accepted invalid semantics (medium severity).**
   Arbitrary nonzero numeric masks (for example 0.5) were silently converted to `True`, and internal mask holes were accepted even though the TCN mask is intended for right padding. V2.3 requires boolean/binary masks whose valid values form a contiguous prefix.

8. **Empty metric arrays could return NaN silently (medium severity).**
   Evaluation utilities now reject zero-row / zero-target arrays instead of emitting `mean of empty slice` results.

9. **Whitespace aliases could evade split/identity checks (medium severity).**
   IDs such as `S01` and ` S01 ` could be treated as different subjects. V2.3 rejects leading/trailing whitespace in subject/condition/pass identifiers.

## Reporting / protocol corrections

- The headroom CLI now also reports median incremental gain and raw negative-transfer rate, including subject-bootstrap CI for the latter.
- Model summaries now report repeated-subject counts, within-subject degrees of freedom, target-wise shared scaling, and an approximate Fisher-z 95% CI for paired SP/HP residual correlations.
- Cross-condition evaluation is described as **held-out cross-condition transfer** unless a global chronology variable proves that query trials occur after the support trial. The code does not infer temporal order merely from SP/HP labels.

## Verification status

The V2.3 package is verified with **38 automated tests**, synthetic end-to-end execution, source compilation, CLI smoke checks, wheel build, and clean-ZIP re-extraction testing. See `AUDIT_AND_LIMITATIONS.md` for what remains a scientific assumption rather than a software bug.
