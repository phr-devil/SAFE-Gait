# Audit Status and Known Limitations — SAFE-Gait V2.3

## Executable audit status

V2.3 was audited from a clean extraction of the V2.2 package. The third audit added identifiability checks, stricter anti-leakage contracts, target-wise shared-covariance capping, mask validation, episode-order controls, and expanded result diagnostics. **38 automated tests pass** in the finalized source tree.

See `THIRD_AUDIT_FIX_REPORT.md` for the newly fixed issues and `AUDIT_FIX_REPORT.md` for the earlier V2.2 fixes.

## Remaining scientific limitations

1. **No full real WearGait-PD run is included here.** Synthetic success validates implementation, not scientific efficacy.
2. **Residual-generator alignment remains upstream-critical.** OOF residuals from a weaker predictor can miscalibrate a stronger final predictor.
3. **Variance components use method of moments, not REML.** REML/multilevel sensitivity analysis is still recommended for imbalanced or heteroscedastic real data.
4. **The shared component uses identity/same-sign loading across SP and HP.** Sign-reversing or condition-scaled transfer requires a richer loading model.
5. **Condition-specific components are assumed independent across conditions after the shared component.**
6. **Gaussianity is a working assumption.** Heavy tails/outliers should be checked on real residuals.
7. **Transfer eligibility remains a training/validation model-selection decision.** Point correlation plus its CI should be reported; do not treat a positive estimate as a guarantee.
8. **TVR is descriptive, not causal and not an individualized guarantee.**
9. **Repeatability-aware harm must be estimated at the same condition and aggregation level as the final test metric.**
10. **“First” requires an explicit scientific ordering.** Use `order_col` or a validated upstream chronological order.
11. **Cross-condition does not imply future-in-time.** If a paper says “future HP after SP calibration,” the raw protocol must prove HP occurred after SP; otherwise use “held-out HP” wording.
12. **External datasets require re-estimation of the statistical components without tuning on external test subjects.**

## Real-data go/no-go order

1. Generate matched cross-fitted pass residuals.
2. Confirm repeated-pass identifiability in every outer fold.
3. Compare raw population, calibrated population, simple correction, diagonal shrinkage and V2.3.
4. Inspect paired correlations + CI, TVR, shared target scales, decomposition error, heavy tails and negative transfer.
5. Freeze model choices before final test folds.
6. Add damping/adapter only if justified by validation data.
