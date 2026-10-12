# Next Codex Task — SAFE-Gait V2.3 Real-Data Integration

Use V2.3 as the statistical core. Do not reintroduce APACE planning.

1. Reconnect the existing WearGait population-model / pass-segmentation pipeline.
2. Generate subject-level cross-fitted pass residuals with mandatory physical `pass_id` and, if available, an explicit `pass_order` or global timestamp.
3. Verify every outer-training fold satisfies V2.3 repeated-pass identifiability guards.
4. Run SP→SP, HP→HP, SP→HP, HP→SP headroom.
5. Compare raw population, calibrated population, simple residual, diagonal shrinkage, same-condition multivariate shrinkage and V2.3.
6. Report paired correlation + CI, TVR, shared target scales, decomposition error, benefit and negative-transfer metrics.
7. Freeze choices before final 5-fold × 3-seed test evaluation.
8. Add damping or a tiny neural adapter only if validation evidence justifies them.
