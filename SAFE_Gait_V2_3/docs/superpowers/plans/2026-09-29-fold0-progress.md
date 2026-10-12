# Fold 0 execution ledger

Plan: 2026-09-29-fold0-sanity.md

Task 1 complete: original60/16/20 snapshot and primary60/15/20 manifests/arrays/splits frozen and hashed. Reference gate passed on239 matched subject-condition pairs. NLS165 SP and NLS186 SP outlier combinations have no primary pass. No outcome-based exclusion.

Task 2 implementation complete: six leakage/sign/pooling/diagonal/episode tests observed failing before implementation, then passing. All15models completed100epochs each.104training OOF rows,25validation rows,30test rows. No validation/test tuning.

Task 3 implementation complete: evaluation/report complete:96summary rows and full subject/query details. Reference split/condition breakdown persisted.15saved-model replay PASS(max difference5.68e-14),frozen/core hashes unchanged.44combined tests passed.

Ruling: no Git repository exists; use new isolated report directory and frozen/core SHA256. No branch/commit operation or core edit.
Ruling: preserve WPD021 original validation membership; exclude all4passes from primary analysis, never refill cohort.
Ruling: diagonal shrinkage is an explicit diagonal covariance ablation of the same training residual decomposition, not an independently tuned method.
Ruling: original60training subjects include IDs with spaces; useJSON orline-based records, never SAFE CLI whitespace-splitting subject-list parser.

Review fixes: dynamically derive pass totals104/25/30; persist reference split/condition table; clarify cross-condition chronology is unverified; show step_time metrics to6decimals. All were reporting fixes; no retraining or core edits.
