# Single outer-seed fivefold extension plan

Goal: extend the frozen fold0 sanity run to five disjoint outer test folds, without changing data, hyperparameters or core algorithms; aggregate test-only results and paired subject differences.

Authority: current user request. Input anchor: reports/fold0_sanity_20260929/frozen and verified fold0 outputs. Work in reports/fivefold_seed7_20260929; no Git repository, preserve input/core SHA256.

## Frozen analysis choices (before new results)

- One outer seed7 only. The existing pipeline includes3 internal cross-fit repeats (partition seeds7/17/27); these are the frozen3-member prediction ensemble within one outer-seed run, not3 outer-seed experiments. Clarification requested about this distinction before training expansion.
- Recover seed7 permutation from original fold0 test+validation+train; assert exact RNG reproduction. Split this96-ID sequence into20/19/19/19/19 outer test blocks. For each fold, validation is the first16 IDs remaining in the same frozen sequence; train is the rest. Remove WPD021 from its assigned roles without replacement. Fold0 original and primary identities must exactly match previous snapshots.
- Main cohort95subjects/159passes. Test coverage once per subject. Expected combined pair n=13/25/23/23.
- Reuse completed fold0 models/results if the unchanged internal protocol is retained. For folds1–4 call the exact existing trainer with experiment-output globals redirected; preserve TCN, preprocessing, seeds, epoch count, inner crossfit, routing hash convention, default SAFE and correction definitions byte-for-byte. Model outputs in each fold are namespaced by outer_fold.
- Fold1 training61subjects naturally gives48/49subject fitting sets for inner5groups. This rounding is part of unchanged np.array_split, not a hyperparameter change. Other folds fit48subjects/model.
- Only trainingOOF residual fits prior; validation remains excluded from model/parameter selection; aggregate outer test results only.
- Metrics per pair/target/method: subject-balanced MAE, mean gain and median gain against calibrated population, benefit rate, negative-transfer rate;2000subject percentile-bootstrap95%CI,seed7. One paired subject row, mean absolute query error before subject averaging.
- Six method pairs: delta=MAE(method_b)-MAE(method_a), positive favors method_a. Report per-subject deltas, mean/median delta and bootstrapCI, win/loss/tie rates and per-fold mean deltas.
- TVR from each fold's training-only prior, per condition/target; report five estimates and descriptive median/range. No fictitious independent-subject CI for five overlapping training sets.
- Interpret 'stable within this run' before results: pooled mean paired advantage CI lower bound>0, positive median advantage, and positive fold mean in>=4 of5 folds. Also report actual fold counts, not just label. This is a descriptive operational rule, not confirmation across seeds or adjusted multiple-comparison inference.
- Cross-condition is condition transfer, not verified future chronology. Bootstrap conditional on fitted models; no retraining uncertainty or full training-overlap uncertainty. Fold0 was inspected previously; no tuning based on it, but this is not an untouched external confirmation study.

## Tasks

- [x] Test exact fold0 preservation, disjoint exhaustive test assignment, no replacement of WPD021, paired delta sign, true median gain, bootstrap subject unit.
- [x] Freeze outer splits/config/source hashes; confirm reference gate and repeated-pass feasibility for every training set.
- [x] Run remaining folds with unchanged trainer and record checkpoints/provenance; allow resume only after checking completed outputs and immutable configuration hashes.
- [x] Evaluate unchanged four methods on each fold test; save every residual and query/subject result and prior TVR.
- [x] Aggregate each subject once, bootstrap and paired method differences, fold consistency; export full tables and readable report answering five user questions.
- [x] Independently reload saved models or replay complete provenance checks, verify hashes and statistics, run tests and request independent review.

Review focus: do not confuse inner repeats with outer seeds; retain fold0; handle fold3 SP->SP n=1 per-fold intervals honestly; use JSON/line-preserving IDs with spaces; never merge validation into combined test metrics; equal query/support cohorts across methods.
