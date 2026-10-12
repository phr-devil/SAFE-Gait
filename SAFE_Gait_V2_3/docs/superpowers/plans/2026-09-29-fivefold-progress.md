# Execution ledger: single outer seed7 fivefold

Input anchor and fold0 outputs verified unchanged. WPD021 excluded without replacement. New split test sizes20/18/19/19/19,95unique subjects,159passes.

Four new tests observed failing for absent implementation, then passing. Existing model/preprocessing code imported exactly from frozen fold0 experiment; no retraining/tuning of fold0. New aggregation code implements median gain, six pairedmethod comparisons and predetermined descriptive stability rule.

Original user requested one seed; interpretation stated before launch: one outerseed7, retaining original3internalcrossfit repeats because user explicitly requires same frozen workflow/hyperparameters. No additional outerseed experiment launched. Async clarification remains available for user steering.

Fold0 evaluation reused verified predictions and checked equality. Folds1–4 completed sequentially. Each fold has 15 models with identical population hyperparameters. All 75 checkpoints replayed successfully; maximum difference 5.684341886080802e-14. All training histories contain 100 epochs.

Final independent review recomputed 36 SAFE-versus-baseline comparisons from subject-level data and found no material reporting errors. All 48 primary rows and 72 paired rows, including bootstrap intervals, passed the complete verification chain. The 48-test suite passed earlier in this run and was not repeated on the later continuation request.

Continuation audit found stage VERIFIED, not interrupted training. Git status/diff confirmed no Git repository; frozen source/input hashes and 397 existing output hashes matched. Remaining work was limited to final review and delivery. Results and completion index are in reports/fivefold_seed7_20260929. No further training is pending.

Key findings: SAFE meets the predefined within-seed stability rule in 6/12 cells versus calibrated population, 1/12 versus simple bias, and 0/12 versus diagonal shrinkage. Cross-condition improvement versus calibrated population is supported in 5/6 cells. All 12 cells have individual negative transfer. HP->SP step_length is worse than diagonal shrinkage in all five folds. No algorithm/hyperparameter changes or additional outer seeds were run.
