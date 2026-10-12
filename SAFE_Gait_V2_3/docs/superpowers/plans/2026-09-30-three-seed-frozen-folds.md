# Three seeds on frozen five-fold memberships

Goal: add run seeds17/27 to completed7, preserve dataset, memberships, preprocessing, hyperparameters and four methods, and judge diagonal shrinkage from repeated training runs.

Authority/spec: user's current explicit request. Native inline execution; no Git repository. New artifacts in reports/three_seed_frozen_folds_20260930; source/anchor SHA256 provide isolation and change guards. Do not delete outputs or mutate earlier reports.

Seed interpretation stated before launch: preserve exact original train/validation/test AND internal cross-fitting memberships. The new run seed changes only population RNG initialization base from1007 to1017/1027, with unchanged per-repeat+100 and inner-fold offsets. Original internal partition seeds7/17/27 remain frozen. This tests training RNG variation at fixed splits, not outer split randomness. Optional clarification was asked; preserving explicit frozen memberships is the default if no steering arrives.

## Analysis decisions before new outcomes

- Reuse all75 seed7 checkpoints. Train150 new checkpoints:2runs×5outer folds×3internal repeats×5inner folds,100epochs unchanged.
- Import original trainer and evaluation functions without editing them. Separate seed/fold directories and persisted console logs. Seed IDs belong in metadata/filenames, not model algorithm. If interrupted, resume only from verified completed artifacts; never silently overwrite checkpoints.
- Four methods only: calibrated population, simple bias, diagonal shrinkage, full SAFE V2.3. Gate remains archived exploratory work and is excluded from main comparison.
- Every run has95 unique outer-test subjects/159passes, identical eligible pair cohorts13/25/23/23. Each subject appears once per run. WPD021 remains excluded.
- Primary aggregation: mean subject MAE across three seeds, then equal subject weights. Do not average predictions into a new ensemble. Subject bootstrap carries all three seed observations together,2000draws,seed7.
- Report MAE, mean/median gain, and CIs; subject-mean benefit/negative transfer PLUS average per-seed harm rate and any/all-seed harm to expose instability hidden by seed averaging.
- Paired comparisons of DS against calibrated population/simple bias/full V2.3: positive delta=comparatorMAE−DSMAE. Give each seed's mean delta, pooled paired95CI, subject win/loss, positive folds per seed. Descriptive cross-seed support rule: pooled meanCI low>0, median delta>0, all3seed mean deltas>0. Also report whether all3individual-seed CIs exclude0; no threshold is tuned.
- Compute β and TVR from each of15 unchanged training-only priors; give point matrices, per-seed fold means/ranges, within-fold variation across seeds, and mean pairwise absolute changes. Do not treat overlapping15priors as independent subjects or average different target units.
- Distinguish practical candidate freezing from proof of statistically best method. Do not assert DS superiority to simple bias or V2.3 without paired evidence. One fixed subject partition family is a limitation even with3RNGseeds.

## Tasks

- [x] Freeze run configuration and test exact membership preservation, sole allowed RNG change, and subject-cluster aggregation with3seeds.
- [x] Run seeds17/27 with unchanged trainer, save all checkpoints, residuals, provenance and logs; do not rerun seed7.
- [x] Aggregate four methods, paired differences, per-seed stability, harm and β/TVR; generate full report.
- [x] Replay all new saved models, recheck preprocessing/subject exclusion, residual sign and downstream calculations; run appropriate tests and one fresh independent review.
- [x] Decide whether DS is ready as a frozen candidate for external GAITRite validation; if supported, outline a locked, independent confirmation protocol without tuning to prior test results.

Review focus: exact seed semantics; repeated-subject rather than independent seed rows; original prediction routing/inner ensemble preserved; no gate included; no changing defaults in response to results.
