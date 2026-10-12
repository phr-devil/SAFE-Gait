# Execution ledger: three seeds on frozen folds

Plan: 2026-09-30-three-seed-frozen-folds.md. User explicitly requested seeds17/27 and exact workflow freeze. Work is isolated in a new reports directory; no Git exists.

Seed semantics clarification offered: retain all memberships and vary training RNG only (recommended/default matching explicit freeze). This interpretation was stated before launch and user then instructed continue. Seeds17/27 launched in separate processes after checking GPU memory (seed17 about2.1GB on16GB GPU); deterministic per-process RNG and all memberships unchanged. Seed7 checkpoints/results are reused.

Task1 complete: 4 targeted tests observed RED then GREEN; seed_protocol changes only init_seed_base, exact inner model plan equality enforced against seed7; subject aggregation rejects duplicate/missing seeds and changed memberships. A fifth bootstrap clustering regression also passes. Original trainer imported byte-for-byte. Task2 running: seed17 session40788; seed27 session9639. Persistent console logs and progress.json under each seed directory. Both fold0 runs completed and fold1 training underway.

Analysis and independent replay scripts prepared. Five targeted tests and 38 package tests pass. Per-fold cells with fewer than two subjects retain their point estimates but omit bootstrap CIs. External GAITRite protocol is a conditional draft only; no external study or algorithm change started.

All prior gate development is stopped; no main gate experiment planned.

Intermediate status: both seeds17/27 completed folds0,1,2 and entered fold3. Independent read-only static reviewer found no material issues: all485 original hashes match, completed new fold0 hashes match, membership/RNG-only changes correct, training-only preprocessing/prior and subject-cluster aggregation correct, five targeted tests pass. Numerical review waits for complete outputs and model replay.

Completed: both new seeds all5folds;150new+75reused models. Aggregation saved48main metric rows,36DS paired rows,90beta/TVR rows plus all subject/query records. Verification replays all225models with maxdifference5.684341886080802e-14 and independently reproduces main/paired/clustered-risk statistics. Final read-only reviewer independently checked943pre-conclusion manifest entries,36raw paired CIs,48harm rows and coefficient summaries, finding no blocking issue. Two reporting clarifications (unique versus independent passes, persisted candidate-only decision) are resolved in the final report and external draft.

Conclusion: DS is a worthwhile frozen candidate for independent GAITRite comparison, not established as universally best. Retain all4methods. Strongest cross-condition evidence is velocity/step_length; both cross-condition step_time pooledCIs include0. Full report and decision: reports/three_seed_frozen_folds_20260930/THREE_SEED_REPORT.md and CONCLUSIONS.md. External study remains unstarted; no algorithm/gate change or further training.
