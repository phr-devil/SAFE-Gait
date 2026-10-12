# GAITRite low-supervision batch implementation plan

> Execution: inline by the current agent, using executing-plans. User has already authorized implementation and unattended execution; no additional approval gate.

**Goal:** Run training seeds 17 then 27, each 225 jobs, under the frozen seed-7 subject split, then aggregate all three seeds.

**Architecture:** An execution-only controller launches isolated job subprocesses with the unchanged seed-7 train_model function. Per-seed adapters change only output location, training initialization seed base (1000 + training_seed), and reporting metadata. Frozen protocol and split files remain byte-identical. Scientific outcomes cannot enter scheduling decisions.

**Spec:** The user's eight-point low-supervision request in this conversation.

**Constraints:** Four workers, four torch threads per worker, fixed 100 epochs and all model/preprocessing/method settings. Inner partition seeds and routing stay fixed. Bootstrap RNG remains 7. No retraining of seed 7. No modification of frozen files.

## Tasks

- [x] Write failing tests for isolated exception continuation, repeated technical failure stop, completed-job hash validation, unchanged training function, seed semantics, and subject-cluster aggregation.
- [x] Implement batch policy and isolated job launcher. Save every attempt's logs and result; completed seals checked before reuse. One retry of a failed job from its existing checkpoint; OOM retries run alone without changing batch size. Five consecutive failures of the same technical category stop the batch. Data/protocol changes, leakage, unavailable disk/environment stop immediately. Isolated checkpoint corruption is quarantined; systematic corruption stops. No metric-dependent stopping.
- [x] Create auditable per-seed adapters and independent replay/audit stage. Only after all jobs have terminal status, evaluate complete folds, bootstrap and summarize. Missing models/folds are explicitly missing, never silently replaced or called complete. Nonfatal incomplete batches may proceed to the next seed.
- [x] Implement three-seed subject-cluster aggregation and a conditional scientific report; keep seed-specific results and all slow analyses. Average each subject's errors across seeds, never treat seeds as independent subjects. Missing coverage is reported and confirmatory conclusions withheld where coverage is incomplete.
- [x] Run policy and adapter tests, frozen preflight, then start a hidden detached controller. Verify startup handshake once and return to the user. No polling during training.

## Review focus

Duplicate controllers: exclusive OS lock. Interrupted processes: Windows job object ties children to controller; atomic job states and existing checkpoint resume. Single NaN/OOM must not abort other jobs. Hash corruption must not be mistaken for a retriable scientific failure. No extra epochs/models during postprocessing. Three-seed rates must distinguish risk over seeds from sign of subject seed-mean gain.

## Execution ledger

2026-10-01: No Git repository exists; separate new report/controller directories provide isolation. Frozen source and seed7 outputs untouched. Ten tests pass, including real child-process failure continuation/re-entry, exact training-function AST, subject aggregation, stage output corruption on resume, and local handling of the frozen prediction assertion. Fresh independent code review confirmed both high-impact findings fixed; no remaining blockers. Preflight verified 1,887 seed7 outputs, 2,724 source data files, all frozen splits/exclusions/physical episodes, matching Python/torch/numpy/GPU, and >3 TiB free disk. No new training was performed during preparation. Launch and final runtime state are recorded in the batch directory.

Startup handshake confirmed RUNNING training_seed=17, split_seed=7, automatic_sequence=[17,27], controller PID35384 (venv launcher PID35216). No critical stop at the handshake. Execution lock SHA256: 5a4f2fd9f31ce97955bbb816337c8b389eab4da7ccd131c7e165f4e6391fad32. Codex stops monitoring here; local program owns the remaining training, technical supervision, stage audits, and final aggregation.
