# APACE-V6 | Real-data repair release (PyCharm)

## Local update — 2026-09-28

The NaN/Inf guard repair and first formal WearGait run are documented in
[the execution report](outputs/guard_validation_20260928/REPORT.md): 89 tests pass,
real A/D early stopping occurred, and the saved model was replayed with the final
guard code. That first report covers one held-out subject. Subsequent
[five-fold validation](outputs/generalization_20260928/REPORT.md) covers 173 subjects.
[G3 Oracle headroom](outputs/g3_20260928/REPORT.md) is now evaluated by a separate
offline command. [G4 Value learnability](outputs/g4_20260928/REPORT.md) returned
NO_GO across three Value seeds: further active optimization is stopped. G5 and
G3/G4 interruptions inside the A–E trainer remain unimplemented.

The explicitly requested [one-step restart](outputs/one_step_20260928/REPORT.md)
adds 328 pressure-only observable state features and restricts decisions to
STOP/SP/HP. Its 5-fold × 3-seed G4 also returned NO_GO (63.59%, 60.54%, 69.21%
Oracle Gain Recovery); further tuning is stopped. See
[the fixed protocol](docs/one_step_20260928.md). Multistep planning remains stopped.

The subsequent [selective calibration study](outputs/selective_20260929/REPORT.md)
also returned NO_GO across 5 folds × 3 seeds. The benefit predictor did not improve
on fixed1; always-calibrate and fixed1 are the same baseline. Training is stopped,
and descriptive calibration benefit/failure analysis is complete: 131 subjects
benefited, 36 had larger errors, and 6 had gains smaller than reference cost.
See [the selective protocol](docs/selective_20260929.md).

See [Chinese audit and run instructions](OPTIMIZATION_20260928.md). Local dataset locations
are configured in `configs/local_data.yaml`, resolved relative to that file. Without
`--raw-dir` or `--manifest`, `run_real.py` uses the configured WearGait directory.

```powershell
.\.venv\Scripts\python.exe run_real.py --prepare-only --out outputs/prepared
.\.venv\Scripts\python.exe run_real.py --manifest outputs/prepared/converted/manifest.csv --out outputs/formal --epochs 100
```

The repair retains valid passes from partially invalid files, audits pass rejection,
preserves string IDs, rejects duplicate exports, balances validation by subject,
and aligns calibration/validation/oracle reference budgets. Policy state now includes
the remaining budget and available protocol counts. Acquisition outcomes are sampled
uniformly with a recorded seed, consistent with the oracle. Frozen oracle predictions
are reused across cost weights. Physical-unit errors are reported per target.

New checkpoints use `format_version=2` and include model dimensions and preprocessing
metadata. Earlier value-model checkpoints are incompatible with the added state features;
retrain them. This does not establish scientific accuracy. The quick demo can return
NO_GO; its test verifies valid execution and truthful guards, not guaranteed improvement.

Exact oracle search is currently limited to reference budgets 0–2. G2 examines the
best single reference (including STOP) on validation subjects; for a two-reference
policy it is diagnostic only, since one-reference failure cannot rule out two-reference
benefit. GAITRite and OptiTrack paths are registered, but their raw-data converters
are still not implemented.
For `--max-refs 0`, the policy acquires no references; calibration/value modules are
still trained with a one-reference budget to retain a common pipeline.

This update **repairs the three implementation blockers** found with actual WearGait-PD CSV files. It does **not** claim the calibration method is scientifically successful.

## Changes in this release

1. **Variable pass length:** `collate_passes()` right-pads short passes with a valid-time mask. Overlong passes are anti-aliased down to at most 384 samples for a bounded GPU footprint. The original pass duration in seconds remains an explicit encoder feature. This is NOT equal to simply pretending each pass has the same duration. The mask is applied after every strided convolution to avoid padding/bias contamination.
2. **Retained temporal order:** a lightweight depthwise-separable strided TCN learns motion features; four ordered within-pass temporal bins are pooled only at the end, followed by sensor attention. It no longer averages the time dimension immediately after the first two convolutions.
3. **Real Stage A–E entry point and guards:** `run_real.py` supports WearGait raw SP/HP CSV files or an already converted manifest; data QC, subject-disjoint train/validation/test, TRAIN-only target normalization, stage-specific held-out-subject early stopping and checkpoint restore, NaN/Inf and nonfinite gradient abort, small-sample safeguards, and saved audit/status JSON are wired into the real run.

## Start here (Windows / PyCharm terminal)

Open the folder containing `run_real.py`, create a Python 3.11/3.12 environment and install:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For the **small uploaded sample** place canonical filenames such as `HC101_SelfPace.csv`, `HC102_SelfPace.csv`, `HC102_HurriedPace.csv`, `NLS036_SelfPace.csv`, `NLS036_HurriedPace.csv` in `D:\WearGait_PD\raw`. Do not use `_mat`, `_matTURN`, Balance, TUG, or TandemGait in the SP/HP calibration experiment. Missing 32-channel subjects such as HC100 will be logged as rejected; missing one condition is allowed in a software pilot but constrains the available reference actions.

```bash
python run_real.py --raw-dir "D:\WearGait_PD\raw" --out outputs/real_pilot --pilot --test-subject NLS036 --val-subject HC101 --epochs 3
```

Alternatively use your preconverted manifest:

```bash
python run_real.py --manifest "D:\WearGait_PD\converted\manifest.csv" --out outputs/real_pilot --pilot --test-subject NLS036 --val-subject HC101 --epochs 3
```

Real mode (after you have **at least 10 usable independent subjects**):

```bash
python run_real.py --raw-dir "D:\WearGait_PD\raw" --out outputs/real_run --epochs 100 --max-refs 2
```

In formal mode the software automatically holds out a test subject and multiple validation subjects. **Do not add `--pilot`, and do not select hyperparameters by looking at test results.** The current entry point is one held-out-subject experiment, not an automatic 185-fold LOSO paper pipeline. Repeat outer folds only after your pilot/validation phase succeeds.

Outputs: `conversion_audit.json`, `converted/manifest.csv`, `real_metrics.json`, `checkpoint.pt`. No original raw CSV is copied into the shared code package. Failed runs save `status=HALTED` and the reason, with process exit status 2. The G1/G2 research gates evaluate only validation subjects and are diagnostic-only for small pilots; formal mode can halt on strongly negative, subject-level evidence when at least 8 validation subjects exist. G3 planning headroom, external replication and full LOSO benchmarking require further experiments and **are not claimed to be automated here**.

**Label caveat:** WearGait's raw walkway-to-pass targets are reconstructed research labels (not official PKMAS per-pass exports). The existing converter uses `GeneralEvent=Walk` annotations for offline pass segmentation, not an on-device segmentation model; exact real-world prospective calibration must validate segmentation and acquisition feasibility separately. The five targets have different units. Training standardizes targets using training subjects only; pilot policy metrics labelled `*_normalized` are averaged normalized errors and are NOT physical cm/s or cm accuracy. High test MAE and STOP-only decisions mean the algorithm is not yet shown to work scientifically.

---

# APACE-V6

**APACE-V6** is a lightweight PyTorch research package for **active personalized calibration of plantar-pressure sensing systems**.

The package implements the algorithmic design agreed in this project:

1. learn a population plantar-pressure predictor;
2. summarize acquired reference trials into a calibration belief state;
3. update personalized predictions without test-time gradient descent;
4. generate leakage-safe counterfactual calibration episodes offline;
5. compute finite-horizon oracle calibration cost-to-go;
6. train a distributional value model for candidate reference protocols;
7. choose QUERY/STOP actions according to expected calibration value;
8. automatically stop failed runs or scientifically unpromising branches with numerical and research guards.

The repository includes a small synthetic dataset so the complete Stage A-E pipeline can be checked before connecting a real public dataset.

## 1. Open in PyCharm

1. Extract the project folder.
2. In PyCharm choose **Open** and select `APACE_V6`.
3. Create a Python 3.11 or 3.12 virtual environment.
4. Open the PyCharm terminal and run:

```bash
python -m pip install -r requirements.txt
```

For an NVIDIA RTX 5070 Laptop, install the CUDA-enabled PyTorch build recommended for your installed NVIDIA driver from the official PyTorch installation selector. The package itself does not hard-code a CUDA version.

## 2. Verify the package

```bash
python -m pytest -q
```

## 3. Run the end-to-end demo

Fast CPU/GPU check:

```bash
python run_demo.py --config configs/default.yaml --quick
```

Default output:

```text
outputs/demo/metrics.json
```

Remove `--quick` for a longer synthetic run.

## 4. Scientific architecture

```text
Plantar pressure
      |
      v
Coordinate-aware temporal encoder
      |
      v
Population probabilistic predictor
      |
      +-----------------------------+
      |                             |
      v                             v
Zero-shot prediction       Calibration context
                                    |
                                    v
                         Calibration belief encoder
                                    |
                                    v
                         Gradient-free personal head
                                    |
                                    v
                     Counterfactual protocol evaluation
                                    |
                                    v
                      Finite-horizon oracle cost-to-go
                                    |
                                    v
                    Distributional calibration value model
                                    |
                           +--------+--------+
                           |                 |
                         QUERY              STOP
```

## 5. Training stages

### Stage A — Population predictor
Train `PressureEncoder` + `ProbabilisticPredictor`, then freeze them.

### Stage B — Calibration model
Train `CalibrationBeliefEncoder` + `PersonalizedPredictor` using variable-size subject contexts, then freeze them.

### Stage C — Counterfactual generation
With all previous modules frozen, use complete training-subject references to construct leakage-safe candidate-action values. The final query trials are disjoint from the calibration pool.

### Stage D — Value model
Train `DistributionalValueModel` to estimate lower/median/upper calibration value for each candidate protocol and acquisition-cost weight.

### Stage E — Sequential evaluation
For a held-out subject, the policy may only use pressure data and already acquired references. It chooses another protocol only when predicted net calibration value is positive; otherwise it stops.

## 6. Guard system

The package deliberately separates **numerical failure** from **scientific no-go**.

- **G0 Data Integrity:** NaN/Inf, flat channels, malformed coordinates/targets, insufficient trial length.
- **G1 Base Learnability:** stop expanding the method if the population model cannot beat simple baselines.
- **G2 Calibration Headroom:** stop calibration research if even oracle references barely improve held-out error.
- **G3 Planning Headroom:** disable active planning if oracle schedules do not outperform the best fixed schedule.
- **G4 Value Learnability:** stop planner training if learned values recover too little oracle gain.
- **G5 External Replication:** if a second dataset fails to replicate the effect, lower the scientific claim rather than hiding the result.

`ResearchGate` uses subject-level bootstrap confidence intervals. A small point estimate alone is not treated as sufficient evidence.

Numerical training guards include NaN/Inf checks and gradient clipping. The real-data entry point implements held-out-SUBJECT validation checkpointing and early stopping in stages A, B and D. The compact synthetic demo retains short fixed epoch budgets.

## 7. Real-data manifest format

Raw public datasets generally use different directory layouts. APACE therefore uses a normalized manifest instead of hard-coding undocumented dataset file names.

Each row of `manifest.csv` represents one **independent trial**, not one gait frame:

```csv
subject_id,trial_id,condition,pressure_path,coords_path,speed,stride_length
S001,T001,normal,arrays/S001_T001.npy,coords/insole.npy,1.21,1.34
S001,T002,fast,arrays/S001_T002.npy,coords/insole.npy,1.58,1.51
```

Requirements:

- `pressure_path`: `.npy`, shape `[n_sensors, n_time]`
- `coords_path`: `.npy`, shape `[n_sensors, 2]`
- reference targets: one or more numeric columns such as `speed`, `stride_length`
- `condition`: the actual calibration protocol available in that dataset (`self_paced`, `hurried`, `slow`, `normal`, `fast`, etc.)

Loading:

```python
from apace.data.manifest import load_manifest

trials = load_manifest(
    "data/manifest.csv",
    target_columns=["speed", "stride_length"],
)
```

Dataset-specific converters should convert WearGait-PD, Insole-GAITRite, OptiTrack, etc. into this manifest. Do not merge raw 16-channel and 32-channel signals directly unless a separate cross-device representation study is being performed.

## 8. Leakage rules — mandatory for publication experiments

- split by **subject**, never randomly by gait frame;
- a held-out subject cannot influence feature normalization, model selection, hyperparameters or early stopping;
- calibration trials and final query trials must be disjoint;
- a future GAITRite/Zeno/OptiTrack label may only enter the calibration state after the policy has explicitly queried that reference;
- counterfactual oracle labels are generated only for training/validation subjects;
- report results at the **subject level**, not merely by treating thousands of steps as independent samples.

## 9. Primary research metrics

Recommended primary metrics are:

- reference efficiency;
- error-reference curve / AUEC;
- normalized calibration regret vs. oracle;
- oracle-gain recovery;
- subject benefit rate;
- MAE / NRMSE for the underlying biomechanical targets.

The key scientific claim should be calibration efficiency, not a tiny improvement in raw MAE.

## 10. Important limitation

The synthetic demo proves software integrity, **not scientific effectiveness**. Go/No-Go decisions for a paper must be recomputed from real held-out subjects in the selected public datasets.

## WearGait-PD adapter (added after validation on NLS036)

The package now includes `apace.data.weargait.WearGaitAdapter` for the WearGait-PD CSV format.

Validated inputs:
- `NLS036_SelfPace.csv`
- `NLS036_HurriedPace.csv`

The adapter:
1. Requires the full Moticon pressure set `LPressure1..16` and `RPressure1..16`.
2. Splits a task file into independent passes using contiguous `GeneralEvent == "Walk"` runs.
3. Extracts a `[32, time]` plantar-pressure matrix for every pass.
4. Reconstructs pass-level Zeno reference metrics from raw `Walkway_X`, `Walkway_Y`, `WalkwayPressureLevel`, and `WalkwayFoot` fields.
5. Rejects a pass if pressure-channel missingness exceeds the configured threshold.
6. Exports APACE-compatible `manifest.csv + .npy` arrays with `export_trials(...)`.

Default pass-level target vector:
`velocity_cm_s, step_length_cm, stride_length_cm, step_time_s, stride_time_s`.

### Important leakage rule
`Walkway_*` fields are used only to construct reference targets. They must never be passed to the pressure encoder.

### Sensor layout note
WearGait-PD exposes fixed sensor IDs but not numeric per-sensor x/y coordinates in these CSV files. The adapter therefore uses deterministic sensor-identity/layout features (normalized sensor index + left/right side) instead of inventing physical coordinates. If verified physical coordinates become available, replace these features with the real layout.

## Local label audit and subject cross-validation

Run from this project using `.venv/Scripts/python.exe`:

```powershell
.venv/Scripts/python.exe run_label_audit.py
.venv/Scripts/python.exe run_cross_validation.py --out outputs/new_cv_study
```

The audit compares exact subject/condition aggregate labels against PKMAS; it does not verify individual passes. The five-fold runner freezes subject splits, data/source hashes and parameters before training. Every eligible subject is tested once; normalization uses only that fold's training subjects. Fixed acquisition schedules are selected only on validation subjects. Random and fixed comparators use the same query pool as the policy. A halted fold remains visible as incomplete coverage; existing study directories are protected against overwrite.

Protocol and limitations: [docs/generalization_20260928.md](docs/generalization_20260928.md). Results: [outputs/generalization_20260928/REPORT.md](outputs/generalization_20260928/REPORT.md). No additional Python packages are required beyond the installed project environment.

## G3 offline Oracle headroom

```powershell
.venv/Scripts/python.exe run_g3.py --cv outputs/generalization_20260928/cv --out outputs/new_g3_study
```

Replays frozen models without training, comparing the exact protocol Oracle with validation-selected fixed lengths 1/2 and exact random-policy expectations. Query labels are used only for the declared clairvoyant diagnostic bound, never for learning. A completed NO_GO or INCONCLUSIVE result exits 2 with `allow_active_optimization=false`; GO exits 0. Check `run_status.json` to distinguish a research decision from a failed run. This standalone decision does not insert a new interruption inside `run_real.py`. Full definitions: [docs/g3_20260928.md](docs/g3_20260928.md).

## G4 Value learnability — NO_GO

`run_g4.py` completed the predefined 3-seed, 5-fold study. Oracle Gain Recovery was 75.01%, 73.85%, and 71.06%; none stably beat fixed1, random1, or uncertainty1. The process exited 2 with `allow_active_optimization=false`. Further active-strategy tuning is stopped; no automatic retraining or G5 run follows. Full protocol: [docs/g4_20260928.md](docs/g4_20260928.md). Results and diagnosis: [G4 report](outputs/g4_20260928/REPORT.md).
