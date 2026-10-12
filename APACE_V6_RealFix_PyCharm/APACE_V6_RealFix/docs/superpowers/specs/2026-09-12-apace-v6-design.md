# APACE-V6 Design Specification

## Goal
Build a lightweight Python/PyTorch research package for active personalized calibration of plantar-pressure models. The system must learn a population gait predictor, update a calibration belief from a variable number of reference trials, estimate the distribution of long-term value for each candidate calibration protocol, and stop when no action is worth its acquisition cost.

## Scientific scope
The first implementation focuses on the algorithmic mechanism, not on a specific hardware file format. Real datasets are connected through a manifest adapter. Different datasets may use different encoders and targets; the active-calibration mechanism and evaluation protocol remain identical.

## Core pipeline
1. Dataset adapter emits subject-level `Trial` objects with pressure, sensor coordinates, protocol/condition and reference targets.
2. Data QC rejects malformed trials and flags unusable subjects before training.
3. A coordinate-aware temporal pressure encoder and probabilistic population predictor learn zero-shot gait estimation.
4. A DeepSets-style calibration belief encoder summarizes zero or more `(embedding, protocol, residual)` reference observations.
5. A lightweight FiLM personalized predictor conditions the population representation on the calibration belief without gradient-based test-time fine-tuning.
6. With the predictor and calibrator frozen, an offline counterfactual episode builder measures the actual error reduction caused by adding candidate reference protocols while preserving a disjoint query set.
7. A dynamic-programming oracle computes long-horizon calibration cost-to-go on the finite action space.
8. A distributional value model predicts calibration gain/cost quantiles for candidate actions and a cost weight `lambda`.
9. The policy queries the action with positive net value and stops when all actions fail the value threshold.
10. Evaluation reports gait error, reference efficiency, normalized regret, AUEC, oracle-gain recovery and subject benefit rate.

## Information-leakage rules
- All model selection is subject-level; gait cycles from a held-out subject never enter training.
- Calibration candidates and final query trials are disjoint.
- Future reference labels are used only during offline counterfactual-label construction for training subjects, never during policy decisions for held-out subjects.
- Test-time calibration updates use only reference observations already acquired by the policy.

## Training stages
A. Train base encoder/predictor; freeze.
B. Train calibration belief + personalized head episodically; freeze.
C. Generate counterfactual records and oracle values offline; no neural optimization.
D. Train distributional value model; freeze.
E. Evaluate the sequential query/stop policy on held-out subjects.

## Guard system
- G0 Data Integrity: reject invalid trials/subjects.
- G1 Base Learnability: halt research pipeline when the base model cannot beat simple baselines by a configured margin.
- G2 Calibration Headroom: halt active-calibration work when oracle references do not provide sufficient improvement.
- G3 Planning Headroom: disable planner when oracle schedules do not beat the best fixed strategy.
- G4 Value Learnability: halt planner training when learned values fail to recover a configured fraction of oracle gain.
- G5 External Replication: downgrade claims rather than crash when an external dataset does not replicate the direction of effect.

Research gates use subject-level bootstrap confidence intervals rather than only point estimates.

## Numerical guards
Immediate stop on NaN/Inf loss or parameters; gradient clipping; configurable early stopping; prediction-collapse detection; optional automatic batch-size reduction is left to the launcher rather than hidden inside the model.

## Model size target
The default model should remain in the low-single-million parameter range and run on CPU for tests and on an 8 GB laptop GPU for research experiments.

## Real-data interface
A CSV manifest points to pressure `.npy` files, target values, subject id, condition, optional trial id, and a sensor-coordinate `.npy` file. This avoids hard-coding undocumented dataset layouts while allowing dataset-specific converters later.
