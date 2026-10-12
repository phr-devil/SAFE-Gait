# APACE-V6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a PyCharm-ready APACE-V6 research package with a runnable synthetic end-to-end demonstration.

**Architecture:** Use small, isolated Python modules for data contracts, QC, models, counterfactual/oracle logic, training guards and evaluation. All learned components use PyTorch; offline planning uses deterministic Python/NumPy dynamic programming. The synthetic demo validates the full stage ordering without requiring a proprietary dataset layout.

**Tech Stack:** Python 3.11+, PyTorch 2.x, NumPy, pandas, scikit-learn, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-apace-v6-design.md`

## Global Constraints
- Subject-level train/validation/test separation is mandatory.
- Calibration candidates and final query trials must remain disjoint.
- No future reference labels may enter a held-out policy state before the corresponding query action.
- Base model is frozen before calibration training; calibrator is frozen before counterfactual record generation.
- Default architecture must fit in 8 GB GPU memory.
- All numerical and research guards are enabled by default in the demo.

---

### Task 1: Define data contracts and QC
**Files:** `apace/data/types.py`, `apace/data/manifest.py`, `apace/data/synthetic.py`, `apace/qc/checks.py`, `tests/test_data_qc.py`
**Produces:** `Trial`, manifest loading, synthetic subject dataset, `DataQualityGuard`.

### Task 2: Implement subject-safe splitting
**Files:** `apace/data/split.py`, `tests/test_split.py`
**Produces:** subject-level fold and calibration/query split utilities with leakage assertions.

### Task 3: Implement pressure and calibration models
**Files:** `apace/models/encoder.py`, `apace/models/predictor.py`, `apace/models/calibration.py`, `tests/test_models.py`
**Produces:** encoder, probabilistic population head, DeepSets belief encoder, FiLM personalized head.

### Task 4: Implement oracle and counterfactual utilities
**Files:** `apace/oracle/dp.py`, `apace/oracle/counterfactual.py`, `tests/test_oracle.py`
**Produces:** finite-horizon dynamic programming and protocol-level gain records.

### Task 5: Implement distributional value model and policy
**Files:** `apace/models/value.py`, `apace/evaluation/policy.py`, `tests/test_value_policy.py`
**Produces:** quantile value estimator and query/stop policy.

### Task 6: Implement guards, metrics and training loops
**Files:** `apace/training/guards.py`, `apace/training/loops.py`, `apace/evaluation/metrics.py`, `tests/test_guards_metrics.py`
**Produces:** numerical guards, early stopping, research gates, bootstrap CIs and core metrics.

### Task 7: Build runnable synthetic pipeline
**Files:** `apace/pipeline.py`, `run_demo.py`, `configs/default.yaml`, `tests/test_pipeline.py`
**Produces:** fast end-to-end CPU demo that executes stages A-E and emits a metrics JSON file.

### Task 8: Package and document
**Files:** `pyproject.toml`, `requirements.txt`, `README.md`, `apace/__init__.py`
**Produces:** PyCharm/open-folder instructions, real manifest format, CLI usage and scientific caveats.

### Task 9: Verify
Run `pytest -q`, then `python run_demo.py --config configs/default.yaml --quick`, then compile all Python modules with `python -m compileall apace run_demo.py`.
