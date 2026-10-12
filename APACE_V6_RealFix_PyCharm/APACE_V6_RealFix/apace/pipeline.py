from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch

from apace.data.synthetic import generate_synthetic_trials
from apace.qc.checks import DataQualityGuard
from apace.models.encoder import PressureEncoder
from apace.models.predictor import ProbabilisticPredictor
from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor
from apace.models.value import DistributionalValueModel
from apace.training.loops import train_population_model, encode_trials, train_calibration_model, train_value_model
from apace.oracle.counterfactual import CounterfactualEvaluator, build_protocol_value_records, planning_state
from apace.evaluation.policy import choose_action
from apace.training.guards import ResearchGate


def _split_subjects(trials, seed: int):
    subjects = sorted({t.subject_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(2, len(subjects) // 5)
    n_val = max(1, len(subjects) // 10)
    test = set(subjects[:n_test])
    val = set(subjects[n_test:n_test+n_val])
    train = set(subjects[n_test+n_val:])
    return [t for t in trials if t.subject_id in train], [t for t in trials if t.subject_id in val], [t for t in trials if t.subject_id in test]


def _cal_query_ids(rows):
    by_cond = defaultdict(list)
    for r in sorted(rows, key=lambda x: x.trial_id):
        by_cond[r.condition].append(r.trial_id)
    cal, query = set(), set()
    for ids in by_cond.values():
        cut = max(1, len(ids) // 2)
        cal.update(ids[:cut])
        query.update(ids[cut:])
    return cal, query


def _evaluate_subject_policy(rows, evaluator, value_model, action_to_index, lam: float, max_refs: int, device, *, seed=0, return_details=False):
    rng = np.random.default_rng(seed)
    lookup = {r.trial_id: r for r in rows}
    cal_ids, query_ids = _cal_query_ids(rows)
    query = [lookup[i] for i in sorted(query_ids)]
    remaining = set(cal_ids)
    context = []
    base_error = evaluator.query_error([], query)
    for step in range(max_refs):
        features = planning_state(evaluator.state_vector(context), [lookup[i] for i in sorted(remaining)],
                                  action_to_index, max_refs-step)
        state = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        net_values = {}
        for action, a_idx in action_to_index.items():
            if not any(lookup[i].condition == action for i in remaining):
                continue
            with torch.no_grad():
                q = value_model(state, torch.tensor([a_idx], device=device), torch.tensor([[lam]], device=device))
            if not bool(torch.isfinite(q).all()):
                raise FloatingPointError('nonfinite_policy_values')
            net_values[action] = float(q[0, 1].cpu())
        action, _ = choose_action(net_values)
        if action == "STOP":
            break
        candidates = sorted(i for i in remaining if lookup[i].condition == action)
        chosen = candidates[int(rng.integers(len(candidates)))]
        context.append(lookup[chosen])
        remaining.remove(chosen)
    calibrated_error = evaluator.query_error(context, query)
    if return_details:
        return {'zero_shot_mae_normalized': base_error, 'policy_mae_normalized': calibrated_error,
                'reference_count': len(context), 'reference_ids': [r.trial_id for r in context],
                'query_ids': sorted(query_ids)}
    return base_error, calibrated_error, len(context)


def run_synthetic_pipeline(output_dir: str | Path, seed: int = 7, quick: bool = False, device: str | None = None) -> dict[str, float | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_subjects = 10 if quick else 18
    trials = generate_synthetic_trials(n_subjects=n_subjects, trials_per_condition=4, n_sensors=8, timepoints=40, seed=seed)
    qc = DataQualityGuard(max_nan_fraction=0.0, max_flat_fraction=0.4)
    trials = [t for t in trials if qc.check_trial(t).ok]
    train_trials, val_trials, test_trials = _split_subjects(trials, seed)
    conditions = sorted({t.condition for t in trials})
    action_to_index = {c: i for i, c in enumerate(conditions)}
    n_targets = len(trials[0].target)
    encoder = PressureEncoder(n_sensors=8, hidden_dim=24, out_dim=32)
    base = ProbabilisticPredictor(32, n_targets, hidden_dim=32)
    train_population_model(train_trials, encoder, base, epochs=16 if quick else 35, batch_size=32, device=device, seed=seed)
    encoded_all = encode_trials(trials, encoder, base, device=device)
    train_subjects = {t.subject_id for t in train_trials}
    test_subjects = {t.subject_id for t in test_trials}
    encoded_train = [t for t in encoded_all if t.subject_id in train_subjects]
    encoded_test = [t for t in encoded_all if t.subject_id in test_subjects]
    for p in encoder.parameters(): p.requires_grad_(False)
    for p in base.parameters(): p.requires_grad_(False)

    belief = CalibrationBeliefEncoder(32, n_targets, len(conditions), belief_dim=16)
    personal = PersonalizedPredictor(32, 16, n_targets, hidden_dim=32)
    train_calibration_model(encoded_train, belief, personal, action_to_index,
                            epochs=20 if quick else 35,
                            episodes_per_epoch=50 if quick else 80,
                            max_context=2, device=device, seed=seed)
    evaluator = CounterfactualEvaluator(belief.to(device), personal.to(device), action_to_index, device=device)

    records = []
    by_subject = defaultdict(list)
    for row in encoded_train:
        by_subject[row.subject_id].append(row)
    for rows in by_subject.values():
        cal_ids, query_ids = _cal_query_ids(rows)
        records.extend(build_protocol_value_records(
            rows, evaluator, cal_ids, query_ids,
            lambda_values=[0.0, 0.002, 0.005] if quick else [0.0, 0.002, 0.005, 0.01],
            max_depth=2,
        ))
    value_model = DistributionalValueModel(16 + 1 + len(conditions), len(conditions), hidden_dim=32)
    train_value_model(records, value_model, action_to_index, epochs=25 if quick else 50, batch_size=64, device=device, seed=seed)
    value_model.to(device).eval()

    test_by_subject = defaultdict(list)
    for row in encoded_test:
        test_by_subject[row.subject_id].append(row)
    base_errors, cal_errors, refs = [], [], []
    for rows in test_by_subject.values():
        b, c, n = _evaluate_subject_policy(rows, evaluator, value_model, action_to_index, lam=0.005, max_refs=2, device=torch.device(device), seed=seed)
        base_errors.append(b); cal_errors.append(c); refs.append(n)
    effects = (np.asarray(base_errors) - np.asarray(cal_errors)) / np.maximum(np.asarray(base_errors), 1e-8)
    decision = ResearchGate(min_effect=0.05).effect_gate(effects, n_boot=300 if quick else 2000, seed=seed)
    metrics = {
        "base_mae": float(np.mean(base_errors)),
        "calibrated_mae": float(np.mean(cal_errors)),
        "mean_references": float(np.mean(refs)),
        "relative_improvement": float(np.mean(effects)),
        "guard_status": decision.status,
        "device": str(device),
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "n_value_records": len(records),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
