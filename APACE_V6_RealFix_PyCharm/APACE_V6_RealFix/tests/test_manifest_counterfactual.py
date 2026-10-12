import numpy as np
import pandas as pd
import torch

from apace.data.manifest import load_manifest
from apace.oracle.counterfactual import EncodedTrial, CounterfactualEvaluator, build_protocol_value_records
from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor


def test_manifest_loader_reads_npy_trials(tmp_path):
    p = np.ones((3, 10), dtype=np.float32)
    c = np.array([[0,0],[1,0],[2,0]], dtype=np.float32)
    np.save(tmp_path / "p.npy", p)
    np.save(tmp_path / "c.npy", c)
    pd.DataFrame([{
        "subject_id":"S1","trial_id":"T1","condition":"normal",
        "pressure_path":"p.npy","coords_path":"c.npy","speed":1.2,"stride":0.7,
    }]).to_csv(tmp_path / "manifest.csv", index=False)
    trials = load_manifest(tmp_path / "manifest.csv", target_columns=["speed","stride"])
    assert len(trials) == 1
    assert trials[0].pressure.shape == (3,10)
    assert np.allclose(trials[0].target, [1.2,0.7])


def test_counterfactual_records_keep_query_disjoint_and_emit_values():
    torch.manual_seed(0)
    belief = CalibrationBeliefEncoder(embedding_dim=4, target_dim=1, n_actions=2, belief_dim=3)
    personal = PersonalizedPredictor(embedding_dim=4, belief_dim=3, target_dim=1, hidden_dim=4)
    evaluator = CounterfactualEvaluator(belief, personal, action_to_index={"normal":0,"fast":1})
    trials = []
    for i, cond in enumerate(["normal","normal","fast","fast"]):
        emb = np.array([1.,0.,0.,0.], dtype=np.float32) + i * 0.01
        trials.append(EncodedTrial("S1", f"T{i}", cond, emb, np.array([0.0], np.float32), np.array([1.0+i*0.1], np.float32)))
    records = build_protocol_value_records(
        trials, evaluator, calibration_ids={"T0","T2"}, query_ids={"T1","T3"},
        lambda_values=[0.1], max_depth=1,
    )
    assert records
    assert all(r.subject_id == "S1" for r in records)
    assert {r.action for r in records}.issubset({"normal","fast"})
    assert all(np.isfinite(r.net_value) for r in records)
