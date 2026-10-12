import numpy as np
from apace.evaluation.baselines import MeanTargetBaseline, fixed_protocol_policy


def test_mean_target_baseline_repeats_training_mean():
    b = MeanTargetBaseline().fit(np.array([[1.,2.],[3.,4.]]))
    pred = b.predict(3)
    assert pred.shape == (3,2)
    assert np.allclose(pred[0], [2.,3.])


def test_fixed_protocol_policy_stops_after_budget():
    assert fixed_protocol_policy("normal", used_references=0, max_references=2) == "normal"
    assert fixed_protocol_policy("normal", used_references=2, max_references=2) == "STOP"
