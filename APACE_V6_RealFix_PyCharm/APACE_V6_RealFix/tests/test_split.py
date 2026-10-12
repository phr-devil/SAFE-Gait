import numpy as np
from apace.data.types import Trial
from apace.data.split import leave_one_subject_out, split_calibration_query


def trial(subject, idx, condition="normal"):
    return Trial(subject, f"{subject}_{idx}", condition,
                 np.ones((2, 8), dtype=np.float32),
                 np.array([[0, 0], [1, 0]], dtype=np.float32),
                 np.array([float(idx)], dtype=np.float32))


def test_loso_has_no_subject_overlap():
    trials = [trial(f"S{s}", i) for s in range(3) for i in range(3)]
    train, test = leave_one_subject_out(trials, "S1")
    assert {t.subject_id for t in test} == {"S1"}
    assert "S1" not in {t.subject_id for t in train}


def test_calibration_query_split_is_disjoint():
    trials = [trial("S1", i, "normal" if i < 3 else "fast") for i in range(6)]
    cal, query = split_calibration_query(trials, calibration_fraction=0.5, seed=3)
    assert {t.trial_id for t in cal}.isdisjoint({t.trial_id for t in query})
    assert len(cal) > 0 and len(query) > 0
