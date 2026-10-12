import random
import numpy as np
import torch
import pytest
from safe_gait.utils.repro import set_seed
from safe_gait.evaluation.bootstrap import subject_bootstrap_ci


def test_set_seed_repeats_python_numpy_torch():
    set_seed(123)
    a=(random.random(),np.random.rand(),torch.rand(1).item())
    set_seed(123)
    b=(random.random(),np.random.rand(),torch.rand(1).item())
    assert a==pytest.approx(b)


def test_subject_bootstrap_rejects_nonfinite():
    with pytest.raises(ValueError):
        subject_bootstrap_ci(np.array([[1.0],[np.nan]]))
