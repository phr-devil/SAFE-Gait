import numpy as np

from apace.data.types import Trial
from apace.qc.checks import DataQualityGuard


def make_trial(flat=False, nan=False):
    pressure = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (4, 1))
    if flat:
        pressure[0] = 1.0
    if nan:
        pressure[1, 0] = np.nan
    return Trial(
        subject_id="S01",
        trial_id="T01",
        condition="normal",
        pressure=pressure,
        coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
        target=np.array([1.2, 0.7], dtype=np.float32),
    )


def test_data_quality_guard_accepts_clean_trial():
    result = DataQualityGuard(max_nan_fraction=0.0, max_flat_fraction=0.5).check_trial(make_trial())
    assert result.ok


def test_data_quality_guard_rejects_nan_trial():
    result = DataQualityGuard(max_nan_fraction=0.0).check_trial(make_trial(nan=True))
    assert not result.ok
    assert "nan_fraction" in result.reasons


def test_data_quality_guard_rejects_excessive_flat_channels():
    result = DataQualityGuard(max_flat_fraction=0.0).check_trial(make_trial(flat=True))
    assert not result.ok
    assert "flat_fraction" in result.reasons
