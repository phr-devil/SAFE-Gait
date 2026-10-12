from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from .types import Trial


def leave_one_subject_out(trials: Sequence[Trial], held_out_subject: str) -> tuple[list[Trial], list[Trial]]:
    train = [t for t in trials if t.subject_id != held_out_subject]
    test = [t for t in trials if t.subject_id == held_out_subject]
    if not test:
        raise ValueError(f"held-out subject {held_out_subject!r} not found")
    if held_out_subject in {t.subject_id for t in train}:
        raise RuntimeError("subject leakage detected")
    return train, test


def split_calibration_query(
    trials: Sequence[Trial], calibration_fraction: float = 0.5, seed: int = 0
) -> tuple[list[Trial], list[Trial]]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if len(trials) < 2:
        raise ValueError("at least two trials are required")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(trials))
    rng.shuffle(idx)
    n_cal = max(1, min(len(trials) - 1, int(round(len(trials) * calibration_fraction))))
    cal_idx = set(idx[:n_cal].tolist())
    cal = [t for i, t in enumerate(trials) if i in cal_idx]
    query = [t for i, t in enumerate(trials) if i not in cal_idx]
    if {t.trial_id for t in cal} & {t.trial_id for t in query}:
        raise RuntimeError("calibration/query leakage detected")
    return cal, query
