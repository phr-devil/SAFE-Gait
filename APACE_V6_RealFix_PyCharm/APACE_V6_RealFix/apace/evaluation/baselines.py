from __future__ import annotations
import numpy as np


class MeanTargetBaseline:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "MeanTargetBaseline":
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 2 or len(arr) == 0:
            raise ValueError("y must be a non-empty 2D array")
        self.mean_ = arr.mean(axis=0)
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("baseline is not fitted")
        return np.repeat(self.mean_[None, :], int(n), axis=0)


def fixed_protocol_policy(protocol: str, used_references: int, max_references: int) -> str:
    return "STOP" if used_references >= max_references else protocol
