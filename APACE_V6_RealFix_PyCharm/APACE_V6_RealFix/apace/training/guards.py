from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


class NumericalGuard:
    def __init__(self, max_grad_norm: float = 5.0) -> None:
        self.max_grad_norm = max_grad_norm

    def check_loss(self, loss: torch.Tensor) -> tuple[bool, str]:
        if not torch.isfinite(loss).all():
            return False, "nonfinite_loss"
        return True, "ok"

    def clip(self, parameters) -> float:
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        return float(norm)


class EarlyStopper:
    def __init__(self, patience: int = 25, min_epochs: int = 40, mode: str = "min", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.bad = 0

    def step(self, metric: float, epoch: int) -> bool:
        if not np.isfinite(metric):
            raise FloatingPointError('nonfinite_early_stopping_metric')
        improved = metric < self.best - self.min_delta if self.mode == "min" else metric > self.best + self.min_delta
        if improved:
            self.best, self.bad = metric, 0
        else:
            self.bad += 1
        return epoch >= self.min_epochs and self.bad >= self.patience


@dataclass(frozen=True)
class GateDecision:
    status: str
    mean: float
    lower: float
    upper: float


class ResearchGate:
    def __init__(self, min_effect: float = 0.05) -> None:
        if not np.isfinite(min_effect):
            raise ValueError('min_effect must be finite')
        self.min_effect = min_effect

    def effect_gate(self, effects: np.ndarray, n_boot: int = 2000, seed: int = 0) -> GateDecision:
        from apace.evaluation.metrics import bootstrap_mean_ci
        mean, lo, hi = bootstrap_mean_ci(np.asarray(effects, dtype=float), n_boot=n_boot, seed=seed)
        if hi < self.min_effect:
            status = "NO_GO"
        elif lo > 0:
            status = "GO"
        else:
            status = "CONTINUE_PILOT"
        return GateDecision(status, mean, lo, hi)
