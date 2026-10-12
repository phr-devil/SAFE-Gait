import numpy as np
import torch
from apace.training.guards import NumericalGuard, EarlyStopper, ResearchGate
from apace.evaluation.metrics import bootstrap_mean_ci, auec, normalized_regret


def test_numerical_guard_rejects_nonfinite_loss():
    guard = NumericalGuard()
    ok, reason = guard.check_loss(torch.tensor(float("nan")))
    assert not ok and "nonfinite" in reason


def test_early_stopper_respects_patience_and_min_epochs():
    es = EarlyStopper(patience=2, min_epochs=2, mode="min")
    assert not es.step(1.0, 1)
    assert not es.step(1.1, 2)
    assert es.step(1.2, 3)


def test_bootstrap_ci_contains_mean_for_constant_values():
    mean, lo, hi = bootstrap_mean_ci(np.ones(20), n_boot=100, seed=1)
    assert mean == lo == hi == 1.0


def test_auec_and_regret():
    assert auec([0,1,2], [10,6,5]) > 0
    assert normalized_regret(6.0, 4.0, 10.0) == 1/3


def test_research_gate_uses_upper_ci_for_no_go():
    gate = ResearchGate(min_effect=0.05)
    decision = gate.effect_gate(np.array([0.01, 0.02, 0.0, 0.01]), n_boot=200, seed=2)
    assert decision.status == "NO_GO"
