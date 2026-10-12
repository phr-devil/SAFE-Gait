from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import numpy as np
import torch
from torch import nn

from apace.data.types import Trial
from apace.data.batching import collate_passes
from apace.models.encoder import PressureEncoder
from apace.models.predictor import ProbabilisticPredictor, gaussian_nll
from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor
from apace.models.value import DistributionalValueModel, quantile_loss
from apace.oracle.counterfactual import EncodedTrial, ValueRecord
from apace.training.guards import NumericalGuard, EarlyStopper
from apace.training.validation import validate_population, validate_calibration, validate_value, BestValidationCheckpoint


def _epoch_diagnostics(history, check, metric, epoch, epochs, diagnostics, on_epoch):
    stop = check.check(metric, epoch) if check else False
    snapshot = {'epochs_run': epoch, 'epoch_limit': epochs, 'training_loss': history[-1],
                'validation_metric': metric, 'best_epoch': check.best_epoch if check else None,
                'best_validation_metric': float(check.stopping.best) if check else None,
                'best_weights_restored': False,
                'stop_reason': 'EARLY_STOPPING' if stop else ('EPOCH_LIMIT' if epoch == epochs else 'RUNNING')}
    if diagnostics is not None: diagnostics.update(snapshot)
    if on_epoch is not None: on_epoch(snapshot)
    return stop


def _restore_validation(check, diagnostics):
    if check:
        check.restore()
        if diagnostics is not None: diagnostics['best_weights_restored'] = True


def _tensor_batch(trials: Sequence[Trial], device: torch.device):
    return collate_passes(trials, device=device)


def train_population_model(
    trials: Sequence[Trial],
    encoder: PressureEncoder,
    predictor: ProbabilisticPredictor,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    val_trials: Sequence | None = None,
    patience: int = 25,
    min_epochs: int = 40,
    diagnostics: dict | None = None,
    on_epoch=None,
) -> list[float]:
    device = torch.device(device)
    encoder.to(device)
    predictor.to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    guard = NumericalGuard()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    check = BestValidationCheckpoint([encoder, predictor],patience=patience,min_epochs=min_epochs) if val_trials is not None else None
    n = len(trials)
    if n == 0:
        raise ValueError("population training requires trials")
    for epoch in range(1,epochs+1):
        order = rng.permutation(n)
        losses = []
        encoder.train(); predictor.train()
        for start in range(0, n, batch_size):
            batch = [trials[int(i)] for i in order[start:start + batch_size]]
            x, coords, y, valid, duration_s = _tensor_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            h = encoder(x, coords, valid, duration_s)
            mean, scale = predictor(h)
            loss = gaussian_nll(mean, scale, y)
            ok, reason = guard.check_loss(loss)
            if not ok:
                raise FloatingPointError(reason)
            loss.backward()
            grad_norm=guard.clip(params)
            if not np.isfinite(grad_norm): raise FloatingPointError("nonfinite_population_gradient")
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        metric=validate_population(val_trials,encoder,predictor,device=device) if check else None
        if _epoch_diagnostics(history,check,metric,epoch,epochs,diagnostics,on_epoch): break
    _restore_validation(check,diagnostics)
    return history


@torch.no_grad()
def encode_trials(
    trials: Sequence[Trial],
    encoder: PressureEncoder,
    predictor: ProbabilisticPredictor,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> list[EncodedTrial]:
    device = torch.device(device)
    encoder.to(device).eval(); predictor.to(device).eval()
    result: list[EncodedTrial] = []
    for start in range(0, len(trials), batch_size):
        batch = list(trials[start:start + batch_size])
        x, coords, _, valid, duration_s = _tensor_batch(batch, device)
        h = encoder(x, coords, valid, duration_s)
        mean, _ = predictor(h)
        for t, e, m in zip(batch, h.cpu().numpy(), mean.cpu().numpy()):
            result.append(EncodedTrial(t.subject_id, t.trial_id, t.condition, e.astype(np.float32), m.astype(np.float32), t.target.astype(np.float32)))
    return result


def train_calibration_model(
    trials: Sequence[EncodedTrial],
    belief_encoder: CalibrationBeliefEncoder,
    personalized_predictor: PersonalizedPredictor,
    action_to_index: dict[str, int],
    epochs: int = 40,
    episodes_per_epoch: int = 64,
    max_context: int = 3,
    lr: float = 2e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    val_trials: Sequence | None = None,
    patience: int = 25,
    min_epochs: int = 40,
    diagnostics: dict | None = None,
    on_epoch=None,
) -> list[float]:
    device = torch.device(device)
    belief_encoder.to(device); personalized_predictor.to(device)
    by_subject: dict[str, list[EncodedTrial]] = defaultdict(list)
    for t in trials:
        by_subject[t.subject_id].append(t)
    subjects = [s for s, rows in by_subject.items() if len(rows) >= 2]
    if not subjects:
        raise ValueError("calibration training needs >=2 trials per subject")
    params = list(belief_encoder.parameters()) + list(personalized_predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    guard = NumericalGuard()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    check = BestValidationCheckpoint([belief_encoder,personalized_predictor],patience=patience,min_epochs=min_epochs) if val_trials is not None else None
    for epoch in range(1,epochs+1):
        belief_encoder.train(); personalized_predictor.train()
        losses = []
        for _ep in range(episodes_per_epoch):
            sid = str(rng.choice(subjects))
            rows = by_subject[sid]
            q_idx = int(rng.integers(0, len(rows)))
            query = rows[q_idx]
            pool = [r for i, r in enumerate(rows) if i != q_idx]
            k = int(rng.integers(1, min(max_context, len(pool)) + 1))
            chosen = rng.choice(len(pool), size=k, replace=False)
            context = [pool[int(i)] for i in np.atleast_1d(chosen)]
            emb = torch.tensor(np.stack([r.embedding for r in context]), dtype=torch.float32, device=device).unsqueeze(0)
            residual = torch.tensor(np.stack([r.target - r.base_mean for r in context]), dtype=torch.float32, device=device).unsqueeze(0)
            actions = torch.tensor([[action_to_index[r.condition] for r in context]], dtype=torch.long, device=device)
            mask = torch.ones((1, k), dtype=torch.bool, device=device)
            belief = belief_encoder(emb, residual, actions, mask)
            q_emb = torch.tensor(query.embedding, dtype=torch.float32, device=device).unsqueeze(0)
            q_target = torch.tensor(query.target, dtype=torch.float32, device=device).unsqueeze(0)
            q_base = torch.tensor(query.base_mean, dtype=torch.float32, device=device).unsqueeze(0)
            mean, scale = personalized_predictor.predict(q_emb, belief, q_base)
            loss = gaussian_nll(mean, scale, q_target)
            ok, reason = guard.check_loss(loss)
            if not ok:
                raise FloatingPointError(reason)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm=guard.clip(params)
            if not np.isfinite(grad_norm): raise FloatingPointError("nonfinite_calibration_gradient")
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        metric=validate_calibration(val_trials,belief_encoder,personalized_predictor,action_to_index,device=device,max_context=max_context) if check else None
        if _epoch_diagnostics(history,check,metric,epoch,epochs,diagnostics,on_epoch): break
    _restore_validation(check,diagnostics)
    return history


def train_value_model(
    records: Sequence[ValueRecord],
    model: DistributionalValueModel,
    action_to_index: dict[str, int],
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 2e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    val_records: Sequence[ValueRecord] | None = None,
    patience: int = 25,
    min_epochs: int = 40,
    diagnostics: dict | None = None,
    on_epoch=None,
) -> list[float]:
    if not records:
        raise ValueError("value training requires records")
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    guard = NumericalGuard()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    check = BestValidationCheckpoint([model],patience=patience,min_epochs=min_epochs) if val_records is not None else None
    for epoch in range(1,epochs+1):
        order = rng.permutation(len(records))
        losses = []
        model.train()
        for start in range(0, len(records), batch_size):
            rows = [records[int(i)] for i in order[start:start + batch_size]]
            state = torch.tensor(np.stack([r.state for r in rows]), dtype=torch.float32, device=device)
            action = torch.tensor([action_to_index[r.action] for r in rows], dtype=torch.long, device=device)
            lam = torch.tensor([[r.cost_weight] for r in rows], dtype=torch.float32, device=device)
            target = torch.tensor([r.net_value for r in rows], dtype=torch.float32, device=device)
            pred = model(state, action, lam)
            loss = quantile_loss(pred, target)
            ok, reason = guard.check_loss(loss)
            if not ok:
                raise FloatingPointError(reason)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm=guard.clip(model.parameters())
            if not np.isfinite(grad_norm): raise FloatingPointError("nonfinite_value_gradient")
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        metric=validate_value(val_records,model,action_to_index,device=device) if check else None
        if _epoch_diagnostics(history,check,metric,epoch,epochs,diagnostics,on_epoch): break
    _restore_validation(check,diagnostics)
    return history
