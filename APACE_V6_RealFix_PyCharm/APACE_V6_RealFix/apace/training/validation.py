"""Held-out-SUBJECT validation and checkpoint selection (never test-subject selection)."""
from __future__ import annotations
from collections import defaultdict
import copy
import numpy as np
import torch
from apace.data.batching import collate_passes
from apace.models.predictor import gaussian_nll
from apace.oracle.counterfactual import CounterfactualEvaluator
from apace.training.guards import EarlyStopper
from apace.models.value import quantile_loss, QUANTILES


def validate_population(rows, encoder, head, device='cpu', batch_size=8):
    encoder.eval(); head.eval(); vals=defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, c, y, valid, seconds = collate_passes(rows[start:start+batch_size], device=device)
            mu, sig = head(encoder(x,c,valid,seconds))
            var = sig.square().clamp_min(1e-8)
            losses = (0.5 * ((y-mu).square()/var + torch.log(var))).mean(1).cpu().tolist()
            for row, loss in zip(rows[start:start+batch_size], losses):
                vals[row.subject_id].append(loss)
    if not vals:
        raise ValueError('empty subject-level population validation')
    return float(np.mean([np.mean(v) for v in vals.values()]))


def validate_calibration(rows, belief, personal, action_to_index, device='cpu', max_context=2):
    from apace.pipeline import _cal_query_ids
    if max_context < 1: raise ValueError('max_context must be positive')
    belief.eval(); personal.eval()
    evaluator=CounterfactualEvaluator(belief,personal,action_to_index,device=device)
    by_subject=defaultdict(list)
    for r in rows: by_subject[r.subject_id].append(r)
    errors=[]
    for subject_rows in by_subject.values():
        lookup={r.trial_id:r for r in subject_rows}
        cal_ids,qry_ids=_cal_query_ids(subject_rows)
        cal=[lookup[i] for i in sorted(cal_ids)]
        qry=[lookup[i] for i in sorted(qry_ids)]
        if cal and qry:
            rng=np.random.default_rng(0)
            order=rng.permutation(len(cal))
            errors.append(np.mean([evaluator.query_error([cal[int(i)] for i in order[:k]],qry)
                                   for k in range(1,min(max_context,len(cal))+1)]))
    if not errors:
        raise ValueError('no independent calibration/query pairs in validation subjects')
    return float(np.mean(errors))


def validate_value(records, model, action_to_index, device='cpu', batch_size=128):
    model.eval(); vals=defaultdict(list)
    with torch.no_grad():
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            states=torch.tensor(np.stack([r.state for r in batch]),dtype=torch.float32,device=device)
            actions=torch.tensor([action_to_index[r.action] for r in batch],device=device)
            costs=torch.tensor([[r.cost_weight] for r in batch],dtype=torch.float32,device=device)
            target=torch.tensor([r.net_value for r in batch],dtype=torch.float32,device=device)
            pred=model(states,actions,costs)
            err=target[:,None]-pred
            q=QUANTILES.to(pred.device,pred.dtype)[None,:]
            losses=torch.maximum(q*err,(q-1)*err).mean(1).cpu().tolist()
            for row,loss in zip(batch,losses): vals[row.subject_id].append(loss)
    if not vals: raise ValueError('no validation value records')
    return float(np.mean([np.mean(v) for v in vals.values()]))


class BestValidationCheckpoint:
    def __init__(self, modules, *, patience=25, min_epochs=40, min_delta=1e-4):
        self.modules=tuple(modules)
        self.stopping=EarlyStopper(patience=patience,min_epochs=min_epochs,min_delta=min_delta)
        self.best_states=None
        self.best_epoch=None
        self.history=[]

    def check(self, metric, epoch):
        if not np.isfinite(metric): raise FloatingPointError('nonfinite_validation_metric')
        old_best=self.stopping.best
        stop=self.stopping.step(metric, epoch)
        if self.stopping.best != old_best:
            self.best_states=[copy.deepcopy(m.state_dict()) for m in self.modules]
            self.best_epoch=epoch
        self.history.append(float(metric))
        return stop

    def restore(self):
        if self.best_states is None: raise RuntimeError('no validation checkpoint to restore')
        for m,weights in zip(self.modules,self.best_states):
            m.load_state_dict(weights)
