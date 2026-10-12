"""Fault injection for tests only. Never imported by the production CLI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from unittest.mock import patch
import apace.real_pipeline as pipeline
import apace.training.loops as loops
from apace.oracle.counterfactual import EncodedTrial
from apace.training.guards import ResearchGate
import run_real


def main():
    scenario,manifest,out=sys.argv[1:]
    torch.set_num_threads(1)
    args=['run_real.py','--manifest',manifest,'--out',out,'--device','cpu','--epochs','1']
    gate_probe=scenario.startswith(('g1_','g2_'))
    if not gate_probe or scenario in ('g1_pilot','g2_pilot'):
        args+=['--pilot']
    if scenario.startswith('g2_'):
        args+=['--max-refs','2' if scenario=='g2_budget2' else '1']
    sys.argv=args
    if gate_probe:
        # Training/encoding substitutes provide deterministic evidence, not research results.
        pipeline.train_population_model=lambda *a,**k:[0.]
        pipeline.train_calibration_model=lambda *a,**k:[0.]
        pipeline.validate_population=lambda *a,**k:0.
        pipeline.validate_calibration=lambda *a,**k:0.
        def encode(rows,encoder,*a,**k):
            dim=encoder.out[0].out_features
            low=scenario.startswith('g1_') and scenario!='g1_nonfinite'
            return [EncodedTrial(r.subject_id,r.trial_id,r.condition,np.zeros(dim,np.float32),
                                np.zeros_like(r.target) if low else r.target.copy(),r.target) for r in rows]
        pipeline.encode_trials=encode
        def reached_next(*a,**k):
            raise FloatingPointError('PROBE_REACHED_NEXT_STAGE')
        if scenario in ('g1_low','g1_pilot','g1_small','g1_nonfinite'):
            pipeline.train_calibration_model=reached_next
        else:
            pipeline._records=reached_next
        if scenario in ('g1_nonfinite','g2_nonfinite'):
            original=ResearchGate.effect_gate
            ResearchGate.effect_gate=lambda self,effects,**k:original(self,np.full(8,np.nan),**k)
            if scenario=='g2_nonfinite':
                # Keep G1 valid so G2 receives the invalid effect vector.
                calls=[0]
                def gate(self,effects,**kwargs):
                    calls[0]+=1
                    return original(self,effects if calls[0]==1 else np.full(8,np.inf),**kwargs)
                ResearchGate.effect_gate=gate
    elif scenario in ('loss_nan','loss_inf'):
        bad=float('nan') if scenario=='loss_nan' else float('inf')
        loops.gaussian_nll=lambda mean,*a:mean.sum()*0+bad
    elif scenario=='gradient_inf':
        original=pipeline.ProbabilisticPredictor
        class BadGradient(original):
            def __init__(self,*a,**k):
                super().__init__(*a,**k)
                next(self.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        pipeline.ProbabilisticPredictor=BadGradient
    elif scenario=='validation_nan':
        loops.validate_population=lambda *a,**k:float('nan')
    elif scenario=='evaluation_nan':
        original=pipeline._evaluate_subject_policy
        def invalid_result(*a,**k):
            result=original(*a,**k)
            result['policy_mae_normalized']=float('nan')
            return result
        pipeline._evaluate_subject_policy=invalid_result
    elif scenario=='poisoned_report':
        def invalid_report(*a,**k):
            k['on_epoch']({'training_loss':float('inf')})
            return [0.]
        pipeline.train_population_model=invalid_report
    elif scenario=='policy_nan':
        from apace.models.value import DistributionalValueModel
        original=DistributionalValueModel.forward
        def invalid_values(self,*a,**k):
            result=original(self,*a,**k)
            return result if self.training else torch.full_like(result,float('nan'))
        # Do not poison D's validation; inject only after D completes.
        evaluate=pipeline._evaluate_subject_policy
        def invalid_policy(*a,**k):
            with patch.object(DistributionalValueModel,'forward',invalid_values):
                return evaluate(*a,**k)
        pipeline._evaluate_subject_policy=invalid_policy
    elif scenario=='stop':
        from apace.models.value import DistributionalValueModel
        # Force a valid STOP decision while retaining real tiny training/evaluation.
        original=DistributionalValueModel.forward
        def negative(self,*a,**k):
            values=original(self,*a,**k)
            return values if self.training else -torch.ones_like(values)
        DistributionalValueModel.forward=negative
    run_real.main()


if __name__=='__main__': main()
