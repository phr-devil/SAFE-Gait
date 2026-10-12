from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
import torch

from apace.training.guards import ResearchGate, EarlyStopper
from apace.evaluation.metrics import bootstrap_mean_ci


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('fn', [bootstrap_mean_ci, ResearchGate().effect_gate])
def test_research_statistics_reject_nonfinite(bad, fn):
    with pytest.raises(FloatingPointError, match='nonfinite'):
        fn(np.array([.1, bad, .2]), n_boot=30)


@pytest.mark.parametrize('fn', [bootstrap_mean_ci, ResearchGate().effect_gate])
def test_research_statistics_reject_empty(fn):
    with pytest.raises(ValueError):
        fn(np.array([]))


@pytest.mark.parametrize('effects,expected', [(np.zeros(8),'NO_GO'), (np.full(8,.1),'GO')])
def test_research_finite_decisions_unchanged(effects, expected):
    assert ResearchGate().effect_gate(effects, n_boot=30).status == expected


@pytest.mark.parametrize('bad', [np.nan,np.inf,-np.inf])
def test_direct_early_stopper_rejects_nonfinite(bad):
    with pytest.raises(FloatingPointError, match='nonfinite'):
        EarlyStopper().step(bad, 1)


def test_training_reports_early_stop_and_restores_best(monkeypatch):
    import apace.training.loops as loops
    from apace.data.synthetic import generate_synthetic_trials
    from apace.models.encoder import PressureEncoder
    from apace.models.predictor import ProbabilisticPredictor
    torch.set_num_threads(1)
    rows=generate_synthetic_trials(n_subjects=3,trials_per_condition=2,n_sensors=4,timepoints=32)
    encoder=PressureEncoder(4,hidden_dim=8,out_dim=8)
    head=ProbabilisticPredictor(8,2,hidden_dim=8)
    values=iter([1.,1.1,1.2]); saved={}; callbacks=[]
    def validate(*args,**kwargs):
        if not saved:
            saved.update({k:v.detach().clone() for k,v in head.state_dict().items()})
        return next(values)
    monkeypatch.setattr(loops,'validate_population',validate)
    diagnostics={}
    history=loops.train_population_model(rows,encoder,head,epochs=8,min_epochs=2,patience=2,
                                         val_trials=rows,diagnostics=diagnostics,on_epoch=lambda d: callbacks.append(dict(d)))
    assert len(history)==3
    assert diagnostics['stop_reason']=='EARLY_STOPPING'
    assert diagnostics['best_epoch']==1 and diagnostics['best_weights_restored']
    assert len(callbacks)==3
    for k,v in head.state_dict().items():
        assert torch.equal(v,saved[k])


@pytest.fixture(scope='module')
def manifests(tmp_path_factory):
    from dataclasses import replace
    from apace.data.synthetic import generate_synthetic_trials
    from apace.data.weargait import export_trials
    root=tmp_path_factory.mktemp('guard_inputs')
    result={}
    for n in (3,10,55):
        rows=generate_synthetic_trials(n_subjects=n,trials_per_condition=2,n_sensors=4,timepoints=32,conditions=('slow',))
        result[n]=export_trials([replace(r,target=np.resize(r.target,5)) for r in rows],root/str(n))
    return result


@pytest.mark.parametrize('scenario,n,stage,reason', [
    ('loss_nan',3,'A','nonfinite_loss'),('loss_inf',3,'A','nonfinite_loss'),
    ('gradient_inf',3,'A','nonfinite_population_gradient'),
    ('validation_nan',3,'A','nonfinite_validation_metric'),
    ('evaluation_nan',3,'E','nonfinite_report_values'),
    ('poisoned_report',3,'A','nonfinite_report_values'),
    ('policy_nan',3,'E','nonfinite_policy_values'),
    ('g1_low',55,'G1','G1_NO_GO'),('g1_nonfinite',55,'G1','nonfinite_research_effects'),
    ('g1_pilot',55,'B','PROBE_REACHED_NEXT_STAGE'),('g1_small',10,'B','PROBE_REACHED_NEXT_STAGE'),
    ('g2_low',55,'G2','G2_NO_GO'),('g2_nonfinite',55,'G2','nonfinite_research_effects'),
    ('g2_budget2',55,'C','PROBE_REACHED_NEXT_STAGE'),('g2_pilot',55,'C','PROBE_REACHED_NEXT_STAGE'),
])
def test_cli_halt_and_gate_activation(manifests,tmp_path,scenario,n,stage,reason):
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/scenario
    proc=subprocess.run([sys.executable,str(root/'tests/guard_cli_probe.py'),scenario,str(manifests[n]),str(out)],
                        cwd=root,capture_output=True,text=True,timeout=90)
    assert proc.returncode==2,proc.stdout+proc.stderr
    report=json.loads((out/'real_metrics.json').read_text())
    assert report['status']=='HALTED' and report['halt_stage']==stage
    assert reason in report['halt_reason']
    assert report['stages'][stage]['status']=='HALTED'
    assert not (out/'checkpoint.pt').exists()
    if scenario=='poisoned_report':
        assert report['invalid_numeric_fields']
        assert report['stages']['A']['training']['training_loss'] is None
    if stage in ('A','G1'): assert 'B' not in report['stages']
    if stage in ('B','G2'): assert 'C' not in report['stages']
    if reason in ('G1_NO_GO','G2_NO_GO'):
        assert report['stages'][stage]['decision']=='NO_GO'
        assert report['stages'][stage]['upper_ci']<.05


def test_stop_is_success_not_failed_training(manifests,tmp_path):
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,str(root/'tests/guard_cli_probe.py'),'stop',str(manifests[3]),str(tmp_path)],
                        cwd=root,capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,proc.stdout+proc.stderr
    report=json.loads((tmp_path/'real_metrics.json').read_text())
    assert report['status']=='PILOT_ONLY'
    assert report['stages']['E']['results'][0]['reference_count']==0
    assert (tmp_path/'checkpoint.pt').exists()


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_oracle_cannot_hide_nonfinite_candidate_behind_stop(bad):
    from apace.real_pipeline import _one_shot_oracle_gain
    class Evaluator:
        def query_error(self,context,query): return bad if context else 1.
    with pytest.raises(FloatingPointError,match='nonfinite'):
        _one_shot_oracle_gain(Evaluator(),['reference'],['query'])


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_policy_refuses_nonfinite_quantiles(bad):
    from apace.pipeline import _evaluate_subject_policy
    from apace.oracle.counterfactual import EncodedTrial
    rows=[EncodedTrial('s',str(i),'walk',np.zeros(2),np.zeros(1),np.ones(1)) for i in range(4)]
    class Evaluator:
        def state_vector(self,context): return np.zeros(2)
        def query_error(self,context,query): return 1.
    class Model(torch.nn.Module):
        def forward(self,*a): return torch.tensor([[bad,1.,1.]])
    with pytest.raises(FloatingPointError,match='nonfinite_policy_values'):
        _evaluate_subject_policy(rows,Evaluator(),Model(),{'walk':0},0.,1,'cpu')


@pytest.mark.parametrize('fault',['missing','duplicate','too_few','pressure_nan','target_inf'])
def test_cli_g0_errors_exit_before_training(manifests,tmp_path,fault):
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    source=manifests[3]
    frame=pd.read_csv(source)
    for c in ('pressure_path','coords_path'):
        frame[c]=frame[c].map(lambda v:str(source.parent/v))
    if fault=='duplicate':frame.loc[1,'trial_id']=frame.loc[0,'trial_id']
    if fault=='too_few':frame=frame.iloc[:4]
    if fault=='pressure_nan':
        broken=tmp_path/'broken.npy';np.save(broken,np.full((4,32),np.nan,dtype=np.float32))
        frame.loc[0,'pressure_path']=str(broken)
    if fault=='target_inf':frame.loc[0,'velocity_cm_s']=np.inf
    manifest=tmp_path/'input.csv';frame.to_csv(manifest,index=False)
    if fault=='missing':manifest=tmp_path/'missing.csv'
    out=tmp_path/'out'
    proc=subprocess.run([sys.executable,str(root/'run_real.py'),'--manifest',str(manifest),'--out',str(out),
                         '--pilot','--epochs','1','--device','cpu'],cwd=root,capture_output=True,text=True,timeout=90)
    assert proc.returncode==2,proc.stdout+proc.stderr
    report=json.loads((out/'real_metrics.json').read_text())
    assert report['halt_stage']=='G0' and 'A' not in report['stages']
    assert not (out/'checkpoint.pt').exists()
