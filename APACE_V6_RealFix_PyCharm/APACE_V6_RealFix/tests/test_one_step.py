import numpy as np
import pytest
import torch


def observation(condition='SelfPace',offset=0.):
    from apace.evaluation.one_step import PressureObservation
    x=np.maximum(0,np.sin(np.arange(100)/8))[None,:]*np.arange(1,5)[:,None]+offset
    return PressureObservation(condition,x,.01,np.arange(3,dtype=np.float32),np.array([1.,2.]),np.array([.1,.2]))


def test_state_schema_and_unlabeled_contract():
    from apace.evaluation.one_step import build_state,feature_names,PressureObservation
    from dataclasses import fields
    actions=['HurriedPace','SelfPace']
    x=build_state([observation()],actions,3,2)
    names=feature_names(actions,3,2)
    assert len(x)==len(names) and np.isfinite(x).all()
    assert 'target' not in {f.name for f in fields(PressureObservation)}
    assert x[names.index('SelfPace.available')]==1 and x[names.index('HurriedPace.available')]==0
    assert any('embedding' in n for n in names) and any('prediction' in n for n in names)
    assert any('uncertainty' in n for n in names) and any('dominant_frequency' in n for n in names)
    assert any('flat_fraction' in n for n in names)
    y=build_state([observation(offset=2.)],actions,3,2)
    assert not np.allclose(x,y)


def test_state_is_order_invariant_and_pressure_only():
    from apace.evaluation.one_step import build_state
    rows=[observation(),observation('HurriedPace',2),observation(offset=1)]
    np.testing.assert_allclose(build_state(rows,['HurriedPace','SelfPace'],3,2),build_state(rows[::-1],['HurriedPace','SelfPace'],3,2))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_state_rejects_nonfinite_inputs(bad):
    from apace.evaluation.one_step import build_state
    from dataclasses import replace
    row=replace(observation(),prediction=np.array([bad,1.]))
    with pytest.raises(FloatingPointError):build_state([row],['SelfPace'],3,2)


def test_scaler_fit_uses_only_explicit_training_states():
    from apace.evaluation.one_step import StateScaler
    scaler=StateScaler.fit(np.array([[1.,2.],[3.,4.]]))
    np.testing.assert_allclose(scaler.mean,[2.,3.])
    test=scaler.transform(np.array([999.,999.]))
    np.testing.assert_allclose(scaler.mean,[2.,3.])
    assert np.isfinite(test).all() and np.max(abs(test))<=8


def test_one_step_records_use_budget_one_oracle_and_expected_outcomes():
    from test_g3 import problem
    from apace.evaluation.one_step import supervision
    study=problem({():2.,('a1',):.2,('a2',):1.8,('b',):.8},budget=1,lam=.1)
    records=supervision(study,np.ones(5,dtype=np.float32),'subject')
    assert len(records)==2
    assert {r.action:r.net_value for r in records}==pytest.approx({'A':.9,'B':1.1})
    assert all(r.state.shape==(5,) for r in records)


def test_decision_only_stop_sp_hp_and_ignores_unavailable_protocol():
    from apace.evaluation.one_step import choose_one_step
    class Model(torch.nn.Module):
        def forward(self,s,a,c):return torch.tensor([[1.,2. if int(a[0])==0 else -1.,3.]])
    actions={'HurriedPace':0,'SelfPace':1}
    assert choose_one_step(Model(),np.zeros(3),actions,['SelfPace'],'cpu')[0]=='STOP'
    assert choose_one_step(Model(),np.zeros(3),actions,list(actions),'cpu')[0]=='HP'
    class Bad(Model):
        def forward(self,s,a,c):return torch.tensor([[1.,2.,float('inf')]])
    with pytest.raises(FloatingPointError):choose_one_step(Bad(),np.zeros(3),actions,list(actions),'cpu')


def test_feature_projection_never_reads_targets_or_query_entries():
    from apace.evaluation.one_step import make_observations,build_state
    class Candidate:
        trial_id='cal';condition='SelfPace';embedding=np.array([1.,2.,3.],dtype=np.float32)
        @property
        def target(self):raise AssertionError('future reference label read')
    class Raw:
        pressure=observation().pressure;sample_interval_s=.01
        @property
        def target(self):raise AssertionError('raw target read')
    class Base(torch.nn.Module):
        def forward(self,x):return x[:,:2],torch.ones_like(x[:,:2])
    # No query entry exists in the accessible mapping.
    obs=make_observations([Candidate()],{'cal':Raw()},Base(),'cpu')
    state=build_state(obs,['SelfPace'],3,2)
    assert np.isfinite(state).all()


def test_real_state_builder_survives_target_poison_and_has_no_query_access():
    from run_one_step import build_unlabeled_state
    from apace.oracle.counterfactual import EncodedTrial
    from apace.data.types import Trial
    from apace.evaluation.oracle_headroom import ProtocolStudy
    rows=[EncodedTrial('s',name,condition,np.ones(3,dtype=np.float32),np.zeros(5),np.ones(5))
          for name,condition in [('sp','SelfPace'),('hp','HurriedPace'),('query','SelfPace')]]
    raw={r.trial_id:Trial('s',r.trial_id,r.condition,observation().pressure,np.zeros((4,2)),np.ones(5))
         for r in rows if r.trial_id!='query'}
    class Evaluator:
        def query_errors(self,*args):raise AssertionError('state requested true error')
    class Base(torch.nn.Module):
        def forward(self,x):return x[:,:1].expand(-1,5),torch.ones((len(x),5))
    study=ProtocolStudy(rows,Evaluator(),{'sp','hp'},{'query'},np.ones(5),list('abcde'),budget=1)
    state=build_unlabeled_state(study,raw,Base(),'cpu',['HurriedPace','SelfPace'],3)
    assert state.shape==(78,) and np.isfinite(state).all()


@pytest.mark.parametrize('decision,expected',[('NO_GO',2),('GO',0)])
def test_one_step_cli_stop_status(decision,expected):
    import subprocess
    import sys
    from pathlib import Path
    code=("import run_one_step; "
          f"run_one_step.run_study=lambda *a,**k: {{'gate':{{'decision':{decision!r},'allow_active_optimization':{decision=='GO'}}},'seeds':{{}}}}; "
          "run_one_step.main()")
    p=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert p.returncode==expected,p.stderr
