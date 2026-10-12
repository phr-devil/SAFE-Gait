import numpy as np
import pytest
import torch


def test_recovery_uses_ratio_of_subject_mean_gains_and_keeps_negative_values():
    from apace.evaluation.value_recovery import gain_recovery
    r=gain_recovery([1.,10.],[.5,5.],[.6,6.])
    assert r['value']==pytest.approx(.8)
    assert gain_recovery([1.,1.],[.5,.5],[1.1,1.1])['value']==pytest.approx(-.2)
    assert gain_recovery([1.,1.],[1.,1.],[.9,.9])['value'] is None


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_recovery_rejects_nonfinite(bad):
    from apace.evaluation.value_recovery import gain_recovery
    with pytest.raises(FloatingPointError):gain_recovery([1.,1.],[.5,.5],[bad,.8])


def test_g4_fails_even_with_80_percent_if_baseline_is_not_beaten():
    from apace.evaluation.value_recovery import summarize_seed
    rows=[{'subject':str(i),'fold':i%5+1,'zero_cost':1.,'oracle_cost':.5,
           'learned':{'cost_normalized':.59},'baselines':{k:{'cost_normalized':.58} for k in ['fixed1','random1','uncertainty1']}} for i in range(20)]
    r=summarize_seed(rows)
    assert r['recovery']['value']>.8 and r['decision']=='NO_GO'
    for row in rows:
        for val in row['baselines'].values():val['cost_normalized']=.7
    assert summarize_seed(rows)['decision']=='GO'
    for row in rows:row['learned']['cost_normalized']=.65
    assert summarize_seed(rows)['decision']=='NO_GO'


def test_g4_requires_fold_stability_and_unique_subjects():
    from apace.evaluation.value_recovery import summarize_seed
    rows=[{'subject':str(i),'fold':i%5+1,'zero_cost':1.,'oracle_cost':.5,
           'learned':{'cost_normalized':.6},'baselines':{k:{'cost_normalized':(.601 if i%5<3 else .5)} for k in ['fixed1','random1','uncertainty1']}} for i in range(20)]
    r=summarize_seed(rows)
    assert r['decision']=='NO_GO' and r['comparisons']['fixed1']['folds_better']==3
    with pytest.raises(ValueError,match='duplicate'):summarize_seed(rows+[rows[0]])


def test_expected_policy_does_not_choose_best_outcome_and_can_stop():
    from apace.evaluation.value_recovery import evaluate_policy
    from test_g3 import problem
    p=problem({():10.,('a1',):1.,('a2',):4.,('b',):9.,('a1','a2'):8.,('a1','b'):2.,('a2','b'):1.},lam=.1)
    def controller(context,remaining):
        if not context:return 'A',{}
        return ('B' if context[0].trial_id=='a2' else 'STOP'),{}
    r=evaluate_policy(p,controller)
    assert r['mae_normalized']==1 and r['reference_count']==1.5
    assert r['cost_normalized']==pytest.approx(1.15)


def test_uncertainty_uses_only_embeddings_not_reference_labels():
    from apace.evaluation.value_recovery import uncertainty_protocol
    from apace.oracle.counterfactual import EncodedTrial
    from dataclasses import replace
    rows=[EncodedTrial('s',str(i),a,np.array([e],dtype=np.float32),np.zeros(1),np.array([y])) for i,(a,e,y) in enumerate([('A',1,999),('A',3,999),('B',4,-999)])]
    class Base(torch.nn.Module):
        def forward(self,x):return x*0,x
    assert uncertainty_protocol(rows,Base(),'cpu')=='B'
    assert uncertainty_protocol([replace(r,target=-r.target) for r in rows],Base(),'cpu')=='B'


def test_value_controller_ignores_unacquired_targets_and_rejects_bad_values():
    from apace.evaluation.value_recovery import ValueController
    from apace.oracle.counterfactual import EncodedTrial
    class Evaluator:
        def state_vector(self,ctx):return np.array([sum(float(r.target[0]) for r in ctx)],dtype=np.float32)
    class Model(torch.nn.Module):
        def forward(self,s,a,lam):return torch.ones((1,3),device=s.device)
    row=EncodedTrial('s','a','A',np.zeros(1),np.zeros(1),np.array([999.]))
    controller=ValueController(Evaluator(),Model(),{'A':0},2,.005,'cpu')
    assert controller([], [row])[0]=='A'
    class Bad(Model):
        def forward(self,s,a,lam):return torch.tensor([[1.,1.,float('nan')]])
    with pytest.raises(FloatingPointError):ValueController(Evaluator(),Bad(),{'A':0},2,.005,'cpu')([], [row])


def test_three_seed_gate_cannot_select_only_favorable_seed():
    from run_g4 import combined_decision
    summaries={str(s):{'decision':'GO','reasons':[]} for s in [7,17,27]}
    assert combined_decision(summaries)['allow_active_optimization']
    summaries['17']={'decision':'NO_GO','reasons':['below80']}
    assert not combined_decision(summaries)['allow_active_optimization']
    with pytest.raises(ValueError):combined_decision({'7':summaries['7']})


@pytest.mark.parametrize('decision,expected',[('NO_GO',2),('GO',0)])
def test_g4_cli_failure_gate_exits_process(decision,expected):
    import subprocess
    import sys
    from pathlib import Path
    code=("import run_g4; "
          f"run_g4.run_study=lambda *a,**k: {{'gate':{{'decision':{decision!r},'allow_active_optimization':{decision=='GO'}}},'seeds':{{}}}}; "
          "run_g4.main()")
    proc=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert proc.returncode==expected,proc.stderr


def test_failed_export_cannot_leave_a_successful_gate_summary(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from run_g4 import publish_results
    summary={'status':'COMPLETED','gate':{'decision':'GO','allow_active_optimization':True}}
    def fail(*a,**k):raise OSError('injected export failure')
    monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    with pytest.raises(OSError):publish_results(tmp_path,summary,{'7':[{'subject':'s'}]},[])
    stored=json.loads((tmp_path/'summary.json').read_text())
    assert stored['status']=='HALTED' and not stored['gate']['allow_active_optimization']
