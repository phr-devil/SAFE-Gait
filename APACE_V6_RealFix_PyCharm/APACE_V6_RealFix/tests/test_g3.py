import numpy as np
import pytest
from apace.oracle.counterfactual import EncodedTrial


def problem(errors, budget=2, lam=0.):
    from apace.evaluation.oracle_headroom import ProtocolStudy
    rows=[EncodedTrial('s',i,a,np.zeros(2),np.zeros(1),np.ones(1))
          for i,a in [('a1','A'),('a2','A'),('b','B'),('q','A')]]
    class Evaluator:
        def query_errors(self,context,query):
            assert [r.trial_id for r in query]==['q']
            return np.array([[errors[tuple(sorted(r.trial_id for r in context))]]])
    return ProtocolStudy(rows,Evaluator(),{'a1','a2','b'},{'q'},np.array([10.]),['speed'],budget=budget,lam=lam)


def test_oracle_averages_unknown_protocol_outcomes_not_best_trial():
    p=problem({():2.,('a1',):.2,('a2',):1.8,('b',):.8},budget=1)
    o=p.oracle()
    assert o['first_action']=='B' and o['mae_normalized']==pytest.approx(.8)
    assert p.fixed(('A',))['mae_normalized']==pytest.approx(1.)
    assert p.random(1)['mae_normalized']==pytest.approx(.9)
    assert o['mae_physical']['speed']==pytest.approx(8.)


def test_oracle_adapts_stop_after_outcome_and_accounts_for_reference_cost():
    p=problem({():10.,('a1',):1.,('a2',):4.,('b',):9.,('a1','a2'):8.,('a1','b'):2.,('a2','b'):1.},lam=.1)
    o=p.oracle()
    assert o['first_action']=='A'
    assert o['mae_normalized']==pytest.approx(1.)
    assert o['reference_count']==pytest.approx(1.5)
    assert o['cost_normalized']==pytest.approx(1.15)
    assert o['cost_normalized']<p.fixed(('A','B'))['cost_normalized']
    assert any(d['context']==['a1'] and d['action']=='STOP' for d in o['decisions'])


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_oracle_rejects_bad_candidate_even_when_other_action_wins(bad):
    p=problem({():2.,('a1',):.2,('a2',):bad,('b',):.1},budget=1)
    with pytest.raises(FloatingPointError,match='nonfinite'):
        p.oracle()


def test_fixed_length_cannot_stop_just_because_second_reference_worsens():
    p=problem({():10.,('a1',):1.,('a2',):1.,('b',):1.,('a1','a2'):3.,('a1','b'):3.,('a2','b'):3.})
    assert p.fixed(('A','B'))['reference_count']==2
    assert p.fixed(('A','B'))['mae_normalized']==3
    assert p.oracle()['reference_count']==1


def test_fixed_two_uses_label_blind_fallback_if_protocol_exhausted():
    p=problem({():10.,('a1',):1.,('a2',):1.,('b',):1.,('a1','a2'):3.,('a1','b'):3.,('a2','b'):3.})
    assert p.fixed(('B','B'))['reference_count']==2


def test_g3_requires_meaningful_advantage_over_all_four_baselines():
    from apace.evaluation.oracle_headroom import decide_g3
    names=['fixed_1','fixed_2','random_1','random_2']
    rows=[{'subject':str(i),'oracle':{'cost_normalized':.96},
           'comparisons':{k:{'cost_normalized':1.} for k in names}} for i in range(12)]
    result=decide_g3(rows)
    assert result['decision']=='NO_GO' and not result['allow_active_optimization']
    for r in rows:r['oracle']['cost_normalized']=.8
    assert decide_g3(rows)['decision']=='GO'
    for r in rows:r['comparisons']['fixed_1']['cost_normalized']=.8
    assert decide_g3(rows)['decision']=='NO_GO'
    with pytest.raises(ValueError,match='duplicate'):decide_g3(rows+[rows[0]])


def test_fixed_selection_is_exact_length_and_subject_weighted():
    from apace.evaluation.oracle_headroom import select_fixed
    p=problem({():2.,('a1',):.2,('a2',):1.8,('b',):.8},budget=1)
    result=select_fixed([p],['A','B'],1)
    assert result['schedule']==['B'] and result['source']=='validation_only'
    assert all(len(c['schedule'])==1 for c in result['candidates'])


def test_g3_does_not_treat_insufficient_evidence_as_permission():
    from apace.evaluation.oracle_headroom import decide_g3
    rows=[{'subject':str(i),'oracle':{'cost_normalized':x},
           'comparisons':{k:{'cost_normalized':1.} for k in ['fixed_1','fixed_2','random_1','random_2']}}
          for i,x in enumerate([1.,1.,1.,.8])]
    result=decide_g3(rows)
    assert result['decision']=='INCONCLUSIVE' and not result['allow_active_optimization']


def test_frozen_checkpoint_study_replays_all_subjects_and_refuses_overwrite(tmp_path):
    import json
    import torch
    from dataclasses import replace
    from apace.data.synthetic import generate_synthetic_trials
    from apace.data.weargait import export_trials
    from apace.real_pipeline import run_real_pipeline
    from apace.evaluation.cross_validation import make_subject_folds
    from run_cross_validation import _fingerprint_data
    from run_g3 import run_study,load_fold
    torch.set_num_threads(1)
    rows=generate_synthetic_trials(n_subjects=10,trials_per_condition=2,n_sensors=4,timepoints=32,conditions=('slow','fast'))
    manifest=export_trials([replace(r,target=np.resize(r.target,5)) for r in rows],tmp_path/'data')
    cv=tmp_path/'cv';cv.mkdir();folds=make_subject_folds({r.subject_id for r in rows})
    for i,split in enumerate(folds,1):
        report=run_real_pipeline(manifest,cv/f'fold_{i}',subject_split=split,pilot=True,epochs=1,device='cpu')
        assert report['status']=='PILOT_ONLY'
    protocol={'manifest':str(manifest),'folds':folds,'data_sha256':_fingerprint_data(manifest),
              'parameters':{'max_refs':2,'lam':.005},'source_sha256':{},'eligible_subjects':10}
    (cv/'protocol.json').write_text(json.dumps(protocol))
    (cv/'summary.json').write_text(json.dumps({'status':'COMPLETED'}))
    result=run_study(cv,tmp_path/'result',device='cpu')
    assert result['evaluated_subjects']==10 and result['checkpoint_replay']=='PASSED'
    assert result['checkpoints_unchanged']
    for row in json.loads((tmp_path/'result/subject_results.json').read_text()):
        assert row['comparisons']['fixed_2']['reference_count']==2
        assert row['comparisons']['random_2']['reference_count']==2
    with pytest.raises(ValueError,match='new or empty'):run_study(cv,tmp_path/'result','cpu')
    report['train_target_mean'][0]+=1
    with pytest.raises(ValueError,match='normalization mismatch'):load_fold(cv/'fold_5/checkpoint.pt',report,'cpu')


@pytest.mark.parametrize('decision,expected',[('NO_GO',2),('INCONCLUSIVE',2),('GO',0)])
def test_cli_gate_exit_status(decision,expected):
    import subprocess
    import sys
    from pathlib import Path
    code=("import run_g3; "
          f"run_g3.run_study=lambda *a,**k: {{'gate':{{'decision':{decision!r},'allow_active_optimization':{decision=='GO'}}}}}; "
          "run_g3.main()")
    proc=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert proc.returncode==expected,proc.stderr
    assert decision in proc.stdout
