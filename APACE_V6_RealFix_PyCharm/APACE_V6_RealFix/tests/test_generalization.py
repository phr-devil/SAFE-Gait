import numpy as np
import pandas as pd
import pytest


def test_five_folds_have_no_overlap_and_test_every_subject_once():
    from apace.evaluation.cross_validation import make_subject_folds
    ids=[f'S{i:03}' for i in range(23)]
    folds=make_subject_folds(ids,seed=7)
    assert folds==make_subject_folds(ids[::-1],seed=7)
    assert len(folds)==5
    tested=[]
    for f in folds:
        tr,va,te=map(set,[f['train'],f['validation'],f['test']])
        assert tr and len(va)>=3 and te
        assert not(tr&va or tr&te or va&te)
        assert tr|va|te==set(ids)
        tested+=f['test']
    assert sorted(tested)==sorted(ids)


def test_explicit_split_rejects_leakage_and_missing_ids():
    from apace.real_pipeline import _explicit_subject_split
    from apace.data.synthetic import generate_synthetic_trials
    rows=generate_synthetic_trials(n_subjects=6)
    split={'train':['S000','S001'],'validation':['S002','S003'],'test':['S004','S005']}
    train,val,test,actual=_explicit_subject_split(rows,split,pilot=True)
    assert {r.subject_id for r in test}=={'S004','S005'}
    with pytest.raises(ValueError,match='overlap'):
        _explicit_subject_split(rows,{**split,'test':['S001','S005']},pilot=True)
    with pytest.raises(ValueError,match='cover'):
        _explicit_subject_split(rows,{**split,'test':['S004']},pilot=True)


def test_pkmass_two_header_columns_and_exact_matching(tmp_path):
    from apace.evaluation.label_audit import load_pkmas, audit_labels
    from apace.data.weargait import TARGET_ORDER
    cols=[('Task',''),('Participant ID',''),('PD vs Control',''),
          ('Velocity (cm./sec.)',''),('Step Length (cm.)','Mean'),('Step Length (cm.)','#Samples'),
          ('Stride Length (cm.)','Mean'),('Step Time (sec.)','Mean'),('Stride Time (sec.)','Mean')]
    official=pd.DataFrame([['SelfPace ','001','Control',100,50,999,100,.5,1]],columns=pd.MultiIndex.from_tuples(cols))
    p=tmp_path/'pkmas.csv';official.to_csv(p,index=False)
    parsed=load_pkmas(p)
    assert parsed.loc[0,'subject_id']=='001'
    assert parsed.loc[0,'step_length_cm']==50
    frame=pd.DataFrame([dict(subject_id=s,condition='SelfPace',**dict(zip(TARGET_ORDER,values)))
                        for s,values in [('001',[90,40,90,.4,.8]),('001',[110,60,110,.6,1.2]),('001_2',[1]*5)]])
    matched,summary=audit_labels(frame,parsed)
    assert len(matched)==1
    assert summary['matched_subjects']==1 and summary['unmatched_reconstructed_pairs']==1
    assert summary['metrics']['step_length_cm']['mae']==pytest.approx(0)


def test_label_audit_refuses_duplicate_official_pairs():
    from apace.evaluation.label_audit import audit_labels
    from apace.data.weargait import TARGET_ORDER
    row=dict(subject_id='s',condition='SelfPace',**{k:1. for k in TARGET_ORDER})
    with pytest.raises(ValueError,match='duplicate'):
        audit_labels(pd.DataFrame([row]),pd.DataFrame([row,row]))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_label_audit_rejects_nonfinite_pass_before_aggregation(bad):
    from apace.evaluation.label_audit import audit_labels
    from apace.data.weargait import TARGET_ORDER
    row=dict(subject_id='s',condition='SelfPace',**{k:1. for k in TARGET_ORDER})
    invalid={**row,TARGET_ORDER[0]:bad}
    with pytest.raises(FloatingPointError,match='nonfinite_reconstructed'):
        audit_labels(pd.DataFrame([row,invalid]),pd.DataFrame([row]))


def test_reference_baselines_do_not_choose_using_query_targets():
    from dataclasses import replace
    from apace.oracle.counterfactual import EncodedTrial
    from apace.evaluation.comparison import evaluate_reference_schedule
    rows=[EncodedTrial('s',str(i),'normal',np.zeros(2),np.zeros(1),np.ones(1)) for i in range(6)]
    class Evaluator:
        def query_errors(self,context,query):
            assert not {r.trial_id for r in context}&{r.trial_id for r in query}
            return np.ones((len(query),1))
    a=evaluate_reference_schedule(rows,Evaluator(),np.array([10.]),['speed'],budget=2,seed=7)
    changed=[replace(r,target=np.array([999.])) if int(r.trial_id)>=3 else r for r in rows]
    b=evaluate_reference_schedule(changed,Evaluator(),np.array([10.]),['speed'],budget=2,seed=7)
    assert a['reference_ids']==b['reference_ids']
    assert a['reference_count']==2 and a['mae_physical']['speed']==10.


def test_subject_summary_has_coverage_ci_and_worsening_rate():
    from apace.evaluation.cross_validation import summarize_subject_results
    rows=[{'subject':'A','zero_shot_mae_normalized':1.,'policy_mae_normalized':.5,'reference_count':1,
           'zero_shot_mae_physical':{'speed':10.},'policy_mae_physical':{'speed':5.}},
          {'subject':'B','zero_shot_mae_normalized':1.,'policy_mae_normalized':1.5,'reference_count':2,
           'zero_shot_mae_physical':{'speed':10.},'policy_mae_physical':{'speed':15.}}]
    s=summarize_subject_results(rows,planned_subjects=3,seed=7)
    assert s['evaluated_subjects']==2 and not s['complete_coverage']
    assert s['worsened_fraction']==.5 and s['policy_physical_mae']['speed']['mean']==10.
    with pytest.raises(ValueError,match='duplicate'):
        summarize_subject_results(rows+[rows[0]],3)


def test_multi_subject_pipeline_uses_train_only_normalization(tmp_path):
    import torch
    from dataclasses import replace
    from apace.data.synthetic import generate_synthetic_trials
    from apace.data.weargait import export_trials
    from apace.real_pipeline import run_real_pipeline
    torch.set_num_threads(1)
    rows=generate_synthetic_trials(n_subjects=5,trials_per_condition=2,n_sensors=4,timepoints=32,conditions=('slow',))
    split={'train':['S000','S001'],'validation':['S002'],'test':['S003','S004']}
    rows=[replace(r,target=np.resize(r.target,5)+(1000 if r.subject_id in split['test'] else 0)) for r in rows]
    expected=np.stack([r.target for r in rows if r.subject_id in split['train']]).mean(0)
    result=run_real_pipeline(export_trials(rows,tmp_path/'data'),tmp_path/'out',pilot=True,epochs=1,device='cpu',
                             subject_split=split,compare_baselines=True)
    assert result['status']=='PILOT_ONLY'
    assert np.allclose(result['train_target_mean'],expected)
    assert {r['subject'] for r in result['stages']['E']['results']}==set(split['test'])
    assert all('comparisons' in r for r in result['stages']['E']['results'])
    assert result['baseline_selection']['source']=='validation_only'


def test_cv_runner_reports_halted_fold_without_hiding_missing_people(tmp_path,monkeypatch):
    import run_cross_validation as runner
    from dataclasses import replace
    from apace.data.synthetic import generate_synthetic_trials
    from apace.data.weargait import export_trials
    rows=generate_synthetic_trials(n_subjects=10,trials_per_condition=2,n_sensors=4,timepoints=32,conditions=('slow',))
    manifest=export_trials([replace(r,target=np.resize(r.target,5)) for r in rows],tmp_path/'data')
    calls=[]
    def fake(manifest,out,**kwargs):
        calls.append(kwargs['subject_split'])
        assert (tmp_path/'out/protocol.json').exists()
        if len(calls)==1:return {'status':'HALTED','halt_stage':'G1','halt_reason':'injected gate'}
        return {'status':'EVALUATED_NOT_PROVEN','stages':{'E':{'results':[
            {'subject':s,'zero_shot_mae_normalized':1.,'policy_mae_normalized':.8,'reference_count':1,
             'zero_shot_mae_physical':{'speed':1.},'policy_mae_physical':{'speed':.8}}
            for s in kwargs['subject_split']['test']]}}}
    monkeypatch.setattr(runner,'run_real_pipeline',fake)
    result=runner.run_study(manifest,tmp_path/'out',epochs=1,device='cpu')
    assert len(calls)==5 and result['evaluated_subjects']==8
    assert result['status']=='INCOMPLETE' and not result['complete_coverage']
    assert result['folds'][0]['status']=='HALTED'
    with pytest.raises(ValueError,match='exists'):
        runner.run_study(manifest,tmp_path/'out',epochs=1,device='cpu')
