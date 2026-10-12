import numpy as np
import pytest


def test_benefit_labels_include_reference_cost_and_refuse_nonfinite():
    from apace.evaluation.selective import benefit_labels
    np.testing.assert_array_equal(benefit_labels([.2,.2,.2],[.1,.2,.3]),[1,0,0])
    with pytest.raises(FloatingPointError):benefit_labels([np.nan],[.1])


def test_threshold_selected_on_supplied_validation_costs_and_has_always_never():
    from apace.evaluation.selective import select_threshold,decisions
    result=select_threshold([.9,.1],[1.,1.],[.5,2.])
    assert result['validation_mean_cost']==pytest.approx(.75)
    assert decisions([.9,.1],result['threshold']).tolist()==[True,False]
    assert select_threshold([.5,.5],[1.,1.],[.5,.5])['threshold']==0.
    assert select_threshold([.5,.5],[1.,1.],[2.,2.])['validation_mean_cost']==1.


def test_selective_gate_requires_all_five_fold_evidence():
    from apace.evaluation.selective import summarize_selective
    rows=[{'subject':str(i),'fold':i%5+1,'probability':.7,'calibrate':True,
           'selected':{'cost_normalized':.9,'mae_normalized':.895,'reference_count':1},
           'fixed1':{'cost_normalized':1.,'mae_normalized':.995,'reference_count':1},
           'never':{'cost_normalized':1.1,'mae_normalized':1.1,'reference_count':0}} for i in range(20)]
    assert summarize_selective(rows)['decision']=='GO'
    for r in rows:r['selected']=r['fixed1'].copy()
    assert summarize_selective(rows)['decision']=='NO_GO'
    with pytest.raises(ValueError,match='duplicate'):summarize_selective(rows+[rows[0]])


def test_predictor_training_is_finite_reproducible_and_has_no_cost_inputs_at_prediction():
    import torch
    from apace.evaluation.selective import train_predictor,predict_probability
    torch.set_num_threads(1)
    x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]],dtype=np.float32);y=np.array([0.,0.,1.,1.],dtype=np.float32)
    a,diag=train_predictor(x,y,x,y,seed=7,device='cpu',epochs=40)
    b,_=train_predictor(x,y,x,y,seed=7,device='cpu',epochs=40)
    p=predict_probability(a,x,'cpu')
    np.testing.assert_array_equal(p,predict_probability(b,x,'cpu'))
    assert np.all(np.isfinite(p)) and p[0]<p[-1] and diag['best_weights_restored']


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_selective_decision_rejects_bad_probability(bad):
    from apace.evaluation.selective import decisions
    with pytest.raises(FloatingPointError):decisions([.5,bad],.5)


def test_failed_seed_switches_to_analysis_mode():
    from run_selective import combined_gate
    summaries={str(s):{'decision':'GO'} for s in [7,17,27]}
    assert combined_gate(summaries)['allow_further_training']
    summaries['17']['decision']='NO_GO'
    gate=combined_gate(summaries)
    assert not gate['allow_further_training'] and gate['next_mode']=='benefit_failure_mechanism_analysis'
    with pytest.raises(ValueError):combined_gate({'7':{'decision':'GO'}})


def test_failure_analysis_separates_harm_cost_and_reference_variation():
    from run_selective import analyze_mechanisms
    from apace.data.weargait import TARGET_ORDER
    rows=[]
    for i in range(9):
        error=[1.1,.998,.8][i%3]
        rows.append({'subject':str(i),'net_gain':1-error-.005,'raw_error_gain':1-error,
                     'never':{'mae_normalized':1.,'mae_physical':{k:1. for k in TARGET_ORDER}},
                     'fixed1':{'mae_physical':{k:error for k in TARGET_ORDER}},
                     'candidate_net_gains':[-.1,.49] if i%3==2 else [-.1,-.1]})
    r=analyze_mechanisms(rows)
    assert r['net_benefit_subjects']==3 and r['raw_error_worsened_subjects']==3
    assert r['small_error_gain_outweighed_by_cost_subjects']==3
    assert r['expected_benefit_but_some_references_harmful']==3


@pytest.mark.parametrize('decision,expected',[('NO_GO',2),('GO',0)])
def test_selective_cli_gate_exit(decision,expected):
    import subprocess
    import sys
    from pathlib import Path
    code=("import run_selective; "
          f"run_selective.run_study=lambda *a,**k: {{'gate':{{'decision':{decision!r},'allow_further_training':{decision=='GO'}}}}}; "
          "run_selective.main()")
    p=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert p.returncode==expected,p.stderr


@pytest.mark.parametrize('changed,reason',[('protocol','split protocol'),('model','model implementation')])
def test_changed_frozen_sources_abort_before_training(tmp_path,monkeypatch,changed,reason):
    import json
    from pathlib import Path
    import run_selective as runner
    cv=tmp_path/'cv';cv.mkdir();source=tmp_path/'source';source.mkdir()
    (cv/'protocol.json').write_text(json.dumps({'folds':[{}]*5,'manifest':'unused','source_sha256':{'model.py':'model_hash'}}))
    (source/'protocol.json').write_text(json.dumps({'source_cv':str(cv),'data_sha256':'data_hash','source_protocol_sha256':'protocol_hash'}))
    monkeypatch.setattr(runner,'_fingerprint_data',lambda path:'data_hash')
    def fake_digest(path):
        if Path(path).name=='protocol.json':return 'changed' if changed=='protocol' else 'protocol_hash'
        return 'changed'
    monkeypatch.setattr(runner,'digest',fake_digest)
    with pytest.raises(ValueError,match=reason):runner.run_study(source,tmp_path/'out','cpu')
    assert not (tmp_path/'out').exists()
