from dataclasses import replace
import numpy as np
import pytest
import torch
from apace.data.types import Trial
from apace.data.batching import collate_passes
from apace.models.encoder import PressureEncoder
from apace.training.validation import BestValidationCheckpoint
from apace.real_pipeline import run_real_pipeline


def _trial(sid,tid,length):
    rng=np.random.default_rng(length)
    return Trial(sid,tid,'SelfPace',rng.random((4,length),dtype=np.float32),
          np.zeros((4,2),dtype=np.float32),np.array([1.],dtype=np.float32),sample_interval_s=.01)


def test_mixed_length_batching_and_time_preserved():
    x,c,y,m,d=collate_passes([_trial('A','a',32),_trial('B','b',51)])
    assert x.shape==(2,4,51) and m.shape==(2,51)
    assert m[0].sum()==32 and m[1].sum()==51
    assert torch.allclose(d,torch.tensor([.31,.50]))
    assert (x[0,:,32:]==0).all()


def test_masked_encoder_ignores_padding_and_tracks_duration():
    torch.manual_seed(7)
    model=PressureEncoder(4,hidden_dim=16,out_dim=8).eval()
    sample=_trial('A','a',43)
    x,c,_,mask,d=collate_passes([sample])
    bat,coords,_,mask2,d2=collate_passes([sample,_trial('B','b',65)])
    with torch.no_grad():
        one=model(x,c,mask,d)[0]
        mixed=model(bat,coords,mask2,d2)[0]
        changed=model(x,c,mask,d*1.4)[0]
    assert torch.allclose(one,mixed,atol=1e-5)
    assert not torch.allclose(one,changed)


def test_long_pass_caps_memory_but_keeps_original_duration():
    x,_,_,m,d=collate_passes([_trial('A','a',1301)],max_points=256)
    assert x.shape==(1,4,256) and int(m.sum())==256
    assert d.item()==pytest.approx(13.0,abs=1e-4)


def test_validation_checkpoint_restores_best_weights():
    net=torch.nn.Linear(1,1,bias=False)
    check=BestValidationCheckpoint([net],patience=1,min_epochs=2)
    with torch.no_grad():net.weight.fill_(2.)
    assert not check.check(1.,1)
    with torch.no_grad():net.weight.fill_(99.)
    assert check.check(2.,2)
    check.restore()
    assert net.weight.item()==pytest.approx(2.)


def test_guard_halts_before_training_if_too_few_subjects(tmp_path):
    import pandas as pd
    t=_trial('A','a',40)
    np.save(tmp_path/'p.npy',t.pressure);np.save(tmp_path/'c.npy',t.coords)
    pd.DataFrame([dict(subject_id='A',trial_id='a',condition='SelfPace',
        pressure_path='p.npy',coords_path='c.npy',velocity_cm_s=1.,step_length_cm=1.,
        stride_length_cm=1.,step_time_s=1.,stride_time_s=1.)]).to_csv(tmp_path/'manifest.csv',index=False)
    result=run_real_pipeline(tmp_path/'manifest.csv',tmp_path/'out',pilot=True,device='cpu')
    assert result['status']=='HALTED' and '3 eligible' in result['halt_reason']
    assert (tmp_path/'out'/'real_metrics.json').exists()
