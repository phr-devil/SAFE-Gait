import numpy as np
import pytest
import torch
from safe_gait.models.damping import GlobalDampingCalibrator
from safe_gait.evaluation.metrics import estimate_repeatability_threshold, subject_personalization_metrics, regression_metrics
from safe_gait.models.tcn import PopulationTCN


def test_damping_is_bounded_and_missing_pair_raises():
    rng=np.random.default_rng(0)
    p=rng.normal(size=(30,3)); y=0.5*p+rng.normal(scale=.1,size=(30,3))
    c=GlobalDampingCalibrator()
    g=c.fit_pair('SP','HP',p,y)
    assert np.all((g>=0)&(g<=1))
    with pytest.raises(KeyError):
        c.get('HP','SP',3)


def test_repeatability_harm_metric():
    a=np.array([[1.,2.],[1.1,2.1],[.9,1.9]])
    b=np.array([[1.05,2.05],[1.0,2.0],[.95,1.95]])
    t=estimate_repeatability_threshold(a,b,0.95)
    out=subject_personalization_metrics(np.ones((3,2)),np.ones((3,2))*1.01,t)
    assert 'repeatability_aware_harm_rate' in out


def test_constant_target_r2_is_nan_not_huge_artifact():
    y=np.ones((4,1)); p=np.array([[1.],[2.],[1.],[0.]])
    out=regression_metrics(y,p)
    assert np.isnan(out['r2'][0])


def test_tcn_shape_and_mask():
    m=PopulationTCN(in_channels=32,n_targets=5,hidden=16,dropout=0.0)
    x=torch.randn(4,32,101)
    mask=torch.ones(4,101,dtype=torch.bool); mask[0,80:]=False
    y=m(x,mask)
    assert tuple(y.shape)==(4,5)


def test_tcn_padding_mask_prevents_convolutional_contamination():
    torch.manual_seed(4)
    m=PopulationTCN(in_channels=2,n_targets=2,hidden=8,dropout=0.0).eval()
    x=torch.randn(1,2,50)
    with torch.no_grad():
        y_short=m(x)
        padded=torch.randn(1,2,83)  # deliberately non-zero garbage in padding
        padded[:,:,:50]=x
        mask=torch.zeros(1,83,dtype=torch.bool); mask[:,:50]=True
        y_padded=m(padded,mask)
    assert torch.allclose(y_short,y_padded,atol=1e-6,rtol=1e-6)


def test_tcn_rejects_all_padding():
    m=PopulationTCN(in_channels=2,n_targets=1,hidden=4,dropout=0.0)
    x=torch.zeros(1,2,10)
    mask=torch.zeros(1,10,dtype=torch.bool)
    with pytest.raises(ValueError):
        m(x,mask)


def test_episode_mae_does_not_cancel_opposite_query_residuals():
    from safe_gait.evaluation.metrics import episode_absolute_errors
    q=np.array([[-1.0],[1.0]])
    raw,cal,pers=episode_absolute_errors(q,np.array([0.0]),np.array([0.0]))
    assert raw[0] == pytest.approx(1.0)
    assert cal[0] == pytest.approx(1.0)
    assert pers[0] == pytest.approx(1.0)
    assert abs(q.mean()) == 0.0  # the old |mean residual| evaluation would incorrectly report zero


def test_tcn_mask_safely_ignores_nan_padding_but_rejects_nan_valid_data():
    torch.manual_seed(6)
    m=PopulationTCN(in_channels=2,n_targets=1,hidden=8,dropout=0.0).eval()
    x=torch.randn(1,2,30)
    with torch.no_grad():
        y=m(x)
        xp=torch.full((1,2,50),float('nan')); xp[:,:,:30]=x
        mask=torch.zeros(1,50,dtype=torch.bool); mask[:,:30]=True
        yp=m(xp,mask)
    assert torch.allclose(y,yp,atol=1e-6,rtol=1e-6)
    bad=x.clone(); bad[:,:,3]=float('nan')
    with pytest.raises(ValueError):
        m(bad)


def test_metrics_reject_empty_arrays():
    from safe_gait.evaluation.metrics import regression_metrics, episode_absolute_errors
    with pytest.raises(ValueError):
        regression_metrics(np.empty((0,1)), np.empty((0,1)))
    with pytest.raises(ValueError):
        episode_absolute_errors(np.empty((0,1)), np.array([0.]), np.array([0.]))


def test_tcn_mask_must_be_binary_and_right_padded():
    m=PopulationTCN(in_channels=2,n_targets=1,hidden=8,dropout=0.0).eval()
    x=torch.randn(1,2,5)
    with pytest.raises(ValueError, match='binary'):
        m(x, torch.tensor([[1.0,1.0,0.5,0.0,0.0]]))
    with pytest.raises(ValueError, match='right-padded'):
        m(x, torch.tensor([[True,True,False,True,False]]))
