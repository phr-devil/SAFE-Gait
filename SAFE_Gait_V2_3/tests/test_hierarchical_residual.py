import numpy as np
import pandas as pd
import pytest
from safe_gait.models.hierarchical_residual import HierarchicalResidualModel
from scripts.run_synthetic_demo import generate, TARGETS


def test_fit_components_are_psd_tvr_valid_and_decomposition_coherent():
    df=generate(seed=1,n_subjects=120,passes=4)
    m=HierarchicalResidualModel(TARGETS,covariance_shrinkage=0.15).fit(df)
    for a in [m.sigma_u_,m.sigma_eps_["SP"],m.sigma_eps_["HP"],m.sigma_v_["SP"],m.sigma_v_["HP"]]:
        assert np.linalg.eigvalsh(a).min()>=-1e-9
    for c in ["SP","HP"]:
        tvr=m.transferable_variance_ratio(c)
        assert np.all((tvr>=0)&(tvr<=1))
        assert np.allclose(m.sigma_u_+m.sigma_v_[c],m.sigma_latent_[c],atol=1e-7,rtol=1e-6)


def test_cross_condition_personalization_has_positive_synthetic_headroom():
    df=generate(seed=3,n_subjects=260,passes=4)
    ids=sorted(df.subject_id.unique())
    tr=df[df.subject_id.isin(ids[:180])]
    te=df[df.subject_id.isin(ids[180:])]
    m=HierarchicalResidualModel(TARGETS,covariance_shrinkage=0.15).fit(tr)
    gains=[]
    for sid,sdf in te.groupby('subject_id'):
        sp=sdf[sdf.condition=='SP'].iloc[0]
        hp=sdf[sdf.condition=='HP']
        rs=sp[TARGETS].to_numpy(float)
        qvals=hp[TARGETS].to_numpy(float)
        mu=m.population_bias('HP')
        corr=m.posterior(rs,'SP','HP',query_n=1).correction_raw
        base=np.mean(np.abs(qvals-mu),axis=0)
        pers=np.mean(np.abs(qvals-mu-corr),axis=0)
        gains.append(base-pers)
    assert np.mean(np.vstack(gains))>0


def test_population_bias_is_subject_balanced():
    rows=[]
    for i in range(10): rows.append({'subject_id':'S1','condition':'SP','pass_id':f'a{i}','x':0.0})
    rows.append({'subject_id':'S2','condition':'SP','pass_id':'b','x':10.0})
    rows.append({'subject_id':'S3','condition':'SP','pass_id':'c','x':5.0})
    for s,v in [('S1',0.0),('S2',10.0),('S3',5.0),('S4',7.0),('S5',3.0)]:
        rows.append({'subject_id':s,'condition':'HP','pass_id':'h1','x':v})
        rows.append({'subject_id':s,'condition':'HP','pass_id':'h2','x':v})
        rows.append({'subject_id':s,'condition':'SP','pass_id':'s_extra','x':v})
    df=pd.DataFrame(rows)
    m=HierarchicalResidualModel(['x'],covariance_shrinkage=1.0).fit(df)
    sm=df[df.condition=='SP'].groupby('subject_id').x.mean().mean()
    assert np.isclose(m.population_bias('SP')[0],sm)


def test_same_condition_and_cross_condition_are_not_identical_when_v_exists():
    df=generate(seed=5,n_subjects=150,passes=4)
    m=HierarchicalResidualModel(TARGETS).fit(df)
    r=df[(df.subject_id=='S000')&(df.condition=='SP')].iloc[0][TARGETS].to_numpy(float)
    a=m.posterior(r,'SP','SP').correction_z
    b=m.posterior(r,'SP','HP').correction_z
    assert not np.allclose(a,b)


def test_negative_cross_condition_target_is_exactly_disabled():
    rng=np.random.default_rng(9)
    rows=[]
    for s in range(80):
        u=rng.normal()
        for p in range(3):
            rows.append({'subject_id':f'S{s}','condition':'SP','pass_id':f's{p}','x':u+rng.normal(scale=.2)})
            rows.append({'subject_id':f'S{s}','condition':'HP','pass_id':f'h{p}','x':-u+rng.normal(scale=.2)})
    df=pd.DataFrame(rows)
    m=HierarchicalResidualModel(['x'],min_transfer_correlation=0.0).fit(df)
    assert m.paired_correlation_[0] < 0
    assert not bool(m.transfer_enabled_[0])
    assert m.transferable_variance_ratio('SP')[0] == 0.0
    r=df[(df.subject_id=='S0')&(df.condition=='SP')].iloc[0][['x']].to_numpy(float)
    assert m.posterior(r,'SP','HP').correction_z[0] == pytest.approx(0.0,abs=1e-12)
    assert m.sigma_u_[0,0] == pytest.approx(0.0,abs=1e-12)


def test_query_mean_predictive_variance_scales_with_query_n():
    df=generate(seed=8,n_subjects=140,passes=4)
    m=HierarchicalResidualModel(TARGETS).fit(df)
    r=df[(df.subject_id=='S000')&(df.condition=='SP')].iloc[0][TARGETS].to_numpy(float)
    one=m.posterior(r,'SP','HP',query_n=1)
    four=m.posterior(r,'SP','HP',query_n=4)
    assert np.all(np.diag(four.predictive_cov_z) <= np.diag(one.predictive_cov_z)+1e-12)


def test_damping_updates_expected_corrected_error_second_moment():
    df=generate(seed=10,n_subjects=140,passes=4)
    m=HierarchicalResidualModel(TARGETS).fit(df)
    r=df[(df.subject_id=='S000')&(df.condition=='SP')].iloc[0][TARGETS].to_numpy(float)
    undamped=m.posterior(r,'SP','HP')
    damped=m.posterior(r,'SP','HP',damping=np.zeros(len(TARGETS)))
    assert np.allclose(damped.correction_z,0)
    assert np.all(np.diag(damped.corrected_error_second_moment_z) >= np.diag(undamped.predictive_cov_z)-1e-12)


def test_nan_inf_and_duplicate_passes_rejected():
    df=generate(seed=11,n_subjects=30,passes=3)
    bad=df.copy(); bad.loc[0,TARGETS[0]]=np.inf
    with pytest.raises(ValueError): HierarchicalResidualModel(TARGETS).fit(bad)
    dup=pd.concat([df,df.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError): HierarchicalResidualModel(TARGETS).fit(dup)


def test_unexpected_condition_is_rejected():
    df=generate(seed=12,n_subjects=30,passes=3)
    extra=df.iloc[[0]].copy(); extra['condition']='BALANCE'; extra['pass_id']='bal0'
    bad=pd.concat([df,extra],ignore_index=True)
    with pytest.raises(ValueError):
        HierarchicalResidualModel(TARGETS).fit(bad)


def test_negative_transfer_threshold_rejected_because_core_shared_loading_is_same_sign():
    with pytest.raises(ValueError):
        HierarchicalResidualModel(['x'],min_transfer_correlation=-0.1)


def test_within_subject_covariance_requires_repeated_passes():
    rows=[]
    for s in range(20):
        rows.append({'subject_id':f'S{s}','condition':'SP','pass_id':'sp0','x':float(s)})
        rows.append({'subject_id':f'S{s}','condition':'HP','pass_id':'hp0','x':float(s)+0.1})
    df=pd.DataFrame(rows)
    with pytest.raises(ValueError, match='insufficient repeated-pass information'):
        HierarchicalResidualModel(['x']).fit(df)


def test_pass_identity_guard_cannot_be_disabled():
    df=generate(seed=15,n_subjects=30,passes=3)
    with pytest.raises(ValueError, match='pass_col is required'):
        HierarchicalResidualModel(TARGETS).fit(df, pass_col=None)  # type: ignore[arg-type]


def test_support_and_query_counts_must_be_positive_integers():
    df=generate(seed=16,n_subjects=40,passes=3)
    m=HierarchicalResidualModel(TARGETS).fit(df)
    r=df[(df.subject_id=='S000')&(df.condition=='SP')].iloc[0][TARGETS].to_numpy(float)
    for bad in (1.5, True, 0, -1):
        with pytest.raises(ValueError):
            m.posterior(r,'SP','HP',support_n=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            m.posterior(r,'SP','HP',query_n=bad)  # type: ignore[arg-type]


def test_targetwise_shared_cap_avoids_one_target_globally_suppressing_others():
    from safe_gait.utils.linalg import targetwise_cap_shared, max_psd_scale
    shared=np.diag([10.0,1.0])
    marg=[np.eye(2),np.eye(2)]
    old_alpha=max_psd_scale(shared,marg)
    capped,factors=targetwise_cap_shared(shared,marg)
    new_alpha=max_psd_scale(capped,marg)
    old=old_alpha*shared
    new=new_alpha*capped
    assert old[1,1] < 0.2  # old one-global-scale logic suppresses target 2
    assert new[0,0] == pytest.approx(1.0,abs=1e-8)
    assert new[1,1] == pytest.approx(1.0,abs=1e-8)
    assert np.all(factors <= 1.0)


def test_paired_correlation_ci_is_finite_for_nonconstant_targets():
    df=generate(seed=18,n_subjects=100,passes=4)
    m=HierarchicalResidualModel(TARGETS).fit(df)
    lo,hi=m.paired_correlation_ci()
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo <= m.paired_correlation_)
    assert np.all(m.paired_correlation_ <= hi)
