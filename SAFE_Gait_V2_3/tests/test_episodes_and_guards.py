import pandas as pd
import pytest
from safe_gait.data.episodes import build_one_shot_episodes, build_all_one_shot_episodes
from safe_gait.data.guards import assert_disjoint_subjects, assert_unique_pass_rows


def sample_df(shared_ids=False):
    rows=[]
    for s in ['A','B']:
        for c in ['SP','HP']:
            for p in range(3):
                pid=str(p) if shared_ids else f'{c}{p}'
                rows.append({'subject_id':s,'condition':c,'pass_id':pid,'r':float(p)})
    return pd.DataFrame(rows)


def test_same_condition_episode_excludes_support_pass():
    ep=build_one_shot_episodes(sample_df(),['r'],'SP','SP')
    for _,r in ep.iterrows():
        assert r.support_pass_id not in r.query_pass_ids.split('|')


def test_cross_condition_can_reuse_bare_pass_numbers():
    ep=build_one_shot_episodes(sample_df(shared_ids=True),['r'],'SP','HP')
    assert len(ep)==2
    assert all(ep.n_query==3)


def test_primary_support_selection_is_deterministic_first():
    a=build_one_shot_episodes(sample_df(),['r'],'SP','HP',seed=1)
    b=build_one_shot_episodes(sample_df(),['r'],'SP','HP',seed=999)
    assert a.support_pass_id.tolist()==b.support_pass_id.tolist()


def test_all_support_episodes_are_enumerated():
    ep=build_all_one_shot_episodes(sample_df(),['r'],'SP','HP')
    assert len(ep)==6  # 2 subjects x 3 possible supports


def test_duplicate_pass_rows_rejected():
    df=sample_df()
    df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError):
        assert_unique_pass_rows(df)
    with pytest.raises(ValueError):
        build_one_shot_episodes(df,['r'],'SP','HP')


def test_subject_split_guard():
    a=pd.DataFrame({'subject_id':['A','B']}); b=pd.DataFrame({'subject_id':['C']}); c=pd.DataFrame({'subject_id':['B']})
    with pytest.raises(ValueError): assert_disjoint_subjects(a,b,c)


def test_first_support_can_use_explicit_order_column_after_row_shuffle():
    df=sample_df().copy()
    # Within each condition, lower numeric order should define first even after shuffle.
    df['pass_order']=df['pass_id'].str.extract(r'(\d+)$').astype(int)
    df=df.sample(frac=1.0,random_state=123).reset_index(drop=True)
    ep=build_one_shot_episodes(df,['r'],'SP','HP',order_col='pass_order')
    assert all(pid=='SP0' for pid in ep.support_pass_id)


def test_absent_requested_condition_is_error_not_silent_empty_result():
    df=sample_df()
    with pytest.raises(ValueError, match='requested condition'):
        build_one_shot_episodes(df,['r'],'SP','MISSING')


def test_subject_split_guard_rejects_missing_subject_ids():
    a=pd.DataFrame({'subject_id':['A',None]}); b=pd.DataFrame({'subject_id':['B']}); c=pd.DataFrame({'subject_id':['C']})
    with pytest.raises(ValueError, match='missing values'):
        assert_disjoint_subjects(a,b,c)


def test_identifier_whitespace_is_rejected_to_prevent_split_aliases():
    df=sample_df()
    bad=df.copy(); bad.loc[0,'subject_id']=' A '
    with pytest.raises(ValueError, match='whitespace'):
        assert_unique_pass_rows(bad)
    a=pd.DataFrame({'subject_id':['A']}); b=pd.DataFrame({'subject_id':[' A ']}); c=pd.DataFrame({'subject_id':['C']})
    with pytest.raises(ValueError, match='whitespace'):
        assert_disjoint_subjects(a,b,c)
