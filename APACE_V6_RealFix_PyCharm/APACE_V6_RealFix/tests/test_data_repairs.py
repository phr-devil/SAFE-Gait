from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch

from apace.data.weargait import WearGaitAdapter, export_trials, TARGET_ORDER
from apace.data.manifest import load_manifest
from apace.training.validation import validate_population
from test_weargait_adapter import _toy_frame
from test_real_fixes import _trial


def test_bad_pass_keeps_good_pass_and_records_reason(tmp_path):
    path = tmp_path / 'HC101_SelfPace.csv'
    _toy_frame().to_csv(path, index=False)
    adapter = WearGaitAdapter()
    rows = adapter.load_file(path)
    assert len(rows) == 1
    assert adapter.rejected_passes[0]['pass_index'] == 2
    assert 'footfalls' in adapter.rejected_passes[0]['reason']


def test_nonmonotonic_time_is_rejected_per_pass(tmp_path):
    frame = _toy_frame()
    frame.loc[4, 'Time'] = '0.01 sec'
    path = tmp_path / 'HC101_SelfPace.csv'
    frame.to_csv(path, index=False)
    adapter = WearGaitAdapter()
    assert adapter.load_file(path) == []
    assert 'time' in adapter.rejected_passes[0]['reason'].lower()


def test_export_rejects_duplicate_before_overwriting(tmp_path):
    t = replace(_trial('001', '001', 32), target=np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError, match='duplicate'):
        export_trials([t, replace(t, pressure=t.pressure * 2)], tmp_path)
    assert not list(tmp_path.rglob('*.npy'))


def test_manifest_preserves_numeric_ids(tmp_path):
    t = replace(_trial('001', '0001', 32), target=np.ones(5, dtype=np.float32))
    path = export_trials([t], tmp_path)
    row = load_manifest(path, TARGET_ORDER)[0]
    assert row.subject_id == '001' and row.trial_id == '0001'


def test_validation_weights_subjects_equally():
    class Encoder(torch.nn.Module):
        def forward(self, x, c, valid, seconds):
            return x.mean((1, 2)).unsqueeze(1) * 0
    class Head(torch.nn.Module):
        def forward(self, x):
            return x, torch.ones_like(x)
    a = replace(_trial('A', 'a', 32), target=np.array([0.], dtype=np.float32))
    b = replace(_trial('B', 'b', 32), target=np.array([2.], dtype=np.float32))
    # Per-subject Gaussian NLLs are 0 and 2, regardless of A's trial count.
    assert validate_population([a, replace(a, trial_id='a2'), b], Encoder(), Head()) == pytest.approx(1.)


def test_policy_uses_seeded_uniform_reference_and_reports_ids():
    from apace.pipeline import _evaluate_subject_policy
    from apace.oracle.counterfactual import EncodedTrial
    rows = [EncodedTrial('s', str(i), 'walk', np.zeros(2), np.zeros(1), np.ones(1)) for i in range(8)]
    class Evaluator:
        def query_error(self, context, query): return 1.
        def state_vector(self, context): return np.zeros(2)
    class Value(torch.nn.Module):
        def forward(self, *args): return torch.ones((1, 3))
    result = _evaluate_subject_policy(rows, Evaluator(), Value(), {'walk': 0}, 0., 1, 'cpu', seed=0, return_details=True)
    expected = str(np.random.default_rng(0).choice(4))
    assert result['reference_ids'] == [expected]
    assert not set(result['reference_ids']) & set(result['query_ids'])


def test_real_pipeline_saves_physical_errors_and_loadable_checkpoint(tmp_path):
    from apace.data.synthetic import generate_synthetic_trials
    from apace.real_pipeline import run_real_pipeline
    rows = generate_synthetic_trials(n_subjects=3, trials_per_condition=4, n_sensors=4, timepoints=32)
    rows = [replace(r, target=np.resize(r.target, 5)) for r in rows]
    manifest = export_trials(rows, tmp_path / 'data')
    report = run_real_pipeline(manifest, tmp_path / 'out', pilot=True, epochs=1, device='cpu')
    assert report['status'] == 'PILOT_ONLY'
    result = report['stages']['E']['results'][0]
    assert set(result['policy_mae_physical']) == set(TARGET_ORDER)
    assert report['stages']['C']['oracle_max_depth'] == 2
    checkpoint = torch.load(tmp_path / 'out' / 'checkpoint.pt', weights_only=True)
    assert checkpoint['target_columns'] == list(TARGET_ORDER)


def test_one_shot_headroom_uses_best_reference_not_first():
    from apace.real_pipeline import _one_shot_oracle_gain
    class Evaluator:
        def query_error(self, context, query):
            return {(): 10., ('bad',): 15., ('good',): 5.}[tuple(context)]
    assert _one_shot_oracle_gain(Evaluator(), ['bad', 'good'], ['q']) == pytest.approx(.5)


def test_oracle_reuses_predictions_across_cost_weights():
    from apace.oracle.counterfactual import EncodedTrial, build_protocol_value_records
    calls = []
    class Evaluator:
        def query_error(self, context, query):
            key = tuple(t.trial_id for t in context)
            calls.append(key)
            return 1. / (len(key) + 1)
        def state_vector(self, context): return np.zeros(2)
    rows = [EncodedTrial('s', str(i), 'walk', np.zeros(2), np.zeros(1), np.ones(1)) for i in range(3)]
    build_protocol_value_records(rows, Evaluator(), {'0', '1'}, {'2'}, [0., .01, .02], max_depth=2)
    assert len(calls) == len(set(calls))


def test_default_paths_resolve_from_config_not_cwd(tmp_path, monkeypatch):
    from run_real import load_data_paths
    config = tmp_path / 'paths.yaml'
    config.write_text('weargait: ../raw\ngaitrite: ../gait\noptitrack: ../opti\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path.parent)
    assert load_data_paths(config)['weargait'] == (tmp_path / '../raw').resolve()


def test_oracle_state_distinguishes_remaining_budget_and_availability():
    from apace.oracle.counterfactual import EncodedTrial, build_protocol_value_records
    class Evaluator:
        action_to_index = {'walk': 0, 'fast': 1}
        def query_error(self, context, query): return 1. / (len(context)+1)
        def state_vector(self, context): return np.zeros(2)
    rows = [EncodedTrial('s', str(i), 'walk', np.zeros(2), np.zeros(1), np.ones(1)) for i in range(3)]
    a = build_protocol_value_records(rows, Evaluator(), {'0','1'}, {'2'}, [0.], max_depth=1)
    b = build_protocol_value_records(rows, Evaluator(), {'0','1'}, {'2'}, [0.], max_depth=2)
    assert not np.array_equal(a[0].state, b[0].state)
    assert b[0].state.shape == (5,)  # belief, budget, two protocol counts
    assert b[0].state[-1] == 0  # no remaining fast trial


def test_value_validation_weights_subjects_equally():
    from apace.training.validation import validate_value
    from apace.oracle.counterfactual import ValueRecord
    class Model(torch.nn.Module):
        def forward(self, state, action, cost): return torch.zeros(len(state),3)
    a=ValueRecord('A',np.zeros(2),'walk',0.,0.,0.,0.)
    b=ValueRecord('B',np.zeros(2),'walk',0.,2.,0.,0.)
    assert validate_value([a,a,b],Model(),{'walk':0}) == pytest.approx(.5)


def test_calibration_validation_respects_one_reference_budget(monkeypatch):
    import apace.training.validation as validation
    from apace.oracle.counterfactual import EncodedTrial
    sizes=[]
    class Evaluator:
        def __init__(self,*args,**kwargs): pass
        def query_error(self,context,query):
            sizes.append(len(context))
            assert not {r.trial_id for r in context} & {r.trial_id for r in query}
            return 1.
    monkeypatch.setattr(validation,'CounterfactualEvaluator',Evaluator)
    rows=[EncodedTrial('s',str(i),c,np.zeros(2),np.zeros(1),np.ones(1))
          for i,c in enumerate(['slow','slow','fast','fast'])]
    validation.validate_calibration(rows,torch.nn.Identity(),torch.nn.Identity(),
                                    {'slow':0,'fast':1},max_context=1)
    assert sizes and max(sizes)==1


def test_g0_excludes_subject_without_independent_query_before_split(tmp_path):
    from apace.real_pipeline import run_real_pipeline
    rows=[]
    for sid, conditions in [('A',['slow','slow']),('B',['slow','slow']),('C',['slow','fast'])]:
        for i,c in enumerate(conditions):
            rows.append(replace(_trial(sid,f'{sid}_{i}',32), condition=c,target=np.ones(5,dtype=np.float32)))
    report=run_real_pipeline(export_trials(rows,tmp_path/'data'),tmp_path/'out',pilot=True,epochs=1,device='cpu')
    assert report['status']=='HALTED'
    assert report['stages']['G0']['valid_subjects']==2
    assert '3 eligible' in report['halt_reason']
