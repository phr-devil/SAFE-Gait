"""Frozen subject folds and subject-weighted out-of-fold summaries."""
import math
import numpy as np
from apace.evaluation.metrics import bootstrap_mean_ci


def make_subject_folds(subject_ids,n_splits=5,seed=7):
    ids=sorted(set(subject_ids))
    if n_splits<2 or len(ids)<max(10,n_splits+3):
        raise ValueError('need >=10 subjects and enough subjects for disjoint validation/training')
    rng=np.random.default_rng(seed);rng.shuffle(ids)
    folds=[]
    for test in np.array_split(ids,n_splits):
        remaining=sorted(set(ids)-set(test))
        rng.shuffle(remaining)
        n_val=min(max(3,math.ceil(len(remaining)*.15)),len(remaining)-1)
        folds.append({'train':sorted(remaining[n_val:]),'validation':sorted(remaining[:n_val]),'test':sorted(test.tolist())})
    return folds


def summarize_subject_results(rows,planned_subjects,seed=7):
    if len({r['subject'] for r in rows})!=len(rows):raise ValueError('duplicate out-of-fold test subject')
    result={'evaluated_subjects':len(rows),'planned_subjects':planned_subjects,
            'complete_coverage':len(rows)==planned_subjects,
            'interval_scope':'subject bootstrap; descriptive, overlapping training folds are not independent replications'}
    if not rows:return result
    def interval(values):
        mean,lo,hi=bootstrap_mean_ci(np.asarray(values,dtype=float),n_boot=2000,seed=seed)
        return {'mean':mean,'ci95':[lo,hi]}
    base=np.asarray([r['zero_shot_mae_normalized'] for r in rows])
    policy=np.asarray([r['policy_mae_normalized'] for r in rows])
    result.update(zero_shot_normalized=interval(base),policy_normalized=interval(policy),
                  mean_subject_relative_improvement=interval((base-policy)/np.maximum(base,1e-8)),
                  worsened_fraction=float(np.mean(policy>base+1e-10)),
                  unchanged_fraction=float(np.mean(np.isclose(policy,base,rtol=0,atol=1e-10))),
                  reference_count=interval([r['reference_count'] for r in rows]))
    for label in ('zero_shot','policy'):
        result[label+'_physical_mae']={k:interval([r[label+'_mae_physical'][k] for r in rows])
                                      for k in rows[0][label+'_mae_physical']}
    names=sorted(set.intersection(*(set(r.get('comparisons',{})) for r in rows)))
    result['comparisons']={}
    for name in names:
        vals=[r['comparisons'][name] for r in rows]
        result['comparisons'][name]={'mae_normalized':interval([v['mae_normalized'] for v in vals]),
                                    'reference_count':interval([v['reference_count'] for v in vals]),
                                    'cost_normalized':interval([v['cost_normalized'] for v in vals]),
                                    'mae_physical':{k:interval([v['mae_physical'][k] for v in vals]) for k in vals[0]['mae_physical']}}
        if all('policy_cost_normalized' in r for r in rows):
            result['comparisons'][name]['policy_minus_baseline_cost']=interval(
                [r['policy_cost_normalized']-r['comparisons'][name]['cost_normalized'] for r in rows])
    return result
