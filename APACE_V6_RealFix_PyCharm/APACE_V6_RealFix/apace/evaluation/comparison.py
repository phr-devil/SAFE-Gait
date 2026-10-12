"""Reference baselines: choose protocols without unacquired reference labels."""
from collections import defaultdict
from itertools import product
import hashlib
import numpy as np


def evaluate_reference_schedule(rows,evaluator,std,target_columns,*,budget,seed=7,schedule=None,lam=0.):
    from apace.pipeline import _cal_query_ids
    lookup={r.trial_id:r for r in rows}
    cal_ids,query_ids=_cal_query_ids(rows)
    query=[lookup[i] for i in sorted(query_ids)]
    subject=rows[0].subject_id
    derived=int.from_bytes(hashlib.sha256(f'{seed}:{subject}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(derived)
    remaining=set(cal_ids);context=[]
    for step in range(budget if schedule is None else min(budget,len(schedule))):
        available=sorted({lookup[i].condition for i in remaining})
        if not available:break
        action=str(rng.choice(available)) if schedule is None else schedule[step]
        if action not in available:break
        candidates=sorted(i for i in remaining if lookup[i].condition==action)
        chosen=candidates[int(rng.integers(len(candidates)))]
        context.append(lookup[chosen]);remaining.remove(chosen)
    errors=evaluator.query_errors(context,query)
    if not np.isfinite(errors).all():raise FloatingPointError('nonfinite_comparison_errors')
    error=float(errors.mean())
    return {'mae_normalized':error,'reference_count':len(context),'cost_normalized':error+lam*len(context),
            'reference_ids':[r.trial_id for r in context],'query_ids':sorted(query_ids),
            'mae_physical':dict(zip(target_columns,map(float,(errors*std).mean(0))))}


def select_fixed_schedule(validation_rows,evaluator,conditions,std,target_columns,*,budget,seed,lam):
    by_subject=defaultdict(list)
    for r in validation_rows:by_subject[r.subject_id].append(r)
    candidates=[()] + [s for n in range(1,budget+1) for s in product(sorted(conditions),repeat=n)]
    scored=[]
    for schedule in candidates:
        costs=[evaluate_reference_schedule(rows,evaluator,std,target_columns,budget=budget,seed=seed,
                                           schedule=schedule,lam=lam)['cost_normalized'] for rows in by_subject.values()]
        scored.append({'schedule':list(schedule),'validation_mean_cost':float(np.mean(costs))})
    best=min(scored,key=lambda r:(r['validation_mean_cost'],len(r['schedule']),r['schedule']))
    return {'source':'validation_only','schedule':best['schedule'],'candidates':scored,
            'objective':'mean subject normalized MAE + lambda * reference_count',
            'unavailable_protocol_rule':'stop','seed':seed}
