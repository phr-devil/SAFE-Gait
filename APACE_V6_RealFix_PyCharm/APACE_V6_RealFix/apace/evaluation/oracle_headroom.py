"""Offline clairvoyant protocol bound; query labels NEVER become training data."""
from functools import lru_cache
from itertools import product
import numpy as np
from apace.evaluation.metrics import bootstrap_mean_ci

BASELINES=('fixed_1','fixed_2','random_1','random_2')


class ProtocolStudy:
    """Exact expectation over uniform, without-replacement outcomes of each protocol.

    Oracle knows all counterfactual query errors, chooses protocols, and may STOP.
    It cannot choose which unknown trial a protocol will yield. Terminal vectors
    contain normalized error, total reference count, and physical target errors.
    """
    def __init__(self,rows,evaluator,calibration_ids,query_ids,std,targets,*,budget=2,lam=.005):
        self.lookup={r.trial_id:r for r in rows}
        if len(self.lookup)!=len(rows) or len({r.subject_id for r in rows})!=1:
            raise ValueError('need one subject and unique trial IDs')
        if not calibration_ids or not query_ids or calibration_ids & query_ids:
            raise ValueError('need nonempty disjoint calibration/query pools')
        if not (calibration_ids|query_ids)<=self.lookup.keys():raise ValueError('unknown trial ID')
        if budget not in (0,1,2) or not np.isfinite(lam) or lam<0:raise ValueError('invalid budget or cost')
        self.std=np.asarray(std,dtype=float);self.targets=list(targets)
        if self.std.shape!=(len(targets),) or not np.isfinite(self.std).all() or np.any(self.std<=0):
            raise ValueError('invalid target scale')
        self.cal=tuple(sorted(calibration_ids));self.query_ids=sorted(query_ids)
        self.query=[self.lookup[i] for i in self.query_ids]
        self.evaluator=evaluator;self.budget=budget;self.lam=lam
        self.actions=sorted({self.lookup[i].condition for i in self.cal})
        self._errors={}

    def available(self,state):
        return {a:[i for i in self.cal if i not in state and self.lookup[i].condition==a]
                for a in self.actions if any(i not in state and self.lookup[i].condition==a for i in self.cal)}

    def terminal(self,state):
        if state not in self._errors:
            errors=np.asarray(self.evaluator.query_errors([self.lookup[i] for i in state],self.query))
            if errors.shape!=(len(self.query),len(self.targets)):raise ValueError('invalid error shape')
            physical=(errors*self.std).mean(0)
            vector=np.r_[errors.mean(),len(state),physical]
            if not np.isfinite(errors).all() or not np.isfinite(vector).all():
                raise FloatingPointError('nonfinite_oracle_errors')
            if np.any(errors<0):raise ValueError('negative absolute errors')
            self._errors[state]=vector
        return self._errors[state]

    def cost(self,vector):return float(vector[0]+self.lam*vector[1])

    def report(self,vector):
        if not np.isfinite(vector).all() or not np.isfinite(self.cost(vector)):
            raise FloatingPointError('nonfinite_oracle_result')
        return {'mae_normalized':float(vector[0]),'reference_count':float(vector[1]),
                'cost_normalized':self.cost(vector),'mae_physical':dict(zip(self.targets,map(float,vector[2:])))}

    def oracle(self):
        decisions={}
        @lru_cache(None)
        def solve(state):
            best=self.terminal(state);action='STOP'
            if len(state)<self.budget:
                for a,ids in self.available(state).items():
                    value=np.mean([solve(tuple(sorted((*state,i)))) for i in ids],axis=0)
                    if self.cost(value)<self.cost(best):best,action=value,a
            decisions[state]=action
            return best
        result=self.report(solve(()))
        result.update(first_action=decisions[()],decisions=[{'context':list(k),'action':a} for k,a in sorted(decisions.items())])
        return result

    def fixed(self,schedule):
        schedule=tuple(schedule)
        if len(schedule)>self.budget:raise ValueError('schedule exceeds budget')
        @lru_cache(None)
        def solve(state,step):
            if step==len(schedule):return self.terminal(state)
            available=self.available(state)
            if not available:return self.terminal(state)
            # Fixed, label-blind fallback preserves the requested count when feasible.
            ids=available.get(schedule[step],available[sorted(available)[0]])
            return np.mean([solve(tuple(sorted((*state,i))),step+1) for i in ids],axis=0)
        return self.report(solve((),0))

    def random(self,count):
        if count<0 or count>self.budget:raise ValueError('count exceeds budget')
        @lru_cache(None)
        def solve(state):
            choices=self.available(state)
            if len(state)==count or not choices:return self.terminal(state)
            return np.mean([np.mean([solve(tuple(sorted((*state,i)))) for i in ids],axis=0)
                            for ids in choices.values()],axis=0)
        return self.report(solve(()))


def select_fixed(studies,actions,count):
    """Caller supplies validation subjects only; exact length, no optional STOP."""
    if not studies:raise ValueError('empty validation cohort')
    scores=[{'schedule':list(seq),'validation_mean_cost':float(np.mean([s.fixed(seq)['cost_normalized'] for s in studies]))}
            for seq in product(sorted(actions),repeat=count)]
    if not scores:raise ValueError('no candidate schedules')
    best=min(scores,key=lambda r:(r['validation_mean_cost'],r['schedule']))
    return {**best,'candidates':scores,'source':'validation_only','requested_count':count}


def decide_g3(rows,*,min_effect=.05,seed=7):
    if len({r['subject'] for r in rows})!=len(rows):raise ValueError('duplicate subject')
    if not rows or not np.isfinite(min_effect) or min_effect<0:raise ValueError('invalid G3 inputs')
    comparisons={}
    oracle=np.array([r['oracle']['cost_normalized'] for r in rows])
    for name in BASELINES:
        base=np.array([r['comparisons'][name]['cost_normalized'] for r in rows])
        if np.any(base<0) or np.any(oracle<0):raise ValueError('negative costs')
        effects=(base-oracle)/np.maximum(base,1e-8)
        mean,lo,hi=bootstrap_mean_ci(effects,seed=seed)
        decision='GO' if mean>=min_effect and lo>0 else ('NO_GO' if hi<min_effect else 'INCONCLUSIVE')
        comparisons[name]={'mean_relative_cost_gain':mean,'ci95':[lo,hi],'decision':decision}
    decision='GO' if all(v['decision']=='GO' for v in comparisons.values()) else (
        'NO_GO' if any(v['decision']=='NO_GO' for v in comparisons.values()) else 'INCONCLUSIVE')
    return {'decision':decision,'allow_active_optimization':decision=='GO','min_effect':min_effect,
            'rule':'all four comparisons: mean relative cost gain >= threshold and paired bootstrap lower CI > 0',
            'comparisons':comparisons,'subjects':len(rows),
            'scope':'offline optimistic bound; descriptive subject bootstrap with overlapping training folds'}
