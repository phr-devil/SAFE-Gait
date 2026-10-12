"""G4 exact learned-policy evaluation and fail-closed Oracle recovery criteria."""
from functools import lru_cache
import numpy as np
import torch
from apace.oracle.counterfactual import planning_state
from apace.evaluation.policy import choose_action
from apace.evaluation.metrics import bootstrap_mean_ci

BASELINES=('fixed1','random1','uncertainty1')


class ValueController:
    def __init__(self,evaluator,model,actions,budget,lam,device):
        self.evaluator=evaluator;self.model=model;self.actions=actions
        self.budget=budget;self.lam=lam;self.device=device

    @torch.no_grad()
    def __call__(self,context,remaining):
        features=planning_state(self.evaluator.state_vector(context),remaining,self.actions,self.budget-len(context))
        if not np.isfinite(features).all():raise FloatingPointError('nonfinite_policy_state')
        state=torch.tensor(features,dtype=torch.float32,device=self.device)[None,:]
        available={r.condition for r in remaining};values={}
        for action,index in self.actions.items():
            if action not in available:continue
            quantiles=self.model(state,torch.tensor([index],device=self.device),torch.tensor([[self.lam]],device=self.device))
            if not torch.isfinite(quantiles).all():raise FloatingPointError('nonfinite_policy_values')
            values[action]=float(quantiles[0,1].cpu())
        return choose_action(values)[0],values


def evaluate_policy(study,controller):
    """Controller sees acquired context and remaining references, never query targets."""
    decisions={}
    @lru_cache(None)
    def solve(state):
        available=study.available(state)
        if len(state)>=study.budget or not available:return study.terminal(state)
        context=[study.lookup[i] for i in state]
        remaining=[study.lookup[i] for i in study.cal if i not in state]
        action,values=controller(context,remaining)
        if any(not np.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite_policy_values')
        decisions[state]={'context':list(state),'action':action,'values':values}
        if action=='STOP':return study.terminal(state)
        if action not in available:raise ValueError('policy chose unavailable action')
        return np.mean([solve(tuple(sorted((*state,i)))) for i in available[action]],axis=0)
    result=study.report(solve(()))
    result['decisions']=[v for _,v in sorted(decisions.items())]
    result['first_action']=decisions.get((),{'action':'STOP'})['action']
    return result


@torch.no_grad()
def uncertainty_protocol(candidates,base,device):
    if not candidates:raise ValueError('no uncertainty candidates')
    embeddings=torch.tensor(np.stack([r.embedding for r in candidates]),dtype=torch.float32,device=device)
    _,scales=base(embeddings)
    if not torch.isfinite(scales).all() or (scales<=0).any():raise FloatingPointError('invalid_predictive_uncertainty')
    per_trial=scales.mean(1).cpu().numpy()
    means={a:float(np.mean([u for r,u in zip(candidates,per_trial) if r.condition==a]))
           for a in sorted({r.condition for r in candidates})}
    return max(means,key=means.get)


def gain_recovery(zero,oracle,learned,seed=7):
    arrays=[np.asarray(x,dtype=float) for x in [zero,oracle,learned]]
    if any(x.ndim!=1 or len(x)==0 or x.shape!=arrays[0].shape for x in arrays):raise ValueError('invalid recovery arrays')
    if not all(np.isfinite(x).all() for x in arrays):raise FloatingPointError('nonfinite_recovery')
    zero,oracle,learned=arrays;denom=zero-oracle;numerator=zero-learned
    d=float(denom.mean());n=float(numerator.mean())
    if d<=1e-8:return {'value':None,'ci95':None,'reason':'no_positive_oracle_headroom','denominator':d}
    index=np.random.default_rng(seed).integers(0,len(zero),size=(2000,len(zero)))
    ds=denom[index].mean(1);valid=ds>1e-8
    ratios=numerator[index].mean(1)[valid]/ds[valid]
    if not np.isfinite(ratios).all():raise FloatingPointError('nonfinite_recovery_bootstrap')
    return {'value':n/d,'ci95':list(map(float,np.quantile(ratios,[.025,.975]))) if len(ratios) else None,
            'denominator':d,'numerator':n,'invalid_bootstrap_samples':int((~valid).sum()),
            'definition':'(mean zero cost - mean learned cost)/(mean zero cost - mean Oracle cost); no clipping'}


def summarize_seed(rows):
    if not rows or len({r['subject'] for r in rows})!=len(rows):raise ValueError('empty or duplicate subject')
    zero=np.array([r['zero_cost'] for r in rows]);oracle=np.array([r['oracle_cost'] for r in rows])
    learned=np.array([r['learned']['cost_normalized'] for r in rows])
    recovery=gain_recovery(zero,oracle,learned)
    folds=sorted({r['fold'] for r in rows});comparisons={};reasons=[]
    if recovery['value'] is None or recovery['value']<.8:reasons.append('Oracle_gain_recovery_below_80_percent')
    for name in BASELINES:
        baseline=np.array([r['baselines'][name]['cost_normalized'] for r in rows])
        delta=learned-baseline;mean,lo,hi=bootstrap_mean_ci(delta,seed=7)
        fold_means={str(f):float(delta[[r['fold']==f for r in rows]].mean()) for f in folds}
        wins=sum(x<0 for x in fold_means.values())
        stable=hi<0 and len(folds)==5 and wins>=4
        comparisons[name]={'mean_policy_minus_baseline_cost':mean,'ci95':[lo,hi],
                           'fold_mean_differences':fold_means,'folds_better':wins,'stable_advantage':stable,
                           'incremental_oracle_gain_recovery':gain_recovery(baseline,oracle,learned)}
        if not stable:reasons.append('no_stable_advantage_over_'+name)
    return {'decision':'NO_GO' if reasons else 'GO','allow_active_optimization':not reasons,
            'recovery':recovery,'comparisons':comparisons,'reasons':reasons,'subjects':len(rows)}
