"""G3 offline Oracle headroom on frozen out-of-fold checkpoints. No training."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from apace.real_pipeline import select_eligible_trials
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.models.encoder import PressureEncoder
from apace.models.predictor import ProbabilisticPredictor
from apace.models.calibration import CalibrationBeliefEncoder,PersonalizedPredictor
from apace.training.loops import encode_trials
from apace.pipeline import _cal_query_ids
from apace.oracle.counterfactual import CounterfactualEvaluator
from apace.evaluation.oracle_headroom import ProtocolStudy,select_fixed,decide_g3,BASELINES
from apace.evaluation.metrics import bootstrap_mean_ci
from run_cross_validation import _fingerprint_data,_write
from run_real import PROJECT_ROOT


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fold(checkpoint,report,device):
    c=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if c['format_version']!=2 or c['max_refs']!=2 or c['target_columns']!=list(TARGET_ORDER):
        raise ValueError('unsupported checkpoint protocol')
    for key in ['train_target_mean','train_target_std']:
        if not np.isfinite(c[key]).all() or not np.array_equal(c[key],report[key]):
            raise ValueError('checkpoint normalization mismatch')
    if np.any(np.asarray(c['train_target_std'])<=0):raise ValueError('invalid checkpoint scale')
    dim=c['embedding_dim'];n=len(TARGET_ORDER);actions=c['action_to_index']
    encoder=PressureEncoder(c['encoder_n_sensors'],hidden_dim=c['hidden_dim'],out_dim=dim)
    base=ProbabilisticPredictor(dim,n,hidden_dim=dim)
    belief=CalibrationBeliefEncoder(dim,n,len(actions),belief_dim=c['belief_dim'])
    personal=PersonalizedPredictor(dim,c['belief_dim'],n,hidden_dim=dim)
    for key,model in [('encoder',encoder),('base',base),('belief',belief),('personal',personal)]:
        model.load_state_dict(c[key]);model.to(device).eval().requires_grad_(False)
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise FloatingPointError('nonfinite_checkpoint')
    return c,encoder,base,CounterfactualEvaluator(belief,personal,actions,device=device)


def run_study(cv,out,device='cuda'):
    cv=Path(cv).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be new or empty')
    p=json.loads((cv/'protocol.json').read_text(encoding='utf-8'))
    old_summary=json.loads((cv/'summary.json').read_text(encoding='utf-8'))
    if old_summary['status']!='COMPLETED':raise ValueError('source cross-validation incomplete')
    if p['parameters']['max_refs']!=2:raise ValueError('G3 requires budget two')
    if _fingerprint_data(Path(p['manifest']))!=p['data_sha256']:raise ValueError('source data changed')
    for name,sha in p['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError(f'source model implementation changed: {name}')
    checkpoints=[cv/f'fold_{i}/checkpoint.pt' for i in range(1,len(p['folds'])+1)]
    metadata={'status':'FROZEN','source_cv':str(cv),'source_protocol_sha256':digest(cv/'protocol.json'),
              'checkpoints':{str(f):digest(f) for f in checkpoints},'data_sha256':p['data_sha256'],
              'implementation':{str(f.relative_to(PROJECT_ROOT)):digest(f) for f in
                                [Path(__file__).resolve(),PROJECT_ROOT/'apace/evaluation/oracle_headroom.py']},
              'budget':2,'lambda':p['parameters']['lam'],'min_effect':.05,'bootstrap_seed':7,
              'rule':'Each of four comparisons: mean relative cost gain >= 5% and 95% paired bootstrap lower bound > 0.',
              'oracle':'Clairvoyant protocol DP; STOP allowed; uniform outcomes without replacement; exact expectation.',
              'baselines':'Exact expected fixed lengths 1/2 selected on validation only; random uniform protocol then outcome.',
              'exhausted_fixed_protocol':'Choose lexicographically first available protocol without reading labels; stop only if all exhausted.',
              'test_use':'Offline upper-bound diagnostic only. No Oracle test records used in training.',
              'failure_rule':'NO_GO or INCONCLUSIVE disables further active optimization; no automatic retraining.',
              'limitations':['Dataset used in prior development; not untouched external replication.',
                             'Bootstrap is descriptive: folds share training samples.',
                             'Oracle may exploit query labels and overstates achievable deployment performance.']}
    out.mkdir(parents=True,exist_ok=True);_write(out/'protocol.json',metadata)
    started=time.perf_counter();results=[];folds=[]
    try:
        raw=load_manifest(p['manifest'],TARGET_ORDER);good,_=select_eligible_trials(raw)
        for i,split in enumerate(p['folds'],1):
            print(f'G3 fold {i}/{len(p["folds"])}',flush=True)
            sets=[set(split[k]) for k in ['train','validation','test']]
            if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('subject leakage')
            report=json.loads((cv/f'fold_{i}/real_metrics.json').read_text(encoding='utf-8'))
            if report['split']!=split:raise ValueError('fold split mismatch')
            c,encoder,base,evaluator=load_fold(checkpoints[i-1],report,device)
            if c['lam']!=metadata['lambda']:raise ValueError('checkpoint cost mismatch')
            mean=np.asarray(c['train_target_mean'],dtype=np.float32);std=np.asarray(c['train_target_std'],dtype=np.float32)
            def studies(subjects):
                trials=[replace(r,target=((r.target-mean)/std).astype(np.float32)) for r in good if r.subject_id in subjects]
                encoded=encode_trials(trials,encoder,base,device=device,batch_size=4)
                by=defaultdict(list)
                for r in encoded:by[r.subject_id].append(r)
                if set(by)!=subjects:raise ValueError('missing subjects')
                return {sid:ProtocolStudy(rows,evaluator,*_cal_query_ids(rows),std,TARGET_ORDER,lam=c['lam']) for sid,rows in by.items()}
            validation=studies(sets[1])
            selected={f'fixed_{n}':select_fixed(list(validation.values()),c['action_to_index'],n) for n in [1,2]}
            folds.append({'fold':i,'fixed_selection':selected,'validation_subjects':split['validation'],'test_subjects':split['test']})
            _write(out/f'fold_{i}_selection.json',folds[-1])
            test=studies(sets[2]);old={r['subject']:r for r in report['stages']['E']['results']}
            for sid,study in test.items():
                zero=study.fixed(())
                np.testing.assert_allclose(zero['mae_normalized'],old[sid]['zero_shot_mae_normalized'],rtol=1e-5,atol=1e-6)
                if study.query_ids!=sorted(old[sid]['query_ids']):raise ValueError('query pool changed')
                prior=study.report(study.terminal(tuple(sorted(old[sid]['reference_ids']))))
                np.testing.assert_allclose(prior['mae_normalized'],old[sid]['policy_mae_normalized'],rtol=1e-5,atol=1e-6)
                oracle=study.oracle()
                comparisons={k:study.fixed(v['schedule']) for k,v in selected.items()}
                comparisons.update({f'random_{n}':study.random(n) for n in [1,2]})
                if any(oracle['cost_normalized']>v['cost_normalized']+1e-7 for v in comparisons.values()):
                    raise ValueError('Oracle fails lower-bound invariant')
                schedules=[()] + [s for n in [1,2] for s in product(sorted(c['action_to_index']),repeat=n)]
                nonadaptive=min((study.fixed(seq) for seq in schedules),key=lambda r:r['cost_normalized'])
                results.append({'subject':sid,'fold':i,'oracle':oracle,'comparisons':comparisons,'zero_shot':zero,
                                'hindsight_best_nonadaptive':nonadaptive,'previous_learned_policy':prior,
                                'calibration_ids':list(study.cal),'query_ids':study.query_ids})
            _write(out/'subject_results.json',results)
            _write(out/'run_status.json',{'status':'RUNNING','completed_folds':i,'evaluated_subjects':len(results)})
        expected=set(sid for split in p['folds'] for sid in split['test'])
        if len(results)!=p['eligible_subjects'] or {r['subject'] for r in results}!=expected:
            raise ValueError('incomplete test coverage')
        gate=decide_g3(results)
        def interval(values):
            mean,lo,hi=bootstrap_mean_ci(np.asarray(values,dtype=float),seed=7)
            return {'mean':mean,'ci95':[lo,hi]}
        strategies={}
        for name in ['zero_shot','oracle','hindsight_best_nonadaptive','previous_learned_policy',*BASELINES]:
            vals=[r[name] if name in r else r['comparisons'][name] for r in results]
            strategies[name]={k:interval([v[k] for v in vals]) for k in ['mae_normalized','reference_count','cost_normalized']}
            strategies[name]['mae_physical']={k:interval([v['mae_physical'][k] for v in vals]) for k in TARGET_ORDER}
        if any(digest(Path(k))!=v for k,v in metadata['checkpoints'].items()):raise ValueError('checkpoint changed')
        summary={'status':'COMPLETED','gate':gate,'evaluated_subjects':len(results),'strategies':strategies,
                 'elapsed_seconds':time.perf_counter()-started,'folds':folds,'checkpoint_replay':'PASSED',
                 'checkpoints_unchanged':True}
        _write(out/'summary.json',summary)
        pd.json_normalize([{k:v for k,v in r.items() if k not in ['calibration_ids','query_ids']} for r in results]).to_csv(out/'subject_results.csv',index=False)
        _write(out/'run_status.json',{'status':'COMPLETED','gate':gate['decision'],'allow_active_optimization':gate['allow_active_optimization']})
        return summary
    except Exception as exc:
        _write(out/'run_status.json',{'status':'HALTED','reason':f'{type(exc).__name__}: {exc}',
                                     'evaluated_subjects':len(results),'allow_active_optimization':False})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cv',type=Path,default=PROJECT_ROOT/'outputs/generalization_20260928/cv')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/g3_20260928')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    summary=run_study(args.cv,args.out,args.device)
    print(json.dumps(summary['gate'],indent=2))
    if not summary['gate']['allow_active_optimization']:raise SystemExit(2)


if __name__=='__main__':main()
