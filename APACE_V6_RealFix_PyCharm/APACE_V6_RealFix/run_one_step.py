"""Explicitly authorized one-step restart after multistep G4 NO_GO. No second action."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from run_real import PROJECT_ROOT
from run_g3 import digest,load_fold
from run_g4 import SEEDS,combined_decision,publish_results,record_failure
from run_cross_validation import _write,_fingerprint_data
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.real_pipeline import select_eligible_trials
from apace.training.loops import encode_trials,train_value_model
from apace.training.validation import validate_value
from apace.models.value import DistributionalValueModel
from apace.pipeline import _cal_query_ids
from apace.evaluation.oracle_headroom import ProtocolStudy,select_fixed
from apace.evaluation.value_recovery import uncertainty_protocol,summarize_seed
from apace.evaluation.one_step import (make_observations,build_state,feature_names,StateScaler,supervision,
                                      choose_one_step,CODE_ACTIONS)


def build_unlabeled_state(study,raw,base,device,actions,embedding_dim):
    candidates=[study.lookup[i] for i in study.cal]
    # Accessible raw dictionary is restricted to calibration-pressure entries.
    pressure_only_pool={i:raw[i] for i in study.cal}
    obs=make_observations(candidates,pressure_only_pool,base,device)
    state=build_state(obs,actions,embedding_dim,len(TARGET_ORDER))
    poisoned=[replace(r,target=np.full_like(r.target,np.nan)) for r in candidates]
    poisoned_raw={i:replace(r,target=np.full_like(r.target,np.nan)) for i,r in pressure_only_pool.items()}
    state_without_labels=build_state(make_observations(poisoned,poisoned_raw,base,device),actions,embedding_dim,len(TARGET_ORDER))
    if not np.array_equal(state,state_without_labels):raise ValueError('target leakage into one-step features')
    return state


def run_study(cv,out,device='cuda'):
    cv=Path(cv).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be new or empty')
    p=json.loads((cv/'protocol.json').read_text());old_summary=json.loads((cv/'summary.json').read_text())
    if old_summary['status']!='COMPLETED' or len(p['folds'])!=5:raise ValueError('complete five-fold source required')
    if _fingerprint_data(Path(p['manifest']))!=p['data_sha256']:raise ValueError('changed source data')
    for name,sha in p['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError('changed frozen model implementation')
    paths=[cv/f'fold_{i}/checkpoint.pt' for i in range(1,6)]
    meta={'source_cv':str(cv),'source_protocol_sha256':digest(cv/'protocol.json'),
          'source_checkpoints':{str(f):digest(f) for f in paths},'data_sha256':p['data_sha256'],
          'implementation':{str(f.relative_to(PROJECT_ROOT)):digest(f) for f in [Path(__file__).resolve(),
                            PROJECT_ROOT/'apace/evaluation/one_step.py',PROJECT_ROOT/'apace/evaluation/value_recovery.py',
                            PROJECT_ROOT/'run_g4.py']},
          'authorization':'User explicitly requested a one-step policy with observable individual features after multistep NO_GO.',
          'actions':['STOP','SP','HP'],'budget':1,'lambda':.005,'seeds':list(SEEDS),
          'epochs':100,'minimum_epochs':35,'patience':20,'batch_size':32,'learning_rate':.002,'hidden_dim':64,
          'state':'Protocol means of unlabeled candidate pressure embedding, zero-shot prediction/SD, prediction/SD across-pass spread, pressure gait proxies and quality; presence/count.',
          'normalization':'Fit state mean/SD on training subjects only, floor tiny SD to1, clip scaled features to[-8,8].',
          'label_rule':'Truth only supervises train/validation Oracle values or evaluates errors; state built from an unlabeled dataclass and checked after poisoning all reference targets.',
          'selection':'Validation quantile-loss checkpoint only. Positive highest median one-step net gain selects SP/HP, otherwise STOP.',
          'oracle':'Recompute budget-1 protocol Oracle with STOP and exact uniform outcome expectation.',
          'gate':'All3seeds OGR>=80%, paired upper cost-difference CI<0 vs fixed1/random1/uncertainty1, and >=4/5 fold wins.',
          'stop':'NO_GO exits2 and stops further optimization; no second acquisition ever executed.',
          'limitations':['Same frozen population/calibration models; only Value seeds repeated.',
                         'Dataset already used for model development; descriptive bootstrap with overlapping training folds.',
                         'Requires unlabeled pressure from candidate SP/HP trials before reference acquisition; no query pressure used.']}
    out.mkdir(parents=True,exist_ok=True);_write(out/'protocol.json',meta)
    rows_by_seed={str(seed):[] for seed in SEEDS};trainings=[];diagnostics=[];fold_info=[];started=time.perf_counter()
    try:
        good,_=select_eligible_trials(load_manifest(p['manifest'],TARGET_ORDER));raw={r.trial_id:r for r in good}
        expected={r.subject_id for r in good}
        for fold,split in enumerate(p['folds'],1):
            print(f'One-step fold {fold}/5: pressure features and budget-1 Oracle',flush=True)
            sets=[set(split[k]) for k in ['train','validation','test']]
            if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]) or set.union(*sets)!=expected:raise ValueError('subject split leakage')
            previous=json.loads((cv/f'fold_{fold}/real_metrics.json').read_text())
            if previous['split']!=split:raise ValueError('checkpoint split mismatch')
            c,encoder,base,evaluator=load_fold(paths[fold-1],previous,device)
            actions=c['action_to_index'];action_order=sorted(actions)
            if set(actions)!=set(CODE_ACTIONS.values()):raise ValueError('SP and HP source models required')
            mean=np.asarray(c['train_target_mean'],dtype=np.float32);std=np.asarray(c['train_target_std'],dtype=np.float32)
            studies={};states={}
            for part in ['train','validation','test']:
                trials=[replace(r,target=((r.target-mean)/std).astype(np.float32)) for r in good if r.subject_id in split[part]]
                by=defaultdict(list)
                for r in encode_trials(trials,encoder,base,device=device,batch_size=4):by[r.subject_id].append(r)
                if set(by)!=set(split[part]):raise ValueError('incomplete feature cohort')
                for sid,rs in by.items():
                    study=ProtocolStudy(rs,evaluator,*_cal_query_ids(rs),std,TARGET_ORDER,budget=1,lam=.005)
                    studies[sid]=study;states[sid]=build_unlabeled_state(study,raw,base,device,action_order,c['embedding_dim'])
            scaler=StateScaler.fit(np.stack([states[sid] for sid in split['train']]))
            scaled={sid:scaler.transform(x) for sid,x in states.items()}
            names=feature_names(action_order,c['embedding_dim'],len(TARGET_ORDER))
            selected=select_fixed([studies[sid] for sid in split['validation']],actions,1)
            info={'fold':fold,'feature_names':names,'state_dim':len(names),'state_mean':scaler.mean.tolist(),
                  'state_scale':scaler.scale.tolist(),'normalization_subjects':split['train'],'fixed1_selection':selected,
                  'label_poison_invariance_checks':len(states),'query_pressure_used':False}
            fold_info.append(info);_write(out/f'fold_{fold}_features.json',info)
            np.savez_compressed(out/f'fold_{fold}_states.npz',subjects=np.array(sorted(states)),
                                states=np.stack([states[sid] for sid in sorted(states)]))
            train_records=[r for sid in split['train'] for r in supervision(studies[sid],scaled[sid],sid)]
            val_records=[r for sid in split['validation'] for r in supervision(studies[sid],scaled[sid],sid)]
            if {r.subject_id for r in train_records}-sets[0] or {r.subject_id for r in val_records}-sets[1]:raise ValueError('supervision leakage')
            baselines={};oracles={};zeros={}
            old_test={r['subject']:r for r in previous['stages']['E']['results']}
            for sid in split['test']:
                study=studies[sid];oracles[sid]=study.oracle();zeros[sid]=study.fixed(())
                np.testing.assert_allclose(zeros[sid]['mae_normalized'],old_test[sid]['zero_shot_mae_normalized'],rtol=1e-5,atol=1e-6)
                if study.query_ids!=sorted(old_test[sid]['query_ids']):raise ValueError('query pool changed')
                uncertainty=uncertainty_protocol([study.lookup[i] for i in study.cal],base,device)
                baselines[sid]={'fixed1':study.fixed(selected['schedule']),'random1':study.random(1),'uncertainty1':study.fixed((uncertainty,))}
                diagnostics.append({'subject':sid,'fold':fold,'oracle':oracles[sid],'zero_shot':zeros[sid],
                                    'available':list(study.available(())),'calibration_ids':list(study.cal),'query_ids':study.query_ids})
            for seed in SEEDS:
                print(f'One-step fold {fold}/5 seed {seed}: training root Value',flush=True)
                torch.manual_seed(seed);np.random.seed(seed)
                model=DistributionalValueModel(len(names),len(actions),hidden_dim=64)
                training={}
                def progress(d):_write(out/'run_status.json',{'status':'RUNNING','fold':fold,'seed':seed,'training':d})
                train_value_model(train_records,model,actions,epochs=100,batch_size=32,lr=.002,device=device,seed=seed,
                                  val_records=val_records,min_epochs=35,patience=20,diagnostics=training,on_epoch=progress)
                model.eval()
                if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise FloatingPointError('nonfinite_one_step_checkpoint')
                fold_rows=[]
                for sid in split['test']:
                    study=studies[sid]
                    action,values=choose_one_step(model,scaled[sid],actions,list(study.available(())),device)
                    learned=study.fixed(()) if action=='STOP' else study.fixed((CODE_ACTIONS[action],))
                    learned.update(first_action=action,predicted_net_values=values)
                    if learned['reference_count']>1 or learned['cost_normalized']<oracles[sid]['cost_normalized']-1e-7:
                        raise ValueError('invalid one-step policy outcome')
                    fold_rows.append({'subject':sid,'fold':fold,'seed':seed,'zero_cost':zeros[sid]['cost_normalized'],
                                      'oracle_cost':oracles[sid]['cost_normalized'],'learned':learned,'baselines':baselines[sid]})
                rows_by_seed[str(seed)].extend(fold_rows)
                detail={'fold':fold,'seed':seed,'training':training,'train_records':len(train_records),'validation_records':len(val_records),
                        'train_quantile_loss':validate_value(train_records,model,actions,device=device),
                        'validation_quantile_loss':validate_value(val_records,model,actions,device=device)}
                trainings.append(detail);_write(out/f'fold_{fold}_seed_{seed}.json',{'training':detail,'results':fold_rows})
                torch.save({'value':{k:v.cpu() for k,v in model.state_dict().items()},'state_mean':scaler.mean.tolist(),
                            'state_scale':scaler.scale.tolist(),'state_dim':len(names),'feature_names':names,'hidden_dim':64,
                            'action_to_index':actions,'budget':1,'lambda':.005,'seed':seed,'training':training,
                            'source_checkpoint':str(paths[fold-1]),'source_checkpoint_sha256':digest(paths[fold-1])},
                           out/f'value_fold_{fold}_seed_{seed}.pt')
        summaries={}
        for seed,rows in rows_by_seed.items():
            if len(rows)!=len(expected) or {r['subject'] for r in rows}!=expected:raise ValueError('incomplete test coverage')
            summaries[seed]=summarize_seed(rows)
        gate=combined_decision(summaries)
        for name,sha in meta['source_checkpoints'].items():
            if digest(name)!=sha:raise ValueError('frozen model changed')
        summary={'status':'COMPLETED','policy_type':'one_step','gate':gate,'seeds':summaries,'subjects_per_seed':len(expected),
                 'trainings':trainings,'elapsed_seconds':time.perf_counter()-started,
                 'state_dimensions':[r['state_dim'] for r in fold_info],
                 'label_poison_invariance_checks':sum(r['label_poison_invariance_checks'] for r in fold_info)}
        publish_results(out,summary,rows_by_seed,diagnostics)
        return summary
    except Exception as exc:
        record_failure(out,exc);raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cv',type=Path,default=PROJECT_ROOT/'outputs/generalization_20260928/cv')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/one_step_20260928')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    result=run_study(args.cv,args.out,args.device)
    print(json.dumps({'gate':result['gate'],'recovery':{k:v['recovery'] for k,v in result['seeds'].items()}},indent=2))
    if not result['gate']['allow_active_optimization']:raise SystemExit(2)


if __name__=='__main__':main()
