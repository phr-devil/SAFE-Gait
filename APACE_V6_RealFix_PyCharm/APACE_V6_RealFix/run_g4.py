"""One predeclared G4 experiment: train Value only, evaluate three seeds, then stop."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from run_g3 import digest,load_fold
from run_cross_validation import _write,_fingerprint_data
from run_real import PROJECT_ROOT
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.real_pipeline import select_eligible_trials,_records
from apace.training.loops import encode_trials,train_value_model
from apace.training.validation import validate_value
from apace.models.value import DistributionalValueModel
from apace.oracle.counterfactual import planning_state
from apace.evaluation.oracle_headroom import ProtocolStudy
from apace.pipeline import _cal_query_ids
from apace.evaluation.value_recovery import ValueController,evaluate_policy,uncertainty_protocol,summarize_seed

SEEDS=(7,17,27)


def record_failure(out,exc):
    failed={'status':'HALTED','reason':f'{type(exc).__name__}: {exc}',
            'gate':{'decision':'HALTED','allow_active_optimization':False},'allow_active_optimization':False}
    _write(out/'summary.json',failed)
    _write(out/'run_status.json',failed)


def publish_results(out,summary,rows_by_seed,root_diagnostics):
    try:
        _write(out/'subject_results.json',rows_by_seed)
        _write(out/'root_state_diagnostics.json',root_diagnostics)
        pd.json_normalize([r for rs in rows_by_seed.values() for r in rs]).to_csv(out/'subject_results.csv',index=False)
        _write(out/'summary.json',summary)
        _write(out/'run_status.json',{'status':'COMPLETED',**summary['gate']})
    except Exception as exc:
        record_failure(out,exc)
        raise


def combined_decision(summaries):
    if set(summaries)!=set(map(str,SEEDS)):raise ValueError('all three predefined seeds required')
    passed=all(s['decision']=='GO' for s in summaries.values())
    return {'decision':'GO' if passed else 'NO_GO','allow_active_optimization':passed,
            'reasons':{k:v['reasons'] for k,v in summaries.items() if v['reasons']},
            'rule':'All 3 seeds: recovery >=80%, upper paired cost-difference CI <0 vs each baseline, better in >=4/5 folds.'}


def run_study(g3,out,device='cuda'):
    g3=Path(g3).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be new or empty')
    g3_summary=json.loads((g3/'summary.json').read_text());g3_protocol=json.loads((g3/'protocol.json').read_text())
    if g3_summary['status']!='COMPLETED' or g3_summary['gate']['decision']!='GO':raise ValueError('G3 does not authorize active optimization')
    cv=Path(g3_protocol['source_cv']);p=json.loads((cv/'protocol.json').read_text())
    if digest(cv/'protocol.json')!=g3_protocol['source_protocol_sha256']:raise ValueError('changed CV protocol')
    if _fingerprint_data(Path(p['manifest']))!=g3_protocol['data_sha256']:raise ValueError('changed data')
    for name,sha in p['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError('changed frozen model code: '+name)
    for name,sha in g3_protocol['implementation'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError('changed G3 implementation')
    for name,sha in g3_protocol['checkpoints'].items():
        if digest(name)!=sha:raise ValueError('changed frozen checkpoint')
    old={r['subject']:r for r in json.loads((g3/'subject_results.json').read_text())}
    meta={'source_g3':str(g3),'g3_summary_sha256':digest(g3/'summary.json'),
          'g3_subject_results_sha256':digest(g3/'subject_results.json'),
          'source_checkpoints':g3_protocol['checkpoints'],'data_sha256':g3_protocol['data_sha256'],
          'implementation':{str(f.relative_to(PROJECT_ROOT)):digest(f) for f in
                            [Path(__file__).resolve(),PROJECT_ROOT/'apace/evaluation/value_recovery.py']},
          'seeds':list(SEEDS),'epochs':100,'minimum_epochs':35,'patience':20,'learning_rate':.002,
          'batch_size':32,'budget':2,'lambda':.005,'training_lambdas':[0.,.005,.01],
          'architecture':'Existing DistributionalValueModel; frozen belief + log1p budget + log1p available counts.',
          'selection':'Validation-only quantile-loss checkpoint; policy uses median net gain, STOP if all <=0.',
          'baselines':'fixed1 from G3 validation selection; random1 exact; uncertainty1 chooses protocol with highest mean predicted normalized SD on candidate pressure embeddings.',
          'recovery':'(mean zero cost - mean learned cost)/(mean zero cost - mean Oracle cost); retain all subjects, no clipping.',
          'stability':'All seeds OGR>=80%; for each baseline upper paired 95% cost-difference CI<0 and lower mean cost in >=4/5 folds.',
          'stop_rule':'NO_GO stops further architecture/hyperparameter optimization; report diagnostics, no automatic retry.',
          'scope':'Three new Value initializations; same frozen population/calibration models. Not 3 independent end-to-end fits.',
          'limitations':['Dataset has development history. Subject bootstrap intervals are descriptive; training folds overlap.',
                         'Oracle query labels are offline diagnostics; only train/validation Oracle records may train/select Value.']}
    out.mkdir(parents=True,exist_ok=True);_write(out/'protocol.json',meta)
    started=time.perf_counter();rows_by_seed={str(seed):[] for seed in SEEDS};trainings=[];root_diagnostics=[]
    try:
        good,_=select_eligible_trials(load_manifest(p['manifest'],TARGET_ORDER))
        expected={r.subject_id for r in good}
        for fold,split in enumerate(p['folds'],1):
            print(f'G4 fold {fold}/5: preparing train/validation Oracle records',flush=True)
            sets=[set(split[k]) for k in ['train','validation','test']]
            if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]) or set.union(*sets)!=expected:raise ValueError('invalid subject split')
            report=json.loads((cv/f'fold_{fold}/real_metrics.json').read_text())
            if report['split']!=split:raise ValueError('checkpoint split mismatch')
            c,encoder,base,evaluator=load_fold(cv/f'fold_{fold}/checkpoint.pt',report,device)
            if c['lam']!=.005 or c['max_refs']!=2:raise ValueError('checkpoint budget/cost mismatch')
            mean=np.asarray(c['train_target_mean'],dtype=np.float32);std=np.asarray(c['train_target_std'],dtype=np.float32)
            encoded={}
            for name in ['train','validation','test']:
                trials=[replace(r,target=((r.target-mean)/std).astype(np.float32)) for r in good if r.subject_id in split[name]]
                encoded[name]=encode_trials(trials,encoder,base,device=device,batch_size=4)
            records=_records(encoded['train'],evaluator,[0.,.005,.01],2)
            val_records=_records(encoded['validation'],evaluator,[0.,.005,.01],2)
            if {r.subject_id for r in records}-sets[0] or {r.subject_id for r in val_records}-sets[1]:raise ValueError('Oracle training leakage')
            by=defaultdict(list)
            for r in encoded['test']:by[r.subject_id].append(r)
            studies={};baselines={}
            for sid,rs in by.items():
                study=ProtocolStudy(rs,evaluator,*_cal_query_ids(rs),std,TARGET_ORDER,lam=.005)
                if study.query_ids!=old[sid]['query_ids'] or list(study.cal)!=old[sid]['calibration_ids']:raise ValueError('changed query/calibration pool')
                oracle=study.oracle();zero=study.fixed(())
                np.testing.assert_allclose([oracle['cost_normalized'],zero['cost_normalized']],
                    [old[sid]['oracle']['cost_normalized'],old[sid]['zero_shot']['cost_normalized']],atol=1e-7,rtol=1e-6)
                candidates=[study.lookup[i] for i in study.cal]
                uncertainty=uncertainty_protocol(candidates,base,device)
                baselines[sid]={'fixed1':old[sid]['comparisons']['fixed_1'],'random1':study.random(1),'uncertainty1':study.fixed((uncertainty,))}
                studies[sid]=study
                root=planning_state(evaluator.state_vector([]),candidates,c['action_to_index'],2)
                root_diagnostics.append({'subject':sid,'fold':fold,'state':root.tolist(),'oracle_first_action':oracle['first_action'],
                                         'uncertainty_action':uncertainty})
            for seed in SEEDS:
                print(f'G4 fold {fold}/5 seed {seed}: training Value',flush=True)
                torch.manual_seed(seed);np.random.seed(seed)
                model=DistributionalValueModel(c['value_state_dim'],len(c['action_to_index']),hidden_dim=c['embedding_dim'])
                diagnostics={}
                def progress(d):
                    _write(out/'run_status.json',{'status':'RUNNING','fold':fold,'seed':seed,'training':d})
                train_value_model(records,model,c['action_to_index'],epochs=100,batch_size=32,lr=.002,
                    device=device,seed=seed,val_records=val_records,min_epochs=35,patience=20,diagnostics=diagnostics,on_epoch=progress)
                model.eval()
                if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise FloatingPointError('nonfinite_value_checkpoint')
                controller=ValueController(evaluator,model,c['action_to_index'],2,.005,device)
                fold_rows=[]
                for sid,study in studies.items():
                    learned=evaluate_policy(study,controller)
                    if learned['cost_normalized']<old[sid]['oracle']['cost_normalized']-1e-7:raise ValueError('learned beats clairvoyant bound')
                    fold_rows.append({'subject':sid,'fold':fold,'seed':seed,'zero_cost':old[sid]['zero_shot']['cost_normalized'],
                                      'oracle_cost':old[sid]['oracle']['cost_normalized'],'learned':learned,'baselines':baselines[sid]})
                rows_by_seed[str(seed)].extend(fold_rows)
                training={'fold':fold,'seed':seed,'training':diagnostics,'train_records':len(records),'validation_records':len(val_records),
                          'train_quantile_loss':validate_value(records,model,c['action_to_index'],device=device),
                          'validation_quantile_loss':validate_value(val_records,model,c['action_to_index'],device=device)}
                trainings.append(training)
                _write(out/f'fold_{fold}_seed_{seed}.json',{'training':training,'results':fold_rows})
                torch.save({'value':{k:v.cpu() for k,v in model.state_dict().items()},'source_checkpoint':str(cv/f'fold_{fold}/checkpoint.pt'),
                            'source_checkpoint_sha256':digest(cv/f'fold_{fold}/checkpoint.pt'),'seed':seed,'training':diagnostics,
                            'value_state_dim':c['value_state_dim'],'action_to_index':c['action_to_index'],'hidden_dim':c['embedding_dim'],
                            'budget':2,'lambda':.005},out/f'value_fold_{fold}_seed_{seed}.pt')
        summaries={}
        for seed,rows in rows_by_seed.items():
            if len(rows)!=len(expected) or {r['subject'] for r in rows}!=expected:raise ValueError('incomplete out-of-fold coverage')
            summaries[seed]=summarize_seed(rows)
        gate=combined_decision(summaries)
        summary={'status':'COMPLETED','gate':gate,'seeds':summaries,'subjects_per_seed':len(expected),
                 'elapsed_seconds':time.perf_counter()-started,'trainings':trainings}
        for name,sha in meta['source_checkpoints'].items():
            if digest(name)!=sha:raise ValueError('source checkpoint changed')
        publish_results(out,summary,rows_by_seed,root_diagnostics)
        return summary
    except Exception as exc:
        record_failure(out,exc)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--g3',type=Path,default=PROJECT_ROOT/'outputs/g3_20260928')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/g4_20260928')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    result=run_study(args.g3,args.out,args.device)
    print(json.dumps({'gate':result['gate'],'recovery':{k:v['recovery'] for k,v in result['seeds'].items()}},indent=2))
    if not result['gate']['allow_active_optimization']:raise SystemExit(2)


if __name__=='__main__':main()
