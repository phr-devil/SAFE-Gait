"""Predict whether fixed1 helps; no action/protocol optimization and no second acquisition."""
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
from run_g4 import SEEDS,publish_results,record_failure
from run_one_step import build_unlabeled_state
from run_cross_validation import _write,_fingerprint_data
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.real_pipeline import select_eligible_trials
from apace.training.loops import encode_trials
from apace.pipeline import _cal_query_ids
from apace.evaluation.oracle_headroom import ProtocolStudy,select_fixed
from apace.evaluation.one_step import StateScaler
from apace.evaluation.selective import (benefit_labels,train_predictor,predict_probability,
                                        select_threshold,decisions,summarize_selective,interval)


def analyze_mechanisms(rows):
    """Outcome-based descriptive analysis only; never used to fit the predictor."""
    gains=np.array([r['net_gain'] for r in rows]);raw=np.array([r['raw_error_gain'] for r in rows])
    base=np.array([r['never']['mae_normalized'] for r in rows])
    groups=[]
    for i,indices in enumerate(np.array_split(np.argsort(base,kind='stable'),4),1):
        groups.append({'zero_shot_error_quartile':i,'subjects':len(indices),'zero_shot_mae_mean':float(base[indices].mean()),
                       'net_benefit_fraction':float(np.mean(gains[indices]>0)),'mean_net_gain':float(gains[indices].mean())})
    return {'subjects':len(rows),'net_gain':interval(gains),'raw_error_gain':interval(raw),
            'net_benefit_subjects':int((gains>0).sum()),'raw_error_worsened_subjects':int((raw<0).sum()),
            'small_error_gain_outweighed_by_cost_subjects':int(((raw>=0)&(gains<=0)).sum()),
            'expected_benefit_but_some_references_harmful':sum(r['net_gain']>0 and min(r['candidate_net_gains'])<0 for r in rows),
            'mixed_sign_candidate_subjects':sum(min(r['candidate_net_gains'])<0<max(r['candidate_net_gains']) for r in rows),
            'zero_shot_error_quartiles':groups,
            'physical_error_gain':{k:interval([r['never']['mae_physical'][k]-r['fixed1']['mae_physical'][k] for r in rows]) for k in TARGET_ORDER},
            'interpretation':'Descriptive association, not causal proof. Actual zero-shot error and candidate gains require labels and are unavailable to deployment features.'}


def combined_gate(summaries):
    if set(summaries)!=set(map(str,SEEDS)):raise ValueError('three predefined seeds required')
    passed=all(s['decision']=='GO' for s in summaries.values())
    return {'decision':'GO' if passed else 'NO_GO','allow_further_training':passed,'allow_active_optimization':False,
            'next_mode':'selective_calibration' if passed else 'benefit_failure_mechanism_analysis',
            'rule':'All3seeds: paired upper95% CI of selective-fixed1 cost <0 and improvement in >=4/5 folds.'}


def run_study(source,out,device='cuda'):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be new or empty')
    source_meta=json.loads((source/'protocol.json').read_text());cv=Path(source_meta['source_cv']);p=json.loads((cv/'protocol.json').read_text())
    if len(p['folds'])!=5 or _fingerprint_data(Path(p['manifest']))!=source_meta['data_sha256']:raise ValueError('invalid or changed source')
    if digest(cv/'protocol.json')!=source_meta['source_protocol_sha256']:raise ValueError('changed source split protocol')
    for name,sha in p['source_sha256'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError('changed frozen model implementation')
    for name,sha in source_meta['source_checkpoints'].items():
        if digest(name)!=sha:raise ValueError('changed frozen checkpoint')
    for name,sha in source_meta['implementation'].items():
        if digest(PROJECT_ROOT/name)!=sha:raise ValueError('changed feature/source implementation')
    inputs=[source/'protocol.json',cv/'protocol.json']+[source/f'fold_{f}_{suffix}' for f in range(1,6) for suffix in ['features.json','states.npz']]
    meta={'source':str(source),'source_cv':str(cv),'inputs':{str(f):digest(f) for f in inputs},
          'source_checkpoints':source_meta['source_checkpoints'],'data_sha256':source_meta['data_sha256'],
          'implementation':{str(f.relative_to(PROJECT_ROOT)):digest(f) for f in [Path(__file__).resolve(),PROJECT_ROOT/'apace/evaluation/selective.py']},
          'seeds':list(SEEDS),'policy':'CALIBRATE using validation-selected fixed1, or STOP; no SP/HP choice by benefit predictor.',
          'benefit_label':'1 if Cnever-Cfixed1>0, where C=normalized MAE+0.005*reference count; otherwise0.',
          'features':'328 frozen unlabeled pressure features; no target/query-pressure; train-only standardization.',
          'predictor':'Linear sigmoid logistic model; random initialization/mini-batch order by seed.',
          'optimizer':'Adam lr0.01; BCE + 0.1*sum(weight^2), unpenalized intercept; batch32; no class rebalance.',
          'epochs':200,'minimum_epochs':30,'patience':20,'selection':'Validation BCE checkpoint, then validation expected-cost threshold selection.',
          'threshold_grid':[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.01],'threshold_ties':'smaller threshold',
          'baselines':'always-calibrate is exactly fixed1; never-calibrate is STOP. Do not count alias as independent comparator.',
          'gate':'All3seeds paired cost-difference upper95%CI<0 vs fixed1 and improvement in >=4/5 folds; no80%OGR criterion.',
          'failure_rule':'Stop selective training/optimization and perform descriptive calibration benefit/failure analysis.',
          'limitations':['Shared validation subjects select fixed1, checkpoint and threshold; test subjects never select anything.',
                         'Existing dataset development history; overlapping training folds, descriptive subject bootstrap.',
                         '3 seeds repeat benefit classifier training only; frozen base/calibration models.']}
    out.mkdir(parents=True,exist_ok=True);_write(out/'protocol.json',meta)
    rows_by_seed={str(s):[] for s in SEEDS};training_rows=[];mechanism_rows=[];started=time.perf_counter();label_checks=0
    try:
        good,_=select_eligible_trials(load_manifest(p['manifest'],TARGET_ORDER));raw={r.trial_id:r for r in good};expected={r.subject_id for r in good}
        for fold,split in enumerate(p['folds'],1):
            print(f'Selective fold {fold}/5: frozen fixed1 labels and pressure states',flush=True)
            sets=[set(split[k]) for k in ['train','validation','test']]
            if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]) or set.union(*sets)!=expected:raise ValueError('subject split leakage')
            previous=json.loads((cv/f'fold_{fold}/real_metrics.json').read_text())
            if previous['split']!=split:raise ValueError('checkpoint split mismatch')
            c,encoder,base,evaluator=load_fold(cv/f'fold_{fold}/checkpoint.pt',previous,device)
            mean=np.array(c['train_target_mean'],dtype=np.float32);std=np.array(c['train_target_std'],dtype=np.float32)
            data=np.load(source/f'fold_{fold}_states.npz');states=dict(zip(data['subjects'].tolist(),data['states']))
            scaler=StateScaler.fit(np.stack([states[sid] for sid in split['train']]))
            feature_info=json.loads((source/f'fold_{fold}_features.json').read_text())
            np.testing.assert_array_equal(scaler.mean,feature_info['state_mean']);np.testing.assert_array_equal(scaler.scale,feature_info['state_scale'])
            studies={}
            for part in ['train','validation','test']:
                trials=[replace(r,target=((r.target-mean)/std).astype(np.float32)) for r in good if r.subject_id in split[part]]
                by=defaultdict(list)
                for r in encode_trials(trials,encoder,base,device=device,batch_size=4):by[r.subject_id].append(r)
                if set(by)!=set(split[part]):raise ValueError('incomplete cohort')
                for sid,rs in by.items():
                    st=ProtocolStudy(rs,evaluator,*_cal_query_ids(rs),std,TARGET_ORDER,budget=1,lam=.005);studies[sid]=st
                    state=build_unlabeled_state(st,raw,base,device,sorted(c['action_to_index']),c['embedding_dim'])
                    np.testing.assert_array_equal(state,states[sid]);label_checks+=1
            fixed=select_fixed([studies[sid] for sid in split['validation']],c['action_to_index'],1)
            if fixed['schedule']!=feature_info['fixed1_selection']['schedule']:raise ValueError('fixed1 changed')
            never={sid:st.fixed(()) for sid,st in studies.items()};cal={sid:st.fixed(fixed['schedule']) for sid,st in studies.items()}
            x={part:scaler.transform(np.stack([states[sid] for sid in split[part]])) for part in ['train','validation','test']}
            y={part:benefit_labels([never[sid]['cost_normalized'] for sid in split[part]],
                                  [cal[sid]['cost_normalized'] for sid in split[part]]) for part in ['train','validation']}
            _write(out/f'fold_{fold}_protocol.json',{'fixed1':fixed,'train_subjects':split['train'],'validation_subjects':split['validation'],
                'test_subjects':split['test'],'state_mean':scaler.mean.tolist(),'state_scale':scaler.scale.tolist(),
                'training_benefit_fraction':float(y['train'].mean()),'validation_benefit_fraction':float(y['validation'].mean())})
            for sid in split['test']:
                st=studies[sid];available=st.available(());action=fixed['schedule'][0]
                if action not in available:action=sorted(available)[0]
                gains=[never[sid]['cost_normalized']-st.report(st.terminal((trial,)))['cost_normalized'] for trial in available[action]]
                mechanism_rows.append({'subject':sid,'fold':fold,'fixed_protocol':action,'never':never[sid],'fixed1':cal[sid],
                                       'net_gain':never[sid]['cost_normalized']-cal[sid]['cost_normalized'],
                                       'raw_error_gain':never[sid]['mae_normalized']-cal[sid]['mae_normalized'],
                                       'candidate_net_gains':gains})
            for seed in SEEDS:
                print(f'Selective fold {fold}/5 seed {seed}: fitting benefit predictor',flush=True)
                def progress(d):_write(out/'run_status.json',{'status':'RUNNING','fold':fold,'seed':seed,'training':d})
                model,training=train_predictor(x['train'],y['train'],x['validation'],y['validation'],seed=seed,device=device,on_epoch=progress)
                pv=predict_probability(model,x['validation'],device)
                threshold=select_threshold(pv,[never[sid]['cost_normalized'] for sid in split['validation']],
                                             [cal[sid]['cost_normalized'] for sid in split['validation']])
                pt=predict_probability(model,x['test'],device);chosen=decisions(pt,threshold['threshold']);fold_rows=[]
                for sid,prob,decision in zip(split['test'],pt,chosen):
                    fold_rows.append({'subject':sid,'fold':fold,'seed':seed,'probability':float(prob),'threshold':threshold['threshold'],
                                      'calibrate':bool(decision),'selected':cal[sid] if decision else never[sid],
                                      'fixed1':cal[sid],'never':never[sid]})
                rows_by_seed[str(seed)].extend(fold_rows)
                detail={'fold':fold,'seed':seed,'training':training,'threshold_selection':threshold};training_rows.append(detail)
                _write(out/f'fold_{fold}_seed_{seed}.json',{'training':detail,'results':fold_rows})
                torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'state_mean':scaler.mean.tolist(),
                            'state_scale':scaler.scale.tolist(),'state_dim':x['train'].shape[1],'threshold':threshold['threshold'],
                            'fixed1':fixed['schedule'],'seed':seed,'training':training},out/f'benefit_fold_{fold}_seed_{seed}.pt')
        summaries={}
        for seed,rows in rows_by_seed.items():
            if len(rows)!=len(expected) or {r['subject'] for r in rows}!=expected:raise ValueError('incomplete subject coverage')
            summaries[seed]=summarize_selective(rows)
        gate=combined_gate(summaries)
        for name,sha in {**meta['inputs'],**meta['source_checkpoints']}.items():
            if digest(name)!=sha:raise ValueError('frozen inputs changed')
        analysis=analyze_mechanisms(mechanism_rows)
        _write(out/'mechanism_data.json',mechanism_rows);_write(out/'mechanism_analysis.json',analysis)
        summary={'status':'COMPLETED','gate':gate,'seeds':summaries,'subjects_per_seed':len(expected),'training':training_rows,
                 'label_poison_checks':label_checks,'elapsed_seconds':time.perf_counter()-started,
                 'analysis_mode_entered':gate['decision']=='NO_GO'}
        publish_results(out,summary,rows_by_seed,mechanism_rows)
        return summary
    except Exception as exc:
        record_failure(out,exc);raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=PROJECT_ROOT/'outputs/one_step_20260928')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/selective_20260929')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if args.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    s=run_study(args.source,args.out,args.device)
    print(json.dumps(s['gate'],indent=2))
    if not s['gate']['allow_further_training']:raise SystemExit(2)


if __name__=='__main__':main()
