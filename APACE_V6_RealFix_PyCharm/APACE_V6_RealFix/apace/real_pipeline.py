"""Actual manifest-driven A-E training, subject-separated validation, guarded pilot.

WARNING: a 3-subject pilot verifies execution, not accuracy/generalization.
"""
from __future__ import annotations
from collections import defaultdict
import math
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from apace.data.types import Trial
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.qc.checks import DataQualityGuard
from apace.models.encoder import PressureEncoder
from apace.models.predictor import ProbabilisticPredictor
from apace.models.calibration import CalibrationBeliefEncoder, PersonalizedPredictor
from apace.models.value import DistributionalValueModel
from apace.training.loops import train_population_model,encode_trials,train_calibration_model,train_value_model
from apace.training.validation import validate_population, validate_calibration, validate_value
from apace.oracle.counterfactual import CounterfactualEvaluator,build_protocol_value_records
from apace.pipeline import _cal_query_ids,_evaluate_subject_policy
from apace.training.guards import ResearchGate


def _finite_report_copy(value, path='report'):
    """Preserve failure evidence as null plus field paths in strict JSON."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None, [path]
    if isinstance(value, dict):
        clean, invalid = {}, []
        for key, item in value.items():
            clean[key], fields = _finite_report_copy(item, f'{path}.{key}')
            invalid.extend(fields)
        return clean, invalid
    if isinstance(value, (list, tuple)):
        clean, invalid = [], []
        for i, item in enumerate(value):
            parsed, fields = _finite_report_copy(item, f'{path}[{i}]')
            clean.append(parsed); invalid.extend(fields)
        return clean, invalid
    return value, []


def _subjects_split(trials, seed, test_subject=None, val_subject=None, *, pilot=False):
    groups=sorted({t.subject_id for t in trials})
    if len(groups) < 3:
        raise ValueError('need >=3 independent subjects for train/validation/test pilot; >=10 recommended for research gate')
    rng=np.random.default_rng(seed); rng.shuffle(groups)
    test_subject=test_subject or groups[0]
    if test_subject not in groups: raise ValueError('unknown test subject')
    available=[x for x in groups if x != test_subject]
    if val_subject is not None:
        if val_subject not in available: raise ValueError('validation subject must be distinct and present')
        if not pilot: raise ValueError('formal mode requires multiple validation subjects; omit --val-subject')
        val_ids={val_subject}
    elif pilot:
        val_ids={available[0]}
    else:
        n_val=max(3, math.ceil(len(available)*0.15))
        val_ids=set(available[:n_val])
    train=set(available)-val_ids
    return ([t for t in trials if t.subject_id in train],
            [t for t in trials if t.subject_id in val_ids],
            [t for t in trials if t.subject_id==test_subject],
            {'train':sorted(train),'validation':sorted(val_ids),'test':[test_subject]})


def _explicit_subject_split(trials, split, *, pilot=False):
    if set(split)!={'train','validation','test'}: raise ValueError('split requires train/validation/test')
    groups={k:set(v) for k,v in split.items()}
    if any(not v or len(v)!=len(split[k]) for k,v in groups.items()):
        raise ValueError('split groups must be nonempty and contain no duplicates')
    a,b,c=(groups[k] for k in ['train','validation','test'])
    if a&b or a&c or b&c: raise ValueError('subject split overlap')
    if a|b|c != {t.subject_id for t in trials}: raise ValueError('split must cover exactly all eligible subjects')
    if not pilot and len(b)<3: raise ValueError('formal split requires multiple validation subjects')
    return (*([t for t in trials if t.subject_id in groups[k]] for k in ['train','validation','test']),
            {k:sorted(v) for k,v in groups.items()})


def select_eligible_trials(raw):
    if not raw: raise ValueError('manifest empty')
    if len({t.trial_id for t in raw})!=len(raw): raise ValueError('duplicate trial_id: data leakage risk')
    qc=DataQualityGuard(max_nan_fraction=0.,max_flat_fraction=.75,min_timepoints=16)
    good=[];excluded=[];by_subject=defaultdict(list)
    for t in raw:
        result=qc.check_trial(t)
        if result.ok and np.isfinite(t.pressure).all(): by_subject[t.subject_id].append(t)
        else: excluded.append({'trial_id':t.trial_id,'reason':result.reasons or 'nonfinite_pressure'})
    for rows in by_subject.values():
        cal_ids,query_ids=_cal_query_ids(rows)
        if cal_ids and query_ids: good.extend(rows)
        else: excluded.extend({'trial_id':t.trial_id,'reason':'no independent calibration/query split'} for t in rows)
    return good,excluded


def _records(encoded, evaluator, lam_values, max_depth):
    by=defaultdict(list)
    for r in encoded: by[r.subject_id].append(r)
    records=[]
    for subject,rows in by.items():
        cal_ids, query_ids=_cal_query_ids(rows)
        if not cal_ids or not query_ids: continue
        records.extend(build_protocol_value_records(rows,evaluator,cal_ids,query_ids,
                        lambda_values=lam_values,max_depth=max_depth))
    return records


def _one_shot_oracle_gain(evaluator, candidates, query):
    """Optimistic one-reference headroom on validation only, including STOP."""
    err0 = evaluator.query_error([], query)
    errors = [err0] + [evaluator.query_error([r], query) for r in candidates]
    if not np.isfinite(errors).all():
        raise FloatingPointError('nonfinite_oracle_errors')
    best = min(errors)
    return (err0 - best) / max(err0, 1e-8)


def run_real_pipeline(manifest: str | Path, output_dir: str | Path, *,
                      pilot: bool=False, seed:int=7, test_subject:str|None=None,
                      val_subject:str|None=None, device:str|None=None,
                      epochs:int|None=None, max_refs:int=2, lam:float=0.005,
                      require_subjects:int=10, subject_split:dict|None=None,
                      compare_baselines:bool=False):
    if epochs is not None and epochs < 1: raise ValueError('epochs must be >=1')
    if max_refs not in (0,1,2) or not np.isfinite(lam) or lam < 0: raise ValueError('reference count must be 0, 1 or 2; lambda must be finite and nonnegative')
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but not available')
    if (device or 'cpu') == 'cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed); np.random.seed(seed)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    report={'status':'STARTED','is_pilot':pilot,'device':device,'manifest':str(manifest),
            'warnings':[], 'stages':{}, 'excluded':[],
            'parameters':{'seed':seed,'max_refs':max_refs,'lam':lam,'epoch_limit':epochs or (3 if pilot else 100)}}
    stage='G0'
    def serialize_report():
        report['current_stage']=stage
        clean,invalid=_finite_report_copy(report)
        if invalid:
            if report['status']!='HALTED':
                raise FloatingPointError('nonfinite_report_values: '+', '.join(invalid))
            report.clear();report.update(clean)
            report['invalid_numeric_fields']=invalid
        return json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    def save():
        text=serialize_report()
        temporary=output/'real_metrics.json.tmp'
        temporary.write_text(text,encoding='utf-8')
        temporary.replace(output/'real_metrics.json')
    def progress(stage_name):
        def record(diagnostics):
            report['stages'][stage_name]['training']=dict(diagnostics)
            save()
        return record
    try:
        raw=load_manifest(manifest,TARGET_ORDER)
        good,report['excluded']=select_eligible_trials(raw)
        subjects={t.subject_id for t in good}
        report['stages']['G0']={'total_trials':len(raw),'valid_trials':len(good),
                                 'valid_subjects':len(subjects),'excluded_trials':len(report['excluded'])}
        save()
        if len(subjects)<3: raise ValueError('G0: <3 eligible subjects')
        if len({t.pressure.shape[0] for t in good}) != 1:
            raise ValueError('G0: mixed sensor counts; train each device/dataset separately')
        if len(subjects)<require_subjects:
            report['warnings'].append(f'Only {len(subjects)} subjects: SOFTWARE SMOKE TEST ONLY; no research GO/NO-GO or publication claim')
            if not pilot: raise ValueError('insufficient independent subjects for formal mode; use --pilot')
        if subject_split is not None:
            if test_subject is not None or val_subject is not None: raise ValueError('explicit split conflicts with single-subject options')
            train,val,test,split=_explicit_subject_split(good,subject_split,pilot=pilot)
        else:
            train,val,test,split=_subjects_split(good,seed,test_subject,val_subject,pilot=pilot)
        report['split']=split; save()
        targets=np.stack([t.target for t in train]); mean=targets.mean(0); std=targets.std(0)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise FloatingPointError('nonfinite_target_normalization')
        std=np.where(std < 1e-4,1.,std)
        report['train_target_mean']=mean.tolist();report['train_target_std']=std.tolist()
        def norm(rows): return [replace(t,target=((t.target-mean)/std).astype(np.float32)) for t in rows]
        train,val,test=norm(train),norm(val),norm(test)
        conditions=sorted({t.condition for t in train})
        if any(t.condition not in conditions for t in val+test):
            raise ValueError('held-out data contains protocol absent from training')
        action_index={c:i for i,c in enumerate(conditions)}
        n_sensor=train[0].pressure.shape[0]; n_target=len(mean)
        dim=32 if pilot else 128
        hidden=16 if pilot else 64
        ep=epochs or (3 if pilot else 100)
        min_ep=2 if pilot else 35
        patience=2 if pilot else 20
        encoder=PressureEncoder(n_sensor,hidden_dim=hidden,out_dim=dim)
        base=ProbabilisticPredictor(dim,n_target,hidden_dim=dim)
        stage='A';training={}
        report['stages']['A']={'status':'RUNNING'};save()
        history=train_population_model(train,encoder,base,epochs=ep,batch_size=4 if pilot else 16,
                                        device=device,seed=seed,val_trials=val,min_epochs=min_ep,patience=patience,
                                        diagnostics=training,on_epoch=progress('A'))
        report['stages']['A']={'status':'COMPLETED','epochs':len(history),'training':training,
                               'validation_nll':validate_population(val,encoder,base,device=device)}
        save()
        stage='G1'
        encoded_train=encode_trials(train,encoder,base,device=device,batch_size=4)
        encoded_val=encode_trials(val,encoder,base,device=device,batch_size=4)
        encoded_test=encode_trials(test,encoder,base,device=device,batch_size=4)
        # G1: held-out-subject baseline comparison, never uses test references.
        by_train_cond=defaultdict(list)
        for row in train: by_train_cond[row.condition].append(row.target)
        cond_mean={c:np.mean(np.stack(v),axis=0) for c,v in by_train_cond.items()}
        by_val=defaultdict(list)
        for row in encoded_val: by_val[row.subject_id].append(row)
        base_gain=[]
        for rows in by_val.values():
            baseline=float(np.mean([np.mean(abs(r.target-cond_mean[r.condition])) for r in rows]))
            current=float(np.mean([np.mean(abs(r.target-r.base_mean)) for r in rows]))
            base_gain.append((baseline-current)/max(baseline,1e-8))
        if not np.isfinite(base_gain).all(): raise FloatingPointError('nonfinite_G1_effects')
        report['stages']['G1']={'status':'DIAGNOSTIC_ONLY' if pilot or len(by_val)<8 else 'CHECKED',
                    'subject_improvement_vs_condition_mean':base_gain}
        if not pilot and len(by_val)>=8:
            gate=ResearchGate(min_effect=.05).effect_gate(np.asarray(base_gain),n_boot=1000,seed=seed)
            report['stages']['G1']['decision']=gate.status
            report['stages']['G1'].update(mean_effect=gate.mean,lower_ci=gate.lower,upper_ci=gate.upper,
                                         min_effect=.05,validation_subjects=len(by_val))
            if gate.status=='NO_GO': raise ValueError('G1_NO_GO: population predictor has insufficient validated improvement over condition mean')
        save()
        encoder.requires_grad_(False);base.requires_grad_(False)
        belief=CalibrationBeliefEncoder(dim,n_target,len(conditions),belief_dim=16 if pilot else 64)
        personal=PersonalizedPredictor(dim,belief.belief_dim,n_target,hidden_dim=dim)
        stage='B';training={}
        report['stages']['B']={'status':'RUNNING'};save()
        history=train_calibration_model(encoded_train,belief,personal,action_index,
                   epochs=ep,episodes_per_epoch=8 if pilot else 100,max_context=max(1,max_refs),
                   device=device,seed=seed,val_trials=encoded_val,min_epochs=min_ep,patience=patience,
                   diagnostics=training,on_epoch=progress('B'))
        report['stages']['B']={'status':'COMPLETED','epochs':len(history),'training':training,
                     'validation_mae_normalized':validate_calibration(encoded_val,belief,personal,action_index,device=device,max_context=max(1,max_refs))}
        save()
        stage='G2'
        belief.requires_grad_(False);personal.requires_grad_(False)
        evaluator=CounterfactualEvaluator(belief.to(device),personal.to(device),action_index,device=device)
        by_val_subject=defaultdict(list)
        for r in encoded_val:by_val_subject[r.subject_id].append(r)
        cal_gains=[]
        for rows in by_val_subject.values():
            cal_ids,query_ids=_cal_query_ids(rows)
            lookup={r.trial_id:r for r in rows}
            query=[lookup[i] for i in sorted(query_ids)]
            candidates=[lookup[i] for i in sorted(cal_ids)]
            if not candidates or not query: continue
            cal_gains.append(_one_shot_oracle_gain(evaluator,candidates,query))
        if not np.isfinite(cal_gains).all(): raise FloatingPointError('nonfinite_G2_effects')
        report['stages']['G2']={'status':'DIAGNOSTIC_ONLY' if pilot or len(cal_gains)<8 or max_refs!=1 else 'CHECKED',
                                'oracle_one_shot_relative_gains':cal_gains,
                                'scope':'optimistic one-reference bound, not a multi-reference bound'}
        if not pilot and len(cal_gains)>=8 and max_refs == 1:
            gate=ResearchGate(min_effect=.05).effect_gate(np.asarray(cal_gains),n_boot=1000,seed=seed)
            report['stages']['G2']['decision']=gate.status
            report['stages']['G2'].update(mean_effect=gate.mean,lower_ci=gate.lower,upper_ci=gate.upper,
                                         min_effect=.05,validation_subjects=len(cal_gains))
            if gate.status=='NO_GO': raise ValueError('G2_NO_GO: one-reference calibration shows insufficient benefit')
        save()
        stage='C'
        report['stages']['C']={'status':'RUNNING'};save()
        # Train and val references only; NEVER construct oracle training records with test labels.
        cost_weights=sorted(set([0.,lam,max(0.01,lam*2)]))
        oracle_depth=max(1,max_refs)
        train_records=_records(encoded_train,evaluator,cost_weights,oracle_depth)
        val_records=_records(encoded_val,evaluator,cost_weights,oracle_depth)
        if len(train_records)<2 or not val_records:
            raise ValueError('G2: not enough independent pass-level calibration/query episodes')
        report['stages']['C']={'status':'COMPLETED','train_value_records':len(train_records),
                               'val_value_records':len(val_records),'oracle_max_depth':oracle_depth};save()
        value_state_dim=belief.belief_dim+1+len(conditions)
        value=DistributionalValueModel(value_state_dim,len(conditions),hidden_dim=dim)
        stage='D';training={}
        report['stages']['D']={'status':'RUNNING'};save()
        history=train_value_model(train_records,value,action_index,epochs=ep,batch_size=32,
                   device=device,seed=seed,val_records=val_records,min_epochs=min_ep,patience=patience,
                   diagnostics=training,on_epoch=progress('D'))
        report['stages']['D']={'status':'COMPLETED','epochs':len(history),'training':training,
                          'validation_quantile_loss':validate_value(val_records,value,action_index,device=device)}
        save();value.to(device).eval()
        if compare_baselines:
            from apace.evaluation.comparison import select_fixed_schedule,evaluate_reference_schedule
            stage='BASELINE_SELECTION'
            report['baseline_selection']=select_fixed_schedule(encoded_val,evaluator,conditions,std,TARGET_ORDER,
                                                                budget=max_refs,seed=seed,lam=lam)
            save()
        stage='E'
        report['stages']['E']={'status':'RUNNING'};save()
        by_test=defaultdict(list)
        for r in encoded_test: by_test[r.subject_id].append(r)
        results=[]
        for sid,rows in by_test.items():
            result=_evaluate_subject_policy(rows,evaluator,value,action_index,lam,max_refs,
                                           torch.device(device),seed=seed,return_details=True)
            lookup={r.trial_id:r for r in rows}
            query=[lookup[i] for i in result['query_ids']]
            context=[lookup[i] for i in result['reference_ids']]
            for label,ctx in [('zero_shot',[]),('policy',context)]:
                physical=(evaluator.query_errors(ctx,query)*std).mean(axis=0)
                result[label+'_mae_physical']=dict(zip(TARGET_ORDER,map(float,physical)))
            result['subject']=sid
            result['policy_cost_normalized']=result['policy_mae_normalized']+lam*result['reference_count']
            if compare_baselines:
                comparison={}
                for budget in range(1,max_refs+1):
                    comparison[f'random_{budget}']=evaluate_reference_schedule(rows,evaluator,std,TARGET_ORDER,
                                                                               budget=budget,seed=seed,lam=lam)
                comparison['fixed_validation_best']=evaluate_reference_schedule(rows,evaluator,std,TARGET_ORDER,
                    budget=max_refs,seed=seed,lam=lam,schedule=report['baseline_selection']['schedule'])
                result['comparisons']=comparison
            results.append(result)
        if not results: raise ValueError('empty held-out evaluation')
        report['stages']['E']={'status':'COMPLETED','results':results}
        report['status']='PILOT_ONLY' if len(subjects)<require_subjects or pilot else 'EVALUATED_NOT_PROVEN'
        report['warnings'].append('Scalar normalized MAE combines five parameters. Physical MAE is reported separately for each named target. No model-selection from test labels.')
        # DO NOT turn held-out test evidence into a training-time early-stopping gate.
        serialize_report()  # Reject invalid evaluation before publishing any model.
        stage='CHECKPOINT'
        torch.save({'encoder':encoder.cpu().state_dict(),'base':base.cpu().state_dict(),
                    'belief':belief.cpu().state_dict(),'personal':personal.cpu().state_dict(),
                    'value':value.cpu().state_dict(),'action_to_index':action_index,
                    'train_target_mean':mean.tolist(),'train_target_std':std.tolist(),
                    'target_columns':list(TARGET_ORDER),'belief_dim':belief.belief_dim,
                    'max_refs':max_refs,'lam':lam,'seed':seed,'format_version':2,
                    'value_state_dim':value_state_dim,'value_state_features':'belief,log1p_budget,log1p_remaining_counts_in_action_index_order',
                    'encoder_n_sensors':n_sensor,'embedding_dim':dim,'hidden_dim':hidden},output/'checkpoint.pt')
        stage='COMPLETE';save()
        return report
    except (ValueError,FloatingPointError,RuntimeError,MemoryError,OSError,KeyError) as exc:
        report['status']='HALTED'
        report['halt_stage']=stage
        report['halt_reason']=f'{type(exc).__name__}: {exc}'
        report['stages'].setdefault(stage,{})['status']='HALTED'
        save()
        return report
