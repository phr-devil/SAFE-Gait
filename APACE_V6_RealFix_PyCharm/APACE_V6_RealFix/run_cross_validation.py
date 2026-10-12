"""Frozen five-fold subject evaluation. Failed folds remain visible; no gate bypass."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from apace.data.manifest import load_manifest
from apace.data.weargait import TARGET_ORDER
from apace.real_pipeline import run_real_pipeline,select_eligible_trials
from apace.evaluation.cross_validation import make_subject_folds,summarize_subject_results
from run_real import PROJECT_ROOT


def _write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _fingerprint_data(manifest):
    digest=hashlib.sha256(manifest.read_bytes())
    frame=pd.read_csv(manifest)
    paths=sorted({str((manifest.parent/str(v)).resolve()) for col in ['pressure_path','coords_path'] for v in frame[col]})
    for value in paths:
        path=Path(value);digest.update(value.encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def run_study(manifest,out,*,epochs=100,device='cuda',seed=7,max_refs=2,lam=.005):
    manifest=Path(manifest).resolve();out=Path(out).resolve()
    if (out/'protocol.json').exists():raise ValueError('study already exists; choose a new output directory')
    raw=load_manifest(manifest,TARGET_ORDER);good,excluded=select_eligible_trials(raw)
    ids=sorted({r.subject_id for r in good})
    folds=make_subject_folds(ids,seed=seed)
    params={'epochs':epochs,'device':device,'seed':seed,'max_refs':max_refs,'lam':lam,'compare_baselines':True}
    source_paths=sorted((PROJECT_ROOT/'apace').rglob('*.py'))+[Path(__file__).resolve()]
    protocol={'manifest':str(manifest),'data_sha256':_fingerprint_data(manifest),
              'source_sha256':{str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              'parameters':params,'n_splits':5,'folds':folds,'eligible_subjects':len(ids),'eligible_trials':len(good),
              'excluded':excluded,'normalization':'each fold training subjects only','fixed_schedule_selection':'validation subjects only',
              'labels':'unchanged reconstructed WearGait targets; PKMAS used only for aggregate audit',
              'limitations':['Prior exploratory model development used this dataset; this is not a new untouched external test.',
                             'One preselected random seed; bootstrap intervals are descriptive, not independent training replications.']}
    out.mkdir(parents=True,exist_ok=True);_write(out/'protocol.json',protocol)
    del raw,good
    results=[];fold_reports=[];started=time.perf_counter()
    for index,split in enumerate(folds,start=1):
        _write(out/'run_status.json',{'status':'RUNNING','current_fold':index,'completed_folds':fold_reports})
        print(f'Fold {index}/5: train={len(split["train"])} validation={len(split["validation"])} test={len(split["test"])}',flush=True)
        report=run_real_pipeline(manifest,out/f'fold_{index}',subject_split=split,**params)
        entry={'fold':index,'status':report['status'],'test_subjects':split['test']}
        if report['status']=='HALTED':
            entry.update(halt_stage=report.get('halt_stage'),halt_reason=report.get('halt_reason'))
        else:
            rows=report['stages']['E']['results']
            if len(rows)!=len(split['test']) or {r['subject'] for r in rows}!=set(split['test']):
                raise ValueError('fold output does not cover the planned test subjects exactly')
            for row in rows:row['fold']=index
            results.extend(rows)
            entry['summary']=summarize_subject_results(rows,len(split['test']),seed=seed)
        fold_reports.append(entry)
        interim=summarize_subject_results(results,len(ids),seed=seed)
        interim.update(status='RUNNING',folds=fold_reports)
        _write(out/'summary.json',interim)
        pd.json_normalize(results).to_csv(out/'subject_results.csv',index=False)
        _write(out/'subject_results.json',results)
        print(f'Fold {index}: {entry["status"]}',flush=True)
    summary=summarize_subject_results(results,len(ids),seed=seed)
    summary.update(status='COMPLETED' if summary['complete_coverage'] else 'INCOMPLETE',folds=fold_reports,
                   elapsed_seconds=time.perf_counter()-started)
    _write(out/'summary.json',summary)
    _write(out/'run_status.json',{'status':summary['status'],'completed_folds':fold_reports})
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=PROJECT_ROOT/'outputs/local_prepared_20260928/converted/manifest.csv')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/generalization_20260928/cv')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    if args.epochs<1:parser.error('epochs must be positive')
    summary=run_study(args.manifest,args.out,epochs=args.epochs,device=args.device,seed=args.seed)
    print(json.dumps({k:summary[k] for k in ['status','evaluated_subjects','planned_subjects']},indent=2))
    if summary['status']!='COMPLETED':raise SystemExit(2)


if __name__=='__main__':main()
