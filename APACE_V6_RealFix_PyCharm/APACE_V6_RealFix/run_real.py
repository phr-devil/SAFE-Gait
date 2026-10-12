"""PyCharm runnable real-data entry point. Run with --help for options."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from apace.real_pipeline import run_real_pipeline
from apace.data.weargait import WearGaitAdapter, export_trials

PROJECT_ROOT = Path(__file__).resolve().parent


def load_data_paths(config):
    config = Path(config).resolve()
    values = yaml.safe_load(config.read_text(encoding='utf-8'))
    return {name: (config.parent / value).resolve() for name, value in values.items()}


def main():
    p=argparse.ArgumentParser(description='APACE-V6 validated real-data stages A-E')
    src=p.add_mutually_exclusive_group()
    src.add_argument('--manifest',type=Path,help='previously generated manifest.csv')
    src.add_argument('--raw-dir',type=Path,help='folder containing canonical *_SelfPace.csv and *_HurriedPace.csv files')
    p.add_argument('--paths-config',type=Path,default=PROJECT_ROOT/'configs/local_data.yaml')
    p.add_argument('--subjects',nargs='+',help='convert only these subject IDs (raw mode)')
    p.add_argument('--prepare-only',action='store_true',help='convert and audit without training')
    p.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/real_run')
    p.add_argument('--pilot',action='store_true',help='3+ subjects only: SOFTWARE SMOKE TEST, NOT scientific validation')
    p.add_argument('--test-subject');p.add_argument('--val-subject')
    p.add_argument('--epochs',type=int)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--max-refs',type=int,default=2)
    p.add_argument('--lam',type=float,default=0.005)
    args=p.parse_args()
    if args.manifest is not None and args.subjects:
        p.error('--subjects is only supported in raw mode')
    if args.epochs is not None and args.epochs < 1: p.error('--epochs must be >=1')
    if args.max_refs not in (0,1,2):
        p.error('--max-refs supports 0, 1 or 2; larger exact-oracle searches require a separate scalability study')
    manifest=args.manifest
    data_paths={}
    if manifest is None and args.raw_dir is None:
        data_paths=load_data_paths(args.paths_config)
        args.raw_dir=data_paths['weargait']
    if args.raw_dir is not None:
        if not args.raw_dir.is_dir(): p.error('--raw-dir must be a folder')
        selected=sorted(f for f in args.raw_dir.rglob('*.csv') if f.name.endswith(('_SelfPace.csv','_HurriedPace.csv')))
        if args.subjects:
            selected=[f for f in selected if f.name.split('_')[0] in args.subjects]
            missing=set(args.subjects)-{f.name.split('_')[0] for f in selected}
            if missing: p.error(f'no canonical files for subjects: {sorted(missing)}')
        if not selected: p.error('no canonical SP/HP raw CSV files found')
        trials=[]; rejected=[]; rejected_passes=[]; sources=[]; adapter=WearGaitAdapter()
        for path in selected:
            try:
                accepted=adapter.load_file(path)
                trials.extend(accepted)
                rejected_passes.extend(adapter.rejected_passes)
                sources.append({'file':str(path.resolve()),'accepted_passes':len(accepted)})
            except (ValueError,KeyError,IndexError,OSError) as exc:
                rejected.append({'file':str(path.resolve()),'reason':str(exc)})
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'conversion_audit.json').write_text(json.dumps({'selected_files':len(selected),
                    'accepted_passes':len(trials),'rejected':rejected,'rejected_passes':rejected_passes,
                    'source_files':sources,'dataset_paths':{k:str(v) for k,v in data_paths.items()}},indent=2),encoding='utf-8')
        if not trials: p.error('no valid pressure+reference passes; see conversion_audit.json')
        manifest=export_trials(trials,args.out/'converted')
    if args.prepare_only:
        if not manifest.is_file(): p.error('manifest does not exist')
        print(json.dumps({'status':'PREPARED','manifest':str(manifest)},indent=2))
        return
    report=run_real_pipeline(manifest,args.out,pilot=args.pilot,seed=args.seed,
        test_subject=args.test_subject,val_subject=args.val_subject,device=args.device,
        epochs=args.epochs,max_refs=args.max_refs,lam=args.lam)
    print(json.dumps({'status':report['status'],'halt_reason':report.get('halt_reason'),
        'stages':report.get('stages'), 'warnings':report.get('warnings')},ensure_ascii=False,indent=2))
    if report['status']=='HALTED': raise SystemExit(2)


if __name__=='__main__': main()
