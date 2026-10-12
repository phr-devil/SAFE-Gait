"""Compare existing rebuilt pass targets with official condition-level summaries."""
import argparse
import json
from pathlib import Path
import pandas as pd
from apace.evaluation.label_audit import load_pkmas,audit_labels
from run_real import PROJECT_ROOT,load_data_paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=PROJECT_ROOT/'outputs/local_prepared_20260928/converted/manifest.csv')
    parser.add_argument('--pkmas',type=Path)
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/generalization_20260928/label_audit')
    args=parser.parse_args()
    official=args.pkmas or load_data_paths(PROJECT_ROOT/'configs/local_data.yaml')['weargait']/'PKMAS Walkway Gait Metrics - HP+SP.csv'
    frame=pd.read_csv(args.manifest,dtype={'subject_id':str,'trial_id':str,'condition':str})
    matched,report=audit_labels(frame,load_pkmas(official))
    report.update(manifest=str(args.manifest.resolve()),pkmas=str(official.resolve()))
    args.out.mkdir(parents=True,exist_ok=True)
    matched.to_csv(args.out/'matched_pairs.csv',index=False)
    (args.out/'label_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','matched_pairs','matched_subjects','metrics']},indent=2))


if __name__=='__main__':main()
