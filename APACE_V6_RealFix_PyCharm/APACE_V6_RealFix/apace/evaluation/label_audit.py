"""Aggregate agreement with PKMAS; never substitutes summaries for pass labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from apace.data.weargait import TARGET_ORDER

PKMAS_COLUMNS = ('Velocity (cm./sec.)','Step Length (cm.)','Stride Length (cm.)',
                 'Step Time (sec.)','Stride Time (sec.)')
KEYS = ['subject_id','condition']


def load_pkmas(path):
    frame=pd.read_csv(Path(path),header=[0,1],dtype=str)
    def column(name,stat=None):
        matches=[c for c in frame.columns if c[0].strip()==name and
                 (c[1]=='Mean' if stat=='Mean' else c[1].startswith('Unnamed:'))]
        if len(matches)!=1: raise ValueError(f'ambiguous/missing PKMAS column: {name} {stat}')
        return frame[matches[0]]
    out=pd.DataFrame({'subject_id':column('Participant ID').str.strip(),
                      'condition':column('Task').str.strip(),
                      'group':column('PD vs Control').str.strip()})
    if out[KEYS].isna().any().any(): raise ValueError('missing official identifier')
    for key,name in zip(TARGET_ORDER,PKMAS_COLUMNS):
        out[key]=pd.to_numeric(column(name,None if key=='velocity_cm_s' else 'Mean'),errors='coerce')
    return out


def audit_labels(reconstructed,official):
    official=official.copy()
    reconstructed=reconstructed.copy()
    for frame in (official,reconstructed):
        if frame[KEYS].isna().any().any(): raise ValueError('missing label identifier')
        for k in KEYS: frame[k]=frame[k].astype(str).str.strip()
    if official.duplicated(KEYS).any(): raise ValueError('duplicate official subject/condition pairs')
    if not np.isfinite(reconstructed[list(TARGET_ORDER)].to_numpy(dtype=float)).all():
        raise FloatingPointError('nonfinite_reconstructed_labels: cannot aggregate NaN/Inf passes')
    averaged=reconstructed.groupby(KEYS,as_index=False)[list(TARGET_ORDER)].mean()
    counts=reconstructed.groupby(KEYS).size().rename('pass_count').reset_index()
    averaged=averaged.merge(counts,on=KEYS,validate='one_to_one')
    joined=averaged.merge(official,on=KEYS,suffixes=('_rebuilt','_official'),how='outer',indicator=True,validate='one_to_one')
    matched=joined.loc[joined['_merge']=='both'].drop(columns='_merge').copy()
    summary={'status':'AGGREGATE_AGREEMENT_ONLY','aggregation':'equal pass means per exact subject/condition',
             'matched_pairs':len(matched),'matched_subjects':matched.subject_id.nunique(),
             'unmatched_reconstructed_pairs':int((joined['_merge']=='left_only').sum()),
             'unmatched_official_pairs':int((joined['_merge']=='right_only').sum()),
             'unmatched_reconstructed':joined.loc[joined['_merge']=='left_only',KEYS].to_dict('records'),
             'unmatched_official':joined.loc[joined['_merge']=='right_only',KEYS].to_dict('records'),
             'metrics':{},'warnings':['Agreement of aggregated summaries does not establish per-pass accuracy.',
                                      'No alias, repeat-visit or condition substitutions are made.',
                                      'PKMAS event-weighted summaries may differ from equal-pass means.']}
    for key in TARGET_ORDER:
        x=matched[key+'_rebuilt'].to_numpy(dtype=float);y=matched[key+'_official'].to_numpy(dtype=float)
        valid=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        err=x[valid]-y[valid]
        corr=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>1 and np.std(x[valid])>0 and np.std(y[valid])>0 else None
        summary['metrics'][key]={'n_pairs':int(valid.sum()),'invalid_pairs':int((~valid).sum()),
            'mae':float(np.abs(err).mean()) if len(err) else None,
            'bias':float(err.mean()) if len(err) else None,
            'rmse':float(np.sqrt(np.mean(err**2))) if len(err) else None,
            'median_absolute_relative_error':float(np.median(abs(err)/y[valid])) if len(err) else None,
            'correlation':corr}
        matched[key+'_difference']=x-y
    return matched,summary
