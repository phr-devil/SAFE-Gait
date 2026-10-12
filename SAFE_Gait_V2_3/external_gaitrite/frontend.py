from dataclasses import dataclass
from collections import Counter
import numpy as np
import pandas as pd

PRESSURE_COLUMNS = [f'PressureSensor {i}' for i in range(32)]

@dataclass
class PressurePass:
    subject_id: str
    pass_id: str
    pressure: np.ndarray

def read_pressure(left_path,right_path,subject_id,pass_id):
    sides = []
    for path in (left_path, right_path):
        frame = pd.read_csv(path)
        columns = [c for c in frame if c.startswith('PressureSensor ')]
        if set(columns) != set(PRESSURE_COLUMNS):
            raise ValueError('Expected all 32 pressure channels per foot')
        sides.append(frame[PRESSURE_COLUMNS].to_numpy(dtype=float).T)
    if sides[0].shape[1] != sides[1].shape[1]:
        raise ValueError('Bilateral sample counts differ')
    pressure = np.concatenate(sides, axis=0)
    _validate_pressure(pressure)
    return PressurePass(str(subject_id), str(pass_id), pressure)

def _validate_pressure(pressure):
    if (pressure.ndim != 2 or pressure.shape[0] != 64 or pressure.shape[1] < 1
            or not np.isfinite(pressure).all() or (pressure < 0).any()):
        raise ValueError('Expected finite nonnegative 64-channel pressure')

class GaitRiteFrontend:
    """Pressure-only frontend; never reads reference or synchronization metadata.

    Channel order is L0..L31,R0..R31. Released clips are kept in full.
    Pooling is temporal only and includes the final incomplete block.
    """
    def __init__(self,pool_factor=4):
        if not isinstance(pool_factor, int) or pool_factor < 1:
            raise ValueError('pool_factor must be a positive integer')
        self.pool_factor=pool_factor

    def pool(self,pressure):
        pressure = np.asarray(pressure, dtype=float)
        _validate_pressure(pressure)
        return np.stack([pressure[:,i:i+self.pool_factor].mean(axis=1)
                         for i in range(0,pressure.shape[1],self.pool_factor)],axis=1)

    def fit(self,records,fit_subjects):
        ids = set(map(str,fit_subjects))
        selected = [r for r in records if r.subject_id in ids]
        if not ids or {r.subject_id for r in selected} != ids:
            raise ValueError('Missing preprocessing fit subjects')
        if len({r.pass_id for r in selected}) != len(selected):
            raise ValueError('Duplicate physical pass')
        counts=Counter(r.subject_id for r in selected)
        w=np.array([1/counts[r.subject_id] for r in selected]);w/=w.sum()
        x=[np.log1p(self.pool(r.pressure)) for r in selected]
        self.mean_=np.sum(np.stack([a.mean(axis=1) for a in x])*w[:,None],axis=0)
        second=np.sum(np.stack([(a*a).mean(axis=1) for a in x])*w[:,None],axis=0)
        sd=np.sqrt(np.maximum(second-self.mean_**2,0))
        self.std_=np.where(sd>1e-6,sd,1.)
        self.fit_subjects_=sorted(ids)
        self.fit_pass_ids_=sorted(r.pass_id for r in selected)
        return self

    def transform(self,records):
        if not hasattr(self,'mean_'):
            raise ValueError('Frontend must be fitted on training subjects first')
        if not records:
            raise ValueError('No pressure passes')
        a=[(np.log1p(self.pool(r.pressure))-self.mean_[:,None])/self.std_[:,None]
           for r in records]
        n=max(z.shape[1] for z in a)
        x=np.zeros((len(a),64,n),dtype=np.float32)
        mask=np.zeros((len(a),n),dtype=bool)
        for i,z in enumerate(a):
            x[i,:,:z.shape[1]]=z;mask[i,:z.shape[1]]=True
        return x,mask

def read_reference(path,subject_id):
    """Complete within-clip heel-based steps: cm/s, cm, seconds.

    Exclude the first available event even when its exported step is positive:
    its predecessor is outside the released reference. Never use stride velocity
    or condition-level summary labels as a physical-pass target.
    """
    f=pd.read_csv(path,sep=';').sort_values('FirstContact').reset_index(drop=True)
    required={'PT_Code','Pt_id','Gait_Id','TestSetId','Event','Foot','FirstContact',
              'LastContact','Step_length','Step_time','Loc_heel_X'}
    if not required <= set(f) or len(f)<3:
        raise ValueError('Missing or insufficient reference')
    if set(f.PT_Code)!={subject_id}:
        raise ValueError('Reference subject mismatch')
    for col in ['Pt_id','Gait_Id','TestSetId']:
        if f[col].isna().any() or f[col].nunique()!=1:
            raise ValueError('Mixed reference identities')
    a=f[['Event','Foot','FirstContact','LastContact','Step_length','Step_time','Loc_heel_X']].to_numpy(float)
    if not np.isfinite(a).all() or not (np.diff(a[:,0])==1).all():
        raise ValueError('Invalid reference events')
    if set(a[:,1])!={0.,1.} or (np.diff(a[:,1])==0).any():
        raise ValueError('Nonalternating reference')
    if not (np.diff(a[:,2])>0).all() or not (a[:,3]>a[:,2]).all():
        raise ValueError('Invalid contact times')
    length=a[1:,4];time=a[1:,5]
    if not (length>0).all() or not (time>0).all():
        raise ValueError('Nonpositive bounded steps')
    if not np.allclose(time,np.diff(a[:,2]),atol=1/180,rtol=0):
        raise ValueError('Step time disagrees with first contacts')
    if not np.allclose(length,abs(np.diff(a[:,6]))*1.27,atol=1e-4,rtol=0):
        raise ValueError('Step length disagrees with heel coordinates')
    return {'targets':np.array([length.sum()/time.sum(),length.mean(),time.mean()]),
            'units':['cm/s','cm','s'],'n_bounded_steps':len(length),
            'leading_step_excluded':True,
            'positive_leading_step_excluded':bool(a[0,4]>0 or a[0,5]>0),
            'gait_id':str(f.Gait_Id.iloc[0]),'subject_id':subject_id}
