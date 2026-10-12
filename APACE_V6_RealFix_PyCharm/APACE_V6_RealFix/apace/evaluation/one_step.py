"""One-step pressure-only state and STOP/SP/HP supervision. No labels in observations."""
from dataclasses import dataclass
import numpy as np
import torch
from apace.oracle.counterfactual import ValueRecord

ACTION_CODES={'SelfPace':'SP','HurriedPace':'HP'}
CODE_ACTIONS={v:k for k,v in ACTION_CODES.items()}
PRESSURE_FEATURES=('duration_s','total_mean','total_std','total_cv','left_right_asymmetry',
                   'left_contact_fraction','right_contact_fraction','dominant_frequency_hz','spectral_peak_fraction',
                   'relative_temporal_change','zero_fraction','negative_fraction','flat_fraction','repeated_frame_fraction')


@dataclass(frozen=True)
class PressureObservation:
    condition:str
    pressure:np.ndarray
    sample_interval_s:float
    embedding:np.ndarray
    prediction:np.ndarray
    uncertainty:np.ndarray


@torch.no_grad()
def make_observations(candidates,raw_by_id,base,device):
    """Whitelist only pressure/model outputs; never access target or query entries."""
    emb=torch.tensor(np.stack([r.embedding for r in candidates]),dtype=torch.float32,device=device)
    predictions,scales=base(emb)
    predictions=predictions.cpu().numpy();scales=scales.cpu().numpy()
    return [PressureObservation(r.condition,raw_by_id[r.trial_id].pressure,raw_by_id[r.trial_id].sample_interval_s,
                                r.embedding,predictions[i],scales[i]) for i,r in enumerate(candidates)]


def pressure_features(observation):
    x=np.asarray(observation.pressure,dtype=float);dt=observation.sample_interval_s
    if x.ndim!=2 or x.shape[0]<2 or x.shape[0]%2 or x.shape[1]<8 or not np.isfinite(dt) or dt<=0:
        raise ValueError('invalid pressure shape or time interval')
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite_pressure_features')
    half=x.shape[0]//2;load=np.maximum(x,0)
    left=load[:half].sum(0);right=load[half:].sum(0);total=left+right
    mean=float(total.mean());std=float(total.std());denom=max(mean,1e-8)
    spectrum=abs(np.fft.rfft(total-mean))**2;frequency=np.fft.rfftfreq(len(total),dt)
    mask=(frequency>=.4)&(frequency<=3.)
    peak=int(np.flatnonzero(mask)[np.argmax(spectrum[mask])]) if mask.any() and spectrum[mask].sum()>1e-12 else 0
    def contact(signal):return float(np.mean(signal>.05*signal.max())) if signal.max()>0 else 0.
    result=np.array([(x.shape[1]-1)*dt,mean,std,std/denom,
                     abs(float(left.mean()-right.mean()))/denom,contact(left),contact(right),
                     frequency[peak] if peak else 0.,spectrum[peak]/max(float(spectrum[1:].sum()),1e-8) if peak else 0.,
                     np.abs(np.diff(total)).mean()/denom,np.mean(x==0),np.mean(x<0),
                     np.mean(x.std(1)<=1e-8),np.mean(np.all(np.diff(x,axis=1)==0,axis=0))],dtype=np.float64)
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite_pressure_features')
    return result


def feature_names(actions,embedding_dim,target_dim):
    local=['available','log1p_count']+[f'embedding_mean_{i}' for i in range(embedding_dim)]
    for prefix in ['prediction_mean','prediction_std','uncertainty_mean','uncertainty_std']:
        local.extend(f'{prefix}_{i}' for i in range(target_dim))
    local.extend(PRESSURE_FEATURES)
    return [f'{a}.{name}' for a in actions for name in local]


def build_state(observations,actions,embedding_dim,target_dim):
    if not observations or len(set(actions))!=len(actions):raise ValueError('invalid state observations/actions')
    for r in observations:
        if r.condition not in actions:raise ValueError('unknown protocol')
        if r.embedding.shape!=(embedding_dim,) or r.prediction.shape!=(target_dim,) or r.uncertainty.shape!=(target_dim,):
            raise ValueError('invalid model feature shape')
        if not all(np.isfinite(v).all() for v in [r.embedding,r.prediction,r.uncertainty]):
            raise FloatingPointError('nonfinite_model_features')
        if np.any(r.uncertainty<=0):raise ValueError('uncertainty must be positive')
    blocks=[];width=2+embedding_dim+4*target_dim+len(PRESSURE_FEATURES)
    for action in actions:
        rows=[r for r in observations if r.condition==action]
        if not rows:blocks.append(np.zeros(width));continue
        e=np.stack([r.embedding for r in rows]);m=np.stack([r.prediction for r in rows]);u=np.stack([r.uncertainty for r in rows])
        blocks.append(np.r_[1.,np.log1p(len(rows)),e.mean(0),m.mean(0),m.std(0),u.mean(0),u.std(0),
                              np.stack([pressure_features(r) for r in rows]).mean(0)])
    state=np.concatenate(blocks).astype(np.float32)
    if not np.isfinite(state).all():raise FloatingPointError('nonfinite_one_step_state')
    return state


@dataclass(frozen=True)
class StateScaler:
    mean:np.ndarray
    scale:np.ndarray

    @classmethod
    def fit(cls,training_states):
        x=np.asarray(training_states,dtype=np.float64)
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('invalid training feature matrix')
        scale=x.std(0);return cls(x.mean(0),np.where(scale<1e-6,1.,scale))

    def transform(self,state):
        x=np.asarray(state,dtype=np.float64)
        if x.shape[-1]!=len(self.mean) or not np.isfinite(x).all():raise FloatingPointError('invalid state transform')
        value=np.clip((x-self.mean)/self.scale,-8.,8.).astype(np.float32)
        if not np.isfinite(value).all():raise FloatingPointError('nonfinite_scaled_state')
        return value


def supervision(study,state,subject):
    if study.budget!=1:raise ValueError('one-step supervision requires budget one')
    zero=study.fixed(())['cost_normalized'];records=[]
    for action in study.available(()):
        cost=study.fixed((action,))['cost_normalized']
        records.append(ValueRecord(subject,state.copy(),action,study.lam,zero-cost,zero,cost))
    return records


@torch.no_grad()
def choose_one_step(model,state,actions,available,device,lam=.005):
    x=np.asarray(state,dtype=np.float32)
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite_decision_state')
    if set(actions)-set(ACTION_CODES):raise ValueError('one-step actions must be SP/HP')
    values={}
    for name,index in actions.items():
        if name not in available:continue
        q=model(torch.tensor(x,device=device)[None,:],torch.tensor([index],device=device),torch.tensor([[lam]],device=device))
        if not torch.isfinite(q).all():raise FloatingPointError('nonfinite_one_step_values')
        values[ACTION_CODES[name]]=float(q[0,1].cpu())
    if not values or max(values.values())<=0:return 'STOP',values
    return max(values,key=values.get),values
