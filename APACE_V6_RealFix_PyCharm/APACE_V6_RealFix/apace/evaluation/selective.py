"""Binary benefit prediction: only fixed1 or STOP, no protocol planning."""
import numpy as np
import torch
from apace.training.validation import BestValidationCheckpoint
from apace.evaluation.metrics import bootstrap_mean_ci


def finite_vector(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x):raise ValueError('nonempty one-dimensional values required')
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite_selective_input')
    return x


def benefit_labels(never,fixed):
    a,b=finite_vector(never),finite_vector(fixed)
    if a.shape!=b.shape:raise ValueError('cost shape mismatch')
    return (a>b).astype(np.float32)


def decisions(probabilities,threshold):
    p=finite_vector(probabilities)
    if not np.isfinite(threshold) or np.any((p<0)|(p>1)):raise FloatingPointError('invalid_selective_probability')
    return p>=threshold


def select_threshold(probabilities,never,fixed):
    p=finite_vector(probabilities);n=finite_vector(never);f=finite_vector(fixed)
    if p.shape!=n.shape or p.shape!=f.shape:raise ValueError('validation size mismatch')
    scores=[{'threshold':float(t),'validation_mean_cost':float(np.where(decisions(p,t),f,n).mean())}
            for t in [*np.linspace(0,1,11),1.01]]
    best=min(scores,key=lambda r:(r['validation_mean_cost'],r['threshold']))
    return {**best,'candidates':scores,'source':'validation_only','tie_break':'lower threshold'}


@torch.no_grad()
def predict_probability(model,states,device):
    x=np.asarray(states,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise FloatingPointError('invalid_predictor_state')
    model.eval();p=torch.sigmoid(model(torch.tensor(x,device=device)).squeeze(-1)).cpu().numpy()
    finite_vector(p)
    return p


def train_predictor(train_x,train_y,val_x,val_y,*,seed,device,epochs=200,on_epoch=None):
    x=np.asarray(train_x,dtype=np.float32);v=np.asarray(val_x,dtype=np.float32)
    y=finite_vector(train_y);vy=finite_vector(val_y)
    if x.ndim!=2 or v.ndim!=2 or x.shape[1]!=v.shape[1] or len(x)!=len(y) or len(v)!=len(vy):raise ValueError('predictor dimensions')
    if not np.isfinite(x).all() or not np.isfinite(v).all():raise FloatingPointError('nonfinite_predictor_features')
    if np.any((y!=0)&(y!=1)) or np.any((vy!=0)&(vy!=1)):raise ValueError('binary targets required')
    torch.manual_seed(seed);model=torch.nn.Linear(x.shape[1],1).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.01);rng=np.random.default_rng(seed)
    tx=torch.tensor(x,device=device);ty=torch.tensor(y,dtype=torch.float32,device=device)
    vx=torch.tensor(v,device=device);vyt=torch.tensor(vy,dtype=torch.float32,device=device)
    criterion=torch.nn.BCEWithLogitsLoss();check=BestValidationCheckpoint([model],patience=20,min_epochs=30)
    for epoch in range(1,epochs+1):
        model.train()
        order=rng.permutation(len(x))
        for start in range(0,len(x),32):
            idx=torch.tensor(order[start:start+32],device=device)
            loss=criterion(model(tx[idx]).squeeze(-1),ty[idx])+.1*model.weight.square().sum()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_benefit_loss')
            opt.zero_grad();loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5.)):raise FloatingPointError('nonfinite_benefit_gradient')
            opt.step()
        model.eval()
        with torch.no_grad():
            metric=float(criterion(model(vx).squeeze(-1),vyt).cpu())
            training_loss=float(criterion(model(tx).squeeze(-1),ty).cpu())
        stop=check.check(metric,epoch)
        if not np.isfinite(training_loss):raise FloatingPointError('nonfinite_training_metric')
        diag={'epochs_run':epoch,'epoch_limit':epochs,'best_epoch':check.best_epoch,'best_validation_bce':check.stopping.best,
              'training_bce':training_loss,'validation_bce':metric,'stop_reason':'EARLY_STOPPING' if stop else 'EPOCH_LIMIT' if epoch==epochs else 'RUNNING'}
        if on_epoch:on_epoch(diag)
        if stop:break
    check.restore();model.eval()
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise FloatingPointError('nonfinite_benefit_model')
    diag['best_weights_restored']=True
    return model,diag


def interval(values):
    mean,lo,hi=bootstrap_mean_ci(finite_vector(values),seed=7)
    return {'mean':mean,'ci95':[lo,hi]}


def summarize_selective(rows):
    if not rows or len({r['subject'] for r in rows})!=len(rows):raise ValueError('empty or duplicate subject')
    strategies={}
    for key,name in [('selected','selective'),('fixed1','fixed1'),('never','never_calibrate')]:
        strategies[name]={metric:interval([r[key][metric] for r in rows]) for metric in ['cost_normalized','mae_normalized','reference_count']}
    strategies['always_calibrate']=strategies['fixed1']
    comparisons={}
    for key,name in [('fixed1','fixed1'),('never','never_calibrate')]:
        delta=np.array([r['selected']['cost_normalized']-r[key]['cost_normalized'] for r in rows])
        folds={str(f):float(delta[[r['fold']==f for r in rows]].mean()) for f in sorted({r['fold'] for r in rows})}
        comparisons[name]={**interval(delta),'fold_mean_differences':folds,'folds_better':sum(v<0 for v in folds.values())}
    comparisons['always_calibrate']=comparisons['fixed1']
    stable=comparisons['fixed1']['ci95'][1]<0 and len(comparisons['fixed1']['fold_mean_differences'])==5 and comparisons['fixed1']['folds_better']>=4
    y=benefit_labels([r['never']['cost_normalized'] for r in rows],[r['fixed1']['cost_normalized'] for r in rows])
    p=finite_vector([r['probability'] for r in rows]);chosen=np.array([r['calibrate'] for r in rows],dtype=bool)
    positives=p[y==1];negatives=p[y==0]
    auc=float(((positives[:,None]>negatives).astype(float)+.5*(positives[:,None]==negatives)).mean()) if len(positives) and len(negatives) else None
    return {'decision':'GO' if stable else 'NO_GO','allow_further_training':bool(stable),'subjects':len(rows),
            'baseline_alias':'always_calibrate is identical to fixed1','strategies':strategies,'comparisons':comparisons,
            'prediction':{'auroc':auc,'brier':float(np.mean((p-y)**2)),'benefit_subjects':int(y.sum()),
                          'beneficial_but_stopped':int(((y==1)&~chosen).sum()),'nonbeneficial_but_calibrated':int(((y==0)&chosen).sum()),
                          'nonbeneficial_correctly_stopped':int(((y==0)&~chosen).sum())}}
