from __future__ import annotations
import json
from pathlib import Path
import sys

# Allow the documented direct source-tree invocation: python scripts/<name>.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from safe_gait.models.hierarchical_residual import HierarchicalResidualModel
from safe_gait.data.episodes import build_one_shot_episodes
from safe_gait.evaluation.metrics import episode_absolute_errors

TARGETS = ["velocity", "step_length", "step_time"]
CONDS = ("SP", "HP")


def generate(seed=7, n_subjects=260, passes=4):
    rng = np.random.default_rng(seed)
    j = len(TARGETS)
    mu = {"SP": np.array([3.0, 1.2, -0.02]), "HP": np.array([5.0, 1.8, -0.03])}
    scale = np.array([10.0, 5.0, 0.10])
    su = np.array([[0.55, 0.15, -0.08], [0.15, 0.45, -0.05], [-0.08, -0.05, 0.35]])
    sv = {
        "SP": np.diag([0.25, 0.20, 0.25]),
        "HP": np.diag([0.35, 0.25, 0.30]),
    }
    se = {"SP": np.diag([0.25, 0.25, 0.20]), "HP": np.diag([0.30, 0.30, 0.25])}
    rows=[]
    for s in range(n_subjects):
        u=rng.multivariate_normal(np.zeros(j),su)
        for c in CONDS:
            v=rng.multivariate_normal(np.zeros(j),sv[c])
            for p in range(passes):
                e=rng.multivariate_normal(np.zeros(j),se[c])
                r=mu[c]+scale*(u+v+e)
                row={"subject_id":f"S{s:03d}","condition":c,"pass_id":f"{c}_{p}"}
                row.update({t:float(r[k]) for k,t in enumerate(TARGETS)})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    df=generate()
    subjects=sorted(df.subject_id.unique())
    train_ids=set(subjects[:180]); test_ids=set(subjects[180:])
    tr=df[df.subject_id.isin(train_ids)].copy(); te=df[df.subject_id.isin(test_ids)].copy()
    model=HierarchicalResidualModel(TARGETS, covariance_shrinkage=0.15).fit(tr)
    report={"model_summary":model.summary(),"pairs":{}}
    for sc,qc in [("SP","SP"),("HP","HP"),("SP","HP"),("HP","SP")]:
        ep=build_one_shot_episodes(te,TARGETS,sc,qc,seed=11)
        raw_base=[]; base=[]; pers=[]
        for _,r in ep.iterrows():
            rs=np.array([r[f"support_{t}"] for t in TARGETS])
            qvals=np.column_stack([np.asarray(r[f"query_values_{t}"],dtype=float) for t in TARGETS])
            post=model.posterior(rs,sc,qc,query_n=1)
            corr=post.correction_raw
            rb,b,p=episode_absolute_errors(qvals,model.population_bias(qc),corr)
            raw_base.append(rb); base.append(b); pers.append(p)
        raw_base=np.vstack(raw_base); base=np.vstack(base); pers=np.vstack(pers)
        report["pairs"][f"{sc}->{qc}"]={
            "n_subjects":len(ep),
            "raw_population_mae":raw_base.mean(axis=0).tolist(),
            "calibrated_population_mae":base.mean(axis=0).tolist(),
            "personalized_mae":pers.mean(axis=0).tolist(),
            "mean_gain":(base-pers).mean(axis=0).tolist(),
            "benefit_rate":np.mean(pers<base,axis=0).tolist(),
        }
    out=Path(__file__).resolve().parents[1]/"reports"/"synthetic_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(f"\nSaved: {out}")

if __name__=="__main__":
    main()
