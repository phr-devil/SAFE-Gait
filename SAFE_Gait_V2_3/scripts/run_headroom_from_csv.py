from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Allow the documented direct source-tree invocation: python scripts/<name>.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from safe_gait.models.hierarchical_residual import HierarchicalResidualModel
from safe_gait.data.episodes import build_one_shot_episodes, build_all_one_shot_episodes
from safe_gait.evaluation.metrics import episode_absolute_errors
from safe_gait.evaluation.bootstrap import subject_bootstrap_ci


def _subject_aggregate(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.groupby("subject_id", as_index=False)[value_cols].mean()


def main():
    ap = argparse.ArgumentParser(
        description="Fit SAFE-Gait V2.3 from cross-fitted residual CSV and run leakage-safe headroom analysis."
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument("--targets", required=True, help="Comma-separated residual columns")
    ap.add_argument("--train-subjects", required=True, help="Text file: one training subject id per line")
    ap.add_argument("--eval-subjects", required=True, help="Text file: one validation/test subject id per line")
    ap.add_argument("--out", default="reports/headroom.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000, help="Subject-level bootstrap replicates")
    ap.add_argument("--order-col", default=None, help="Optional pass-order column used to define deterministic first support.")
    ap.add_argument(
        "--support-mode",
        choices=["first", "random", "all"],
        default="first",
        help="Primary deployment should normally use 'first'; 'all' is a subject-aggregated sensitivity analysis.",
    )
    args = ap.parse_args()

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    if not targets:
        raise ValueError("at least one target is required")
    df = pd.read_csv(args.csv)
    trids = set(Path(args.train_subjects).read_text().split())
    evids = set(Path(args.eval_subjects).read_text().split())
    if not trids or not evids:
        raise ValueError("train/eval subject lists must both be non-empty")
    if trids & evids:
        raise ValueError("train/eval subject overlap")

    available=set(df.subject_id.astype(str).unique())
    missing_train=trids-available; missing_eval=evids-available
    if missing_train or missing_eval:
        raise ValueError(f"split lists reference absent subjects: train={sorted(missing_train)}, eval={sorted(missing_eval)}")
    tr = df[df.subject_id.astype(str).isin(trids)].copy()
    ev = df[df.subject_id.astype(str).isin(evids)].copy()
    if tr.empty or ev.empty:
        raise ValueError("no rows matched train or eval subject lists")

    model = HierarchicalResidualModel(targets).fit(tr)
    result = {
        "summary": model.summary(),
        "support_mode": args.support_mode,
        "pairs": {},
    }

    for sc, qc in [("SP", "SP"), ("HP", "HP"), ("SP", "HP"), ("HP", "SP")]:
        if args.support_mode == "all":
            ep = build_all_one_shot_episodes(ev, targets, sc, qc, order_col=args.order_col)
        else:
            ep = build_one_shot_episodes(
                ev,
                targets,
                sc,
                qc,
                seed=args.seed,
                selection=args.support_mode,
                order_col=args.order_col,
            )

        rows = []
        for _, r in ep.iterrows():
            rs = np.array([r[f"support_{t}"] for t in targets], dtype=float)
            qvals = np.column_stack([np.asarray(r[f"query_values_{t}"], dtype=float) for t in targets])
            mu_q = model.population_bias(qc)
            post = model.posterior(
                rs,
                sc,
                qc,
                support_n=int(r.get("n_support", 1)),
                query_n=1,  # deployment predicts each future pass; repeated queries are evaluation replicates
            )
            raw_abs, calibrated_abs, personalized_abs = episode_absolute_errors(
                qvals, mu_q, post.correction_raw
            )
            row = {"subject_id": r["subject_id"]}
            for k, t in enumerate(targets):
                row[f"raw_{t}"] = raw_abs[k]
                row[f"cal_{t}"] = calibrated_abs[k]
                row[f"pers_{t}"] = personalized_abs[k]
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        if args.support_mode == "all" and not metrics_df.empty:
            # Multiple support choices from one subject are sensitivity replicates, not
            # independent subjects. Aggregate within subject before any headline metric.
            value_cols = [c for c in metrics_df.columns if c != "subject_id"]
            metrics_df = _subject_aggregate(metrics_df, value_cols)

        if metrics_df.empty:
            result["pairs"][f"{sc}->{qc}"] = {"n_subjects": 0}
            continue

        raw = np.column_stack([metrics_df[f"raw_{t}"].to_numpy() for t in targets])
        cal = np.column_stack([metrics_df[f"cal_{t}"].to_numpy() for t in targets])
        pers = np.column_stack([metrics_df[f"pers_{t}"].to_numpy() for t in targets])
        intercept_gain = raw - cal
        personalization_gain = cal - pers
        benefit = (pers < cal).astype(float)
        negative_transfer = (pers > cal).astype(float)
        pair_result = {
            "n_subjects": int(len(metrics_df)),
            "raw_population_mae": raw.mean(axis=0).tolist(),
            "calibrated_population_mae": cal.mean(axis=0).tolist(),
            "personalized_mae": pers.mean(axis=0).tolist(),
            "population_intercept_gain": intercept_gain.mean(axis=0).tolist(),
            "incremental_personalization_gain": personalization_gain.mean(axis=0).tolist(),
            "median_incremental_personalization_gain": np.median(personalization_gain, axis=0).tolist(),
            "subject_benefit_rate_vs_calibrated_population": benefit.mean(axis=0).tolist(),
            "raw_negative_transfer_rate_vs_calibrated_population": negative_transfer.mean(axis=0).tolist(),
        }
        if len(metrics_df) >= 2 and args.bootstrap >= 100:
            lo, hi = subject_bootstrap_ci(personalization_gain, n_boot=args.bootstrap, seed=args.seed)
            blo, bhi = subject_bootstrap_ci(benefit, n_boot=args.bootstrap, seed=args.seed + 1)
            nlo, nhi = subject_bootstrap_ci(negative_transfer, n_boot=args.bootstrap, seed=args.seed + 2)
            pair_result["incremental_personalization_gain_95ci"] = [lo.tolist(), hi.tolist()]
            pair_result["subject_benefit_rate_95ci"] = [blo.tolist(), bhi.tolist()]
            pair_result["raw_negative_transfer_rate_95ci"] = [nlo.tolist(), nhi.tolist()]
        result["pairs"][f"{sc}->{qc}"] = pair_result

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
