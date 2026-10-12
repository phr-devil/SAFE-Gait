from __future__ import annotations
import numpy as np
import pandas as pd
from .guards import assert_support_query_independent, assert_unique_pass_rows


def _validate_episode_df(
    residual_df: pd.DataFrame,
    targets: list[str],
    subject_col: str,
    condition_col: str,
    pass_col: str,
    order_col: str | None = None,
) -> None:
    needed = {subject_col, condition_col, pass_col, *targets}
    if order_col is not None:
        needed.add(order_col)
    missing = needed - set(residual_df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must be a non-empty list of unique column names")
    vals = residual_df[targets].to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError("residual target columns contain NaN/Inf")
    assert_unique_pass_rows(residual_df, subject_col, condition_col, pass_col)


def _validate_requested_conditions(
    residual_df: pd.DataFrame, support_condition: str, query_condition: str, condition_col: str
) -> None:
    observed = set(residual_df[condition_col].astype(str).unique())
    missing = {str(support_condition), str(query_condition)} - observed
    if missing:
        raise ValueError(f"requested condition(s) absent from residual table: {sorted(missing)}")


def _sort_condition_rows(df: pd.DataFrame, order_col: str | None) -> pd.DataFrame:
    if order_col is None:
        return df
    if df[order_col].isna().any():
        raise ValueError(f"order column {order_col!r} contains missing values")
    return df.sort_values(order_col, kind="mergesort")


def build_one_shot_episodes(
    residual_df: pd.DataFrame,
    targets: list[str],
    support_condition: str,
    query_condition: str,
    seed: int = 0,
    selection: str = "first",
    subject_col: str = "subject_id",
    condition_col: str = "condition",
    pass_col: str = "pass_id",
    order_col: str | None = None,
) -> pd.DataFrame:
    """Create exactly one subject-level one-shot episode per eligible subject.

    `selection='first'` is the deterministic default because the primary deployment
    protocol should not change its support pass with the model seed. Use
    `selection='random'` only for a predeclared sensitivity analysis.

    Query-pass residual vectors are retained for correct pass-wise MAE evaluation;
    their mean is also stored for residual-correlation analyses. For same-condition
    episodes, the support pass is excluded from the query set.
    """
    _validate_episode_df(residual_df, targets, subject_col, condition_col, pass_col, order_col)
    _validate_requested_conditions(residual_df, support_condition, query_condition, condition_col)
    residual_df = residual_df.copy()
    residual_df[subject_col] = residual_df[subject_col].astype(str)
    residual_df[condition_col] = residual_df[condition_col].astype(str)
    residual_df[pass_col] = residual_df[pass_col].astype(str)
    if selection not in {"first", "random"}:
        raise ValueError("selection must be 'first' or 'random'")
    rng = np.random.default_rng(seed)
    rows = []
    for sid, sdf in residual_df.groupby(subject_col, sort=True):
        sup = _sort_condition_rows(
            sdf[sdf[condition_col].astype(str) == str(support_condition)].copy(), order_col
        )
        qry = _sort_condition_rows(
            sdf[sdf[condition_col].astype(str) == str(query_condition)].copy(), order_col
        )
        if len(sup) < 1 or len(qry) < 1:
            continue
        # Stable input order is retained for the deterministic 'first' policy. Upstream
        # adapters should export passes chronologically if 'first' has temporal meaning.
        if selection == "random":
            idx = int(rng.integers(0, len(sup)))
        else:
            idx = 0
        srow = sup.iloc[idx]
        support_pid = str(srow[pass_col])
        if str(support_condition) == str(query_condition):
            qry = qry[qry[pass_col].astype(str) != support_pid]
        if len(qry) < 1:
            continue
        query_ids = qry[pass_col].astype(str).tolist()
        assert_support_query_independent(
            support_pid,
            query_ids,
            support_condition=support_condition,
            query_condition=query_condition,
        )
        out = {
            "subject_id": sid,
            "support_condition": str(support_condition),
            "query_condition": str(query_condition),
            "support_pass_id": support_pid,
            "query_pass_ids": "|".join(query_ids),
            "n_support": 1,
            "n_query": len(query_ids),
        }
        for t in targets:
            out[f"support_{t}"] = float(srow[t])
            qvals = tuple(float(v) for v in qry[t].to_numpy(float))
            out[f"query_values_{t}"] = qvals
            out[f"query_{t}"] = float(np.mean(qvals))
        rows.append(out)
    return pd.DataFrame(rows)


def build_all_one_shot_episodes(
    residual_df: pd.DataFrame,
    targets: list[str],
    support_condition: str,
    query_condition: str,
    subject_col: str = "subject_id",
    condition_col: str = "condition",
    pass_col: str = "pass_id",
    order_col: str | None = None,
) -> pd.DataFrame:
    """Enumerate every eligible support pass for sensitivity/headroom analysis.

    These rows are NOT independent subjects. Any metric computed from this output must
    first aggregate support-pass results within subject, then bootstrap/average subjects.
    """
    _validate_episode_df(residual_df, targets, subject_col, condition_col, pass_col, order_col)
    _validate_requested_conditions(residual_df, support_condition, query_condition, condition_col)
    residual_df = residual_df.copy()
    residual_df[subject_col] = residual_df[subject_col].astype(str)
    residual_df[condition_col] = residual_df[condition_col].astype(str)
    residual_df[pass_col] = residual_df[pass_col].astype(str)
    rows = []
    for sid, sdf in residual_df.groupby(subject_col, sort=True):
        sup = _sort_condition_rows(
            sdf[sdf[condition_col].astype(str) == str(support_condition)].copy(), order_col
        )
        qry_all = _sort_condition_rows(
            sdf[sdf[condition_col].astype(str) == str(query_condition)].copy(), order_col
        )
        if len(sup) < 1 or len(qry_all) < 1:
            continue
        for _, srow in sup.iterrows():
            support_pid = str(srow[pass_col])
            qry = qry_all.copy()
            if str(support_condition) == str(query_condition):
                qry = qry[qry[pass_col].astype(str) != support_pid]
            if len(qry) < 1:
                continue
            query_ids = qry[pass_col].astype(str).tolist()
            assert_support_query_independent(
                support_pid,
                query_ids,
                support_condition=support_condition,
                query_condition=query_condition,
            )
            out = {
                "subject_id": sid,
                "support_condition": str(support_condition),
                "query_condition": str(query_condition),
                "support_pass_id": support_pid,
                "query_pass_ids": "|".join(query_ids),
                "n_support": 1,
                "n_query": len(query_ids),
            }
            for t in targets:
                out[f"support_{t}"] = float(srow[t])
                qvals = tuple(float(v) for v in qry[t].to_numpy(float))
                out[f"query_values_{t}"] = qvals
                out[f"query_{t}"] = float(np.mean(qvals))
            rows.append(out)
    return pd.DataFrame(rows)
