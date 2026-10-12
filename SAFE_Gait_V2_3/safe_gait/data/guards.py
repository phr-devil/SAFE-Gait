from __future__ import annotations
from typing import Iterable, Set
import pandas as pd


def _subjects(df: pd.DataFrame, subject_col: str = "subject_id") -> Set[str]:
    if subject_col not in df.columns:
        raise ValueError(f"missing subject column: {subject_col}")
    if df[subject_col].isna().any():
        raise ValueError(f"missing values in subject column: {subject_col}")
    ids = df[subject_col].astype(str)
    if (ids.str.len() == 0).any():
        raise ValueError(f"empty subject identifiers in column: {subject_col}")
    if (ids != ids.str.strip()).any():
        raise ValueError(f"subject identifiers contain leading/trailing whitespace: {subject_col}")
    return set(ids.unique())


def assert_disjoint_subjects(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    subject_col: str = "subject_id",
) -> None:
    tr, va, te = _subjects(train, subject_col), _subjects(val, subject_col), _subjects(test, subject_col)
    overlaps = {
        "train_val": tr & va,
        "train_test": tr & te,
        "val_test": va & te,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"subject leakage detected: {bad}")


def assert_unique_pass_rows(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    condition_col: str = "condition",
    pass_col: str = "pass_id",
) -> None:
    """Require one row per physical pass in the residual table.

    Duplicate subject-condition-pass rows commonly indicate cycle-level pseudo-replication
    or accidental duplicate exports, both of which invalidate the variance-component model.
    """
    needed = {subject_col, condition_col, pass_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"missing pass identity columns: {sorted(missing)}")
    if df[[subject_col, condition_col, pass_col]].isna().any().any():
        raise ValueError("subject/condition/pass identifiers must not be missing")
    keys = df[[subject_col, condition_col, pass_col]].astype(str)
    for col in (subject_col, condition_col, pass_col):
        if (keys[col].str.len() == 0).any():
            raise ValueError(f"empty identifier in column: {col}")
        if (keys[col] != keys[col].str.strip()).any():
            raise ValueError(f"identifier contains leading/trailing whitespace in column: {col}")
    dup = keys.duplicated(keep=False)
    if dup.any():
        sample = (
            df.loc[dup, [subject_col, condition_col, pass_col]]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "duplicate subject-condition-pass rows detected; expected pass-level residuals. "
            f"Examples: {sample}"
        )


def assert_support_query_independent(
    support_pass_id: str,
    query_pass_ids: Iterable[str],
    support_condition: str | None = None,
    query_condition: str | None = None,
) -> None:
    """Reject support/query reuse within the same condition.

    Pass identifiers are allowed to repeat across different conditions (e.g. pass '1' in
    both SP and HP); equality of a bare pass_id across conditions is not leakage.
    """
    if (
        support_condition is not None
        and query_condition is not None
        and str(support_condition) != str(query_condition)
    ):
        return
    q = {str(x) for x in query_pass_ids}
    if str(support_pass_id) in q:
        raise ValueError("support/query leakage: the support pass also appears in query passes")
