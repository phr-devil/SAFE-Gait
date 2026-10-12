from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd

from .types import Trial


def load_manifest(
    manifest_path: str | Path,
    target_columns: Sequence[str],
    pressure_column: str = "pressure_path",
    coords_column: str = "coords_path",
) -> list[Trial]:
    """Load a normalized APACE manifest.

    Paths are resolved relative to the manifest. Pressure arrays must be
    [n_sensors, n_time] and coordinate arrays [n_sensors, 2].
    """
    path = Path(manifest_path)
    frame = pd.read_csv(path, dtype={'subject_id': str, 'trial_id': str, 'condition': str})
    required = {"subject_id", "trial_id", "condition", pressure_column, coords_column, *target_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if frame[['subject_id', 'trial_id', 'condition']].isna().any().any():
        raise ValueError('manifest has missing subject/trial/condition identifiers')
    trials: list[Trial] = []
    for row in frame.to_dict(orient="records"):
        pressure = np.load(path.parent / str(row[pressure_column])).astype(np.float32)
        coords = np.load(path.parent / str(row[coords_column])).astype(np.float32)
        target = np.asarray([row[c] for c in target_columns], dtype=np.float32)
        trials.append(
            Trial(
                subject_id=str(row["subject_id"]),
                trial_id=str(row["trial_id"]),
                condition=str(row["condition"]),
                pressure=pressure,
                coords=coords,
                target=target,
                site=None if pd.isna(row.get("site")) else str(row.get("site")),
                sample_interval_s=float(row.get("sample_interval_s", 0.01)),
            )
        )
    return trials
