from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from collections import defaultdict

import numpy as np
import pandas as pd

from .types import Trial

PRESSURE_COLUMNS = [f"LPressure{i}" for i in range(1, 17)] + [f"RPressure{i}" for i in range(1, 17)]
WALKWAY_COLUMNS = ["Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"]
TARGET_ORDER = ("velocity_cm_s", "step_length_cm", "stride_length_cm", "step_time_s", "stride_time_s")
ZENO_GRID_PITCH_CM = 1.27  # 0.5 inch sensor pitch


def _seconds(series: pd.Series) -> np.ndarray:
    out = series.astype(str).str.extract(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:sec|s)?\s*$", expand=False)
    return pd.to_numeric(out, errors="coerce").to_numpy(dtype=float)


def extract_walk_runs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Return half-open [start, end) row ranges for independent Walk passes."""
    if "GeneralEvent" not in frame.columns:
        raise ValueError("WearGait CSV is missing GeneralEvent")
    events = frame["GeneralEvent"].fillna("unlabeled").astype(str).to_numpy()
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(events) + 1):
        if i == len(events) or events[i] != events[start]:
            if events[start] == "Walk":
                runs.append((start, i))
            start = i
    return runs


def _parse_cells(row: pd.Series) -> list[tuple[float, float, float, str]]:
    if any(pd.isna(row.get(c)) for c in WALKWAY_COLUMNS):
        return []
    values = [str(row[c]).split("|") for c in WALKWAY_COLUMNS]
    n = min(map(len, values))
    cells: list[tuple[float, float, float, str]] = []
    for x, y, p, foot in zip(*(v[:n] for v in values)):
        try:
            cells.append((float(x), float(y), float(p), foot))
        except (TypeError, ValueError):
            continue
    return cells


def _footfalls(pass_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    """Extract reference footfalls from raw Zeno walkway cell activations."""
    times = _seconds(pass_frame["Time"])
    per_side: dict[str, list[tuple[int, float, float, float, float]]] = {"L": [], "R": []}
    for local_i, (_, row) in enumerate(pass_frame.iterrows()):
        grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for x, y, pressure, foot in _parse_cells(row):
            if foot in ("L", "R"):
                grouped[foot].append((x, y, pressure))
        for side in ("L", "R"):
            if grouped[side]:
                arr = np.asarray(grouped[side], dtype=float)
                weights = arr[:, 2].copy()
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    weights = np.ones(len(arr), dtype=float)
                cx = float(np.average(arr[:, 0], weights=weights))
                cy = float(np.average(arr[:, 1], weights=weights))
                per_side[side].append((local_i, times[local_i], cx, cy, float(weights.sum())))
            else:
                per_side[side].append((local_i, times[local_i], np.nan, np.nan, 0.0))

    episodes: list[dict[str, float | str]] = []
    for side in ("L", "R"):
        rows = per_side[side]
        active = np.asarray([r[4] > 0 for r in rows], dtype=bool)
        start: int | None = None
        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i
            if start is not None and ((not is_active) or i == len(active) - 1):
                end = i if is_active and i == len(active) - 1 else i - 1
                if end - start + 1 >= 2:
                    seg = rows[start : end + 1]
                    weights = np.asarray([r[4] for r in seg], dtype=float)
                    episodes.append({
                        "side": side,
                        "t0": float(seg[0][1]),
                        "t1": float(seg[-1][1]),
                        "x": float(np.average([r[2] for r in seg], weights=weights)),
                        "y": float(np.average([r[3] for r in seg], weights=weights)),
                    })
                start = None
    episodes.sort(key=lambda r: float(r["t0"]))
    return episodes


def walkway_pass_metrics(pass_frame: pd.DataFrame, grid_pitch_cm: float = ZENO_GRID_PITCH_CM) -> dict[str, float]:
    """Derive pass-level reference gait metrics from raw walkway activations.

    Uses footfall centroids on the Zeno grid. Metrics are intentionally limited to
    quantities that can be reconstructed robustly and independently for each pass.
    """
    required = {"Time", *WALKWAY_COLUMNS}
    missing = sorted(required - set(pass_frame.columns))
    if missing:
        raise ValueError(f"walkway columns missing: {missing}")
    footfalls = _footfalls(pass_frame)
    if len(footfalls) < 3:
        raise ValueError("fewer than 3 valid walkway footfalls in pass")
    if any(a['side'] == b['side'] for a, b in zip(footfalls, footfalls[1:])):
        raise ValueError('nonalternating footfalls: incomplete or fragmented reference contacts')

    step_lengths, step_times = [], []
    stride_lengths, stride_times = [], []
    for i in range(1, len(footfalls)):
        step_lengths.append(abs(float(footfalls[i]["x"]) - float(footfalls[i - 1]["x"])) * grid_pitch_cm)
        step_times.append(float(footfalls[i]["t0"]) - float(footfalls[i - 1]["t0"]))
    for i in range(2, len(footfalls)):
        if footfalls[i]["side"] == footfalls[i - 2]["side"]:
            stride_lengths.append(abs(float(footfalls[i]["x"]) - float(footfalls[i - 2]["x"])) * grid_pitch_cm)
            stride_times.append(float(footfalls[i]["t0"]) - float(footfalls[i - 2]["t0"]))

    elapsed = float(footfalls[-1]["t0"]) - float(footfalls[0]["t0"])
    if elapsed <= 0:
        raise ValueError("invalid footfall timing")
    velocity = abs(float(footfalls[-1]["x"]) - float(footfalls[0]["x"])) * grid_pitch_cm / elapsed
    return {
        "velocity_cm_s": float(velocity),
        "step_length_cm": float(np.mean(step_lengths)),
        "stride_length_cm": float(np.mean(stride_lengths)),
        "step_time_s": float(np.mean(step_times)),
        "stride_time_s": float(np.mean(stride_times)),
        "n_footfalls": int(len(footfalls)),
    }


def default_sensor_layout_features() -> np.ndarray:
    """Return deterministic identity/layout features for the fixed Moticon layout.

    WearGait-PD CSV files do not expose numeric (x,y) coordinates for the 16 sensing
    elements. APACE therefore uses two *identity* features rather than pretending
    these are physical coordinates: normalized sensor index and left/right side.
    """
    index = np.tile(np.linspace(-1.0, 1.0, 16, dtype=np.float32), 2)
    side = np.concatenate([np.full(16, -1.0, np.float32), np.full(16, 1.0, np.float32)])
    return np.stack([index, side], axis=1).astype(np.float32)


@dataclass
class WearGaitAdapter:
    max_pressure_nan_fraction: float = 0.05
    min_footfalls: int = 3
    rejected_passes: list[dict] = field(default_factory=list, init=False)

    def validate_columns(self, frame: pd.DataFrame) -> None:
        required = {"Time", "GeneralEvent", *PRESSURE_COLUMNS, *WALKWAY_COLUMNS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"WearGait CSV missing required columns: {missing}")

    def extract_pressure(self, pass_frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(PRESSURE_COLUMNS) - set(pass_frame.columns))
        if missing:
            raise ValueError(f"missing Moticon pressure channels: {missing}")
        numeric = pass_frame[PRESSURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        nan_fraction = float(numeric.isna().mean().mean())
        if nan_fraction > self.max_pressure_nan_fraction:
            raise ValueError(f"pressure NaN fraction {nan_fraction:.3f} exceeds limit {self.max_pressure_nan_fraction:.3f}")
        # Interpolate only the small packet gaps within an otherwise valid pass.
        numeric = numeric.interpolate(axis=0, limit_direction="both")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("pressure contains unrecoverable NaN/Inf")
        return numeric.to_numpy(dtype=np.float32).T

    def load_file(self, csv_path: str | Path, subject_id: str | None = None, condition: str | None = None) -> list[Trial]:
        path = Path(csv_path)
        self.rejected_passes.clear()
        required = {'Time', 'GeneralEvent', *PRESSURE_COLUMNS, *WALKWAY_COLUMNS}
        frame = pd.read_csv(path, usecols=lambda c: c in required, low_memory=False)
        self.validate_columns(frame)
        subject_id = subject_id or path.name.split("_")[0]
        if condition is None:
            if "SelfPace" in path.name:
                condition = "SelfPace"
            elif "HurriedPace" in path.name:
                condition = "HurriedPace"
            else:
                condition = "Unknown"
        layout = default_sensor_layout_features()
        trials: list[Trial] = []
        for pass_idx, (start, end) in enumerate(extract_walk_runs(frame), start=1):
            pass_frame = frame.iloc[start:end].copy()
            try:
                times = _seconds(pass_frame['Time'])
                delta = np.diff(times)
                if len(delta) == 0 or not np.isfinite(times).all() or np.any(delta <= 0):
                    raise ValueError('time must be finite and strictly increasing')
                dt = float(np.median(delta))
                if not np.allclose(delta, dt, rtol=0.05, atol=1e-6):
                    raise ValueError('irregular time intervals require resampling')
                pressure = self.extract_pressure(pass_frame)
                metrics = walkway_pass_metrics(pass_frame)
                if metrics['n_footfalls'] < self.min_footfalls:
                    raise ValueError('too few footfalls')
                target = np.asarray([metrics[k] for k in TARGET_ORDER], dtype=np.float32)
                if not np.isfinite(target).all() or np.any(target <= 0):
                    raise ValueError('reference targets must be finite and positive')
            except ValueError as exc:
                self.rejected_passes.append({'file': str(path.resolve()), 'pass_index': pass_idx,
                                            'start_row': start, 'end_row': end, 'reason': str(exc)})
                continue
            trials.append(Trial(
                subject_id=str(subject_id),
                trial_id=f"{subject_id}_{condition}_pass{pass_idx}",
                condition=str(condition),
                pressure=pressure,
                coords=layout.copy(),
                target=target,
                sample_interval_s=dt,
            ))
        return trials


def export_trials(trials: Iterable[Trial], output_dir: str | Path) -> Path:
    """Export WearGait trials into the generic APACE manifest layout."""
    trials = list(trials)
    ids = [t.trial_id for t in trials]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate trial_id: refusing to overwrite pressure arrays')
    if not trials:
        raise ValueError('cannot export empty trials')
    if any(len(t.target) != len(TARGET_ORDER) for t in trials):
        raise ValueError('WearGait export requires all five targets')
    output = Path(output_dir)
    pressure_dir = output / "pressure"
    coords_dir = output / "coords"
    pressure_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in trials:
        pressure_rel = Path("pressure") / f"{trial.trial_id}.npy"
        coords_rel = Path("coords") / f"{trial.trial_id}.npy"
        np.save(output / pressure_rel, trial.pressure.astype(np.float32))
        np.save(output / coords_rel, trial.coords.astype(np.float32))
        row = {
            "subject_id": trial.subject_id,
            "trial_id": trial.trial_id,
            "condition": trial.condition,
            "pressure_path": pressure_rel.as_posix(),
            "coords_path": coords_rel.as_posix(),
            "sample_interval_s": trial.sample_interval_s,
        }
        for key, value in zip(TARGET_ORDER, trial.target.tolist()):
            row[key] = float(value)
        rows.append(row)
    manifest = output / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
