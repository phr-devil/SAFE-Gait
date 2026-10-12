import numpy as np
import pandas as pd

from apace.data.weargait import (
    WearGaitAdapter,
    extract_walk_runs,
    walkway_pass_metrics,
)


def _toy_frame():
    rows = []
    # standing + two short walk passes separated by turn
    events = ['Standing']*2 + ['Walk']*6 + ['Turn']*2 + ['Walk']*6
    times = np.arange(len(events))*0.01
    for i, (t, ev) in enumerate(zip(times, events)):
        row = {
            'Time': f'{t:.2f} sec', 'GeneralEvent': ev,
            'Walkway_X': np.nan, 'Walkway_Y': np.nan,
            'WalkwayPressureLevel': np.nan, 'WalkwayFoot': np.nan,
        }
        for s in range(1,17):
            row[f'LPressure{s}'] = float(i+s)
            row[f'RPressure{s}'] = float(i+s+20)
        rows.append(row)
    # Create alternating walkway footfalls inside first walk run.
    # Each footfall lasts two frames and shifts 40 grid cells.
    for idx, (x, side) in zip([2,4,6], [(10,'L'),(50,'R'),(90,'L')]):
        for j in [idx, idx+1]:
            rows[j]['Walkway_X'] = f'{x}|{x+1}'
            rows[j]['Walkway_Y'] = '10|11'
            rows[j]['WalkwayPressureLevel'] = '5|5'
            rows[j]['WalkwayFoot'] = f'{side}|{side}'
    return pd.DataFrame(rows)


def test_extract_walk_runs_returns_independent_passes():
    df = _toy_frame()
    runs = extract_walk_runs(df)
    assert runs == [(2, 8), (10, 16)]


def test_weargait_pressure_extraction_is_32_by_time():
    df = _toy_frame()
    adapter = WearGaitAdapter()
    pressure = adapter.extract_pressure(df.iloc[2:8])
    assert pressure.shape == (32, 6)
    assert np.isfinite(pressure).all()


def test_walkway_metrics_recovers_step_and_stride_geometry():
    df = _toy_frame()
    m = walkway_pass_metrics(df.iloc[2:8])
    # Consecutive footfalls are 40 grid cells apart. Zeno grid pitch = 1.27 cm.
    assert np.isclose(m['step_length_cm'], 50.8, atol=0.5)
    # Same-foot stride is 80 grid cells.
    assert np.isclose(m['stride_length_cm'], 101.6, atol=0.5)
    assert m['n_footfalls'] == 3


def test_export_trials_writes_manifest_and_arrays(tmp_path):
    from apace.data.weargait import export_trials
    df = _toy_frame()
    adapter = WearGaitAdapter()
    # first toy walk has valid walkway labels, second does not, so create one Trial manually
    pass_df = df.iloc[2:8]
    metrics = walkway_pass_metrics(pass_df)
    from apace.data.types import Trial
    trial = Trial(
        subject_id='S1', trial_id='S1_SelfPace_pass1', condition='SelfPace',
        pressure=adapter.extract_pressure(pass_df),
        coords=np.zeros((32,2), dtype=np.float32),
        target=np.asarray([metrics[k] for k in ('velocity_cm_s','step_length_cm','stride_length_cm','step_time_s','stride_time_s')], dtype=np.float32),
    )
    manifest = export_trials([trial], tmp_path)
    assert manifest.exists()
    out = pd.read_csv(manifest)
    assert len(out) == 1
    assert (tmp_path / out.loc[0,'pressure_path']).exists()
    assert (tmp_path / out.loc[0,'coords_path']).exists()
