from __future__ import annotations

import numpy as np
from .types import Trial


def generate_synthetic_trials(
    n_subjects: int = 12,
    trials_per_condition: int = 4,
    n_sensors: int = 8,
    timepoints: int = 48,
    conditions: tuple[str, ...] = ("slow", "normal", "fast"),
    seed: int = 0,
) -> list[Trial]:
    """Generate a small dataset with genuine subject-specific calibration headroom."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_sensors, dtype=np.float32)
    coords = np.stack([x, np.sin(np.pi * (x + 1) / 2)], axis=1).astype(np.float32)
    speed_map = {c: v for c, v in zip(conditions, np.linspace(0.8, 1.6, len(conditions)))}
    trials: list[Trial] = []
    t = np.linspace(0, 2 * np.pi, timepoints, dtype=np.float32)
    for s in range(n_subjects):
        subject_bias = rng.normal(0, [0.12, 0.08]).astype(np.float32)
        foot_pattern = rng.normal(0, 0.10, size=n_sensors).astype(np.float32)
        for cond in conditions:
            base_speed = float(speed_map[cond])
            for r in range(trials_per_condition):
                phase = rng.uniform(-0.15, 0.15)
                amp = base_speed * (1.0 + 0.10 * subject_bias[0])
                waves = []
                for i in range(n_sensors):
                    wave = amp * (1 + 0.2 * foot_pattern[i]) * (np.sin(t + phase + i * 0.12).clip(min=0))
                    wave += 0.15 * np.sin(2 * t + i * 0.05)
                    wave += rng.normal(0, 0.03, size=timepoints)
                    waves.append(wave)
                pressure = np.asarray(waves, dtype=np.float32)
                target = np.asarray([
                    base_speed + subject_bias[0] + rng.normal(0, 0.015),
                    0.55 + 0.35 * base_speed + subject_bias[1] + rng.normal(0, 0.012),
                ], dtype=np.float32)
                trials.append(Trial(
                    subject_id=f"S{s:03d}",
                    trial_id=f"S{s:03d}_{cond}_{r}",
                    condition=cond,
                    pressure=pressure,
                    coords=coords.copy(),
                    target=target,
                ))
    return trials
