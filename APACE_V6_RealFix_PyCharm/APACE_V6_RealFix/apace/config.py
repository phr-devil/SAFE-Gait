from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data
