from __future__ import annotations

import argparse
import json
from pathlib import Path

from apace.config import load_config
from apace.pipeline import run_synthetic_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Run APACE-V6 synthetic end-to-end demo")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="outputs/demo")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    metrics = run_synthetic_pipeline(
        output_dir=Path(args.output),
        seed=int(cfg.get("seed", 7)),
        quick=bool(args.quick),
        device=cfg.get("device"),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
