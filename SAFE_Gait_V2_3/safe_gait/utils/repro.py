from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int = 0, deterministic_torch: bool = True) -> None:
    """Seed Python/NumPy/PyTorch and request deterministic PyTorch execution.

    Note: changing PYTHONHASHSEED at runtime only affects child processes launched
    afterwards; Python's current-process hash seed is fixed at interpreter startup.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
