"""Shared utilities for MARS agents."""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger("mars.utils")


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed for full reproducibility.

    Call at the start of training/evaluation, not at import time.
    Covers: Python random, NumPy, PyTorch CPU+GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic CuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Global seed set to %d", seed)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config, return empty dict if not found."""
    from pathlib import Path
    import yaml
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}
