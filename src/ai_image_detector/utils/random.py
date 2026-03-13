import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may not be installed in all environments
    torch = None  # type: ignore


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to improve reproducibility.

    Parameters
    ----------
    seed: int
        Seed value to use.
    deterministic: bool
        If True and torch is available, enable deterministic cuDNN behavior (may reduce performance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
