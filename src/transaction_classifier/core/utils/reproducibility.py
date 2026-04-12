"""Seed management and environment snapshot for reproducibility."""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Pin all random-number generators to a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)


def get_reproducibility_info() -> dict[str, str]:
    """Capture library versions and platform details."""
    import platform

    import pandas
    import scipy
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pandas.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "os": platform.platform(),
    }
