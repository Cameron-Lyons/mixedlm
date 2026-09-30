from numbers import Real

import numpy as np


def _validate_confidence_level(level: float) -> float:
    """Return a finite real confidence level before model or simulation work."""
    if isinstance(level, bool | np.bool_) or not isinstance(level, Real):
        raise TypeError("level must be a finite number strictly between 0 and 1")
    value = float(level)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")
    return value
