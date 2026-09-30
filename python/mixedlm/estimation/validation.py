from __future__ import annotations

from typing import Any

import numpy as np


def validate_finite_real(name: str, values: Any, shape: tuple[int, ...]) -> None:
    """Validate final estimates without coercing away complex or malformed values."""
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isrealobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
