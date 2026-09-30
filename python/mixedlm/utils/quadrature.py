from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite

_CACHE_SIZE = 32
_MAX_CACHED_ORDER = 1024


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _compute_rule(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = roots_hermite(n)
    # Immutable backing buffers prevent a caller from re-enabling writes to cached data.
    return np.frombuffer(nodes.tobytes(), dtype=np.float64), np.frombuffer(
        weights.tobytes(), dtype=np.float64
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_rule(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return _compute_rule(n)


def hermite_rule(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return an immutable rule for integration against exp(-x**2)."""
    n = _positive_integer(n, "n")
    # Bound retained bytes as well as the number of cached rules.
    return _cached_rule(n) if n <= _MAX_CACHED_ORDER else _compute_rule(n)
