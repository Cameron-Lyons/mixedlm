"""Input checks shared by normalized conditional likelihoods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def likelihood_inputs(
    y: NDArray[np.floating],
    mu: NDArray[np.floating],
    wt: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    response, mean, weights = (np.asarray(value, dtype=np.float64) for value in (y, mu, wt))
    if response.ndim != 1 or mean.shape != response.shape or weights.shape != response.shape:
        raise ValueError("Likelihood responses, means, and weights must be matching vectors")
    if not all(np.all(np.isfinite(value)) for value in (response, mean, weights)):
        raise ValueError("Likelihood responses, means, and weights must be finite")
    if np.any(weights <= 0):
        raise ValueError("Likelihood weights must be positive")
    return response, mean, weights


def whole_counts(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    """Allow floating-point roundoff from recovering counts from proportions."""
    rounded = np.rint(values)
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(values))
    if np.any(values < 0) or np.any(np.abs(values - rounded) > tolerance):
        raise ValueError(f"Normalized {name} likelihood requires nonnegative whole-number counts")
    return rounded
