import math
from numbers import Integral, Real

import numpy as np


def _validate_confidence_level(level: float, name: str = "level") -> float:
    """Return a finite real confidence level before model or simulation work."""
    if isinstance(level, bool | np.bool_) or not isinstance(level, Real):
        raise TypeError(f"{name} must be a finite number strictly between 0 and 1")
    value = float(level)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite number strictly between 0 and 1")
    return value


def _validate_inner_controls(
    maxiter: int | None, tol: float, *, maxiter_name: str, tol_name: str
) -> None:
    """Check an inner solver's iteration limit (None for no limit) and tolerance."""
    if maxiter is not None and (
        isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter <= 0
    ):
        raise ValueError(f"{maxiter_name} must be a positive integer or None")
    valid_tol = isinstance(tol, Real) and not isinstance(tol, bool)
    try:
        valid_tol = valid_tol and math.isfinite(tol) and tol > 0
    except OverflowError:
        valid_tol = False
    if not valid_tol:
        raise ValueError(f"{tol_name} must be positive and finite")
