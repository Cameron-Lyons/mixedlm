"""Validation shared by nonlinear fitting and direct PNLS entry points."""

import math
from numbers import Integral, Real


def validate_pnls_controls(maxiter: int | None, tol: float) -> None:
    if maxiter is not None and (
        isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter <= 0
    ):
        raise ValueError("pnls_maxiter must be a positive integer or None")
    valid_tol = isinstance(tol, Real) and not isinstance(tol, bool)
    try:
        valid_tol = valid_tol and math.isfinite(tol) and tol > 0
    except OverflowError:
        valid_tol = False
    if not valid_tol:
        raise ValueError("pnls_tol must be positive and finite")
