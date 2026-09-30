"""Validation shared by fitting controls and direct PIRLS entry points."""

import math
from numbers import Integral, Real


def validate_pirls_controls(
    maxiter: int | None, tol: float, *, tol_name: str = "pirls_tol"
) -> None:
    if maxiter is not None and (
        isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter <= 0
    ):
        raise ValueError("pirls_maxiter must be a positive integer or None")
    valid_tol = isinstance(tol, Real) and not isinstance(tol, bool)
    try:
        valid_tol = valid_tol and math.isfinite(tol) and tol > 0
    except OverflowError:
        valid_tol = False
    if not valid_tol:
        raise ValueError(f"{tol_name} must be positive and finite")
