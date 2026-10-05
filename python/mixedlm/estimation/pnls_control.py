"""Validation shared by nonlinear fitting and direct PNLS entry points."""

from mixedlm.utils.validation import _validate_inner_controls


def validate_pnls_controls(maxiter: int | None, tol: float) -> None:
    _validate_inner_controls(maxiter, tol, maxiter_name="pnls_maxiter", tol_name="pnls_tol")
