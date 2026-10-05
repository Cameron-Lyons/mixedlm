"""Validation shared by fitting controls and direct PIRLS entry points."""

from mixedlm.utils.validation import _validate_inner_controls


def validate_pirls_controls(
    maxiter: int | None, tol: float, *, tol_name: str = "pirls_tol"
) -> None:
    _validate_inner_controls(maxiter, tol, maxiter_name="pirls_maxiter", tol_name=tol_name)
