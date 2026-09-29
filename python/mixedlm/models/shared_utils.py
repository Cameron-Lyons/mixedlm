from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from mixedlm.utils.dataframe import dataframe_length, get_column_numpy, get_columns

_MAX_QUADRATIC_FORM_ELEMENTS = 1_000_000
_SPARSE_PROJECTION_MIN_RANDOM = 256


class _RandomEffectFactor:
    """Reuse a precision solve, materializing its dense Cholesky only on demand."""

    def __init__(self, precision: sparse.spmatrix, *, jitter: float = 0.0) -> None:
        self.precision = sparse.csc_matrix(precision)
        self._dense_factor: NDArray[np.float64] | None = None
        self._sparse_factor: sparse_linalg.SuperLU | None = None
        try:
            self._factorize()
        except (linalg.LinAlgError, RuntimeError):
            if jitter == 0.0:
                raise
            self.precision = self.precision + jitter * sparse.eye(
                self.precision.shape[0], format="csc"
            )
            self._factorize()

    def _factorize(self) -> None:
        if self.precision.shape[0] >= _SPARSE_PROJECTION_MIN_RANDOM:
            self._sparse_factor = sparse_linalg.splu(self.precision)
        else:
            self._dense_factor = linalg.cholesky(self.precision.toarray(), lower=True)

    def __reduce__(self) -> tuple[type[_RandomEffectFactor], tuple[sparse.csc_matrix]]:
        # SuperLU objects cannot be pickled; rebuild from the effective precision.
        return type(self), (self.precision,)

    @cached_property
    def cholesky(self) -> NDArray[np.float64]:
        if self._dense_factor is not None:
            return self._dense_factor
        return linalg.cholesky(self.precision.toarray(), lower=True)

    @cached_property
    def logdet(self) -> float:
        if self._sparse_factor is not None:
            return float(np.sum(np.log(np.abs(self._sparse_factor.U.diagonal()))))
        return float(2.0 * np.sum(np.log(np.diag(self.cholesky))))

    def solve(self, rhs: NDArray[np.floating]) -> NDArray[np.float64]:
        if rhs.size == 0:
            return np.asarray(rhs, dtype=np.float64).copy()
        if self._sparse_factor is not None:
            return self._sparse_factor.solve(rhs)
        return linalg.cho_solve((self.cholesky, True), rhs)

    def quadratic_diagonal(self, design: sparse.spmatrix) -> NDArray[np.float64]:
        factor = self.solve if self._sparse_factor is not None else self.cholesky
        return sparse_quadratic_form_diagonal(design, factor)

    def crossproduct(self, rhs: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return B.T C^-1 B, using only a forward solve for dense factors."""
        if self._sparse_factor is not None:
            return rhs.T @ self.solve(rhs)
        whitened = linalg.solve_triangular(self.cholesky, rhs, lower=True)
        return whitened.T @ whitened

    def solve_with_crossproduct(
        self, rhs: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return C^-1 B and B.T C^-1 B, retaining the dense Cholesky arithmetic."""
        if self._sparse_factor is not None:
            solved = self.solve(rhs)
            return solved, rhs.T @ solved
        whitened = linalg.solve_triangular(self.cholesky, rhs, lower=True)
        solved = linalg.solve_triangular(self.cholesky.T, whitened, lower=False)
        return solved, whitened.T @ whitened


def sparse_quadratic_form_diagonal(
    design: sparse.spmatrix,
    factor: NDArray[np.floating] | Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> NDArray[np.float64]:
    """Compute diag(A C^-1 A.T) with bounded dense solve buffers.

    ``design`` is A. ``factor`` is either C's lower Cholesky factor or a
    callable computing C^-1 times its argument without modifying that argument.
    Each right-hand-side buffer holds at most one million elements, or one
    row of A if its width exceeds that limit.
    """
    design = design.tocsr()
    n_rows, width = design.shape
    if width == 0:
        return np.zeros(n_rows, dtype=np.float64)

    result = np.empty(n_rows, dtype=np.float64)
    chunk_size = max(1, _MAX_QUADRATIC_FORM_ELEMENTS // width)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        rhs = design[start:stop].toarray().T
        if callable(factor):
            solved = factor(rhs)
            result[start:stop] = np.einsum("ij,ij->j", rhs, solved)
        else:
            solved = linalg.solve_triangular(factor, rhs, lower=True, overwrite_b=True)
            result[start:stop] = np.einsum("ij,ij->j", solved, solved)
        del rhs, solved
    return result


def symmetric_inverse(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Invert a symmetric matrix with a positive-definite fast path."""
    try:
        factor = linalg.cholesky(matrix, lower=True)
        return linalg.cho_solve((factor, True), np.eye(matrix.shape[0]))
    except linalg.LinAlgError:
        return linalg.pinvh(matrix)


def is_singular_theta(theta: NDArray[np.floating], structures: list[Any], tol: float) -> bool:
    """Return whether any random-effect covariance block is rank deficient."""
    theta_idx = 0

    for struct in structures:
        q = struct.n_terms
        cov_type = getattr(struct, "cov_type", "us")

        if cov_type in ("cs", "ar1"):
            sigma_rel = theta[theta_idx]
            theta_idx += 2 if q > 1 else 1
            if abs(sigma_rel) < tol:
                return True
        elif struct.correlated:
            n_theta = q * (q + 1) // 2
            theta_block = theta[theta_idx : theta_idx + n_theta]
            theta_idx += n_theta

            diag_idx = 0
            for i in range(q):
                if abs(theta_block[diag_idx]) < tol:
                    return True
                diag_idx += i + 2
        else:
            theta_block = theta[theta_idx : theta_idx + q]
            theta_idx += q
            if np.any(np.abs(theta_block) < tol):
                return True

    return False


def resolve_optional_vector(
    data: Any,
    value: NDArray[np.floating] | str | None,
    name: str,
) -> NDArray[np.floating] | None:
    """Resolve optional vector-like inputs from array or data column name."""
    if value is None:
        return None

    if isinstance(value, str):
        if value not in get_columns(data):
            raise ValueError(f"{name} column '{value}' not found in data")
        return get_column_numpy(data, value, dtype=np.float64)

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.shape[0] != dataframe_length(data):
        raise ValueError(
            f"{name} length {arr.shape[0]} does not match data length {dataframe_length(data)}"
        )
    return arr
