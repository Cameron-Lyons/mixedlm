from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from mixedlm.utils.dataframe import (
    dataframe_length,
    ensure_dataframe,
    get_column_numpy,
    get_columns,
)

try:
    from mixedlm._rust import SparseCholeskySymbolic

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

_MAX_QUADRATIC_FORM_ELEMENTS = 1_000_000
# Random-effect precisions of at least this order use a sparse Cholesky factor
# (native with a fill-reducing ordering, else SuperLU); smaller ones are dense.
_SPARSE_PROJECTION_MIN_RANDOM = 256


def _lower_triangle(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    lower = sparse.tril(matrix, format="csc")
    lower.sum_duplicates()
    lower.sort_indices()
    return lower


def _entry_keys(matrix: sparse.csc_matrix) -> NDArray[np.int64]:
    """Order canonical column-compressed entries by column, then row."""
    n = matrix.shape[0]
    columns = np.repeat(np.arange(matrix.shape[1], dtype=np.int64), np.diff(matrix.indptr))
    return columns * n + matrix.indices


class _SparseCholeskyPattern:
    """Native symbolic Cholesky analysis shared by precisions with one structure.

    ``pattern`` must include every entry any of those precisions can store,
    so it is built from structure rather than from values that may be zero.
    """

    def __init__(self, pattern: sparse.spmatrix) -> None:
        self.lower = _lower_triangle(pattern)
        self.keys = _entry_keys(self.lower)
        self.symbolic = SparseCholeskySymbolic(
            self.lower.indices.astype(np.int64),
            self.lower.indptr.astype(np.int64),
            self.lower.shape[0],
        )

    def __reduce__(self) -> tuple[type[_SparseCholeskyPattern], tuple[sparse.csc_matrix]]:
        # Native analyses cannot be pickled; repeat the analysis on load.
        return type(self), (self.lower,)

    def factor(self, precision: sparse.spmatrix) -> Any:
        lower = _lower_triangle(precision)
        keys = _entry_keys(lower)
        positions = np.searchsorted(self.keys, keys)
        if not np.array_equal(self.keys.take(positions, mode="clip"), keys):
            raise ValueError("precision has entries outside the analyzed pattern")
        values = np.zeros(len(self.keys))
        values[positions] = lower.data
        return self.symbolic.factor(values)


class _RandomEffectFactor:
    """Factor a random-effect precision once for repeated solves.

    The dense Cholesky factor of a sparse precision is materialized only on
    demand. ``pattern`` reuses a symbolic analysis of the precision structure.
    """

    def __init__(
        self,
        precision: sparse.spmatrix,
        *,
        jitter: float = 0.0,
        pattern: _SparseCholeskyPattern | None = None,
    ) -> None:
        # Small precisions are densified, so only sparse factors need column storage.
        self.precision = precision if sparse.issparse(precision) else sparse.csc_matrix(precision)
        self._pattern = pattern
        self._dense_factor: NDArray[np.float64] | None = None
        self._sparse_factor: Any = None
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
        if self.precision.shape[0] < _SPARSE_PROJECTION_MIN_RANDOM:
            self._dense_factor = linalg.cholesky(self.precision.toarray(), lower=True)
        elif _HAS_RUST:
            pattern = self._pattern or _SparseCholeskyPattern(self.precision)
            try:
                self._sparse_factor = pattern.factor(self.precision)
            except ValueError as error:
                raise linalg.LinAlgError(str(error)) from error
        else:
            self._sparse_factor = sparse_linalg.splu(self.precision.tocsc())

    def __reduce__(self) -> tuple[type[_RandomEffectFactor], tuple[sparse.spmatrix]]:
        # Sparse factors cannot be pickled; rebuild from the effective precision.
        return type(self), (self.precision,)

    @cached_property
    def cholesky(self) -> NDArray[np.float64]:
        if self._dense_factor is not None:
            return self._dense_factor
        return linalg.cholesky(self.precision.toarray(), lower=True)

    @cached_property
    def logdet(self) -> float:
        if isinstance(self._sparse_factor, sparse_linalg.SuperLU):
            return float(np.sum(np.log(np.abs(self._sparse_factor.U.diagonal()))))
        if self._sparse_factor is not None:
            return float(self._sparse_factor.logdet())
        return float(2.0 * np.sum(np.log(np.diag(self.cholesky))))

    def solve(self, rhs: NDArray[np.floating]) -> NDArray[np.float64]:
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.size == 0:
            return rhs.copy()
        if self._sparse_factor is None:
            return linalg.cho_solve((self.cholesky, True), rhs)
        if isinstance(self._sparse_factor, sparse_linalg.SuperLU):
            return self._sparse_factor.solve(rhs)
        return self._sparse_factor.solve(rhs.reshape(len(rhs), -1)).reshape(rhs.shape)

    @cached_property
    def _dense_inverse(self) -> NDArray[np.float64]:
        return linalg.cho_solve((self.cholesky, True), np.eye(self.precision.shape[0]))

    def inverse_entries(
        self, rows: NDArray[np.integer], columns: NDArray[np.integer]
    ) -> NDArray[np.float64]:
        """Return the paired entries C^-1[rows, columns].

        Sparse factors solve for the requested unit columns in bounded batches.
        """
        if self._sparse_factor is None:
            return self._dense_inverse[rows, columns]
        q = self.precision.shape[0]
        values = np.empty(len(rows), dtype=np.float64)
        order = np.argsort(columns, kind="stable")
        needed, first = np.unique(columns[order], return_index=True)
        first = np.append(first, len(order))
        batch = max(1, _MAX_QUADRATIC_FORM_ELEMENTS // q)
        for start in range(0, len(needed), batch):
            block = needed[start : start + batch]
            unit = np.zeros((q, len(block)), dtype=np.float64)
            unit[block, np.arange(len(block))] = 1.0
            solved = self.solve(unit)
            selected = order[first[start] : first[start + len(block)]]
            values[selected] = solved[rows[selected], np.searchsorted(block, columns[selected])]
        return values

    def quadratic_diagonal(self, design: sparse.spmatrix) -> NDArray[np.float64]:
        """Compute diag(A C^-1 A.T) from C^-1 at the column pairs sharing a row of A."""
        design = sparse.csr_matrix(design, dtype=np.float64)
        design.sum_duplicates()
        n_rows, q = design.shape
        result = np.zeros(n_rows, dtype=np.float64)
        if design.nnz == 0:
            return result

        if self._sparse_factor is None:
            inverse = self._dense_inverse

            def lookup(rows: NDArray[np.intp], columns: NDArray[np.intp]) -> NDArray[np.float64]:
                return inverse[rows, columns]

        else:
            # Only the entries of C^-1 on the pattern of A.T A are needed.
            structure = sparse.csr_matrix(
                (np.ones(design.nnz), design.indices, design.indptr), shape=design.shape
            )
            gram = (structure.T @ structure).tocsr()
            gram.sort_indices()
            gram_rows = np.repeat(np.arange(q), np.diff(gram.indptr))
            keys = gram_rows * q + gram.indices
            entries = self.inverse_entries(gram_rows, gram.indices)

            def lookup(rows: NDArray[np.intp], columns: NDArray[np.intp]) -> NDArray[np.float64]:
                return entries[np.searchsorted(keys, rows.astype(np.int64) * q + columns)]

        widths = np.diff(design.indptr)
        rows_per_chunk = max(1, _MAX_QUADRATIC_FORM_ELEMENTS // int(widths.max()) ** 2)
        for start in range(0, n_rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_rows)
            # Enumerate every ordered pair of stored entries within each row.
            width = widths[start:stop]
            pairs = width**2
            span = np.repeat(width, pairs)
            local = np.arange(pairs.sum()) - np.repeat(np.cumsum(pairs) - pairs, pairs)
            first = np.repeat(design.indptr[start:stop], pairs) + local // span
            second = first - local // span + local % span
            products = design.data[first] * design.data[second]
            products *= lookup(design.indices[first], design.indices[second])
            result[start:stop] = np.bincount(
                np.repeat(np.arange(stop - start), pairs), weights=products, minlength=stop - start
            )
        return result

    def crossproduct(
        self, rhs: NDArray[np.floating], other: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        """Return B.T C^-1 B, followed by the columns of B.T C^-1 D if ``other`` is given.

        Dense factors use only forward solves; sparse factors solve [B D] at once.
        """
        if self._sparse_factor is not None:
            return rhs.T @ self.solve(rhs if other is None else np.column_stack((rhs, other)))
        whitened = linalg.solve_triangular(self.cholesky, rhs, lower=True)
        gram = whitened.T @ whitened
        if other is None:
            return gram
        whitened_other = linalg.solve_triangular(self.cholesky, other, lower=True)
        return np.column_stack((gram, whitened.T @ whitened_other))

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


def dense_quadratic_form_diagonal(
    design: NDArray[np.floating], covariance: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Compute diag(A C A.T) with one bounded projection buffer."""
    n_rows, width = design.shape
    if width == 0:
        return np.zeros(n_rows, dtype=np.float64)
    result = np.empty(n_rows, dtype=np.float64)
    chunk_size = max(1, _MAX_QUADRATIC_FORM_ELEMENTS // width)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        chunk = design[start:stop]
        projected = chunk @ covariance
        result[start:stop] = np.einsum("ij,ij->i", projected, chunk)
        del projected
    return result


def resolve_prediction_vector(
    data: Any,
    value: ArrayLike | str | None,
    *,
    name: str,
    default: float,
) -> NDArray[np.float64]:
    """Validate prediction rows, retaining scalar inputs as broadcast views."""
    data = ensure_dataframe(data)
    n_rows = dataframe_length(data)
    if value is None:
        # Defaults are internal scalar constants; no user array needs conversion.
        scalar = np.asarray(default, dtype=np.float64)
        if not np.isfinite(scalar):
            raise ValueError(f"Prediction {name} must contain only finite values.")
        return np.broadcast_to(scalar, (n_rows,))
    raw: ArrayLike
    if isinstance(value, str):
        if value not in get_columns(data):
            raise ValueError(f"New data is missing {name} column '{value}'.")
        raw = get_column_numpy(data, value)
    else:
        raw = value
    if np.ma.is_masked(raw):
        raise ValueError(f"Prediction {name} must not contain masked values.")
    try:
        values = np.asarray(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Prediction {name} must contain numeric values.") from None
    if (
        values.dtype.kind in "mMV"
        or np.iscomplexobj(values)
        or (values.dtype.kind == "O" and any(np.iscomplexobj(item) for item in values.flat))
    ):
        raise ValueError(f"Prediction {name} must contain real numeric values.")
    if values.dtype.kind == "O" and any(np.ma.is_masked(item) for item in values.flat):
        raise ValueError(f"Prediction {name} must not contain masked values.")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Prediction {name} must contain numeric values.") from None
    if values.ndim > 1:
        raise ValueError(f"Prediction {name} must be a scalar or one-dimensional array.")
    if values.ndim == 1 and len(values) != n_rows:
        raise ValueError(
            f"Prediction {name} has length {len(values)}; expected {n_rows} for new data."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Prediction {name} must contain only finite values.")
    if values.ndim == 0:
        return np.broadcast_to(values, (n_rows,))
    return values


def sparse_covariance_factor_diagonal(
    design: sparse.spmatrix, factor: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Compute diag(A L L.T A.T) without forming the covariance or a full dense A."""
    design = design.tocsr()
    n_rows, width = design.shape
    if width == 0:
        return np.zeros(n_rows, dtype=np.float64)
    result = np.empty(n_rows, dtype=np.float64)
    chunk_size = max(1, _MAX_QUADRATIC_FORM_ELEMENTS // width)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        projected = np.asarray(design[start:stop] @ factor)
        result[start:stop] = np.einsum("ij,ij->i", projected, projected)
        del projected
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
