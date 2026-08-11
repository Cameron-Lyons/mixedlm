from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

try:
    from mixedlm._rust import (
        SparseCholeskySymbolic,
        sparse_cholesky_logdet,
        sparse_cholesky_solve,
        update_cholesky_factor,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


def sparse_arguments(matrix):
    csc = sparse.csc_matrix(matrix)
    return (
        csc,
        csc.data,
        csc.indices.astype(np.int64),
        csc.indptr.astype(np.int64),
        csc.shape,
    )


class TestSparseCholeskySolve:
    def test_solves_many_right_hand_sides(self):
        matrix = np.diag(np.full(20, 4.0))
        matrix += np.diag(np.full(19, -1.0), 1)
        matrix += np.diag(np.full(19, -1.0), -1)
        csc, data, indices, indptr, shape = sparse_arguments(matrix)
        rhs = np.random.default_rng(42).normal(size=(20, 32))

        actual = sparse_cholesky_solve(data, indices, indptr, shape, rhs)

        assert_allclose(actual, np.linalg.solve(csc.toarray(), rhs), rtol=1e-12, atol=1e-12)

    def test_rejects_mismatched_right_hand_side(self):
        _, data, indices, indptr, shape = sparse_arguments(np.eye(2))

        with pytest.raises(ValueError, match="right-hand side has 3 rows, expected 2"):
            sparse_cholesky_solve(data, indices, indptr, shape, np.ones((3, 1)))

    def test_rejects_non_square_matrix(self):
        _, data, indices, indptr, shape = sparse_arguments(np.ones((2, 3)))

        with pytest.raises(ValueError, match="matrix must be square, got 2x3"):
            sparse_cholesky_solve(data, indices, indptr, shape, np.ones((2, 1)))

    def test_logdet_rejects_non_square_matrix(self):
        _, data, indices, indptr, shape = sparse_arguments(np.ones((2, 3)))

        with pytest.raises(ValueError, match="matrix must be square, got 2x3"):
            sparse_cholesky_logdet(data, indices, indptr, shape)

    def test_factor_update_rejects_non_square_matrix(self):
        _, data, indices, indptr, shape = sparse_arguments(np.ones((2, 3)))

        with pytest.raises(ValueError, match="matrix must be square, got 2x3"):
            update_cholesky_factor(data, indices, indptr, shape, np.array([1.0]))

    def test_cached_factor_rejects_mismatched_right_hand_side(self):
        _, data, indices, indptr, _ = sparse_arguments(np.eye(2))
        symbolic = SparseCholeskySymbolic(indices, indptr, 2)
        numeric = symbolic.factor(data)

        with pytest.raises(ValueError, match="right-hand side has 3 rows, expected 2"):
            numeric.solve(np.ones((3, 1)))
