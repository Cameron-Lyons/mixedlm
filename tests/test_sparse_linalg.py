from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
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

    @pytest.mark.parametrize(("size", "density"), [(8, 0.35), (31, 0.12), (96, 0.04)])
    def test_random_sparse_spd_matches_dense_reference(self, size, density):
        """Exercise sparse patterns and numerical values independently of one fixture."""
        rng = np.random.default_rng(10_000 + size)
        lower = sparse.random(
            size,
            size,
            density=density,
            format="csc",
            random_state=rng,
            data_rvs=rng.standard_normal,
        )
        lower = sparse.tril(lower, format="csc")
        lower.setdiag(rng.uniform(0.5, 1.5, size=size))
        matrix = (lower @ lower.T + sparse.eye(size, format="csc") * 0.25).tocsc()
        data = matrix.data.astype(np.float64)
        indices = matrix.indices.astype(np.int64)
        indptr = matrix.indptr.astype(np.int64)
        rhs = rng.standard_normal((size, 5))

        actual = sparse_cholesky_solve(data, indices, indptr, matrix.shape, rhs)
        expected = np.linalg.solve(matrix.toarray(), rhs)
        assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

        actual_logdet = sparse_cholesky_logdet(data, indices, indptr, matrix.shape)
        sign, expected_logdet = np.linalg.slogdet(matrix.toarray())
        assert sign == 1.0
        assert_allclose(actual_logdet, expected_logdet, rtol=2e-11, atol=2e-11)

        symbolic = SparseCholeskySymbolic(indices, indptr, size)
        numeric = symbolic.factor(data)
        assert_allclose(numeric.solve(rhs), expected, rtol=2e-11, atol=2e-11)
        assert_allclose(numeric.logdet(), expected_logdet, rtol=2e-11, atol=2e-11)

    def test_symbolic_cache_accepts_new_values_with_same_pattern(self):
        size = 40
        offdiag = np.full(size - 1, -0.2)
        first = sparse.diags((offdiag, np.full(size, 2.0), offdiag), (-1, 0, 1), format="csc")
        first.sum_duplicates()
        first.sort_indices()
        second = first.copy()
        second.data = first.data * np.linspace(0.8, 1.2, first.nnz)
        second = (second + second.T) * 0.5 + sparse.eye(size, format="csc")
        # Explicitly retain the original pattern while changing all numeric values.
        second = second.tocsc()
        second.sum_duplicates()
        second.sort_indices()
        assert np.array_equal(first.indices, second.indices)
        assert np.array_equal(first.indptr, second.indptr)

        symbolic = SparseCholeskySymbolic(
            first.indices.astype(np.int64), first.indptr.astype(np.int64), size
        )
        rhs = np.arange(1, size + 1, dtype=np.float64)[:, None]
        for matrix in (first, second):
            numeric = symbolic.factor(matrix.data)
            assert_allclose(
                numeric.solve(rhs),
                np.linalg.solve(matrix.toarray(), rhs),
                rtol=2e-11,
                atol=2e-11,
            )

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


def _ordering_system(layout, size):
    if layout == "banded":
        offdiag = np.full(size - 1, -1.0)
        matrix = sparse.diags((offdiag, np.full(size, 4.0), offdiag), (-1, 0, 1), format="csc")
    else:
        from tests.test_sparse_ordering import arrowhead_system

        matrix = arrowhead_system(size)[0].tocsc()
    return matrix, matrix.data, matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64)


class TestSparseCholeskyOrdering:
    @pytest.mark.parametrize("ordering", ["amd", "natural"])
    @pytest.mark.parametrize("hub", [0, 27])
    def test_one_shot_functions_accept_either_ordering(self, ordering, hub):
        from tests.test_sparse_ordering import arrowhead_system

        matrix, rhs, expected, logdet = arrowhead_system(61, hub)
        csc = matrix.tocsc()
        arguments = (csc.data, csc.indices.astype(np.int64), csc.indptr.astype(np.int64), csc.shape)

        actual = sparse_cholesky_solve(*arguments, rhs, ordering=ordering)

        assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        assert sparse_cholesky_logdet(*arguments, ordering=ordering) == pytest.approx(
            logdet, rel=2e-13
        )

    @pytest.mark.parametrize("ordering", ["AMD", "identity", ""])
    def test_one_shot_functions_reject_unsupported_orderings(self, ordering):
        _, data, indices, indptr, shape = sparse_arguments(np.eye(2))

        with pytest.raises(ValueError, match="ordering must be 'amd' or 'natural'"):
            sparse_cholesky_solve(data, indices, indptr, shape, np.ones((2, 1)), ordering=ordering)
        with pytest.raises(ValueError, match="ordering must be 'amd' or 'natural'"):
            sparse_cholesky_logdet(data, indices, indptr, shape, ordering=ordering)

    @pytest.mark.parametrize("layout", ["banded", "hub"])
    @pytest.mark.parametrize("n_rhs", [1, 3, 128])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_amd_solves_match_natural_order_for_many_right_hand_sides(self, layout, n_rhs, order):
        matrix, data, indices, indptr = _ordering_system(layout, 300)
        rhs = np.asarray(np.random.default_rng(n_rhs).standard_normal((300, n_rhs)), order=order)
        solutions = {}
        for ordering in ["amd", "natural"]:
            symbolic = SparseCholeskySymbolic(indices, indptr, 300, ordering=ordering)
            solutions[ordering] = symbolic.factor(data).solve(rhs)
            one_shot = sparse_cholesky_solve(
                data, indices, indptr, matrix.shape, rhs, ordering=ordering
            )
            assert_array_equal(one_shot, solutions[ordering])

        assert_allclose(solutions["amd"], solutions["natural"], rtol=1e-12, atol=1e-12)
        assert_allclose(matrix @ solutions["amd"], rhs, rtol=1e-12, atol=1e-12)
