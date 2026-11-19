from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

rust = pytest.importorskip("mixedlm._rust")


def make_solver(matrix, cached):
    matrix = sparse.csc_matrix(matrix)
    indices = matrix.indices.astype(np.int64)
    indptr = matrix.indptr.astype(np.int64)
    if cached:
        symbolic = rust.SparseCholeskySymbolic(indices, indptr, matrix.shape[0])
        return symbolic.factor(matrix.data).solve
    return lambda rhs: rust.sparse_cholesky_solve(matrix.data, indices, indptr, matrix.shape, rhs)


def make_rhs(rng, n, n_rhs, layout):
    values = rng.standard_normal((n, n_rhs))
    if layout == "fortran":
        return np.asfortranarray(values)
    if layout == "strided":
        storage = np.zeros((n * 2, n_rhs * 2))
        storage[::2, ::2] = values
        return storage[::2, ::2]
    if layout == "reversed":
        return values[::-1, ::-1]
    if layout == "broadcast":
        return np.broadcast_to(rng.standard_normal((n, 1)), (n, n_rhs))
    if layout == "readonly":
        values.flags.writeable = False
    return values


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("n_rhs", [0, 1, 3, 4, 5, 16])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed", "broadcast", "readonly"])
def test_sparse_solve_preserves_inputs_and_column_order(cached, n_rhs, layout):
    rng = np.random.default_rng(42)
    n = 13
    lower = rng.standard_normal((n, n))
    matrix = lower @ lower.T + np.eye(n)
    rhs = make_rhs(rng, n, n_rhs, layout)
    original = rhs.copy()
    solve = make_solver(matrix, cached)

    actual = solve(rhs)

    assert actual.shape == rhs.shape
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, rhs)
    np.testing.assert_array_equal(rhs, original)
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-12, atol=2e-12)
    # Returned arrays are independent of the factor and subsequent results.
    actual[...] = 0
    np.testing.assert_allclose(solve(rhs), np.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("n_rhs", [1, 4, 9])
def test_sparse_solve_with_no_rows(cached, n_rhs):
    solve = make_solver(sparse.csc_matrix((0, 0)), cached)
    actual = solve(np.empty((0, n_rhs)))
    assert actual.shape == (0, n_rhs)


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("shape", [(2, 0), (2, 5), (4, 5)])
def test_sparse_solve_validates_rows_even_without_columns(cached, shape):
    solve = make_solver(np.eye(3), cached)
    with pytest.raises(ValueError, match=f"right-hand side has {shape[0]} rows, expected 3"):
        solve(np.zeros(shape))


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_sparse_solve_accepts_nested_sequences(cached):
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = make_solver(matrix, cached)(rhs.tolist())
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_sparse_solve_retains_array_dtype_validation(cached, dtype):
    with pytest.raises(TypeError):
        make_solver(np.eye(2), cached)(np.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("ordering", ["natural", "amd"])
def test_inverse_subset_on_fill_pattern(ordering):
    # Eliminating the hub in natural order fills the first component.
    # The isolated final node supplies a pair outside either fill pattern.
    n = 13
    dense = np.eye(n) * 3.0
    dense[0, 1:-1] = dense[1:-1, 0] = np.linspace(-0.6, 0.5, n - 2)
    lower = sparse.tril(dense, format="csc")
    cache = rust.SparseCholeskySymbolic(
        lower.indices.astype(np.int64), lower.indptr.astype(np.int64), n, ordering=ordering
    )
    factor = cache.factor(lower.data)
    rows, cols = np.nonzero(dense)
    actual = factor._inverse_subset(rows, cols)
    inverse = np.linalg.inv(dense)
    np.testing.assert_allclose(actual, inverse[rows, cols], atol=1e-14)
    if ordering == "natural":
        np.testing.assert_allclose(factor._inverse_subset([1, 2], [2, 1]), inverse[[1, 2], [2, 1]])
    assert factor._inverse_subset([0], [n - 1]) is None
    assert factor._inverse_subset([], []).size == 0
    for i, j in (([0], []), ([n], [0]), ([0], [-1])):
        with pytest.raises(ValueError):
            factor._inverse_subset(i, j)
