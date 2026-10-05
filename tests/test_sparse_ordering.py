"""Sparse ordering preserves analytical solutions and avoids hub-induced fill."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose
from scipy import sparse


def arrowhead_system(size, hub=0):
    """Build a sparse SPD system with a closed-form Schur complement oracle."""
    leaves = np.delete(np.arange(size), hub)
    diagonal = np.linspace(1.2, 3.7, size)
    coupling = 0.09 * np.cos(leaves + 0.3)
    schur = 0.8
    diagonal[hub] = schur + np.sum(coupling**2 / diagonal[leaves])
    rows = np.concatenate((np.arange(size), leaves, np.full(size - 1, hub)))
    columns = np.concatenate((np.arange(size), np.full(size - 1, hub), leaves))
    values = np.concatenate((diagonal, coupling, coupling))
    matrix = sparse.csc_matrix((values, (rows, columns)), shape=(size, size))
    rng = np.random.default_rng(440 + hub)
    rhs = rng.normal(size=(size, 7))
    solution = np.empty_like(rhs)
    solution[hub] = (rhs[hub] - coupling @ (rhs[leaves] / diagonal[leaves, None])) / schur
    solution[leaves] = (rhs[leaves] - coupling[:, None] * solution[hub]) / diagonal[leaves, None]
    logdet = np.log(diagonal[leaves]).sum() + np.log(schur)
    return matrix, rhs, solution, logdet


def sparse_arguments(matrix, storage):
    matrix = sparse.tril(matrix, format="csc") if storage == "lower" else matrix
    if storage != "noncanonical":
        return matrix.data.copy(), matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64)
    # Valid raw CSC can contain duplicates and descending row indices. The
    # symbolic mapping must retain the numeric sources after canonicalization.
    values, rows, offsets = [], [], [0]
    for column in range(matrix.shape[1]):
        for entry in range(matrix.indptr[column + 1] - 1, matrix.indptr[column] - 1, -1):
            rows.extend([matrix.indices[entry]] * 2)
            values.extend([0.25 * matrix.data[entry], 0.75 * matrix.data[entry]])
        offsets.append(len(values))
    return np.asarray(values), np.asarray(rows, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


@pytest.mark.parametrize("ordering", ["amd", "natural"])
@pytest.mark.parametrize("storage", ["full", "lower", "noncanonical"])
@pytest.mark.parametrize("hub", [0, 27])
def test_ordering_matches_arrowhead_schur_complement(ordering, storage, hub):
    matrix, rhs, expected, logdet = arrowhead_system(61, hub)
    data, indices, offsets = sparse_arguments(matrix, storage)
    symbolic = _rust.SparseCholeskySymbolic(indices, offsets, len(rhs), ordering=ordering)
    factor = symbolic.factor(data)

    actual = factor.solve(rhs)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(matrix @ actual, rhs, rtol=2e-13, atol=2e-13)
    assert factor.logdet() == pytest.approx(logdet, rel=2e-13, abs=2e-13)
    assert_allclose(
        _rust.sparse_cholesky_solve(data, indices, offsets, matrix.shape, rhs),
        expected,
        rtol=2e-13,
        atol=2e-13,
    )
    assert _rust.sparse_cholesky_logdet(data, indices, offsets, matrix.shape) == pytest.approx(
        logdet, rel=2e-13, abs=2e-13
    )


def test_default_ordering_keeps_hub_factor_storage_linear():
    size = 512
    matrix, rhs, expected, _ = arrowhead_system(size)
    data, indices, offsets = sparse_arguments(matrix, "full")
    default = _rust.SparseCholeskySymbolic(indices, offsets, size)
    natural = _rust.SparseCholeskySymbolic(indices, offsets, size, ordering="natural")

    assert default.factor_nonzeros() <= 2 * size
    assert natural.factor_nonzeros() == size * (size + 1) // 2
    assert_allclose(default.factor(data).solve(rhs), expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("ordering", ["amd", "natural"])
def test_shared_symbolic_factors_and_solutions_remain_independent_in_threads(ordering):
    matrix, rhs, expected, logdet = arrowhead_system(83, 19)
    data, indices, offsets = sparse_arguments(matrix, "noncanonical")
    symbolic = _rust.SparseCholeskySymbolic(indices, offsets, len(rhs), ordering=ordering)
    scales = [0.25, 1.0, 1.7, 4.0] * 4

    def factor_and_solve(scale):
        factor = symbolic.factor(data * scale)
        return factor, factor.solve(rhs), factor.logdet()

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(factor_and_solve, scales))
    for scale, (factor, solution, determinant) in zip(scales, outcomes, strict=True):
        assert_allclose(solution, expected / scale, rtol=2e-13, atol=2e-13)
        assert determinant == pytest.approx(logdet + len(rhs) * np.log(scale), rel=2e-13)
        # A later factorization or returned result mutation cannot alter a factor.
        solution[:] = 0
        assert_allclose(factor.solve(rhs), expected / scale, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("ordering", ["AMD", "identity", "", "random"])
def test_ordering_rejects_unsupported_names(ordering):
    with pytest.raises(ValueError, match="ordering must be 'amd' or 'natural'"):
        _rust.SparseCholeskySymbolic(
            np.array([0], dtype=np.int64), np.array([0, 1], dtype=np.int64), 1, ordering=ordering
        )
