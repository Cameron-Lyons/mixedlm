"""Sparse ordering preserves analytical solutions and avoids hub-induced fill."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose

from tests._sparse_systems import arrowhead_system, sparse_arguments

pytestmark = pytest.mark.installed_wheel


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
