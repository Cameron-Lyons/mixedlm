from __future__ import annotations

import numpy as np
import pytest
from mixedlm.estimation.reml import _build_lambda
from mixedlm.matrices.design import RandomEffectStructure
from mixedlm.utils.variance import getL
from scipy import linalg, sparse


def _structure(n_terms=3, n_levels=4, correlated=True, cov_type="us"):
    return RandomEffectStructure(
        grouping_factor="group",
        term_names=[f"term{i}" for i in range(n_terms)],
        n_levels=n_levels,
        n_terms=n_terms,
        correlated=correlated,
        level_map={},
        cov_type=cov_type,
    )


@pytest.mark.parametrize("n_levels", [0, 1, 5])
@pytest.mark.parametrize("cov_type", ["us", "cs", "ar1", "diagonal"])
@pytest.mark.parametrize("scale", [0.0, 1.5])
def test_repeated_factors_match_dense_reference(n_levels, cov_type, scale):
    structure = _structure(n_levels=n_levels, cov_type=cov_type)
    if cov_type == "us":
        factor = scale * np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.3, 0.4, 0.8]])
        theta = factor[np.tril_indices(3)]
    elif cov_type == "diagonal":
        structure.cov_type = "us"
        structure.correlated = False
        theta = scale * np.array([1.0, 0.0, 0.8])
        factor = np.diag(theta)
    else:
        # A structured covariance takes precedence over the double-bar flag.
        structure.correlated = False
        rho = -0.2
        correlation = (
            np.full((3, 3), rho)
            if cov_type == "cs"
            else rho ** np.abs(np.arange(3)[:, None] - np.arange(3))
        )
        np.fill_diagonal(correlation, 1.0)
        factor = scale * linalg.cholesky(correlation, lower=True)
        theta = np.array([scale, rho])

    original_theta = theta.copy()
    actual = _build_lambda(theta, [structure])
    expected = np.kron(np.eye(n_levels), factor)

    assert sparse.isspmatrix_csc(actual)
    assert actual.dtype == np.float64
    assert actual.has_canonical_format
    assert actual.nnz == n_levels * np.count_nonzero(factor)
    np.testing.assert_allclose(actual.toarray(), expected, atol=1e-15)
    np.testing.assert_allclose(getL(theta, [structure], sigma=2.5, as_blocks=True), [2.5 * factor])
    np.testing.assert_allclose(getL(theta, [structure]).toarray(), expected, atol=1e-15)
    np.testing.assert_array_equal(theta, original_theta)


@pytest.mark.parametrize("cov_type", ["us", "cs", "ar1"])
def test_intercept_only_factor_consumes_one_parameter(cov_type):
    structures = [_structure(n_terms=1, n_levels=3, cov_type=cov_type), _structure(n_terms=1)]
    actual = _build_lambda(np.array([0.4, 1.2]), structures)

    np.testing.assert_allclose(actual.toarray(), np.diag([0.4] * 3 + [1.2] * 4))


def test_mixed_structures_keep_parameter_and_column_order():
    structures = [
        _structure(n_terms=2, n_levels=2),
        _structure(n_terms=1, n_levels=0),
        _structure(n_terms=2, n_levels=3, correlated=False),
        _structure(n_terms=2, n_levels=2, cov_type="cs"),
    ]
    theta = np.array([1.0, -0.2, 0.7, 9.0, 0.0, 0.4, 1.5, 0.3])
    unstructured = np.array([[1.0, 0.0], [-0.2, 0.7]])
    diagonal = np.diag([0.0, 0.4])
    cs = 1.5 * linalg.cholesky([[1.0, 0.3], [0.3, 1.0]], lower=True)
    expected = linalg.block_diag(unstructured, unstructured, diagonal, diagonal, diagonal, cs, cs)

    actual = _build_lambda(theta, structures)

    assert actual.has_canonical_format
    np.testing.assert_allclose(actual.toarray(), expected)


def test_no_random_structures_return_empty_factor():
    actual = _build_lambda(np.array([]), [])

    assert actual.shape == (0, 0)
    assert actual.nnz == 0
    assert actual.has_canonical_format
    assert getL(np.array([]), [], as_blocks=True) == []


def test_large_factor_avoids_intermediate_sparse_block_matrices(monkeypatch):
    def reject_intermediate(*args, **kwargs):
        raise AssertionError("assemble the repeated factor directly")

    monkeypatch.setattr(sparse, "kron", reject_intermediate)
    monkeypatch.setattr(sparse, "block_diag", reject_intermediate)
    structure = _structure(n_levels=50_000)

    actual = _build_lambda(np.array([1.0, -0.2, 0.7, 0.0, 0.3, 0.4]), [structure])

    assert actual.shape == (150_000, 150_000)
    assert actual.nnz == 250_000
    assert actual.has_canonical_format
    np.testing.assert_allclose(actual.diagonal(), np.tile([1.0, 0.7, 0.4], 50_000))
