from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lFormula, set_cov_type
from mixedlm.estimation.em_reml import (
    _add_random_precision,
    _m_step_update_sigma,
    em_reml_simple,
)
from mixedlm.matrices.design import RandomEffectStructure
from scipy import linalg


def _structure(n_terms, n_levels):
    return RandomEffectStructure(
        "group", [str(i) for i in range(n_terms)], n_levels, n_terms, True, {}
    )


@pytest.mark.parametrize("strided", [False, True])
def test_prior_precision_matches_dense_blocks(strided):
    rng = np.random.default_rng(91)
    parent = rng.normal(size=(10, 10))
    original = parent.copy()
    precision = parent[::2, ::2] if strided else parent[:5, :5]
    expected = precision.copy()
    covariance = np.array([[1.0, -0.4], [-0.4, 2.0]])
    covariance_copy = covariance.copy()
    expected += linalg.block_diag(np.linalg.inv(covariance), np.linalg.inv(covariance), [[4.0]])

    _add_random_precision(
        precision, [_structure(2, 2), _structure(1, 1)], [covariance, np.array([[0.25]])]
    )

    np.testing.assert_allclose(precision, expected, atol=1e-14)
    np.testing.assert_array_equal(covariance, covariance_copy)
    if strided:
        original[::2, ::2] = expected
    else:
        original[:5, :5] = expected
    np.testing.assert_allclose(parent, original, atol=1e-14)


@pytest.mark.parametrize("n_terms", [1, 2, 3])
def test_covariance_update_uses_diagonal_level_blocks_in_strided_view(n_terms):
    rng = np.random.default_rng(12)
    n_levels = 5
    size = n_terms * n_levels
    matrix = rng.normal(size=(size + 4, size + 4))
    full_covariance = matrix @ matrix.T + np.eye(size + 4)
    covariance = full_covariance[2:-2, 2:-2]
    assert not covariance.flags.c_contiguous
    original = full_covariance.copy()
    u = rng.normal(size=size)
    expected = (
        sum(
            covariance[i : i + n_terms, i : i + n_terms]
            + np.outer(u[i : i + n_terms], u[i : i + n_terms])
            for i in range(0, size, n_terms)
        )
        / n_levels
    )

    actual = _m_step_update_sigma(_structure(n_terms, n_levels), u, covariance, 1e-8)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(full_covariance, original)


@pytest.mark.parametrize("kind", ["intercept", "slope", "diagonal", "cs", "crossed"])
def test_weighted_em_step_matches_dense_joint_posterior(kind):
    rng = np.random.default_rng(92026)
    n = 96
    x = rng.normal(size=n)
    group = np.arange(n) % 8
    crossed = np.arange(n) % 6
    offset = np.linspace(-0.3, 0.6, n)
    weights = rng.uniform(0.5, 2.0, n)
    y = (
        2.0
        + 0.8 * x
        + rng.normal(size=8)[group]
        + rng.normal(size=6)[crossed]
        + offset
        + rng.normal(scale=0.4, size=n)
    )
    data = pd.DataFrame({"y": y, "x": x, "group": group, "crossed": crossed})
    formula = {
        "intercept": "y ~ x + (1 | group)",
        "slope": "y ~ x + (x | group)",
        "diagonal": "y ~ x + (x || group)",
        "cs": "y ~ x + (x | group)",
        "crossed": "y ~ x + (x | group) + (1 | crossed)",
    }[kind]
    if kind == "cs":
        formula = set_cov_type(formula, "cs")
    matrices = lFormula(formula, data, weights=weights, offset=offset).matrices
    X = matrices.X
    Z = matrices.Z.toarray()
    p = X.shape[1]
    y_adj = y - offset
    beta_ols = linalg.lstsq(np.sqrt(weights)[:, None] * X, np.sqrt(weights) * y_adj)[0]
    residual_variance = np.sum(weights * (y_adj - X @ beta_ols) ** 2) / n
    prior_variance = max(0.5 * residual_variance, 0.1)
    joint_design = np.column_stack([X, Z])
    joint_precision = joint_design.T @ (weights[:, None] * joint_design) / residual_variance
    joint_precision[p:, p:] += np.eye(Z.shape[1]) / prior_variance
    joint_covariance = np.linalg.inv(joint_precision)
    solution = joint_covariance @ (joint_design.T @ (weights * y_adj)) / residual_variance
    random_covariance = joint_covariance[p:, p:]
    residuals = y_adj - joint_design @ solution
    # Evaluate uncertainty in observation space, independently of the trace contraction.
    uncertainty = np.sum(weights * np.diag(Z @ random_covariance @ Z.T))
    sigma2 = (np.sum(weights * residuals**2) + uncertainty) / n
    expected_theta = []
    start = p
    for structure in matrices.random_structures:
        q = structure.n_terms
        covariance = (
            sum(
                joint_covariance[i : i + q, i : i + q]
                + np.outer(solution[i : i + q], solution[i : i + q])
                for i in range(start, start + structure.n_levels * q, q)
            )
            / structure.n_levels
        )
        if structure.cov_type == "cs":
            variance = np.diag(covariance).mean()
            correlation = covariance[0, 1] / variance
            expected_theta.extend([np.sqrt(variance / sigma2), correlation])
        elif structure.correlated:
            factor = linalg.cholesky(covariance / sigma2, lower=True)
            expected_theta.extend(factor[np.tril_indices(q)])
        else:
            expected_theta.extend(np.sqrt(np.diag(covariance) / sigma2))
        start += structure.n_levels * q
    original_Z = matrices.Z.copy()

    result = em_reml_simple(matrices, max_iter=1)

    np.testing.assert_allclose(result.beta, solution[:p], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result.theta, expected_theta, atol=1e-12, rtol=1e-12)
    assert result.sigma == pytest.approx(np.sqrt(sigma2), rel=1e-12)
    assert result.n_iter == 1
    np.testing.assert_array_equal(matrices.y, y)
    np.testing.assert_array_equal(matrices.weights, weights)
    np.testing.assert_array_equal(matrices.offset, offset)
    assert (original_Z != matrices.Z).nnz == 0
