from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lFormula, set_cov_type
from mixedlm.estimation.em_reml import em_reml_simple
from scipy import linalg


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
    residuals = y_adj - joint_design @ solution
    # Evaluate uncertainty in observation space, independently of the trace contraction.
    uncertainty = np.sum(weights * np.diag(joint_design @ joint_covariance @ joint_design.T))
    sigma2 = (np.sum(weights * residuals**2) + uncertainty) / n
    expected_theta = []
    prior_blocks = []
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
            np.fill_diagonal(covariance, variance)
            expected_theta.extend([np.sqrt(variance / sigma2), correlation])
        elif structure.correlated:
            factor = linalg.cholesky(covariance / sigma2, lower=True)
            expected_theta.extend(factor[np.tril_indices(q)])
        else:
            covariance = np.diag(np.diag(covariance))
            expected_theta.extend(np.sqrt(np.diag(covariance) / sigma2))
        prior_blocks.extend([covariance] * structure.n_levels)
        start += structure.n_levels * q
    original_Z = matrices.Z.copy()

    result = em_reml_simple(matrices, max_iter=1)

    # Evaluate the restricted likelihood directly in observation space at the
    # updated covariance. Its GLS solution must match the reported beta.
    V = sigma2 * np.diag(1 / weights) + Z @ linalg.block_diag(*prior_blocks) @ Z.T
    VinvX = linalg.solve(V, X, assume_a="pos")
    beta = linalg.solve(X.T @ VinvX, VinvX.T @ y_adj, assume_a="pos")
    residual = y_adj - X @ beta
    loglik = -0.5 * (
        (n - p) * np.log(2 * np.pi)
        + np.linalg.slogdet(V)[1]
        + np.linalg.slogdet(X.T @ VinvX)[1]
        + residual @ linalg.solve(V, residual, assume_a="pos")
    )
    np.testing.assert_allclose(result.beta, beta, atol=1e-12, rtol=1e-12)
    assert result.final_loglik == pytest.approx(loglik, abs=1e-10)
    np.testing.assert_allclose(result.theta, expected_theta, atol=1e-12, rtol=1e-12)
    assert result.sigma == pytest.approx(np.sqrt(sigma2), rel=1e-12)
    assert result.n_iter == 1
    np.testing.assert_array_equal(matrices.y, y)
    np.testing.assert_array_equal(matrices.weights, weights)
    np.testing.assert_array_equal(matrices.offset, offset)
    assert (original_Z != matrices.Z).nnz == 0


@pytest.mark.parametrize("use_native", [True, False])
def test_sparse_em_matches_balanced_random_intercept_solution(monkeypatch, use_native):
    from mixedlm.models import shared_utils

    rng = np.random.default_rng(20261119)
    groups = 350
    group_means = rng.normal(size=groups)
    within = rng.normal(scale=0.3, size=(groups, 4))
    within -= within.mean(axis=1, keepdims=True)
    y = group_means[:, None] + within
    data = pd.DataFrame({"y": y.ravel(), "g": np.repeat(np.arange(groups), y.shape[1])})
    matrices = lFormula("y ~ 1 + (1 | g)", data).matrices
    residual_variance = np.sum(within**2) / (groups * (y.shape[1] - 1))
    group_variance = np.var(group_means, ddof=1) - residual_variance / y.shape[1]
    monkeypatch.setattr(shared_utils, "_HAS_RUST", use_native and shared_utils._HAS_RUST)

    def reject_dense(_self):
        raise AssertionError("large EM systems should keep their precision sparse")

    monkeypatch.setattr(shared_utils._RandomEffectFactor, "cholesky", property(reject_dense))
    result = em_reml_simple(matrices, tol=1e-13)

    assert result.converged
    np.testing.assert_allclose(result.sigma**2, residual_variance, rtol=1e-6)
    np.testing.assert_allclose((result.theta * result.sigma) ** 2, group_variance, rtol=1e-6)


def test_em_likelihood_is_invariant_to_predictor_origin():
    from mixedlm import load_sleepstudy

    data = load_sleepstudy()
    original = em_reml_simple(lFormula("Reaction ~ Days + (1 | Subject)", data).matrices)
    data["Days"] += 1e7
    shifted = em_reml_simple(lFormula("Reaction ~ Days + (1 | Subject)", data).matrices)
    assert shifted.final_loglik == pytest.approx(original.final_loglik, abs=1e-7)
    np.testing.assert_allclose(shifted.theta, original.theta, rtol=1e-8)
    assert shifted.beta[1] == pytest.approx(original.beta[1], abs=1e-7)
