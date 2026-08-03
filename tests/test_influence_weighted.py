from __future__ import annotations

import numpy as np
import pandas as pd
from mixedlm import glmer, lmer
from mixedlm.datasets import load_cbpp
from mixedlm.diagnostics import (
    cooks_distance,
    dfbeta,
    dfbetas,
    dffits,
    influence,
    influence_summary,
    influential_obs,
    leverage,
)
from mixedlm.estimation.reml import _build_lambda, _profiled_deviance_core
from mixedlm.families import Binomial
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import LmerControl
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose
from scipy import linalg, sparse


def _weighted_lmm_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(823)
    n_groups = 9
    observations_per_group = 7
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.4, 1.4, observations_per_group), n_groups)
    intercepts = np.repeat(
        np.array([-1.2, 0.3, 1.4, -0.6, 0.8, -1.0, 0.5, 1.1, -0.2]),
        observations_per_group,
    )
    slopes = np.repeat(
        np.array([0.7, -0.5, 0.15, 0.55, -0.2, -0.65, 0.4, -0.1, 0.25]),
        observations_per_group,
    )
    base_y = 1.5 + 0.75 * x + intercepts + slopes * x + rng.normal(0.0, 0.16, len(x))
    offset = 0.2 * np.cos(np.arange(len(x)) / 5.0) + 0.08 * x
    weights = np.exp(np.linspace(-0.7, 0.8, len(x)))
    base = pd.DataFrame({"y": base_y, "x": x, "group": group.astype(str)})
    shifted = base.assign(y=base_y + offset)
    return base, shifted, weights, offset


def _fit_lmm(
    data: pd.DataFrame,
    weights: np.ndarray,
    offset: np.ndarray | None = None,
) -> LmerResult:
    return lmer(
        "y ~ x + (x | group)",
        data,
        weights=weights,
        offset=offset,
        control=LmerControl(use_rust=False, em_init=False),
    )


def _fixed_only_result(
    data: pd.DataFrame,
    weights: np.ndarray,
    offset: np.ndarray,
) -> LmerResult:
    formula = parse_formula("y ~ x")
    matrices = build_model_matrices(
        formula,
        data,
        weights=weights,
        offset=offset,
        na_action=None,
    )
    core = _profiled_deviance_core(np.empty(0), matrices, REML=False)
    assert core is not None
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=core.beta,
        sigma=core.sigma,
        u=core.u,
        deviance=core.deviance,
        REML=False,
        converged=True,
        n_iter=0,
    )


def test_weighted_fixed_only_dfbeta_matches_exact_case_deletion() -> None:
    x = np.linspace(-1.5, 1.5, 14)
    offset = 0.15 * np.sin(np.arange(len(x)))
    weights = np.linspace(0.4, 2.2, len(x))
    y = (
        2.0
        + 1.3 * x
        + offset
        + np.array(
            [0.1, -0.2, 0.05, 0.15, -0.1, 0.3, -0.25, 0.05, 0.1, -0.1, 0.2, -0.05, 0.1, -0.15]
        )
    )
    data = pd.DataFrame({"y": y, "x": x})
    result = _fixed_only_result(data, weights, offset)
    actual = influence(result).dfbeta

    expected = np.empty_like(actual)
    for row in range(len(data)):
        keep = np.arange(len(data)) != row
        deleted = _fixed_only_result(
            data.loc[keep].reset_index(drop=True),
            weights[keep],
            offset[keep],
        )
        expected[row] = result.beta - deleted.beta

    assert_allclose(actual, expected, rtol=0, atol=2e-14)


def test_lmm_influence_matches_model_and_preserves_equivalent_fits() -> None:
    base, shifted, weights, offset = _weighted_lmm_data()
    scale = 25.0
    baseline = _fit_lmm(base, weights)
    shifted_fit = _fit_lmm(shifted, weights, offset)
    scaled_fit = _fit_lmm(shifted, scale * weights, offset)
    baseline_influence = influence(baseline)

    assert_allclose(baseline_influence.hat_values, baseline.hatvalues(), rtol=0, atol=0)
    assert_allclose(baseline_influence.cooks_distance, baseline.cooks_distance(), atol=1e-14)
    assert_allclose(
        baseline_influence.residuals,
        np.sqrt(weights) * baseline.residuals(type="response", na_expand=False),
        rtol=0,
        atol=1e-14,
    )

    for equivalent in (shifted_fit, scaled_fit):
        equivalent_influence = influence(equivalent)
        assert_allclose(equivalent_influence.hat_values, baseline_influence.hat_values, atol=2e-5)
        assert_allclose(
            equivalent_influence.residuals / equivalent_influence.sigma,
            baseline_influence.residuals / baseline_influence.sigma,
            atol=2e-5,
        )
        assert_allclose(equivalent_influence.dfbeta, baseline_influence.dfbeta, atol=2e-5)
        assert_allclose(equivalent_influence.dfbetas, baseline_influence.dfbetas, atol=2e-5)
        assert_allclose(
            equivalent_influence.cooks_distance,
            baseline_influence.cooks_distance,
            atol=2e-5,
        )
        assert_allclose(equivalent_influence.dffits, baseline_influence.dffits, atol=2e-5)


def test_influence_helpers_accept_fitted_model() -> None:
    base, _, weights, _ = _weighted_lmm_data()
    result = _fit_lmm(base, weights)
    diagnostics = influence(result)

    assert_allclose(dfbeta(result), diagnostics.dfbeta)
    assert_allclose(dfbetas(result), diagnostics.dfbetas)
    assert_allclose(cooks_distance(result), diagnostics.cooks_distance)
    assert_allclose(dffits(result), diagnostics.dffits)
    assert_allclose(leverage(result), diagnostics.leverage)
    assert influence_summary(result).shape[0] == result.matrices.n_obs
    assert influential_obs(result).ndim == 1


def test_weighted_glmm_projection_matches_direct_identities() -> None:
    data = load_cbpp()
    data["y"] = data["incidence"] / data["size"]
    prior_weights = data["size"].to_numpy(dtype=np.float64)
    result = glmer(
        "y ~ period + (1 | herd)",
        data,
        family=Binomial(),
        weights=prior_weights,
    )

    matrices = result.matrices
    mu = np.clip(result.fitted(type="response", na_expand=False), 1e-10, 1 - 1e-10)
    working_weights = result.family.weights(mu) * prior_weights
    sqrt_weights = np.sqrt(working_weights)
    weighted_X = sqrt_weights[:, None] * matrices.X
    weighted_Z = matrices.Z.multiply(sqrt_weights[:, None]).tocsc()
    lambda_matrix = _build_lambda(result.theta, matrices.random_structures)
    C = weighted_Z.T @ weighted_Z + lambda_matrix.T @ lambda_matrix
    L_C = linalg.cholesky(C.toarray(), lower=True)
    expected_RZX = linalg.solve_triangular(L_C, weighted_Z.T @ weighted_X, lower=True)
    information = weighted_X.T @ weighted_X - expected_RZX.T @ expected_RZX
    information = (information + information.T) / 2.0
    information_inv = linalg.inv(information)
    weighted_ZLambda = (weighted_Z @ lambda_matrix).toarray()
    C_inv_ZLambda_t = linalg.cho_solve((L_C, True), weighted_ZLambda.T)
    expected_hat = np.einsum("ij,ij->i", weighted_X @ information_inv, weighted_X)
    expected_hat += np.einsum("ij,ji->i", weighted_ZLambda, C_inv_ZLambda_t)

    assert result._working_projection is result._working_projection
    assert_allclose(result._working_projection.working_weights, working_weights, atol=1e-12)
    assert_allclose(result.vcov(), information_inv, atol=1e-11)
    assert_allclose(result.getME("RZX"), expected_RZX, atol=1e-11)
    assert_allclose(result.getME("RX").T @ result.getME("RX"), information, atol=1e-11)
    assert_allclose(result.hatvalues(), np.clip(expected_hat, 0, 1 - 1e-10), atol=1e-11)

    first_hat = result.hatvalues()
    first_hat[0] = np.nan
    assert np.isfinite(result.hatvalues()[0])


def test_weighted_glmm_influence_uses_final_fit_and_prior_weights() -> None:
    data = load_cbpp()
    data["y"] = data["incidence"] / data["size"]
    prior_weights = data["size"].to_numpy(dtype=np.float64)
    result = glmer(
        "y ~ period + (1 | herd)",
        data,
        family=Binomial(),
        weights=prior_weights,
    )
    diagnostics = influence(result)
    mu = result.fitted(type="response", na_expand=False)
    response_residual = result.matrices.y - mu
    expected_pearson = np.sqrt(prior_weights) * response_residual
    expected_pearson /= np.sqrt(result.family.variance(mu))

    assert_allclose(result.residuals(type="pearson", na_expand=False), expected_pearson)
    assert_allclose(diagnostics.residuals, expected_pearson)
    assert_allclose(diagnostics.hat_values, result.hatvalues(), rtol=0, atol=0)
    assert_allclose(diagnostics.cooks_distance, result.cooks_distance(), atol=1e-14)
    assert_allclose(cooks_distance(result), result.cooks_distance(), atol=1e-14)

    conditional = result.ranef(condVar=True).condVar
    assert conditional is not None
    projection = result._working_projection
    assert projection.lambda_matrix is not None
    ZtWZ = projection.weighted_Z.T @ projection.weighted_Z
    V = projection.lambda_matrix.T @ ZtWZ @ projection.lambda_matrix
    V = V + sparse.eye(result.matrices.n_random, format="csc")
    expected_condvar = projection.lambda_matrix @ linalg.solve(
        V.toarray(),
        projection.lambda_matrix.T.toarray(),
        assume_a="pos",
    )
    expected_diagonal = np.diag(expected_condvar).reshape(
        result.matrices.random_structures[0].n_levels,
        result.matrices.random_structures[0].n_terms,
    )
    assert_allclose(conditional["herd"]["(Intercept)"], expected_diagonal[:, 0], atol=1e-11)


def test_weighted_glmm_diagnostics_match_integer_row_replication() -> None:
    rng = np.random.default_rng(22)
    groups = np.repeat(np.arange(8), 6)
    x = np.tile(np.linspace(-1.2, 1.2, 6), 8)
    group_effects = np.repeat(np.array([-0.8, 0.3, 1.0, -0.4, 0.6, -0.7, 0.2, 0.9]), 6)
    eta = -0.4 + 0.8 * x + group_effects
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    integer_weights = np.tile(np.array([1, 2, 3, 1, 2, 3]), 8)
    data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})

    weighted = glmer(
        "y ~ x + (1 | group)",
        data,
        family=Binomial(),
        weights=integer_weights,
    )
    repeated_rows = np.repeat(np.arange(len(data)), integer_weights)
    replicated = glmer(
        "y ~ x + (1 | group)",
        data.iloc[repeated_rows].reset_index(drop=True),
        family=Binomial(),
    )

    assert weighted.converged
    assert replicated.converged
    assert_allclose(weighted.beta, replicated.beta, rtol=0, atol=2e-11)
    assert_allclose(weighted.theta, replicated.theta, rtol=0, atol=2e-11)
    assert_allclose(weighted.vcov(), replicated.vcov(), rtol=0, atol=3e-12)

    replicated_hat = replicated.hatvalues()
    row_starts = np.cumsum(np.r_[0, integer_weights[:-1]])
    aggregated_hat = np.array(
        [
            replicated_hat[start : start + count].sum()
            for start, count in zip(row_starts, integer_weights, strict=True)
        ]
    )
    assert_allclose(weighted.hatvalues(), aggregated_hat, rtol=0, atol=3e-12)

    weighted_pearson = weighted.residuals(type="pearson", na_expand=False)
    replicated_pearson = replicated.residuals(type="pearson", na_expand=False)
    aggregated_pearson_square = np.array(
        [
            np.square(replicated_pearson[start : start + count]).sum()
            for start, count in zip(row_starts, integer_weights, strict=True)
        ]
    )
    assert_allclose(np.square(weighted_pearson), aggregated_pearson_square, atol=3e-12)
