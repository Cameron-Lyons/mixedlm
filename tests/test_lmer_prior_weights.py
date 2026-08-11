from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation.reml import (
    _HAS_RUST,
    _build_lambda,
    _profiled_deviance_core,
    profiled_deviance_fast,
)
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose
from scipy import linalg


def _weighted_slope_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(124)
    n_groups = 12
    observations_per_group = 10
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.5, 1.5, observations_per_group), n_groups)
    intercepts = np.array([-2.0, -1.0, 0.5, 1.5, 2.5, -0.5, 1.0, -2.5, 0.0, 2.0, -1.5, 0.8])
    slopes = np.array([1.4, -0.8, 0.5, -1.3, 1.0, 0.1, -0.4, 1.2, -1.1, 0.8, -0.2, -1.5])
    y = 1.5 + 0.8 * x + intercepts[group] + slopes[group] * x + rng.normal(0.0, 0.12, len(group))
    weights = np.linspace(0.4, 2.0, len(group))
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})
    return data, weights


def _fit_weighted(
    data: pd.DataFrame,
    weights: np.ndarray,
    reml: bool,
):
    return lmer(
        "y ~ x + (x | group)",
        data,
        weights=weights,
        REML=reml,
        control=LmerControl(use_rust=False, em_init=False),
    )


@pytest.mark.parametrize("reml", [False, True])
def test_global_weight_rescaling_preserves_fitted_model(reml: bool) -> None:
    data, weights = _weighted_slope_data()
    scale = 25.0
    result = _fit_weighted(data, weights, reml)
    scaled = _fit_weighted(data, scale * weights, reml)

    assert result.converged
    assert scaled.converged
    assert_allclose(scaled.beta, result.beta, rtol=2e-5, atol=1e-7)
    assert_allclose(scaled.u, result.u, rtol=2e-5, atol=1e-6)
    assert_allclose(scaled.fitted(), result.fitted(), rtol=2e-5, atol=1e-6)
    assert_allclose(scaled.theta * np.sqrt(scale), result.theta, rtol=2e-5, atol=1e-6)
    assert scaled.sigma == pytest.approx(result.sigma * np.sqrt(scale), rel=2e-5)
    assert scaled.deviance == pytest.approx(result.deviance, abs=1e-6)
    assert scaled.logLik().value == pytest.approx(result.logLik().value, abs=5e-7)
    assert scaled.AIC() == pytest.approx(result.AIC(), abs=1e-6)
    assert_allclose(
        scaled.VarCorr().get_cov("group"),
        result.VarCorr().get_cov("group"),
        rtol=3e-5,
        atol=1e-7,
    )
    assert_allclose(scaled.vcov(), result.vcov(), rtol=3e-5, atol=1e-7)
    assert_allclose(scaled.hatvalues(), result.hatvalues(), rtol=3e-5, atol=1e-7)
    assert_allclose(
        scaled.residuals(type="pearson"),
        result.residuals(type="pearson"),
        rtol=3e-5,
        atol=1e-7,
    )

    original_condvar = result.ranef(condVar=True).condVar
    scaled_condvar = scaled.ranef(condVar=True).condVar
    assert original_condvar is not None
    assert scaled_condvar is not None
    for term in result.matrices.random_structures[0].term_names:
        assert_allclose(
            scaled_condvar["group"][term],
            original_condvar["group"][term],
            rtol=3e-5,
            atol=1e-7,
        )


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize(
    "use_rust",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _HAS_RUST, reason="Rust unavailable"))],
)
def test_profiled_deviance_is_normalized_for_weight_scale(reml: bool, use_rust: bool) -> None:
    data, weights = _weighted_slope_data()
    matrices = build_model_matrices(
        parse_formula("y ~ x + (x | group)"), data, weights=weights, na_action=None
    )
    theta = np.array([1.3, -0.2, 0.9])
    scale = 16.0
    scaled_matrices = replace(matrices, weights=scale * weights)

    deviance = profiled_deviance_fast(theta, matrices, REML=reml, use_rust=use_rust)
    scaled_deviance = profiled_deviance_fast(
        theta / np.sqrt(scale), scaled_matrices, REML=reml, use_rust=use_rust
    )

    assert scaled_deviance == pytest.approx(deviance, abs=1e-9)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize(
    "use_rust",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _HAS_RUST, reason="Rust unavailable"))],
)
def test_fixed_only_profiled_deviance_includes_weight_normalization(
    reml: bool, use_rust: bool
) -> None:
    data, weights = _weighted_slope_data()
    matrices = build_model_matrices(parse_formula("y ~ x"), data, weights=weights, na_action=None)
    scale = 9.0
    scaled_matrices = replace(matrices, weights=scale * weights)

    deviance = profiled_deviance_fast(np.array([]), matrices, REML=reml, use_rust=use_rust)
    scaled_deviance = profiled_deviance_fast(
        np.array([]), scaled_matrices, REML=reml, use_rust=use_rust
    )
    core = _profiled_deviance_core(np.array([]), matrices, REML=reml)
    scaled_core = _profiled_deviance_core(np.array([]), scaled_matrices, REML=reml)

    assert core is not None
    assert scaled_core is not None
    assert scaled_deviance == pytest.approx(deviance, abs=1e-9)
    assert_allclose(scaled_core.beta, core.beta, rtol=0, atol=1e-11)
    assert scaled_core.sigma == pytest.approx(np.sqrt(scale) * core.sigma, abs=1e-10)


def test_weighted_projection_matches_direct_matrix_identities() -> None:
    data, weights = _weighted_slope_data()
    result = _fit_weighted(data, weights, reml=True)
    matrices = result.matrices
    sqrt_weights = np.sqrt(weights)
    weighted_X = sqrt_weights[:, None] * matrices.X
    lambda_matrix = _build_lambda(result.theta, matrices.random_structures).toarray()
    B = sqrt_weights[:, None] * (matrices.Z @ lambda_matrix)
    V = np.eye(matrices.n_random) + B.T @ B
    L_V = linalg.cholesky(V, lower=True)
    V_inv_Bt = linalg.cho_solve((L_V, True), B.T)
    projected_X = weighted_X - B @ (V_inv_Bt @ weighted_X)
    information = weighted_X.T @ projected_X
    information_inv = linalg.inv(information)

    expected_vcov = result.sigma**2 * information_inv
    expected_rzx = linalg.solve_triangular(L_V, B.T @ weighted_X, lower=True)
    expected_cond_cov = (
        result.sigma**2 * lambda_matrix @ linalg.cho_solve((L_V, True), lambda_matrix.T)
    )
    expected_hat = np.einsum("ij,ji->i", B, V_inv_Bt) + np.einsum(
        "ij,ij->i", projected_X @ information_inv, projected_X
    )

    assert_allclose(result.vcov(), expected_vcov, rtol=0, atol=1e-10)
    assert_allclose(result.getME("RZX"), expected_rzx, rtol=0, atol=1e-10)
    RX = result.getME("RX")
    assert_allclose(RX.T @ RX, information, rtol=0, atol=1e-10)
    assert_allclose(result.hatvalues(), expected_hat, rtol=0, atol=1e-10)

    condvar = result.ranef(condVar=True).condVar
    assert condvar is not None
    expected_diagonal = np.diag(expected_cond_cov).reshape(
        matrices.random_structures[0].n_levels,
        matrices.random_structures[0].n_terms,
    )
    for term_index, term in enumerate(matrices.random_structures[0].term_names):
        assert_allclose(condvar["group"][term], expected_diagonal[:, term_index], atol=1e-10)


def test_weighted_residual_and_influence_diagnostics() -> None:
    data, weights = _weighted_slope_data()
    result = _fit_weighted(data, weights, reml=True)
    response_residual = result.residuals(type="response")
    pearson_residual = result.residuals(type="pearson")
    leverage = result.hatvalues()
    expected_pearson = np.sqrt(weights) * response_residual / result.sigma
    expected_cooks = (
        expected_pearson**2 / result.matrices.n_fixed * leverage / (1.0 - leverage) ** 2
    )

    assert_allclose(pearson_residual, expected_pearson, rtol=0, atol=1e-12)
    assert_allclose(result.cooks_distance(), expected_cooks, rtol=0, atol=1e-12)
    assert_allclose(
        result.influence()["std_resid"],
        expected_pearson / np.sqrt(1.0 - leverage),
        rtol=0,
        atol=1e-12,
    )


def test_in_sample_prediction_intervals_use_observation_variances() -> None:
    data, weights = _weighted_slope_data()
    result = _fit_weighted(data, weights, reml=True)
    confidence = result.predict(interval="confidence")
    prediction = result.predict(interval="prediction")
    z_crit = 1.959963984540054
    confidence_variance = ((confidence.upper - confidence.lower) / (2.0 * z_crit)) ** 2
    prediction_variance = ((prediction.upper - prediction.lower) / (2.0 * z_crit)) ** 2

    assert_allclose(
        prediction_variance - confidence_variance,
        result.sigma**2 / weights,
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("weights_factory", "message"),
    [
        (lambda n: np.ones((n, 1)), "one-dimensional"),
        (lambda n: np.ones(n - 1), "weights.*length"),
        (lambda n: np.where(np.arange(n) == 0, np.nan, 1.0), "finite values"),
        (lambda n: np.where(np.arange(n) == 0, np.inf, 1.0), "finite values"),
        (lambda n: np.where(np.arange(n) == 0, 0.0, 1.0), "strictly positive"),
        (lambda n: np.where(np.arange(n) == 0, -1.0, 1.0), "strictly positive"),
    ],
)
def test_invalid_prior_weights_fail_before_optimization(weights_factory, message: str) -> None:
    data, _ = _weighted_slope_data()

    with pytest.raises(ValueError, match=message):
        lmer(
            "y ~ x + (1 | group)",
            data,
            weights=weights_factory(len(data)),
            na_action=None,
        )


def test_weight_shape_is_validated_before_default_na_handling() -> None:
    data, _ = _weighted_slope_data()

    with pytest.raises(ValueError, match="one-dimensional"):
        lmer("y ~ x + (1 | group)", data, weights=np.ones((len(data), 1)))


@pytest.mark.skipif(not _HAS_RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(0.0, "strictly positive"), (-1.0, "strictly positive"), (np.nan, "finite values")],
)
def test_native_backend_rejects_invalid_prior_weights(bad_value: float, message: str) -> None:
    data, weights = _weighted_slope_data()
    matrices = build_model_matrices(
        parse_formula("y ~ x + (1 | group)"), data, weights=weights, na_action=None
    )
    invalid = weights.copy()
    invalid[0] = bad_value

    with pytest.raises(ValueError, match=message):
        profiled_deviance_fast(
            np.array([1.0]), replace(matrices, weights=invalid), REML=True, use_rust=True
        )
