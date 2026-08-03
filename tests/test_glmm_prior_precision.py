from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import laplace as laplace_module
from mixedlm.estimation.laplace import (
    GLMMOptimizer,
    _build_lambda,
    _get_lambda_cached,
    adaptive_gh_deviance,
    clear_lambda_cache,
    laplace_deviance,
    pirls,
)
from mixedlm.families import Binomial, Gaussian
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.matrices.design import ModelMatrices, build_model_matrices
from mixedlm.models.glmer import GlmerResult
from numpy.testing import assert_allclose
from scipy import integrate, sparse, stats


def _gaussian_random_slope_matrices() -> ModelMatrices:
    group = np.repeat(["a", "b", "c", "d"], 4)
    x = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), 4)
    group_shift = np.repeat(np.array([-0.8, 0.3, 0.9, -0.2]), 4)
    group_slope = np.repeat(np.array([0.4, -0.1, 0.25, -0.35]), 4)
    offset = np.linspace(-0.2, 0.2, len(x))
    y = 1.2 + 0.7 * x + group_shift + group_slope * x + offset
    data = pd.DataFrame({"y": y, "x": x, "group": group})
    weights = np.linspace(0.7, 2.2, len(data))
    return build_model_matrices(
        parse_formula("y ~ x + (x | group)"),
        data,
        weights=weights,
        offset=offset,
    )


def _direct_gaussian_system(
    matrices: ModelMatrices, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lambda = _build_lambda(theta, matrices.random_structures).toarray()
    A = matrices.Z @ Lambda
    A = A.toarray() if sparse.issparse(A) else np.asarray(A)
    X = matrices.X
    W = matrices.weights
    design = np.column_stack((X, A))
    penalty = np.zeros((design.shape[1], design.shape[1]))
    penalty[X.shape[1] :, X.shape[1] :] = np.eye(A.shape[1])
    information = design.T @ (W[:, None] * design) + penalty
    rhs = design.T @ (W * (matrices.y - matrices.offset))
    solution = np.linalg.solve(information, rhs)
    beta = solution[: X.shape[1]]
    spherical = solution[X.shape[1] :]
    random_effects = Lambda @ spherical
    return beta, spherical, random_effects, design, information


def test_pirls_uses_spherical_prior_for_correlated_random_slopes() -> None:
    matrices = _gaussian_random_slope_matrices()
    family = Gaussian()
    theta = np.array([0.75, 0.2, 0.45])

    beta, random_effects, deviance, converged = pirls(matrices, family, theta)
    expected_beta, spherical, expected_random, _, _ = _direct_gaussian_system(matrices, theta)

    assert converged
    assert_allclose(beta, expected_beta, rtol=1e-11, atol=1e-11)
    assert_allclose(random_effects, expected_random, rtol=1e-11, atol=1e-11)

    mu = matrices.X @ expected_beta + matrices.Z @ expected_random + matrices.offset
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
    expected_deviance = np.sum(family.deviance_resids(matrices.y, mu, matrices.weights))
    expected_deviance += spherical @ spherical
    assert deviance == pytest.approx(expected_deviance, rel=1e-12, abs=1e-12)


def test_laplace_adjustment_matches_spherical_hessian() -> None:
    matrices = _gaussian_random_slope_matrices()
    family = Gaussian()
    theta = np.array([0.75, 0.2, 0.45])
    beta, spherical, random_effects, design, information = _direct_gaussian_system(matrices, theta)

    deviance, actual_beta, actual_random = laplace_deviance(theta, matrices, family)
    mu = matrices.X @ beta + matrices.Z @ random_effects + matrices.offset
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
    conditional = np.sum(family.deviance_resids(matrices.y, mu, matrices.weights))
    conditional += spherical @ spherical
    q = matrices.n_random
    random_hessian = information[-q:, -q:]
    _, logdet = np.linalg.slogdet(random_hessian)

    assert_allclose(actual_beta, beta, rtol=1e-11, atol=1e-11)
    assert_allclose(actual_random, random_effects, rtol=1e-11, atol=1e-11)
    assert deviance == pytest.approx(conditional + logdet, rel=1e-11, abs=1e-11)
    assert design.shape[1] == matrices.n_fixed + matrices.n_random


def test_zero_variance_is_a_stable_fixed_effect_boundary() -> None:
    matrices = _gaussian_random_slope_matrices()
    theta = np.zeros(3)

    beta, random_effects, deviance, converged = pirls(matrices, Gaussian(), theta)
    X = matrices.X
    W = matrices.weights
    expected_beta = np.linalg.solve(
        X.T @ (W[:, None] * X), X.T @ (W * (matrices.y - matrices.offset))
    )

    assert converged
    assert_allclose(beta, expected_beta, rtol=1e-11, atol=1e-11)
    assert_allclose(random_effects, 0.0, rtol=0.0, atol=0.0)
    assert np.isfinite(deviance)


def test_result_projection_matches_full_joint_information() -> None:
    matrices = _gaussian_random_slope_matrices()
    family = Gaussian()
    theta = np.array([0.75, 0.2, 0.45])
    beta, _, random_effects, design, information = _direct_gaussian_system(matrices, theta)
    result = GlmerResult(
        formula=parse_formula("y ~ x + (x | group)"),
        matrices=matrices,
        family=family,
        theta=theta,
        beta=beta,
        u=random_effects,
        deviance=0.0,
        converged=True,
        n_iter=1,
        nAGQ=1,
    )

    joint_covariance = np.linalg.inv(information)
    p = matrices.n_fixed
    assert_allclose(result.vcov(), joint_covariance[:p, :p], rtol=1e-11, atol=1e-11)

    expected_hat = matrices.weights * np.sum((design @ joint_covariance) * design, axis=1)
    assert_allclose(result.hatvalues(), expected_hat, rtol=1e-10, atol=1e-10)

    projection = result._working_projection
    assert_allclose(
        result.getME("RX").T @ result.getME("RX"),
        projection.fixed_information,
        rtol=1e-11,
        atol=1e-11,
    )
    assert_allclose(result.getME("RZX"), projection.RZX, rtol=0.0, atol=0.0)

    Lambda = projection.Lambda.toarray()
    conditional_random = (
        Lambda @ np.linalg.inv(projection.random_cholesky @ projection.random_cholesky.T) @ Lambda.T
    )
    condvar = result._compute_condVar()["group"]
    diagonal = np.diag(conditional_random).reshape(4, 2)
    assert_allclose(condvar["(Intercept)"], diagonal[:, 0], rtol=1e-11, atol=1e-11)
    assert_allclose(condvar["x"], diagonal[:, 1], rtol=1e-11, atol=1e-11)


def _binomial_random_intercept_matrices() -> ModelMatrices:
    group = np.repeat(["a", "b", "c"], 5)
    x = np.tile(np.linspace(-1.0, 1.0, 5), 3)
    eta = -0.4 + 0.8 * x + np.repeat([-0.7, 0.1, 0.9], 5)
    y = (np.arange(len(x)) % 4 < (1.0 / (1.0 + np.exp(-eta))) * 4).astype(float)
    data = pd.DataFrame({"y": y, "x": x, "group": group})
    return build_model_matrices(parse_formula("y ~ x + (1 | group)"), data)


def test_adaptive_quadrature_matches_direct_normalized_integrals() -> None:
    matrices = _binomial_random_intercept_matrices()
    family = Binomial()
    theta = np.array([0.65])
    deviance, beta, _ = adaptive_gh_deviance(theta, matrices, family, nAGQ=25)
    Lambda = _build_lambda(theta, matrices.random_structures).toarray()
    eta_fixed = matrices.X @ beta + matrices.offset

    log_integral = 0.0
    for group_index in range(3):
        rows = matrices.Z[:, group_index].toarray().ravel() != 0

        def integrand(
            value: float,
            group_index: int = group_index,
            rows: np.ndarray = rows,
        ) -> float:
            spherical = np.zeros(3)
            spherical[group_index] = value
            mu = family.link.inverse(eta_fixed + matrices.Z @ (Lambda @ spherical))
            conditional = np.sum(
                family.deviance_resids(matrices.y[rows], mu[rows], matrices.weights[rows])
            )
            return float(np.exp(-0.5 * conditional) * stats.norm.pdf(value))

        integral, error = integrate.quad(integrand, -9.0, 9.0, epsabs=1e-12, epsrel=1e-12)
        assert error < 1e-9
        log_integral += np.log(integral)

    assert deviance == pytest.approx(-2.0 * log_integral, rel=2e-8, abs=2e-8)


def test_native_and_python_paths_agree_away_from_unit_scale() -> None:
    native = pytest.importorskip("mixedlm._rust")
    matrices = _binomial_random_intercept_matrices()
    family = Binomial()
    theta = np.array([0.65])
    z = matrices.Z.tocsc()
    args = (
        matrices.y,
        matrices.X,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        matrices.weights,
        matrices.offset,
        theta,
        [3],
        [1],
        [True],
        "binomial",
        "logit",
    )

    py_beta, py_random, py_deviance, py_converged = pirls(matrices, family, theta)
    rust_beta, rust_random, rust_deviance, rust_converged = native.pirls(*args)
    assert rust_converged == py_converged
    assert_allclose(rust_beta, py_beta, rtol=1e-11, atol=1e-11)
    assert_allclose(rust_random, py_random, rtol=1e-11, atol=1e-11)
    assert rust_deviance == pytest.approx(py_deviance, rel=1e-11, abs=1e-11)

    py_laplace = laplace_deviance(theta, matrices, family)
    rust_laplace = native.laplace_deviance(*args)
    assert rust_laplace[0] == pytest.approx(py_laplace[0], rel=1e-11, abs=1e-11)
    assert_allclose(rust_laplace[1], py_laplace[1], rtol=1e-11, atol=1e-11)
    assert_allclose(rust_laplace[2], py_laplace[2], rtol=1e-11, atol=1e-11)

    py_agh = adaptive_gh_deviance(theta, matrices, family, nAGQ=9)
    rust_agh = native.adaptive_gh_deviance(*args, 9)
    assert rust_agh[0] == pytest.approx(py_agh[0], rel=1e-10, abs=1e-10)
    assert_allclose(rust_agh[1], py_agh[1], rtol=1e-11, atol=1e-11)
    assert_allclose(rust_agh[2], py_agh[2], rtol=1e-11, atol=1e-11)


def test_native_correlated_random_slopes_match_python() -> None:
    native = pytest.importorskip("mixedlm._rust")
    rng = np.random.default_rng(42)
    group = np.repeat(["a", "b", "c", "d"], 10)
    x = np.tile(np.linspace(-1.5, 1.5, 10), 4)
    group_shift = np.repeat(np.array([-0.8, 0.3, 0.9, -0.2]), 10)
    group_slope = np.repeat(np.array([0.4, -0.1, 0.25, -0.35]), 10)
    probability = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x + group_shift + group_slope * x)))
    data = pd.DataFrame({"y": rng.binomial(1, probability).astype(float), "x": x, "group": group})
    matrices = build_model_matrices(parse_formula("y ~ x + (x | group)"), data)
    theta = np.array([0.75, 0.2, 0.45])
    z = matrices.Z.tocsc()
    args = (
        matrices.y,
        matrices.X,
        z.data,
        z.indices.astype(np.int64),
        z.indptr.astype(np.int64),
        z.shape,
        matrices.weights,
        matrices.offset,
        theta,
        [4],
        [2],
        [True],
        "binomial",
        "logit",
    )

    py_beta, py_random, py_deviance, py_converged = pirls(matrices, Binomial(), theta)
    rust_beta, rust_random, rust_deviance, rust_converged = native.pirls(*args)
    assert rust_converged == py_converged
    assert_allclose(rust_beta, py_beta, rtol=1e-11, atol=1e-11)
    assert_allclose(rust_random, py_random, rtol=1e-11, atol=1e-11)
    assert rust_deviance == pytest.approx(py_deviance, rel=1e-11, abs=1e-11)


def test_optimizer_objective_is_repeatable() -> None:
    optimizer = GLMMOptimizer(_binomial_random_intercept_matrices(), Binomial())
    theta_a = np.array([0.35])
    theta_b = np.array([1.1])
    first = optimizer.objective(theta_a)
    optimizer.objective(theta_b)
    second = optimizer.objective(theta_a)
    assert second == pytest.approx(first, rel=0.0, abs=1e-13)


def test_covariance_factor_cache_distinguishes_structured_models() -> None:
    data = pd.DataFrame(
        {
            "y": np.linspace(0.1, 0.9, 12),
            "x": np.tile([-1.0, 0.0, 1.0], 4),
            "z": np.tile([0.5, -1.0, 0.5], 4),
            "group": np.repeat(["a", "b", "c", "d"], 3),
        }
    )
    cs = build_model_matrices(set_cov_type("y ~ x + z + (x + z | group)", "cs"), data)
    ar1 = build_model_matrices(set_cov_type("y ~ x + z + (x + z | group)", "ar1"), data)
    theta = np.array([0.7, 0.35])

    clear_lambda_cache()
    lambda_cs = _get_lambda_cached(theta, cs.random_structures).toarray()
    lambda_ar1 = _get_lambda_cached(theta, ar1.random_structures).toarray()
    clear_lambda_cache()

    assert not np.allclose(lambda_cs, lambda_ar1)
    assert_allclose(lambda_cs, _build_lambda(theta, cs.random_structures).toarray())
    assert_allclose(lambda_ar1, _build_lambda(theta, ar1.random_structures).toarray())


def test_structured_covariance_avoids_unsupported_native_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame(
        {
            "y": np.linspace(0.1, 0.9, 12),
            "x": np.tile([-1.0, 0.0, 1.0], 4),
            "z": np.tile([0.5, -1.0, 0.5], 4),
            "group": np.repeat(["a", "b", "c", "d"], 3),
        }
    )
    matrices = build_model_matrices(set_cov_type("y ~ x + z + (x + z | group)", "ar1"), data)

    def unsupported(*args: object, **kwargs: object) -> tuple[float, np.ndarray, np.ndarray]:
        raise AssertionError("unsupported native covariance layout was dispatched")

    monkeypatch.setattr(laplace_module, "_laplace_deviance_rust", unsupported)
    deviance, beta, random_effects = laplace_module.laplace_deviance_fast(
        np.array([0.7, 0.35]), matrices, Gaussian()
    )
    assert np.isfinite(deviance)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(random_effects))
