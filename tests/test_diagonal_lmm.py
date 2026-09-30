"""Diagonal likelihood systems agree with independent observation covariances."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation import reml
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure, build_model_matrices
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse


def diagonal_matrices(q=8, slope=False):
    rng = np.random.default_rng(774)
    active = max(1, q - 1)
    rows = np.arange(6 * active)
    group = rows % active
    x = rng.normal(size=len(rows))
    values = rng.uniform(-1.5, 1.5, len(rows)) if slope else np.ones(len(rows))
    values[::17] = 0
    design = sparse.coo_matrix((values, (rows, group)), shape=(len(rows), q)).tocsc()
    # Preserve duplicate entries, explicit zero rows, and an unused group.
    design = sparse.csc_matrix(
        (np.repeat(design.data / 2, 2), np.repeat(design.indices, 2), 2 * design.indptr),
        shape=design.shape,
    )
    offset = 0.15 * np.cos(rows)
    weights = 0.5 + (rows % 7) / 4
    y = 0.8 + 0.3 * x + offset + values * rng.normal(size=active)[group]
    y += rng.normal(scale=0.4 / np.sqrt(weights))
    structure = RandomEffectStructure(
        "g", ["x" if slope else "(Intercept)"], q, 1, True, {str(i): i for i in range(q)}
    )
    return ModelMatrices(
        y,
        np.column_stack((np.ones(len(rows)), x)),
        design,
        ["(Intercept)", "x"],
        [structure],
        len(rows),
        2,
        q,
        weights,
        offset,
    )


def observation_reference(matrices, covariance_factor, restricted):
    design = matrices.Z.toarray()
    transformed = design @ covariance_factor
    covariance = np.diag(1 / matrices.weights) + transformed @ transformed.T
    factor = linalg.cho_factor(covariance, lower=True)
    inverse_x = linalg.cho_solve(factor, matrices.X)
    information = matrices.X.T @ inverse_x
    response = matrices.y - matrices.offset
    beta = np.linalg.solve(information, inverse_x.T @ response)
    residual = response - matrices.X @ beta
    inverse_residual = linalg.cho_solve(factor, residual)
    random = covariance_factor @ transformed.T @ inverse_residual
    spherical = transformed.T @ inverse_residual
    conditional = residual - design @ random
    wrss = np.dot(matrices.weights * conditional, conditional)
    ussq = np.dot(spherical, spherical)
    pwrss = residual @ inverse_residual
    denominator = matrices.n_obs - matrices.n_fixed if restricted else matrices.n_obs
    sigma = np.sqrt(pwrss / denominator)
    logdet_cov = 2 * np.log(np.diag(factor[0])).sum()
    logdet_fixed = np.linalg.slogdet(information)[1]
    deviance = denominator * (1 + np.log(2 * np.pi * sigma**2)) + logdet_cov
    if restricted:
        deviance += logdet_fixed
    return dict(
        deviance=deviance,
        beta=beta,
        u=random,
        sigma=sigma,
        wrss=wrss,
        ussq=ussq,
        pwrss=pwrss,
        fixed_information=information,
        ldL2=logdet_cov + np.log(matrices.weights).sum(),
        ldRX2=logdet_fixed if restricted else 0,
    )


@pytest.mark.parametrize("q", [1, 8, 255, 256])
@pytest.mark.parametrize("slope", [False, True])
@pytest.mark.parametrize("theta", [0.0, 0.6, 2.0])
@pytest.mark.parametrize("restricted", [False, True])
def test_diagonal_system_matches_observation_likelihood(q, slope, theta, restricted):
    matrices = diagonal_matrices(q, slope)
    products = reml._LMMCrossproducts.from_matrices(matrices)
    assert products.ZtWZ_diagonal is not None
    expected = observation_reference(matrices, theta * np.eye(q), restricted)
    with patch.object(reml, "_build_lambda", side_effect=AssertionError("diagonal factor only")):
        actual = reml._profiled_deviance_core(
            np.array([theta]),
            matrices,
            restricted,
            crossproducts=products,
        )
    assert actual is not None
    for name, value in expected.items():
        assert_allclose(getattr(actual, name), value, rtol=2e-11, atol=2e-10)


@pytest.mark.parametrize("covariance", ["us", "diagonal", "cs", "ar1"])
@pytest.mark.parametrize("coupled", [False, True])
def test_orthogonal_slopes_use_actual_covariance_structure(covariance, coupled):
    group = np.repeat(np.arange(5), 4)
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 5)
    rng = np.random.default_rng(608)
    data = pd.DataFrame({"y": rng.normal(size=len(x)), "x": x, "g": group})
    text = "y ~ x + (x || g)" if covariance == "diagonal" else "y ~ x + (x | g)"
    formula = set_cov_type(text, covariance) if covariance in {"cs", "ar1"} else parse_formula(text)
    matrices = build_model_matrices(formula, data, weights=np.repeat([0.5, 1, 2, 3, 4], 4))
    rho = 0.3 if coupled else 0.0
    if covariance in {"cs", "ar1"}:
        theta = np.array([0.7, rho])
        block = 0.7 * np.array([[1, 0], [rho, np.sqrt(1 - rho**2)]])
        factor = np.kron(np.eye(5), block)
    elif covariance == "diagonal":
        theta = np.array([0.7, 0.4])
        factor = np.diag(np.tile([0.7, 0.4], 5))
    else:
        theta = np.array([0.7, rho, 0.4])
        factor = np.kron(np.eye(5), np.array([[0.7, 0], [rho, 0.4]]))
    products = reml._LMMCrossproducts.from_matrices(matrices)
    assert products.ZtWZ_diagonal is not None
    expected = observation_reference(matrices, factor, True)
    build = reml._build_lambda
    with patch.object(reml, "_build_lambda", wraps=build) as general:
        actual = reml._profiled_deviance_core(theta, matrices, crossproducts=products)
    assert general.call_count == int(coupled and covariance != "diagonal")
    assert actual is not None
    for name, value in expected.items():
        assert_allclose(getattr(actual, name), value, rtol=2e-11, atol=2e-11)


def test_coupled_design_retains_general_factorization():
    matrices = diagonal_matrices()
    design = matrices.Z.copy().tolil()
    design[1, 0] = 0.5
    matrices = replace(matrices, Z=design.tocsc())
    products = reml._LMMCrossproducts.from_matrices(matrices)
    assert products.ZtWZ_diagonal is None
    expected = observation_reference(matrices, 0.6 * np.eye(matrices.n_random), False)
    actual = reml._profiled_deviance_core(np.array([0.6]), matrices, False, crossproducts=products)
    assert actual is not None
    assert_allclose(actual.deviance, expected["deviance"], atol=1e-11)


@pytest.mark.parametrize("restricted", [False, True])
def test_multiple_diagonal_structures_without_fixed_coefficients(restricted):
    matrices = diagonal_matrices(9)
    structures = [
        RandomEffectStructure("first", ["a"], 3, 1, True, {}),
        RandomEffectStructure("second", ["b", "c"], 3, 2, False, {}),
    ]
    matrices = replace(
        matrices,
        X=np.empty((matrices.n_obs, 0)),
        n_fixed=0,
        fixed_names=[],
        random_structures=structures,
    )
    factor = np.diag(np.r_[np.full(3, 0.8), np.tile([0.5, 1.1], 3)])
    expected = observation_reference(matrices, factor, restricted)
    actual = reml._profiled_deviance_core(np.array([0.8, 0.5, 1.1]), matrices, restricted)
    assert actual is not None
    for name, value in expected.items():
        assert_allclose(getattr(actual, name), value, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("theta,weight", [(1e160, 1e-300), (1e-160, 1e300)])
def test_opposite_parameter_and_weight_scales_keep_finite_information(theta, weight):
    matrices = diagonal_matrices(1)
    matrices = replace(
        matrices,
        X=np.empty((matrices.n_obs, 0)),
        n_fixed=0,
        fixed_names=[],
        weights=np.full(matrices.n_obs, weight),
    )
    z = matrices.Z.toarray().ravel()
    response = matrices.y - matrices.offset
    information = (np.dot(matrices.weights * z, z) * theta) * theta
    spherical = (theta * np.dot(matrices.weights * z, response)) / (1 + information)
    random = theta * spherical
    residual = response - z * random
    pwrss = np.dot(matrices.weights * residual, residual) + spherical**2
    expected = (
        matrices.n_obs * (1 + np.log(2 * np.pi * pwrss / matrices.n_obs))
        + np.log1p(information)
        - np.log(matrices.weights).sum()
    )
    actual = reml._profiled_deviance_core(np.array([theta]), matrices, False)
    assert actual is not None
    assert_allclose(actual.deviance, expected, atol=1e-10)
    assert_allclose(actual.ldL2, np.log1p(information), rtol=1e-14, atol=0)
    assert_allclose(actual.u, [random], rtol=1e-14, atol=0)


def test_large_diagonal_system_does_not_build_or_factor_random_precision(monkeypatch):
    matrices = diagonal_matrices(4096, True)
    products = reml._LMMCrossproducts.from_matrices(matrices)
    original = linalg.cholesky

    def fixed_only(matrix, *args, **kwargs):
        assert matrix.shape == (matrices.n_fixed, matrices.n_fixed)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(linalg, "cholesky", fixed_only)
    with (
        patch.object(reml, "_build_lambda", side_effect=AssertionError("random matrix assembly")),
        patch.object(reml.sparse_linalg, "splu", side_effect=AssertionError("random sparse solve")),
    ):
        for theta in [0, 0.5, 2]:
            actual = reml._profiled_deviance_core(
                np.array([theta]), matrices, crossproducts=products
            )
            assert actual is not None and np.isfinite(actual.deviance)
            assert actual.u[-1] == 0


def test_tiny_variance_retains_log_determinant_information():
    matrices = diagonal_matrices(8)
    theta = 1e-9
    products = reml._LMMCrossproducts.from_matrices(matrices)
    information = theta**2 * products.ZtWZ.diagonal()
    assert_array_equal(1 + information, np.ones(matrices.n_random))
    actual = reml._profiled_deviance_core(np.array([theta]), matrices, crossproducts=products)
    assert actual is not None
    assert_allclose(actual.ldL2, np.log1p(information).sum(), rtol=1e-15, atol=0)
    assert actual.ldL2 > 0


@pytest.mark.parametrize("dense", [False, True])
def test_diagonal_detection_handles_storage_without_modifying_input(dense):
    matrix = sparse.csc_matrix(
        ([1.0, 2.0, 0.0, -2.0, 2.0, 4.0], [0, 0, 1, 0, 0, 1], [0, 3, 6]), shape=(2, 2)
    )
    source = matrix.toarray() if dense else matrix
    before = source.copy()
    assert_allclose(reml._diagonal_entries(source), [3, 4])
    assert_allclose(source if dense else source.toarray(), before if dense else before.toarray())
    if not dense:
        assert_array_equal(source.data, before.data)
        assert_array_equal(source.indices, before.indices)


def test_python_fits_and_profiles_reuse_diagonal_preparation():
    rng = np.random.default_rng(502)
    group = np.repeat(np.arange(8), 5)
    x = rng.normal(size=len(group))
    data = pd.DataFrame(
        {
            "y": 0.4 + 0.8 * x + rng.normal(size=8)[group] + rng.normal(size=len(x)),
            "x": x,
            "g": group,
        }
    )
    fitted = lmer("y ~ x + (1 | g)", data, REML=False, control=LmerControl(use_rust=False))
    with patch.object(reml, "_diagonal_entries", wraps=reml._diagonal_entries) as prepare:
        profile = fitted.profile("x", n_points=7)["x"]
    assert prepare.call_count == 1
    assert profile.ci_lower < profile.mle < profile.ci_upper
