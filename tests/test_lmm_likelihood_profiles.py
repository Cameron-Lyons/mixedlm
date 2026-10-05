"""LMM profiles optimize the full ML likelihood under coefficient constraints."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer, slice2D
from mixedlm.estimation import reml
from mixedlm.formula.parser import set_cov_type
from mixedlm.inference import lmm_profile
from mixedlm.inference.profile import profile_lmer
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, optimize, stats


def data_fixture():
    rng = np.random.default_rng(672)
    group = np.repeat(np.arange(7), [4, 6, 5, 7, 3, 8, 5])
    x = rng.normal(size=len(group)) + 0.15 * group
    z = rng.normal(size=len(group))
    offset = 0.2 * np.cos(np.arange(len(group)))
    weights = np.geomspace(0.5, 2.0, len(group))
    effects = rng.normal(scale=[0.8, 0.3, 0.25], size=(7, 3))
    y = 1.1 + 0.5 * x - 0.3 * z + offset
    y += effects[group, 0] + effects[group, 1] * x + effects[group, 2] * z
    y += rng.normal(scale=0.4 / np.sqrt(weights))
    return pd.DataFrame(dict(y=y, x=x, z=z, g=group)), weights, offset


def fitted_model(kind="intercept", reml=False):
    data, weights, offset = data_fixture()
    formula = {
        "fixed": "y ~ x + z",
        "intercept": "y ~ x + z + (1 | g)",
        "correlated": "y ~ x + z + (x | g)",
        "cs": "y ~ x + z + (x + z | g)",
        "ar1": "y ~ x + z + (x + z | g)",
    }[kind]
    if kind in {"cs", "ar1"}:
        formula = set_cov_type(formula, kind)
    return lmer(formula, data, weights=weights, offset=offset, REML=reml)


def marginal_deviance(result, theta, fixed):
    """Independent observation-covariance calculation with analytic beta/sigma."""
    matrices = result.matrices
    blocks = []
    position = 0
    for structure in matrices.random_structures:
        width = structure.n_terms
        if structure.cov_type in {"cs", "ar1"}:
            scale, correlation = theta[position : position + 2]
            position += 2
            if structure.cov_type == "cs":
                covariance = scale**2 * ((1 - correlation) * np.eye(width) + correlation)
            else:
                distances = np.abs(np.arange(width)[:, None] - np.arange(width))
                covariance = scale**2 * correlation**distances
        else:
            factor = np.zeros((width, width))
            count = width * (width + 1) // 2
            factor[np.tril_indices(width)] = theta[position : position + count]
            position += count
            covariance = factor @ factor.T
        blocks.append(np.kron(np.eye(structure.n_levels), covariance))
    covariance = np.diag(1 / matrices.weights)
    if blocks:
        design = matrices.Z.toarray()
        covariance += design @ linalg.block_diag(*blocks) @ design.T
    factor = linalg.cho_factor(covariance, lower=True)
    keep = [i for i in range(matrices.n_fixed) if i not in fixed]
    design = matrices.X[:, keep]
    response = matrices.y - matrices.offset
    for i, value in fixed.items():
        response = response - value * matrices.X[:, i]
    if keep:
        inverse_design = linalg.cho_solve(factor, design)
        beta = np.linalg.solve(design.T @ inverse_design, inverse_design.T @ response)
        residual = response - design @ beta
    else:
        residual = response
    rss = residual @ linalg.cho_solve(factor, residual)
    n = matrices.n_obs
    return n * (1 + np.log(2 * np.pi * rss / n)) + 2 * np.log(np.diag(factor[0])).sum()


def intercept_reference(result, fixed):
    fit = optimize.minimize_scalar(
        lambda theta: marginal_deviance(result, np.array([theta]), fixed),
        bounds=(0, 40),
        method="bounded",
        options={"xatol": 1e-10},
    )
    assert fit.success and fit.x < 39
    return min(fit.fun, marginal_deviance(result, np.zeros(1), fixed))


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("level", [0.9, 0.99])
def test_fixed_only_profile_matches_exact_weighted_gaussian_likelihood(reml, level):
    result = fitted_model("fixed", reml)
    matrices = result.matrices
    information = matrices.X.T @ (matrices.weights[:, None] * matrices.X)
    beta = np.linalg.solve(
        information, matrices.X.T @ (matrices.weights * (matrices.y - matrices.offset))
    )
    residual = matrices.y - matrices.offset - matrices.X @ beta
    rss = np.dot(matrices.weights * residual, residual)
    index = matrices.fixed_names.index("x")
    half_width = np.sqrt(
        rss
        * np.linalg.inv(information)[index, index]
        * np.expm1(stats.chi2.isf(1 - level, 1) / matrices.n_obs)
    )
    profile = profile_lmer(result, "x", n_points=7, level=level)["x"]
    assert_allclose(
        [profile.ci_lower, profile.ci_upper],
        beta[index] + half_width * np.array([-1, 1]),
        atol=2e-9,
    )
    assert_allclose(
        profile.zeta[[0, -1]],
        [-np.sqrt(stats.chi2.isf(1 - level, 1)), np.sqrt(stats.chi2.isf(1 - level, 1))],
        atol=2e-8,
    )
    assert profile.values[3] == profile.mle
    assert profile.zeta[3] == 0


@pytest.mark.parametrize("reml", [False, True])
def test_random_intercept_limits_match_independent_marginal_optimization(reml):
    result = fitted_model(reml=reml)
    before = result.theta.copy(), result.beta.copy(), result.sigma, result.matrices.offset.copy()
    profile = profile_lmer(result, "x", n_points=5)["x"]
    minimum = intercept_reference(result, {})
    index = result.matrices.fixed_names.index("x")
    cutoff = stats.chi2.isf(0.05, 1)
    for endpoint in [profile.ci_lower, profile.ci_upper]:
        assert_allclose(intercept_reference(result, {index: endpoint}) - minimum, cutoff, atol=2e-7)
    for value, zeta in zip(profile.values, profile.zeta, strict=True):
        assert_allclose(intercept_reference(result, {index: value}) - minimum, zeta**2, atol=3e-7)
    assert_array_equal(result.theta, before[0])
    assert_array_equal(result.beta, before[1])
    assert result.sigma == before[2]
    assert_array_equal(result.matrices.offset, before[3])
    assert reml == result.REML


@pytest.mark.parametrize("kind", ["correlated", "cs", "ar1"])
@pytest.mark.parametrize("indices", [(1,), (0, 1)])
def test_constrained_covariance_fits_match_dense_marginal_likelihood(kind, indices):
    result = fitted_model(kind)
    likelihood = lmm_profile._LMMProfileLikelihood(result.matrices)
    optimum = likelihood.fit(result.theta)
    values = tuple(float(result.beta[i] + 0.5) for i in indices)
    constrained = likelihood.fit(optimum.theta, indices, values)
    fixed = dict(zip(indices, values, strict=True))
    expected = optimize.minimize(
        lambda theta: marginal_deviance(result, theta, fixed),
        optimum.theta,
        method="Nelder-Mead",
        bounds=likelihood.bounds,
        options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-10},
    )
    assert expected.success
    assert_allclose(constrained.deviance, expected.fun, atol=2e-7)
    assert_allclose(marginal_deviance(result, constrained.theta, fixed), expected.fun, atol=2e-7)
    assert np.max(np.abs(constrained.theta - optimum.theta)) > 1e-3


@pytest.mark.parametrize("jobs", [1, 2])
def test_two_parameter_profile_matches_independent_constrained_likelihood(jobs):
    result = fitted_model()
    surface = slice2D(result, "(Intercept)", "x", n_points=3, n_jobs=jobs, profile_covariance=True)
    assert surface.profile_covariance
    minimum = intercept_reference(result, {})
    for i, first in enumerate(surface.values1):
        for j, second in enumerate(surface.values2):
            expected = intercept_reference(result, {0: first, 1: second}) - minimum
            assert_allclose(surface.zeta[i, j] ** 2, expected, atol=2e-7)
    assert surface.zeta[1, 1] == 0
    assert surface.values1[1] == surface.mle1
    assert surface.values2[1] == surface.mle2
    assert not slice2D(result, "(Intercept)", "x", n_points=3).profile_covariance


@pytest.mark.parametrize("kind", ["intercept", "ar1"])
def test_prepared_products_and_fitted_points_are_reused_and_grid_does_not_change_limits(kind):
    result = fitted_model(kind)
    native = reml._HAS_RUST and kind != "ar1"
    with (
        patch.object(
            reml._RustMatrixCache, "from_matrices", wraps=reml._RustMatrixCache.from_matrices
        ) as native_designs,
        patch.object(
            reml._LMMCrossproducts, "from_matrices", wraps=reml._LMMCrossproducts.from_matrices
        ) as python_products,
    ):
        likelihood, optimum, _ = lmm_profile._reference(result)
        parameter = lmm_profile._LMMParameterProfile(likelihood, 1, optimum)
        value = parameter.mle + 0.2
        first = parameter.deviance(value)
        with patch.object(
            likelihood, "fit", side_effect=AssertionError("cached point must not refit")
        ):
            assert parameter.deviance(value) == first
        parameter.deviance(value + 0.1)
        other = lmm_profile._LMMParameterProfile(likelihood, 2, optimum)
        other.deviance(other.mle + 0.1)
    if native:
        # One design per set of free coefficients serves all of its held values.
        # Only the latest is retained, so memory does not grow with the number
        # of profiled coefficients.
        assert (native_designs.call_count, python_products.call_count) == (3, 0)
        assert likelihood._design is not None and likelihood._design[0] == (0, 1)
    else:
        # The Python likelihood slices one set of weighted products.
        assert (native_designs.call_count, python_products.call_count) == (0, 1)
        assert likelihood._design is None
    short = result.confint("x", method="profile")["x"]
    dense = result.profile("x", n_points=15)["x"]
    assert_allclose(short, [dense.ci_lower, dense.ci_upper], atol=1e-10)


@pytest.mark.parametrize("value", [True, 0, -1, 2, 3.0, np.nan])
def test_invalid_plot_sizes_are_rejected(value):
    with pytest.raises(ValueError, match="n_points"):
        profile_lmer(fitted_model(), n_points=value)


@pytest.mark.parametrize(
    "value,error",
    [(True, TypeError), (0, ValueError), (-2, ValueError), (1.0, TypeError), (np.nan, TypeError)],
)
def test_invalid_worker_counts_are_rejected(value, error):
    with pytest.raises(error, match="n_jobs"):
        profile_lmer(fitted_model(), n_jobs=value)


def test_nonconverged_input_is_rejected():
    result = replace(fitted_model(), converged=False)
    with pytest.raises(ValueError, match="converged fitted model"):
        result.confint(method="profile")


def test_failed_nuisance_optimization_is_reported(monkeypatch):
    result = fitted_model()
    monkeypatch.setattr(
        lmm_profile.optimize,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="forced failure"),
    )
    with pytest.raises(RuntimeError, match="nuisance optimization failed.*forced failure"):
        result.confint(method="profile")


def test_failed_gradient_optimization_retries_the_likelihood_with_cobyqa(monkeypatch):
    result = fitted_model()
    likelihood = lmm_profile._LMMProfileLikelihood(result.matrices)
    original = optimize.minimize
    methods = []

    def fail_first(fun, start, **kwargs):
        methods.append(kwargs["method"])
        if len(methods) == 1:
            return SimpleNamespace(success=False, message="line search failed")
        return original(fun, start, **kwargs)

    monkeypatch.setattr(lmm_profile.optimize, "minimize", fail_first)
    value = float(result.beta[1] + 0.5)
    fitted = likelihood.fit(result.theta * 2, (1,), (value,))
    assert methods == ["L-BFGS-B", "COBYQA"]
    assert_allclose(fitted.deviance, intercept_reference(result, {1: value}), atol=2e-7)


def test_failed_factorization_is_reported(monkeypatch):
    result = fitted_model()
    monkeypatch.setattr(lmm_profile.LMMOptimizer, "_evaluate_core", lambda self, theta: None)
    with pytest.raises(RuntimeError, match="factorization failed"):
        result.confint(method="profile")


def boundary_model():
    data = pd.DataFrame({"y": np.tile([-1.0, -1.0, 1.0, 1.0], 6), "g": np.repeat(np.arange(6), 4)})
    with pytest.warns(UserWarning, match="singular"):
        result = lmer("y ~ 1 + (1 | g)", data, REML=False)
    assert result.theta[0] == 0
    return result


@pytest.mark.parametrize("start", [0.0, 1e-8])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constrained_variance_can_leave_a_zero_start(start, value):
    result = boundary_model()
    likelihood = lmm_profile._LMMProfileLikelihood(result.matrices)
    fitted = likelihood.fit(np.array([start]), (0,), (value,))
    assert_allclose(fitted.deviance, intercept_reference(result, {0: value}), atol=2e-7)
    if value == 1:
        assert fitted.theta[0] > 0.5
    else:
        assert fitted.theta[0] < 1e-6


def test_boundary_fit_profile_endpoints_reoptimize_variance():
    result = boundary_model()
    profile = result.profile("(Intercept)", level=0.999, n_points=3)["(Intercept)"]
    minimum = intercept_reference(result, {})
    cutoff = stats.chi2.isf(0.001, 1)
    for endpoint in [profile.ci_lower, profile.ci_upper]:
        assert_allclose(intercept_reference(result, {0: endpoint}) - minimum, cutoff, atol=2e-7)


@pytest.mark.parametrize("start", [0.8, 2.0, 10.0])
def test_gradient_fit_landing_at_zero_does_not_hide_positive_variance(start):
    result = boundary_model()
    response = result.matrices.y + np.repeat(np.tile([-0.7, 0.7], 3), 4)
    result = replace(result, matrices=replace(result.matrices, y=response))
    likelihood = lmm_profile._LMMProfileLikelihood(result.matrices)
    fitted = likelihood.fit(np.array([start]))
    assert fitted.theta[0] > 0.3
    assert_allclose(fitted.deviance, intercept_reference(result, {}), atol=2e-7)


def test_unbracketed_interval_raises_instead_of_substituting_wald(monkeypatch):
    result = fitted_model()
    monkeypatch.setattr(
        lmm_profile._LMMParameterProfile, "deviance", lambda self, value: self.minimum
    )
    with pytest.raises(RuntimeError, match="Could not bracket"):
        result.confint("x", method="profile")


def test_lower_than_reference_deviance_is_reported(monkeypatch):
    result = fitted_model()
    likelihood, optimum, _ = lmm_profile._reference(result)
    profile = lmm_profile._LMMParameterProfile(likelihood, 1, optimum)
    bad = replace(optimum, evaluation=replace(optimum.evaluation, deviance=optimum.deviance - 1))
    monkeypatch.setattr(likelihood, "fit", lambda *args, **kwargs: bad)
    with pytest.raises(RuntimeError, match="lower deviance than the ML optimum"):
        profile.deviance(profile.mle + 0.1)


def test_large_random_system_stays_sparse_during_covariance_profiling(monkeypatch):
    from scipy import sparse

    rng = np.random.default_rng(217)
    groups = 300
    group = np.repeat(np.arange(groups), 4)
    x = rng.normal(size=len(group))
    y = 0.3 + 0.7 * x + rng.normal(scale=0.5, size=groups)[group]
    y += rng.normal(scale=0.4, size=len(group))
    result = lmer("y ~ x + (1 | g)", pd.DataFrame(dict(y=y, x=x, g=group)), REML=False)
    for matrix_type in (sparse.csc_matrix, sparse.csr_matrix):
        original = matrix_type.toarray

        def guarded(self, *args, _original=original, **kwargs):
            if self.shape == (groups, groups):
                pytest.fail("large random-effect precision must stay sparse")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(matrix_type, "toarray", guarded)
    profile = result.profile("x", n_points=3)["x"]
    assert profile.ci_lower < profile.mle < profile.ci_upper
    assert_allclose(
        profile.zeta**2, [stats.chi2.isf(0.05, 1), 0, stats.chi2.isf(0.05, 1)], atol=2e-7
    )


def test_ml_and_reml_inputs_have_the_same_profile_reference():
    ml, reml = fitted_model(reml=False), fitted_model(reml=True)
    first = ml.profile("x", n_points=5)["x"]
    with pytest.warns(UserWarning, match="ML refit"):
        second = reml.profile("x", n_points=5)["x"]
    assert_allclose(first.values, second.values, atol=2e-7)
    assert_allclose(first.zeta, second.zeta, atol=2e-6)


@pytest.mark.parametrize("first,second", [("x", "x"), ("missing", "x")])
def test_joint_surface_requires_two_known_distinct_parameters(first, second):
    with pytest.raises(ValueError, match="distinct|not found"):
        slice2D(fitted_model(), first, second, profile_covariance=True)


@pytest.mark.parametrize("value", [0, 1, "yes", None])
def test_surface_profile_mode_requires_a_boolean(value):
    with pytest.raises(ValueError, match="profile_covariance must be a boolean"):
        slice2D(fitted_model(), "x", "z", profile_covariance=value)
