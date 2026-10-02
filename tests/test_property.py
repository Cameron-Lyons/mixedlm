"""Generated designs must obey Gaussian likelihood and fitted-model identities."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from mixedlm import lmer
from mixedlm.estimation.reml import _HAS_RUST, LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose
from scipy import linalg

KINDS = ["intercept", "slope", "independent", "crossed"]
FORMULAS = {
    "intercept": "y ~ x + (1 | group)",
    "slope": "y ~ x + (x | group)",
    "independent": "y ~ x + (x || group)",
    "crossed": "y ~ x + (x | group) + (1 | second)",
}
BACKENDS = [
    False,
    pytest.param(True, marks=pytest.mark.skipif(not _HAS_RUST, reason="Rust not available")),
]
FINITE = dict(allow_nan=False, allow_infinity=False, allow_subnormal=False)
SCALE = st.one_of(st.just(0.0), st.floats(min_value=0.1, max_value=2.0, **FINITE))


@st.composite
def random_problem(draw, min_groups=3):
    """Vary unbalanced group sizes, row order, weights, offsets, and covariance."""
    sizes = draw(st.lists(st.integers(3, 7), min_size=min_groups, max_size=6))
    group = np.repeat(np.arange(len(sizes)), sizes)
    n = len(group)
    row = np.arange(n)
    # Within-group spread keeps the fixed design full rank even when Hypothesis
    # shrinks every generated noise value to a constant.
    x = np.concatenate([np.linspace(-1.0, 1.0, size) for size in sizes])
    x += draw(arrays(np.float64, n, elements=st.floats(-0.2, 0.2, **FINITE)))
    noise = draw(arrays(np.float64, n, elements=st.floats(-0.25, 0.25, **FINITE)))
    offset = draw(arrays(np.float64, n, elements=st.floats(-0.5, 0.5, **FINITE)))
    weights = draw(arrays(np.float64, n, elements=st.floats(0.25, 4.0, **FINITE)))
    y = 1.2 + 0.7 * x + np.sin(group) + np.cos(row * 1.7) + noise + offset
    order = np.asarray(draw(st.permutations(range(n))))
    frame = pd.DataFrame(dict(y=y[order], x=x[order], group=group[order], second=(row % 3)[order]))
    parameters = np.array(
        [draw(SCALE), draw(st.floats(-0.8, 0.8, **FINITE)), draw(SCALE), draw(SCALE)]
    )
    return frame, weights[order], offset[order], parameters


def theta_for(kind, parameters):
    if kind == "intercept":
        return parameters[:1]
    if kind == "independent":
        return parameters[[0, 2]]
    return parameters[: 4 if kind == "crossed" else 3]


def observation_covariance(frame, weights, kind, theta):
    """Build marginal covariance from row labels, without model matrix helpers."""
    same_group = frame["group"].to_numpy()[:, None] == frame["group"].to_numpy()
    if kind == "intercept":
        random = theta[0] ** 2 * same_group
    else:
        intercept, slope = theta[[0, 1]] if kind == "independent" else theta[[0, 2]]
        cross = 0.0 if kind == "independent" else theta[1]
        terms = np.column_stack((np.ones(len(frame)), frame["x"].to_numpy()))
        covariance = np.array(
            [[intercept**2, intercept * cross], [intercept * cross, slope**2 + cross**2]]
        )
        random = (terms @ covariance @ terms.T) * same_group
    if kind == "crossed":
        second = frame["second"].to_numpy()
        random += theta[3] ** 2 * (second[:, None] == second)
    return np.diag(1.0 / weights) + random, random


def gaussian_reference(frame, weights, offset, kind, theta, restricted):
    """Use dense observation-space GLS, independent of the profiled solvers."""
    covariance, random = observation_covariance(frame, weights, kind, theta)
    design = np.column_stack((np.ones(len(frame)), frame["x"].to_numpy()))
    response = frame["y"].to_numpy() - offset
    factor = linalg.cho_factor(covariance, lower=True)
    inverse_x = linalg.cho_solve(factor, design)
    information = design.T @ inverse_x
    beta = np.linalg.solve(information, inverse_x.T @ response)
    residual = response - design @ beta
    projected = linalg.cho_solve(factor, residual)
    rss = residual @ projected
    df = len(frame) - design.shape[1] if restricted else len(frame)
    sigma = np.sqrt(rss / df)
    deviance = df * (1.0 + np.log(2.0 * np.pi * sigma**2))
    deviance += 2.0 * np.log(np.diag(factor[0])).sum()
    if restricted:
        deviance += np.linalg.slogdet(information)[1]
    return beta, sigma, deviance, random @ projected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("native", BACKENDS)
@given(problem=random_problem())
@settings(max_examples=25, deadline=None, derandomize=True)
def test_generated_profiles_match_independent_gaussian_likelihood(
    kind, restricted, native, problem
):
    frame, weights, offset, parameters = problem
    theta = theta_for(kind, parameters)
    matrices = build_model_matrices(parse_formula(FORMULAS[kind]), frame, weights, offset)
    optimizer = LMMOptimizer(matrices, REML=restricted, use_rust=native)
    beta, sigma, deviance, random_prediction = gaussian_reference(
        frame, weights, offset, kind, theta, restricted
    )

    assert_allclose(optimizer.objective(theta), deviance, rtol=2e-12, atol=2e-10)
    actual = optimizer._final_evaluation(theta)
    assert_allclose(actual.beta, beta, rtol=2e-12, atol=2e-11)
    assert_allclose(actual.sigma, sigma, rtol=2e-12, atol=2e-11)
    assert_allclose(actual.deviance, deviance, rtol=2e-12, atol=2e-10)
    assert_allclose(matrices.Z @ actual.u, random_prediction, rtol=2e-11, atol=2e-10)
    assert_allclose(actual.wrss + actual.ussq, actual.pwrss, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("native", BACKENDS)
@given(
    problem=random_problem(),
    kind=st.sampled_from(KINDS),
    scale=st.sampled_from([-3.0, -0.25, 0.5, 2.0]),
    shift=st.floats(-10.0, 10.0, **FINITE),
)
@settings(max_examples=25, deadline=None, derandomize=True)
def test_affine_response_transform_preserves_profile_geometry(
    restricted, native, problem, kind, scale, shift
):
    frame, weights, offset, parameters = problem
    theta = theta_for(kind, parameters)
    matrices = build_model_matrices(parse_formula(FORMULAS[kind]), frame, weights, offset)
    optimizer = LMMOptimizer(matrices, REML=restricted, use_rust=native)
    original = optimizer._final_evaluation(theta)
    response = scale * (matrices.y - matrices.offset) + shift + matrices.offset
    transformed = optimizer.with_response(response)
    actual = transformed._final_evaluation(theta)
    fresh = LMMOptimizer(replace(matrices, y=response), REML=restricted, use_rust=native)
    expected_beta = scale * original.beta + np.array([shift, 0.0])
    df = matrices.n_obs - matrices.n_fixed if restricted else matrices.n_obs

    assert_allclose(actual.beta, expected_beta, rtol=2e-11, atol=2e-10)
    assert_allclose(actual.sigma, abs(scale) * original.sigma, rtol=2e-11, atol=2e-10)
    assert_allclose(actual.u, scale * original.u, rtol=2e-11, atol=2e-10)
    assert_allclose(
        actual.deviance,
        original.deviance + 2.0 * df * np.log(abs(scale)),
        rtol=2e-12,
        atol=2e-10,
    )
    assert_allclose(transformed.objective(theta), fresh.objective(theta), atol=2e-10)
    # Shared design state must not let a response refit overwrite its parent.
    assert optimizer.objective(theta) == original.deviance


@given(problem=random_problem(min_groups=5))
@settings(max_examples=15, deadline=None, derandomize=True)
def test_generated_public_fits_satisfy_gls_and_prediction_identities(problem):
    frame, weights, offset, _ = problem
    model = lmer(FORMULAS["intercept"], frame, weights=weights, offset=offset)
    beta, sigma, deviance, random_prediction = gaussian_reference(
        frame, weights, offset, "intercept", model.theta, True
    )
    design = np.column_stack((np.ones(len(frame)), frame["x"].to_numpy()))
    expected = design @ beta + random_prediction + offset

    assert model.converged
    boundary = LMMOptimizer(model.matrices, REML=True).objective(np.zeros(1))
    assert model.deviance <= boundary + 1e-8
    assert_allclose(model.beta, beta, rtol=2e-11, atol=2e-10)
    assert_allclose(model.sigma, sigma, rtol=2e-11, atol=2e-10)
    assert_allclose(model.deviance, deviance, rtol=2e-11, atol=2e-10)
    assert_allclose(model.fitted(), expected, rtol=2e-11, atol=2e-10)
    assert_allclose(model.predict(), expected, rtol=2e-11, atol=2e-10)
    assert_allclose(model.fitted() + model.residuals(), frame["y"], atol=2e-10)
    assert_allclose(float(model.logLik()), -0.5 * deviance, atol=2e-10)


@given(
    intercept=st.floats(-100.0, 100.0, **FINITE),
    slope=st.floats(-10.0, 10.0, **FINITE),
)
@settings(max_examples=15, deadline=None, derandomize=True)
def test_lmer_recovers_known_parameters(intercept, slope):
    rng = np.random.default_rng(42)
    n_groups, per_group = 8, 30
    group = np.repeat(np.arange(n_groups), per_group)
    # Identical within-group observation times make fixed-effect estimates
    # exactly equal pooled OLS, giving a reference independent of the fit.
    x = np.tile(np.linspace(-1.0, 1.0, per_group), n_groups)
    effects = rng.normal(0.0, 0.5, n_groups)
    effects -= effects.mean()
    noise = rng.normal(0.0, 0.1, len(x))
    y = intercept + slope * x + effects[group] + noise
    data = pd.DataFrame(dict(y=y, x=x, group=group))
    model = lmer(FORMULAS["intercept"], data)
    expected = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]

    assert model.converged
    assert_allclose(model.beta, expected, rtol=2e-12, atol=2e-10)
    assert_allclose(model.beta, [intercept, slope], rtol=0.0, atol=0.025)
