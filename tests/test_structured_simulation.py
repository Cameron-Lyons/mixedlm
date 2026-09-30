from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import families
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.bootstrap import _simulate_glmer, _simulate_lmer
from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


def _result(kind, cov_type="us", n_groups=3, n_terms=3):
    correlated = cov_type != "diagonal"
    structure = RandomEffectStructure(
        grouping_factor="group",
        term_names=[f"term{i}" for i in range(n_terms)],
        n_levels=n_groups,
        n_terms=n_terms,
        correlated=correlated,
        level_map={},
        cov_type="us" if cov_type == "diagonal" else cov_type,
    )
    if cov_type in ("cs", "ar1"):
        theta = np.array([0.8, -0.2]) if n_terms > 1 else np.array([0.8])
    elif correlated:
        factor = np.array([[0.8, 0.0, 0.0], [-0.2, 0.7, 0.0], [0.3, -0.1, 0.6]])
        theta = factor[:n_terms, :n_terms][np.tril_indices(n_terms)]
    else:
        theta = np.array([0.8, 0.0, 0.6])[:n_terms]
    n = n_groups * n_terms
    matrices = ModelMatrices(
        y=np.zeros(n),
        X=np.ones((n, 1)),
        Z=sparse.eye(n, format="csc"),
        fixed_names=["(Intercept)"],
        random_structures=[structure],
        n_obs=n,
        n_fixed=1,
        n_random=n,
        weights=np.tile(np.array([1.0, 4.0, 9.0])[:n_terms], n_groups),
        offset=np.tile(np.array([0.1, -0.2, 0.3])[:n_terms], n_groups),
    )
    common = dict(
        formula=parse_formula("y ~ 1 + (1 | group)"),
        matrices=matrices,
        theta=theta,
        beta=np.array([0.25]),
        u=np.zeros(n),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmer":
        return LmerResult(sigma=1.4, REML=True, **common)
    return GlmerResult(family=families.Gaussian(), nAGQ=1, **common)


def _expected_covariance(cov_type):
    if cov_type == "us":
        factor = np.array([[0.8, 0.0, 0.0], [-0.2, 0.7, 0.0], [0.3, -0.1, 0.6]])
        return factor @ factor.T
    if cov_type == "diagonal":
        return np.diag(np.array([0.8, 0.0, 0.6]) ** 2)
    correlation = (
        np.full((3, 3), -0.2)
        if cov_type == "cs"
        else (-0.2) ** np.abs(np.arange(3)[:, None] - np.arange(3))
    )
    np.fill_diagonal(correlation, 1.0)
    return 0.8**2 * correlation


@pytest.mark.parametrize("kind", ["lmer", "glmer"])
@pytest.mark.parametrize("cov_type", ["us", "diagonal", "cs", "ar1"])
@pytest.mark.parametrize("method", ["single", "batch", "bootstrap"])
def test_response_covariance_matches_fitted_model(kind, cov_type, method):
    # Identity Z exposes each coefficient directly, so empirical covariances can
    # be checked against the model definition without using its factor builder.
    result = _result(kind, cov_type, n_groups=3 if method == "batch" else 30_000)
    nsim = 10_000 if method == "batch" else 1
    if method == "bootstrap":
        np.random.seed(63)
        simulated = _simulate_lmer(result) if kind == "lmer" else _simulate_glmer(result)
    else:
        simulated = result.simulate(nsim=nsim, seed=63)
    mean = result.matrices.X @ result.beta + result.matrices.offset
    centered = simulated - (mean if nsim == 1 else mean[:, None])
    draws = centered.reshape(-1, 3, nsim).transpose(0, 2, 1).reshape(-1, 3)
    expected = _expected_covariance(cov_type)
    if kind == "lmer":
        expected = result.sigma**2 * (expected + np.diag(1 / np.array([1.0, 4.0, 9.0])))
    else:
        expected += np.eye(3)

    assert_allclose(draws.mean(axis=0), 0.0, atol=0.035)
    assert_allclose(np.cov(draws, rowvar=False), expected, atol=0.06, rtol=0.03)


@pytest.mark.parametrize("nsim", [1, 5])
@pytest.mark.parametrize(
    "random_options", [{"use_re": False}, {"re_form": "~0"}, {"re_form": "NA"}, {}]
)
def test_lmm_residual_draws_scale_with_prior_weights(nsim, random_options):
    weighted = _result("lmer")
    weighted.theta[:] = 0.0
    unweighted = replace(weighted, matrices=replace(weighted.matrices, weights=np.ones(9)))
    mean = weighted.matrices.X @ weighted.beta + weighted.matrices.offset
    scale = np.sqrt(weighted.matrices.weights)
    if nsim > 1:
        mean = mean[:, None]
        scale = scale[:, None]

    weighted_draws = weighted.simulate(nsim=nsim, seed=17, **random_options)
    unweighted_draws = unweighted.simulate(nsim=nsim, seed=17, **random_options)

    assert_allclose(weighted_draws - mean, (unweighted_draws - mean) / scale)
    assert_array_equal(weighted_draws, weighted.simulate(nsim=nsim, seed=17, **random_options))


@pytest.mark.parametrize("kind", ["lmer", "glmer"])
@pytest.mark.parametrize("nsim", [1, 4])
@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_mixed_structures_preserve_parameter_order(kind, nsim, cov_type):
    structured = _result(kind, cov_type)
    # Follow a structured block with a diagonal block, then compare with the
    # equivalent unstructured encoding under the same random stream.
    structure = structured.matrices.random_structures[0]
    second = replace(structure, grouping_factor="other", correlated=False, cov_type="us")
    structured.matrices = replace(
        structured.matrices,
        random_structures=[replace(structure, correlated=False), second],
        Z=sparse.hstack([structured.matrices.Z, structured.matrices.Z], format="csc"),
        n_random=18,
    )
    structured.theta = np.concatenate([structured.theta, [0.3, 0.0, 0.9]])
    structured.u = np.zeros(18)
    factor = np.linalg.cholesky(_expected_covariance(cov_type))
    equivalent = replace(
        structured,
        theta=np.concatenate([factor[np.tril_indices(3)], [0.3, 0.0, 0.9]]),
        matrices=replace(
            structured.matrices,
            random_structures=[replace(structure, cov_type="us"), second],
        ),
    )

    assert_allclose(structured.simulate(nsim, seed=29), equivalent.simulate(nsim, seed=29))
    bootstrap = _simulate_lmer if kind == "lmer" else _simulate_glmer
    np.random.seed(29)
    first = bootstrap(structured)
    np.random.seed(29)
    assert_allclose(first, bootstrap(equivalent))


@pytest.mark.parametrize("kind", ["lmer", "glmer"])
@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_intercept_only_consumes_one_parameter(kind, cov_type):
    structured = _result(kind, cov_type, n_terms=1)
    equivalent = _result(kind, "us", n_terms=1)
    for nsim in (1, 3):
        assert_allclose(structured.simulate(nsim, seed=11), equivalent.simulate(nsim, seed=11))


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("cov_type", ["us", "diagonal", "cs", "ar1"])
def test_zero_glmm_variance_adds_no_artificial_random_effects(monkeypatch, bootstrap, cov_type):
    result = _result("glmer", cov_type)
    result.theta[:] = 0.0
    monkeypatch.setattr(result.family, "simulate", lambda mu, rng=None: mu)
    expected = result.matrices.X @ result.beta + result.matrices.offset

    actual = _simulate_glmer(result) if bootstrap else result.simulate(seed=8)

    assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["lmer", "glmer"])
def test_structured_batch_falls_back_without_native_sampler(monkeypatch, kind):
    import mixedlm._rust as native

    monkeypatch.delattr(native, "simulate_re_batch")
    result = _result(kind, "ar1")
    np.random.seed(41)
    expected = np.column_stack([result._simulate_once() for _ in range(4)])

    assert_allclose(result.simulate(4, seed=41), expected)
