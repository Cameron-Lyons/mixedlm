from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation.reml import (
    _HAS_RUST,
    LMMOptimizer,
    _profiled_deviance_core,
    profiled_deviance,
)
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose

from tests._lmm_oracles import direct_profiled_likelihood


def _weighted_random_slope_matrices() -> ModelMatrices:
    group = np.repeat(["a", "b", "c"], 4)
    x = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), 3)
    intercepts = np.repeat(np.array([-1.0, 0.75, 1.5]), 4)
    slopes = np.repeat(np.array([0.4, -0.2, 0.7]), 4)
    noise = np.array([0.2, -0.1, 0.05, -0.15] * 3)
    offset = np.linspace(-0.15, 0.2, len(x))
    y = 2.0 + 1.25 * x + intercepts + slopes * x + offset + noise
    weights = np.linspace(0.7, 1.6, len(x))
    data = pd.DataFrame({"y": y, "x": x, "group": group})

    return build_model_matrices(
        parse_formula("y ~ x + (x | group)"),
        data,
        weights=weights,
        offset=offset,
        na_action=None,
    )


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("sparse_profile", [False, True])
def test_profiled_core_matches_direct_marginal_likelihood(
    reml: bool, sparse_profile: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mixedlm.models.shared_utils._SPARSE_PROJECTION_MIN_RANDOM",
        0 if sparse_profile else np.inf,
    )
    matrices = _weighted_random_slope_matrices()
    theta = np.array([0.8, 0.15, 0.45])

    result = _profiled_deviance_core(theta, matrices, REML=reml)
    expected = direct_profiled_likelihood(theta, matrices, reml)

    assert result is not None
    assert result.deviance == pytest.approx(expected["deviance"], abs=1e-10)
    assert_allclose(result.beta, expected["beta"], rtol=0, atol=1e-12)
    assert result.sigma == pytest.approx(expected["sigma"], abs=1e-12)
    assert_allclose(result.u, expected["u"], rtol=0, atol=1e-12)
    assert result.wrss == pytest.approx(expected["wrss"], abs=1e-12)
    assert result.ussq == pytest.approx(expected["ussq"], abs=1e-12)
    assert result.pwrss == pytest.approx(expected["pwrss"], abs=1e-12)
    assert result.pwrss == pytest.approx(result.wrss + result.ussq, abs=1e-12)
    assert_allclose(result.fixed_information, expected["fixed_information"], atol=1e-12)


def test_profiled_reml_is_a_deprecated_reml_deviance_alias() -> None:
    from mixedlm.estimation import profiled_reml

    matrices = _weighted_random_slope_matrices()
    theta = np.array([0.8, 0.15, 0.45])
    with pytest.warns(DeprecationWarning, match="profiled_deviance"):
        deviance = profiled_reml(theta, matrices)
    assert deviance == profiled_deviance(theta, matrices, REML=True)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("cov_type", ["us", "cs", "ar1", "diagonal"])
@pytest.mark.parametrize("sparse_profile", [False, True])
def test_cached_optimizer_matches_direct_likelihood(
    reml: bool, cov_type: str, sparse_profile: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "mixedlm.models.shared_utils._SPARSE_PROJECTION_MIN_RANDOM",
        0 if sparse_profile else np.inf,
    )
    matrices = _weighted_random_slope_matrices()
    structure = matrices.random_structures[0]
    if cov_type == "diagonal":
        structure.correlated = False
        theta_values = ([0.8, 0.45], [0.0, 0.6], [1.2, 0.0])
    elif cov_type in ("cs", "ar1"):
        structure.cov_type = cov_type
        theta_values = ([0.8, -0.25], [0.0, 0.4], [1.2, 0.6])
    else:
        theta_values = ([0.8, 0.15, 0.45], [0.0, 0.0, 0.6], [1.2, -0.3, 0.0])
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=False)

    for values in (*theta_values, theta_values[0]):
        theta = np.array(values)
        expected = direct_profiled_likelihood(theta, matrices, reml)
        assert optimizer.objective(theta) == pytest.approx(expected["deviance"], abs=1e-10)
        estimates = optimizer._final_evaluation(theta)
        assert_allclose(estimates.beta, expected["beta"], atol=1e-12)
        assert estimates.sigma == pytest.approx(expected["sigma"], abs=1e-12)
        assert_allclose(estimates.u, expected["u"], atol=1e-12)


def test_optimizer_reuses_weighted_products(monkeypatch: pytest.MonkeyPatch) -> None:
    from mixedlm.estimation import reml

    matrices = _weighted_random_slope_matrices()
    optimizer = LMMOptimizer(matrices, use_rust=False)
    theta = np.array([0.8, 0.15, 0.45])
    expected = optimizer.objective(theta)

    def reject_recomputation(*args, **kwargs):
        raise AssertionError("weighted products must be reused across parameter evaluations")

    monkeypatch.setattr(reml._LMMCrossproducts, "from_matrices", reject_recomputation)
    theta[0] = 1.2
    assert optimizer.objective(theta) != pytest.approx(expected)
    theta[0] = 0.8
    assert optimizer.objective(theta) == expected
    optimizer._final_evaluation(theta)


def test_new_optimizer_uses_updated_response_and_weights() -> None:
    matrices = _weighted_random_slope_matrices()
    theta = np.array([0.8, 0.15, 0.45])
    original = LMMOptimizer(matrices, use_rust=False).objective(theta)
    updated = replace(
        matrices,
        y=matrices.y + np.linspace(-0.4, 0.9, matrices.n_obs),
        weights=matrices.weights[::-1].copy(),
        offset=matrices.offset * 2.0,
    )
    actual = LMMOptimizer(updated, use_rust=False).objective(theta)
    expected = direct_profiled_likelihood(theta, updated, reml=True)

    assert actual != pytest.approx(original)
    assert actual == pytest.approx(expected["deviance"], abs=1e-10)


@pytest.mark.parametrize("reml", [False, True])
def test_fixed_only_optimizer_matches_uncached_deviance(reml: bool) -> None:
    from scipy import sparse

    matrices = _weighted_random_slope_matrices()
    matrices = replace(
        matrices,
        Z=sparse.csc_matrix((matrices.n_obs, 0)),
        n_random=0,
        random_structures=[],
    )
    optimizer = LMMOptimizer(matrices, REML=reml, use_rust=False)
    theta = np.array([])

    assert optimizer.objective(theta) == profiled_deviance(theta, matrices, REML=reml)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")
@pytest.mark.parametrize("reml", [False, True])
def test_rust_profiled_deviance_matches_direct_marginal_likelihood(reml: bool) -> None:
    matrices = _weighted_random_slope_matrices()
    theta = np.array([0.8, 0.15, 0.45])
    expected = direct_profiled_likelihood(theta, matrices, reml)

    deviance = LMMOptimizer(matrices, REML=reml, use_rust=True).objective(theta)

    assert deviance == pytest.approx(expected["deviance"], abs=1e-10)


def test_strong_group_signal_does_not_collapse_to_boundary() -> None:
    rng = np.random.default_rng(91)
    n_groups = 12
    observations_per_group = 8
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.0, 1.0, observations_per_group), n_groups)
    group_effect = np.linspace(-3.0, 3.0, n_groups)
    y = 4.0 + 1.5 * x + group_effect[group] + rng.normal(0.0, 0.15, len(group))
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})

    result = lmer(
        "y ~ x + (1 | group)",
        data,
        control=LmerControl(use_rust=False, em_init=False),
    )
    boundary_deviance = profiled_deviance(np.zeros(1), result.matrices, REML=True)

    assert result.converged
    assert not result.isSingular()
    assert result.theta[0] > 5.0
    assert result.sigma < 0.25
    assert result.deviance < boundary_deviance - 100.0
    assert result.logLik().value == pytest.approx(-0.5 * result.deviance, abs=1e-12)
