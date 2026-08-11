from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation.reml import _build_lambda, _profiled_deviance_core
from mixedlm.formula.parser import parse_formula
from mixedlm.inference.profile import (
    _profile_deviance_2d,
    _profile_deviance_at_beta,
    profile_lmer,
    slice2D,
)
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.control import LmerControl
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose
from scipy import linalg


def _profile_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(417)
    n_groups = 8
    observations_per_group = 6
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = np.tile(np.linspace(-1.5, 1.5, observations_per_group), n_groups)
    intercepts = np.repeat(
        np.array([-1.4, 0.2, 1.6, -0.7, 0.9, -1.1, 0.4, 1.3]),
        observations_per_group,
    )
    slopes = np.repeat(
        np.array([0.8, -0.55, 0.1, 0.65, -0.25, -0.75, 0.45, -0.05]),
        observations_per_group,
    )
    base_y = 1.2 + 0.65 * x + intercepts + slopes * x + rng.normal(0.0, 0.18, len(x))
    offset = 0.25 * np.sin(np.arange(len(x)) / 4.0) + 0.1 * x
    weights = np.exp(np.linspace(-0.8, 0.9, len(x)))

    base = pd.DataFrame({"y": base_y, "x": x, "group": group.astype(str)})
    shifted = base.assign(y=base_y + offset)
    return base, shifted, weights, offset


def _fit(
    formula: str,
    data: pd.DataFrame,
    weights: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    reml: bool = True,
):
    return lmer(
        formula,
        data,
        weights=weights,
        offset=offset,
        REML=reml,
        control=LmerControl(use_rust=False, em_init=False),
    )


def _fit_fixed_only(
    data: pd.DataFrame,
    weights: np.ndarray,
    offset: np.ndarray,
    reml: bool,
) -> LmerResult:
    formula = parse_formula("y ~ x")
    matrices = build_model_matrices(
        formula,
        data,
        weights=weights,
        offset=offset,
        na_action=None,
    )
    core = _profiled_deviance_core(np.empty(0), matrices, REML=reml)
    assert core is not None
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=core.beta,
        sigma=core.sigma,
        u=core.u,
        deviance=core.deviance,
        REML=reml,
        converged=True,
        n_iter=0,
    )


def _direct_held_deviance(result, held: dict[int, float]) -> float:
    matrices = result.matrices
    keep_idx = [column for column in range(matrices.n_fixed) if column not in held]
    y_adjusted = matrices.y - matrices.offset
    for column, value in held.items():
        y_adjusted = y_adjusted - value * matrices.X[:, column]

    sqrt_weights = np.sqrt(matrices.weights)
    weighted_y = sqrt_weights * y_adjusted
    weighted_X = sqrt_weights[:, None] * matrices.X[:, keep_idx]

    if matrices.n_random == 0:
        covariance = np.eye(matrices.n_obs)
    else:
        lambda_matrix = _build_lambda(result.theta, matrices.random_structures).toarray()
        weighted_random = sqrt_weights[:, None] * (matrices.Z @ lambda_matrix)
        covariance = np.eye(matrices.n_obs) + weighted_random @ weighted_random.T

    covariance_factor = linalg.cho_factor(covariance, lower=True)
    covariance_inverse_y = linalg.cho_solve(covariance_factor, weighted_y)
    if keep_idx:
        covariance_inverse_X = linalg.cho_solve(covariance_factor, weighted_X)
        information = weighted_X.T @ covariance_inverse_X
        beta = linalg.solve(
            information,
            weighted_X.T @ covariance_inverse_y,
            assume_a="pos",
        )
    else:
        information = np.empty((0, 0), dtype=np.float64)
        beta = np.empty(0, dtype=np.float64)

    weighted_residual = weighted_y - weighted_X @ beta
    pwrss = float(
        weighted_residual @ linalg.cho_solve(covariance_factor, weighted_residual)
    )
    denominator = matrices.n_obs - len(keep_idx) if result.REML else matrices.n_obs
    sigma2 = pwrss / denominator
    deviance = (
        denominator * (1.0 + np.log(2.0 * np.pi * sigma2))
        + 2.0 * np.sum(np.log(np.diag(covariance_factor[0])))
        - np.sum(np.log(matrices.weights))
    )
    if result.REML and keep_idx:
        deviance += np.linalg.slogdet(information)[1]
    return float(deviance)


@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("formula", ["y ~ x", "y ~ x + (x | group)"])
def test_weighted_profile_deviance_matches_direct_likelihood(formula: str, reml: bool) -> None:
    _, shifted, weights, offset = _profile_data()
    if formula == "y ~ x":
        result = _fit_fixed_only(shifted, weights, offset, reml)
    else:
        result = _fit(formula, shifted, weights, offset=offset, reml=reml)
    slope_idx = result.matrices.fixed_names.index("x")
    slope_value = float(
        result.beta[slope_idx] + 0.35 * np.sqrt(result.vcov()[slope_idx, slope_idx])
    )

    actual = _profile_deviance_at_beta(result, slope_idx, slope_value)
    expected = _direct_held_deviance(result, {slope_idx: slope_value})

    assert actual == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("reml", [False, True])
def test_weighted_2d_profile_deviance_matches_direct_likelihood(reml: bool) -> None:
    _, shifted, weights, offset = _profile_data()
    result = _fit("y ~ x + (x | group)", shifted, weights, offset=offset, reml=reml)
    intercept_idx = result.matrices.fixed_names.index("(Intercept)")
    slope_idx = result.matrices.fixed_names.index("x")
    intercept_value = float(result.beta[intercept_idx] - 0.2)
    slope_value = float(result.beta[slope_idx] + 0.1)

    actual = _profile_deviance_2d(
        result,
        intercept_idx,
        intercept_value,
        slope_idx,
        slope_value,
    )
    expected = _direct_held_deviance(
        result,
        {intercept_idx: intercept_value, slope_idx: slope_value},
    )

    assert actual == pytest.approx(expected, abs=1e-10)


def test_weighted_profiles_preserve_offsets_scale_and_parallel_fallback(monkeypatch) -> None:
    import mixedlm.inference.profile as profile_module

    base, shifted, weights, offset = _profile_data()
    formula = "y ~ x + (x | group)"
    scale = 16.0
    baseline = _fit(formula, base, weights)
    shifted_fit = _fit(formula, shifted, weights, offset=offset)
    scaled_fit = _fit(formula, shifted, scale * weights, offset=offset)

    baseline_profile = profile_lmer(baseline, which="x", n_points=9)["x"]
    shifted_profile = profile_lmer(shifted_fit, which="x", n_points=9)["x"]
    scaled_profile = profile_lmer(scaled_fit, which="x", n_points=9)["x"]

    class UnavailableExecutor:
        def __init__(self, *args, **kwargs):
            raise PermissionError("process semaphores are unavailable")

    monkeypatch.setattr(profile_module, "ProcessPoolExecutor", UnavailableExecutor)
    with pytest.warns(RuntimeWarning, match="falling back to serial execution"):
        fallback_profile = profile_lmer(shifted_fit, which="x", n_points=9, n_jobs=2)["x"]

    for profile in (shifted_profile, scaled_profile, fallback_profile):
        assert_allclose(profile.values, baseline_profile.values, rtol=3e-5, atol=1e-7)
        assert_allclose(profile.zeta, baseline_profile.zeta, rtol=3e-5, atol=1e-7)
        assert profile.ci_lower == pytest.approx(baseline_profile.ci_lower, rel=3e-5)
        assert profile.ci_upper == pytest.approx(baseline_profile.ci_upper, rel=3e-5)

    baseline_slice = slice2D(baseline, "(Intercept)", "x", n_points=5)
    shifted_slice = slice2D(shifted_fit, "(Intercept)", "x", n_points=5)
    scaled_slice = slice2D(scaled_fit, "(Intercept)", "x", n_points=5)

    for profile_slice in (shifted_slice, scaled_slice):
        assert_allclose(profile_slice.values1, baseline_slice.values1, rtol=3e-5, atol=1e-7)
        assert_allclose(profile_slice.values2, baseline_slice.values2, rtol=3e-5, atol=1e-7)
        assert_allclose(profile_slice.zeta, baseline_slice.zeta, rtol=3e-5, atol=1e-6)
