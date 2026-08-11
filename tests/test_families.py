from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import glmer
from mixedlm.estimation import laplace
from mixedlm.estimation.laplace import pirls
from mixedlm.families import Binomial, Gamma, InverseGaussian, NegativeBinomial, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices import build_model_matrices


def test_binomial_deviance_handles_boundary_outcomes_without_warnings() -> None:
    y = np.array([0.0, 1.0])
    mu = np.array([0.25, 0.75])
    weights = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        deviance = Binomial().deviance_resids(y, mu, weights)

    expected = -2 * weights * np.log(np.array([0.75, 0.75]))
    np.testing.assert_allclose(deviance, expected)


def test_poisson_deviance_handles_zero_counts_without_warnings() -> None:
    y = np.array([0.0, 2.0])
    mu = np.array([0.5, 1.5])
    weights = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        deviance = Poisson().deviance_resids(y, mu, weights)

    expected = 2 * weights * np.array([0.5, 2 * np.log(4 / 3) - 0.5])
    np.testing.assert_allclose(deviance, expected)


def test_gamma_deviance_matches_unit_deviance() -> None:
    y = np.array([0.5, 1.0, 2.0, 4.0])
    mu = np.array([1.0, 2.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        deviance = Gamma().deviance_resids(y, mu, weights)

    ratio = y / mu
    expected = 2 * weights * (ratio - 1 - np.log(ratio))
    np.testing.assert_allclose(deviance, expected)
    assert np.all(deviance >= 0)


def test_gamma_deviance_is_zero_at_observed_mean() -> None:
    y = np.array([0.25, 1.0, 4.0])

    deviance = Gamma().deviance_resids(y, y, np.ones_like(y))

    np.testing.assert_array_equal(deviance, np.zeros_like(y))


def test_gamma_fit_recovers_finite_coefficients_and_deviance() -> None:
    rng = np.random.default_rng(123)
    n_groups = 12
    observations_per_group = 30
    n = n_groups * observations_per_group
    group = np.repeat(np.arange(n_groups), observations_per_group)
    x = rng.normal(size=n)
    random_intercepts = rng.normal(0.0, 0.15, n_groups)
    eta = 0.7 + 0.35 * x + random_intercepts[group]
    mu = np.exp(eta)
    y = rng.gamma(12.0, mu / 12.0)
    data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})

    result = glmer("y ~ x + (1 | group)", data, family=Gamma())

    assert result.converged
    assert np.isfinite(result.deviance)
    assert result.deviance >= 0
    np.testing.assert_allclose(result.beta, np.array([0.7, 0.35]), atol=0.06)
    assert np.all((result.fitted() > 0) & (result.fitted() < 10))


def test_python_pirls_preserves_poisson_means_above_one() -> None:
    n_groups = 8
    observations_per_group = 5
    data = pd.DataFrame(
        {
            "y": np.full(n_groups * observations_per_group, 5.0),
            "group": np.repeat(np.arange(n_groups), observations_per_group).astype(str),
        }
    )
    matrices = build_model_matrices(parse_formula("y ~ 1 + (1 | group)"), data)

    beta, u, deviance, converged = pirls(matrices, Poisson(), np.array([0.5]))
    fitted = np.exp(matrices.X @ beta + matrices.Z @ u)

    assert converged
    assert np.isfinite(deviance)
    np.testing.assert_allclose(fitted, 5.0, rtol=1e-6)


@pytest.mark.parametrize("family", [Gamma(), InverseGaussian(), NegativeBinomial()])
def test_unsupported_native_families_use_python_backend(monkeypatch, family) -> None:
    expected = (1.0, np.array([2.0]), np.array([3.0]))

    monkeypatch.setattr(laplace, "_HAS_RUST", True)
    monkeypatch.setattr(
        laplace,
        "_rust_laplace_deviance",
        lambda *args: pytest.fail("unsupported family was sent to the Rust backend"),
    )
    monkeypatch.setattr(laplace, "laplace_deviance", lambda *args: expected)

    actual = laplace.laplace_deviance_fast(np.array([0.5]), object(), family)

    assert actual is expected
