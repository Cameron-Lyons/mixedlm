from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import laplace
from mixedlm.estimation.laplace import pirls
from mixedlm.families import (
    Binomial,
    Gamma,
    GammaInverse,
    Gaussian,
    InverseGaussian,
    InverseGaussianCanonical,
    NegativeBinomial,
    Poisson,
)
from mixedlm.families.custom import QuasiFamily
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


def test_binomial_mean_domain_is_bounded_on_both_sides() -> None:
    mu = np.array([-2.0, 0.25, 4.0])

    bounded = Binomial().clamp_mu(mu, eps=1e-6)

    np.testing.assert_allclose(bounded, [1e-6, 0.25, 1 - 1e-6])


@pytest.mark.parametrize(
    "family",
    [
        Poisson(),
        Gamma(),
        GammaInverse(),
        InverseGaussian(),
        InverseGaussianCanonical(),
        NegativeBinomial(),
    ],
)
def test_positive_mean_families_have_no_artificial_upper_bound(family) -> None:
    mu = np.array([-2.0, 0.25, 40.0])
    out = np.empty_like(mu)

    bounded = family.clamp_mu(mu, eps=1e-6, out=out)

    assert bounded is out
    np.testing.assert_allclose(bounded, [1e-6, 0.25, 40.0])


def test_unbounded_and_quasi_family_mean_domains() -> None:
    mu = np.array([-2.0, 40.0])

    assert Gaussian().clamp_mu(mu) is mu
    np.testing.assert_allclose(QuasiFamily(Poisson()).clamp_mu(mu), [1e-10, 40.0])


def test_poisson_simulation_preserves_large_means() -> None:
    class RecordingRng:
        def poisson(self, mu):
            self.mu = mu.copy()
            return np.zeros_like(mu)

    rng = RecordingRng()

    Poisson().simulate(np.array([2.0, 20.0]), rng=rng)

    np.testing.assert_allclose(rng.mu, [2.0, 20.0])


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
