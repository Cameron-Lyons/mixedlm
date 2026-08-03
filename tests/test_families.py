from __future__ import annotations

import warnings

import numpy as np
import pytest
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
