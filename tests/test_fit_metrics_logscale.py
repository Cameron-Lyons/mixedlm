"""Extreme GLMM variances verified against decimal distribution formulas."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from mixedlm.diagnostics.fit_metrics import _distribution_specific_variance
from mixedlm.families import (
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    QuasiFamily,
)
from mixedlm.families.base import LogLink
from numpy.testing import assert_allclose


def decimal_residual(family, means, weights, approximation):
    with localcontext() as context:
        context.prec = 500
        base = getattr(family, "base_family", family)
        dispersion = Decimal(str(getattr(family, "phi", 1)))
        contributions = []
        for mean, weight in zip(means, weights, strict=True):
            mu = Decimal(str(mean))
            if isinstance(base, Gaussian):
                variance = Decimal(1)
            elif isinstance(base, Poisson):
                variance = mu
            elif isinstance(base, Gamma):
                variance = mu**2
            elif isinstance(base, InverseGaussian):
                variance = mu**3
            elif isinstance(base, NegativeBinomial):
                variance = mu + mu**2 / Decimal(str(base.theta))
            else:
                raise AssertionError("Unknown oracle distribution")
            ratio = dispersion * variance / (Decimal(str(weight)) * mu**2)
            contributions.append((1 + ratio).ln() if approximation == "lognormal" else ratio)
        return float(sum(contributions) / len(contributions))


@pytest.mark.parametrize(
    "family",
    [
        Gaussian(link="log"),
        Poisson(),
        Gamma(),
        InverseGaussian(),
        NegativeBinomial(theta=4),
        QuasiFamily(Gamma(), phi=2.5),
    ],
)
@pytest.mark.parametrize("mean", [1e-150, 0.7, 1e150, 1e200])
@pytest.mark.parametrize("approximation", ["lognormal", "delta"])
def test_builtin_log_variance_matches_high_precision_distribution(family, mean, approximation):
    means = np.array([mean, 2 * mean])
    weights = np.array([2.0, 8.0])
    expected = decimal_residual(family, means, weights, approximation)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual, used = _distribution_specific_variance(family, means, weights, approximation)
    assert used == approximation
    assert_allclose(actual, expected, rtol=2e-12, atol=0)


def test_delta_averages_before_converting_back_to_float():
    # First contribution exceeds float64; their arithmetic mean is representable.
    means = np.array([1e308, 1.0])
    weights = np.array([0.5, 1.0])
    family = InverseGaussian()
    expected = decimal_residual(family, means, weights, "delta")
    actual, _ = _distribution_specific_variance(family, means, weights, "delta")
    assert_allclose(actual, expected, rtol=2e-12, atol=0)


def test_custom_variance_override_is_preserved():
    class CustomPoisson(Poisson):
        def variance(self, mu):
            return 7 * mu

    family = CustomPoisson()
    means = np.array([2.0, 4.0])
    weights = np.array([1.0, 2.0])
    actual, used = _distribution_specific_variance(family, means, weights, "lognormal")
    assert used == "lognormal"
    assert actual == pytest.approx((np.log1p(3.5) + np.log1p(0.875)) / 2)


def test_custom_log_link_derivative_is_preserved():
    class ScaledLogLink(LogLink):
        def link(self, mu):
            return 2 * np.log(mu)

        def inverse(self, eta):
            return np.exp(eta / 2)

        def deriv(self, mu):
            return 2 / mu

    actual, used = _distribution_specific_variance(
        Poisson(link=ScaledLogLink()), np.array([2.0, 4.0]), np.array([1.0, 2.0]), "delta"
    )
    assert used == "delta"
    assert actual == pytest.approx((4 / 2 + 4 / (2 * 4)) / 2)


def test_unrepresentable_delta_variance_raises():
    with pytest.raises(ValueError, match="Delta-method residual variance is not finite"):
        _distribution_specific_variance(
            Gaussian(link="log"), np.array([1e-200]), np.ones(1), "delta"
        )
