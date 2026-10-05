"""Custom and quasi families satisfy the family contract that glmer relies on."""

import numpy as np
import pytest
from mixedlm.families import (
    Binomial,
    CustomFamily,
    Gamma,
    Gaussian,
    InverseGaussian,
    LogLink,
    NegativeBinomial,
    Poisson,
    QuasiFamily,
    validate_family,
)
from numpy.testing import assert_allclose

_UNBOUNDED_TEST_MEANS = pytest.mark.xfail(
    strict=True,
    reason="validate_family picks its test means from the link type, so binomial "
    "links other than logit/probit/cloglog are checked at means above 1",
)


@pytest.mark.parametrize(
    "family",
    [
        Gaussian(),
        Gaussian(link="log"),
        Binomial(),
        Binomial(link="probit"),
        Binomial(link="cloglog"),
        pytest.param(Binomial(link="cauchit"), marks=_UNBOUNDED_TEST_MEANS),
        pytest.param(Binomial(link="log"), marks=_UNBOUNDED_TEST_MEANS),
        Poisson(),
        Poisson(link="sqrt"),
        Gamma(),
        InverseGaussian(),
        NegativeBinomial(theta=2.0),
    ],
    ids=lambda family: f"{type(family).__name__}-{type(family.link).__name__}",
)
def test_validate_family_accepts_builtin_families(family):
    assert validate_family(family) is True


class _UnboundedSlopeLog(LogLink):
    def deriv(self, mu):
        return np.full_like(mu, np.inf)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weights": None}, "must have a callable 'weights' method"),
        ({"weights": lambda mu: np.ones(1)}, r"weights\(\) returned wrong shape"),
        ({"weights": lambda mu: np.full_like(mu, np.nan)}, r"weights\(\) returned non-finite"),
        (
            {"link": _UnboundedSlopeLog(), "weights": lambda mu: mu},
            r"link.deriv\(\) returned non-finite",
        ),
    ],
    ids=["missing-weights", "weights-shape", "weights-nan", "infinite-link-derivative"],
)
def test_validate_family_rejects_defective_weights_and_links(overrides, message):
    family = Poisson()
    for name, value in overrides.items():
        setattr(family, name, value)

    with pytest.raises(ValueError, match=message):
        validate_family(family)


def test_validate_family_evaluates_the_supplied_means():
    # A Poisson variance of zero at mu = 0 is only reached through explicit means.
    with pytest.raises(ValueError, match=r"variance\(\) must return positive"):
        validate_family(Poisson(), test_mu=np.array([0.0, 1.0]))
    assert validate_family(Binomial(link="cauchit"), test_mu=np.array([0.1, 0.5, 0.9])) is True


def test_quasi_family_scales_the_base_family_by_its_dispersion():
    base = Poisson()
    quasi = QuasiFamily(base, phi=2.0)
    mu = np.array([1.0, 2.0, 5.0])
    y = np.array([0.0, 3.0, 4.0])
    wt = np.array([1.0, 0.5, 2.0])

    assert quasi.link is base.link
    assert quasi.mean_bounds == base.mean_bounds
    assert_allclose(quasi.variance(mu), 2.0 * base.variance(mu), rtol=1e-15)
    assert_allclose(quasi.deviance_resids(y, mu, wt), base.deviance_resids(y, mu, wt) / 2.0)
    assert_allclose(quasi.weights(mu), base.weights(mu) / 2.0, rtol=1e-15)
    assert validate_family(quasi) is True


def test_custom_family_working_weights_follow_its_link_and_variance():
    class PowerVariance(CustomFamily):
        def __init__(self):
            self.link = LogLink()

        def variance(self, mu):
            return mu**1.5

        def deviance_resids(self, y, mu, wt):
            return 2 * wt * (y - mu)

    family = PowerVariance()
    mu = np.array([0.5, 1.0, 4.0])

    # 1 / (V(mu) g'(mu)^2) with g = log: 1 / (mu^1.5 / mu^2) = sqrt(mu).
    assert_allclose(family.weights(mu), np.sqrt(mu), rtol=1e-15)
    assert validate_family(family) is True
