from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer
from mixedlm.estimation import laplace
from mixedlm.estimation.laplace import pirls
from mixedlm.families import (
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    ProbitLink,
    QuasiFamily,
)
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices import build_model_matrices

from tests._lmer_data import CBPP


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


@pytest.mark.parametrize(
    ("family", "expected_link"),
    [
        (Gaussian(), "identity"),
        (Binomial(), "logit"),
        (Poisson(), "log"),
        (Gamma(), "log"),
        (InverseGaussian(), "log"),
        (NegativeBinomial(), "log"),
        (Gaussian(link="log"), "log"),
        (Binomial(link="probit"), "probit"),
        (Poisson(link="sqrt"), "sqrt"),
        (Gamma(link="inverse"), "inverse"),
        (InverseGaussian(link="1/mu^2"), "1/mu^2"),
    ],
)
def test_family_link_configuration(family, expected_link: str) -> None:
    assert family.link.name == expected_link
    assert f"link={expected_link}" in repr(family)


def test_family_accepts_link_instances() -> None:
    link = ProbitLink()

    family = Binomial(link=link)

    assert family.link is link


@pytest.mark.parametrize(
    ("factory", "link"),
    [
        (Gaussian, "probit"),
        (Binomial, "sqrt"),
        (Poisson, "logit"),
        (Gamma, "probit"),
        (InverseGaussian, "cauchit"),
        (NegativeBinomial, "identity"),
    ],
)
def test_family_rejects_incompatible_named_links(factory, link: str) -> None:
    with pytest.raises(ValueError, match="not supported"):
        factory(link=link)


@pytest.mark.parametrize("theta", [0.0, -1.0, np.inf, np.nan])
def test_negative_binomial_rejects_invalid_theta(theta: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        NegativeBinomial(theta=theta)


@pytest.mark.parametrize("phi", [0.0, -1.0, np.inf, np.nan])
def test_quasi_family_rejects_invalid_dispersion(phi: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        QuasiFamily(Poisson(), phi=phi)


def test_documented_family_helpers() -> None:
    family = Binomial(link="probit")
    mu = np.array([0.2, 0.8])
    y = np.array([0.0, 1.0])

    eta = family.link(mu)

    np.testing.assert_allclose(family.linkinv(eta), mu)
    np.testing.assert_allclose(
        family.deviance_residuals(y, mu),
        family.deviance_resids(y, mu, np.ones_like(y)),
    )


@pytest.mark.parametrize("link", ["probit", "cloglog", "cauchit"])
def test_binomial_alternative_links_fit_end_to_end(link: str) -> None:
    result = glmer(
        "y ~ period + (1 | herd)",
        CBPP,
        family=families.Binomial(link=link),
        weights=CBPP["size"].to_numpy(),
    )

    assert result.converged
    assert result.family.link.name == link
    assert np.isfinite(result.deviance)
    assert np.all((result.fitted() > 0) & (result.fitted() < 1))


def test_positive_family_canonical_links_fit_end_to_end() -> None:
    rng = np.random.default_rng(123)
    n_groups = 8
    observations_per_group = 12
    n = n_groups * observations_per_group
    group = np.repeat(np.arange(n_groups), observations_per_group).astype(str)
    x = rng.uniform(0.5, 1.5, n)
    mu = np.exp(0.5 + 0.2 * x)

    cases = [
        (Gamma(link="inverse"), rng.gamma(10.0, mu / 10.0, n)),
        (InverseGaussian(link="1/mu^2"), rng.wald(mu, 10.0, n)),
    ]

    for family, y in cases:
        data = pd.DataFrame({"y": y, "x": x, "group": group})
        result = glmer("y ~ x + (1 | group)", data, family=family)

        assert result.converged
        assert np.isfinite(result.deviance)
        assert np.all(result.fitted() > 0)


@pytest.mark.parametrize("link", ["identity", "sqrt"])
def test_poisson_alternative_links_fit_end_to_end(link: str) -> None:
    data = pd.DataFrame(
        {
            "y": np.full(40, 5.0),
            "group": np.repeat(np.arange(8), 5).astype(str),
        }
    )

    result = glmer("y ~ 1 + (1 | group)", data, family=Poisson(link=link))

    assert result.converged
    np.testing.assert_allclose(result.fitted(), 5.0, rtol=1e-6)


@pytest.mark.parametrize("family", [Binomial(link="probit"), Poisson(link="sqrt")])
def test_non_native_links_use_python_backend(monkeypatch, family) -> None:
    expected = (1.0, np.array([2.0]), np.array([3.0]))

    monkeypatch.setattr(laplace, "_HAS_RUST", True)
    monkeypatch.setattr(
        laplace,
        "_rust_laplace_deviance",
        lambda *args: pytest.fail("non-native link was sent to the Rust backend"),
    )
    monkeypatch.setattr(laplace, "laplace_deviance", lambda *args: expected)

    actual = laplace.laplace_deviance_fast(np.array([0.5]), object(), family)

    assert actual is expected
