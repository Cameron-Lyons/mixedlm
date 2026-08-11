from __future__ import annotations

import warnings

import numpy as np
import pytest
from mixedlm.families import Link, resolve_link
from mixedlm.families.base import LogitLink


def test_inverse_logit_handles_extreme_predictors_without_warnings() -> None:
    eta = np.array([-1000.0, 0.0, 1000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = LogitLink().inverse(eta)

    np.testing.assert_array_equal(probabilities, np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("identity", np.array([-2.0, 0.0, 3.0])),
        ("log", np.array([0.2, 1.0, 5.0])),
        ("logit", np.array([0.1, 0.5, 0.9])),
        ("probit", np.array([0.1, 0.5, 0.9])),
        ("cloglog", np.array([0.1, 0.5, 0.9])),
        ("cauchit", np.array([0.1, 0.5, 0.9])),
        ("inverse", np.array([0.2, 1.0, 5.0])),
        ("sqrt", np.array([0.2, 1.0, 5.0])),
        ("1/mu^2", np.array([0.2, 1.0, 5.0])),
    ],
)
def test_named_links_round_trip(name: str, values: np.ndarray) -> None:
    link = resolve_link(name, default="identity")

    assert link.name == name
    assert isinstance(link, Link)
    np.testing.assert_allclose(link.inverse(link(values)), values)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("complementary log-log", "cloglog"),
        ("comploglog", "cloglog"),
        ("inverse_squared", "1/mu^2"),
        ("1/mu2", "1/mu^2"),
    ],
)
def test_link_aliases(alias: str, expected: str) -> None:
    assert resolve_link(alias, default="identity").name == expected


def test_cloglog_inverse_handles_extreme_predictors_without_warnings() -> None:
    eta = np.array([-1000.0, 0.0, 1000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probabilities = resolve_link("cloglog", default="logit").inverse(eta)

    np.testing.assert_array_equal(probabilities, np.array([0.0, 1 - np.exp(-1), 1.0]))


def test_resolve_link_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown link"):
        resolve_link("missing", default="identity")

    with pytest.raises(TypeError, match="string, Link instance, or None"):
        resolve_link(42, default="identity")  # type: ignore[arg-type]
