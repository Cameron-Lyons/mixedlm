from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import vcconv, vec2STlist
from mixedlm.matrices.design import RandomEffectStructure
from numpy.testing import assert_allclose, assert_array_equal


def _structure(cov_type="us", n_terms=4, correlated=True, group="group"):
    return RandomEffectStructure(
        grouping_factor=group,
        term_names=[f"term{i}" for i in range(n_terms)],
        n_levels=5,
        n_terms=n_terms,
        correlated=correlated,
        level_map={},
        cov_type=cov_type,
    )


def _parameters(cov_type, n_terms=4):
    if cov_type == "us":
        factor = np.array(
            [[1.0, 0, 0, 0], [0.2, 0.8, 0, 0], [-0.3, 0.4, 0.6, 0], [0.1, -0.2, 0.5, 0.9]]
        )[:n_terms, :n_terms]
        return factor[np.tril_indices(n_terms)], factor @ factor.T
    rho = -0.2
    covariance = (
        np.full((n_terms, n_terms), rho)
        if cov_type == "cs"
        else rho ** np.abs(np.arange(n_terms)[:, None] - np.arange(n_terms))
    )
    np.fill_diagonal(covariance, 1.0)
    return np.array([1.5, rho]) if n_terms > 1 else np.array([1.5]), 1.5**2 * covariance


@pytest.mark.parametrize("cov_type", ["us", "cs", "ar1"])
@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.5])
@pytest.mark.parametrize("n_terms", [1, 3, 4])
def test_conversions_match_covariance_and_original_parameter_order(cov_type, sigma, n_terms):
    structure = _structure(cov_type, n_terms)
    theta, covariance = _parameters(cov_type, n_terms)
    covariance *= sigma**2
    expected_sd = np.sqrt(np.diag(covariance))
    off_diagonal = np.triu_indices(n_terms, k=1)
    denominator = np.outer(expected_sd, expected_sd)
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator != 0
    )

    varcov = vcconv(theta, [structure], sigma, to="varcov")["group"]
    sdcorr = vcconv(theta, [structure], sigma, to="sdcorr")["group"]
    original = vcconv(theta, [structure], sigma, to="theta")["group"]

    assert_allclose(varcov["var"], np.diag(covariance), atol=1e-15)
    assert_allclose(varcov["cov"], covariance[off_diagonal], atol=1e-15)
    assert_allclose(sdcorr["sd"], expected_sd, atol=1e-15)
    assert_allclose(sdcorr["corr"], correlation[off_diagonal], atol=1e-15)
    assert_array_equal(original["theta"], theta)
    assert varcov["terms"] == sdcorr["terms"] == original["terms"] == structure.term_names


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_covariance_takes_precedence_over_double_bar(cov_type):
    theta, _ = _parameters(cov_type)
    structure = _structure(cov_type)

    for target in ("sdcorr", "varcov", "theta"):
        assert vcconv(theta, [replace(structure, correlated=False)], to=target) == vcconv(
            theta, [structure], to=target
        )


@pytest.mark.parametrize("target", ["sdcorr", "varcov", "theta"])
def test_mixed_structures_keep_parameter_boundaries(target):
    first_theta, _ = _parameters("cs")
    second_theta = np.array([-0.4, 0.0, 0.7])
    third_theta, _ = _parameters("ar1", 1)
    fourth_theta, _ = _parameters("us")
    structures = [
        _structure("cs", group="first"),
        _structure(n_terms=3, correlated=False, group="second"),
        _structure("ar1", n_terms=1, group="third"),
        _structure("us", group="fourth"),
    ]
    blocks = [first_theta, second_theta, third_theta, fourth_theta]
    original = np.concatenate(blocks)

    combined = vcconv(original, structures, sigma=1.7, to=target)

    assert list(combined) == ["first", "second", "third", "fourth"]
    for structure, block in zip(structures, blocks, strict=True):
        assert (
            combined[structure.grouping_factor]
            == vcconv(block, [structure], sigma=1.7, to=target)[structure.grouping_factor]
        )
    assert_array_equal(original, np.concatenate(blocks))


@pytest.mark.parametrize("target", ["sdcorr", "varcov"])
def test_independent_variances_use_linear_memory(monkeypatch, target):
    from mixedlm.estimation import reml

    def reject_dense_factor(*args, **kwargs):
        raise AssertionError("independent variances do not need a covariance factor")

    monkeypatch.setattr(reml, "_build_lambda_blocks", reject_dense_factor)
    structure = _structure(n_terms=10_000, correlated=False)
    theta = np.resize(np.array([-0.4, 0.0, 0.7]), structure.n_terms)

    actual = vcconv(theta, [structure], sigma=2.0, to=target)["group"]

    if target == "sdcorr":
        assert_allclose(actual["sd"], 2.0 * np.abs(theta))
        assert actual["corr"] == []
    else:
        assert_allclose(actual["var"], 4.0 * theta**2)
        assert actual["cov"] == []


def test_theta_passthrough_avoids_covariance_work_and_returns_owned_lists(monkeypatch):
    from mixedlm.estimation import reml

    def reject_factor(*args, **kwargs):
        raise AssertionError("theta passthrough does not need a covariance factor")

    monkeypatch.setattr(reml, "_build_lambda_blocks", reject_factor)
    structure = _structure()
    theta, _ = _parameters("us")
    actual = vcconv(theta, [structure], sigma=0.0, to="theta")["group"]
    actual["theta"][0] = 99.0
    actual["terms"][0] = "changed"

    assert theta[0] == 1.0
    assert structure.term_names[0] == "term0"


def test_singular_unstructured_factor_has_finite_zero_correlations():
    structure = _structure(n_terms=3)
    theta = np.array([0.0, 0.0, 1.0, 0.0, -0.2, 0.3])

    actual = vcconv(theta, [structure], sigma=2.0)["group"]

    assert_allclose(actual["sd"], [0.0, 2.0, 2 * np.sqrt(0.13)])
    assert_allclose(actual["corr"], [0.0, 0.0, -0.2 / np.sqrt(0.13)])


@pytest.mark.parametrize("target", ["sdcorr", "varcov", "theta"])
def test_empty_structures_return_empty_dictionary(target):
    assert vcconv(np.array([]), [], to=target) == {}


@pytest.mark.parametrize("sigma", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_sigma_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
        vcconv(np.array([1.0]), [_structure(n_terms=1)], sigma=sigma)


@pytest.mark.parametrize("theta", [[], [1.0, 2.0], [[1.0]], 1.0])
def test_wrong_theta_shape_is_rejected(theta):
    with pytest.raises(ValueError, match="one-dimensional with exactly 1 values"):
        vcconv(theta, [_structure(n_terms=1)])


@pytest.mark.parametrize("theta", [[np.nan], [np.inf], [-np.inf]])
def test_nonfinite_theta_is_rejected(theta):
    with pytest.raises(ValueError, match="theta must contain only finite values"):
        vcconv(theta, [_structure(n_terms=1)])


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="to must be"):
        vcconv(np.array([1.0]), [_structure(n_terms=1)], to="variance")


def test_vec2stlist_matches_fitted_theta_row_order():
    theta, covariance = _parameters("us")

    factor, intercept = vec2STlist(np.concatenate([theta, [1.5]]), [4, 1])

    assert_allclose(factor @ factor.T, covariance)
    assert_array_equal(factor[np.tril_indices(4)], theta)
    assert_array_equal(intercept, [[1.5]])
