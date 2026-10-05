from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, lmer, load_sleepstudy, parse_formula, set_cov_type
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from tests._lmm_oracles import observation_likelihood

Result = LmerResult | GlmerResult


@pytest.fixture
def structured_data() -> pd.DataFrame:
    n_groups = 6
    n_per_group = 4
    n = n_groups * n_per_group
    return pd.DataFrame(
        {
            "y": np.arange(n) % 2,
            "x": np.tile([-1.0, -0.25, 0.5, 1.25], n_groups),
            "z": np.tile([0.0, 1.0, -1.0, 0.5], n_groups),
            "group": np.repeat([f"g{i}" for i in range(n_groups)], n_per_group),
        }
    )


@pytest.fixture(params=[LmerResult, GlmerResult], ids=["lmer", "glmer"])
def result_factory(
    request: pytest.FixtureRequest,
    structured_data: pd.DataFrame,
) -> Callable[[str, NDArray[np.floating]], Result]:
    result_type = request.param

    def make_result(cov_type: str, theta: NDArray[np.floating]) -> Result:
        formula = set_cov_type("y ~ x + z + (x + z | group)", cov_type)
        matrices = build_model_matrices(formula, structured_data)
        common = {
            "formula": formula,
            "matrices": matrices,
            "theta": np.asarray(theta, dtype=np.float64),
            "beta": np.zeros(matrices.n_fixed),
            "u": np.zeros(matrices.n_random),
            "deviance": 0.0,
            "converged": True,
            "n_iter": 0,
        }
        if result_type is LmerResult:
            return LmerResult(sigma=2.0, REML=True, **common)
        return GlmerResult(family=families.Binomial(), nAGQ=1, **common)

    return make_result


def _expected_correlation(cov_type: str, rho: float, q: int) -> NDArray[np.floating]:
    if cov_type == "cs":
        result = np.full((q, q), rho, dtype=np.float64)
        np.fill_diagonal(result, 1.0)
        return result
    indices = np.arange(q)
    return rho ** np.abs(indices[:, None] - indices[None, :])


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_varcorr_uses_fitted_covariance_factor(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
) -> None:
    relative_sd = 1.5
    rho = 0.4
    result = result_factory(cov_type, np.array([relative_sd, rho]))
    scale = result.sigma**2 if isinstance(result, LmerResult) else 1.0
    expected_corr = _expected_correlation(cov_type, rho, q=3)
    expected_cov = relative_sd**2 * scale * expected_corr

    group = result.VarCorr().groups["group"]

    np.testing.assert_allclose(group.cov, expected_cov)
    np.testing.assert_allclose(group.corr, expected_corr)
    np.testing.assert_allclose(list(group.variance.values()), np.diag(expected_cov))


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_varcorr_reports_correlation_for_double_bar_formula(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
) -> None:
    result = result_factory(cov_type, np.array([1.25, 0.3]))
    result.matrices.random_structures[0].correlated = False

    corr = result.VarCorr().groups["group"].corr

    np.testing.assert_allclose(corr, _expected_correlation(cov_type, 0.3, q=3))


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_repca_matches_reported_covariance(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
) -> None:
    result = result_factory(cov_type, np.array([1.25, 0.35]))
    cov = result.VarCorr().groups["group"].cov
    expected_eigenvalues = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)

    pca = result.rePCA().groups["group"]

    np.testing.assert_allclose(pca.sdev, np.sqrt(expected_eigenvalues))
    np.testing.assert_allclose(pca.proportion, expected_eigenvalues / expected_eigenvalues.sum())
    np.testing.assert_allclose(pca.cumulative, np.cumsum(pca.proportion))


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_singularity_uses_covariance_eigenvalues(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
) -> None:
    result = result_factory(cov_type, np.array([1.5, 0.25]))
    assert not result.isSingular()

    result.theta = np.array([1.5, 1.0 - 1e-10])
    assert result.isSingular()

    result.theta = np.array([0.0, 0.0])
    assert result.isSingular()


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="getME('Lind') assumes one theta per Lambda entry, so cs/ar1 indices run past theta",
)
@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_lind_is_rejected_for_nonlinear_covariance_maps(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
) -> None:
    result = result_factory(cov_type, np.array([1.5, 0.4]))
    factor = result.getME("Lambda").toarray()[:3, :3]

    # Products such as sd * rho fill the factor, so no index map into
    # theta = (sd, rho) reproduces Lambda as lme4's Lind contract requires.
    assert_allclose(factor @ factor.T, 1.5**2 * _expected_correlation(cov_type, 0.4, q=3))
    with pytest.raises((ValueError, NotImplementedError), match="Lind"):
        result.getME("Lind")


def test_singularity_tolerance_validation(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
) -> None:
    result = result_factory("cs", np.array([1.0, 0.25]))

    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite, non-negative"):
            result.isSingular(invalid)


@pytest.mark.parametrize(
    ("cov_type", "expected_correlation_lower"),
    [("cs", -0.5 + 1e-6), ("ar1", -1.0 + 1e-6)],
)
def test_structured_lower_bounds_match_optimizer_constraints(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    cov_type: str,
    expected_correlation_lower: float,
) -> None:
    result = result_factory(cov_type, np.array([1.0, 0.25]))

    lower = result.getME("lower")

    assert len(lower) == len(result.theta)
    np.testing.assert_allclose(lower, [0.0, expected_correlation_lower])


def test_unstructured_covariance_results_are_unchanged(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
) -> None:
    theta = np.array([1.0, 0.2, 0.8, -0.1, 0.3, 0.6])
    result = result_factory("us", theta)
    factor = result.getME("Lambda")[:3, :3].toarray()
    scale = result.sigma**2 if isinstance(result, LmerResult) else 1.0

    np.testing.assert_allclose(result.VarCorr().groups["group"].cov, factor @ factor.T * scale)
    np.testing.assert_allclose(result.getME("lower"), [0.0, -np.inf, 0.0, -np.inf, -np.inf, 0.0])


def test_covariance_reporting_does_not_expand_group_level_factors(
    result_factory: Callable[[str, NDArray[np.floating]], Result],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mixedlm.estimation import reml

    result = result_factory("ar1", np.array([1.25, 0.3]))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("full group-level covariance factor should not be built")

    monkeypatch.setattr(reml, "_build_lambda", fail_if_called)

    cov = result.VarCorr().groups["group"].cov

    assert cov.shape == (3, 3)


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_set_cov_type_marks_the_random_term(cov_type: str) -> None:
    formula = set_cov_type(parse_formula("Reaction ~ Days + (Days | Subject)"), cov_type)

    assert [(term.grouping, term.cov_type) for term in formula.random] == [("Subject", cov_type)]


def test_set_cov_type_maps_each_grouping_factor() -> None:
    formula = set_cov_type(
        parse_formula("y ~ x + (1 | group1) + (1 | group2)"), {"group1": "cs", "group2": "ar1"}
    )

    assert {term.grouping: term.cov_type for term in formula.random} == {
        "group1": "cs",
        "group2": "ar1",
    }


def test_set_cov_type_rejects_unknown_structures() -> None:
    with pytest.raises(ValueError, match="Invalid cov_type"):
        set_cov_type(parse_formula("y ~ x + (1 | group)"), "invalid")


def test_two_term_structures_fit_one_nested_model() -> None:
    data = load_sleepstudy()
    formula = "Reaction ~ Days + (Days | Subject)"
    unstructured = lmer(formula, data)
    compound = lmer(set_cov_type(formula, "cs"), data)
    autoregressive = lmer(set_cov_type(formula, "ar1"), data)

    assert compound.converged and autoregressive.converged
    assert (len(unstructured.theta), len(compound.theta)) == (3, 2)
    # With two terms both structures are one shared variance and one correlation.
    assert autoregressive.deviance == pytest.approx(compound.deviance, rel=1e-9)
    assert_allclose(autoregressive.theta, compound.theta, rtol=1e-5)
    assert compound.deviance > unstructured.deviance
    assert compound.deviance == pytest.approx(
        observation_likelihood(compound.matrices, compound.theta, reml=True), rel=1e-10
    )
