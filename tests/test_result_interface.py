"""Interface shared by fitted linear and generalized linear mixed models."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import devcomp, families, glmer, lmer, load_cbpp, load_sleepstudy
from mixedlm.estimation.reml import _build_lambda
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

SLEEPSTUDY = load_sleepstudy()
CBPP = load_cbpp()
BASE_GETME_NAMES = [
    "X",
    "Z",
    "Zt",
    "y",
    "beta",
    "theta",
    "Lambda",
    "Lambdat",
    "u",
    "b",
    "n",
    "n_obs",
    "p",
    "n_fixed",
    "q",
    "n_random",
    "lower",
    "weights",
    "offset",
    "deviance",
    "fixef_names",
    "flist",
    "cnms",
    "Gp",
    "RX",
    "RZX",
    "Lind",
    "devcomp",
]


@pytest.fixture(scope="module")
def lmm():
    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="module")
def weighted_ml_lmm():
    weights = np.linspace(0.5, 2.0, len(SLEEPSTUDY))
    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, weights=weights, REML=False)


@pytest.fixture(scope="module")
def glmm():
    return glmer("incidence / size ~ period + (1 | herd)", CBPP, family=families.Binomial())


def _pls_spherical_modes(result, theta):
    """Solve the penalized least-squares system for u with dense algebra."""
    matrices = result.matrices
    Lambda = _build_lambda(theta, matrices.random_structures).toarray()
    sqrt_w = np.sqrt(matrices.weights)
    ZL = sqrt_w[:, None] * (matrices.Z.toarray() @ Lambda)
    residual = sqrt_w * (matrices.y - matrices.offset - matrices.X @ result.beta)
    return np.linalg.solve(ZL.T @ ZL + np.eye(matrices.n_random), ZL.T @ residual)


def _assert_same_components(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert_allclose(actual[name], value, rtol=1e-12, equal_nan=True, err_msg=name)


class TestGetME:
    @pytest.mark.parametrize("fixture", ["lmm", "glmm"])
    def test_base_names_are_shared(self, fixture, request) -> None:
        result = request.getfixturevalue(fixture)

        for name in BASE_GETME_NAMES:
            result.getME(name)
        assert result.getME("fixef_names") == list(result.fixef())
        with pytest.raises(ValueError, match="Valid names are: .*'fixef_names'"):
            result.getME("bogus")

    @pytest.mark.parametrize("fixture", ["lmm", "weighted_ml_lmm"])
    def test_u_is_the_spherical_pls_solution(self, fixture, request) -> None:
        result = request.getfixturevalue(fixture)
        u = result.getME("u")

        assert_allclose(u, _pls_spherical_modes(result, result.theta), rtol=1e-6, atol=1e-8)
        assert_allclose(result.getME("Lambda") @ u, result.getME("b"), rtol=1e-12)
        assert_allclose(result.getME("b"), result.u)
        assert u @ u == pytest.approx(result.getME("devcomp")["cmp"]["ussq"], rel=1e-12)
        assert not np.allclose(u, result.getME("b"))

    def test_u_has_no_component_in_the_null_space_of_a_singular_factor(self, lmm) -> None:
        # A rank-one slope factor: the slope effect is a multiple of the intercept.
        theta = np.array([0.8, 0.05, 0.0])
        u_pls = _pls_spherical_modes(lmm, theta)
        b = _build_lambda(theta, lmm.matrices.random_structures) @ u_pls
        singular = replace(lmm, theta=theta, u=b)

        assert singular.isSingular()
        assert_allclose(singular.getME("u"), u_pls, rtol=1e-10, atol=1e-10)

    def test_glmm_u_reproduces_the_laplace_deviance(self, glmm) -> None:
        u = glmm.getME("u")
        cmp = glmm.getME("devcomp")["cmp"]

        assert_allclose(glmm.getME("Lambda") @ u, glmm.getME("b"), rtol=1e-12)
        assert glmm.nAGQ == 1
        assert cmp["drsum"] + u @ u + cmp["ldL2"] == pytest.approx(glmm.deviance, rel=1e-10)

    @pytest.mark.parametrize("kind", ["lmer", "glmer"])
    def test_lind_indexes_theta_into_the_covariance_factor(self, kind) -> None:
        n = 24
        rows = np.arange(n)
        data = pd.DataFrame(
            {
                "y": rows % 2,
                "x": np.linspace(-1, 1, n),
                "z": np.cos(rows),
                "g": rows % 4,
                "h": rows % 3,
            }
        )
        formula = parse_formula("y ~ x + (x | g) + (z || h)")
        matrices = build_model_matrices(formula, data)
        theta = np.array([0.9, 0.2, 0.5, 0.7, 0.3])
        common = dict(
            formula=formula,
            matrices=matrices,
            theta=theta,
            beta=np.zeros(matrices.n_fixed),
            u=np.zeros(matrices.n_random),
            deviance=0.0,
            converged=True,
            n_iter=0,
        )
        if kind == "lmer":
            result = LmerResult(sigma=1.0, REML=True, **common)
        else:
            result = GlmerResult(family=families.Binomial(), nAGQ=1, **common)
        correlated = np.array([[0.9, 0.0], [0.2, 0.5]])
        expected = linalg.block_diag(*[correlated] * 4, *[np.diag([0.7, 0.3])] * 3)

        assert_array_equal(result.getME("Lambda").toarray(), expected)
        assert_array_equal(result.getME("Lambdat").toarray(), expected.T)
        # lme4's contract: Lambdat's stored entries, column by column, are theta[Lind].
        stored = result.getME("Lambdat").tocsc()
        stored.sort_indices()
        assert_array_equal(stored.data, theta[result.getME("Lind")])

    def test_rx_is_the_triangular_factor_of_the_fixed_effect_precision(self, lmm) -> None:
        RX = lmm.getME("RX")

        assert_array_equal(RX, np.triu(RX))
        assert np.all(np.diag(RX) > 0)
        assert_allclose(lmm.sigma**2 * np.linalg.inv(RX.T @ RX), lmm.vcov(), rtol=1e-10)
        assert lmm.getME("RZX").shape == (lmm.matrices.n_random, lmm.matrices.n_fixed)


class TestDevcomp:
    def test_lmm_components_reproduce_the_reml_fit(self, lmm) -> None:
        parts = lmm.getME("devcomp")
        cmp, dims = parts["cmp"], parts["dims"]
        n, p = lmm.matrices.n_obs, lmm.matrices.n_fixed
        RX = lmm.getME("RX")

        assert cmp["wrss"] == pytest.approx(np.sum(lmm.residuals() ** 2), rel=1e-12)
        assert cmp["pwrss"] == pytest.approx(cmp["wrss"] + cmp["ussq"], rel=1e-12)
        assert cmp["sigmaREML"] == pytest.approx(lmm.sigma, rel=1e-8)
        assert cmp["sigmaML"] == pytest.approx(lmm.sigma * np.sqrt((n - p) / n), rel=1e-8)
        assert cmp["ldRX2"] == pytest.approx(np.linalg.slogdet(RX.T @ RX)[1], rel=1e-10)
        assert cmp["REML"] == lmm.deviance
        assert np.isnan(cmp["dev"]) and np.isnan(cmp["drsum"])
        assert dims == {
            "n": 180,
            "p": 2,
            "q": 36,
            "nmp": 178,
            "nth": 3,
            "REML": 1,
            "useSc": 1,
            "nAGQ": 1,
            "q0": 36,
            "q1": 0,
            "qrx": 2,
            "ngrps": 1,
        }

    def test_weighted_ml_components_reproduce_the_fit(self, weighted_ml_lmm) -> None:
        result = weighted_ml_lmm
        cmp = result.getME("devcomp")["cmp"]
        n = result.matrices.n_obs
        weighted_rss = np.sum(result.matrices.weights * result.residuals() ** 2)
        ldRX2 = np.linalg.slogdet(result.getME("RX").T @ result.getME("RX"))[1]

        assert cmp["wrss"] == pytest.approx(weighted_rss, rel=1e-12)
        assert cmp["sigmaML"] == pytest.approx(result.sigma, rel=1e-8)
        assert np.sqrt(cmp["pwrss"] / n) == pytest.approx(result.sigma, rel=1e-8)
        assert cmp["dev"] == result.deviance
        assert cmp["ldRX2"] == pytest.approx(ldRX2, rel=1e-10)
        assert np.isnan(cmp["REML"])

    def test_glmm_components_are_computed_not_placeholders(self, glmm) -> None:
        cmp = glmm.getME("devcomp")["cmp"]
        dims = glmm.getME("devcomp")["dims"]
        pearson = glmm.residuals(type="pearson")
        RX = glmm.getME("RX")

        assert cmp["wrss"] == pytest.approx(np.sum(pearson**2), rel=1e-12)
        assert cmp["drsum"] == pytest.approx(np.sum(glmm.residuals() ** 2), rel=1e-10)
        assert cmp["pwrss"] == pytest.approx(cmp["wrss"] + cmp["ussq"], rel=1e-12)
        assert cmp["ldRX2"] == pytest.approx(np.linalg.slogdet(RX.T @ RX)[1], rel=1e-10)
        assert cmp["ldL2"] > 0 and cmp["dev"] == glmm.deviance
        assert all(np.isnan(cmp[name]) for name in ("REML", "sigmaML", "sigmaREML"))
        assert (dims["useSc"], dims["REML"], dims["nAGQ"], dims["ngrps"]) == (0, 0, 1, 1)

    def test_deviance_components_reassemble_the_reml_criterion(self, lmm) -> None:
        parts = lmm.get_deviance_components()
        cmp = lmm.getME("devcomp")["cmp"]
        df = lmm.matrices.n_obs - lmm.matrices.n_fixed

        assert parts.REML is True
        assert parts.total == pytest.approx(lmm.deviance, rel=1e-10)
        assert parts.sigma2 == pytest.approx(lmm.sigma**2, rel=1e-8)
        for name in ("ldL2", "ldRX2", "wrss", "ussq", "pwrss"):
            assert getattr(parts, name) == pytest.approx(cmp[name], rel=1e-8), name
        assert parts.total == pytest.approx(
            parts.ldL2 + parts.ldRX2 + df * (1 + np.log(2 * np.pi * parts.pwrss / df)), rel=1e-12
        )
        assert f"Total deviance:     {parts.total:.4f}" in str(parts)

    def test_split_terms_of_one_factor_count_one_grouping_factor(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", SLEEPSTUDY)

        assert result.getME("devcomp")["dims"]["ngrps"] == 1

    @pytest.mark.parametrize("fixture", ["lmm", "weighted_ml_lmm", "glmm"])
    def test_devcomp_function_matches_getme(self, fixture, request) -> None:
        result = request.getfixturevalue(fixture)
        expected = result.getME("devcomp")
        actual = devcomp(result)

        _assert_same_components(actual.cmp, expected["cmp"])
        assert actual.dims == expected["dims"]


class TestSharedAccessors:
    def test_glmer_weights_and_offset_accept_copy(self, glmm) -> None:
        assert glmm.weights(copy=False) is glmm.matrices.weights
        assert glmm.offset(copy=False) is glmm.matrices.offset
        weights = glmm.weights()
        assert weights is not glmm.matrices.weights
        assert_allclose(weights, CBPP["size"])

    @pytest.mark.parametrize(
        ("fixture", "expected"), [("lmm", 6), ("weighted_ml_lmm", 6), ("glmm", 5)]
    )
    def test_loglik_df_is_npar(self, fixture, expected, request) -> None:
        result = request.getfixturevalue(fixture)

        assert result.npar() == expected
        assert result.logLik().df == expected
        assert result.extractAIC() == (float(expected), result.AIC())

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("fe_params", lambda r: r.beta),
            ("re_params", lambda r: r.theta),
            ("fittedvalues", lambda r: r.fitted()),
            ("resid", lambda r: r.residuals()),
        ],
    )
    def test_statsmodels_aliases_are_deprecated(self, lmm, name, expected) -> None:
        with pytest.warns(DeprecationWarning, match=f"LmerResult.{name} is deprecated"):
            value = getattr(lmm, name)

        assert_allclose(value, expected(lmm))
