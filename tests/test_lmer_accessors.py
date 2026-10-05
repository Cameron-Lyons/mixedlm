"""Fitted-model accessors checked against lme4's published sleepstudy and CBPP fits.

Published values: https://lme4.github.io/lme4/reference/lmer.html and Bates et al.
(2015), "Fitting Linear Mixed-Effects Models Using lme4", J. Stat. Softw. 67(1).
"""

from copy import copy
from dataclasses import replace
from pickle import dumps, loads

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    AllFitResult,
    RanefResult,
    allFit,
    families,
    glmer,
    lmer,
    nlme,
    nlmer,
    parse_formula,
)
from mixedlm.families.base import LogitLink, LogLink
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from scipy.special import expit

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY, grouped_data


def cbpp_working_weights(model):
    """Binomial IRLS weights at the conditional modes of a CBPP fit."""
    eta = model.getME("X") @ model.beta + model.getME("Z") @ model.getME("b")
    mu = expit(eta)
    return mu, CBPP["size"].to_numpy() * mu * (1 - mu)


@pytest.fixture(scope="module")
def crossed_lmm():
    rng = np.random.default_rng(42)
    group1 = np.repeat(np.arange(10), 20)
    group2 = np.tile(np.arange(5), 40)
    x = rng.standard_normal(200)
    effects1, effects2 = rng.normal(0.0, 0.5, 10), rng.normal(0.0, 0.5, 5)
    y = 2.0 + 1.5 * x + effects1[group1] + effects2[group2] + rng.normal(0.0, 0.5, 200)
    data = pd.DataFrame({"y": y, "x": x, "g1": group1.astype(str), "g2": group2.astype(str)})
    return lmer("y ~ x + (1 | g1) + (1 | g2)", data)


@pytest.fixture(scope="module")
def logistic_nlmm():
    rng = np.random.default_rng(42)
    index = np.repeat(np.arange(5), 20)
    x = np.tile(np.linspace(0, 10, 20), 5)
    asym = 200 + rng.normal(0.0, 10.0, 5)
    xmid = 5 + rng.normal(0.0, 0.5, 5)
    y = asym[index] / (1 + np.exp(xmid[index] - x)) + rng.normal(0.0, 5.0, len(x))
    data = pd.DataFrame({"y": y, "x": x, "group": [f"g{i}" for i in index]})
    return nlmer(
        model=nlme.SSlogis(),
        data=data,
        x_var="x",
        y_var="y",
        group_var="group",
        random_params=[0, 1],
        start={"Asym": 200, "xmid": 5, "scal": 1},
    )


class TestSizesAndParameterCounts:
    def test_linear_model(self, grouped_lmm) -> None:
        assert grouped_lmm.nobs() == 200
        assert grouped_lmm.ngrps() == {"group": 10}
        # Two fixed effects, one relative SD and the residual SD.
        assert grouped_lmm.npar() == 4
        assert grouped_lmm.df_residual() == 198
        assert grouped_lmm.get_sigma() == grouped_lmm.sigma

    def test_correlated_random_slopes(self, sleepstudy_slopes_lmm) -> None:
        assert sleepstudy_slopes_lmm.npar() == 2 + 3 + 1
        assert sleepstudy_slopes_lmm.df_residual() == 178

    def test_crossed_grouping_factors(self, crossed_lmm) -> None:
        assert crossed_lmm.ngrps() == {"g1": 10, "g2": 5}
        assert crossed_lmm.npar() == 2 + 2 + 1

    def test_generalized_models_have_unit_scale(self, cbpp_glmm, grouped_glmm) -> None:
        assert cbpp_glmm.nobs() == 56
        assert cbpp_glmm.ngrps() == {"herd": 15}
        assert cbpp_glmm.npar() == 4 + 1
        assert cbpp_glmm.df_residual() == 52
        assert grouped_glmm.df_residual() == 198
        for model in (cbpp_glmm, grouped_glmm):
            assert model.sigma == model.get_sigma() == 1.0

    def test_nonlinear_model(self, logistic_nlmm) -> None:
        assert logistic_nlmm.nobs() == 100
        # Three fixed parameters, a 2x2 covariance factor and the residual SD.
        assert logistic_nlmm.npar() == 3 + 3 + 1
        assert logistic_nlmm.df_residual() == 97

    def test_model_type_predicates(self, sleepstudy_lmm, cbpp_glmm, logistic_nlmm) -> None:
        predicates = [
            (model.isLMM(), model.isGLMM(), model.isNLMM())
            for model in (sleepstudy_lmm, cbpp_glmm, logistic_nlmm)
        ]
        assert predicates == [(True, False, False), (False, True, False), (False, False, True)]


class TestModelFrame:
    def test_holds_the_model_variables(self, sleepstudy_lmm) -> None:
        frame = sleepstudy_lmm.model_frame()

        assert sorted(frame.columns) == ["Days", "Reaction", "Subject"]
        pd.testing.assert_frame_equal(frame[list(SLEEPSTUDY.columns)], SLEEPSTUDY)

    def test_omits_incomplete_rows(self) -> None:
        data = SLEEPSTUDY.copy()
        data.loc[0, "Reaction"] = np.nan
        data.loc[5, "Days"] = np.nan

        frame = lmer("Reaction ~ Days + (1 | Subject)", data, na_action="omit").model_frame()

        expected = data.dropna().reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[list(data.columns)], expected)

    def test_lists_each_variable_once(self, crossed_lmm, cbpp_glmm) -> None:
        data = grouped_data().assign(x2=np.linspace(-1.0, 1.0, 200))
        interaction = lmer("y ~ x * x2 + (1 | group)", data)

        assert sorted(interaction.model_frame().columns) == ["group", "x", "x2", "y"]
        assert sorted(crossed_lmm.model_frame().columns) == ["g1", "g2", "x", "y"]
        assert sorted(cbpp_glmm.model_frame().columns) == ["herd", "incidence", "period", "size"]


class TestWeightsOffsetAndFamily:
    def test_defaults_are_unit_weights_and_zero_offset(self, sleepstudy_lmm) -> None:
        assert_array_equal(sleepstudy_lmm.weights(), np.ones(180))
        assert_array_equal(sleepstudy_lmm.offset(), np.zeros(180))

    def test_supplied_values_are_returned(self) -> None:
        rng = np.random.default_rng(3)
        weights = rng.uniform(0.5, 1.5, 180)
        offset = rng.standard_normal(180)

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, weights=weights, offset=offset)

        assert_array_equal(result.weights(), weights)
        assert_array_equal(result.offset(), offset)

    def test_accessors_return_copies(self, sleepstudy_lmm) -> None:
        weights, offset = sleepstudy_lmm.weights(), sleepstudy_lmm.offset()
        weights[0] = offset[0] = 999.0

        assert sleepstudy_lmm.weights()[0] == 1.0
        assert sleepstudy_lmm.offset()[0] == 0.0

    def test_grouped_binomial_trials_are_prior_weights(self, cbpp_glmm) -> None:
        assert_array_equal(cbpp_glmm.weights(), CBPP["size"])

    def test_generalized_offset(self) -> None:
        offset = np.log(CBPP["size"].to_numpy())

        result = glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), offset=offset)

        assert_array_equal(result.offset(), offset)

    def test_family_and_link(self, cbpp_glmm) -> None:
        poisson = glmer("y ~ x + (1 | group)", grouped_data("poisson"), family=families.Poisson())

        assert isinstance(cbpp_glmm.get_family(), families.Binomial)
        assert isinstance(cbpp_glmm.get_family().link, LogitLink)
        assert isinstance(poisson.get_family(), families.Poisson)
        assert isinstance(poisson.get_family().link, LogLink)


class TestCondVar:
    def test_random_intercepts_match_the_conditional_variance_formula(self, sleepstudy_lmm) -> None:
        theta, sigma = sleepstudy_lmm.theta[0], sleepstudy_lmm.sigma
        Z = sleepstudy_lmm.getME("Z").toarray()
        expected = sigma**2 * theta**2 * np.diag(np.linalg.inv(np.eye(18) + theta**2 * Z.T @ Z))

        ranefs = sleepstudy_lmm.ranef(condVar=True)

        assert isinstance(ranefs, RanefResult)
        assert_allclose(ranefs.condVar["Subject"]["(Intercept)"], expected)
        assert_allclose(
            ranefs["Subject"]["(Intercept)"],
            sleepstudy_lmm.ranef(condVar=False)["Subject"]["(Intercept)"],
        )

    def test_random_slopes_match_the_conditional_variance_formula(
        self, sleepstudy_slopes_lmm
    ) -> None:
        Lambda = sleepstudy_slopes_lmm.getME("Lambda").toarray()
        Z = sleepstudy_slopes_lmm.getME("Z").toarray()
        precision = np.eye(36) + Lambda.T @ Z.T @ Z @ Lambda
        covariance = sleepstudy_slopes_lmm.sigma**2 * Lambda @ np.linalg.inv(precision) @ Lambda.T

        cond_var = sleepstudy_slopes_lmm.ranef(condVar=True).condVar["Subject"]

        assert_allclose(cond_var["(Intercept)"], np.diag(covariance)[0::2])
        assert_allclose(cond_var["Days"], np.diag(covariance)[1::2])

    def test_glmm_matches_the_laplace_conditional_variance(self, cbpp_glmm) -> None:
        theta = cbpp_glmm.theta[0]
        Z = cbpp_glmm.getME("Z").toarray()
        _, weights = cbpp_working_weights(cbpp_glmm)
        precision = np.eye(15) + theta**2 * Z.T @ (weights[:, None] * Z)

        ranefs = cbpp_glmm.ranef(condVar=True)

        assert_allclose(ranefs["herd"]["(Intercept)"], cbpp_glmm.getME("b"))
        assert np.all(ranefs["herd"]["(Intercept)"] != 0)
        assert_allclose(
            ranefs.condVar["herd"]["(Intercept)"], theta**2 * np.diag(np.linalg.inv(precision))
        )

    def test_ranef_result_is_dict_like(self, sleepstudy_lmm) -> None:
        ranefs = sleepstudy_lmm.ranef(condVar=True)

        assert "Subject" in ranefs
        assert list(ranefs.keys()) == ["Subject"]
        assert [(group, list(terms)) for group, terms in ranefs.items()] == [
            ("Subject", ["(Intercept)"])
        ]


class TestDrop1:
    def test_matches_an_independent_reduced_fit(self) -> None:
        full = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        reduced = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, REML=False)

        result = full.drop1(data=SLEEPSTUDY)

        lrt = reduced.deviance - full.deviance
        assert result.terms == ["Days"]
        assert result.lrt[0] == pytest.approx(lrt)
        assert result.p_value[0] == pytest.approx(stats.chi2.sf(lrt, 1))
        assert result.aic[0] == pytest.approx(reduced.AIC())
        assert result.full_model_aic == pytest.approx(full.AIC())

    def test_drops_each_term_separately(self) -> None:
        data = grouped_data().assign(x2=np.cos(np.arange(200.0)))
        full = lmer("y ~ x + x2 + (1 | group)", data, REML=False)

        result = full.drop1(data=data)

        assert result.terms == ["x", "x2"]
        for term, kept, lrt in zip(result.terms, ["x2", "x"], result.lrt, strict=True):
            reduced = lmer(f"y ~ {kept} + (1 | group)", data, REML=False)
            assert lrt == pytest.approx(reduced.deviance - full.deviance), term

    def test_glmm_factor_term(self, cbpp_glmm) -> None:
        reduced = glmer("incidence / size ~ 1 + (1 | herd)", CBPP, family=families.Binomial())

        result = cbpp_glmm.drop1(data=CBPP)

        lrt = 2 * (cbpp_glmm.logLik().value - reduced.logLik().value)
        assert result.terms == ["period"]
        assert result.lrt[0] == pytest.approx(lrt, abs=1e-6)
        assert result.p_value[0] == pytest.approx(stats.chi2.sf(lrt, 3), rel=1e-5)

    def test_function_matches_method_and_prints_a_table(self) -> None:
        from mixedlm.inference import drop1_lmer

        model = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        result = drop1_lmer(model, data=SLEEPSTUDY)

        assert result.lrt == pytest.approx(model.drop1(data=SLEEPSTUDY).lrt)
        output = str(result)
        for text in ("Single term deletions", "AIC", "LRT", "Days"):
            assert text in output


class TestIsSingular:
    def test_interior_fits_are_not_singular(self, sleepstudy_lmm, grouped_lmm, cbpp_glmm) -> None:
        uncorrelated = lmer("Reaction ~ Days + (Days || Subject)", SLEEPSTUDY)

        for model in (sleepstudy_lmm, grouped_lmm, cbpp_glmm, uncorrelated):
            assert model.isSingular() is False
            assert model.is_singular() is False
        assert grouped_lmm.isSingular(tol=0.01) is False

    def test_tolerance_is_a_lower_bound_on_theta(self, sleepstudy_lmm, cbpp_glmm) -> None:
        assert sleepstudy_lmm.isSingular(tol=1e10) is True
        assert cbpp_glmm.isSingular(tol=1e10) is True

    def test_zero_between_group_variance_is_singular(self) -> None:
        # Every group has the same responses, so the group variance is zero.
        rng = np.random.default_rng(42)
        x = np.tile(np.linspace(-1, 1, 20), 5)
        y = 2.0 + 1.5 * x + np.tile(rng.standard_normal(20), 5)
        data = pd.DataFrame({"y": y, "x": x, "group": np.repeat(list("abcde"), 20)})

        with pytest.warns(UserWarning, match="Model is singular"):
            result = lmer("y ~ x + (1 | group)", data)

        assert_array_equal(result.theta, [0.0])
        assert result.isSingular() is True

    def test_bernoulli_coded_cbpp_proportions_are_singular(self, singular_cbpp_glmm) -> None:
        assert_array_equal(singular_cbpp_glmm.theta, [0.0])
        assert singular_cbpp_glmm.isSingular() is True
        assert singular_cbpp_glmm.is_singular() is True

    def test_detects_near_zero_theta(self) -> None:
        from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
        from mixedlm.models.lmer import LmerResult
        from scipy import sparse

        matrices = ModelMatrices(
            y=np.array([1.0, 2.0, 3.0]),
            X=np.array([[1.0], [1.0], [1.0]]),
            Z=sparse.csc_matrix(np.eye(3)),
            fixed_names=["(Intercept)"],
            random_structures=[
                RandomEffectStructure(
                    grouping_factor="g",
                    term_names=["(Intercept)"],
                    n_levels=3,
                    n_terms=1,
                    correlated=False,
                    level_map={"0": 0, "1": 1, "2": 2},
                )
            ],
            n_obs=3,
            n_fixed=1,
            n_random=3,
            weights=np.ones(3),
            offset=np.zeros(3),
        )

        def result(theta):
            return LmerResult(
                formula=parse_formula("y ~ 1 + (1 | g)"),
                matrices=matrices,
                theta=np.array([theta]),
                beta=np.array([2.0]),
                sigma=1.0,
                u=np.zeros(3),
                deviance=10.0,
                REML=True,
                converged=True,
                n_iter=1,
            )

        assert result(0.0).isSingular() is True
        assert result(1.0).isSingular() is False


class TestAllFit:
    def test_tables_report_each_optimizer_fit(self, sleepstudy_lmm) -> None:
        result = sleepstudy_lmm.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        assert isinstance(result, AllFitResult)
        assert list(result.fits) == ["L-BFGS-B", "Nelder-Mead"]
        assert result.errors == {}
        for name, fit in result.fits.items():
            assert fit.converged
            assert_allclose(fit.beta, sleepstudy_lmm.beta, atol=1e-4)
            assert result.fixef_table()[name] == fit.fixef()
            assert result.theta_table()[name] == list(fit.theta)
        deviances = {name: fit.deviance for name, fit in result.fits.items()}
        for criterion in ("deviance", "AIC", "BIC"):
            assert result.best_fit(criterion) is result.fits[min(deviances, key=deviances.get)]
        assert result.is_consistent()

    @pytest.mark.filterwarnings(
        # allFit records non-converged optimizers; their fit warnings still escape.
        "ignore:Model failed to converge:UserWarning"
    )
    def test_default_optimizers_reach_the_same_optimum(self, sleepstudy_lmm) -> None:
        from mixedlm.inference.allfit import _default_optimizers

        result = sleepstudy_lmm.allFit(SLEEPSTUDY)

        assert list(result.fits) == _default_optimizers()
        assert result.warnings.keys() == result.fits.keys()
        assert result.errors == {}
        assert result.best_fit().deviance <= sleepstudy_lmm.deviance + 1e-6
        for name, fit in result.fits.items():
            if fit.converged:
                assert fit.deviance == pytest.approx(sleepstudy_lmm.deviance, abs=1e-3), name
            else:
                assert result.warnings[name] == ["Did not converge"]
        assert result.is_consistent()

    def test_glmm_optimizers_agree(self, cbpp_glmm) -> None:
        result = cbpp_glmm.allFit(CBPP, optimizers=["L-BFGS-B", "Nelder-Mead"])

        assert list(result.fits) == ["L-BFGS-B", "Nelder-Mead"]
        for fit in result.fits.values():
            assert fit.converged
            assert fit.deviance == pytest.approx(cbpp_glmm.deviance, abs=1e-4)
            assert_allclose(fit.beta, cbpp_glmm.beta, atol=1e-3)

    def test_singular_refits_are_flagged(self, singular_cbpp_glmm) -> None:
        data = CBPP.assign(y=CBPP["incidence"] / CBPP["size"])

        with pytest.warns(UserWarning, match="Model is singular"):
            result = singular_cbpp_glmm.allFit(data, optimizers=["L-BFGS-B"])

        assert result.warnings == {"L-BFGS-B": ["Singular fit"]}

    def test_is_consistent_ignores_fits_that_did_not_converge(self, sleepstudy_lmm) -> None:
        stalled = replace(sleepstudy_lmm, deviance=sleepstudy_lmm.deviance + 50.0, converged=False)
        fits = {"COBYQA": sleepstudy_lmm, "stalled": stalled}
        result = AllFitResult(fits=fits, errors={}, warnings={})

        assert result.is_consistent()
        fits["stalled"] = replace(stalled, converged=True)
        assert not result.is_consistent()

    def test_methods_validate_n_jobs(self, sleepstudy_lmm, cbpp_glmm) -> None:
        for result, data in ((sleepstudy_lmm, SLEEPSTUDY), (cbpp_glmm, CBPP)):
            with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
                result.allFit(data, optimizers=["Nelder-Mead"], n_jobs=0)
            with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
                result.drop1(data, n_jobs=0)

    def test_str_repr(self, sleepstudy_lmm) -> None:
        result = sleepstudy_lmm.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        str_output = str(result)
        assert "allFit summary:" in str_output
        assert "L-BFGS-B" in str_output
        assert "Nelder-Mead" in str_output

        repr_output = repr(result)
        assert "AllFitResult" in repr_output
        assert "successful" in repr_output

    def test_function_and_method_share_result_interface(self) -> None:
        from mixedlm.inference import AllFitResult as InferenceAllFitResult

        formula_result = allFit(
            "Reaction ~ Days + (1 | Subject)",
            SLEEPSTUDY,
            optimizers=["L-BFGS-B"],
        )

        assert AllFitResult is InferenceAllFitResult
        assert isinstance(formula_result, AllFitResult)
        assert formula_result.results["L-BFGS-B"] is formula_result.fits["L-BFGS-B"]
        assert formula_result.best_optimizer == "L-BFGS-B"
        assert list(formula_result.summary["optimizer"]) == ["L-BFGS-B"]

    def test_legacy_result_interface_preserves_failures(self) -> None:
        result = AllFitResult(
            fits={"broken": None},
            errors={"broken": "optimizer failed"},
            warnings={"broken": []},
        )

        assert isinstance(result.results["broken"], RuntimeError)
        assert str(result.results["broken"]) == "optimizer failed"
        assert result.summary.loc[0, "error"] == "optimizer failed"
        assert result.best_optimizer == "broken"


class TestVarCorr:
    def test_random_intercepts_match_published_lme4(self, sleepstudy_lmm) -> None:
        vc = sleepstudy_lmm.VarCorr()
        subject = vc.groups["Subject"]

        assert subject.variance["(Intercept)"] == pytest.approx(1378.2, abs=0.05)
        assert subject.stddev["(Intercept)"] == pytest.approx(37.12, abs=0.005)
        assert vc.residual == pytest.approx(960.5, abs=0.05)
        assert subject.variance["(Intercept)"] == pytest.approx(
            (sleepstudy_lmm.theta[0] * sleepstudy_lmm.sigma) ** 2
        )
        assert vc.as_dict() == {"Subject": {"(Intercept)": subject.variance["(Intercept)"]}}

    def test_correlated_slopes_match_published_lme4(self, sleepstudy_slopes_lmm) -> None:
        theta, sigma = sleepstudy_slopes_lmm.theta, sleepstudy_slopes_lmm.sigma
        lower = np.array([[theta[0], 0.0], [theta[1], theta[2]]])
        covariance = sigma**2 * lower @ lower.T
        vc = sleepstudy_slopes_lmm.VarCorr()
        subject = vc.groups["Subject"]

        # Published to two decimals; lme4's optimizer stops within 0.02 of the optimum.
        assert subject.variance["(Intercept)"] == pytest.approx(612.10, abs=0.02)
        assert subject.variance["Days"] == pytest.approx(35.07, abs=0.005)
        assert subject.corr[0, 1] == pytest.approx(0.07, abs=0.005)
        assert vc.residual == pytest.approx(654.94, abs=0.005)
        assert_allclose(vc.get_cov("Subject"), covariance)
        expected_corr = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        assert_allclose(subject.corr, [[1.0, expected_corr], [expected_corr, 1.0]])

    def test_uncorrelated_slopes_match_published_lme4(self) -> None:
        vc = lmer("Reaction ~ Days + (Days || Subject)", SLEEPSTUDY).VarCorr()
        subject = vc.groups["Subject"]

        assert subject.variance["(Intercept)"] == pytest.approx(627.57, abs=0.005)
        assert subject.variance["Days"] == pytest.approx(35.86, abs=0.005)
        assert vc.residual == pytest.approx(653.58, abs=0.005)
        assert subject.corr is None

    def test_glmm_variance_is_the_squared_relative_sd(self, cbpp_glmm) -> None:
        herd = cbpp_glmm.VarCorr().groups["herd"]

        assert herd.variance["(Intercept)"] == pytest.approx(cbpp_glmm.theta[0] ** 2)
        assert herd.stddev["(Intercept)"] == pytest.approx(cbpp_glmm.theta[0])
        assert cbpp_glmm.VarCorr().as_dict() == {
            "herd": {"(Intercept)": herd.variance["(Intercept)"]}
        }

    def test_text_output(self, sleepstudy_lmm, sleepstudy_slopes_lmm, cbpp_glmm) -> None:
        output = str(sleepstudy_lmm.VarCorr())
        assert "Random effects:" in output
        assert "1378.1785" in output
        assert "37.1238" in output
        assert "Residual" in output
        assert "Days" in str(sleepstudy_slopes_lmm.VarCorr())
        assert "herd" in str(cbpp_glmm.VarCorr())
        assert repr(sleepstudy_lmm.VarCorr()) == "VarCorr(1 groups, residual=960.4566)"


class TestLogLik:
    def test_reml_criterion_matches_published_lme4(self, sleepstudy_lmm) -> None:
        ll = sleepstudy_lmm.logLik()

        assert ll.value == pytest.approx(-1786.5 / 2, abs=0.025)
        assert ll.value == pytest.approx(-sleepstudy_lmm.deviance / 2)
        assert (ll.df, ll.nobs, ll.REML) == (4, 180, True)
        assert sleepstudy_lmm.AIC() == pytest.approx(-2 * ll.value + 2 * ll.df)
        assert sleepstudy_lmm.BIC() == pytest.approx(-2 * ll.value + ll.df * np.log(180))

    def test_maximum_likelihood_matches_published_lme4(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        ll = result.logLik()

        assert ll.value == pytest.approx(-897.04, abs=0.005)
        assert ll.REML is False
        assert result.AIC() == pytest.approx(1802.1, abs=0.05)
        assert result.BIC() == pytest.approx(-2 * ll.value + 4 * np.log(180))

    def test_glmm_is_the_full_binomial_laplace_approximation(self, cbpp_glmm) -> None:
        theta = cbpp_glmm.theta[0]
        Z = cbpp_glmm.getME("Z").toarray()
        u = cbpp_glmm.getME("u")
        mu, weights = cbpp_working_weights(cbpp_glmm)
        _, logdet = np.linalg.slogdet(np.eye(15) + theta**2 * Z.T @ (weights[:, None] * Z))
        conditional = stats.binom.logpmf(CBPP["incidence"], CBPP["size"], mu).sum()

        ll = cbpp_glmm.logLik()

        assert ll.value == pytest.approx(conditional - 0.5 * (u @ u + logdet), abs=1e-6)
        assert (ll.df, ll.nobs, ll.REML) == (5, 56, False)
        assert cbpp_glmm.AIC() == pytest.approx(-2 * ll.value + 10)
        assert cbpp_glmm.BIC() == pytest.approx(-2 * ll.value + 5 * np.log(56))

    def test_text_and_numeric_protocols(self, sleepstudy_lmm) -> None:
        ll = sleepstudy_lmm.logLik()

        assert float(ll) == ll.value
        assert ll < 0
        assert -2 * ll == -2 * ll.value
        output = str(ll)
        assert "log Lik." in output
        assert "df=4" in output
        assert "REML" in output
        assert "LogLik" in repr(ll)
        assert "value=" in repr(ll)

    def test_preserves_metadata_when_copied_or_pickled(self, sleepstudy_lmm) -> None:
        ll = sleepstudy_lmm.logLik()

        for restored in (copy(ll), loads(dumps(ll))):
            assert restored == ll
            assert restored.df == ll.df
            assert restored.nobs == ll.nobs
            assert restored.REML == ll.REML


class TestDeviance:
    def test_reml_and_ml_criteria_match_published_lme4(self, sleepstudy_lmm) -> None:
        ml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        assert sleepstudy_lmm.isREML()
        assert sleepstudy_lmm.REMLcrit() == pytest.approx(1786.5, abs=0.05)
        assert sleepstudy_lmm.get_deviance() == sleepstudy_lmm.REMLcrit()
        assert sleepstudy_lmm.get_deviance() == sleepstudy_lmm.deviance
        assert not ml.isREML()
        assert ml.REMLcrit() == pytest.approx(1794.1, abs=0.05)
        assert ml.get_deviance() == ml.deviance

    def test_glmm_deviance_is_relative_to_the_saturated_model(self, cbpp_glmm) -> None:
        incidence, size = CBPP["incidence"], CBPP["size"]
        saturated = stats.binom.logpmf(incidence, size, incidence / size).sum()

        assert not cbpp_glmm.isREML()
        assert cbpp_glmm.get_deviance() == pytest.approx(cbpp_glmm.deviance - 2 * saturated)
        assert cbpp_glmm.REMLcrit() == pytest.approx(cbpp_glmm.get_deviance())
        assert cbpp_glmm.get_deviance() == pytest.approx(-2 * cbpp_glmm.logLik().value)


class TestModelMatrix:
    def test_fixed_and_random_designs(self, sleepstudy_lmm) -> None:
        X = np.column_stack((np.ones(180), SLEEPSTUDY["Days"]))
        Z = pd.get_dummies(SLEEPSTUDY["Subject"]).to_numpy(dtype=float)

        assert_array_equal(sleepstudy_lmm.model_matrix("fixed"), X)
        assert_array_equal(sleepstudy_lmm.model_matrix("X"), X)
        assert_array_equal(sleepstudy_lmm.model_matrix("random").toarray(), Z)
        assert_array_equal(sleepstudy_lmm.model_matrix("Z").toarray(), Z)
        both = sleepstudy_lmm.model_matrix("both")
        assert_array_equal(both[0], X)
        assert_array_equal(both[1].toarray(), Z)

    def test_glmm_designs(self, cbpp_glmm) -> None:
        periods = pd.get_dummies(CBPP["period"]).to_numpy(dtype=float)
        herds = pd.get_dummies(CBPP["herd"]).to_numpy(dtype=float)

        assert_array_equal(
            cbpp_glmm.model_matrix("fixed"), np.column_stack((np.ones(56), periods[:, 1:]))
        )
        assert_array_equal(cbpp_glmm.model_matrix("random").toarray(), herds)


class TestTerms:
    def test_lmer_terms_basic(self, sleepstudy_lmm) -> None:
        t = sleepstudy_lmm.terms()

        assert t.response == "Reaction"
        assert t.fixed_terms == ["(Intercept)", "Days"]
        assert t.random_terms == {"Subject": ["(Intercept)"]}
        assert "Days" in t.fixed_variables
        assert "Subject" in t.grouping_factors
        assert t.has_intercept

    def test_lmer_terms_random_slope(self, sleepstudy_slopes_lmm) -> None:
        t = sleepstudy_slopes_lmm.terms()

        assert t.random_terms["Subject"] == ["(Intercept)", "Days"]
        assert "Days" in t.random_variables

    def test_lmer_terms_merge_split_group_terms(self) -> None:
        result = lmer(
            "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
            SLEEPSTUDY,
        )
        t = result.terms()

        assert t.random_terms["Subject"] == ["(Intercept)", "Days"]

    def test_lmer_terms_str(self, sleepstudy_lmm) -> None:
        output = str(sleepstudy_lmm.terms())

        assert "Response" in output
        assert "Reaction" in output
        assert "Fixed effects" in output

    def test_get_formula(self, sleepstudy_lmm, cbpp_glmm) -> None:
        assert str(sleepstudy_lmm.get_formula()) == "Reaction ~ Days + (1 | Subject)"
        assert str(cbpp_glmm.get_formula()) == CBPP_FORMULA

    def test_glmer_terms_basic(self, cbpp_glmm) -> None:
        t = cbpp_glmm.terms()

        assert t.response == "incidence"
        assert t.fixed_terms == cbpp_glmm.matrices.fixed_names
        assert t.random_terms == {"herd": ["(Intercept)"]}
        assert t.grouping_factors == {"herd"}


class TestCoef:
    def test_random_intercepts_shift_only_the_intercept(self, sleepstudy_lmm) -> None:
        coef = sleepstudy_lmm.coef()["Subject"]
        fixef = sleepstudy_lmm.fixef()
        ranef = sleepstudy_lmm.ranef()["Subject"]

        assert list(coef) == ["(Intercept)", "Days"]
        assert_allclose(coef["(Intercept)"], fixef["(Intercept)"] + ranef["(Intercept)"])
        assert_allclose(coef["Days"], np.full(18, fixef["Days"]))

    def test_random_slopes_shift_both_coefficients(self, sleepstudy_slopes_lmm) -> None:
        coef = sleepstudy_slopes_lmm.coef()["Subject"]
        fixef = sleepstudy_slopes_lmm.fixef()
        ranef = sleepstudy_slopes_lmm.ranef()["Subject"]

        for term in ("(Intercept)", "Days"):
            assert_allclose(coef[term], fixef[term] + ranef[term])

    def test_lmer_coef_merges_split_terms_for_same_group(self) -> None:
        result = lmer(
            "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
            SLEEPSTUDY,
        )
        ranef = result.ranef(condVar=True)
        coef = result.coef()

        assert set(ranef["Subject"]) == {"(Intercept)", "Days"}
        assert set(ranef.condVar["Subject"]) == {"(Intercept)", "Days"}
        assert set(coef["Subject"]) == {"(Intercept)", "Days"}
        assert np.allclose(
            coef["Subject"]["(Intercept)"],
            result.fixef()["(Intercept)"] + ranef["Subject"]["(Intercept)"],
        )
        assert np.allclose(
            coef["Subject"]["Days"],
            result.fixef()["Days"] + ranef["Subject"]["Days"],
        )

    def test_lmer_coef_includes_random_only_term(self) -> None:
        result = lmer("Reaction ~ 1 + (0 + Days | Subject)", SLEEPSTUDY)
        ranef = result.ranef()
        coef = result.coef()

        assert list(coef["Subject"]) == ["Days", "(Intercept)"]
        assert np.allclose(coef["Subject"]["Days"], ranef["Subject"]["Days"])
        assert np.allclose(coef["Subject"]["(Intercept)"], result.fixef()["(Intercept)"])

    def test_glmm_herd_effects_shift_the_intercept(self, cbpp_glmm) -> None:
        coef = cbpp_glmm.coef()["herd"]
        fixef = cbpp_glmm.fixef()

        assert set(coef) == set(fixef)
        assert_allclose(coef["(Intercept)"], fixef["(Intercept)"] + cbpp_glmm.getME("b"))
        for term in set(fixef) - {"(Intercept)"}:
            assert_allclose(coef[term], np.full(15, fixef[term]))
