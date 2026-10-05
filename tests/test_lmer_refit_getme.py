"""getME components, update() and refit()/refitML() against direct fits."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer
from numpy.testing import assert_allclose, assert_array_equal

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY, grouped_data


def assert_same_fit(actual, expected, *, atol=1e-4) -> None:
    assert actual.converged and expected.converged
    assert_allclose(actual.beta, expected.beta, rtol=0, atol=atol)
    assert_allclose(actual.theta, expected.theta, rtol=0, atol=atol)
    assert actual.deviance == pytest.approx(expected.deviance, abs=1e-6)


class TestGetME:
    def test_design_matrices_and_response(self, sleepstudy_lmm) -> None:
        subjects = pd.get_dummies(SLEEPSTUDY["Subject"]).to_numpy(dtype=float)

        assert_array_equal(
            sleepstudy_lmm.getME("X"), np.column_stack((np.ones(180), SLEEPSTUDY["Days"]))
        )
        # Columns follow the sorted subject levels used by ranef().
        assert_array_equal(sleepstudy_lmm.getME("Z").toarray(), subjects)
        assert_array_equal(sleepstudy_lmm.getME("y"), SLEEPSTUDY["Reaction"])

    def test_parameters_match_the_fit(self, sleepstudy_lmm) -> None:
        assert_array_equal(sleepstudy_lmm.getME("beta"), sleepstudy_lmm.beta)
        assert_array_equal(sleepstudy_lmm.getME("beta"), list(sleepstudy_lmm.fixef().values()))
        assert_array_equal(sleepstudy_lmm.getME("theta"), sleepstudy_lmm.theta)
        assert sleepstudy_lmm.getME("sigma") == sleepstudy_lmm.sigma
        assert sleepstudy_lmm.getME("deviance") == sleepstudy_lmm.deviance
        assert sleepstudy_lmm.getME("REML") is True

    def test_scalar_relative_covariance_factor(self, sleepstudy_lmm) -> None:
        Lambda = sleepstudy_lmm.getME("Lambda").toarray()

        assert_allclose(Lambda, sleepstudy_lmm.theta[0] * np.eye(18))
        assert_allclose(sleepstudy_lmm.getME("Lambdat").toarray(), Lambda.T)

    def test_correlated_factor_repeats_the_lower_cholesky_block(
        self, sleepstudy_slopes_lmm
    ) -> None:
        theta = sleepstudy_slopes_lmm.theta
        block = np.array([[theta[0], 0.0], [theta[1], theta[2]]])

        Lambda = sleepstudy_slopes_lmm.getME("Lambda").toarray()

        assert_allclose(Lambda, np.kron(np.eye(18), block))
        assert_array_equal(sleepstudy_slopes_lmm.getME("lower"), [0.0, -np.inf, 0.0])
        assert_array_equal(sleepstudy_slopes_lmm.getME("Gp"), [0, 36])
        assert sleepstudy_slopes_lmm.getME("flist") == ["Subject"]
        assert sleepstudy_slopes_lmm.getME("cnms") == {"Subject": ["(Intercept)", "Days"]}

    def test_spherical_and_conditional_random_effects(self, sleepstudy_slopes_lmm) -> None:
        u = sleepstudy_slopes_lmm.getME("u")
        b = sleepstudy_slopes_lmm.getME("b")
        ranef = sleepstudy_slopes_lmm.ranef()["Subject"]

        assert_allclose(b, sleepstudy_slopes_lmm.getME("Lambda") @ u)
        assert_allclose(b[0::2], ranef["(Intercept)"])
        assert_allclose(b[1::2], ranef["Days"])
        assert u @ u == pytest.approx(sleepstudy_slopes_lmm.getME("devcomp")["cmp"]["ussq"])

    def test_dimensions_names_weights_and_offset(self, sleepstudy_lmm) -> None:
        assert sleepstudy_lmm.getME("n") == sleepstudy_lmm.getME("n_obs") == 180
        assert sleepstudy_lmm.getME("p") == sleepstudy_lmm.getME("n_fixed") == 2
        assert sleepstudy_lmm.getME("q") == sleepstudy_lmm.getME("n_random") == 18
        assert sleepstudy_lmm.getME("fixef_names") == ["(Intercept)", "Days"]
        assert_array_equal(sleepstudy_lmm.getME("lower"), [0.0])
        assert_array_equal(sleepstudy_lmm.getME("weights"), np.ones(180))
        assert_array_equal(sleepstudy_lmm.getME("offset"), np.zeros(180))

    def test_invalid_component(self, sleepstudy_lmm) -> None:
        with pytest.raises(ValueError, match="Unknown component name"):
            sleepstudy_lmm.getME("invalid_name")

    def test_glmm_components(self, cbpp_glmm) -> None:
        periods = pd.get_dummies(CBPP["period"]).to_numpy(dtype=float)
        b = cbpp_glmm.getME("b")

        assert_array_equal(cbpp_glmm.getME("X"), np.column_stack((np.ones(56), periods[:, 1:])))
        assert_array_equal(cbpp_glmm.getME("beta"), cbpp_glmm.beta)
        assert isinstance(cbpp_glmm.getME("family"), families.Binomial)
        assert cbpp_glmm.getME("nAGQ") == 1
        assert_allclose(b, cbpp_glmm.ranef()["herd"]["(Intercept)"])
        assert_allclose(b, cbpp_glmm.theta[0] * cbpp_glmm.getME("u"))
        assert np.all(b != 0)


class TestUpdate:
    def test_reml_switch_matches_a_direct_ml_fit(self, sleepstudy_lmm) -> None:
        updated = sleepstudy_lmm.update(REML=False)

        assert sleepstudy_lmm.REML is True
        assert updated.REML is False
        assert_same_fit(updated, lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False))

    def test_without_arguments_reproduces_the_fit(self, sleepstudy_lmm) -> None:
        assert_same_fit(sleepstudy_lmm.update(), sleepstudy_lmm, atol=1e-6)
        # A no-op formula edit reuses the stored model frame.
        assert_same_fit(sleepstudy_lmm.update(". ~ . + 1"), sleepstudy_lmm, atol=1e-6)

    @pytest.mark.parametrize(
        ("original", "change", "expected"),
        [
            ("Reaction ~ 1 + (1 | Subject)", ". ~ . + Days", "Reaction ~ Days + (1 | Subject)"),
            ("Reaction ~ Days + (1 | Subject)", ". ~ . - Days", "Reaction ~ 1 + (1 | Subject)"),
            (
                "Reaction ~ Days + (1 | Subject)",
                ". ~ 1 + (1 | Subject)",
                "Reaction ~ 1 + (1 | Subject)",
            ),
            (
                "Reaction ~ Days + (1 | Subject)",
                ". ~ . + Days2",
                "Reaction ~ Days + Days2 + (1 | Subject)",
            ),
            (
                "Reaction ~ 1 + (1 | Subject)",
                "Reaction ~ Days + (Days | Subject)",
                "Reaction ~ Days + (1 + Days | Subject)",
            ),
        ],
    )
    def test_formula_changes_match_direct_fits(self, original, change, expected) -> None:
        data = SLEEPSTUDY.assign(Days2=SLEEPSTUDY["Days"] ** 2)

        updated = lmer(original, data).update(change, data=data)

        assert str(updated.formula) == expected
        assert updated.formula.response == "Reaction"
        assert_same_fit(updated, lmer(expected, data))

    def test_new_data_matches_a_direct_fit(self, sleepstudy_lmm) -> None:
        subset = SLEEPSTUDY[SLEEPSTUDY["Days"] <= 5]

        updated = sleepstudy_lmm.update(data=subset)

        assert updated.getME("n") == 108
        assert_same_fit(updated, lmer("Reaction ~ Days + (1 | Subject)", subset))

    def test_new_weights_match_a_direct_fit(self, sleepstudy_lmm) -> None:
        weights = np.ones(180)
        weights[:90] = 2.0

        updated = sleepstudy_lmm.update(weights=weights)

        assert_array_equal(sleepstudy_lmm.getME("weights"), np.ones(180))
        assert_array_equal(updated.getME("weights"), weights)
        assert_same_fit(
            updated, lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, weights=weights)
        )

    def test_glmm_added_term_matches_a_direct_fit(self, cbpp_glmm) -> None:
        intercept_only = glmer(
            "incidence / size ~ 1 + (1 | herd)", CBPP, family=families.Binomial()
        )

        updated = intercept_only.update(". ~ . + period", data=CBPP)

        assert updated.matrices.fixed_names == cbpp_glmm.matrices.fixed_names
        assert_same_fit(updated, cbpp_glmm)

    def test_glmm_family_change_matches_a_direct_fit(self, cbpp_glmm) -> None:
        updated = cbpp_glmm.update(
            formula="incidence ~ period + (1 | herd)", family=families.Poisson()
        )

        assert isinstance(cbpp_glmm.getME("family"), families.Binomial)
        assert isinstance(updated.getME("family"), families.Poisson)
        assert_same_fit(
            updated, glmer("incidence ~ period + (1 | herd)", CBPP, family=families.Poisson())
        )


class TestRefit:
    def test_same_response_reproduces_the_fit(self, sleepstudy_lmm) -> None:
        refitted = sleepstudy_lmm.refit(SLEEPSTUDY["Reaction"].to_numpy())

        assert_same_fit(refitted, sleepstudy_lmm, atol=1e-6)
        assert_same_fit(refitted.refit(SLEEPSTUDY["Reaction"].to_numpy()), refitted, atol=1e-6)

    def test_simulated_response_matches_a_direct_fit(self, sleepstudy_lmm) -> None:
        newresp = sleepstudy_lmm.simulate(seed=123)

        refitted = sleepstudy_lmm.refit(newresp)

        assert refitted.REML is True
        assert (refitted.getME("n"), refitted.getME("p"), refitted.getME("q")) == (180, 2, 18)
        direct = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY.assign(Reaction=newresp))
        assert_same_fit(refitted, direct)
        assert not np.allclose(refitted.beta, sleepstudy_lmm.beta, rtol=1e-4)

    def test_glmm_success_counts_match_a_direct_fit(self, cbpp_glmm) -> None:
        # Grouped binomial refits take success counts, like the formula response.
        incidence = np.minimum(CBPP["incidence"].to_numpy()[::-1], CBPP["size"])

        refitted = cbpp_glmm.refit(incidence)

        direct = glmer(CBPP_FORMULA, CBPP.assign(incidence=incidence), family=families.Binomial())
        assert_same_fit(refitted, direct)

    def test_glmm_bernoulli_response_matches_a_direct_fit(self, grouped_glmm) -> None:
        data = grouped_data("binomial")
        y = np.random.default_rng(7).binomial(1, grouped_glmm.fitted()).astype(float)

        refitted = grouped_glmm.refit(y)

        direct = glmer("y ~ x + (1 | group)", data.assign(y=y), family=families.Binomial())
        assert_same_fit(refitted, direct)

    @pytest.mark.parametrize("model", ["sleepstudy_lmm", "cbpp_glmm"])
    def test_wrong_length_response_is_rejected(self, request, model) -> None:
        result = request.getfixturevalue(model)

        with pytest.raises(ValueError, match="newresp has length 3"):
            result.refit(np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        "formula", ["Reaction ~ Days + (1 | Subject)", "Reaction ~ Days + (Days | Subject)"]
    )
    def test_refit_ml_matches_a_direct_ml_fit(self, formula) -> None:
        reml = lmer(formula, SLEEPSTUDY)

        ml = reml.refitML()

        assert reml.REML is True and reml.isREML()
        assert ml.REML is False and not ml.isREML()
        assert_same_fit(ml, lmer(formula, SLEEPSTUDY, REML=False))
        assert ml.logLik().value == pytest.approx(-ml.deviance / 2)

    def test_refit_ml_returns_maximum_likelihood_fits_unchanged(self, cbpp_glmm) -> None:
        ml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        assert ml.refitML() is ml
        assert not cbpp_glmm.isREML()
        assert cbpp_glmm.refitML() is cbpp_glmm
