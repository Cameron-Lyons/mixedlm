"""End-to-end lmer, glmer and nlmer fits checked against published or simulated truth."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    families,
    glmer,
    glmerControl,
    lmer,
    lmerControl,
    nlme,
    nlmer,
)
from numpy.testing import assert_allclose
from scipy.special import expit, xlogy

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY, grouped_data


def asymptotic_growth_data():
    """Simulate SSasymp curves long enough to identify the asymptote."""
    rng = np.random.default_rng(42)
    asym = 200 + rng.normal(0.0, 20.0, 8)
    r0 = 50 + rng.normal(0.0, 10.0, 8)
    subject = np.repeat(np.arange(8), 10)
    time = np.tile(np.arange(10) * 2.5, 8)
    y = asym[subject] + (r0[subject] - asym[subject]) * np.exp(-np.exp(-2.0) * time)
    data = pd.DataFrame(
        {"y": y + rng.normal(0.0, 5.0, len(time)), "time": time, "subject": subject.astype(str)}
    )
    return data, asym, r0


@pytest.fixture(scope="module")
def growth():
    data, asym, r0 = asymptotic_growth_data()
    result = nlmer(
        model=nlme.SSasymp(),
        data=data,
        x_var="time",
        y_var="y",
        group_var="subject",
        random_params=["Asym", "R0"],
    )
    return result, data, asym, r0


@pytest.fixture(scope="module")
def growth_random_asymptote(growth):
    _, data, _, _ = growth
    return nlmer(
        model=nlme.SSasymp(),
        data=data,
        x_var="time",
        y_var="y",
        group_var="subject",
        random_params=["Asym"],
    )


class TestLmer:
    def test_random_intercepts_match_published_lme4(self, sleepstudy_lmm) -> None:
        assert sleepstudy_lmm.converged
        assert list(sleepstudy_lmm.fixef()) == ["(Intercept)", "Days"]
        assert_allclose(sleepstudy_lmm.beta, [251.4051, 10.4673], atol=5e-5)
        assert_allclose(np.sqrt(np.diag(sleepstudy_lmm.vcov())), [9.7467, 0.8042], atol=5e-5)
        assert sleepstudy_lmm.deviance == pytest.approx(1786.5, abs=0.05)

    def test_random_slopes_match_published_lme4(self, sleepstudy_slopes_lmm) -> None:
        ranefs = sleepstudy_slopes_lmm.ranef()["Subject"]

        assert sleepstudy_slopes_lmm.converged
        assert_allclose(sleepstudy_slopes_lmm.beta, [251.4051, 10.4673], atol=5e-5)
        assert_allclose(np.sqrt(np.diag(sleepstudy_slopes_lmm.vcov())), [6.825, 1.546], atol=5e-4)
        assert list(ranefs) == ["(Intercept)", "Days"]
        assert len(ranefs["Days"]) == 18

    def test_fitted_and_residuals_partition_the_response(self, sleepstudy_lmm) -> None:
        X = sleepstudy_lmm.getME("X")
        Z = sleepstudy_lmm.getME("Z")
        fitted = sleepstudy_lmm.fitted()

        assert_allclose(fitted, X @ sleepstudy_lmm.beta + Z @ sleepstudy_lmm.getME("b"))
        assert_allclose(fitted + sleepstudy_lmm.residuals(), SLEEPSTUDY["Reaction"])

    def test_summary(self, sleepstudy_lmm) -> None:
        summary = sleepstudy_lmm.summary()

        assert "Linear mixed model fit by REML" in summary
        assert "Formula: Reaction ~ Days + (1 | Subject)" in summary
        for value in ("1378.1785", "251.4051", "9.7467", "10.4673", "0.8042"):
            assert value in summary
        assert "convergence: yes" in summary

    def test_summary_convergence_recommendation(self) -> None:
        ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2)
        with pytest.warns(UserWarning, match="Model failed to converge"):
            result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)
        summary = result.summary()

        assert "convergence: no" in summary
        assert "allFit()" in summary

    def test_summary_singular_fit_message(self) -> None:
        # Every group has the same responses, so the group variance is zero.
        rng = np.random.default_rng(42)
        x = np.tile(np.linspace(-1, 1, 10), 5)
        y = 1.0 + 2.0 * x + np.tile(rng.normal(0.0, 0.5, 10), 5)
        data = pd.DataFrame({"y": y, "x": x, "group": np.repeat(list("abcde"), 10)})

        with pytest.warns(UserWarning, match="Model is singular"):
            result = lmer("y ~ x + (1 | group)", data)
        summary = result.summary()

        assert "singular" in summary
        assert "simplifying" in summary

    def test_em_initialization_reaches_the_default_optimum(self, sleepstudy_slopes_lmm) -> None:
        ctrl = lmerControl(em_init=True, em_maxiter=20)

        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)

        assert result.converged
        assert_allclose(result.beta, sleepstudy_slopes_lmm.beta, atol=1e-4)
        assert result.deviance == pytest.approx(sleepstudy_slopes_lmm.deviance, abs=1e-6)

    def test_em_initialization_is_skipped_with_explicit_start(self, sleepstudy_slopes_lmm) -> None:
        ctrl = lmerControl(em_init=True, em_maxiter=20)
        with patch(
            "mixedlm.estimation.em_reml.em_reml_simple",
            side_effect=AssertionError("an explicit start must skip EM"),
        ):
            result = lmer(
                "Reaction ~ Days + (Days | Subject)",
                SLEEPSTUDY,
                start=np.array([1.0, 0.0, 1.0]),
                control=ctrl,
            )

        assert result.deviance == pytest.approx(sleepstudy_slopes_lmm.deviance, abs=1e-6)

    def test_em_init_warns_on_fallback_error(self, sleepstudy_lmm) -> None:
        with (
            patch(
                "mixedlm.estimation.em_reml.em_reml_simple",
                side_effect=NotImplementedError("unsupported"),
            ),
            pytest.warns(RuntimeWarning, match="EM initialization failed"),
        ):
            ctrl = lmerControl(em_init=True)
            result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)

        assert result.converged
        assert_allclose(result.beta, sleepstudy_lmm.beta, atol=1e-6)


class TestGlmer:
    def test_fitted_values_apply_the_inverse_link(self, cbpp_glmm) -> None:
        eta = cbpp_glmm.getME("X") @ cbpp_glmm.beta + cbpp_glmm.getME("Z") @ cbpp_glmm.getME("b")

        assert cbpp_glmm.converged
        assert_allclose(cbpp_glmm.fitted(type="link"), eta)
        assert_allclose(cbpp_glmm.fitted(type="response"), expit(eta))
        assert_allclose(cbpp_glmm.fitted(), expit(eta))

    def test_residual_types(self, cbpp_glmm) -> None:
        size = CBPP["size"].to_numpy()
        y = CBPP["incidence"].to_numpy() / size
        mu = cbpp_glmm.fitted()
        unit_deviance = 2 * size * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)))

        assert_allclose(cbpp_glmm.residuals(type="response"), y - mu)
        assert_allclose(
            cbpp_glmm.residuals(type="pearson"), (y - mu) / np.sqrt(mu * (1 - mu) / size)
        )
        deviance_residuals = np.sign(y - mu) * np.sqrt(unit_deviance)
        assert_allclose(cbpp_glmm.residuals(type="deviance"), deviance_residuals)
        assert_allclose(cbpp_glmm.residuals(), deviance_residuals)

    def test_vcov_is_the_schur_complement_at_the_mode(self, cbpp_glmm) -> None:
        X = cbpp_glmm.getME("X")
        Z = cbpp_glmm.getME("Z").toarray() * cbpp_glmm.theta[0]
        mu = cbpp_glmm.fitted()
        weights = CBPP["size"].to_numpy() * mu * (1 - mu)
        XtWZ = X.T @ (weights[:, None] * Z)
        precision = Z.T @ (weights[:, None] * Z) + np.eye(15)
        information = X.T @ (weights[:, None] * X) - XtWZ @ np.linalg.solve(precision, XtWZ.T)

        assert_allclose(cbpp_glmm.vcov(), np.linalg.inv(information), rtol=1e-8)

    def test_summary(self, cbpp_glmm) -> None:
        summary = cbpp_glmm.summary()

        assert "Generalized linear mixed model fit by maximum likelihood (Laplace)" in summary
        assert "Family: Binomial" in summary
        # lme4 prints AIC 194.1, BIC 204.2 and logLik -92.0 for this model.
        for value in ("194.1", "204.2", "-92.0", "(Intercept)"):
            assert value in summary

    def test_summary_without_a_normalized_likelihood(self, singular_cbpp_glmm) -> None:
        # Proportions fitted as single trials are not whole-number success counts.
        with pytest.raises(ValueError, match="requires nonnegative whole-number counts"):
            singular_cbpp_glmm.logLik()

        summary = singular_cbpp_glmm.summary()

        lines = summary.splitlines()
        assert lines[lines.index("     AIC      BIC   logLik  -2logL") + 1].split() == ["NA"] * 4
        assert "requires nonnegative whole-number counts" in summary
        assert "boundary (singular) fit" in summary
        rows = {line.split()[0]: line.split() for line in lines if line.startswith(("(I", "per"))}
        assert rows["(Intercept)"][-2:] == ["0.0422", "*"]
        assert rows["period.1"][-1] == "0.2923"

    def test_em_init_warns_on_fallback_error(self, cbpp_glmm) -> None:
        with (
            patch(
                "mixedlm.estimation.em_reml.em_reml_simple",
                side_effect=NotImplementedError("unsupported"),
            ),
            pytest.warns(RuntimeWarning, match="EM initialization failed"),
        ):
            ctrl = glmerControl(em_init=True)
            result = glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), control=ctrl)

        assert result.converged
        assert_allclose(result.beta, cbpp_glmm.beta, atol=1e-4)

    def test_em_initialization_reaches_the_default_optimum(self, cbpp_glmm) -> None:
        ctrl = glmerControl(em_init=True, em_maxiter=20)

        result = glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), control=ctrl)

        assert result.converged
        assert_allclose(result.beta, cbpp_glmm.beta, atol=1e-4)
        assert result.deviance == pytest.approx(cbpp_glmm.deviance, abs=1e-6)

    def test_summary_convergence_recommendation(self) -> None:
        ctrl = glmerControl(optimizer="Nelder-Mead", maxiter=2)
        with pytest.warns(UserWarning, match="Model failed to converge"):
            result = glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), control=ctrl)
        summary = result.summary()

        assert "convergence: no" in summary
        assert "allFit()" in summary

    def test_poisson_model_recovers_the_simulated_slope(self) -> None:
        data = grouped_data("poisson")

        result = glmer("y ~ x + (1 | group)", data, family=families.Poisson())

        assert result.converged
        se = np.sqrt(np.diag(result.vcov()))
        assert abs(result.beta[1] - 0.3) < 3 * se[1]
        # The intercept score equation makes fitted means average the data.
        assert result.fitted(type="response").mean() == pytest.approx(data["y"].mean(), rel=0.02)

    def test_poisson_model_recovers_means_above_one(self) -> None:
        rng = np.random.default_rng(42)
        n_groups = 12
        n_per_group = 10
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.tile(np.linspace(-1, 1, n_per_group), n_groups)
        group_effects = rng.normal(0, 0.2, n_groups)
        y = rng.poisson(np.exp(3.0 + 0.4 * x + group_effects[group]))
        data = pd.DataFrame({"y": y, "x": x, "group": group.astype(str)})

        result = glmer("y ~ x + (1 | group)", data, family=families.Poisson())

        assert result.converged
        assert 2.5 < result.beta[0] < 3.5
        assert 0.2 < result.beta[1] < 0.7
        assert np.max(result.fitted()) > 10

    @pytest.mark.parametrize(
        ("family", "draw", "weights"),
        [
            # GLMM dispersion is fixed at one, so prior weights carry the precision.
            (families.Gamma(), lambda rng, mu: rng.gamma(10.0, mu / 10.0), 10.0),
            (families.InverseGaussian(), lambda rng, mu: rng.wald(mu, 10.0), 10.0),
            (
                families.NegativeBinomial(theta=5.0),
                lambda rng, mu: rng.negative_binomial(5.0, 5.0 / (mu + 5.0)),
                1.0,
            ),
        ],
        ids=["gamma", "inverse_gaussian", "negative_binomial"],
    )
    def test_log_link_families_recover_the_simulated_coefficients(
        self, family, draw, weights
    ) -> None:
        rng = np.random.default_rng(321)
        group = np.repeat(np.arange(10), 25)
        x = rng.uniform(-1.0, 1.0, len(group))
        mu = np.exp(1.0 + 0.3 * x + rng.normal(0.0, 0.2, 10)[group])
        data = pd.DataFrame({"y": draw(rng, mu), "x": x, "group": group.astype(str)})

        result = glmer(
            "y ~ x + (1 | group)", data, family=family, weights=np.full(len(group), weights)
        )

        assert result.converged
        se = np.sqrt(np.diag(result.vcov()))
        assert np.all(np.abs(result.beta - [1.0, 0.3]) < 3 * se)
        assert result.theta[0] > 0


class TestNlmer:
    def test_asymptotic_model_recovers_the_simulated_parameters(self, growth) -> None:
        result, _, asym, r0 = growth
        se = np.sqrt(np.diag(result.vcov()))

        assert result.converged
        assert list(result.fixef()) == ["Asym", "R0", "lrc"]
        assert np.all(np.abs(result.phi - [asym.mean(), r0.mean(), -2.0]) < 3 * se)

    def test_random_effects_track_the_simulated_subject_deviations(self, growth) -> None:
        result, _, asym, r0 = growth
        ranefs = result.ranef()["subject"]

        assert list(ranefs) == ["Asym", "R0"]
        assert len(ranefs["Asym"]) == 8
        assert np.corrcoef(ranefs["Asym"], asym - asym.mean())[0, 1] > 0.95
        assert np.corrcoef(ranefs["R0"], r0 - r0.mean())[0, 1] > 0.9

    def test_fitted_and_residuals_partition_the_response(self, growth_random_asymptote) -> None:
        data, _, _ = asymptotic_growth_data()

        fitted = growth_random_asymptote.fitted()

        assert_allclose(fitted + growth_random_asymptote.residuals(), data["y"])

    def test_summary(self, growth_random_asymptote) -> None:
        summary = growth_random_asymptote.summary()

        assert "Nonlinear mixed model" in summary
        assert "SSasymp" in summary
        assert "Asym" in summary

    def test_information_criteria(self, growth_random_asymptote) -> None:
        ll = growth_random_asymptote.logLik()

        assert ll.value == float(ll)
        # Three fixed parameters, one random-effect SD and the residual SD.
        assert ll.df == growth_random_asymptote.npar() == 5
        assert ll.nobs == growth_random_asymptote.nobs() == 80
        assert ll.REML is False
        assert growth_random_asymptote.AIC() == pytest.approx(-2 * ll.value + 10)
        assert growth_random_asymptote.BIC() == pytest.approx(-2 * ll.value + 5 * np.log(80))

    def test_logistic_model_recovers_the_midpoint(self) -> None:
        rng = np.random.default_rng(123)
        subject = np.repeat(np.arange(6), 12)
        time = np.tile(np.arange(12.0), 6)
        asym = 100 + rng.normal(0.0, 10.0, 6)
        xmid = 5 + rng.normal(0.0, 0.5, 6)
        y = asym[subject] / (1 + np.exp(xmid[subject] - time)) + rng.normal(0.0, 3.0, len(time))
        data = pd.DataFrame({"y": y, "time": time, "subject": subject.astype(str)})

        result = nlmer(
            model=nlme.SSlogis(),
            data=data,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            start={"Asym": 100, "xmid": 5, "scal": 1},
        )

        assert result.converged
        assert list(result.fixef()) == ["Asym", "xmid", "scal"]
        # The midpoint varies by subject but is modelled as fixed: allow its spread.
        assert result.fixef()["xmid"] == pytest.approx(xmid.mean(), abs=0.5)
        assert result.fixef()["scal"] == pytest.approx(1.0, abs=0.15)

    def test_michaelis_menten_model_recovers_the_half_saturation(self) -> None:
        rng = np.random.default_rng(456)
        subject = np.repeat(np.arange(5), 8)
        conc = np.tile(0.1 * np.arange(1, 9), 5)
        vm = 200 + rng.normal(0.0, 20.0, 5)
        y = vm[subject] * conc / (0.5 + conc) + rng.normal(0.0, 5.0, len(conc))
        data = pd.DataFrame({"y": y, "conc": conc, "subject": subject.astype(str)})

        result = nlmer(
            model=nlme.SSmicmen(),
            data=data,
            x_var="conc",
            y_var="y",
            group_var="subject",
            random_params=["Vm"],
        )

        se = np.sqrt(np.diag(result.vcov()))
        assert result.converged
        assert list(result.fixef()) == ["Vm", "K"]
        assert abs(result.fixef()["K"] - 0.5) < 3 * se[1]
