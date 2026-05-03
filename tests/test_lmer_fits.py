# ruff: noqa: F401
import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    anova,
    families,
    findbars,
    glmer,
    glmerControl,
    is_mixed_formula,
    lmer,
    lmerControl,
    nlme,
    nlmer,
    nobars,
    parse_formula,
    subbars,
)
from mixedlm.formula.terms import InteractionTerm, VariableTerm
from mixedlm.matrices import build_model_matrices
from mixedlm.models.control import GlmerControl, LmerControl

from tests._lmer_data import CBPP, SLEEPSTUDY


class TestLmer:
    def test_random_intercept_model(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.converged
        assert len(result.fixef()) == 2
        assert "(Intercept)" in result.fixef()
        assert "Days" in result.fixef()

        assert 240 < result.fixef()["(Intercept)"] < 280
        assert 5 < result.fixef()["Days"] < 15

    def test_random_slope_model(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        assert result.converged
        assert len(result.fixef()) == 2

        ranefs = result.ranef()
        assert "Subject" in ranefs
        assert "(Intercept)" in ranefs["Subject"]
        assert "Days" in ranefs["Subject"]

    def test_fitted_and_residuals(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        fitted = result.fitted()
        residuals = result.residuals()

        assert len(fitted) == 180
        assert len(residuals) == 180
        assert np.allclose(fitted + residuals, SLEEPSTUDY["Reaction"].values)

    def test_summary(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        summary = result.summary()

        assert "Linear mixed model" in summary
        assert "REML" in summary
        assert "(Intercept)" in summary
        assert "Days" in summary

    def test_vcov(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        vcov = result.vcov()

        assert vcov.shape == (2, 2)
        assert np.all(np.diag(vcov) > 0)

    def test_aic_bic(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        aic = result.AIC()
        bic = result.BIC()

        assert aic > 0
        assert bic > 0
        assert bic > aic

    def test_summary_convergence_recommendation(self) -> None:
        ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2)
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)
        summary = result.summary()

        assert "convergence: no" in summary
        assert "allFit()" in summary

    def test_summary_singular_fit_message(self) -> None:
        np.random.seed(42)
        n_per_group = 5
        n_groups = 10
        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n_per_group * n_groups)
        y = 1.0 + 2.0 * x + np.random.randn(n_per_group * n_groups) * 0.5
        data = pd.DataFrame({"y": y, "x": x, "group": group})

        result = lmer("y ~ x + (x | group)", data)
        if result.isSingular():
            summary = result.summary()
            assert "singular" in summary
            assert "simplifying" in summary

    def test_em_init_control(self) -> None:
        ctrl = lmerControl(em_init=True, em_maxiter=20)
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)

        assert result.converged
        assert len(result.fixef()) == 2
        assert 200 < result.beta[0] < 300
        assert 5 < result.beta[1] < 15

    def test_em_init_skipped_with_explicit_start(self) -> None:
        start = np.array([1.0, 0.0, 1.0])
        ctrl = lmerControl(em_init=True, em_maxiter=20)
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, start=start, control=ctrl)

        assert len(result.fixef()) == 2

    def test_em_init_warns_on_fallback_error(self) -> None:
        from unittest.mock import patch

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
        assert len(result.fixef()) == 2


class TestGlmer:
    def test_binomial_random_intercept(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        assert result.converged
        assert len(result.fixef()) == 4
        assert "(Intercept)" in result.fixef()

    def test_binomial_fitted_values(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        fitted = result.fitted(type="response")
        assert len(fitted) == len(CBPP)
        assert np.all(fitted >= 0) and np.all(fitted <= 1)

        fitted_link = result.fitted(type="link")
        assert len(fitted_link) == len(CBPP)

    def test_binomial_residuals(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        resid_response = result.residuals(type="response")
        resid_pearson = result.residuals(type="pearson")
        resid_deviance = result.residuals(type="deviance")

        assert len(resid_response) == len(CBPP)
        assert len(resid_pearson) == len(CBPP)
        assert len(resid_deviance) == len(CBPP)

    def test_binomial_summary(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        summary = result.summary()
        assert "Generalized linear mixed model" in summary
        assert "Binomial" in summary
        assert "(Intercept)" in summary

    def test_binomial_vcov(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        vcov = result.vcov()
        assert vcov.shape == (4, 4)
        assert np.all(np.diag(vcov) > 0)

    def test_em_init_warns_on_fallback_error(self) -> None:
        from unittest.mock import patch

        with (
            patch(
                "mixedlm.estimation.em_reml.em_reml_simple",
                side_effect=NotImplementedError("unsupported"),
            ),
            pytest.warns(RuntimeWarning, match="EM initialization failed"),
        ):
            ctrl = glmerControl(em_init=True)
            result = glmer(
                "y ~ period + (1 | herd)",
                CBPP,
                family=families.Binomial(),
                control=ctrl,
            )

        assert result.converged
        assert len(result.fixef()) == 4

    def test_poisson_model(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        eta = 0.5 + 0.3 * x + group_effects[group]
        y = np.random.poisson(np.exp(eta))

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer("y ~ x + (1 | group)", data, family=families.Poisson())

        assert result.converged
        assert len(result.fixef()) == 2

    def test_glmer_aic_bic(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        aic = result.AIC()
        bic = result.BIC()

        assert np.isfinite(aic)
        assert np.isfinite(bic)

    def test_gamma_model(self) -> None:
        np.random.seed(123)
        n_groups = 8
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.uniform(0.5, 2.0, n)
        group_effects = np.random.randn(n_groups) * 0.1
        mu = np.exp(1.0 + 0.2 * x + group_effects[group])
        shape = 10.0
        y = np.random.gamma(shape, mu / shape, n)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer("y ~ x + (1 | group)", data, family=families.Gamma())

        assert len(result.fixef()) == 2
        assert np.all(result.fitted(type="response") > 0)

    def test_negative_binomial_model(self) -> None:
        np.random.seed(789)
        n_groups = 8
        n_per_group = 25
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.2
        mu = np.exp(1.5 + 0.3 * x + group_effects[group])
        theta = 5.0
        y = np.random.negative_binomial(theta, theta / (mu + theta), n)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer("y ~ x + (1 | group)", data, family=families.NegativeBinomial(theta=theta))

        assert len(result.fixef()) == 2
        assert np.all(result.fitted(type="response") >= 0)

    def test_inverse_gaussian_model(self) -> None:
        np.random.seed(321)
        n_groups = 8
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.uniform(0.5, 1.5, n)
        group_effects = np.random.randn(n_groups) * 0.1
        mu = np.exp(0.5 + 0.3 * x + group_effects[group])
        lam = 10.0
        y = np.random.wald(mu, lam, n)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer("y ~ x + (1 | group)", data, family=families.InverseGaussian())

        assert len(result.fixef()) == 2
        assert np.all(result.fitted(type="response") > 0)

    def test_summary_convergence_recommendation(self) -> None:
        ctrl = glmerControl(optimizer="Nelder-Mead", maxiter=2)
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial(), control=ctrl)
        summary = result.summary()

        assert "convergence: no" in summary
        assert "allFit()" in summary

    def test_em_init_control(self) -> None:
        ctrl = glmerControl(em_init=True, em_maxiter=20)
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial(), control=ctrl)

        assert len(result.fixef()) == 4


def generate_nlme_data() -> pd.DataFrame:
    np.random.seed(42)
    n_subjects = 8
    n_times = 10

    Asym_pop = 200
    R0_pop = 50
    lrc_pop = -2

    Asym_sd = 20
    R0_sd = 10

    data_rows = []
    for subj in range(n_subjects):
        Asym_i = Asym_pop + np.random.randn() * Asym_sd
        R0_i = R0_pop + np.random.randn() * R0_sd

        for t in range(n_times):
            time = t * 0.5
            y_true = Asym_i + (R0_i - Asym_i) * np.exp(-np.exp(lrc_pop) * time)
            y = y_true + np.random.randn() * 5

            data_rows.append(
                {
                    "y": y,
                    "time": time,
                    "subject": str(subj),
                }
            )

    return pd.DataFrame(data_rows)


NLME_DATA = generate_nlme_data()


class TestNlmer:
    def test_ssasymp_model(self) -> None:
        model = nlme.SSasymp()

        result = nlmer(
            model=model,
            data=NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym", "R0"],
        )

        assert result.converged or result.n_iter > 0
        assert len(result.fixef()) == 3
        assert "Asym" in result.fixef()
        assert "R0" in result.fixef()
        assert "lrc" in result.fixef()

    def test_sslogis_model(self) -> None:
        np.random.seed(123)
        n_subjects = 6
        n_times = 12

        data_rows = []
        for subj in range(n_subjects):
            Asym_i = 100 + np.random.randn() * 10
            xmid_i = 5 + np.random.randn() * 0.5
            scal = 1.0

            for t in range(n_times):
                time = t * 1.0
                y_true = Asym_i / (1 + np.exp((xmid_i - time) / scal))
                y = y_true + np.random.randn() * 3

                data_rows.append(
                    {
                        "y": max(y, 0.1),
                        "time": time,
                        "subject": str(subj),
                    }
                )

        data = pd.DataFrame(data_rows)
        model = nlme.SSlogis()

        result = nlmer(
            model=model,
            data=data,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
            start={"Asym": 100, "xmid": 5, "scal": 1},
        )

        assert len(result.fixef()) == 3

    def test_ssmicmen_model(self) -> None:
        np.random.seed(456)
        n_subjects = 5
        n_conc = 8

        data_rows = []
        for subj in range(n_subjects):
            Vm_i = 200 + np.random.randn() * 20
            K = 0.5

            for c in range(n_conc):
                conc = 0.1 * (c + 1)
                y_true = Vm_i * conc / (K + conc)
                y = y_true + np.random.randn() * 5

                data_rows.append(
                    {
                        "y": max(y, 0.1),
                        "conc": conc,
                        "subject": str(subj),
                    }
                )

        data = pd.DataFrame(data_rows)
        model = nlme.SSmicmen()

        result = nlmer(
            model=model,
            data=data,
            x_var="conc",
            y_var="y",
            group_var="subject",
            random_params=["Vm"],
        )

        assert len(result.fixef()) == 2
        assert "Vm" in result.fixef()
        assert "K" in result.fixef()

    def test_nlmer_fitted_residuals(self) -> None:
        model = nlme.SSasymp()

        result = nlmer(
            model=model,
            data=NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
        )

        fitted = result.fitted()
        residuals = result.residuals()

        assert len(fitted) == len(NLME_DATA)
        assert len(residuals) == len(NLME_DATA)
        assert np.allclose(fitted + residuals, NLME_DATA["y"].values)

    def test_nlmer_ranef(self) -> None:
        model = nlme.SSasymp()

        result = nlmer(
            model=model,
            data=NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym", "R0"],
        )

        ranefs = result.ranef()
        assert "subject" in ranefs
        assert "Asym" in ranefs["subject"]
        assert "R0" in ranefs["subject"]
        assert len(ranefs["subject"]["Asym"]) == 8

    def test_nlmer_summary(self) -> None:
        model = nlme.SSasymp()

        result = nlmer(
            model=model,
            data=NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
        )

        summary = result.summary()
        assert "Nonlinear mixed model" in summary
        assert "SSasymp" in summary
        assert "Asym" in summary

    def test_nlmer_aic_bic(self) -> None:
        model = nlme.SSasymp()

        result = nlmer(
            model=model,
            data=NLME_DATA,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
        )

        aic = result.AIC()
        bic = result.BIC()

        assert np.isfinite(aic)
        assert np.isfinite(bic)
