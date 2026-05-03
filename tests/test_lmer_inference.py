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


class TestInference:
    def test_lmer_confint_wald(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ci = result.confint(method="Wald")
        assert "(Intercept)" in ci
        assert "Days" in ci
        assert ci["(Intercept)"][0] < result.fixef()["(Intercept)"] < ci["(Intercept)"][1]
        assert ci["Days"][0] < result.fixef()["Days"] < ci["Days"][1]

    def test_lmer_confint_profile(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ci = result.confint(parm="Days", method="profile")
        assert "Days" in ci
        assert ci["Days"][0] < result.fixef()["Days"] < ci["Days"][1]

    def test_lmer_confint_boot(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ci = result.confint(parm="Days", method="boot", n_boot=50, seed=42)
        assert "Days" in ci
        assert ci["Days"][0] < ci["Days"][1]

    def test_profile_lmer(self) -> None:
        from mixedlm.inference import profile_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which="Days", n_points=10)

        assert "Days" in profiles
        profile = profiles["Days"]
        assert len(profile.values) == 10
        assert len(profile.zeta) == 10
        assert profile.ci_lower < profile.mle < profile.ci_upper

    def test_bootstrap_lmer(self) -> None:
        from mixedlm.inference import bootstrap_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        boot_result = bootstrap_lmer(result, n_boot=30, seed=42)

        assert boot_result.n_boot == 30
        assert boot_result.beta_samples.shape == (30, 2)
        se = boot_result.se()
        assert "(Intercept)" in se
        assert "Days" in se
        assert se["Days"] > 0

    def test_bootstrap_lmer_ci(self) -> None:
        from mixedlm.inference import bootstrap_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        boot_result = bootstrap_lmer(result, n_boot=30, seed=42)

        ci_pct = boot_result.ci(method="percentile")
        ci_basic = boot_result.ci(method="basic")
        ci_normal = boot_result.ci(method="normal")

        assert "Days" in ci_pct
        assert "Days" in ci_basic
        assert "Days" in ci_normal

    def test_glmer_confint_wald(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        ci = result.confint(method="Wald")
        assert "(Intercept)" in ci
        assert ci["(Intercept)"][0] < result.fixef()["(Intercept)"] < ci["(Intercept)"][1]

    def test_glmer_confint_profile(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        ci = result.confint(parm="(Intercept)", method="profile")
        assert "(Intercept)" in ci
        assert ci["(Intercept)"][0] < ci["(Intercept)"][1]

    def test_bootstrap_glmer(self) -> None:
        from mixedlm.inference import bootstrap_glmer

        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        boot_result = bootstrap_glmer(result, n_boot=20, seed=42)
        assert boot_result.n_boot == 20
        assert boot_result.sigma_samples is None

    def test_profile_result_summary(self) -> None:
        from mixedlm.inference import bootstrap_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        boot_result = bootstrap_lmer(result, n_boot=30, seed=42)

        summary = boot_result.summary()
        assert "Parametric bootstrap" in summary
        assert "30 samples" in summary

    def test_anova_lmer(self) -> None:
        model1 = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, REML=False)
        model2 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        result = anova(model1, model2)

        assert len(result.models) == 2
        assert result.chi_sq[0] is None
        assert result.chi_sq[1] is not None
        assert result.chi_sq[1] > 0
        assert result.chi_df[1] == 1
        assert result.p_value[1] is not None
        assert 0 <= result.p_value[1] <= 1

    def test_anova_multiple_models(self) -> None:
        model1 = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, REML=False)
        model2 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        model3 = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, REML=False)

        result = anova(model1, model2, model3)

        assert len(result.models) == 3
        assert all(aic > 0 for aic in result.aic)
        assert all(bic > 0 for bic in result.bic)

    def test_anova_output(self) -> None:
        model1 = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, REML=False)
        model2 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        result = anova(model1, model2)
        output = str(result)

        assert "AIC" in output
        assert "BIC" in output
        assert "logLik" in output
        assert "Chisq" in output

    def test_anova_glmer(self) -> None:
        model1 = glmer("y ~ 1 + (1 | herd)", CBPP, family=families.Binomial())
        model2 = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        result = anova(model1, model2)

        assert len(result.models) == 2
        assert result.chi_sq[1] is not None
        assert result.chi_df[1] == 3

    def test_lmer_simulate_single(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        y_sim = result.simulate(nsim=1, seed=42)

        assert y_sim.shape == (180,)
        assert np.isfinite(y_sim).all()

    def test_lmer_simulate_multiple(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        y_sim = result.simulate(nsim=10, seed=42)

        assert y_sim.shape == (180, 10)
        assert np.isfinite(y_sim).all()

    def test_lmer_simulate_no_re(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        y_sim_re = result.simulate(nsim=1, seed=42, use_re=True)
        y_sim_no_re = result.simulate(nsim=1, seed=42, use_re=False)

        assert not np.allclose(y_sim_re, y_sim_no_re)

    def test_lmer_simulate_reproducible(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        y_sim1 = result.simulate(nsim=1, seed=123)
        y_sim2 = result.simulate(nsim=1, seed=123)

        assert np.allclose(y_sim1, y_sim2)

    def test_glmer_simulate_binomial(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        y_sim = result.simulate(nsim=5, seed=42)

        assert y_sim.shape == (56, 5)
        assert np.all((y_sim == 0) | (y_sim == 1))

    def test_glmer_simulate_poisson(self) -> None:
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

        y_sim = result.simulate(nsim=1, seed=42)

        assert y_sim.shape == (n,)
        assert np.all(y_sim >= 0)
        assert np.all(y_sim == y_sim.astype(int))


class TestWeightsOffset:
    def test_lmer_weights(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        weights = np.abs(np.random.randn(n)) + 0.1

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result_unweighted = lmer("y ~ x + (1 | group)", data)
        result_weighted = lmer("y ~ x + (1 | group)", data, weights=weights)

        assert result_weighted.converged
        assert result_weighted.fixef()["x"] != result_unweighted.fixef()["x"]
        assert len(result_weighted.fitted()) == n
        assert len(result_weighted.residuals()) == n

    def test_lmer_offset(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        offset_vals = np.random.randn(n) * 0.5
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + offset_vals + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = lmer("y ~ x + (1 | group)", data, offset=offset_vals)

        assert result.converged
        fitted = result.fitted()
        assert len(fitted) == n

    def test_glmer_weights(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p)

        weights = np.abs(np.random.randn(n)) + 0.1

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), weights=weights)

        assert result.converged
        assert len(result.fitted()) == n
        assert len(result.residuals()) == n

    def test_glmer_offset(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        log_exposure = np.random.randn(n) * 0.5
        group_effects = np.random.randn(n_groups) * 0.3
        eta = 0.5 + 0.3 * x + log_exposure + group_effects[group]
        y = np.random.poisson(np.exp(eta))

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = glmer(
            "y ~ x + (1 | group)",
            data,
            family=families.Poisson(),
            offset=log_exposure,
        )

        assert result.converged
        fitted = result.fitted()
        assert len(fitted) == n
        assert np.all(fitted > 0)

    def test_lmer_weights_and_offset(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        offset_vals = np.random.randn(n) * 0.5
        weights = np.abs(np.random.randn(n)) + 0.1
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + offset_vals + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = lmer("y ~ x + (1 | group)", data, weights=weights, offset=offset_vals)

        assert result.converged
        assert len(result.fitted()) == n


class TestRefit:
    def test_lmer_refit(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y1 = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y1, "x": x, "group": [str(g) for g in group]})
        result1 = lmer("y ~ x + (1 | group)", data)

        y2 = 3.0 + 2.0 * x + group_effects[group] + np.random.randn(n) * 0.5
        result2 = result1.refit(y2)

        assert result2.converged
        assert result2.fixef()["(Intercept)"] != result1.fixef()["(Intercept)"]
        assert result2.fixef()["x"] != result1.fixef()["x"]
        assert len(result2.fitted()) == n
        assert result2.matrices.n_obs == result1.matrices.n_obs

    def test_lmer_refit_simulated(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        y_sim = result.simulate(nsim=1, seed=123)
        result_refit = result.refit(y_sim)

        assert result_refit.converged
        assert len(result_refit.fitted()) == n

    def test_glmer_refit(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y1 = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y1, "x": x, "group": [str(g) for g in group]})
        result1 = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        y2 = np.random.binomial(1, p).astype(float)
        result2 = result1.refit(y2)

        assert result2.converged
        assert len(result2.fitted()) == n
        assert result2.matrices.n_obs == result1.matrices.n_obs

    def test_refit_wrong_length(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        with pytest.raises(ValueError, match="newresp has length"):
            result.refit(np.random.randn(n + 10))

    def test_lmer_refitML(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result_reml = lmer("y ~ x + (1 | group)", data, REML=True)

        assert result_reml.REML is True
        assert result_reml.isREML() is True
        assert result_reml.isGLMM() is False

        result_ml = result_reml.refitML()

        assert result_ml.REML is False
        assert result_ml.isREML() is False
        assert result_ml.converged
        assert abs(result_ml.fixef()["x"] - result_reml.fixef()["x"]) < 0.1

        result_ml2 = result_ml.refitML()
        assert result_ml2 is result_ml

    def test_lmer_refitML_ml_model(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result_ml = lmer("y ~ x + (1 | group)", data, REML=False)

        assert result_ml.REML is False
        result_ml2 = result_ml.refitML()
        assert result_ml2 is result_ml

    def test_glmer_refitML(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = -0.5 + 0.5 * x + group_effects[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        assert result.isREML() is False
        assert result.isGLMM() is True

        result2 = result.refitML()
        assert result2 is result


