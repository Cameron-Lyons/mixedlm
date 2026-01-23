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
from mixedlm.matrices import build_model_matrices
from mixedlm.models.control import GlmerControl, LmerControl

SLEEPSTUDY = pd.DataFrame(
    {
        "Reaction": [
            249.56,
            258.70,
            250.80,
            321.44,
            356.85,
            414.69,
            382.20,
            290.15,
            430.59,
            466.35,
            222.73,
            205.27,
            202.98,
            204.71,
            207.72,
            215.94,
            213.63,
            217.73,
            224.30,
            237.31,
            199.05,
            194.33,
            234.32,
            232.84,
            229.31,
            220.46,
            235.42,
            255.75,
            261.01,
            247.52,
            321.42,
            300.01,
            283.86,
            285.13,
            285.80,
            297.59,
            280.24,
            318.26,
            305.35,
            354.04,
            287.60,
            285.00,
            301.80,
            320.12,
            316.28,
            293.32,
            290.08,
            334.82,
            293.70,
            371.58,
            234.92,
            242.81,
            272.95,
            309.17,
            317.96,
            310.00,
            454.16,
            346.82,
            330.30,
            253.92,
            283.84,
            289.00,
            276.77,
            299.80,
            297.50,
            338.10,
            340.80,
            305.55,
            354.10,
            357.70,
            265.35,
            276.37,
            243.43,
            254.72,
            279.59,
            284.26,
            305.60,
            331.84,
            335.45,
            377.32,
            241.58,
            273.94,
            254.44,
            270.04,
            251.47,
            254.38,
            245.37,
            235.70,
            235.98,
            249.55,
            312.17,
            313.59,
            291.64,
            346.12,
            365.16,
            391.84,
            404.29,
            416.56,
            455.86,
            458.96,
            292.63,
            308.52,
            324.28,
            320.90,
            305.30,
            350.56,
            300.01,
            327.33,
            335.63,
            392.03,
            290.14,
            262.46,
            253.66,
            267.51,
            296.00,
            304.56,
            350.78,
            369.47,
            364.88,
            370.63,
            263.99,
            289.65,
            276.77,
            299.09,
            297.43,
            310.73,
            287.17,
            329.61,
            334.48,
            343.22,
            237.45,
            301.79,
            311.90,
            282.73,
            285.05,
            240.09,
            275.19,
            238.49,
            266.54,
            207.72,
            286.95,
            288.54,
            245.18,
            276.11,
            266.42,
            250.13,
            269.84,
            281.05,
            284.78,
            306.72,
            271.98,
            268.70,
            257.72,
            266.66,
            310.04,
            309.18,
            327.21,
            347.79,
            341.82,
            373.73,
            346.00,
            344.00,
            358.00,
            399.00,
            363.00,
            400.00,
            416.00,
            376.00,
            441.00,
            466.00,
            269.41,
            273.47,
            297.60,
            310.63,
            287.17,
            329.61,
            334.48,
            343.22,
            369.14,
            364.06,
        ],
        "Days": list(range(10)) * 18,
        "Subject": [
            str(i)
            for i in [308] * 10
            + [309] * 10
            + [310] * 10
            + [330] * 10
            + [331] * 10
            + [332] * 10
            + [333] * 10
            + [334] * 10
            + [335] * 10
            + [337] * 10
            + [349] * 10
            + [350] * 10
            + [351] * 10
            + [352] * 10
            + [369] * 10
            + [370] * 10
            + [371] * 10
            + [372] * 10
        ],
    }
)


class TestFormulaParser:
    def test_simple_random_intercept(self) -> None:
        formula = parse_formula("y ~ x + (1 | group)")
        assert formula.response == "y"
        assert formula.fixed.has_intercept
        assert len(formula.random) == 1
        assert formula.random[0].grouping == "group"
        assert formula.random[0].correlated

    def test_random_slope(self) -> None:
        formula = parse_formula("y ~ x + (x | group)")
        assert len(formula.random) == 1
        assert formula.random[0].has_intercept

    def test_uncorrelated_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (x || group)")
        assert len(formula.random) == 1
        assert not formula.random[0].correlated

    def test_nested_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (1 | group/subgroup)")
        assert len(formula.random) == 1
        assert formula.random[0].is_nested
        assert formula.random[0].grouping == ("group", "subgroup")

    def test_crossed_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
        assert len(formula.random) == 2

    def test_no_intercept(self) -> None:
        formula = parse_formula("y ~ 0 + x + (1 | group)")
        assert not formula.fixed.has_intercept


class TestModelMatrices:
    def test_fixed_matrix_with_intercept(self) -> None:
        formula = parse_formula("Reaction ~ Days + (1 | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_obs == 180
        assert matrices.n_fixed == 2
        assert matrices.fixed_names == ["(Intercept)", "Days"]
        assert matrices.X.shape == (180, 2)
        assert np.allclose(matrices.X[:, 0], 1.0)

    def test_random_matrix(self) -> None:
        formula = parse_formula("Reaction ~ Days + (1 | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_random == 18
        assert matrices.Z.shape == (180, 18)

    def test_random_slope_matrix(self) -> None:
        formula = parse_formula("Reaction ~ Days + (Days | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_random == 36
        assert len(matrices.random_structures) == 1
        assert matrices.random_structures[0].n_terms == 2


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


CBPP = pd.DataFrame(
    {
        "incidence": [
            2,
            3,
            4,
            0,
            3,
            1,
            3,
            2,
            0,
            2,
            0,
            1,
            1,
            2,
            0,
            0,
            1,
            0,
            2,
            0,
            4,
            3,
            0,
            2,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            2,
            1,
            2,
            0,
            1,
            0,
            3,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
        ],
        "size": [
            14,
            12,
            9,
            5,
            22,
            18,
            21,
            22,
            16,
            16,
            20,
            10,
            10,
            9,
            6,
            18,
            25,
            24,
            13,
            11,
            10,
            5,
            6,
            8,
            3,
            3,
            5,
            3,
            2,
            2,
            10,
            8,
            4,
            2,
            14,
            11,
            9,
            8,
            4,
            5,
            7,
            3,
            7,
            8,
            4,
            4,
            2,
            4,
            6,
            5,
            5,
            3,
            3,
            2,
            2,
            2,
        ],
        "period": ["1", "2", "3", "4"] * 14,
        "herd": [
            str(i)
            for i in [1] * 4
            + [2] * 4
            + [3] * 4
            + [4] * 4
            + [5] * 4
            + [6] * 4
            + [7] * 4
            + [8] * 4
            + [9] * 4
            + [10] * 4
            + [11] * 4
            + [12] * 4
            + [13] * 4
            + [14] * 4
        ],
    }
)
CBPP["y"] = CBPP["incidence"] / CBPP["size"]


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


class TestAccessors:
    def test_lmer_nobs(self) -> None:
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

        assert result.nobs() == n

    def test_lmer_ngrps(self) -> None:
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

        ngrps = result.ngrps()
        assert "group" in ngrps
        assert ngrps["group"] == n_groups

    def test_lmer_get_sigma(self) -> None:
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

        assert result.get_sigma() == result.sigma
        assert result.get_sigma() > 0

    def test_lmer_df_residual(self) -> None:
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

        assert result.df_residual() == n - 2

    def test_glmer_accessors(self) -> None:
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

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        assert result.nobs() == n
        assert result.ngrps()["group"] == n_groups
        assert result.df_residual() == n - 2
        assert result.sigma == 1.0
        assert result.get_sigma() == 1.0

    def test_ngrps_multiple_grouping(self) -> None:
        np.random.seed(42)
        n = 200

        group1 = np.random.choice(10, n)
        group2 = np.random.choice(5, n)
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        data = pd.DataFrame(
            {
                "y": y,
                "x": x,
                "group1": [str(g) for g in group1],
                "group2": [str(g) for g in group2],
            }
        )
        result = lmer("y ~ x + (1 | group1) + (1 | group2)", data)

        ngrps = result.ngrps()
        assert "group1" in ngrps
        assert "group2" in ngrps
        assert ngrps["group1"] == 10
        assert ngrps["group2"] == 5


class TestGetME:
    def test_lmer_getME_matrices(self) -> None:
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

        X = result.getME("X")
        assert X.shape == (n, 2)

        Z = result.getME("Z")
        assert Z.shape[0] == n

        y_out = result.getME("y")
        assert len(y_out) == n
        np.testing.assert_array_equal(y_out, y)

    def test_lmer_getME_parameters(self) -> None:
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

        beta = result.getME("beta")
        assert len(beta) == 2
        np.testing.assert_array_equal(beta, result.beta)

        theta = result.getME("theta")
        np.testing.assert_array_equal(theta, result.theta)

        sigma = result.getME("sigma")
        assert sigma == result.sigma

        u = result.getME("u")
        np.testing.assert_array_equal(u, result.u)

    def test_lmer_getME_lambda(self) -> None:
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

        Lambda = result.getME("Lambda")
        assert Lambda.shape[0] == n_groups
        assert Lambda.shape[1] == n_groups

        Lambdat = result.getME("Lambdat")
        assert Lambdat.shape == Lambda.T.shape

    def test_lmer_getME_misc(self) -> None:
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

        assert result.getME("n_obs") == n
        assert result.getME("n_fixed") == 2
        assert result.getME("REML") is True
        assert result.getME("fixef_names") == ["(Intercept)", "x"]

    def test_lmer_getME_invalid(self) -> None:
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

        with pytest.raises(ValueError, match="Unknown component"):
            result.getME("invalid_component")

    def test_glmer_getME(self) -> None:
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

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        X = result.getME("X")
        assert X.shape == (n, 2)

        beta = result.getME("beta")
        np.testing.assert_array_equal(beta, result.beta)

        family = result.getME("family")
        assert family.__class__.__name__ == "Binomial"

        assert result.getME("nAGQ") == 1


class TestCondVar:
    def test_lmer_ranef_condVar(self) -> None:
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

        ranef_no_condVar = result.ranef(condVar=False)
        assert isinstance(ranef_no_condVar, dict)
        assert "group" in ranef_no_condVar

        ranef_with_condVar = result.ranef(condVar=True)
        assert hasattr(ranef_with_condVar, "values")
        assert hasattr(ranef_with_condVar, "condVar")
        assert ranef_with_condVar.condVar is not None
        assert "group" in ranef_with_condVar.condVar
        assert "(Intercept)" in ranef_with_condVar.condVar["group"]

        cond_var = ranef_with_condVar.condVar["group"]["(Intercept)"]
        assert len(cond_var) == n_groups
        assert all(v >= 0 for v in cond_var)

    def test_lmer_ranef_condVar_random_slope(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        ranef_with_condVar = result.ranef(condVar=True)
        assert ranef_with_condVar.condVar is not None
        assert "Subject" in ranef_with_condVar.condVar

        assert "(Intercept)" in ranef_with_condVar.condVar["Subject"]
        assert "Days" in ranef_with_condVar.condVar["Subject"]

        for term in ["(Intercept)", "Days"]:
            cond_var = ranef_with_condVar.condVar["Subject"][term]
            assert len(cond_var) == 18
            assert all(v >= 0 for v in cond_var)

    def test_ranef_result_dict_like(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ranef_result = result.ranef(condVar=True)

        assert "Subject" in ranef_result
        assert list(ranef_result.keys()) == ["Subject"]
        for group, terms in ranef_result.items():
            assert group == "Subject"
            assert "(Intercept)" in terms

    def test_glmer_ranef_condVar(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        ranef_no_condVar = result.ranef(condVar=False)
        assert isinstance(ranef_no_condVar, dict)
        assert "herd" in ranef_no_condVar

        ranef_with_condVar = result.ranef(condVar=True)
        assert ranef_with_condVar.condVar is not None
        assert "herd" in ranef_with_condVar.condVar
        assert "(Intercept)" in ranef_with_condVar.condVar["herd"]

        cond_var = ranef_with_condVar.condVar["herd"]["(Intercept)"]
        assert len(cond_var) == result.ngrps()["herd"]
        assert all(v >= 0 for v in cond_var)


class TestUpdate:
    def test_lmer_update_add_term(self) -> None:
        result1 = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY)

        result2 = result1.update(". ~ . + Days", data=SLEEPSTUDY)

        assert "Days" in result2.fixef()
        assert "(Intercept)" in result2.fixef()
        assert len(result2.fixef()) == 2

    def test_lmer_update_remove_term(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        result2 = result1.update(". ~ . - Days", data=SLEEPSTUDY)

        assert "Days" not in result2.fixef()
        assert "(Intercept)" in result2.fixef()
        assert len(result2.fixef()) == 1

    def test_lmer_update_replace_formula(self) -> None:
        result1 = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY)

        result2 = result1.update("Reaction ~ Days + (Days | Subject)", data=SLEEPSTUDY)

        assert "Days" in result2.fixef()
        assert "(Intercept)" in result2.fixef()

    def test_lmer_update_change_REML(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        assert result1.REML is True

        result2 = result1.update(data=SLEEPSTUDY, REML=False)
        assert result2.REML is False

    def test_lmer_update_keep_response(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        result2 = result1.update(". ~ 1 + (1 | Subject)", data=SLEEPSTUDY)

        assert result2.formula.response == "Reaction"
        assert "Days" not in result2.fixef()

    def test_glmer_update_add_term(self) -> None:
        result1 = glmer("y ~ 1 + (1 | herd)", CBPP, family=families.Binomial())

        result2 = result1.update(". ~ . + period", data=CBPP)

        assert any("period" in k for k in result2.fixef())
        assert "(Intercept)" in result2.fixef()

    def test_glmer_update_change_family(self) -> None:
        np.random.seed(42)
        n_groups = 8
        n_per_group = 15
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.3
        eta = 0.5 + 0.3 * x + group_effects[group]
        mu = np.exp(eta)
        y = np.random.poisson(mu)

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result1 = glmer("y ~ x + (1 | group)", data, family=families.Poisson())

        result2 = result1.update(data=data, family=families.Poisson())
        assert result2.family.__class__.__name__ == "Poisson"

    def test_update_uses_stored_data(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        result2 = result.update(". ~ . + 1")
        assert result2.converged


class TestUpdateFormula:
    def test_update_formula_add_variable(self) -> None:
        from mixedlm.formula.parser import parse_formula, update_formula

        old = parse_formula("y ~ x + (1 | group)")
        new = update_formula(old, ". ~ . + z")

        assert str(new) == "y ~ x + z + (1 | group)"

    def test_update_formula_remove_variable(self) -> None:
        from mixedlm.formula.parser import parse_formula, update_formula

        old = parse_formula("y ~ x + z + (1 | group)")
        new = update_formula(old, ". ~ . - z")

        assert "z" not in str(new)
        assert "x" in str(new)

    def test_update_formula_change_response(self) -> None:
        from mixedlm.formula.parser import parse_formula, update_formula

        old = parse_formula("y ~ x + (1 | group)")
        new = update_formula(old, "z ~ .")

        assert new.response == "z"

    def test_update_formula_replace_rhs(self) -> None:
        from mixedlm.formula.parser import parse_formula, update_formula

        old = parse_formula("y ~ x + (1 | group)")
        new = update_formula(old, ". ~ a + b + (1 | subject)")

        assert "a" in str(new)
        assert "b" in str(new)
        assert "subject" in str(new)
        assert new.response == "y"


class TestDrop1:
    def test_drop1_lmer_basic(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        drop1_result = result.drop1(data=SLEEPSTUDY)

        assert len(drop1_result.terms) == 1
        assert "Days" in drop1_result.terms
        assert drop1_result.lrt[0] is not None
        assert drop1_result.lrt[0] > 0
        assert drop1_result.p_value[0] is not None
        assert 0 <= drop1_result.p_value[0] <= 1

    def test_drop1_lmer_multiple_terms(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x1 + 0.8 * x2 + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": [str(g) for g in group]})
        result = lmer("y ~ x1 + x2 + (1 | group)", data, REML=False)

        drop1_result = result.drop1(data=data)

        assert len(drop1_result.terms) == 2
        assert "x1" in drop1_result.terms
        assert "x2" in drop1_result.terms
        assert all(lrt is not None for lrt in drop1_result.lrt)
        assert all(p is not None for p in drop1_result.p_value)

    def test_drop1_lmer_output(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        drop1_result = result.drop1(data=SLEEPSTUDY)
        output = str(drop1_result)

        assert "Single term deletions" in output
        assert "AIC" in output
        assert "LRT" in output
        assert "Days" in output

    def test_drop1_glmer_basic(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        drop1_result = result.drop1(data=CBPP)

        assert len(drop1_result.terms) >= 1
        assert any("period" in t for t in drop1_result.terms)
        assert drop1_result.full_model_aic > 0

    def test_drop1_via_inference_module(self) -> None:
        from mixedlm.inference import drop1_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        drop1_result = drop1_lmer(result, data=SLEEPSTUDY)

        assert len(drop1_result.terms) == 1
        assert "Days" in drop1_result.terms

    def test_drop1_aic_comparison(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        drop1_result = result.drop1(data=SLEEPSTUDY)

        assert drop1_result.aic[0] > drop1_result.full_model_aic


class TestIsSingular:
    def test_lmer_isSingular_returns_bool(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert isinstance(result.isSingular(), bool)

    def test_lmer_singular_with_high_tolerance(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.isSingular(tol=1e10) is True

    def test_lmer_not_singular_with_real_variance(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 30
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        group_effects = np.random.randn(n_groups) * 5.0
        x = np.random.randn(n)
        y = 10.0 + 2.0 * x + group_effects[group] + np.random.randn(n) * 1.0

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        if result.theta[0] > 0.1:
            assert result.isSingular(tol=0.01) is False

    def test_lmer_singular_when_theta_zero(self) -> None:
        np.random.seed(42)
        n_groups = 5
        n_per_group = 50
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (1 | group)", data)

        if result.theta[0] < 1e-4:
            assert result.isSingular() is True

    def test_glmer_isSingular_returns_bool(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        assert isinstance(result.isSingular(), bool)

    def test_glmer_singular_with_high_tolerance(self) -> None:
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())

        assert result.isSingular(tol=1e10) is True

    def test_singular_uncorrelated_random_effects(self) -> None:
        result = lmer("Reaction ~ Days + (Days || Subject)", SLEEPSTUDY)

        assert isinstance(result.isSingular(), bool)

    def test_isSingular_detects_near_zero_theta(self) -> None:
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

        result_singular = LmerResult(
            formula=parse_formula("y ~ 1 + (1 | g)"),
            matrices=matrices,
            theta=np.array([0.0]),
            beta=np.array([2.0]),
            sigma=1.0,
            u=np.zeros(3),
            deviance=10.0,
            REML=True,
            converged=True,
            n_iter=1,
        )
        assert result_singular.isSingular() is True

        result_not_singular = LmerResult(
            formula=parse_formula("y ~ 1 + (1 | g)"),
            matrices=matrices,
            theta=np.array([1.0]),
            beta=np.array([2.0]),
            sigma=1.0,
            u=np.zeros(3),
            deviance=10.0,
            REML=True,
            converged=True,
            n_iter=1,
        )
        assert result_not_singular.isSingular() is False


class TestAllFit:
    def test_allfit_lmer_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        assert len(allfit_result.fits) == 2
        assert "L-BFGS-B" in allfit_result.fits
        assert "Nelder-Mead" in allfit_result.fits

    def test_allfit_lmer_default_optimizers(self):
        from mixedlm.inference.allfit import _get_available_optimizers

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY)

        expected_count = len(_get_available_optimizers())
        assert len(allfit_result.fits) == expected_count

    def test_allfit_glmer_basic(self):
        result = glmer(
            "y ~ period + (1 | herd)",
            CBPP,
            family=families.Binomial(),
        )
        allfit_result = result.allFit(CBPP, optimizers=["L-BFGS-B", "Nelder-Mead"])

        assert len(allfit_result.fits) == 2
        assert "L-BFGS-B" in allfit_result.fits
        assert "Nelder-Mead" in allfit_result.fits

    def test_allfit_best_fit(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        best = allfit_result.best_fit()
        assert best is not None
        assert hasattr(best, "deviance")

        best_aic = allfit_result.best_fit(criterion="AIC")
        assert best_aic is not None

        best_bic = allfit_result.best_fit(criterion="BIC")
        assert best_bic is not None

    def test_allfit_is_consistent(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        consistent = allfit_result.is_consistent()
        assert isinstance(consistent, bool)

    def test_allfit_fixef_table(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        fixef_table = allfit_result.fixef_table()
        assert isinstance(fixef_table, dict)
        assert len(fixef_table) > 0
        for _opt_name, fixefs in fixef_table.items():
            assert "(Intercept)" in fixefs
            assert "Days" in fixefs

    def test_allfit_theta_table(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        theta_table = allfit_result.theta_table()
        assert isinstance(theta_table, dict)
        assert len(theta_table) > 0
        for _opt_name, thetas in theta_table.items():
            assert isinstance(thetas, list)

    def test_allfit_str_repr(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        str_output = str(allfit_result)
        assert "allFit summary:" in str_output
        assert "L-BFGS-B" in str_output
        assert "Nelder-Mead" in str_output

        repr_output = repr(allfit_result)
        assert "AllFitResult" in repr_output
        assert "successful" in repr_output


class TestVarCorr:
    def test_lmer_varcorr_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        assert "Subject" in vc.groups
        assert "(Intercept)" in vc.groups["Subject"].variance
        assert vc.residual > 0

    def test_lmer_varcorr_random_slope(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        assert "Subject" in vc.groups
        group = vc.groups["Subject"]
        assert "(Intercept)" in group.variance
        assert "Days" in group.variance
        assert group.corr is not None
        assert group.corr.shape == (2, 2)
        assert np.allclose(np.diag(group.corr), 1.0)

    def test_lmer_varcorr_uncorrelated(self):
        result = lmer("Reaction ~ Days + (Days || Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        assert "Subject" in vc.groups
        group = vc.groups["Subject"]
        assert "(Intercept)" in group.variance
        assert "Days" in group.variance
        assert group.corr is None

    def test_lmer_varcorr_cov_matrix(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        cov = vc.get_cov("Subject")
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)
        assert np.all(np.diag(cov) >= 0)

    def test_lmer_varcorr_as_dict(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        d = vc.as_dict()
        assert isinstance(d, dict)
        assert "Subject" in d
        assert "(Intercept)" in d["Subject"]

    def test_lmer_varcorr_str(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        str_output = str(vc)
        assert "Random effects:" in str_output
        assert "Subject" in str_output
        assert "(Intercept)" in str_output
        assert "Days" in str_output
        assert "Residual" in str_output

    def test_lmer_varcorr_repr(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        vc = result.VarCorr()

        repr_output = repr(vc)
        assert "VarCorr" in repr_output
        assert "1 groups" in repr_output

    def test_glmer_varcorr_basic(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        vc = result.VarCorr()

        assert "herd" in vc.groups
        assert "(Intercept)" in vc.groups["herd"].variance

    def test_glmer_varcorr_as_dict(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        vc = result.VarCorr()

        d = vc.as_dict()
        assert isinstance(d, dict)
        assert "herd" in d

    def test_glmer_varcorr_str(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        vc = result.VarCorr()

        str_output = str(vc)
        assert "Random effects:" in str_output
        assert "herd" in str_output


class TestLogLik:
    def test_lmer_loglik_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        assert ll.value < 0
        assert ll.df > 0
        assert ll.nobs == len(SLEEPSTUDY)
        assert ll.REML is True

    def test_lmer_loglik_ml(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        ll = result.logLik()

        assert ll.value < 0
        assert ll.REML is False

    def test_lmer_loglik_df(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        assert ll.df == 4

    def test_lmer_loglik_str(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        str_output = str(ll)
        assert "log Lik." in str_output
        assert "df=" in str_output
        assert "REML" in str_output

    def test_lmer_loglik_repr(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        repr_output = repr(ll)
        assert "LogLik" in repr_output
        assert "value=" in repr_output
        assert "df=" in repr_output

    def test_lmer_loglik_float(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        assert float(ll) == ll.value

    def test_lmer_aic_bic_consistency(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        expected_aic = -2 * ll.value + 2 * ll.df
        expected_bic = -2 * ll.value + ll.df * np.log(ll.nobs)

        assert np.isclose(result.AIC(), expected_aic)
        assert np.isclose(result.BIC(), expected_bic)

    def test_glmer_loglik_basic(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        ll = result.logLik()

        assert ll.value < 0
        assert ll.df > 0
        assert ll.nobs == len(CBPP)
        assert ll.REML is False

    def test_glmer_loglik_df(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        ll = result.logLik()

        assert ll.df == 5

    def test_glmer_aic_bic_consistency(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        ll = result.logLik()

        expected_aic = -2 * ll.value + 2 * ll.df
        expected_bic = -2 * ll.value + ll.df * np.log(ll.nobs)

        assert np.isclose(result.AIC(), expected_aic)
        assert np.isclose(result.BIC(), expected_bic)


class TestDeviance:
    def test_lmer_get_deviance(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        dev = result.get_deviance()

        assert isinstance(dev, float)
        assert dev > 0
        assert dev == result.deviance

    def test_lmer_remlcrit_reml(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        crit = result.REMLcrit()

        assert isinstance(crit, float)
        assert crit > 0
        assert crit == result.deviance
        assert result.isREML()

    def test_lmer_remlcrit_ml(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        crit = result.REMLcrit()

        assert isinstance(crit, float)
        assert crit > 0
        assert crit == result.deviance
        assert not result.isREML()

    def test_lmer_deviance_consistency(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ll = result.logLik()

        assert result.get_deviance() == result.REMLcrit()
        expected = result.deviance + (result.matrices.n_obs - result.matrices.n_fixed) * np.log(
            2 * np.pi
        )
        assert np.isclose(-2 * ll.value, expected, rtol=1e-6)

    def test_glmer_get_deviance(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        dev = result.get_deviance()

        assert isinstance(dev, float)
        assert dev > 0
        assert dev == result.deviance

    def test_glmer_remlcrit(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        crit = result.REMLcrit()

        assert isinstance(crit, float)
        assert crit > 0
        assert crit == result.deviance
        assert not result.isREML()

    def test_glmer_deviance_loglik_relation(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        ll = result.logLik()

        assert np.isclose(-2 * ll.value, result.deviance, rtol=1e-6)


class TestModelMatrix:
    def test_lmer_model_matrix_fixed(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        X = result.model_matrix("fixed")

        assert X.shape[0] == len(SLEEPSTUDY)
        assert X.shape[1] == 2

    def test_lmer_model_matrix_random(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Z = result.model_matrix("random")

        assert Z.shape[0] == len(SLEEPSTUDY)
        assert Z.shape[1] == 18

    def test_lmer_model_matrix_both(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        X, Z = result.model_matrix("both")

        assert X.shape[0] == len(SLEEPSTUDY)
        assert Z.shape[0] == len(SLEEPSTUDY)

    def test_lmer_model_matrix_aliases(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        X1 = result.model_matrix("fixed")
        X2 = result.model_matrix("X")
        assert np.allclose(X1, X2)

        Z1 = result.model_matrix("random")
        Z2 = result.model_matrix("Z")
        assert (Z1 != Z2).nnz == 0

    def test_glmer_model_matrix_fixed(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        X = result.model_matrix("fixed")

        assert X.shape[0] == len(CBPP)
        assert X.shape[1] == 4

    def test_glmer_model_matrix_random(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        Z = result.model_matrix("random")

        assert Z.shape[0] == len(CBPP)


class TestTerms:
    def test_lmer_terms_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        t = result.terms()

        assert t.response == "Reaction"
        assert "(Intercept)" in t.fixed_terms
        assert "Days" in t.fixed_terms
        assert "Subject" in t.random_terms
        assert "(Intercept)" in t.random_terms["Subject"]

    def test_lmer_terms_variables(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        t = result.terms()

        assert "Days" in t.fixed_variables
        assert "Subject" in t.grouping_factors
        assert t.has_intercept

    def test_lmer_terms_random_slope(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        t = result.terms()

        assert "(Intercept)" in t.random_terms["Subject"]
        assert "Days" in t.random_terms["Subject"]
        assert "Days" in t.random_variables

    def test_lmer_terms_str(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        t = result.terms()

        output = str(t)
        assert "Response" in output
        assert "Reaction" in output
        assert "Fixed effects" in output

    def test_lmer_get_formula(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        f = result.get_formula()

        assert f.response == "Reaction"
        assert str(f) == "Reaction ~ Days + (1 | Subject)"

    def test_glmer_terms_basic(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        t = result.terms()

        assert t.response == "y"
        assert "(Intercept)" in t.fixed_terms
        assert "herd" in t.random_terms
        assert "herd" in t.grouping_factors

    def test_glmer_get_formula(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        f = result.get_formula()

        assert f.response == "y"


class TestFormulaUtilities:
    def test_nobars_simple(self):
        f = nobars("y ~ x + (1 | group)")

        assert f.response == "y"
        assert len(f.random) == 0
        assert f.fixed.has_intercept

    def test_nobars_multiple_random(self):
        f = nobars("y ~ x + z + (x | group) + (1 | subject)")

        assert len(f.random) == 0
        assert "x" in str(f)
        assert "z" in str(f)

    def test_nobars_with_formula_object(self):
        original = parse_formula("y ~ x + (1 | group)")
        f = nobars(original)

        assert len(f.random) == 0
        assert f.response == original.response
        assert f.fixed == original.fixed

    def test_findbars_simple(self):
        bars = findbars("y ~ x + (1 | group)")

        assert len(bars) == 1
        assert bars[0].grouping == "group"
        assert bars[0].has_intercept

    def test_findbars_multiple(self):
        bars = findbars("y ~ x + (x | group) + (1 | subject)")

        assert len(bars) == 2
        groupings = {b.grouping for b in bars}
        assert "group" in groupings
        assert "subject" in groupings

    def test_findbars_with_formula_object(self):
        original = parse_formula("y ~ x + (x | group)")
        bars = findbars(original)

        assert len(bars) == 1
        assert bars[0].grouping == "group"

    def test_findbars_no_random(self):
        bars = findbars("y ~ x + z")

        assert len(bars) == 0

    def test_subbars_simple(self):
        result = subbars("y ~ x + (1 | group)")

        assert "group" in result
        assert "|" not in result
        assert "y ~" in result

    def test_subbars_with_slope(self):
        result = subbars("y ~ x + (x | group)")

        assert "group" in result
        assert "group:x" in result
        assert "|" not in result

    def test_is_mixed_formula_true(self):
        assert is_mixed_formula("y ~ x + (1 | group)")
        assert is_mixed_formula("y ~ x + (x | group) + (1 | subject)")

    def test_is_mixed_formula_false(self):
        assert not is_mixed_formula("y ~ x")
        assert not is_mixed_formula("y ~ x + z")

    def test_is_mixed_formula_with_object(self):
        mixed = parse_formula("y ~ x + (1 | group)")
        not_mixed = parse_formula("y ~ x + z")

        assert is_mixed_formula(mixed)
        assert not is_mixed_formula(not_mixed)

    def test_nobars_preserves_fixed_structure(self):
        f = nobars("y ~ x * z + (1 | group)")

        assert f.fixed.has_intercept
        fixed_str = str(f)
        assert "x" in fixed_str
        assert "z" in fixed_str

    def test_findbars_uncorrelated(self):
        bars = findbars("y ~ x + (x || group)")

        assert len(bars) == 1
        assert not bars[0].correlated

    def test_findbars_nested(self):
        bars = findbars("y ~ x + (1 | group/subgroup)")

        assert len(bars) == 1
        assert bars[0].is_nested
        assert bars[0].grouping_factors == ("group", "subgroup")


class TestCoef:
    def test_lmer_coef_basic(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        coef = result.coef()

        assert "Subject" in coef
        assert "(Intercept)" in coef["Subject"]
        assert len(coef["Subject"]["(Intercept)"]) == 18

    def test_lmer_coef_combines_fixed_and_random(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        coef = result.coef()
        fixef = result.fixef()
        ranef = result.ranef()

        for i in range(len(ranef["Subject"]["(Intercept)"])):
            expected = fixef["(Intercept)"] + ranef["Subject"]["(Intercept)"][i]
            assert np.isclose(coef["Subject"]["(Intercept)"][i], expected)

    def test_lmer_coef_random_slope(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        coef = result.coef()

        assert "Subject" in coef
        assert "(Intercept)" in coef["Subject"]
        assert "Days" in coef["Subject"]

        fixef = result.fixef()
        ranef = result.ranef()

        for i in range(len(ranef["Subject"]["Days"])):
            expected_days = fixef["Days"] + ranef["Subject"]["Days"][i]
            assert np.isclose(coef["Subject"]["Days"][i], expected_days)

    def test_glmer_coef_basic(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        coef = result.coef()

        assert "herd" in coef
        assert "(Intercept)" in coef["herd"]

    def test_glmer_coef_combines_fixed_and_random(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        coef = result.coef()
        fixef = result.fixef()
        ranef = result.ranef()

        for i in range(len(ranef["herd"]["(Intercept)"])):
            expected = fixef["(Intercept)"] + ranef["herd"]["(Intercept)"][i]
            assert np.isclose(coef["herd"]["(Intercept)"][i], expected)


class TestPredict:
    def test_lmer_predict_no_newdata(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred = result.predict()
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_lmer_predict_same_data(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred = result.predict(newdata=SLEEPSTUDY)
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_lmer_predict_fixed_only(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        pred_fixed = result.predict(newdata=SLEEPSTUDY, re_form="NA")

        fixef = result.fixef()
        expected_fixed = fixef["(Intercept)"] + fixef["Days"] * SLEEPSTUDY["Days"].values
        assert np.allclose(pred_fixed, expected_fixed)

    def test_lmer_predict_new_levels_error(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        with pytest.raises(ValueError, match="New level"):
            result.predict(newdata=new_data)

    def test_lmer_predict_new_levels_allowed(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": ["999"]})

        pred = result.predict(newdata=new_data, allow_new_levels=True)
        fixef = result.fixef()
        expected = fixef["(Intercept)"] + fixef["Days"] * 5.0

        assert np.isclose(pred[0], expected)

    def test_lmer_predict_random_slope(self):
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        subject = SLEEPSTUDY["Subject"].iloc[0]
        new_data = pd.DataFrame({"Reaction": [300.0], "Days": [5.0], "Subject": [subject]})

        pred = result.predict(newdata=new_data)
        pred_fixed = result.predict(newdata=new_data, re_form="NA")

        assert pred[0] != pred_fixed[0]

    def test_glmer_predict_no_newdata(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred = result.predict()
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_glmer_predict_same_data(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred = result.predict(newdata=CBPP)
        fitted = result.fitted()

        assert np.allclose(pred, fitted)

    def test_glmer_predict_fixed_only(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred_fixed = result.predict(newdata=CBPP, re_form="NA")
        pred_full = result.predict(newdata=CBPP)

        assert not np.allclose(pred_fixed, pred_full)

    def test_glmer_predict_link_scale(self):
        result = glmer("y ~ period + (1 | herd)", CBPP, family=families.Binomial())
        pred_response = result.predict(newdata=CBPP, type="response")
        pred_link = result.predict(newdata=CBPP, type="link")

        assert np.all(pred_response >= 0) and np.all(pred_response <= 1)
        assert not np.all(pred_link >= 0) or not np.all(pred_link <= 1)

    def test_glmer_predict_new_levels_allowed(self):
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

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

        new_data = pd.DataFrame({"y": [0], "x": [0.5], "group": ["999"]})
        pred = result.predict(newdata=new_data, allow_new_levels=True)

        assert len(pred) == 1
        assert 0 <= pred[0] <= 1


class TestNAAction:
    def test_lmer_na_omit(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan
        data.loc[10, "group"] = np.nan

        result = lmer("y ~ x + (1 | group)", data, na_action="omit")

        assert result.matrices.n_obs == n - 3
        assert len(result.fitted()) == n - 3
        assert len(result.residuals()) == n - 3
        assert result.converged

    def test_lmer_na_exclude(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = lmer("y ~ x + (1 | group)", data, na_action="exclude")

        assert result.matrices.n_obs == n - 2

        fitted_vals = result.fitted()
        assert len(fitted_vals) == n
        assert np.isnan(fitted_vals[0])
        assert np.isnan(fitted_vals[5])
        assert not np.isnan(fitted_vals[1])

        resid = result.residuals()
        assert len(resid) == n
        assert np.isnan(resid[0])
        assert np.isnan(resid[5])

    def test_lmer_na_fail(self) -> None:
        np.random.seed(42)
        n = 50
        group = np.repeat(np.arange(5), 10)
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        data.loc[0, "y"] = np.nan

        with pytest.raises(ValueError, match="Missing values"):
            lmer("y ~ x + (1 | group)", data, na_action="fail")

    def test_lmer_no_na(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})

        result = lmer("y ~ x + (1 | group)", data, na_action="omit")

        assert result.matrices.n_obs == n
        assert len(result.fitted()) == n

    def test_glmer_na_omit(self) -> None:
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
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), na_action="omit")

        assert result.matrices.n_obs == n - 2
        assert result.converged

    def test_glmer_na_exclude(self) -> None:
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
        data.loc[0, "y"] = np.nan
        data.loc[5, "x"] = np.nan

        result = glmer("y ~ x + (1 | group)", data, family=families.Binomial(), na_action="exclude")

        assert result.matrices.n_obs == n - 2

        fitted_vals = result.fitted()
        assert len(fitted_vals) == n
        assert np.isnan(fitted_vals[0])
        assert np.isnan(fitted_vals[5])
        assert not np.isnan(fitted_vals[1])


class TestInfluenceDiagnostics:
    def test_lmer_hatvalues(self) -> None:
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

        h = result.hatvalues()

        assert len(h) == n
        assert np.all(h >= 0)
        assert np.all(h < 1)
        assert np.sum(h) > 0

    def test_lmer_cooks_distance(self) -> None:
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

        cooks_d = result.cooks_distance()

        assert len(cooks_d) == n
        assert np.all(cooks_d >= 0)
        assert np.all(np.isfinite(cooks_d))

    def test_lmer_influence(self) -> None:
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

        infl = result.influence()

        assert "hat" in infl
        assert "cooks_d" in infl
        assert "std_resid" in infl
        assert "student_resid" in infl

        assert len(infl["hat"]) == n
        assert len(infl["cooks_d"]) == n
        assert len(infl["std_resid"]) == n
        assert len(infl["student_resid"]) == n

    def test_lmer_hatvalues_sum_constraint(self) -> None:
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

        h = result.hatvalues()

        assert np.sum(h) > 0
        assert np.mean(h) > 0
        assert np.mean(h) < 1

    def test_glmer_hatvalues(self) -> None:
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

        h = result.hatvalues()

        assert len(h) == n
        assert np.all(h >= 0)
        assert np.all(h < 1)

    def test_glmer_cooks_distance(self) -> None:
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

        cooks_d = result.cooks_distance()

        assert len(cooks_d) == n
        assert np.all(cooks_d >= 0)
        assert np.all(np.isfinite(cooks_d))

    def test_glmer_influence(self) -> None:
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

        infl = result.influence()

        assert "hat" in infl
        assert "cooks_d" in infl
        assert "pearson_resid" in infl
        assert "deviance_resid" in infl

        assert len(infl["hat"]) == n
        assert len(infl["cooks_d"]) == n


class TestRePCA:
    def test_repca_basic(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        pca = result.rePCA()

        assert "group" in pca.groups
        group_pca = pca["group"]
        assert group_pca.n_terms == 2
        assert len(group_pca.sdev) == 2
        assert len(group_pca.proportion) == 2
        assert len(group_pca.cumulative) == 2

        assert np.all(group_pca.sdev >= 0)
        assert np.all(group_pca.proportion >= 0)
        assert np.all(group_pca.proportion <= 1)
        assert np.isclose(group_pca.cumulative[-1], 1.0, atol=1e-6) or group_pca.cumulative[-1] == 0

    def test_repca_single_term(self) -> None:
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

        pca = result.rePCA()

        assert pca["group"].n_terms == 1
        assert len(pca["group"].sdev) == 1
        assert pca["group"].sdev[0] >= 0

    def test_repca_is_singular(self) -> None:
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

        pca = result.rePCA()
        singular = pca.is_singular()

        assert isinstance(singular, dict)
        assert "group" in singular
        assert isinstance(singular["group"], (bool, np.bool_))

    def test_repca_str_output(self) -> None:
        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_effects = np.random.randn(n_groups) * 0.5
        y = 2.0 + 1.5 * x + group_effects[group] + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        pca = result.rePCA()
        output = str(pca)

        assert "Random effect PCA" in output
        assert "group" in output
        assert "PC1" in output
        assert "PC2" in output

    def test_glmer_repca(self) -> None:
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

        pca = result.rePCA()

        assert "group" in pca.groups
        assert pca["group"].n_terms == 1


class TestDotplot:
    def test_dotplot_basic(self) -> None:
        pytest.importorskip("matplotlib")

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

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dotplot_multiple_terms(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_dotplot_specific_term(self) -> None:
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        n_groups = 10
        n_per_group = 20
        n = n_groups * n_per_group

        group = np.repeat(np.arange(n_groups), n_per_group)
        x = np.random.randn(n)
        group_int = np.random.randn(n_groups) * 0.5
        group_slope = np.random.randn(n_groups) * 0.3
        y = 2.0 + 1.5 * x + group_int[group] + group_slope[group] * x + np.random.randn(n) * 0.5

        data = pd.DataFrame({"y": y, "x": x, "group": [str(g) for g in group]})
        result = lmer("y ~ x + (x | group)", data)

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot(term="(Intercept)")
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_glmer_dotplot(self) -> None:
        pytest.importorskip("matplotlib")

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

        import matplotlib

        matplotlib.use("Agg")

        fig = result.dotplot()
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestContrasts:
    def test_treatment_contrasts_default(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data)

        fixed_names = result.matrices.fixed_names
        assert "(Intercept)" in fixed_names
        assert any("group" in name for name in fixed_names)
        assert len(fixed_names) == 4

    def test_sum_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "sum"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert "(Intercept)" in fixed_names

    def test_helmert_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 2, 3])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "helmert"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert result.converged

    def test_poly_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C", "D"], n // 4)
        subject = np.repeat(np.arange(12), n // 12)
        effects = np.array([0, 1, 4, 9])[np.searchsorted(["A", "B", "C", "D"], group)]
        y = np.random.randn(n) + effects

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": "poly"})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 4
        assert result.converged

    def test_custom_contrast_matrix(self) -> None:
        np.random.seed(42)
        n = 90
        group = np.repeat(["A", "B", "C"], n // 3)
        subject = np.repeat(np.arange(9), n // 9)
        y = np.random.randn(n) + np.array([0, 1, 2])[np.searchsorted(["A", "B", "C"], group)]

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        custom_contrasts = np.array([[-1, -1], [1, 0], [0, 1]], dtype=np.float64)

        result = lmer("y ~ group + (1 | subject)", data, contrasts={"group": custom_contrasts})

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 3
        assert result.converged

    def test_glmer_contrasts(self) -> None:
        np.random.seed(42)
        n = 120
        group = np.repeat(["A", "B", "C"], n // 3)
        subject = np.repeat(np.arange(12), n // 12)
        eta = np.array([-1.0, 0.0, 1.0])[np.searchsorted(["A", "B", "C"], group)]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p).astype(float)

        data = pd.DataFrame({"y": y, "group": group, "subject": [str(s) for s in subject]})

        result = glmer(
            "y ~ group + (1 | subject)",
            data,
            family=families.Binomial(),
            contrasts={"group": "sum"},
        )

        fixed_names = result.matrices.fixed_names
        assert len(fixed_names) == 3
        assert "(Intercept)" in fixed_names

    def test_contrasts_different_effects(self) -> None:
        from mixedlm.utils.contrasts import contr_sum, contr_treatment

        n = 3
        treatment_matrix = contr_treatment(n)
        sum_matrix = contr_sum(n)

        assert treatment_matrix.shape == (3, 2)
        assert sum_matrix.shape == (3, 2)
        assert not np.allclose(treatment_matrix, sum_matrix)

        assert np.allclose(treatment_matrix[0, :], [0, 0])
        assert np.allclose(sum_matrix[-1, :], [-1, -1])

    def test_contrasts_with_interactions(self) -> None:
        np.random.seed(42)
        n = 240
        group1 = np.tile(np.repeat(["A", "B"], n // 4), 2)
        group2 = np.repeat(["X", "Y"], n // 2)
        subject = np.repeat(np.arange(24), n // 24)
        y = np.random.randn(n)

        data = pd.DataFrame(
            {"y": y, "group1": group1, "group2": group2, "subject": [str(s) for s in subject]}
        )

        result = lmer(
            "y ~ group1 * group2 + (1 | subject)",
            data,
            contrasts={"group1": "sum", "group2": "treatment"},
        )

        assert result.converged
        assert "(Intercept)" in result.matrices.fixed_names


class TestControl:
    def test_lmer_control_default(self) -> None:
        ctrl = LmerControl()
        assert ctrl.optimizer == "bobyqa"
        assert ctrl.maxiter == 1000
        assert ctrl.ftol == 1e-8
        assert ctrl.gtol == 1e-5
        assert ctrl.boundary_tol == 1e-4
        assert ctrl.check_conv is True
        assert ctrl.check_singular is True
        assert ctrl.use_rust is True

    def test_lmer_control_custom(self) -> None:
        ctrl = LmerControl(
            optimizer="Nelder-Mead",
            maxiter=2000,
            ftol=1e-6,
            boundary_tol=1e-5,
            check_singular=False,
        )
        assert ctrl.optimizer == "Nelder-Mead"
        assert ctrl.maxiter == 2000
        assert ctrl.ftol == 1e-6
        assert ctrl.boundary_tol == 1e-5
        assert ctrl.check_singular is False

    def test_lmer_control_invalid_optimizer(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimizer"):
            LmerControl(optimizer="invalid")

    def test_lmer_control_invalid_maxiter(self) -> None:
        with pytest.raises(ValueError, match="maxiter must be at least 1"):
            LmerControl(maxiter=0)

    def test_lmer_control_invalid_boundary_tol(self) -> None:
        with pytest.raises(ValueError, match="boundary_tol must be non-negative"):
            LmerControl(boundary_tol=-1)

    def test_lmer_control_function(self) -> None:
        ctrl = lmerControl(optimizer="BFGS", maxiter=500)
        assert isinstance(ctrl, LmerControl)
        assert ctrl.optimizer == "BFGS"
        assert ctrl.maxiter == 500

    def test_lmer_control_scipy_options(self) -> None:
        ctrl = LmerControl(optimizer="L-BFGS-B", maxiter=500, gtol=1e-4, ftol=1e-7)
        options = ctrl.get_scipy_options()
        assert options["maxiter"] == 500
        assert options["gtol"] == 1e-4
        assert options["ftol"] == 1e-7

    def test_lmer_with_control(self) -> None:
        ctrl = lmerControl(maxiter=100, check_singular=False)
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged
        assert result.fixef is not None

    def test_lmer_control_nelder_mead(self) -> None:
        ctrl = lmerControl(optimizer="Nelder-Mead", maxiter=2000)
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged

    def test_glmer_control_default(self) -> None:
        ctrl = GlmerControl()
        assert ctrl.optimizer == "bobyqa"
        assert ctrl.maxiter == 1000
        assert ctrl.tolPwrss == 1e-7
        assert ctrl.compDev is True
        assert ctrl.nAGQ0initStep is True

    def test_glmer_control_custom(self) -> None:
        ctrl = GlmerControl(optimizer="BFGS", maxiter=500, tolPwrss=1e-6, nAGQ0initStep=False)
        assert ctrl.optimizer == "BFGS"
        assert ctrl.maxiter == 500
        assert ctrl.tolPwrss == 1e-6
        assert ctrl.nAGQ0initStep is False

    def test_glmer_control_invalid_tolPwrss(self) -> None:
        with pytest.raises(ValueError, match="tolPwrss must be positive"):
            GlmerControl(tolPwrss=0)

    def test_glmer_control_function(self) -> None:
        ctrl = glmerControl(optimizer="BFGS", tolPwrss=1e-6)
        assert isinstance(ctrl, GlmerControl)
        assert ctrl.optimizer == "BFGS"
        assert ctrl.tolPwrss == 1e-6

    def test_glmer_with_control(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        ctrl = glmerControl(maxiter=100, check_singular=False)
        result = glmer(
            "y ~ period + (1 | herd)",
            data,
            family=families.Binomial(),
            weights=data["size"].values,
            control=ctrl,
        )
        assert result.converged

    def test_lmer_control_opt_ctrl(self) -> None:
        ctrl = lmerControl(optCtrl={"disp": True})
        options = ctrl.get_scipy_options()
        assert options.get("disp") is True

    def test_glmer_control_opt_ctrl(self) -> None:
        ctrl = glmerControl(optCtrl={"disp": True})
        options = ctrl.get_scipy_options()
        assert options.get("disp") is True

    def test_lmer_control_repr(self) -> None:
        ctrl = LmerControl(optimizer="BFGS", maxiter=500)
        repr_str = repr(ctrl)
        assert "BFGS" in repr_str
        assert "500" in repr_str

    def test_glmer_control_repr(self) -> None:
        ctrl = GlmerControl(optimizer="BFGS", maxiter=500, tolPwrss=1e-6)
        repr_str = repr(ctrl)
        assert "BFGS" in repr_str
        assert "500" in repr_str
        assert "1e-06" in repr_str

    def test_lmer_control_bobyqa_valid(self) -> None:
        ctrl = LmerControl(optimizer="bobyqa")
        assert ctrl.optimizer == "bobyqa"

    def test_glmer_control_bobyqa_valid(self) -> None:
        ctrl = GlmerControl(optimizer="bobyqa")
        assert ctrl.optimizer == "bobyqa"


class TestBobyqaOptimizer:
    @pytest.fixture
    def has_bobyqa(self) -> bool:
        from mixedlm.estimation.optimizers import has_bobyqa

        return has_bobyqa()

    def test_lmer_bobyqa(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        ctrl = lmerControl(optimizer="bobyqa")
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged
        assert result.fixef() is not None
        assert result.sigma > 0

    def test_lmer_bobyqa_random_slope(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        ctrl = lmerControl(optimizer="bobyqa")
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged
        fe = result.fixef()
        assert "(Intercept)" in fe
        assert "Days" in fe

    def test_lmer_bobyqa_optctrl(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        ctrl = lmerControl(optimizer="bobyqa", optCtrl={"rhobeg": 0.5, "rhoend": 1e-4})
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged

    def test_glmer_bobyqa(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        ctrl = glmerControl(optimizer="bobyqa")
        result = glmer(
            "y ~ period + (1 | herd)",
            data,
            family=families.Binomial(),
            weights=data["size"].values,
            control=ctrl,
        )
        assert result.converged
        assert result.fixef() is not None

    def test_bobyqa_vs_lbfgsb_consistency(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        ctrl_bobyqa = lmerControl(optimizer="bobyqa")
        ctrl_lbfgsb = lmerControl(optimizer="L-BFGS-B")

        result_bobyqa = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl_bobyqa)
        result_lbfgsb = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl_lbfgsb)

        assert abs(result_bobyqa.deviance - result_lbfgsb.deviance) < 0.1
        fe_bobyqa = result_bobyqa.fixef()
        fe_lbfgsb = result_lbfgsb.fixef()
        assert abs(fe_bobyqa["(Intercept)"] - fe_lbfgsb["(Intercept)"]) < 1.0
        assert abs(fe_bobyqa["Days"] - fe_lbfgsb["Days"]) < 0.5

    def test_has_bobyqa_function(self) -> None:
        from mixedlm.estimation.optimizers import has_bobyqa

        result = has_bobyqa()
        assert isinstance(result, bool)

    def test_allfit_includes_bobyqa(self, has_bobyqa: bool) -> None:
        if not has_bobyqa:
            pytest.skip("pybobyqa not installed")

        from mixedlm.inference.allfit import _get_available_optimizers

        optimizers = _get_available_optimizers()
        assert "bobyqa" in optimizers


class TestNloptOptimizer:
    @pytest.fixture
    def has_nlopt(self) -> bool:
        from mixedlm.estimation.optimizers import has_nlopt

        return has_nlopt()

    def test_lmer_nlopt_bobyqa(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        ctrl = lmerControl(optimizer="nloptwrap_BOBYQA")
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged
        assert result.fixef() is not None
        assert result.sigma > 0

    def test_lmer_nlopt_neldermead(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        ctrl = lmerControl(optimizer="nloptwrap_NELDERMEAD")
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged

    def test_lmer_nlopt_sbplx(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        ctrl = lmerControl(optimizer="nloptwrap_SBPLX")
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)
        assert result.converged

    def test_glmer_nlopt_bobyqa(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        ctrl = glmerControl(optimizer="nloptwrap_BOBYQA")
        result = glmer(
            "y ~ period + (1 | herd)",
            data,
            family=families.Binomial(),
            weights=data["size"].values,
            control=ctrl,
        )
        assert result.converged

    def test_nlopt_vs_lbfgsb_consistency(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        ctrl_nlopt = lmerControl(optimizer="nloptwrap_BOBYQA")
        ctrl_lbfgsb = lmerControl(optimizer="L-BFGS-B")

        result_nlopt = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl_nlopt)
        result_lbfgsb = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl_lbfgsb)

        assert abs(result_nlopt.deviance - result_lbfgsb.deviance) < 0.1
        fe_nlopt = result_nlopt.fixef()
        fe_lbfgsb = result_lbfgsb.fixef()
        assert abs(fe_nlopt["(Intercept)"] - fe_lbfgsb["(Intercept)"]) < 1.0

    def test_has_nlopt_function(self) -> None:
        from mixedlm.estimation.optimizers import has_nlopt

        result = has_nlopt()
        assert isinstance(result, bool)

    def test_allfit_includes_nlopt(self, has_nlopt: bool) -> None:
        if not has_nlopt:
            pytest.skip("nlopt not installed")

        from mixedlm.inference.allfit import _get_available_optimizers

        optimizers = _get_available_optimizers()
        assert "nloptwrap_BOBYQA" in optimizers

    def test_nlopt_control_valid(self) -> None:
        ctrl = LmerControl(optimizer="nloptwrap_BOBYQA")
        assert ctrl.optimizer == "nloptwrap_BOBYQA"

        ctrl = GlmerControl(optimizer="nloptwrap_SBPLX")
        assert ctrl.optimizer == "nloptwrap_SBPLX"


class TestMkReTrms:
    def test_mkretrms_basic(self) -> None:
        from mixedlm.models.modular import mkReTrms

        re_terms = mkReTrms("Reaction ~ Days + (1|Subject)", SLEEPSTUDY)

        assert re_terms.Zt is not None
        assert re_terms.theta is not None
        assert re_terms.Lind is not None
        assert re_terms.Gp is not None
        assert "Subject" in re_terms.flist
        assert "Subject" in re_terms.cnms

    def test_mkretrms_dimensions(self) -> None:
        from mixedlm.models.modular import mkReTrms

        re_terms = mkReTrms("Reaction ~ Days + (1|Subject)", SLEEPSTUDY)

        n_subjects = SLEEPSTUDY["Subject"].nunique()
        assert re_terms.Zt.shape[0] == n_subjects
        assert re_terms.Zt.shape[1] == len(SLEEPSTUDY)
        assert re_terms.nl == [n_subjects]

    def test_mkretrms_random_slope(self) -> None:
        from mixedlm.models.modular import mkReTrms

        re_terms = mkReTrms("Reaction ~ Days + (Days|Subject)", SLEEPSTUDY)

        n_subjects = SLEEPSTUDY["Subject"].nunique()
        assert re_terms.Zt.shape[0] == n_subjects * 2
        assert len(re_terms.theta) == 3

    def test_mkretrms_multiple_grouping(self) -> None:
        from mixedlm.models.modular import mkReTrms

        np.random.seed(42)
        n = 100
        group1 = np.repeat(np.arange(10), 10).astype(str)
        group2 = np.tile(np.arange(5), 20).astype(str)
        y = np.random.randn(n)
        x = np.random.randn(n)
        data = pd.DataFrame({"y": y, "x": x, "g1": group1, "g2": group2})

        re_terms = mkReTrms("y ~ x + (1|g1) + (1|g2)", data)

        assert "g1" in re_terms.flist
        assert "g2" in re_terms.flist
        assert len(re_terms.nl) == 2


class TestSimulateFormula:
    def test_simulate_formula_basic(self) -> None:
        from mixedlm.models.modular import simulate_formula

        result = simulate_formula(
            "Reaction ~ Days + (1|Subject)",
            SLEEPSTUDY,
            beta=np.array([250.0, 10.0]),
            theta=np.array([1.0]),
            sigma=25.0,
            seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert "Reaction" in result.columns
        assert len(result) == len(SLEEPSTUDY)

    def test_simulate_formula_multiple_sims(self) -> None:
        from mixedlm.models.modular import simulate_formula

        result = simulate_formula(
            "Reaction ~ Days + (1|Subject)",
            SLEEPSTUDY,
            beta=np.array([250.0, 10.0]),
            theta=np.array([1.0]),
            sigma=25.0,
            nsim=5,
            seed=42,
        )

        assert isinstance(result, list)
        assert len(result) == 5
        for df in result:
            assert isinstance(df, pd.DataFrame)
            assert "Reaction" in df.columns

    def test_simulate_formula_with_dict_beta(self) -> None:
        from mixedlm.models.modular import simulate_formula

        result = simulate_formula(
            "Reaction ~ Days + (1|Subject)",
            SLEEPSTUDY,
            beta={"(Intercept)": 250.0, "Days": 10.0},
            theta=np.array([1.0]),
            sigma=25.0,
            seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert "Reaction" in result.columns

    def test_simulate_formula_reproducibility(self) -> None:
        from mixedlm.models.modular import simulate_formula

        result1 = simulate_formula(
            "Reaction ~ Days + (1|Subject)",
            SLEEPSTUDY,
            beta=np.array([250.0, 10.0]),
            theta=np.array([1.0]),
            sigma=25.0,
            seed=123,
        )

        result2 = simulate_formula(
            "Reaction ~ Days + (1|Subject)",
            SLEEPSTUDY,
            beta=np.array([250.0, 10.0]),
            theta=np.array([1.0]),
            sigma=25.0,
            seed=123,
        )

        np.testing.assert_array_equal(result1["Reaction"].values, result2["Reaction"].values)


class TestDevfun2:
    def test_devfun2_basic(self) -> None:
        from mixedlm.models.modular import devfun2, lFormula, mkLmerDevfun

        parsed = lFormula("Reaction ~ Days + (1|Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        profile_devfun = devfun2(devfun, result.theta)

        dev = profile_devfun(result.theta)
        assert isinstance(dev, float)
        assert np.isfinite(dev)

    def test_devfun2_which_parameter(self) -> None:
        from mixedlm.models.modular import devfun2, lFormula, mkLmerDevfun

        parsed = lFormula("Reaction ~ Days + (Days|Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        profile_devfun = devfun2(devfun, result.theta, which=[0])

        dev = profile_devfun(np.array([result.theta[0]]))
        assert isinstance(dev, float)
        assert np.isfinite(dev)

    def test_devfun2_profile_value(self) -> None:
        from mixedlm.models.modular import devfun2, lFormula, mkLmerDevfun

        parsed = lFormula("Reaction ~ Days + (1|Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        profile_devfun = devfun2(devfun, result.theta)

        dev_opt = profile_devfun(result.theta)
        dev_perturbed = profile_devfun(result.theta * 1.5)
        assert dev_perturbed >= dev_opt - 0.01


class TestModelFrame:
    def test_model_frame_basic(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        mf = result.model_frame()

        assert isinstance(mf, pd.DataFrame)
        assert "Reaction" in mf.columns
        assert "Days" in mf.columns
        assert "Subject" in mf.columns
        assert len(mf) == len(SLEEPSTUDY)

    def test_model_frame_multiple_random_effects(self) -> None:
        np.random.seed(42)
        n = 100
        group1 = np.repeat(np.arange(10), 10).astype(str)
        group2 = np.tile(np.arange(10), 10).astype(str)
        x = np.random.randn(n)
        y = np.random.randn(n)

        data = pd.DataFrame({"y": y, "x": x, "group1": group1, "group2": group2})

        result = lmer("y ~ x + (1 | group1) + (1 | group2)", data)
        mf = result.model_frame()

        assert "y" in mf.columns
        assert "x" in mf.columns
        assert "group1" in mf.columns
        assert "group2" in mf.columns
        assert len(mf) == n

    def test_model_frame_interaction(self) -> None:
        np.random.seed(42)
        n = 60
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        group = np.repeat(np.arange(6), 10).astype(str)
        y = np.random.randn(n)

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})

        result = lmer("y ~ x1 * x2 + (1 | group)", data)
        mf = result.model_frame()

        assert "y" in mf.columns
        assert "x1" in mf.columns
        assert "x2" in mf.columns
        assert "group" in mf.columns

    def test_model_frame_na_omit(self) -> None:
        data = SLEEPSTUDY.copy()
        data.loc[0, "Reaction"] = np.nan
        data.loc[5, "Days"] = np.nan

        result = lmer("Reaction ~ Days + (1 | Subject)", data, na_action="omit")
        mf = result.model_frame()

        assert len(mf) == len(SLEEPSTUDY) - 2
        assert not mf["Reaction"].isna().any()
        assert not mf["Days"].isna().any()

    def test_glmer_model_frame(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        mf = result.model_frame()

        assert isinstance(mf, pd.DataFrame)
        assert "y" in mf.columns
        assert "period" in mf.columns
        assert "herd" in mf.columns

    def test_model_frame_random_slope(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        mf = result.model_frame()

        assert "Reaction" in mf.columns
        assert "Days" in mf.columns
        assert "Subject" in mf.columns


class TestRanefCondVar:
    def test_ranef_condvar_basic(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ranef_no_var = result.ranef(condVar=False)
        assert isinstance(ranef_no_var, dict)

        ranef_with_var = result.ranef(condVar=True)
        assert hasattr(ranef_with_var, "condVar")
        assert ranef_with_var.condVar is not None

        assert "Subject" in ranef_with_var.values
        assert "Subject" in ranef_with_var.condVar

    def test_ranef_condvar_values(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ranef_result = result.ranef(condVar=True)

        cond_var = ranef_result.condVar["Subject"]["(Intercept)"]
        assert len(cond_var) == result.ngrps()["Subject"]
        assert np.all(cond_var >= 0)

    def test_ranef_condvar_random_slope(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        ranef_result = result.ranef(condVar=True)

        assert "(Intercept)" in ranef_result.condVar["Subject"]
        assert "Days" in ranef_result.condVar["Subject"]

        intercept_var = ranef_result.condVar["Subject"]["(Intercept)"]
        slope_var = ranef_result.condVar["Subject"]["Days"]

        assert len(intercept_var) == result.ngrps()["Subject"]
        assert len(slope_var) == result.ngrps()["Subject"]
        assert np.all(intercept_var >= 0)
        assert np.all(slope_var >= 0)

    def test_ranef_condvar_dict_interface(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ranef_result = result.ranef(condVar=True)

        assert "Subject" in ranef_result
        assert list(ranef_result.keys()) == ["Subject"]

        for _group, terms in ranef_result.items():
            assert "(Intercept)" in terms

    def test_glmer_ranef_condvar(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )

        ranef_result = result.ranef(condVar=True)
        assert ranef_result.condVar is not None
        assert "herd" in ranef_result.condVar

        cond_var = ranef_result.condVar["herd"]["(Intercept)"]
        assert len(cond_var) == result.ngrps()["herd"]
        assert np.all(cond_var >= 0)

    def test_ranef_result_export(self) -> None:
        from mixedlm import RanefResult

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        ranef_result = result.ranef(condVar=True)
        assert isinstance(ranef_result, RanefResult)


class TestModelTypeChecks:
    def test_lmer_is_lmm(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        assert result.isLMM() is True
        assert result.isGLMM() is False
        assert result.isNLMM() is False

    def test_glmer_is_glmm(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        assert result.isGLMM() is True
        assert result.isLMM() is False
        assert result.isNLMM() is False

    def test_nlmer_is_nlmm(self) -> None:
        np.random.seed(42)
        n_groups = 5
        n_per_group = 20
        x = np.tile(np.linspace(0, 10, n_per_group), n_groups)
        groups = np.repeat([f"g{i}" for i in range(n_groups)], n_per_group)

        asym = 200 + np.random.randn(n_groups) * 10
        xmid = 5 + np.random.randn(n_groups) * 0.5
        scal = 1.0

        y = np.zeros(len(x))
        for i in range(n_groups):
            mask = groups == f"g{i}"
            y[mask] = asym[i] / (1 + np.exp((xmid[i] - x[mask]) / scal))
        y += np.random.randn(len(y)) * 5

        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = nlmer(
            model=nlme.SSlogis(),
            data=data,
            x_var="x",
            y_var="y",
            group_var="group",
            random_params=[0, 1],
            start={"Asym": 200, "xmid": 5, "scal": 1},
        )
        assert result.isNLMM() is True
        assert result.isLMM() is False
        assert result.isGLMM() is False


class TestNpar:
    def test_lmer_npar_simple(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        npar = result.npar()

        n_fixed = 2
        n_theta = 1
        n_sigma = 1
        assert npar == n_fixed + n_theta + n_sigma

    def test_lmer_npar_random_slope(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        npar = result.npar()

        n_fixed = 2
        n_theta = 3
        n_sigma = 1
        assert npar == n_fixed + n_theta + n_sigma

    def test_lmer_npar_multiple_random(self) -> None:
        np.random.seed(42)
        n = 100
        group1 = np.repeat(np.arange(10), 10).astype(str)
        group2 = np.tile(np.arange(10), 10).astype(str)
        x = np.random.randn(n)
        y = np.random.randn(n)

        data = pd.DataFrame({"y": y, "x": x, "group1": group1, "group2": group2})

        result = lmer("y ~ x + (1 | group1) + (1 | group2)", data)
        npar = result.npar()

        n_fixed = 2
        n_theta = 2
        n_sigma = 1
        assert npar == n_fixed + n_theta + n_sigma

    def test_glmer_npar(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        npar = result.npar()

        n_fixed = 4
        n_theta = 1
        assert npar == n_fixed + n_theta

    def test_nlmer_npar(self) -> None:
        np.random.seed(42)
        n_groups = 5
        n_per_group = 20
        x = np.tile(np.linspace(0, 10, n_per_group), n_groups)
        groups = np.repeat([f"g{i}" for i in range(n_groups)], n_per_group)

        asym = 200 + np.random.randn(n_groups) * 10
        xmid = 5 + np.random.randn(n_groups) * 0.5
        scal = 1.0

        y = np.zeros(len(x))
        for i in range(n_groups):
            mask = groups == f"g{i}"
            y[mask] = asym[i] / (1 + np.exp((xmid[i] - x[mask]) / scal))
        y += np.random.randn(len(y)) * 5

        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = nlmer(
            model=nlme.SSlogis(),
            data=data,
            x_var="x",
            y_var="y",
            group_var="group",
            random_params=[0, 1],
            start={"Asym": 200, "xmid": 5, "scal": 1},
        )
        npar = result.npar()

        n_fixed = 3
        n_theta = 3
        n_sigma = 1
        assert npar == n_fixed + n_theta + n_sigma


class TestDfResidual:
    def test_lmer_df_residual(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        df_res = result.df_residual()

        n = len(SLEEPSTUDY)
        p = 2
        assert df_res == n - p

    def test_lmer_df_residual_multiple_fixed(self) -> None:
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        group = np.repeat(np.arange(10), 10).astype(str)
        y = np.random.randn(n)

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})

        result = lmer("y ~ x1 + x2 + (1 | group)", data)
        df_res = result.df_residual()

        assert df_res == n - 3

    def test_glmer_df_residual(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        df_res = result.df_residual()

        n = len(data)
        p = 4
        assert df_res == n - p

    def test_nlmer_df_residual(self) -> None:
        np.random.seed(42)
        n_groups = 5
        n_per_group = 20
        x = np.tile(np.linspace(0, 10, n_per_group), n_groups)
        groups = np.repeat([f"g{i}" for i in range(n_groups)], n_per_group)

        asym = 200 + np.random.randn(n_groups) * 10
        xmid = 5 + np.random.randn(n_groups) * 0.5
        scal = 1.0

        y = np.zeros(len(x))
        for i in range(n_groups):
            mask = groups == f"g{i}"
            y[mask] = asym[i] / (1 + np.exp((xmid[i] - x[mask]) / scal))
        y += np.random.randn(len(y)) * 5

        data = pd.DataFrame({"y": y, "x": x, "group": groups})

        result = nlmer(
            model=nlme.SSlogis(),
            data=data,
            x_var="x",
            y_var="y",
            group_var="group",
            random_params=[0, 1],
            start={"Asym": 200, "xmid": 5, "scal": 1},
        )

        n = len(x)
        p = 3
        assert result.df_residual() == n - p


try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestProfilePlotting:
    def test_profile_plot_basic(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.inference import profile_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["Days"], n_points=10)

        ax = profiles["Days"].plot()
        assert ax is not None
        plt.close("all")

    def test_profile_plot_density(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.inference import profile_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["Days"], n_points=10)

        ax = profiles["Days"].plot_density()
        assert ax is not None
        plt.close("all")

    def test_plot_profiles(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.inference import plot_profiles, profile_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["(Intercept)", "Days"], n_points=10)

        fig = plot_profiles(profiles)
        assert fig is not None
        plt.close("all")

    def test_splom_profiles(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.inference import profile_lmer, splom_profiles

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["(Intercept)", "Days"], n_points=10)

        fig = splom_profiles(profiles)
        assert fig is not None
        plt.close("all")

    def test_profile_plot_no_ci(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.inference import profile_lmer

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        profiles = profile_lmer(result, which=["Days"], n_points=10)

        ax = profiles["Days"].plot(show_ci=False, show_mle=False)
        assert ax is not None
        plt.close("all")


class TestAccessorsWeightsOffset:
    def test_lmer_weights_default(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        w = result.weights()

        assert len(w) == len(SLEEPSTUDY)
        assert np.allclose(w, 1.0)

    def test_lmer_weights_custom(self) -> None:
        weights = np.random.rand(len(SLEEPSTUDY)) + 0.5
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, weights=weights)
        w = result.weights()

        assert len(w) == len(SLEEPSTUDY)
        assert np.allclose(w, weights)

    def test_lmer_offset_default(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        o = result.offset()

        assert len(o) == len(SLEEPSTUDY)
        assert np.allclose(o, 0.0)

    def test_lmer_offset_custom(self) -> None:
        offset = np.random.randn(len(SLEEPSTUDY))
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, offset=offset)
        o = result.offset()

        assert len(o) == len(SLEEPSTUDY)
        assert np.allclose(o, offset)

    def test_glmer_weights(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]
        weights = data["size"].values

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial(), weights=weights)
        w = result.weights()

        assert len(w) == len(data)
        assert np.allclose(w, weights)

    def test_glmer_offset(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]
        offset = np.log(data["size"].values)

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial(), offset=offset)
        o = result.offset()

        assert len(o) == len(data)
        assert np.allclose(o, offset)

    def test_glmer_get_family(self) -> None:
        from mixedlm.families.base import LogitLink

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        fam = result.get_family()

        assert isinstance(fam, families.Binomial)
        assert isinstance(fam.link, LogitLink)

    def test_glmer_get_family_poisson(self) -> None:
        from mixedlm.families.base import LogLink

        np.random.seed(42)
        n = 100
        group = np.repeat(np.arange(10), 10).astype(str)
        x = np.random.randn(n)
        y = np.random.poisson(np.exp(0.5 + 0.3 * x), n).astype(float)

        data = pd.DataFrame({"y": y, "x": x, "group": group})

        result = glmer("y ~ x + (1 | group)", data, family=families.Poisson())
        fam = result.get_family()

        assert isinstance(fam, families.Poisson)
        assert isinstance(fam.link, LogLink)

    def test_weights_returns_copy(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        w1 = result.weights()
        w2 = result.weights()

        w1[0] = 999
        assert w2[0] != 999

    def test_offset_returns_copy(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        o1 = result.offset()
        o2 = result.offset()

        o1[0] = 999
        assert o2[0] != 999


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestQQmath:
    def test_qqmath_basic(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.qqmath()

        assert fig is not None
        plt.close("all")

    def test_qqmath_specific_term(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        fig = result.qqmath(term="(Intercept)")

        assert fig is not None
        plt.close("all")

    def test_qqmath_multiple_terms(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        fig = result.qqmath()

        assert fig is not None
        plt.close("all")

    def test_qqmath_glmer(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        fig = result.qqmath()

        assert fig is not None
        plt.close("all")

    def test_qqmath_custom_figsize(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.qqmath(figsize=(8, 6))

        assert fig is not None
        plt.close("all")

    def test_qqmath_invalid_group(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="not found"):
            result.qqmath(group="InvalidGroup")

    def test_qqmath_invalid_term(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="not found"):
            result.qqmath(term="InvalidTerm")


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotDiagnostics:
    def test_plot_basic_lmer(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.plot()

        assert fig is not None
        plt.close("all")

    def test_plot_basic_glmer(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer(
            "y ~ period + (1 | herd)", data, family=families.Binomial(), weights=data["size"].values
        )
        fig = result.plot()

        assert fig is not None
        plt.close("all")

    def test_plot_subset_which(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.plot(which=[1, 2])

        assert fig is not None
        plt.close("all")

    def test_plot_single_which(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.plot(which=[1])

        assert fig is not None
        plt.close("all")

    def test_plot_custom_figsize(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.plot(figsize=(10, 8))

        assert fig is not None
        plt.close("all")

    def test_plot_all_panels(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        fig = result.plot(which=[1, 2, 3, 4])

        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close("all")

    def test_plot_no_random_effects_skips_panel4(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.diagnostics.plots import plot_diagnostics

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        fig = plot_diagnostics(result, which=[1, 2, 3])
        axes = [ax for ax in fig.get_axes() if ax.get_visible()]
        assert len(axes) == 3
        plt.close("all")

    def test_plot_ranef_function(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.diagnostics import plot_ranef

        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        ax = plot_ranef(result, term="(Intercept)")

        assert ax is not None
        plt.close("all")

    def test_plot_individual_functions(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mixedlm.diagnostics import (
            plot_qq,
            plot_resid_fitted,
            plot_resid_group,
            plot_scale_location,
        )

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        ax1 = plot_resid_fitted(result)
        assert ax1 is not None

        ax2 = plot_qq(result)
        assert ax2 is not None

        ax3 = plot_scale_location(result)
        assert ax3 is not None

        ax4 = plot_resid_group(result)
        assert ax4 is not None

        plt.close("all")


class TestModularInterface:
    def test_lFormula_basic(self) -> None:
        from mixedlm import lFormula

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert parsed.n_obs == 180
        assert parsed.n_fixed == 2
        assert parsed.n_random == 18
        assert parsed.n_theta == 1
        assert parsed.X.shape == (180, 2)
        assert parsed.y.shape == (180,)
        assert parsed.REML is True

    def test_lFormula_with_REML_false(self) -> None:
        from mixedlm import lFormula

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)

        assert parsed.REML is False

    def test_mkLmerDevfun_basic(self) -> None:
        from mixedlm import lFormula, mkLmerDevfun

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)

        start = devfun.get_start()
        assert len(start) == 1
        assert start[0] > 0.0
        assert start[0] < 10.0

        dev = devfun(start)
        assert isinstance(dev, float)
        assert dev > 0

    def test_mkLmerDevfun_bounds(self) -> None:
        from mixedlm import lFormula, mkLmerDevfun

        parsed = lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)

        bounds = devfun.get_bounds()
        assert len(bounds) == 3
        assert bounds[0] == (0.0, None)
        assert bounds[1] == (None, None)
        assert bounds[2] == (0.0, None)

    def test_optimizeLmer_basic(self) -> None:
        from mixedlm import lFormula, mkLmerDevfun, optimizeLmer

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        opt = optimizeLmer(devfun)

        assert opt.converged
        assert len(opt.theta) == 1
        assert opt.deviance > 0

    def test_mkLmerMod_basic(self) -> None:
        from mixedlm import lFormula, mkLmerDevfun, mkLmerMod, optimizeLmer

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        opt = optimizeLmer(devfun)
        result = mkLmerMod(devfun, opt)

        assert result.converged
        assert len(result.fixef()) == 2
        assert result.sigma > 0

        ranefs = result.ranef()
        assert "Subject" in ranefs

    def test_modular_matches_lmer(self) -> None:
        from mixedlm import lFormula, mkLmerDevfun, mkLmerMod, optimizeLmer

        direct_result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)
        opt = optimizeLmer(devfun)
        modular_result = mkLmerMod(devfun, opt)

        direct_fixef = np.array(list(direct_result.fixef().values()))
        modular_fixef = np.array(list(modular_result.fixef().values()))
        assert np.allclose(direct_fixef, modular_fixef, rtol=1e-4)
        assert np.allclose(direct_result.theta, modular_result.theta, rtol=1e-4)
        assert np.allclose(direct_result.sigma, modular_result.sigma, rtol=1e-4)

    def test_glFormula_basic(self) -> None:
        from mixedlm import glFormula

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        parsed = glFormula("y ~ period + (1 | herd)", data, family=families.Binomial())

        assert parsed.n_obs == 56
        assert parsed.n_fixed == 4
        assert parsed.family is not None
        assert parsed.n_theta == 1

    def test_mkGlmerDevfun_basic(self) -> None:
        from mixedlm import glFormula, mkGlmerDevfun

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        parsed = glFormula("y ~ period + (1 | herd)", data, family=families.Binomial())
        devfun = mkGlmerDevfun(parsed)

        start = devfun.get_start()
        assert len(start) == 1

        dev = devfun(start)
        assert isinstance(dev, float)

    def test_optimizeGlmer_basic(self) -> None:
        from mixedlm import glFormula, mkGlmerDevfun, optimizeGlmer

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        parsed = glFormula("y ~ period + (1 | herd)", data, family=families.Binomial())
        devfun = mkGlmerDevfun(parsed)
        opt = optimizeGlmer(devfun)

        assert opt.converged
        assert len(opt.theta) == 1

    def test_mkGlmerMod_basic(self) -> None:
        from mixedlm import glFormula, mkGlmerDevfun, mkGlmerMod, optimizeGlmer

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        parsed = glFormula("y ~ period + (1 | herd)", data, family=families.Binomial())
        devfun = mkGlmerDevfun(parsed)
        opt = optimizeGlmer(devfun)
        result = mkGlmerMod(devfun, opt)

        assert result.converged
        assert len(result.fixef()) == 4

        ranefs = result.ranef()
        assert "herd" in ranefs

    def test_glmer_modular_matches_glmer(self) -> None:
        from mixedlm import glFormula, mkGlmerDevfun, mkGlmerMod, optimizeGlmer

        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        direct_result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        parsed = glFormula("y ~ period + (1 | herd)", data, family=families.Binomial())
        devfun = mkGlmerDevfun(parsed)
        opt = optimizeGlmer(devfun)
        modular_result = mkGlmerMod(devfun, opt)

        direct_fixef = np.array(list(direct_result.fixef().values()))
        modular_fixef = np.array(list(modular_result.fixef().values()))
        assert np.allclose(direct_fixef, modular_fixef, rtol=1e-3)
        assert np.allclose(direct_result.theta, modular_result.theta, rtol=1e-3)

    def test_custom_optimizer_lmer(self) -> None:
        from mixedlm import OptimizeResult, lFormula, mkLmerDevfun, mkLmerMod

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        devfun = mkLmerDevfun(parsed)

        from scipy.optimize import minimize

        start = devfun.get_start()
        result = minimize(devfun, start, method="Nelder-Mead", options={"maxiter": 500})

        opt = OptimizeResult(
            theta=result.x,
            deviance=result.fun,
            converged=result.success,
            n_iter=result.nit,
            message="Custom optimizer",
        )

        model_result = mkLmerMod(devfun, opt)
        assert len(model_result.fixef()) == 2

    def test_lFormula_with_weights(self) -> None:
        from mixedlm import lFormula

        weights = np.ones(180)
        weights[:90] = 2.0

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, weights=weights)

        assert np.allclose(parsed.matrices.weights[:90], 2.0)
        assert np.allclose(parsed.matrices.weights[90:], 1.0)

    def test_lFormula_with_offset(self) -> None:
        from mixedlm import lFormula

        offset = np.zeros(180)
        offset[:90] = 10.0

        parsed = lFormula("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, offset=offset)

        assert np.allclose(parsed.matrices.offset[:90], 10.0)
        assert np.allclose(parsed.matrices.offset[90:], 0.0)

    def test_parsed_formula_properties(self) -> None:
        from mixedlm import lFormula

        parsed = lFormula("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        assert parsed.n_theta == 3
        assert parsed.Z.shape[1] == 36


class TestGetMEComponents:
    def test_getME_X(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        X = result.getME("X")

        assert X.shape == (180, 2)
        assert np.allclose(X[:, 0], 1.0)

    def test_getME_Z(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Z = result.getME("Z")

        assert Z.shape == (180, 18)

    def test_getME_y(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        y = result.getME("y")

        assert len(y) == 180
        assert np.allclose(y, SLEEPSTUDY["Reaction"].values)

    def test_getME_beta(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        beta = result.getME("beta")

        assert len(beta) == 2
        assert np.allclose(beta, list(result.fixef().values()))

    def test_getME_theta(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        theta = result.getME("theta")

        assert len(theta) == 1
        assert theta[0] >= 0

    def test_getME_Lambda(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        Lambda = result.getME("Lambda")

        assert Lambda.shape == (18, 18)

    def test_getME_u_b(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        u = result.getME("u")
        b = result.getME("b")

        assert len(u) == 18
        assert np.allclose(u, b)

    def test_getME_sigma(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        sigma = result.getME("sigma")

        assert sigma > 0
        assert sigma == result.sigma

    def test_getME_dimensions(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.getME("n") == 180
        assert result.getME("n_obs") == 180
        assert result.getME("p") == 2
        assert result.getME("n_fixed") == 2
        assert result.getME("q") == 18
        assert result.getME("n_random") == 18

    def test_getME_lower(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        lower = result.getME("lower")

        assert len(lower) == 1
        assert lower[0] == 0.0

    def test_getME_lower_correlated(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        lower = result.getME("lower")

        assert len(lower) == 3
        assert lower[0] == 0.0
        assert lower[1] == -np.inf
        assert lower[2] == 0.0

    def test_getME_weights_offset(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        weights = result.getME("weights")
        assert len(weights) == 180
        assert np.allclose(weights, 1.0)

        offset = result.getME("offset")
        assert len(offset) == 180
        assert np.allclose(offset, 0.0)

    def test_getME_REML_deviance(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        assert result.getME("REML") is True
        assert result.getME("deviance") > 0

    def test_getME_flist_cnms(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)

        flist = result.getME("flist")
        assert flist == ["Subject"]

        cnms = result.getME("cnms")
        assert "Subject" in cnms
        assert "(Intercept)" in cnms["Subject"]
        assert "Days" in cnms["Subject"]

    def test_getME_Gp(self) -> None:
        result = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)
        Gp = result.getME("Gp")

        assert len(Gp) == 2
        assert Gp[0] == 0
        assert Gp[1] == 36

    def test_getME_invalid(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="Unknown component name"):
            result.getME("invalid_name")

    def test_getME_glmer(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        X = result.getME("X")
        assert X.shape[1] == 4

        family = result.getME("family")
        assert isinstance(family, families.Binomial)


class TestUpdateSleepstudy:
    def test_update_REML(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        assert result1.REML is True

        result2 = result1.update(REML=False)
        assert result2.REML is False

    def test_update_same_formula(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        result2 = result1.update()

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_update_new_formula(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result1 = lmer("Reaction ~ Days + (1 | Subject)", data)
        result2 = result1.update("Reaction ~ Days + Days2 + (1 | Subject)", data=data)

        assert len(result1.fixef()) == 2
        assert len(result2.fixef()) == 3

    def test_update_new_data(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        subset = SLEEPSTUDY[SLEEPSTUDY["Days"] <= 5].copy()
        result2 = result1.update(data=subset)

        assert result2.getME("n") < result1.getME("n")

    def test_update_glmer_family(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result1 = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        data["count"] = np.round(data["incidence"]).astype(int)
        result2 = result1.update(
            formula="count ~ period + (1 | herd)", data=data, family=families.Poisson()
        )

        assert isinstance(result1.getME("family"), families.Binomial)
        assert isinstance(result2.getME("family"), families.Poisson)

    def test_update_with_weights(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        weights = np.ones(180)
        weights[:90] = 2.0
        result2 = result1.update(weights=weights)

        w1 = result1.getME("weights")
        w2 = result2.getME("weights")

        assert np.allclose(w1, 1.0)
        assert np.allclose(w2[:90], 2.0)

    def test_update_formula_dot_syntax_add(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result1 = lmer("Reaction ~ Days + (1 | Subject)", data)
        result2 = result1.update(". ~ . + Days2", data=data)

        assert len(result1.fixef()) == 2
        assert len(result2.fixef()) == 3


class TestRefitSleepstudy:
    def test_refit_same_response(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values.copy()
        result2 = result1.refit(newresp)

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_refit_new_response(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        np.random.seed(42)
        newresp = SLEEPSTUDY["Reaction"].values + np.random.normal(0, 50, 180)
        result2 = result1.refit(newresp)

        fixef1 = np.array(list(result1.fixef().values()))
        fixef2 = np.array(list(result2.fixef().values()))
        assert not np.allclose(fixef1, fixef2, rtol=1e-4)

    def test_refit_wrong_size(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="length"):
            result.refit(np.array([1.0, 2.0, 3.0]))

    def test_refit_preserves_structure(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values + 100
        result2 = result1.refit(newresp)

        assert result1.getME("n") == result2.getME("n")
        assert result1.getME("p") == result2.getME("p")
        assert result1.getME("q") == result2.getME("q")

    def test_refit_multiple_times(self) -> None:
        result1 = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        newresp = SLEEPSTUDY["Reaction"].values.copy()
        result2 = result1.refit(newresp)
        result3 = result2.refit(newresp)

        fixef2 = np.array(list(result2.fixef().values()))
        fixef3 = np.array(list(result3.fixef().values()))
        assert np.allclose(fixef2, fixef3, rtol=1e-4)

    def test_refit_glmer_basic(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result1 = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        np.random.seed(42)
        newresp = np.clip(data["y"].values + np.random.normal(0, 0.1, len(data)), 0.01, 0.99)
        result2 = result1.refit(newresp)

        assert result1.getME("n") == result2.getME("n")

    def test_refit_glmer_wrong_size(self) -> None:
        data = CBPP.copy()
        data["y"] = data["incidence"] / data["size"]

        result = glmer("y ~ period + (1 | herd)", data, family=families.Binomial())

        with pytest.raises(ValueError, match="length"):
            result.refit(np.array([0.1, 0.2, 0.3]))

    def test_refit_simulation_workflow(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        np.random.seed(123)
        fixef_samples = []
        for _ in range(3):
            newresp = SLEEPSTUDY["Reaction"].values + np.random.normal(0, 30, 180)
            refit_result = result.refit(newresp)
            fixef_samples.append(list(refit_result.fixef().values()))

        fixef_array = np.array(fixef_samples)
        assert fixef_array.shape == (3, 2)


class TestRefitML:
    def test_refitML_basic(self) -> None:
        result_reml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        assert result_reml.REML is True

        result_ml = result_reml.refitML()
        assert result_ml.REML is False

    def test_refitML_preserves_structure(self) -> None:
        result_reml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=True)
        result_ml = result_reml.refitML()

        assert result_reml.getME("n") == result_ml.getME("n")
        assert result_reml.getME("p") == result_ml.getME("p")
        assert result_reml.getME("q") == result_ml.getME("q")

    def test_refitML_different_estimates(self) -> None:
        result_reml = lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY, REML=True)
        result_ml = result_reml.refitML()

        theta_reml = result_reml.theta
        theta_ml = result_ml.theta

        assert result_reml.REML is True
        assert result_ml.REML is False
        assert theta_reml.shape == theta_ml.shape

    def test_refitML_already_ML_returns_self(self) -> None:
        result_ml = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
        result_ml2 = result_ml.refitML()

        assert result_ml is result_ml2

    def test_refitML_for_LRT(self) -> None:
        data = SLEEPSTUDY.copy()
        data["Days2"] = data["Days"] ** 2

        result_full = lmer("Reaction ~ Days + Days2 + (1 | Subject)", data, REML=True)
        result_reduced = lmer("Reaction ~ Days + (1 | Subject)", data, REML=True)

        ml_full = result_full.refitML()
        ml_reduced = result_reduced.refitML()

        ll_full = ml_full.logLik().value
        ll_reduced = ml_reduced.logLik().value

        assert ll_full > ll_reduced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
