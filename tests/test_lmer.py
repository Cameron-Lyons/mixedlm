import numpy as np
import pandas as pd
import pytest
from mixedlm import anova, families, glmer, lmer, nlme, nlmer, parse_formula
from mixedlm.matrices import build_model_matrices

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

    def test_update_requires_data(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)

        with pytest.raises(ValueError, match="data must be provided"):
            result.update(". ~ . + 1")


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

        data = pd.DataFrame(
            {"y": y, "x1": x1, "x2": x2, "group": [str(g) for g in group]}
        )
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
        from mixedlm.models.lmer import LmerResult
        from mixedlm.matrices.design import ModelMatrices, RandomEffectStructure
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
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY)

        assert len(allfit_result.fits) == 6

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
        for opt_name, fixefs in fixef_table.items():
            assert "(Intercept)" in fixefs
            assert "Days" in fixefs

    def test_allfit_theta_table(self):
        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
        allfit_result = result.allFit(SLEEPSTUDY, optimizers=["L-BFGS-B", "Nelder-Mead"])

        theta_table = allfit_result.theta_table()
        assert isinstance(theta_table, dict)
        assert len(theta_table) > 0
        for opt_name, thetas in theta_table.items():
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
