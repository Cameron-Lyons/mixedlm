import numpy as np
import pandas as pd
import pytest

from mixedlm import lmer, parse_formula
from mixedlm.matrices import build_model_matrices


SLEEPSTUDY = pd.DataFrame(
    {
        "Reaction": [
            249.56, 258.70, 250.80, 321.44, 356.85, 414.69, 382.20, 290.15, 430.59, 466.35,
            222.73, 205.27, 202.98, 204.71, 207.72, 215.94, 213.63, 217.73, 224.30, 237.31,
            199.05, 194.33, 234.32, 232.84, 229.31, 220.46, 235.42, 255.75, 261.01, 247.52,
            321.42, 300.01, 283.86, 285.13, 285.80, 297.59, 280.24, 318.26, 305.35, 354.04,
            287.60, 285.00, 301.80, 320.12, 316.28, 293.32, 290.08, 334.82, 293.70, 371.58,
            234.92, 242.81, 272.95, 309.17, 317.96, 310.00, 454.16, 346.82, 330.30, 253.92,
            283.84, 289.00, 276.77, 299.80, 297.50, 338.10, 340.80, 305.55, 354.10, 357.70,
            265.35, 276.37, 243.43, 254.72, 279.59, 284.26, 305.60, 331.84, 335.45, 377.32,
            241.58, 273.94, 254.44, 270.04, 251.47, 254.38, 245.37, 235.70, 235.98, 249.55,
            312.17, 313.59, 291.64, 346.12, 365.16, 391.84, 404.29, 416.56, 455.86, 458.96,
            292.63, 308.52, 324.28, 320.90, 305.30, 350.56, 300.01, 327.33, 335.63, 392.03,
            290.14, 262.46, 253.66, 267.51, 296.00, 304.56, 350.78, 369.47, 364.88, 370.63,
            263.99, 289.65, 276.77, 299.09, 297.43, 310.73, 287.17, 329.61, 334.48, 343.22,
            237.45, 301.79, 311.90, 282.73, 285.05, 240.09, 275.19, 238.49, 266.54, 207.72,
            286.95, 288.54, 245.18, 276.11, 266.42, 250.13, 269.84, 281.05, 284.78, 306.72,
            271.98, 268.70, 257.72, 266.66, 310.04, 309.18, 327.21, 347.79, 341.82, 373.73,
            346.00, 344.00, 358.00, 399.00, 363.00, 400.00, 416.00, 376.00, 441.00, 466.00,
            269.41, 273.47, 297.60, 310.63, 287.17, 329.61, 334.48, 343.22, 369.14, 364.06,
        ],
        "Days": list(range(10)) * 18,
        "Subject": [str(i) for i in [308] * 10 + [309] * 10 + [310] * 10 + [330] * 10 +
                    [331] * 10 + [332] * 10 + [333] * 10 + [334] * 10 + [335] * 10 +
                    [337] * 10 + [349] * 10 + [350] * 10 + [351] * 10 + [352] * 10 +
                    [369] * 10 + [370] * 10 + [371] * 10 + [372] * 10],
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
