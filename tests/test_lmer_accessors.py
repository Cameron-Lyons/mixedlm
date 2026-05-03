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
