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
