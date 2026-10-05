"""lmerControl/glmerControl settings and the optimizers they select."""

from importlib.util import find_spec

import numpy as np
import pytest
from mixedlm import (
    families,
    glmer,
    glmerControl,
    lmer,
    lmerControl,
)
from mixedlm.models.control import GlmerControl, LmerControl
from numpy.testing import assert_allclose

from tests._datasets import CBPP, CBPP_FORMULA, SLEEPSTUDY

HAS_NLOPT = find_spec("nlopt") is not None
requires_nlopt = pytest.mark.skipif(not HAS_NLOPT, reason="nlopt not installed")


def fit_sleepstudy(formula="Reaction ~ Days + (1 | Subject)", **control):
    return lmer(formula, SLEEPSTUDY, control=lmerControl(**control))


def fit_cbpp(**control):
    return glmer(CBPP_FORMULA, CBPP, family=families.Binomial(), control=glmerControl(**control))


def assert_same_optimum(result, reference, *, atol=1e-4) -> None:
    assert result.converged
    assert result.deviance == pytest.approx(reference.deviance, abs=1e-6)
    assert_allclose(result.beta, reference.beta, rtol=0, atol=atol)
    assert_allclose(result.theta, reference.theta, rtol=0, atol=atol)


class TestControl:
    def test_lmer_control_default(self) -> None:
        ctrl = LmerControl()
        assert ctrl.optimizer == "auto"
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

    def test_lmer_control_method_specific_options(self) -> None:
        ctrl = LmerControl(maxiter=500, ftol=1e-7, gtol=1e-4, xtol=1e-6)

        tnc_options = ctrl.get_scipy_options(optimizer="TNC", maxiter=250)
        assert tnc_options == {
            "maxfun": 250,
            "ftol": 1e-7,
            "gtol": 1e-4,
            "xtol": 1e-6,
        }

        powell_options = ctrl.get_scipy_options(optimizer="Powell")
        assert powell_options == {"maxiter": 500, "ftol": 1e-7, "xtol": 1e-6}

    def test_lmer_fit_forwards_control_tolerances(self, monkeypatch) -> None:
        from mixedlm.estimation import reml as reml_module

        original_run_optimizer = reml_module.run_optimizer
        captured_options = {}

        def capture_options(*args, **kwargs):
            captured_options.update(kwargs["options"])
            return original_run_optimizer(*args, **kwargs)

        monkeypatch.setattr(reml_module, "run_optimizer", capture_options)
        ctrl = LmerControl(
            optimizer="L-BFGS-B",
            maxiter=321,
            ftol=2e-7,
            gtol=3e-6,
            check_singular=False,
        )

        result = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, control=ctrl)

        assert result.converged
        assert captured_options["maxiter"] == 321
        assert captured_options["ftol"] == 2e-7
        assert captured_options["gtol"] == 3e-6

    def test_glmer_fit_forwards_control_tolerances(self, monkeypatch) -> None:
        from mixedlm.estimation import laplace as laplace_module

        original_run_optimizer = laplace_module.run_optimizer
        captured_options = {}

        def capture_options(*args, **kwargs):
            captured_options.update(kwargs["options"])
            return original_run_optimizer(*args, **kwargs)

        monkeypatch.setattr(laplace_module, "run_optimizer", capture_options)

        result = fit_cbpp(
            optimizer="L-BFGS-B", maxiter=321, ftol=2e-7, gtol=3e-6, check_singular=False
        )

        assert result.converged
        assert captured_options["maxiter"] == 321
        assert captured_options["ftol"] == 2e-7
        assert captured_options["gtol"] == 3e-6

    def test_lmer_with_control(self, sleepstudy_lmm) -> None:
        result = fit_sleepstudy(maxiter=100, check_singular=False)

        assert_same_optimum(result, sleepstudy_lmm, atol=1e-5)
        assert_allclose(list(result.fixef().values()), sleepstudy_lmm.beta, rtol=1e-5)

    def test_glmer_control_default(self) -> None:
        ctrl = GlmerControl()
        assert ctrl.optimizer == "COBYQA"
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

    def test_glmer_with_control(self, cbpp_glmm) -> None:
        assert_same_optimum(fit_cbpp(maxiter=100, check_singular=False), cbpp_glmm)

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

    @pytest.mark.parametrize("optimizer", ["bobyqa", "nloptwrap_BOBYQA", "nloptwrap_SBPLX"])
    def test_lme4_optimizer_names_are_accepted(self, optimizer) -> None:
        assert LmerControl(optimizer=optimizer).optimizer == optimizer
        assert GlmerControl(optimizer=optimizer).optimizer == optimizer


class TestScipyOptimizers:
    @pytest.mark.parametrize(
        ("optimizer", "opt_ctrl"),
        [
            ("Nelder-Mead", {}),
            ("COBYQA", {}),
            ("COBYQA", {"initial_tr_radius": 0.5, "final_tr_radius": 1e-4}),
        ],
    )
    def test_reach_the_default_optimum(self, sleepstudy_lmm, optimizer, opt_ctrl) -> None:
        result = fit_sleepstudy(optimizer=optimizer, maxiter=2000, optCtrl=opt_ctrl)

        assert_same_optimum(result, sleepstudy_lmm)

    def test_cobyqa_random_slopes(self, sleepstudy_slopes_lmm) -> None:
        result = fit_sleepstudy("Reaction ~ Days + (Days | Subject)", optimizer="COBYQA")

        assert_same_optimum(result, sleepstudy_slopes_lmm, atol=1e-3)

    def test_cobyqa_and_lbfgsb_agree(self) -> None:
        result_cobyqa = fit_sleepstudy(optimizer="COBYQA")
        result_lbfgsb = fit_sleepstudy(optimizer="L-BFGS-B")

        assert result_cobyqa.deviance == pytest.approx(result_lbfgsb.deviance, abs=1e-6)
        np.testing.assert_allclose(result_cobyqa.beta, result_lbfgsb.beta, rtol=0, atol=1e-4)
        np.testing.assert_allclose(result_cobyqa.theta, result_lbfgsb.theta, rtol=0, atol=1e-4)

    def test_glmer_cobyqa(self, cbpp_glmm) -> None:
        assert_same_optimum(fit_cbpp(optimizer="COBYQA"), cbpp_glmm)

    def test_cobyqa_is_available_to_allfit(self) -> None:
        from mixedlm.estimation.optimizers import has_cobyqa
        from mixedlm.inference.allfit import _default_optimizers

        assert has_cobyqa() is True
        assert "COBYQA" in _default_optimizers()


class TestNloptOptimizer:
    def test_has_nlopt_reports_the_installed_package(self) -> None:
        from mixedlm.estimation.optimizers import has_nlopt

        assert has_nlopt() is HAS_NLOPT

    @requires_nlopt
    @pytest.mark.parametrize(
        "optimizer", ["nloptwrap_BOBYQA", "nloptwrap_NELDERMEAD", "nloptwrap_SBPLX"]
    )
    def test_reach_the_default_optimum(self, sleepstudy_lmm, optimizer) -> None:
        assert_same_optimum(fit_sleepstudy(optimizer=optimizer), sleepstudy_lmm)

    @requires_nlopt
    def test_glmer_bobyqa(self, cbpp_glmm) -> None:
        result = fit_cbpp(optimizer="nloptwrap_BOBYQA")

        assert result.converged
        assert result.deviance == pytest.approx(cbpp_glmm.deviance, abs=1e-5)

    @requires_nlopt
    def test_bobyqa_and_lbfgsb_agree_on_random_slopes(self, sleepstudy_slopes_lmm) -> None:
        result = fit_sleepstudy("Reaction ~ Days + (Days | Subject)", optimizer="nloptwrap_BOBYQA")

        assert result.converged
        assert result.deviance == pytest.approx(sleepstudy_slopes_lmm.deviance, abs=1e-4)
        np.testing.assert_allclose(result.beta, sleepstudy_slopes_lmm.beta, rtol=0, atol=1e-2)

    @requires_nlopt
    def test_allfit_includes_nlopt(self) -> None:
        from mixedlm.inference.allfit import _default_optimizers

        assert "nloptwrap_BOBYQA" in _default_optimizers()
