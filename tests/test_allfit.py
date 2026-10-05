"""allFit refits one model per optimizer through a single engine."""

import mixedlm as mlm
import pytest
from mixedlm.estimation.optimizers import ALL_OPTIMIZERS, COMPATIBILITY_OPTIMIZERS
from mixedlm.inference.allfit import _default_optimizers, allfit_lmer
from mixedlm.models.control import LmerControl, lmerControl

FORMULA = "Reaction ~ Days + (Days | Subject)"


@pytest.fixture(scope="module")
def sleepstudy():
    return mlm.load_sleepstudy()


def test_serial_and_worker_processes_report_identical_outcomes(sleepstudy):
    # Scalar random effects keep worker solves off the native thread pool, so a
    # fork regression fails test_process_parallel_safety instead of hanging here.
    formula = "Reaction ~ Days + (1 | Subject)"
    fitted = mlm.lmer(formula, sleepstudy)
    optimizers = ["Nelder-Mead", "bogus", "COBYQA"]
    serial = allfit_lmer(fitted, sleepstudy, optimizers, n_jobs=1)
    parallel = allfit_lmer(fitted, sleepstudy, optimizers, n_jobs=2)

    for result in (serial, parallel):
        assert list(result.fits) == list(result.warnings) == optimizers
        assert result.fits["bogus"] is None
        assert result.warnings["bogus"] == []
        assert list(result.errors) == ["bogus"]
        assert "Unknown optimizer 'bogus'" in result.errors["bogus"]
    assert parallel.errors == serial.errors
    assert parallel.warnings == serial.warnings
    for name in ("Nelder-Mead", "COBYQA"):
        expected = mlm.lmer(formula, sleepstudy, control=lmerControl(optimizer=name))
        for result in (serial, parallel):
            assert result.fits[name].deviance == expected.deviance
            assert list(result.fits[name].theta) == list(expected.theta)


def test_formula_allfit_keeps_the_callers_control(sleepstudy):
    control = lmerControl(maxiter=3, check_conv=False)
    result = mlm.allFit(FORMULA, sleepstudy, ["L-BFGS-B", "bogus"], control=control)

    expected = mlm.lmer(
        FORMULA, sleepstudy, control=lmerControl(optimizer="L-BFGS-B", maxiter=3, check_conv=False)
    )
    fit = result.fits["L-BFGS-B"]
    assert fit.n_iter == expected.n_iter <= 3
    assert fit.deviance == expected.deviance
    assert not fit.converged
    assert result.warnings == {"L-BFGS-B": ["Did not converge"], "bogus": []}
    assert list(result.errors) == ["bogus"]
    assert control.optimizer == "COBYQA"


def test_default_optimizers_are_the_valid_names_without_aliases():
    defaults = _default_optimizers()
    assert len(defaults) == len(set(defaults))
    assert set(defaults) <= ALL_OPTIMIZERS - COMPATIBILITY_OPTIMIZERS
    assert {"COBYQA", "L-BFGS-B", "Nelder-Mead", "BFGS"} <= set(defaults)
    for name in defaults:
        assert LmerControl(optimizer=name).optimizer == name
