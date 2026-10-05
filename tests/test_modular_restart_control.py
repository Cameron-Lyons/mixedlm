"""Modular LMM fits inherit restart controls and retain explicit overrides."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm import lFormula, lmerControl, mkLmerDevfun, optimizeLmer
from mixedlm.models.modular import LmerDevfun
from numpy.testing import assert_allclose, assert_array_equal

from tests._lmm_oracles import linear_data


def deviance_function(enabled, native=True, analytic=False, reml=False):
    control = lmerControl(restart_edge=enabled, use_rust=native, use_analytic_gradient=analytic)
    parsed = lFormula("y ~ 1 + (1 | g)", linear_data(), REML=reml)
    return mkLmerDevfun(parsed, control=control)


def assert_variance(result, enabled, reml=False):
    assert result.converged, result.message
    if enabled:
        residual_variance = 4 / 3
        between_variance = 0.7**2 * (6 / 5 if reml else 1)
        expected = np.sqrt((between_variance - residual_variance / 4) / residual_variance)
        assert_allclose(result.theta, [expected], atol=2e-4)
    else:
        assert_array_equal(result.theta, [0.0])
        assert result.n_iter == 0


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize(
    "enabled,override,expected",
    [
        (False, "omitted", False),
        (True, "omitted", True),
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, False, False),
    ],
)
def test_restart_choice_is_inherited_or_explicitly_overridden(
    native, analytic, reml, enabled, override, expected
):
    devfun = deviance_function(enabled, native, analytic, reml)
    kwargs = {} if override == "omitted" else {"restart_edge": override}
    result = optimizeLmer(devfun, start=np.zeros(1), **kwargs)
    assert_variance(result, expected, reml)
    assert devfun.control.restart_edge is enabled


@pytest.mark.parametrize("kwargs", [{}, {"restart_edge": None}])
def test_missing_control_preserves_enabled_restarts(kwargs):
    devfun = replace(deviance_function(False), control=None)
    result = optimizeLmer(devfun, start=np.zeros(1), **kwargs)
    assert_variance(result, True)


def test_override_does_not_change_later_fits_or_the_stored_control():
    devfun = deviance_function(False)
    for kwargs, enabled in [({}, False), ({"restart_edge": True}, True), ({}, False)]:
        result = optimizeLmer(devfun, start=np.zeros(1), **kwargs)
        assert_variance(result, enabled)
        assert devfun.control.restart_edge is False


@pytest.mark.parametrize("enabled", [np.bool_(False), np.bool_(True)])
def test_numpy_boolean_override_is_preserved(enabled):
    result = optimizeLmer(deviance_function(not enabled), start=np.zeros(1), restart_edge=enabled)
    assert_variance(result, enabled)


@pytest.mark.parametrize("invalid", [0, 1, "false", [], np.nan])
def test_invalid_overrides_are_rejected_before_evaluating_the_objective(invalid):
    devfun = deviance_function(False)
    with (
        patch.object(devfun.optimizer, "objective", side_effect=AssertionError("unexpected call")),
        pytest.raises(ValueError, match="restart_edge must be a boolean"),
    ):
        optimizeLmer(devfun, start=np.zeros(1), restart_edge=invalid)


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_deviance_callable_inherits_the_restart_choice(enabled):
    class ShiftedDeviance(LmerDevfun):
        def __call__(self, theta):
            return super().__call__(theta) + 1e4

    original = deviance_function(enabled, analytic=True)
    devfun = ShiftedDeviance(original.parsed, original.optimizer, original.control)
    result = optimizeLmer(devfun, start=np.zeros(1))
    assert_variance(result, enabled)
    assert_allclose(result.deviance, devfun(result.theta), rtol=0, atol=1e-10)
