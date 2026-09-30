from __future__ import annotations

from types import SimpleNamespace

import mixedlm.inference.ddf as ddf_module
import numpy as np
import pytest
from mixedlm import lmer, load_sleepstudy
from mixedlm.inference.anova import anova_type3
from mixedlm.inference.ddf import pvalues_with_ddf, satterthwaite_df
from mixedlm.inference.reporting import tidy
from mixedlm.models.control import LmerControl
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module", params=[(True, False), (True, True), (False, False), (False, True)])
def fitted_model(request):
    reml, weighted = request.param
    data = load_sleepstudy()
    weights = np.linspace(0.5, 2.0, len(data)) if weighted else None
    control = LmerControl(use_rust=False, em_init=False)
    model = lmer(
        "Reaction ~ Days + (Days | Subject)", data, weights=weights, REML=reml, control=control
    )
    return model, data, weights, control


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3])
def test_response_units_preserve_ddf_and_inference(fitted_model, scale):
    model, data, weights, control = fitted_model
    scaled_data = data.copy()
    scaled_data["Reaction"] *= scale
    scaled = lmer(
        "Reaction ~ Days + (Days | Subject)",
        scaled_data,
        weights=weights,
        REML=model.REML,
        control=control,
    )

    expected_df = satterthwaite_df(model)
    actual_df = satterthwaite_df(scaled)
    assert_allclose(actual_df.df, expected_df.df, rtol=1e-4, atol=1e-4)
    if model.REML and weights is None:
        assert_allclose(actual_df.df, [17.0, 17.0], atol=1e-3, rtol=0)

    expected_pvalues = pvalues_with_ddf(model)
    actual_pvalues = pvalues_with_ddf(scaled)
    for name in model.matrices.fixed_names:
        # Very small tails amplify the finite-difference curvature tolerance.
        assert_allclose(actual_pvalues[name][1:], expected_pvalues[name][1:], rtol=3e-4, atol=0)

    expected_table = tidy(model, conf_int=True)
    actual_table = tidy(scaled, conf_int=True)
    assert_allclose(
        actual_table[["df", "statistic", "p.value"]],
        expected_table[["df", "statistic", "p.value"]],
        rtol=3e-4,
        atol=0,
    )
    assert_allclose(
        actual_table[["estimate", "std.error", "conf.low", "conf.high"]] / scale,
        expected_table[["estimate", "std.error", "conf.low", "conf.high"]],
        rtol=1e-4,
        atol=1e-5,
    )
    expected_anova = anova_type3(model)
    actual_anova = anova_type3(scaled)
    assert_allclose(actual_anova.den_df, expected_anova.den_df, rtol=1e-4, atol=0)
    assert_allclose(actual_anova.p_value, expected_anova.p_value, rtol=1e-4, atol=0)


@pytest.mark.parametrize("scale", [1e-3, 1e3, 1e6])
def test_fixed_predictor_units_preserve_ddf_and_pvalues(scale):
    data = load_sleepstudy()
    control = LmerControl(use_rust=False, em_init=False)
    model = lmer("Reaction ~ Days + (1 | Subject)", data, control=control)
    data["Days"] *= scale
    scaled = lmer("Reaction ~ Days + (1 | Subject)", data, control=control)

    assert_allclose(satterthwaite_df(scaled).df, satterthwaite_df(model).df, rtol=1e-4, atol=0)
    expected = pvalues_with_ddf(model)
    actual = pvalues_with_ddf(scaled)
    for name in expected:
        assert_allclose(actual[name][1:], expected[name][1:], rtol=1e-4, atol=0)


@pytest.mark.parametrize("scale", [1e-200, 1e-10, 1.0, 1e200])
def test_variance_magnitude_does_not_change_satterthwaite_ratio(monkeypatch, scale):
    variances = scale * np.array([2.0, 3.0, 4.0])
    covariance = np.diag(variances)
    gradients = [
        np.diag(variances * [0.1, 0.3, 0.0]),
        np.diag(variances * [0.2, -0.4, 0.0]),
    ]
    parameter_covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    model = SimpleNamespace(
        vcov=lambda: covariance,
        matrices=SimpleNamespace(n_obs=103, n_fixed=3, fixed_names=["a", "b", "c"]),
    )
    monkeypatch.setattr(
        ddf_module, "_vcov_derivatives", lambda _: (gradients, parameter_covariance)
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = satterthwaite_df(model)

    # Relative uncertainty is 0.08, 0.22, and zero, plus 2 / 100
    # from the residual scale. These ratios do not depend on the units.
    assert_allclose(actual.df, [20.0, 25.0 / 3.0, 100.0], rtol=1e-14, atol=0)
    assert actual.param_names == ["a", "b", "c"]


@pytest.mark.parametrize("n_fixed", [0, 3])
def test_no_covariance_parameters_retains_residual_df(monkeypatch, n_fixed):
    model = SimpleNamespace(
        vcov=lambda: np.eye(n_fixed),
        matrices=SimpleNamespace(
            n_obs=100, n_fixed=n_fixed, fixed_names=list(map(str, range(n_fixed)))
        ),
    )
    monkeypatch.setattr(ddf_module, "_vcov_derivatives", lambda _: ([], np.empty((0, 0))))

    actual = satterthwaite_df(model)

    assert_array_equal(actual.df, np.full(n_fixed, 100 - n_fixed))


def test_zero_variance_retains_residual_df(monkeypatch):
    model = SimpleNamespace(
        vcov=lambda: np.zeros((1, 1)),
        matrices=SimpleNamespace(n_obs=31, n_fixed=1, fixed_names=["a"]),
    )
    monkeypatch.setattr(
        ddf_module, "_vcov_derivatives", lambda _: ([np.zeros((1, 1))], np.ones((1, 1)))
    )

    with np.errstate(divide="raise", invalid="raise"):
        actual = satterthwaite_df(model)

    assert_array_equal(actual.df, [30.0])
