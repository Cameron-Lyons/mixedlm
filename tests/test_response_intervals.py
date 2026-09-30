from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, emmeans, families, ggpredict, glmer, parse_formula
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal

shared = import_module("mixedlm.models.shared_utils")


def _model(family, *, crossing=False, kind="glmm"):
    x = np.tile([-0.4, 0.0, 0.4], 4)
    frame = pd.DataFrame({"x": x, "g": np.repeat(np.arange(4), 3), "y": np.ones(12)})
    formula = parse_formula("y ~ x + (1 | g)")
    matrices = build_model_matrices(formula, frame)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=np.array([0.05, 0.0]) if crossing else np.array([1.5, 0.2]),
        u=np.linspace(-0.05, 0.05, matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        GlmerResult(**common, family=family, nAGQ=1)
        if kind == "glmm"
        else LmerResult(**common, sigma=0.5, REML=True)
    )
    covariance = np.array([[0.16 if crossing else 0.04, 0.002], [0.002, 0.01]])
    model.vcov = lambda: covariance.copy()
    return model


def _interval(model, api, scale):
    if api == "prediction":
        result = model.predict(
            pd.DataFrame({"x": [0.0]}), re_form="NA", type=scale, interval="confidence"
        )
        return result.fit, result.se_fit, result.lower, result.upper
    if api == "marginal":
        result = emmeans(model, [], at={"x": 0.0}, type=scale).result
        return result.emmean, result.se, result.lower, result.upper
    result = (
        ggpredict(model, "x", at={"x": [0.0]}, type=scale)
        if api == "effect"
        else allEffects(model, at={"x": 0.0}, type=scale)["x"]
    )
    return tuple(
        result[name].to_numpy() for name in ["predicted", "std.error", "conf.low", "conf.high"]
    )


FAMILIES = [
    families.Gamma(link="inverse"),
    families.InverseGaussian(link="1/mu^2"),
    families.Poisson(link="sqrt"),
    families.Poisson(),
    families.Binomial(),
    families.Gaussian(),
]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.link.name)
@pytest.mark.parametrize("api", ["prediction", "marginal", "effect", "all_effects"])
def test_response_intervals_contain_predictions_and_follow_inverse_link(family, api):
    crossing = family.link.name == "sqrt"
    model = _model(family, crossing=crossing)
    eta, se_eta, eta_lower, eta_upper = _interval(model, api, "link")

    predicted, se, lower, upper = _interval(model, api, "response")

    transformed = family.link.inverse(np.stack([eta_lower, eta_upper]))
    expected_lower = np.zeros_like(eta) if crossing else transformed.min(axis=0)
    assert_allclose(lower, expected_lower)
    assert_allclose(upper, transformed.max(axis=0))
    assert_allclose(predicted, family.link.inverse(eta))
    assert_allclose(se, se_eta / np.abs(family.link.deriv(predicted)))
    assert np.all(lower <= predicted)
    assert np.all(predicted <= upper)


@pytest.mark.parametrize(
    "link",
    [
        families.IdentityLink(),
        families.LogLink(),
        families.LogitLink(),
        families.ProbitLink(),
        families.CloglogLink(),
        families.CauchitLink(),
        families.InverseLink(),
        families.InverseSquaredLink(),
        families.SqrtLink(),
    ],
    ids=lambda link: link.name,
)
def test_inverse_interval_contains_interior_values_without_changing_inputs(link):
    lower = np.array([-2.0, -0.1, 0.0, 0.5, 2.0])
    upper = np.array([-0.5, 0.3, 0.0, 1.0, 3.0])
    original = lower.copy(), upper.copy()
    lower.setflags(write=False)
    upper.setflags(write=False)

    response_lower, response_upper = link.inverse_interval(lower, upper)

    values = link.inverse(lower + np.linspace(0, 1, 101)[:, None] * (upper - lower))
    assert np.all(values >= response_lower - 1e-12)
    assert np.all(values <= response_upper + 1e-12)
    assert_array_equal(lower, original[0])
    assert_array_equal(upper, original[1])
    if link.name == "sqrt":
        assert_array_equal(response_lower, [0.25, 0.0, 0.0, 0.25, 4.0])
        assert_array_equal(response_upper, [4.0, 0.09, 0.0, 1.0, 9.0])


@pytest.mark.parametrize("api", ["prediction", "marginal", "effect", "all_effects"])
def test_custom_link_interval_override_is_honored(api):
    class CustomSqrt(families.SqrtLink):
        def inverse_interval(self, lower, upper):
            return np.zeros_like(lower), np.full_like(upper, 42.0)

    model = _model(families.Poisson(link=CustomSqrt()), crossing=True)

    _, _, lower, upper = _interval(model, api, "response")

    assert_array_equal(lower, [0.0])
    assert_array_equal(upper, [42.0])


@pytest.mark.parametrize("source", ["in_sample", "pandas", "polars"])
@pytest.mark.parametrize("re_form", [None, "NA", "~0"])
@pytest.mark.parametrize("interval", ["none", "confidence"])
def test_prediction_modes_preserve_point_values_and_delta_standard_errors(
    source, re_form, interval
):
    model = _model(families.Gamma(link="inverse"))
    data = None if source == "in_sample" else model.matrices.frame.iloc[[0, 4, 8]].copy()
    if source == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(data.to_dict("list"))
    linked = model.predict(data, type="link", re_form=re_form, se_fit=True, interval=interval)

    response = model.predict(data, re_form=re_form, se_fit=True, interval=interval)

    assert_allclose(response.fit, model.family.link.inverse(linked.fit))
    assert_allclose(response.se_fit, linked.se_fit / np.abs(model.family.link.deriv(response.fit)))
    assert_allclose(model.predict(data, re_form=re_form), response.fit)
    assert response.interval == interval
    assert response.level == 0.95
    if interval == "confidence":
        assert_allclose(response.lower, model.family.link.inverse(linked.upper))
        assert_allclose(response.upper, model.family.link.inverse(linked.lower))
    else:
        assert response.lower is response.upper is None


@pytest.mark.parametrize(
    "option,value,match",
    [
        ("type", "typo", "type must"),
        ("type", None, "type must"),
        ("type", np.array(["link"]), "type must"),
        ("type", [], "type must"),
        ("interval", "typo", "Unknown interval"),
        ("interval", None, "Unknown interval"),
        ("interval", np.array(["confidence"]), "Unknown interval"),
        ("interval", [], "Unknown interval"),
        ("interval", "prediction", "not available for GLMMs"),
    ],
)
def test_invalid_options_fail_before_model_or_covariance_work(option, value, match):
    with pytest.raises(ValueError, match=match):
        GlmerResult.predict(object(), **{option: value})


@pytest.mark.parametrize("scale", ["link", "response"])
def test_point_predictions_do_not_request_covariance(scale):
    model = _model(families.Poisson())

    def unexpected():
        raise AssertionError("Point predictions do not need covariance")

    model.vcov = unexpected
    assert np.isfinite(model.predict(type=scale, re_form="NA")).all()


@pytest.mark.parametrize("scale", ["link", "response"])
def test_empty_prediction_intervals_keep_empty_arrays(scale):
    model = _model(families.Gamma(link="inverse"))

    result = model.predict(pd.DataFrame({"x": []}), re_form="NA", type=scale, interval="confidence")

    for value in [result.fit, result.se_fit, result.lower, result.upper]:
        assert value.shape == (0,)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("limit", [1, 7, 1_000_000])
def test_prediction_covariance_projection_uses_bounded_buffers(monkeypatch, kind, limit):
    model = _model(families.Poisson(), kind=kind)
    processed = []

    class BoundedDesign(np.ndarray):
        def __matmul__(self, other):
            if other.ndim == 2:
                assert self.size <= max(limit, self.shape[1])
                processed.append(len(self))
            return np.asarray(self) @ other

    design = model.matrices.X.copy()
    model.matrices.X = design.view(BoundedDesign)
    monkeypatch.setattr(shared, "_MAX_QUADRATIC_FORM_ELEMENTS", limit)
    kwargs = {"type": "link"} if kind == "glmm" else {}

    result = model.predict(re_form="NA", se_fit=True, **kwargs)

    expected_variance = np.diag(design @ model.vcov() @ design.T)
    assert_allclose(result.se_fit**2, expected_variance, rtol=1e-12)
    assert sum(processed) == len(design)
    assert_array_equal(model.matrices.X, design)


@pytest.mark.parametrize("shape", [(0, 3), (4, 0)])
def test_dense_variance_handles_empty_model_dimensions(shape):
    design = np.empty(shape)
    result = shared.dense_quadratic_form_diagonal(design, np.eye(shape[1]))
    assert_array_equal(result, np.zeros(shape[0]))


def test_fitted_gamma_inverse_predictions_have_ordered_response_intervals():
    rng = np.random.default_rng(202609)
    x = np.tile(np.linspace(-0.5, 0.5, 12), 10)
    eta = 1.2 + 0.2 * x
    frame = pd.DataFrame(
        {"x": x, "g": np.repeat(np.arange(10), 12), "y": rng.gamma(30.0, 1 / (30 * eta))}
    )
    model = glmer("y ~ x + (1 | g)", frame, family=families.Gamma(link="inverse"))

    result = model.predict(re_form="NA", interval="confidence")

    assert np.all(result.lower <= result.fit)
    assert np.all(result.fit <= result.upper)
