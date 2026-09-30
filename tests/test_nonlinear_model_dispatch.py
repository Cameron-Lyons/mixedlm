from types import MethodType
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import nlmer
from mixedlm.estimation import nlmm
from mixedlm.nlme.models import CustomModel, SSasymp, SSbiexp, SSfpl, SSgompertz, SSlogis, SSmicmen

BUILTINS = [
    (SSasymp, "ssasymp", [10.0, 2.0, -1.0]),
    (SSlogis, "sslogis", [10.0, 2.0, 1.0]),
    (SSmicmen, "ssmicmen", [10.0, 2.0]),
    (SSfpl, "ssfpl", [1.0, 10.0, 2.0, 1.0]),
    (SSgompertz, "ssgompertz", [10.0, 2.0, 0.5]),
    (SSbiexp, "ssbiexp", [5.0, -1.0, 3.0, -2.0]),
]


def problem(model, phi):
    rng = np.random.default_rng(42)
    x = np.tile(np.linspace(0.1, 5, 10), 4)
    groups = np.repeat(np.arange(4), 10)
    y = (
        model.predict(np.asarray(phi), x)
        + np.repeat([-0.3, -0.1, 0.1, 0.3], 10)
        + rng.normal(0, 0.1, len(x))
    )
    return y, x, groups


def optimizer_for(model, phi, **kwargs):
    optimizer = nlmm.NLMMOptimizer(*problem(model, phi), model, [0], **kwargs)
    optimizer._start_phi = np.asarray(phi, dtype=float)
    return optimizer


def shifted_predict(original):
    def predict(self, params, x):
        return original(self, params, x) + 7

    return predict


def scaled_gradient(original):
    def gradient(self, params, x):
        return original(self, params, x) * 0.75

    return gradient


def assert_python_evaluation(model, phi):
    with (
        patch.object(nlmm, "_HAS_RUST", True),
        patch.object(
            nlmm,
            "_nlmm_deviance_rust_with_status",
            side_effect=AssertionError("custom model reached native evaluator"),
        ) as native,
    ):
        default = optimizer_for(model, phi)
        python = optimizer_for(model, phi, use_rust=False)
        actual = default._evaluate(np.array([0.8]))
        expected = python._evaluate(np.array([0.8]))
    assert not default.use_rust
    native.assert_not_called()
    assert actual[0] != 1e100
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("model_type,name,phi", BUILTINS)
@pytest.mark.parametrize("customization", ["predict", "gradient"])
def test_subclasses_keep_their_implementation(model_type, name, phi, customization):
    method = (
        shifted_predict(model_type.predict)
        if customization == "predict"
        else scaled_gradient(model_type.gradient)
    )
    subclass = type("CustomizedModel", (model_type,), {customization: method})
    assert_python_evaluation(subclass(), phi)


@pytest.mark.parametrize("model_type,name,phi", BUILTINS)
@pytest.mark.parametrize("customization", ["predict", "gradient"])
@pytest.mark.parametrize("scope", ["instance", "class"])
def test_replaced_builtin_methods_use_python(model_type, name, phi, customization, scope):
    method = (
        shifted_predict(model_type.predict)
        if customization == "predict"
        else scaled_gradient(model_type.gradient)
    )
    model = model_type()
    target = model if scope == "instance" else model_type
    replacement = MethodType(method, model) if scope == "instance" else method
    with patch.object(target, customization, replacement):
        assert_python_evaluation(model, phi)


@pytest.mark.parametrize("model_type,name,phi", BUILTINS)
@pytest.mark.parametrize("spelling", ["lower", "upper"])
def test_custom_model_name_never_selects_a_builtin_formula(model_type, name, phi, spelling):
    # Deliberately use fewer parameters than the native formulas require.
    model = CustomModel(
        lambda p, x: p[0] + 2 * x,
        lambda p, x: np.ones((len(x), 1)),
        ["a"],
        name=getattr(name, spelling)(),
    )
    assert_python_evaluation(model, [3.0])


@pytest.mark.parametrize("model_type,name,phi", BUILTINS)
def test_original_builtin_uses_native_formula_even_with_a_different_display_name(
    model_type, name, phi
):
    model = model_type()
    y, x, groups = problem(model, phi)
    output = (-12.0, np.asarray(phi), np.zeros((4, 1)), 0.1, True)
    with (
        patch.object(nlmm, "_HAS_RUST", True),
        patch.object(model_type, "name", property(lambda self: "display label")),
        patch.object(
            nlmm, "_rust_nlmm_deviance_with_status", return_value=output, create=True
        ) as native,
    ):
        optimizer = nlmm.NLMMOptimizer(y, x, groups, model, [0])
        optimizer._start_phi = np.asarray(phi)
        actual = optimizer._evaluate(np.array([0.8]))
    assert optimizer.use_rust
    assert native.call_count == 1
    assert native.call_args.args[4] == name
    for value, reference in zip(actual, output, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("available,requested", [(False, True), (True, False), (False, False)])
def test_python_fallback_when_native_is_unavailable_or_disabled(available, requested):
    with (
        patch.object(nlmm, "_HAS_RUST", available),
        patch.object(
            nlmm, "_nlmm_deviance_rust_with_status", side_effect=AssertionError("native disabled")
        ) as native,
    ):
        optimizer = optimizer_for(SSasymp(), [10.0, 2.0, -1.0], use_rust=requested)
        assert optimizer.objective(np.array([0.8])) != 1e100
    assert not optimizer.use_rust
    native.assert_not_called()


@pytest.mark.parametrize("model_type,name,phi", BUILTINS)
def test_native_wrapper_rejects_unsupported_models_before_calling_extension(model_type, name, phi):
    model = CustomModel(lambda p, x: np.full(len(x), p[0]), None, ["a"], name=name)
    y, x, groups = problem(model, [3.0])
    with (
        patch.object(
            nlmm,
            "_rust_nlmm_deviance_with_status",
            side_effect=AssertionError("unsupported native model"),
            create=True,
        ) as native,
        pytest.raises(ValueError, match="unmodified built-in model"),
    ):
        nlmm._nlmm_deviance_rust(
            np.array([0.8]),
            y,
            x,
            groups,
            model,
            np.array([3.0]),
            np.zeros((4, 1)),
            [0],
            0.1,
            np.ones(len(y)),
        )
    native.assert_not_called()


@pytest.mark.parametrize("kind", ["subclass", "custom", "instance"])
@pytest.mark.parametrize("weighted", [False, True])
def test_customized_model_fits_and_refits_match_its_python_implementation(kind, weighted):
    phi = np.array([10.0, 2.0, -1.0])
    predict = shifted_predict(SSasymp.predict)
    if kind == "subclass":
        model = type("ShiftedAsymp", (SSasymp,), {"predict": predict})()
    elif kind == "custom":
        original = SSasymp()
        model = CustomModel(
            lambda p, x: original.predict(p, x) + 7,
            original.gradient,
            original.param_names,
            name=original.name,
        )
    else:
        model = SSasymp()
        model.predict = MethodType(predict, model)
    y, x, groups = problem(model, phi)
    offset = np.linspace(1.0, 2.0, len(y))
    data = pd.DataFrame(dict(y=y + offset, x=x, g=groups))
    kwargs = dict(
        x_var="x",
        y_var="y",
        group_var="g",
        random_params=["Asym"],
        start=dict(zip(model.param_names, phi, strict=True)),
        offset=offset,
    )
    if weighted:
        kwargs["weights"] = np.linspace(0.5, 2.0, len(y))
    with patch.object(nlmm, "_HAS_RUST", False):
        expected = nlmer(model, data, **kwargs)
        expected_refit = expected.refit(data.y.to_numpy() + 0.1)
    with (
        patch.object(nlmm, "_HAS_RUST", True),
        patch.object(
            nlmm,
            "_nlmm_deviance_rust_with_status",
            side_effect=AssertionError("custom model reached native evaluator"),
        ) as native,
    ):
        actual = nlmer(model, data, **kwargs)
        actual_refit = actual.refit(data.y.to_numpy() + 0.1)
    native.assert_not_called()
    for fit, reference in [(actual, expected), (actual_refit, expected_refit)]:
        for attribute in ("phi", "theta", "b", "sigma", "deviance", "converged"):
            np.testing.assert_array_equal(getattr(fit, attribute), getattr(reference, attribute))
        np.testing.assert_array_equal(fit.fitted(), reference.fitted())
        assert abs(np.mean(fit.fitted() - fit.y)) < 0.1
