from __future__ import annotations

import warnings
from decimal import Decimal, localcontext
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, parse_formula
from mixedlm.inference import linear_hypothesis
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats


def _model(kind="lmm", covariance=None, beta=None):
    rng = np.random.default_rng(20260930)
    frame = pd.DataFrame(
        {
            "y": np.ones(48),
            "x": rng.normal(size=48),
            "z": rng.normal(size=48),
            "g": np.repeat(np.arange(6), 8),
        }
    )
    formula = parse_formula("y ~ x + z + (1 | g)")
    matrices = build_model_matrices(formula, frame)
    beta = np.array([1.25, 0.3, -0.2]) if beta is None else np.asarray(beta, dtype=float)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=beta,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.8, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    covariance = (
        np.array([[0.07, 0.01, -0.002], [0.01, 0.04, 0.003], [-0.002, 0.003, 0.025]])
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    covariance.setflags(write=False)
    model.vcov = lambda: covariance
    return model


def _product(*values):
    with localcontext() as context:
        context.prec = 100
        result = Decimal(1)
        for value in values:
            result *= Decimal.from_float(float(value))
        return float(result)


def _assert_scaled(actual, expected):
    assert_allclose(actual, expected, rtol=5e-12, atol=2 * np.nextafter(0.0, 1.0))


SCALES = [
    [1.0, 1e-8],
    [1e-250, 1e250],
    [-1e250, 1e-250],
    [1e-200, 1e-200],
    [1e200, 1e200],
    [np.ldexp(1.0, -1070), np.ldexp(1.0, 1023)],
    [np.finfo(float).max, -np.finfo(float).max],
]


@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("test", ["auto", "F", "Chisq"])
def test_equivalent_equations_preserve_joint_and_row_tests(kind, test, scales):
    model = _model(kind)
    constraints = np.array([[0.0, 1.0, -1.0], [1.0, -0.5, 0.25]])
    rhs = np.array([0.25, 1.0])
    options = {"test": test, "level": 0.9}
    if test == "F":
        options["denominator_df"] = 21.5
    baseline = linear_hypothesis(model, constraints, rhs=rhs, **options)
    scales = np.asarray(scales)
    scaled = constraints * scales[:, None]
    targets = rhs * scales
    scaled.setflags(write=False)
    targets.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = linear_hypothesis(model, scaled, rhs=targets, **options)

    assert_allclose(result.statistic, baseline.statistic, rtol=5e-12)
    assert_allclose(result.p_value, baseline.p_value, rtol=5e-12)
    assert_allclose(result.row_statistic, baseline.row_statistic * np.sign(scales), rtol=5e-12)
    assert_allclose(result.row_p_value, baseline.row_p_value, rtol=5e-12)
    assert result.test == baseline.test
    assert result.denominator_df == baseline.denominator_df
    assert result.numerator_df == baseline.numerator_df
    for name in ["estimate", "difference", "std_error"]:
        factors = np.abs(scales) if name == "std_error" else scales
        expected = [
            _product(value, scale)
            for value, scale in zip(getattr(baseline, name), factors, strict=True)
        ]
        _assert_scaled(getattr(result, name), expected)
    expected_covariance = [
        [_product(baseline.covariance[i, j], scales[i], scales[j]) for j in range(2)]
        for i in range(2)
    ]
    _assert_scaled(result.covariance, expected_covariance)
    lower = [
        _product(baseline.conf_low[i] if scales[i] > 0 else baseline.conf_high[i], scales[i])
        for i in range(2)
    ]
    upper = [
        _product(baseline.conf_high[i] if scales[i] > 0 else baseline.conf_low[i], scales[i])
        for i in range(2)
    ]
    _assert_scaled(result.conf_low, lower)
    _assert_scaled(result.conf_high, upper)
    assert_array_equal(result.constraints, scaled)
    assert_array_equal(result.rhs, targets)
    for name in ["constraints", "rhs", "estimate", "difference", "std_error", "covariance"]:
        assert not getattr(result, name).flags.writeable
    assert not np.shares_memory(result.constraints, scaled)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize(
    "format", ["array", "list", "object", "frame", "rows", "nested", "unmasked"]
)
def test_all_constraint_formats_preserve_scaled_equations(kind, format):
    model = _model(kind)
    constraints = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 0.0]])
    baseline = linear_hypothesis(model, constraints, rhs=[0.25, 1.0])
    scale = np.array([1e-200, 1e200])
    values = constraints * scale[:, None]
    names = model.matrices.fixed_names
    rows = [dict(zip(names, row, strict=True)) for row in values]
    alternatives = {
        "array": values,
        "list": values.tolist(),
        "object": values.astype(object),
        "frame": pd.DataFrame(values, columns=names)[names[::-1]],
        "rows": rows,
        "nested": {"A": rows[0], "B": rows[1]},
        "unmasked": np.ma.array(values, mask=False),
    }

    actual = linear_hypothesis(model, alternatives[format], rhs=np.array([0.25, 1.0]) * scale)

    assert_allclose(actual.statistic, baseline.statistic, rtol=5e-12)
    assert_allclose(actual.row_statistic, baseline.row_statistic, rtol=5e-12)
    assert_array_equal(actual.constraints, values)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize(
    "units", [[1e-150, 1.0, 1e150], [1e150, 1e-150, 1.0], [1.0, 1e150, 1e-150]]
)
def test_coefficient_unit_changes_preserve_hypotheses(kind, units):
    model = _model(kind)
    constraints = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 1.0]])
    expected = linear_hypothesis(model, constraints, rhs=[1.0, 0.5])
    units = np.asarray(units)
    covariance = model.vcov() * units[:, None] * units[None, :]
    changed = _model(kind, covariance=covariance, beta=model.beta * units)

    actual = linear_hypothesis(changed, constraints / units, rhs=[1.0, 0.5])

    for name in [
        "estimate",
        "difference",
        "std_error",
        "covariance",
        "row_statistic",
        "conf_low",
        "conf_high",
    ]:
        assert_allclose(getattr(actual, name), getattr(expected, name), rtol=5e-12, atol=1e-14)
    assert_allclose(actual.statistic, expected.statistic, rtol=5e-12)
    assert_allclose(actual.p_value, expected.p_value, rtol=5e-12)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_coefficient_variances_span_the_finite_float_range(kind):
    variances = np.array([np.nextafter(0.0, 1.0), 1.0, np.finfo(float).max])
    z = np.array([0.25, -1.0, 2.0])
    model = _model(kind, covariance=np.diag(variances), beta=np.sqrt(variances) * z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = linear_hypothesis(model, np.eye(3))

    assert_allclose(result.row_statistic, z, rtol=5e-15)
    assert np.isfinite(result.covariance).all()
    _assert_scaled(np.diag(result.covariance), variances)
    divisor = 3 if kind == "lmm" else 1
    assert_allclose(result.statistic, np.sum(z**2) / divisor, rtol=5e-15)


@pytest.mark.parametrize(
    "scale,beta,variance,rhs,expected_z",
    [
        (1e-300, 0.0, 1e300, 1e10, -1e160),
        (1e300, 1e100, 1e300, 0.0, 1e-50),
        (1e-300, 1e-100, 1e-300, 0.0, 1e50),
        (
            np.ldexp(1.0, -1020),
            0.0,
            np.ldexp(1.0, -1000),
            np.ldexp(1.0, -1020),
            -np.ldexp(1.0, 500),
        ),
        (np.finfo(float).max, 1.25, 0.04, np.finfo(float).max, 1.25),
    ],
)
def test_nonzero_nulls_and_unrepresentable_output_units_keep_finite_tests(
    scale, beta, variance, rhs, expected_z
):
    model = _model(covariance=np.diag([1.0, variance, 1.0]), beta=[0.0, beta, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = linear_hypothesis(model, {"x": scale}, rhs=rhs)

    assert_allclose(actual.row_statistic, [expected_z], rtol=5e-12)
    assert_allclose(
        actual.row_p_value, 2 * stats.t.sf(abs(expected_z), model.df_residual()), rtol=5e-12
    )
    for name in ["estimate", "difference", "std_error", "conf_low", "conf_high", "covariance"]:
        assert not np.isnan(getattr(actual, name)).any()
    if scale == np.finfo(float).max:
        assert np.isinf(actual.estimate[0])
        assert np.isfinite(actual.difference[0])
        assert np.isfinite(actual.conf_low[0])


@pytest.mark.parametrize(
    "constraints,beta,variances,estimate",
    [
        ([3e250, 3e250, 1.0], [1.0, -1.0, -7.0], [1e-300, 1e-300, 1.0], -7.0),
        ([1e300, 1e300, 1e-300], [1e300, -1e300, 1e300], [0.0, 0.0, 1e300], 1.0),
        ([1e300, 1e-300, 0.0], [0.0, 1e300, 0.0], [0.0, 1e300, 1.0], 1.0),
    ],
)
def test_cancelling_and_out_of_range_products_retain_representable_estimates(
    constraints, beta, variances, estimate
):
    model = _model(covariance=np.diag(variances), beta=beta)

    result = linear_hypothesis(model, constraints)

    assert_allclose(result.estimate, [estimate], rtol=5e-15)
    with localcontext() as context:
        context.prec = 100
        variance = sum(
            Decimal.from_float(c) ** 2 * Decimal.from_float(v)
            for c, v in zip(constraints, variances, strict=True)
        )
        se = float(variance.sqrt())
    assert_allclose(result.std_error, [se], rtol=5e-15)
    assert_allclose(result.row_statistic, [estimate / se], rtol=5e-15)


def test_f_statistic_avoids_overflow_before_dividing_by_restriction_count():
    model = _model(covariance=np.eye(3), beta=[2e154, 0.0, 0.0])

    result = linear_hypothesis(model, np.eye(3))

    with localcontext() as context:
        context.prec = 100
        expected = float(Decimal.from_float(2e154) ** 2 / 3)
    assert np.isfinite(result.statistic)
    assert_allclose(result.statistic, expected, rtol=5e-15)


INVALID_NUMERIC = [
    (np.array([0, 1 + 2j, 0]), 0.0, "real"),
    (np.array([0, np.complex128(1 + 2j), 0], dtype=object), 0.0, "real"),
    ([0, 1 + 0j, 0], 0.0, "real"),
    ({"x": np.complex64(1 + 1j)}, 0.0, "real"),
    (pd.DataFrame({"x": [1 + 1j]}), 0.0, "real"),
    ([{"x": 1 + 1j}], 0.0, "real"),
    ({"A": {"x": 1 + 1j}}, 0.0, "real"),
    ({"x": 1.0}, np.complex128(1 + 2j), "real"),
    ({"x": 1.0}, np.array([1 + 0j]), "real"),
    ({"x": 1.0}, np.array([np.complex128(1 + 1j)], dtype=object), "real"),
    (np.ma.array([0.0, 1.0, 0.0], mask=[False, True, False]), 0.0, "masked"),
    ({"x": np.ma.masked}, 0.0, "masked"),
    ({"x": 1.0}, np.ma.array([1.0], mask=[True]), "masked"),
    ({"x": 1.0}, np.ma.masked, "masked"),
]


@pytest.mark.parametrize("hypothesis,rhs,match", INVALID_NUMERIC)
def test_complex_and_masked_inputs_fail_before_covariance(hypothesis, rhs, match):
    model = _model()

    def unexpected():
        raise AssertionError("Invalid inputs do not need covariance")

    model.vcov = unexpected
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=match):
            linear_hypothesis(model, hypothesis, rhs=rhs)


@pytest.mark.parametrize(
    "options",
    [
        {"test": "invalid"},
        {"test": None},
        {"test": "F", "denominator_df": 0},
        {"test": "F", "denominator_df": "bad"},
        {"test": "Chisq", "denominator_df": 5},
    ],
)
def test_invalid_test_options_fail_before_covariance(options):
    model = _model()

    def unexpected():
        raise AssertionError("Invalid options do not need covariance")

    model.vcov = unexpected
    with pytest.raises((TypeError, ValueError)):
        linear_hypothesis(model, {"x": 1}, **options)


@pytest.mark.parametrize("scales", [[1e-250, 1e250], [1e250, -1e-250]])
def test_dependent_equations_fail_before_covariance_at_different_scales(scales):
    model = _model()

    def unexpected():
        raise AssertionError("Dependent equations do not need covariance")

    model.vcov = unexpected
    with pytest.raises(ValueError, match="linearly independent"):
        linear_hypothesis(
            model, np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]) * np.array(scales)[:, None]
        )


@pytest.mark.parametrize(
    "block,match",
    [
        ([[1.0, 1.0], [1.0, 1.0]], "not estimable"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive definite"),
        ([[-1.0, 0.0], [0.0, 1.0]], "variance"),
        ([[0.0, 1.0], [1.0, 1.0]], "variance"),
        ([[0.0, 0.0], [0.0, 1.0]], "variance|estimable"),
    ],
)
def test_invalid_hypothesis_covariance_is_rejected(block, match):
    covariance = np.eye(3)
    covariance[:2, :2] = block
    model = _model(covariance=covariance)

    with pytest.raises(ValueError, match=match):
        linear_hypothesis(model, np.eye(3)[:2])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_estimable_restrictions_allow_unrelated_zero_variance_coefficients(kind):
    model = _model(kind, covariance=np.diag([0.0, 1.0, 4.0]), beta=[1.0, 0.5, -1.0])

    result = linear_hypothesis(model, np.eye(3)[1:])

    assert_allclose(result.row_statistic, [0.5, -0.5])
    assert_allclose(result.statistic, 0.25 if kind == "lmm" else 0.5)


def test_single_restriction_needs_no_matrix_decomposition(monkeypatch):
    model = _model()
    module = import_module("mixedlm.inference.hypothesis")

    def unexpected(*args, **kwargs):
        raise AssertionError("A single restriction needs no matrix decomposition")

    monkeypatch.setattr(np.linalg, "matrix_rank", unexpected)
    monkeypatch.setattr(module.linalg, "cholesky", unexpected)
    for hypothesis in [{"x": 1e250}, {"x": 1e-250, "z": -1e-250}]:
        result = linear_hypothesis(model, hypothesis)
        assert np.isfinite(result.statistic)


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int64", "Int64", "uint64", "UInt64", "float64", "Float64", "object", "string"],
)
def test_dataframe_numeric_dtypes_keep_their_existing_conversion(dtype):
    model = _model()
    frame = pd.DataFrame({"x": pd.Series([1], dtype=dtype)})

    result = linear_hypothesis(model, frame)
    expected = linear_hypothesis(model, {"x": 1})

    assert_array_equal(result.constraints, expected.constraints)
    assert_array_equal(result.estimate, expected.estimate)
    assert result.statistic == expected.statistic


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "string"])
def test_nullable_dataframe_missing_constraints_fail_before_covariance(dtype):
    model = _model()
    frame = pd.DataFrame({"x": pd.Series([1, None], dtype=dtype), "z": [0, 1]})

    def unexpected():
        raise AssertionError("Missing constraints do not need covariance")

    model.vcov = unexpected
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        linear_hypothesis(model, frame)
