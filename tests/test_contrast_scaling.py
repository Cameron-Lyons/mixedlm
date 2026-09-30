from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_emmeans import _synthetic_emmeans

module = import_module("mixedlm.inference.emmeans")


def _pairwise_rows():
    return np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]], dtype=float)


@pytest.mark.parametrize("df", [17.0, np.inf])
@pytest.mark.parametrize("adjust", ["none", "bonferroni", "holm", "fdr", "tukey", "dunnett"])
@pytest.mark.parametrize("exponent", [-300, -200, 200, 300])
def test_contrast_tests_do_not_depend_on_coefficient_units(df, adjust, exponent):
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    means._df = df
    coefficients = _pairwise_rows()
    reference = means.contrast(coefficients, adjust=adjust)
    scales = np.array([1.0, -1.0, 0.5]) * 10.0**exponent
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = means.contrast(coefficients * scales[:, None], adjust=adjust)

    assert_allclose(result.estimate / scales, reference.estimate, rtol=1e-12)
    assert_allclose(result.se / np.abs(scales), reference.se, rtol=1e-12)
    assert_allclose(result.t_ratio, reference.t_ratio * np.sign(scales), rtol=1e-12)
    assert_allclose(result.p_value, reference.p_value, rtol=1e-12, atol=1e-15)
    assert result.contrast == reference.contrast


@pytest.mark.parametrize("layout", ["readonly", "strided", "fortran"])
@pytest.mark.parametrize("limit", [1, 13, 1_000_000])
def test_general_rows_with_mixed_scales_match_direct_matrix_inference(monkeypatch, layout, limit):
    means = _synthetic_emmeans(n_levels=7, n_beta=5)
    rng = np.random.default_rng(420)
    coefficients = rng.normal(size=(5, 7))
    scales = np.array([1e-250, -1e250, 1.0, -0.125, 2e-200])
    custom = coefficients * scales[:, None]
    if layout == "readonly":
        custom.setflags(write=False)
    elif layout == "strided":
        custom = np.repeat(custom, 2, axis=0)[::2]
    else:
        custom = np.asfortranarray(custom)
    original = custom.copy()
    monkeypatch.setattr(module, "_MAX_CONTRAST_ELEMENTS", limit, raising=False)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = means.contrast(custom, adjust="none")

    projected = coefficients @ means._L
    expected = projected @ means._beta
    expected_se = np.sqrt(np.diag(projected @ means._vcov @ projected.T))
    assert_allclose(result.estimate / scales, expected, rtol=1e-12)
    assert_allclose(result.se / np.abs(scales), expected_se, rtol=1e-12)
    assert_allclose(result.t_ratio, expected / expected_se * np.sign(scales), rtol=1e-12)
    assert_array_equal(custom, original)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.longdouble, np.int64, np.uint64]
)
def test_real_numeric_dtypes_use_supported_reference_distribution_inputs(dtype):
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    coefficients = np.array([[2, 0, 1, 0], [0, 3, 0, 1]], dtype=dtype)

    actual = means.contrast(coefficients)
    reference = means.contrast(coefficients.astype(np.float64))

    for name in ("estimate", "se", "t_ratio", "p_value"):
        assert_allclose(getattr(actual, name), getattr(reference, name), rtol=1e-12)
        assert getattr(actual, name).dtype == np.float64


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_limits_are_converted_before_row_scale_calculation(dtype):
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    limits = np.iinfo(dtype)
    coefficients = np.array([[limits.min, limits.max, 0, 1]], dtype=dtype)

    actual = means.contrast(coefficients)
    reference = means.contrast(coefficients.astype(np.float64))

    assert_allclose(actual.estimate, reference.estimate)
    assert_allclose(actual.se, reference.se)
    assert_allclose(actual.t_ratio, reference.t_ratio)


@pytest.mark.parametrize("scale", [1e-310, -1e-310])
def test_subnormal_coefficients_keep_full_precision_test_statistics(scale):
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    coefficients = _pairwise_rows()
    reference = means.contrast(coefficients)

    actual = means.contrast(coefficients * scale)

    assert_allclose(actual.estimate / scale, reference.estimate, rtol=1e-12)
    assert_allclose(actual.se / abs(scale), reference.se, rtol=1e-12)
    assert_allclose(actual.t_ratio, reference.t_ratio * np.sign(scale), rtol=1e-12)
    assert_allclose(actual.p_value, reference.p_value, rtol=1e-12)


@pytest.mark.parametrize("exponent", [-400, 400])
def test_extended_range_coefficients_are_normalized_before_float64_conversion(exponent):
    if np.finfo(np.longdouble).maxexp == np.finfo(np.float64).maxexp:
        pytest.skip("This platform's long double has no extended exponent range")
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    coefficients = _pairwise_rows()
    reference = means.contrast(coefficients)
    scale = np.longdouble(10) ** exponent

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = means.contrast(coefficients.astype(np.longdouble) * scale)

    assert_allclose(actual.t_ratio, reference.t_ratio, rtol=1e-12)
    assert_allclose(actual.p_value, reference.p_value, rtol=1e-12)
    # Estimates outside float64's range cannot be represented, but their tests can.
    if exponent < 0:
        assert_array_equal(actual.estimate, np.zeros(3))
        assert_array_equal(actual.se, np.zeros(3))
    else:
        assert np.isinf(actual.estimate).all()
        assert np.isposinf(actual.se).all()


@pytest.mark.parametrize("scale", [np.finfo(np.float64).max, np.nextafter(0.0, 1.0)])
def test_float64_limits_do_not_overflow_or_underflow_the_test_statistic(scale):
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    coefficients = _pairwise_rows()
    reference = means.contrast(coefficients)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        actual = means.contrast(coefficients * scale)

    assert_allclose(actual.t_ratio, reference.t_ratio, rtol=1e-12)
    assert_allclose(actual.p_value, reference.p_value, rtol=1e-12)


@pytest.mark.parametrize("n_means,n_beta", [(23, 3), (3, 23)])
def test_custom_projection_batches_bound_both_mean_and_model_dimensions(
    monkeypatch, n_means, n_beta
):
    means = _synthetic_emmeans(n_levels=n_means, n_beta=n_beta)
    rng = np.random.default_rng(421)
    custom = rng.normal(size=(17, n_means))
    limit = 50
    monkeypatch.setattr(module, "_MAX_CONTRAST_ELEMENTS", limit, raising=False)
    sizes = []
    original = module._rowwise_quadratic_form

    def observe(coefficients, covariance):
        sizes.append(len(coefficients))
        return original(coefficients, covariance)

    monkeypatch.setattr(module, "_rowwise_quadratic_form", observe)
    means.contrast(custom)

    assert sum(sizes) == len(custom)
    assert max(sizes) * max(n_means, n_beta) <= limit


@pytest.mark.parametrize("n_means,n_beta,n_rows", [(4, 3, 0), (4, 0, 3), (0, 0, 3)])
def test_empty_and_zero_dimensional_custom_contrasts(n_means, n_beta, n_rows):
    means = _synthetic_emmeans(n_levels=n_means, n_beta=n_beta)
    custom = np.zeros((n_rows, n_means))

    with np.errstate(divide="raise", invalid="raise"):
        result = means.contrast(custom)

    assert_array_equal(result.estimate, np.zeros(n_rows))
    assert_array_equal(result.se, np.zeros(n_rows))
    assert np.isnan(result.t_ratio).all()
    assert np.isnan(result.p_value).all()


def test_zero_rows_do_not_affect_other_contrasts():
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    coefficients = np.vstack([np.zeros(4), _pairwise_rows() * 1e200])
    reference = means.contrast(_pairwise_rows())

    result = means.contrast(coefficients)

    assert result.estimate[0] == result.se[0] == 0.0
    assert np.isnan(result.t_ratio[0])
    assert np.isnan(result.p_value[0])
    assert_allclose(result.t_ratio[1:], reference.t_ratio)
    assert_allclose(result.p_value[1:], reference.p_value)


def test_complex_coefficients_are_not_silently_discarded():
    means = _synthetic_emmeans(n_levels=4, n_beta=3)
    with pytest.raises(TypeError, match="real numeric"):
        means.contrast(np.ones((1, 4), dtype=complex) * (1 + 1j))


def test_normalization_preserves_cancellation_of_large_common_model_terms():
    means = _synthetic_emmeans(n_levels=3, n_beta=2)
    means._L = np.column_stack([np.full(3, 2.0**400), [1.0, 2.0, 4.0]])
    means._beta = np.array([2.0**400, 1.0])
    means._vcov = np.eye(2)
    coefficients = np.array([[3.0, -1.0, -2.0], [-5.0, 3.0, 2.0]])

    actual = means.contrast(coefficients)

    assert_array_equal(actual.estimate, [-7.0, 9.0])
    assert_array_equal(actual.se, [7.0, 9.0])
    assert_array_equal(actual.t_ratio, [-1.0, 1.0])
