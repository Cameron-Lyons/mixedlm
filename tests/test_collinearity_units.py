"""Collinearity depends on information geometry, not numerical units."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from mixedlm import check_collinearity
from numpy.testing import assert_allclose


def _model(design, weights, *, nonlinear=False):
    if nonlinear:
        return SimpleNamespace(
            isLMM=lambda: False,
            isGLMM=lambda: False,
            isNLMM=lambda: True,
            model=SimpleNamespace(gradient=lambda phi, x: design, param_names=["x", "z"]),
            phi=np.zeros(2),
            x=np.zeros(len(design)),
            _weights=weights,
        )
    return SimpleNamespace(
        isLMM=lambda: True,
        matrices=SimpleNamespace(X=design, fixed_names=["x", "z"], weights=weights),
    )


@pytest.mark.parametrize(
    ("scale", "shift", "weight_scale"),
    [(1.0, 1e8, 1.0), (1e200, 0.0, 1e200), (1e-200, 0.0, 1e-200), (1.0, 0.0, 1e308)],
)
def test_vif_preserves_predictors_under_changes_of_units(scale, shift, weight_scale):
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 4)
    z = 0.75 * x + 0.25 * np.tile([-1.0, 1.0, -1.0, 1.0], 4)
    weights = np.linspace(0.25, 1.0, len(x))
    expected = check_collinearity(_model(np.column_stack([x, z]), weights))

    with np.errstate(all="raise"):
        result = check_collinearity(
            _model(np.column_stack([scale * x + shift, z]), weights * weight_scale)
        )

    assert result.terms == ("x", "z")
    assert result.rank == result.n_columns == 2
    assert_allclose(result.vif, expected.vif, rtol=1e-13)
    assert_allclose(result.condition_indices, expected.condition_indices, rtol=1e-13)


def test_opposite_extreme_predictors_keep_closed_form_vif():
    x = np.tile([-1e308, -1e308, 1e308, 1e308], 4)
    z = np.tile([-1e308, 1e308, -1e308, 1e308], 4)

    with np.errstate(all="raise"):
        result = check_collinearity(_model(np.column_stack([x, z]), np.full(len(x), 1e308)))

    assert result.terms == ("x", "z")
    assert_allclose(result.vif, 1.0, atol=1e-15)
    assert result.condition_number == pytest.approx(1.0)


def test_subnormal_predictor_is_retained():
    tiny = np.nextafter(0.0, 1.0)
    x = np.tile([-tiny, -tiny, tiny, tiny], 4)
    z = np.tile([-1.0, 1.0, -1.0, 1.0], 4)

    result = check_collinearity(_model(np.column_stack([x, z]), np.ones(len(x))))

    assert result.terms == ("x", "z")
    assert_allclose(result.vif, 1.0, atol=1e-15)


def test_extreme_nonlinear_gradient_units_preserve_uncentered_geometry():
    x = np.arange(1.0, 9.0)
    z = np.tile([1.0, 2.0], 4)
    weights = np.linspace(0.5, 1.0, len(x))
    expected = check_collinearity(_model(np.column_stack([x, z]), weights, nonlinear=True))

    result = check_collinearity(
        _model(np.column_stack([x * 1e200, z * 1e-200]), weights * 1e200, nonlinear=True)
    )

    assert result.terms == ("x", "z")
    assert_allclose(result.vif, expected.vif, rtol=1e-13)
    assert_allclose(result.condition_indices, expected.condition_indices, rtol=1e-13)


def test_empty_model_frame_is_rejected():
    with pytest.raises(ValueError, match="at least one observation"):
        check_collinearity(_model(np.empty((0, 2)), np.empty(0)))
