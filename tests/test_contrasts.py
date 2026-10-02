from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.utils import contrasts as contrasts_module
from mixedlm.utils.contrasts import (
    apply_contrasts,
    apply_contrasts_array,
    contr_poly,
    contr_treatment,
)


@pytest.mark.parametrize("n_levels", [3, 4, 5, 10])
def test_polynomial_contrasts_are_centered_and_orthonormal(n_levels: int) -> None:
    contrasts = contr_poly(n_levels)

    assert contrasts.shape == (n_levels, n_levels - 1)
    np.testing.assert_allclose(contrasts.sum(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        contrasts.T @ contrasts,
        np.eye(n_levels - 1),
        atol=1e-14,
    )
    assert np.all(contrasts[-1, :] > 0)


def test_polynomial_contrasts_match_four_level_reference() -> None:
    expected = np.array(
        [
            [-0.6708203932, 0.5, -0.2236067977],
            [-0.2236067977, -0.5, 0.6708203932],
            [0.2236067977, -0.5, -0.6708203932],
            [0.6708203932, 0.5, 0.2236067977],
        ]
    )

    np.testing.assert_allclose(contr_poly(4), expected, atol=1e-10)


def test_array_encoding_handles_known_and_unknown_levels() -> None:
    categories = ["A", "B", "C"]
    values = np.array(["B", "A", "C", "unknown", "B"], dtype=object)

    columns, names = apply_contrasts_array(
        values,
        "group",
        contr_treatment(len(categories)),
        categories,
    )

    encoded = np.column_stack(columns)
    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [np.nan, np.nan],
            [1.0, 0.0],
        ]
    )
    assert names == ["group.1", "group.2"]
    np.testing.assert_allclose(encoded, expected, equal_nan=True)


def test_series_and_array_encoding_match() -> None:
    categories = ["low", "medium", "high"]
    values = np.array(["low", "high", "medium", "low"], dtype=object)
    matrix = contr_poly(len(categories))

    array_columns, array_names = apply_contrasts_array(values, "dose", matrix, categories)
    series_columns, series_names = apply_contrasts(
        pd.Series(values),
        "dose",
        matrix,
        categories,
    )

    assert series_names == array_names
    np.testing.assert_allclose(
        np.column_stack(series_columns),
        np.column_stack(array_columns),
    )


@pytest.fixture(params=[0, 4096], ids=["vectorized", "small"])
def encoding_size_threshold(request, monkeypatch):
    monkeypatch.setattr(contrasts_module, "_CONTRAST_VECTORIZE_MIN_ROWS", request.param)


@pytest.mark.parametrize(
    "categories,values,expected_codes",
    [
        (["z", "a", "m"], ["a", "z", "unknown", "m", None], [1, 0, -1, 2, -1]),
        ([2, 1], [1, 2, 3, np.nan], [1, 0, -1, -1]),
        ([1, "1"], [True, 1, "1", "unknown"], [0, 0, 1, -1]),
        ([False, True], [0, 1, False, True], [0, 1, 0, 1]),
        (["a", "b", "a"], ["a", "b", "unknown"], [2, 1, -1]),
        ([None, "a"], [None, "a", np.nan], [0, 1, -1]),
        (["a", "b"], [None, "unknown", np.nan], [-1, -1, -1]),
        ([], ["unknown", None], [-1, -1]),
    ],
)
def test_contrast_encoding_preserves_level_order_and_lookup_semantics(
    categories, values, expected_codes, encoding_size_threshold
) -> None:
    matrix = np.arange(len(categories) * 3, dtype=np.float64).reshape(len(categories), 3)
    columns, names = apply_contrasts_array(
        np.asarray(values, dtype=object), "x", matrix, categories
    )
    expected = np.array(
        [matrix[index] if index >= 0 else np.full(3, np.nan) for index in expected_codes]
    )

    assert names == ["x.1", "x.2", "x.3"]
    np.testing.assert_array_equal(np.column_stack(columns), expected)


@pytest.mark.parametrize(
    "values,categories",
    [
        (np.array([False, True]), [0, 1]),
        (np.array([False, True], dtype=object), [0, 1]),
        (np.array([0, 1]), [False, True]),
        (np.array([0.0, 1.0]), [False, True]),
        (np.array([0, 1], dtype=object), [False, True]),
    ],
)
def test_numeric_and_boolean_levels_keep_dictionary_equality(
    values, categories, encoding_size_threshold
) -> None:
    columns, _ = apply_contrasts_array(values, "x", np.array([[2.0], [3.0]]), categories)

    np.testing.assert_array_equal(columns[0], [2.0, 3.0])


def test_missing_category_keys_keep_identity_sensitive_lookup(encoding_size_threshold) -> None:
    nan_level = float("nan")
    values = np.array([nan_level, float("nan"), "a"], dtype=object)
    columns, _ = apply_contrasts_array(values, "x", np.array([[2.0], [3.0]]), [nan_level, "a"])

    np.testing.assert_array_equal(columns[0], [2.0, np.nan, 3.0])


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_contrast_encoding_outputs_independent_float64_columns(
    dtype, encoding_size_threshold
) -> None:
    matrix = np.arange(12, dtype=dtype).reshape(3, 4)
    original_matrix = matrix.copy()
    columns, _ = apply_contrasts_array(np.array(["c", "a", "b"]), "x", matrix, ["a", "b", "c"])

    assert all(column.dtype == np.float64 for column in columns)
    np.testing.assert_array_equal(np.column_stack(columns), original_matrix[[2, 0, 1]])
    columns[0][:] = -1
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(columns[1], original_matrix[[2, 0, 1], 1])


def test_empty_contrast_inputs(encoding_size_threshold) -> None:
    columns, names = apply_contrasts_array(np.empty(0, dtype=object), "x", np.zeros((2, 3)), [0, 1])
    assert names == ["x.1", "x.2", "x.3"]
    assert all(column.shape == (0,) for column in columns)

    columns, names = apply_contrasts_array(np.array([0, 1]), "x", np.empty((2, 0)), [0, 1])
    assert columns == []
    assert names == []


def test_tuple_category_levels(encoding_size_threshold) -> None:
    values = np.empty(4, dtype=object)
    values[:] = [(1, "a"), (2, "b"), (1, "a"), (3, "c")]
    categories = [(2, "b"), (1, "a")]
    columns, _ = apply_contrasts_array(values, "x", np.array([[2.0], [3.0]]), categories)

    np.testing.assert_array_equal(columns[0], [3.0, 2.0, 3.0, np.nan])
