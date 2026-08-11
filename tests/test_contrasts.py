from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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
