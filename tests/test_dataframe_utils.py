from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.utils.dataframe import concat_columns_as_string, factorize_levels


class TestFactorizeLevels:
    def test_factorizes_strings_in_sorted_order(self) -> None:
        values = np.array(["b", "a", "missing", "b"], dtype=object)

        indices, level_map = factorize_levels(values)

        assert indices.dtype == np.int64
        assert indices.tolist() == [1, 0, 2, 1]
        assert level_map == {"a": 0, "b": 1, "missing": 2}

    def test_factorizes_numeric_numpy_scalars(self) -> None:
        values = np.array([20, 10, 30, 10], dtype=np.int64)

        indices, level_map = factorize_levels(values)

        assert indices.tolist() == [1, 0, 2, 0]
        assert level_map == {10: 0, 20: 1, 30: 2}

    def test_missing_value_does_not_alias_literal_nan_level(self) -> None:
        values = np.array([np.nan, "nan"], dtype=object)

        indices, level_map = factorize_levels(values)

        assert indices.tolist() == [-1, 0]
        assert level_map == {"nan": 0}

    def test_empty_values_return_empty_outputs(self) -> None:
        indices, level_map = factorize_levels(np.array([], dtype=object))

        assert indices.tolist() == []
        assert level_map == {}


class TestConcatColumnsAsString:
    def test_vectorized_pandas_result_matches_row_keys(self) -> None:
        data = pd.DataFrame(
            {
                "site": ["north", "south", "north"],
                "subject": [10, 11, 12],
                "visit": [1, 2, 3],
            },
            index=[9, 4, 7],
        )

        result = concat_columns_as_string(data, ["site", "subject", "visit"])

        assert result.tolist() == ["north/10/1", "south/11/2", "north/12/3"]

    def test_preserves_string_representations_of_missing_values(self) -> None:
        data = pd.DataFrame(
            {
                "left": ["a", None, "c"],
                "right": [1.0, 2.0, np.nan],
            }
        )

        result = concat_columns_as_string(data, ["left", "right"], separator=":")

        assert result.tolist() == ["a:1.0", "None:2.0", "c:nan"]

    def test_empty_column_list_returns_empty_keys(self) -> None:
        data = pd.DataFrame({"x": [1, 2, 3]})

        result = concat_columns_as_string(data, [])

        assert result.tolist() == ["", "", ""]

    def test_polars_path_matches_pandas(self) -> None:
        pl = pytest.importorskip("polars")
        pandas_data = pd.DataFrame({"g1": ["a", "b"], "g2": [1, 2]})
        polars_data = pl.DataFrame({"g1": ["a", "b"], "g2": [1, 2]})

        pandas_result = concat_columns_as_string(pandas_data, ["g1", "g2"])
        polars_result = concat_columns_as_string(polars_data, ["g1", "g2"])

        assert polars_result.tolist() == pandas_result.tolist()


def test_nested_design_uses_distinct_vectorized_group_keys() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 0.0, 1.0],
            "site": ["a", "a", "b", "b"],
            "subject": [1, 2, 1, 2],
        }
    )
    formula = parse_formula("y ~ x + (1 | site/subject)")

    matrices = build_model_matrices(formula, data)

    structure = matrices.random_structures[0]
    assert structure.level_map == {"a/1": 0, "a/2": 1, "b/1": 2, "b/2": 3}
    assert matrices.Z.shape == (4, 4)
    assert np.array_equal(matrices.Z.toarray(), np.eye(4))


def test_missing_group_does_not_alias_literal_nan_group() -> None:
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0],
            "group": [np.nan, "nan", "a"],
        }
    )
    formula = parse_formula("y ~ x + (1 | group)")

    matrices = build_model_matrices(formula, data)

    assert matrices.random_structures[0].level_map == {"a": 0, "nan": 1}
    assert np.array_equal(
        matrices.Z.toarray(),
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
    )


def test_nested_polars_design_matches_pandas() -> None:
    pl = pytest.importorskip("polars")
    values = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x": [0.0, 1.0, 0.0, 1.0],
        "site": ["a", "a", "b", "b"],
        "subject": [1, 2, 1, 2],
    }
    formula = parse_formula("y ~ x + (1 | site/subject)")

    pandas_matrices = build_model_matrices(formula, pd.DataFrame(values))
    polars_matrices = build_model_matrices(formula, pl.DataFrame(values))

    assert polars_matrices.random_structures[0].level_map == {
        "a/1": 0,
        "a/2": 1,
        "b/1": 2,
        "b/2": 3,
    }
    assert np.array_equal(polars_matrices.Z.toarray(), pandas_matrices.Z.toarray())
