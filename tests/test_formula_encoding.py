import gc
import tracemalloc
import weakref

import numpy as np
import pandas as pd
import pytest
from mixedlm import parse_formula
from mixedlm.matrices import design
from mixedlm.utils.contrasts import get_contrast_matrix

LEVELS = {"a": ["c", "a", "b", "unused"], "b": ["v", "u"]}


def _data(backend):
    values = {
        "a": ["b", "c", None, "a", "c"],
        "b": ["v", "u", "v", "v", "u"],
        "x": [-2.0, -0.0, 1.5, 3.0, -1.0],
        "g": ["g1", "g0", "g1", "g0", "g1"],
    }
    if backend.startswith("polars"):
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(values)
        if backend == "polars_categorical":
            data = data.with_columns(pl.col("a", "b").cast(pl.Categorical))
        return data
    data = pd.DataFrame(values)
    if backend == "category":
        # Fitted levels must take precedence over the new frame's category order.
        for name, levels in LEVELS.items():
            data[name] = pd.Categorical(data[name], categories=levels[::-1])
    elif backend == "string":
        data[["a", "b"]] = data[["a", "b"]].astype("string")
    return data


def _expected(contrast, intercept):
    matrices = {
        name: get_contrast_matrix(len(levels), contrast[name]) for name, levels in LEVELS.items()
    }
    a_values = ["b", "c", None, "a", "c"]
    b_values = ["v", "u", "v", "v", "u"]

    def encode(values, levels, matrix):
        return np.array(
            [
                matrix[levels.index(value)] if value in levels else np.full(matrix.shape[1], np.nan)
                for value in values
            ]
        )

    a = encode(a_values, LEVELS["a"], matrices["a"])
    b = encode(b_values, LEVELS["b"], matrices["b"])
    x2 = np.array([-2.0, -0.0, 1.5, 3.0, -1.0]) ** 2
    a_names = [f"a.{i + 1}" for i in range(a.shape[1])]
    b_names = [f"b.{i + 1}" for i in range(b.shape[1])]
    if intercept:
        columns = [np.ones(5), *a.T]
        names = ["(Intercept)", *a_names]
    else:
        columns = list(encode(a_values, LEVELS["a"], np.eye(4)).T)
        names = [f"a.{level}" for level in LEVELS["a"]]
    columns.extend(b.T)
    names.extend(b_names)
    # Spell out the formula's term order; the final factor varies fastest.
    for i, a_name in enumerate(a_names):
        for j, b_name in enumerate(b_names):
            columns.append(a[:, i] * b[:, j])
            names.append(f"{a_name}:{b_name}")
    columns.append(x2)
    names.append("I(x**2)")
    for i, a_name in enumerate(a_names):
        columns.append(a[:, i] * x2)
        names.append(f"{a_name}:I(x**2)")
    for i, a_name in enumerate(a_names):
        for j, b_name in enumerate(b_names):
            columns.append(a[:, i] * b[:, j] * x2)
            names.append(f"{a_name}:{b_name}:I(x**2)")
    return np.column_stack(columns), names


@pytest.mark.parametrize(
    "backend", ["object", "category", "string", "polars", "polars_categorical"]
)
@pytest.mark.parametrize("spec", ["treatment", "sum", "custom"])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("target", ["fixed", "random"])
def test_repeated_factors_preserve_schema_values_and_column_order(backend, spec, intercept, target):
    data = _data(backend)
    contrast = (
        {
            "a": np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0], [0.5, -2.0]]),
            "b": np.array([[2.0], [-3.0]]),
        }
        if spec == "custom"
        else {"a": spec, "b": spec}
    )
    terms = "a + b + a:b + I(x**2) + a:I(x**2) + a:b:I(x**2)"
    if not intercept:
        terms = "0 + " + terms
    expected, names = _expected(contrast, intercept)
    if target == "fixed":
        matrix, actual_names = design.build_fixed_matrix(
            parse_formula(f"y ~ {terms}"), data, contrast, LEVELS
        )
        assert actual_names == names
        assert matrix.flags.c_contiguous
        assert matrix.dtype == np.float64
        np.testing.assert_array_equal(matrix, expected)
    else:
        matrix, structures = design.build_random_matrix(
            parse_formula(f"y ~ 1 + ({terms} | g)"), data, contrast, LEVELS
        )
        structure = structures[0]
        assert structure.term_names == names
        dense = np.zeros(matrix.shape)
        groups = ["g1", "g0", "g1", "g0", "g1"]
        for row, group in enumerate(groups):
            start = structure.level_map[group] * len(names)
            dense[row, start : start + len(names)] = expected[row]
        np.testing.assert_array_equal(matrix.toarray(), dense)


@pytest.mark.parametrize("n", [0, 1, 7])
@pytest.mark.parametrize("intercept", [False, True])
def test_numeric_powers_preserve_order_empty_rows_and_input(n, intercept):
    x = np.linspace(-2.0, 3.0, n)
    data = pd.DataFrame({"x": x, "z": x + 0.5})
    original = data.copy(deep=True)
    prefix = "" if intercept else "0 + "
    matrix, names = design.build_fixed_matrix(
        parse_formula(f"y ~ {prefix}x*z + I(x**2)*I(z**3) + I(x**0)"), data
    )
    expected = np.column_stack(
        [x, x + 0.5, x * (x + 0.5), x**2, (x + 0.5) ** 3, x**2 * (x + 0.5) ** 3, np.ones(n)]
    )
    expected_names = ["x", "z", "x:z", "I(x**2)", "I(z**3)", "I(x**2):I(z**3)", "I(x**0)"]
    if intercept:
        expected = np.column_stack([np.ones(n), expected])
        expected_names.insert(0, "(Intercept)")
    assert names == expected_names
    np.testing.assert_array_equal(matrix, expected)
    matrix[:] = -999
    pd.testing.assert_frame_equal(data, original)


def test_encodings_do_not_survive_a_build_or_mutate_contrasts():
    data = pd.DataFrame({"a": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]})
    formula = parse_formula("y ~ a*x")
    contrasts = {"a": np.array([[-1.0], [2.0]])}
    original = contrasts["a"].copy()
    first, _ = design.build_fixed_matrix(formula, data, contrasts)
    np.testing.assert_array_equal(contrasts["a"], original)
    data["x"] += 5
    contrasts["a"] *= 3
    second, _ = design.build_fixed_matrix(formula, data, contrasts)
    np.testing.assert_array_equal(first, [[1, -1, 1, -1], [1, 2, 2, 4], [1, -1, 3, -3]])
    np.testing.assert_array_equal(second, [[1, -3, 6, -18], [1, 6, 7, 42], [1, -3, 8, -24]])


def test_wide_interaction_matrix_does_not_keep_a_second_copy_of_product_columns():
    n = 4_000
    data = pd.DataFrame({name: pd.Categorical(np.arange(n) % 5) for name in "abc"})
    formula = parse_formula("y ~ a*b*c")
    # Warm dtype/import paths before measuring the build's allocations.
    design.build_fixed_matrix(formula, data.iloc[:1])
    gc.collect()
    tracemalloc.start()
    try:
        matrix, names = design.build_fixed_matrix(formula, data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert matrix.shape == (n, 125)
    assert len(names) == 125
    assert peak < matrix.nbytes * 1.4


def test_encoded_arrays_are_released_without_cyclic_collection(monkeypatch):
    data = pd.DataFrame({"a": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    formula = parse_formula("y ~ a*I(x**2)")
    references = []
    original = design._encode_categorical

    def encode(*args, **kwargs):
        columns, names = original(*args, **kwargs)
        references.extend(weakref.ref(column) for column in columns)
        return columns, names

    monkeypatch.setattr(design, "_encode_categorical", encode)
    collecting = gc.isenabled()
    gc.disable()
    try:
        matrix, _ = design.build_fixed_matrix(formula, data)
        assert np.isfinite(matrix).all()
        assert references
        assert all(reference() is None for reference in references)
    finally:
        if collecting:
            gc.enable()


@pytest.mark.parametrize(
    "backend", ["object", "category", "string", "polars", "polars_categorical"]
)
def test_empty_categorical_prediction_frame_preserves_fitted_columns(backend):
    data = _data(backend).head(0)
    matrix, names = design.build_fixed_matrix(
        parse_formula("y ~ a*b"), data, category_levels=LEVELS
    )
    assert matrix.shape == (0, 8)
    assert names == ["(Intercept)", "a.1", "a.2", "a.3", "b.1", "a.1:b.1", "a.2:b.1", "a.3:b.1"]


def test_interaction_before_main_effect_keeps_reduced_and_full_rank_encodings_distinct():
    data = pd.DataFrame({"a": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    matrix, names = design.build_fixed_matrix(parse_formula("y ~ 0 + a:x + a + a:I(x**2)"), data)
    assert names == ["a.1:x", "a.2:x", "a.a", "a.b", "a.c", "a.1:I(x**2)", "a.2:I(x**2)"]
    np.testing.assert_array_equal(
        matrix, [[0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 1, 0, 4, 0], [0, 3, 0, 0, 1, 0, 9]]
    )


def test_numeric_columns_use_fitted_categorical_levels():
    data = pd.DataFrame({"a": [2.0, 1.0, 2.0], "x": [1.0, 2.0, 3.0]})
    matrix, names = design.build_fixed_matrix(
        parse_formula("y ~ a*x"), data, category_levels={"a": [2.0, 1.0, 3.0]}
    )
    assert names == ["(Intercept)", "a.1", "a.2", "x", "a.1:x", "a.2:x"]
    np.testing.assert_array_equal(
        matrix, [[1, 0, 0, 1, 0, 0], [1, 1, 0, 2, 2, 0], [1, 0, 0, 3, 0, 0]]
    )


@pytest.mark.parametrize("n", [0, 1, 3])
def test_single_level_and_zero_column_contrasts(n):
    data = pd.DataFrame(
        {
            "a": pd.Categorical(["a"] * n, categories=["a"]),
            "b": pd.Categorical(["b"] * n, categories=["b", "c"]),
            "x": np.arange(n, dtype=np.float32),
        }
    )
    matrix, names = design.build_fixed_matrix(
        parse_formula("y ~ a*x + b*x"), data, contrasts={"b": np.empty((2, 0))}
    )
    assert names == ["(Intercept)", "a", "x", "a:x"]
    np.testing.assert_array_equal(
        matrix, np.column_stack([np.ones(n), np.ones(n), np.arange(n), np.arange(n)])
    )


def test_interactions_keep_left_to_right_arithmetic_and_signed_zero():
    data = pd.DataFrame(
        {
            "x": [1e200, 1e-200, -0.0, 2.0],
            "y": [1e200, 1e-200, 2.0, 3.0],
            "z": [1e-200, 1e200, 1.0, -4.0],
        }
    )
    with np.errstate(over="ignore", under="ignore"):
        matrix, names = design.build_fixed_matrix(parse_formula("response ~ x:y:z"), data)
        expected = (data["x"].to_numpy() * data["y"].to_numpy()) * data["z"].to_numpy()
    assert names == ["(Intercept)", "x:y:z"]
    np.testing.assert_array_equal(matrix[:, 1], expected)
    np.testing.assert_array_equal(np.signbit(matrix[:, 1]), np.signbit(expected))


@pytest.mark.parametrize(
    "term, expected_name",
    [("``:x", "x"), ("x:``:z", "x::z"), ("`:x`:z", ":x:z"), ("x:`z:`", "x:z:")],
)
def test_interactions_preserve_quoted_empty_and_colon_names(term, expected_name):
    data = pd.DataFrame(
        {"": [2.0, 3.0], "x": [4.0, 5.0], "z": [6.0, 7.0], ":x": [2.0, 3.0], "z:": [6.0, 7.0]}
    )
    _, names = design.build_fixed_matrix(parse_formula(f"y ~ {term}"), data)
    assert names == ["(Intercept)", expected_name]


@pytest.mark.parametrize("empty_factor", [None, "a", "b", "c", "d"])
def test_deep_interaction_products_and_empty_factor_positions(empty_factor):
    from itertools import product

    levels = np.array(list(product(range(3), repeat=4)))
    data = pd.DataFrame({name: pd.Categorical(levels[:, j]) for j, name in enumerate("abcd")})
    contrasts = {} if empty_factor is None else {empty_factor: np.empty((3, 0))}
    matrix, names = design.build_fixed_matrix(parse_formula("y ~ a:b:c:d"), data, contrasts)
    if empty_factor is not None:
        np.testing.assert_array_equal(matrix, np.ones((81, 1)))
        assert names == ["(Intercept)"]
        return
    expected = np.zeros((81, 17))
    expected[:, 0] = 1
    for j, combination in enumerate(product([1, 2], repeat=4), start=1):
        expected[:, j] = (levels == combination).all(axis=1)
    np.testing.assert_array_equal(matrix, expected)
    assert names[1] == "a.1:b.1:c.1:d.1"
    assert names[-1] == "a.2:b.2:c.2:d.2"
