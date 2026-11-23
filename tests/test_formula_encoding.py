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


@pytest.mark.parametrize("n", [0, 4])
def test_single_random_slope_with_missing_groups_and_zero_values(n):
    frame = pd.DataFrame({"g": ["b", None, "a", "b"], "x": [2.0, 4.0, 0.0, np.nan]}).head(n)
    matrix, _ = design.build_random_matrix(parse_formula("y ~ (0 + x | g)"), frame)
    expected = np.array([[0, 2], [0, 0], [0, 0], [0, np.nan]]) if n else np.empty((0, 0))
    np.testing.assert_array_equal(matrix.toarray(), expected)
    assert matrix.format == "csc" and matrix.has_canonical_format
    assert matrix.nnz == (2 if n else 0)


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
    # As in R, terms are ordered by degree and the first factor of an interaction
    # varies fastest; b has one column here, so a:b keeps the order of a.
    columns.append(x2)
    names.append("I(x**2)")
    for i, a_name in enumerate(a_names):
        for j, b_name in enumerate(b_names):
            columns.append(a[:, i] * b[:, j])
            names.append(f"{a_name}:{b_name}")
    for i, a_name in enumerate(a_names):
        columns.append(a[:, i] * x2)
        names.append(f"{a_name}:I(x**2)")
    # As in R, a uses indicators in a:b:I(x**2) because b:I(x**2) is absent,
    # while b keeps contrasts because a:I(x**2) is present.
    a_full = encode(a_values, LEVELS["a"], np.eye(4))
    for i, level in enumerate(LEVELS["a"]):
        for j, b_name in enumerate(b_names):
            columns.append(a_full[:, i] * b[:, j] * x2)
            names.append(f"a.{level}:{b_name}:I(x**2)")
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
    # As in R, terms are ordered by degree.
    expected = np.column_stack(
        [x, x + 0.5, x**2, (x + 0.5) ** 3, np.ones(n), x * (x + 0.5), x**2 * (x + 0.5) ** 3]
    )
    expected_names = ["x", "z", "I(x**2)", "I(z**3)", "I(x**0)", "x:z", "I(x**2):I(z**3)"]
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
    # As in R, a uses contrasts in a:x, whose x margin is present, and an
    # indicator per level in a:z, whose z margin is absent.
    data = pd.DataFrame({"a": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]})
    matrix, names = design.build_fixed_matrix(parse_formula("y ~ a:z + a + x + a:x"), data)
    assert names == [
        "(Intercept)",
        "a.1",
        "a.2",
        "x",
        "a.a:z",
        "a.b:z",
        "a.c:z",
        "a.1:x",
        "a.2:x",
    ]
    np.testing.assert_array_equal(
        matrix,
        [[1, 0, 0, 1, 4, 0, 0, 0, 0], [1, 1, 0, 2, 0, 5, 0, 2, 0], [1, 0, 1, 3, 0, 0, 6, 0, 3]],
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # Each factor uses contrasts when the term without it is in an earlier term.
        ("y ~ a + a:x", ["(Intercept)", "a.1", "a.2", "a.a:x", "a.b:x", "a.c:x"]),
        ("y ~ a/x", ["(Intercept)", "a.1", "a.2", "a.a:x", "a.b:x", "a.c:x"]),
        ("y ~ a*x", ["(Intercept)", "a.1", "a.2", "x", "a.1:x", "a.2:x"]),
        ("y ~ 0 + a + b", ["a.a", "a.b", "a.c", "b.1"]),
        ("y ~ 0 + x + a", ["x", "a.a", "a.b", "a.c"]),
        # As R's terms() does, the interaction lists its factors in order of first
        # appearance, b:a, and its first factor varies fastest.
        ("y ~ b + a:b", ["(Intercept)", "b.1", "b.u:a.1", "b.v:a.1", "b.u:a.2", "b.v:a.2"]),
        ("y ~ a + b + b:a", ["(Intercept)", "a.1", "a.2", "b.1", "a.1:b.1", "a.2:b.1"]),
    ],
)
def test_factor_coding_follows_r_marginality(formula, expected):
    data = pd.DataFrame(
        {"a": ["a", "b", "c", "a"], "b": ["u", "v", "u", "v"], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    _, names = design.build_fixed_matrix(parse_formula(formula), data)
    assert names == expected


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
    # Every margin is present, so every factor in a:b:c:d uses its contrasts.
    matrix, names = design.build_fixed_matrix(
        parse_formula("y ~ (a + b + c + d)^4"), data, contrasts
    )
    if empty_factor is not None:
        # Terms with the empty factor have no columns: the rest is (3 factors)^3.
        assert matrix.shape == (81, 27)
        assert all(f"{empty_factor}." not in name for name in names)
        return
    assert matrix.shape == (81, 81)
    expected = np.zeros((81, 16))
    # As in R, the first factor varies fastest within the interaction.
    for j, combination in enumerate(product([1, 2], repeat=4)):
        expected[:, j] = (levels == combination[::-1]).all(axis=1)
    np.testing.assert_array_equal(matrix[:, -16:], expected)
    assert names[-16] == "a.1:b.1:c.1:d.1"
    assert names[-1] == "a.2:b.2:c.2:d.2"


def _crossed_factors():
    rng = np.random.default_rng(3)
    n = 600
    g = rng.integers(0, 30, n)
    f = rng.choice(["A", "B", "C"], n)
    k = rng.choice(["u", "v", "w"], n)
    h = rng.choice(["p", "q"], n)
    effects = rng.normal(size=(30, 2))
    y = (
        1
        + pd.Series(f).map({"A": 0.0, "B": 0.5, "C": -0.3}).to_numpy()
        + (k == "v") * 0.4
        + effects[g, 0]
        + effects[g, 1] * (h == "q")
        + rng.normal(size=n)
    )
    return pd.DataFrame({"y": y, "f": f, "k": k, "h": h, "g": [f"g{i:02d}" for i in g]})


def test_factor_interactions_vary_the_first_factor_fastest_as_in_lme4():
    import warnings

    from mixedlm import lmer
    from mixedlm.inference.variance_profile import sdcor_from_theta

    data = _crossed_factors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = lmer("y ~ 1 + (0 + h:f | g)", data)
    names = ["h.p:f.A", "h.q:f.A", "h.p:f.B", "h.q:f.B", "h.p:f.C", "h.q:f.C"]
    assert fit.matrices.random_structures[0].term_names == names
    # lme4 2.1.0: .sig02 is sd_hq:fA|g, estimated as 1.27284.
    assert fit.variance_parameters()[1].label == "sd_h.q:f.A|g"
    sds = sdcor_from_theta(fit.theta, fit.sigma, fit.matrices.random_structures)
    assert sds[1] == pytest.approx(1.27284, abs=1e-4)
    interactions = list(lmer("y ~ f*k + (1 | g)", data).fixef().items())[5:]
    # lme4 2.1.0: fixef(lmer(y ~ f*k + (1 | g), d))[6:9] is fB:kv, fC:kv, fB:kw, fC:kw
    assert [name for name, _ in interactions] == ["f.1:k.1", "f.2:k.1", "f.1:k.2", "f.2:k.2"]
    np.testing.assert_allclose(
        [value for _, value in interactions], [-0.27488, 0.11560, 0.31185, 0.29198], atol=1e-5
    )


def test_models_saved_before_r_coding_predict_with_their_own_columns():
    from dataclasses import replace

    from mixedlm import lmer

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "a": rng.choice(["1", "2", "3"], 120),
            "x": rng.normal(size=120),
            "g": np.repeat([f"g{i}" for i in range(12)], 10),
        }
    )
    data["y"] = rng.normal(size=120)
    fit = lmer("y ~ a + a:x + (1 | g)", data)
    assert fit.matrices.factor_coding == "R"
    assert fit.matrices.fixed_names == ["(Intercept)", "a.1", "a.2", "a.1:x", "a.2:x", "a.3:x"]
    X, names = design.build_fixed_matrix(fit.formula, data, coding="legacy")
    assert names == ["(Intercept)", "a.1", "a.2", "a.1:x", "a.2:x"]
    legacy = replace(fit.matrices, X=X, fixed_names=names)
    del legacy.__dict__["factor_coding"]  # as restored from a pickle that predates the field
    assert legacy.factor_coding == "legacy"
    fit.matrices = legacy
    fit.beta = np.linspace(-1.0, 1.0, len(names))
    np.testing.assert_allclose(fit.predict(data, re_form="NA"), X @ fit.beta)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # R 4.6.2 model.matrix() on the same data, with R's names a2 -> a.1 and so on.
        ("y ~ a + b + b:a", ["(Intercept)", "a.1", "a.2", "b.1", "b.2", "a.1:b.1", "a.2:b.1"]),
        ("y ~ b + a:b", ["(Intercept)", "b.1", "b.2", "b.u:a.1", "b.v:a.1", "b.w:a.1"]),
        ("y ~ a + x:a", ["(Intercept)", "a.1", "a.2", "a.a:x", "a.b:x", "a.c:x"]),
        ("y ~ x:a + a", ["(Intercept)", "a.1", "a.2", "x:a.a", "x:a.b", "x:a.c"]),
        ("y ~ b*a + x:b:a", None),
    ],
)
def test_interactions_order_factors_by_first_appearance_as_in_r(formula, expected):
    data = pd.DataFrame(
        {
            "a": ["a", "b", "c", "a", "b", "c"],
            "b": ["u", "v", "w", "w", "u", "v"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    if expected is None:
        # R labels the three-way term b:a:x, from b*a before x:b:a.
        parsed = parse_formula(formula)
        assert str(parsed) == "y ~ b + a + b:a + b:a:x"
        return
    _, names = design.build_fixed_matrix(parse_formula(formula), data)
    assert names[: len(expected)] == expected


def test_random_interactions_and_updates_order_factors_as_in_r():
    from mixedlm.formula.parser import update_formula

    # lme4 builds each bar from its own formula: cnms of (0 + a + b + b:a | g).
    parsed = parse_formula("y ~ b:a + (0 + a + b + b:a | g)")
    assert str(parsed) == "y ~ b:a + (0 + a + b + a:b | g)"
    # As R's update(), '.' brings the old variables first.
    assert str(update_formula(parse_formula("y ~ a + b"), ". ~ . + b:a")) == "y ~ a + b + a:b"
    assert str(update_formula(parse_formula("y ~ x"), ". ~ . + b:a")) == "y ~ x + b:a"


def _legacy_data():
    rng = np.random.default_rng(0)
    n = 216
    g = np.repeat([f"g{i:02d}" for i in range(12)], 18)
    a = np.tile(["A1", "A2", "A3"], n // 3)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    groups = pd.factorize(g)[0]
    slope = np.select([a == "A1", a == "A2"], [0.3, -0.8], 1.2)
    y = (
        1
        + (a == "A2")
        + slope * x
        + 0.4 * z
        + rng.normal(size=12)[groups]
        + 0.5 * rng.normal(size=12)[groups] * x
        + rng.normal(scale=0.7, size=n)
    )
    return pd.DataFrame({"y": y, "g": g, "a": pd.Categorical(a), "x": x, "z": z})


def _saved_model(formula, data):
    """A model as mixedlm 1.3 saved it: legacy coding, and no stored coding field."""
    import warnings

    from mixedlm import lmer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lmer(formula, data, REML=False, _factor_coding="legacy")
    del model.matrices.__dict__["factor_coding"]
    return model


def test_refits_of_saved_models_keep_their_coding():
    import warnings

    from mixedlm import anova_type3, cross_validate, powerCurve

    data = _legacy_data()
    model = _saved_model("y ~ x:a + a + z + (1 + x | g)", data)
    assert model.matrices.factor_coding == "legacy"
    # Legacy coding gives x:a contrasts, so 6 columns where R coding gives 7.
    assert model.matrices.n_fixed == 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        updated = model.update()
        assert updated.matrices.fixed_names == model.matrices.fixed_names
        assert updated.deviance == pytest.approx(model.deviance, rel=1e-8)
        deletions = model.drop1(data)
        assert all(lrt is not None and lrt > 0 for lrt in deletions.lrt)
        allfit = model.allFit(data, optimizers=["L-BFGS-B"])
        assert allfit.fits["L-BFGS-B"].deviance == pytest.approx(model.deviance, rel=1e-6)
        cross_validate(model, cv=3, random_state=1)
    # Fits saved by mixedlm 1.3 lack term labels, which anova_type3 then replays.
    model.matrices.fixed_term_labels = None
    assert len(anova_type3(model).terms) == 3

    sloped = _saved_model("y ~ a + x + (1 + a:x | g)", data)
    assert sloped.matrices.random_structures[0].term_names == ["(Intercept)", "a.1:x", "a.2:x"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        powerCurve(sloped, test="x", along="n_groups", values=[8], nsim=2, seed=1)


def test_updates_remove_bars_saved_with_slopes_in_written_order():
    from dataclasses import replace

    from mixedlm.formula.parser import update_formula

    formula = parse_formula("y ~ x * z + (1 + x:z + x | g)")
    bar = formula.random[0]
    # mixedlm 1.3 stored the slopes as written, not by degree.
    saved = replace(formula, random=(replace(bar, expr=tuple(reversed(bar.expr))),))
    updated = update_formula(saved, ". ~ . - (1 + x:z + x | g) + (1 | g)")
    assert str(updated) == "y ~ x + z + x:z + (1 | g)"
    assert str(update_formula(saved, ". ~ . + (1 + x:z + x | g)")) == str(saved)
