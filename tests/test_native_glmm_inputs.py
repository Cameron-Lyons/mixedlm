"""Native GLMM entry points reject inconsistent dimensions before fitting."""

import sys

import numpy as np
import pytest
from mixedlm.estimation.laplace import _native_glmm_args
from numpy.testing import assert_array_equal

from tests.test_glmm_final_state import mode_problem

native = pytest.importorskip("mixedlm._rust")
FUNCTIONS = ["glmm_deviance", "prepared"]
KEYS = (
    "y",
    "x",
    "z_data",
    "z_indices",
    "z_indptr",
    "z_shape",
    "weights",
    "offset",
    "theta",
    "n_levels",
    "n_terms",
    "correlated",
    "family",
    "link",
)


def arguments(kind="gaussian", layout="intercept"):
    matrices, family, theta = mode_problem(kind, layout)
    return dict(zip(KEYS, _native_glmm_args(theta, matrices, family), strict=True))


def evaluate(function, args, order=1):
    if function == "prepared":
        problem = native.GlmmProblem(
            **{key: value for key, value in args.items() if key != "theta"}
        )
        return problem.evaluate(args["theta"], order)
    return native.glmm_deviance(**args, n_agq=order)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("field", ["x", "weights", "offset", "theta"])
@pytest.mark.parametrize("change", [-1, 1])
def test_short_and_extra_arrays_are_rejected(function, field, change):
    args = arguments()
    value = args[field]
    args[field] = value[:-1] if change == -1 else np.concatenate([value, value[:1]])
    with pytest.raises(ValueError, match=rf"{field} must have"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("change", [-1, 1])
def test_random_design_rows_must_match_the_response(function, change):
    args = arguments()
    n, q = args["z_shape"]
    args["z_shape"] = (n + change, q)
    with pytest.raises(ValueError, match="z must have .* rows to match y"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("field", ["n_levels", "n_terms", "correlated"])
@pytest.mark.parametrize("extra", [False, True])
def test_structure_arrays_cannot_be_silently_truncated(function, field, extra):
    args = arguments()
    args[field] = args[field] * 2 if extra else []
    with pytest.raises(ValueError, match="must have equal lengths"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("field", ["n_levels", "n_terms"])
def test_empty_structure_dimensions_are_rejected(function, field):
    args = arguments()
    args[field] = [0]
    with pytest.raises(ValueError, match="positive dimensions"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("change", [-1, 1])
def test_structure_columns_must_match_the_random_design(function, change):
    args = arguments()
    args["n_levels"] = [args["n_levels"][0] + change]
    with pytest.raises(ValueError, match="structures describe .* columns, but z has"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("change", [-1, 1])
def test_covariance_parameter_count_uses_each_structure(function, correlated, change):
    args = arguments(layout="slope")
    args["correlated"] = [correlated]
    expected = 3 if correlated else 2
    args["theta"] = np.full(expected + change, 0.3)
    with pytest.raises(ValueError, match=rf"theta must have length {expected}, got"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_no_observations_are_rejected(function):
    args = arguments(layout="fixed_only")
    args.update(
        y=np.array([]),
        x=np.empty((0, 2)),
        weights=np.array([]),
        offset=np.array([]),
        z_shape=(0, 0),
    )
    with pytest.raises(ValueError, match="at least one observation"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("overflow", ["columns", "column_sum", "parameters", "indptr"])
def test_dimension_arithmetic_cannot_wrap_or_allocate_huge_matrices(function, overflow):
    args = arguments()
    maximum = 2 * sys.maxsize + 1
    if overflow == "columns":
        args.update(n_levels=[maximum], n_terms=[2])
    elif overflow == "column_sum":
        args.update(n_levels=[maximum, 1], n_terms=[1, 1], correlated=[False, False])
    elif overflow == "parameters":
        args.update(n_levels=[1], n_terms=[maximum])
    else:
        args.update(n_levels=[maximum], z_shape=(len(args["y"]), maximum))
    with pytest.raises(ValueError, match="overflow"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "error",
    [
        "data_length",
        "negative_row",
        "row_past_end",
        "indptr_length",
        "indptr_start",
        "indptr_end",
        "indptr_order",
    ],
)
def test_sparse_format_errors_remain_ordinary_exceptions(function, error):
    args = arguments()
    if error == "data_length":
        args["z_data"] = args["z_data"][:-1]
    elif error in {"negative_row", "row_past_end"}:
        args["z_indices"][0] = -1 if error == "negative_row" else len(args["y"])
    elif error == "indptr_length":
        args["z_indptr"] = args["z_indptr"][:-1]
    elif error == "indptr_start":
        args["z_indptr"][0] = 1
    elif error == "indptr_end":
        args["z_indptr"][-1] -= 1
    else:
        args["z_indptr"][1:3] = [24, 12]
    with pytest.raises(ValueError, match="Invalid sparse matrix format"):
        evaluate(function, args)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("layout", ["intercept", "slope", "crossed", "fixed_only", "mode_only"])
def test_lists_and_zero_covariance_keep_valid_layouts(function, layout):
    args = arguments(layout=layout)
    args["theta"][:] = 0
    expected = evaluate(function, args)
    listed = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in args.items()
    }
    actual = evaluate(function, listed)
    for value, reference in zip(actual, expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("field", ["y", "x", "offset"])
def test_strided_views_keep_their_values_and_order(function, field):
    args = arguments()
    expected = evaluate(function, args)
    value = args[field]
    backing = np.zeros((len(value) * 2, *value.shape[1:]))
    backing[::2] = value
    args[field] = backing[::2]
    assert not args[field].flags.c_contiguous
    actual = evaluate(function, args)
    for value, reference in zip(actual, expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_fortran_fixed_design_is_preserved(function):
    args = arguments()
    expected = evaluate(function, args)
    args["x"] = np.asfortranarray(args["x"])
    for value, reference in zip(evaluate(function, args), expected, strict=True):
        assert_array_equal(value, reference)


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson"])
def test_prepared_and_one_shot_entry_points_agree_at_higher_order(kind):
    args = arguments(kind=kind)
    actual = evaluate("prepared", args, order=7)
    expected = evaluate("glmm_deviance", args, order=7)
    for value, reference in zip(actual, expected, strict=True):
        assert_array_equal(value, reference)
    assert actual[3]
