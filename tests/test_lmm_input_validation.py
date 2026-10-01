"""Raw LMM calls share checked preparation and preserve valid gradients."""

from dataclasses import replace

import numpy as np
import pytest
from mixedlm import _rust
from numpy.testing import assert_allclose, assert_array_equal

from tests.test_lmm_prepared_design import (
    matrices_fixture,
    native_arguments,
    observation_likelihood,
    parameters,
)

APIS = ["profiled_deviance", "profiled_deviance_cached", "profiled_deviance_with_gradient"]


def likelihood_arguments(kind="correlated"):
    matrices = matrices_fixture(kind)
    arguments = native_arguments(matrices)
    arguments.update(y=matrices.y.copy(), theta=parameters(matrices))
    return arguments


def evaluate(api, arguments):
    if api == "profiled_deviance_cached":
        # Use supplied products so this entry point exercises its cache branch.
        matrices = matrices_fixture("fixed" if arguments["z_shape"][1] == 0 else "correlated")
        cache = (matrices.Z.T @ matrices.Z.multiply(matrices.weights[:, None])).toarray().ravel()
        return getattr(_rust, api)(**arguments, ztwz_cache=cache)
    return getattr(_rust, api)(**arguments)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "field,change,message",
    [
        ("theta", lambda a: a[:-1], "theta"),
        ("theta", lambda a: np.append(a, 1.0), "theta"),
        ("y", lambda a: a[:-1], "response"),
        ("y", lambda a: np.append(a, 1.0), "response"),
        ("x", lambda a: a[:-1], "row count"),
        ("x", lambda a: np.vstack((a, a[:1])), "row count"),
        ("weights", lambda a: a[:-1], "row count"),
        ("offset", lambda a: a[:-1], "row count"),
        ("offset", lambda a: np.append(a, 0.0), "row count"),
        ("weights", lambda a: np.zeros_like(a), "strictly positive"),
        ("weights", lambda a: -a, "strictly positive"),
        ("z_shape", lambda a: (a[0] + 1, a[1]), "row count"),
        ("n_levels", lambda a: [], "equal lengths"),
        ("n_levels", lambda a: a + [1], "equal lengths"),
        ("n_terms", lambda a: [], "equal lengths"),
        ("n_terms", lambda a: a + [1], "equal lengths"),
        ("correlated", lambda a: [], "equal lengths"),
        ("correlated", lambda a: a + [True], "equal lengths"),
        ("n_levels", lambda a: [0], "positive dimensions"),
        ("n_terms", lambda a: [0], "positive dimensions"),
        ("n_levels", lambda a: [a[0] + 1], "column count"),
        ("z_indices", lambda a: np.full_like(a, -1), "indices"),
        ("z_indptr", lambda a: a[:-1], "indptr"),
    ],
)
def test_raw_calls_reject_malformed_inputs(api, field, change, message):
    arguments = likelihood_arguments()
    arguments[field] = change(arguments[field])
    with pytest.raises(ValueError, match=message):
        evaluate(api, arguments)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("field", ["theta", "y", "x", "z_data", "weights", "offset"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_raw_calls_reject_nonfinite_values_with_valid_shapes(api, field, value):
    arguments = likelihood_arguments()
    arguments[field].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        evaluate(api, arguments)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("kind", ["fixed", "correlated"])
@pytest.mark.parametrize("extra_columns", [0, 1])
def test_raw_reml_calls_reject_nonpositive_residual_degrees_of_freedom(api, kind, extra_columns):
    arguments = likelihood_arguments(kind)
    n = len(arguments["y"])
    arguments["x"] = np.eye(n, n + extra_columns)
    with pytest.raises(ValueError, match="REML requires more observations"):
        evaluate(api, arguments)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("malformation", ["response", "theta", "sparse", "empty"])
def test_fixed_effects_only_calls_validate_inputs_before_solving(api, malformation):
    arguments = likelihood_arguments("fixed")
    if malformation == "response":
        arguments["y"][0] = np.nan
    elif malformation == "theta":
        arguments["theta"] = np.ones(1)
    elif malformation == "sparse":
        arguments["z_indptr"] = np.array([1], dtype=np.int64)
    else:
        for field in ["x", "y", "weights", "offset"]:
            arguments[field] = arguments[field][:0]
        arguments["z_shape"] = (0, 0)
    with pytest.raises(ValueError):
        evaluate(api, arguments)


@pytest.mark.parametrize("field", ["n_levels", "n_terms"])
def test_gradient_rejects_overflowing_structure_dimensions(field):
    arguments = likelihood_arguments()
    arguments[field] = [int(np.iinfo(np.uintp).max)]
    with pytest.raises(ValueError, match="overflow"):
        _rust.profiled_deviance_with_gradient(**arguments)


@pytest.mark.parametrize("kind", ["fixed", "no_fixed", "crossed"])
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("layout", ["fortran", "reversed"])
def test_gradient_preserves_strided_inputs_and_matches_observation_likelihood(kind, reml, layout):
    matrices = matrices_fixture(kind)
    if layout == "fortran":
        matrices = replace(matrices, X=np.asfortranarray(matrices.X))
    else:
        matrices = replace(
            matrices,
            y=matrices.y[::-1],
            X=matrices.X[::-1],
            Z=matrices.Z[::-1].tocsc(),
            weights=matrices.weights[::-1],
            offset=matrices.offset[::-1],
        )
    arguments = native_arguments(matrices)
    # Keep the noncontiguous arrays instead of the fixture's owned copies.
    arguments.update(x=matrices.X, weights=matrices.weights, offset=matrices.offset)
    theta = parameters(matrices)
    value, gradient = _rust.profiled_deviance_with_gradient(
        theta=theta, y=matrices.y, reml=reml, **arguments
    )
    assert_allclose(value, observation_likelihood(matrices, theta, reml), rtol=2e-13, atol=2e-12)
    assert_array_equal(
        value, _rust.LmmDesign(**arguments).with_response(matrices.y).deviance(theta, reml)
    )
    expected = []
    for index in range(len(theta)):
        step = np.zeros_like(theta)
        step[index] = 1e-5
        expected.append(
            (
                observation_likelihood(matrices, theta + step, reml)
                - observation_likelihood(matrices, theta - step, reml)
            )
            / 2e-5
        )
    assert_allclose(gradient, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("kind", ["fixed", "correlated"])
@pytest.mark.parametrize("reml", [False, True])
def test_gradient_preserves_factorization_failure_result(kind, reml):
    arguments = likelihood_arguments(kind)
    arguments["x"][:] = 0
    value, gradient = _rust.profiled_deviance_with_gradient(**arguments, reml=reml)
    assert value == 1e10
    assert_array_equal(gradient, np.zeros_like(arguments["theta"]))
