"""Prepared designs preserve the likelihood and isolate repeated responses."""

import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from mixedlm import _rust, lmer
from mixedlm.estimation.reml import LMMOptimizer, _LMMCrossproducts, _RustMatrixCache
from mixedlm.formula.parser import parse_formula, set_cov_type
from mixedlm.inference import bootstrap
from mixedlm.matrices.design import build_model_matrices
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, sparse

from tests.test_lmm_likelihood_profiles import data_fixture


def matrices_fixture(kind="intercept"):
    data, weights, offset = data_fixture()
    data["h"] = np.arange(len(data)) % 5
    formula = {
        "fixed": "y ~ x + z",
        "no_fixed": "y ~ 0 + (1 | g)",
        "intercept": "y ~ x + z + (1 | g)",
        "correlated": "y ~ x + z + (x | g)",
        "slope": "y ~ x + z + (0 + x | g)",
        "crossed": "y ~ x + z + (x | g) + (1 | h)",
        "cs": "y ~ x + z + (x + z | g)",
        "ar1": "y ~ x + z + (x + z | g)",
    }[kind]
    parsed = set_cov_type(formula, kind) if kind in {"cs", "ar1"} else parse_formula(formula)
    return build_model_matrices(parsed, data, weights=weights, offset=offset)


def native_arguments(matrices):
    z = matrices.Z.tocsc()
    return dict(
        x=matrices.X.copy(),
        z_data=z.data.copy(),
        z_indices=z.indices.astype(np.int64),
        z_indptr=z.indptr.astype(np.int64),
        z_shape=z.shape,
        weights=matrices.weights.copy(),
        offset=matrices.offset.copy(),
        n_levels=[s.n_levels for s in matrices.random_structures],
        n_terms=[s.n_terms for s in matrices.random_structures],
        correlated=[s.correlated for s in matrices.random_structures],
    )


def parameters(matrices):
    values = []
    for structure in matrices.random_structures:
        if structure.cov_type in {"cs", "ar1"}:
            values.extend([0.8, 0.2])
        elif structure.correlated:
            values.extend(
                0.8 if i == j else 0.1 for i in range(structure.n_terms) for j in range(i + 1)
            )
        else:
            values.extend([0.8] * structure.n_terms)
    return np.array(values)


def observation_likelihood(matrices, theta, reml):
    """Dense observation-space oracle, independent of the profiled solver."""
    covariance = np.diag(1 / matrices.weights)
    position = 0
    blocks = []
    for structure in matrices.random_structures:
        width = structure.n_terms
        if structure.cov_type in {"cs", "ar1"}:
            scale, rho = theta[position : position + 2]
            position += 2
            correlation = (
                (1 - rho) * np.eye(width) + rho
                if structure.cov_type == "cs"
                else rho ** np.abs(np.arange(width)[:, None] - np.arange(width))
            )
            block = scale**2 * correlation
        else:
            count = width * (width + 1) // 2 if structure.correlated else width
            factor = np.zeros((width, width))
            if structure.correlated:
                factor[np.tril_indices(width)] = theta[position : position + count]
            else:
                np.fill_diagonal(factor, theta[position : position + count])
            position += count
            block = factor @ factor.T
        blocks.extend([block] * structure.n_levels)
    if blocks:
        z = matrices.Z.toarray()
        covariance += z @ linalg.block_diag(*blocks) @ z.T
    inverse_x = np.linalg.solve(covariance, matrices.X)
    information = matrices.X.T @ inverse_x
    y = matrices.y - matrices.offset
    beta = np.linalg.solve(information, inverse_x.T @ y)
    residual = y - matrices.X @ beta
    rss = residual @ np.linalg.solve(covariance, residual)
    df = matrices.n_obs - matrices.n_fixed if reml else matrices.n_obs
    return (
        df * (1 + np.log(2 * np.pi * rss / df))
        + np.linalg.slogdet(covariance)[1]
        + (np.linalg.slogdet(information)[1] if reml else 0)
    )


@pytest.mark.parametrize(
    "kind", ["fixed", "no_fixed", "intercept", "correlated", "slope", "crossed"]
)
@pytest.mark.parametrize("reml", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
def test_native_prepared_likelihood_matches_observation_covariance(kind, reml, boundary):
    matrices = matrices_fixture(kind)
    theta = parameters(matrices)
    if boundary:
        theta[:] = 0
    design = _rust.LmmDesign(**native_arguments(matrices))
    for y in (matrices.y, matrices.y[::-1] + 0.3 * matrices.weights):
        response = design.with_response(y)
        expected = observation_likelihood(replace(matrices, y=y), theta, reml)
        assert_allclose(response.deviance(theta, reml), expected, rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize(
    "native,kind",
    [
        (True, "crossed"),
        (False, "crossed"),
        (True, "fixed"),
        (False, "fixed"),
        (True, "cs"),
        (True, "ar1"),
    ],
)
@pytest.mark.parametrize("reml", [False, True])
def test_response_refits_share_design_and_match_independent_fits(native, kind, reml):
    matrices = matrices_fixture(kind)
    original = LMMOptimizer(matrices, REML=reml, use_rust=native)
    theta = parameters(matrices)
    original_value = original.objective(theta)
    y = matrices.y[::-1].copy()
    fitted = original.with_response(y)
    y[:] = -100  # The new optimizer owns its response.
    fresh = LMMOptimizer(replace(matrices, y=matrices.y[::-1].copy()), REML=reml, use_rust=native)
    assert fitted._crossproducts.XtWX is original._crossproducts.XtWX
    assert fitted._crossproducts.ZtWZ is original._crossproducts.ZtWZ
    assert fitted._crossproducts.ZtWX is original._crossproducts.ZtWX
    assert fitted.matrices.Zt is original.matrices.Zt
    if original.use_rust:
        assert fitted._rust_cache.design is original._rust_cache.design
        assert fitted._rust_cache.response is not original._rust_cache.response
    assert_array_equal(fitted.objective(theta), fresh.objective(theta))
    assert_allclose(
        fitted.objective(theta), observation_likelihood(fitted.matrices, theta, reml), rtol=2e-13
    )
    actual, expected = fitted.optimize(start=theta), fresh.optimize(start=theta)
    for field in ("theta", "beta", "sigma", "u", "deviance", "converged", "n_iter"):
        assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert original.objective(theta) == original_value
    assert_array_equal(original.matrices.y, matrices.y)


def test_native_snapshot_owns_inputs_and_response_outlives_design():
    matrices = matrices_fixture("correlated")
    arguments = native_arguments(matrices)
    design = _rust.LmmDesign(**arguments)
    y = matrices.y.copy()
    first = design.with_response(y)
    second = design.with_response(y[::-1])
    theta = parameters(matrices)
    expected = first.deviance(theta)
    other = second.deviance(theta)
    for value in arguments.values():
        if isinstance(value, np.ndarray):
            value[:] = 0
    y[:] = 1000
    del design
    gc.collect()
    assert first.deviance(theta) == expected
    assert second.deviance(theta) == other
    assert first.deviance(theta) == expected


def test_prepared_design_canonicalizes_duplicate_unsorted_sparse_entries():
    matrices = matrices_fixture()
    z = matrices.Z
    data, indices, indptr = [], [], [0]
    for column in range(z.shape[1]):
        start, stop = z.indptr[column : column + 2]
        for row, value in zip(z.indices[start:stop][::-1], z.data[start:stop][::-1], strict=True):
            indices.extend([row, row])
            data.extend([0.25 * value, 0.75 * value])
        indptr.append(len(data))
    duplicate = sparse.csc_matrix((data, indices, indptr), shape=z.shape)
    assert not duplicate.has_canonical_format
    cache = _RustMatrixCache.from_matrices(replace(matrices, Z=duplicate))
    theta = parameters(matrices)
    assert_allclose(
        cache.response.deviance(theta), observation_likelihood(matrices, theta, True), rtol=2e-13
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", np.zeros((3, 2))),
        ("x", np.full((38, 2), np.nan)),
        ("weights", np.ones(2)),
        ("weights", np.zeros(38)),
        ("weights", np.full(38, np.inf)),
        ("offset", np.zeros(2)),
        ("offset", np.full(38, np.nan)),
        ("z_data", np.array([np.inf])),
        ("z_indices", np.array([-1], dtype=np.int64)),
        ("z_indptr", np.array([0, 100], dtype=np.int64)),
        ("z_shape", (1, 7)),
        ("n_levels", []),
        ("n_levels", [0]),
        ("n_levels", [8]),
        ("n_levels", [2**63]),
        ("n_terms", [2**63]),
        ("n_terms", [0]),
        ("correlated", []),
    ],
)
def test_invalid_native_design_is_rejected(field, value):
    arguments = native_arguments(matrices_fixture())
    arguments[field] = value
    with pytest.raises(ValueError):
        _rust.LmmDesign(**arguments)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "y", [np.zeros(2), np.zeros((38, 1)), np.full(38, np.nan), np.full(38, np.inf), np.full(38, 1j)]
)
def test_optimizer_rejects_invalid_response(native, y):
    optimizer = LMMOptimizer(matrices_fixture(), use_rust=native)
    with pytest.raises(ValueError, match="response"):
        optimizer.with_response(y)


@pytest.mark.parametrize("values", [np.zeros(2), np.full(38, np.nan), np.full(38, np.inf)])
def test_native_rejects_invalid_response(values):
    design = _rust.LmmDesign(**native_arguments(matrices_fixture()))
    with pytest.raises(ValueError, match="response"):
        design.with_response(values)


@pytest.mark.parametrize(
    "theta", [np.array([]), np.ones(2), np.array([np.nan]), np.array([np.inf])]
)
def test_native_rejects_invalid_parameters(theta):
    matrices = matrices_fixture()
    response = _RustMatrixCache.from_matrices(matrices).response
    with pytest.raises(ValueError, match="theta"):
        response.deviance(theta)


def test_reml_rejects_nonpositive_residual_degrees_of_freedom():
    matrices = matrices_fixture("fixed")
    matrices = replace(matrices, X=np.eye(matrices.n_obs))
    response = _RustMatrixCache.from_matrices(matrices).response
    with pytest.raises(ValueError, match="REML"):
        response.deviance(np.array([]))


def fitted_model():
    data, weights, offset = data_fixture()
    return lmer("y ~ x + z + (1 | g)", data, weights=weights, offset=offset)


def test_serial_bootstrap_prepares_design_once_and_matches_fresh_refits():
    result = fitted_model()
    with (
        patch.object(
            _RustMatrixCache, "from_matrices", wraps=_RustMatrixCache.from_matrices
        ) as native,
        patch.object(
            _LMMCrossproducts, "from_matrices", wraps=_LMMCrossproducts.from_matrices
        ) as python,
    ):
        actual = bootstrap.bootstrap_lmer(result, n_boot=4, seed=56)
    assert native.call_count == python.call_count == 1
    refit = bootstrap._refit_lmer_response

    def independent(matrices, response, theta, reml, **kwargs):
        return refit(matrices, response, theta, reml)

    with patch.object(bootstrap, "_refit_lmer_response", side_effect=independent):
        expected = bootstrap.bootstrap_lmer(result, n_boot=4, seed=56)
    assert actual.n_failed == expected.n_failed == 0
    for name in ("beta_samples", "theta_samples", "sigma_samples"):
        assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_spawn_bootstrap_creates_native_state_in_workers():
    result = fitted_model()
    serial = bootstrap.bootstrap_lmer(result, n_boot=4, seed=56)
    with patch.object(
        bootstrap,
        "ProcessPoolExecutor",
        partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    ):
        parallel = bootstrap.bootstrap_lmer(result, n_boot=4, seed=56, n_jobs=2)
    assert serial.n_failed == parallel.n_failed == 0
    for name in ("beta_samples", "theta_samples", "sigma_samples"):
        assert_array_equal(getattr(serial, name), getattr(parallel, name))


@pytest.mark.parametrize("crossed", [False, True])
def test_large_python_refits_keep_sparse_design_products(crossed):
    import pandas as pd

    rng = np.random.default_rng(821)
    n = 1028
    data = pd.DataFrame(
        dict(y=rng.normal(size=n), x=rng.normal(size=n), g=np.arange(n) % 257, h=np.arange(n) % 7)
    )
    formula = "y ~ x + (1 | g)" + (" + (1 | h)" if crossed else "")
    matrices = build_model_matrices(parse_formula(formula), data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    response = matrices.y[::-1].copy()
    repeated = optimizer.with_response(response)
    fresh = LMMOptimizer(replace(matrices, y=response), use_rust=False)
    assert sparse.issparse(repeated._crossproducts.ZtWZ)
    assert repeated._crossproducts.ZtWZ is optimizer._crossproducts.ZtWZ
    assert_array_equal(
        repeated.objective(parameters(matrices)), fresh.objective(parameters(matrices))
    )


def test_worker_prepares_design_once_for_multiple_responses():
    result = fitted_model()
    data = tuple(bootstrap._prepare_lmer_worker_data(result).values())
    with (
        patch.object(
            _RustMatrixCache, "from_matrices", wraps=_RustMatrixCache.from_matrices
        ) as native,
        patch.object(
            _LMMCrossproducts, "from_matrices", wraps=_LMMCrossproducts.from_matrices
        ) as python,
    ):
        bootstrap._initialize_bootstrap_worker(bootstrap._lmer_bootstrap_worker, data)
        samples = [bootstrap._run_bootstrap_task((i, 23 + i)) for i in range(3)]
    assert native.call_count == python.call_count == 1
    assert all(sample[1] is not None for sample in samples)


@pytest.mark.parametrize("field", ["x", "z_data", "offset", "weights"])
def test_native_rejects_nonfinite_values_with_valid_shapes(field):
    arguments = native_arguments(matrices_fixture())
    arguments[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _rust.LmmDesign(**arguments)


@pytest.mark.parametrize("reml", [False, True])
def test_uncorrelated_slopes_match_observation_covariance(reml):
    matrices = matrices_fixture("correlated")
    matrices = replace(
        matrices,
        random_structures=[replace(s, correlated=False) for s in matrices.random_structures],
    )
    theta = parameters(matrices)
    base = LMMOptimizer(matrices, use_rust=True, REML=reml)
    repeated = base.with_response(matrices.y[::-1])
    assert_allclose(
        repeated.objective(theta),
        observation_likelihood(repeated.matrices, theta, reml),
        rtol=2e-13,
    )
