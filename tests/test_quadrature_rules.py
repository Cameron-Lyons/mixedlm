from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm import GQN, GHrule, GQdk
from mixedlm.estimation.laplace import _get_gh_nodes_weights, adaptive_gh_deviance
from mixedlm.families import Gaussian
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices


def normalized_rule(kind, order):
    if kind == "matrix":
        rule = GHrule(order)
        return rule[:, 0], rule[:, 1]
    if kind == "dict":
        rule = GHrule(order, asMatrix=False)
        return rule["nodes"], rule["weights"]
    if kind == "tensor":
        nodes, weights = GQdk(1, order)
        return nodes[:, 0], weights
    return GQN(order)


KINDS = ["normal", "matrix", "dict", "tensor"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", [1, 3, 9, 25, 100, 200, 400, 1000])
def test_rules_integrate_normal_moments(kind, order):
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        nodes, weights = normalized_rule(kind, order)
    assert np.isfinite(nodes).all()
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (np.diff(nodes) > 0).all()
    np.testing.assert_allclose(nodes, -nodes[::-1], atol=1e-13, rtol=1e-13)
    for moment in range(min(order, 4)):
        expected = math.prod(range(1, 2 * moment, 2))
        assert weights @ nodes ** (2 * moment) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_public_arrays_are_writable_and_cannot_corrupt_later_calls(kind):
    expected = tuple(value.copy() for value in normalized_rule(kind, 17))
    nodes, weights = normalized_rule(kind, 17)
    nodes[:] = np.nan
    weights[:] = -1
    for actual, reference in zip(normalized_rule(kind, 17), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", [0, -1, True, False, 2.5, 3.0, np.nan, np.inf, "3", None])
def test_invalid_orders_are_rejected_before_cache_lookup(kind, order):
    with pytest.raises(ValueError, match="must be a positive integer"):
        normalized_rule(kind, order)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", [np.int32(7), np.int64(7)])
def test_numpy_integer_orders_are_supported(kind, order):
    nodes, weights = normalized_rule(kind, order)
    assert nodes.shape == weights.shape == (7,)


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5, 2.0, np.nan, "2", None])
def test_invalid_dimensions_are_rejected_before_rule_generation(dimension):
    from mixedlm.utils import quadrature

    with (
        patch.object(quadrature, "_compute_rule", side_effect=AssertionError("unexpected work")),
        pytest.raises(ValueError, match="d must be a positive integer"),
    ):
        GQdk(dimension, 3)


def test_tensor_rule_integrates_independent_moments():
    nodes, weights = GQdk(3, 5)
    assert nodes.shape == (125, 3)
    assert weights.sum() == pytest.approx(1.0)
    assert weights @ (nodes[:, 0] ** 2 * nodes[:, 1] ** 4) == pytest.approx(3.0)
    assert weights @ (nodes[:, 0] * nodes[:, 1]) == pytest.approx(0.0, abs=1e-14)


def test_public_helpers_and_fitting_share_one_rule_generation():
    from mixedlm.utils import quadrature

    quadrature._cached_rule.cache_clear()
    with patch.object(quadrature, "_compute_rule", wraps=quadrature._compute_rule) as generate:
        for kind in KINDS:
            normalized_rule(kind, 13)
        _get_gh_nodes_weights(13)
        assert generate.call_count == 1


def test_cached_rules_cannot_be_mutated_or_made_writable():
    for values in _get_gh_nodes_weights(13):
        with pytest.raises(ValueError):
            values[0] = 0
        with pytest.raises(ValueError):
            values.flags.writeable = True


def test_cache_bounds_entries_and_bypasses_large_orders():
    from mixedlm.utils import quadrature

    quadrature._cached_rule.cache_clear()
    for order in range(1, quadrature._CACHE_SIZE + 3):
        _get_gh_nodes_weights(order)
    before = quadrature._cached_rule.cache_info()
    assert before.currsize == quadrature._CACHE_SIZE
    nodes, weights = _get_gh_nodes_weights(quadrature._MAX_CACHED_ORDER + 1)
    assert len(nodes) == len(weights) == quadrature._MAX_CACHED_ORDER + 1
    assert quadrature._cached_rule.cache_info() == before


def test_concurrent_calls_return_independent_public_arrays():
    orders = [9, 25, 100, 400] * 8
    references = {n: GQN(n) for n in set(orders)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(GQN, orders))
    for order, (nodes, weights) in zip(orders, results, strict=True):
        np.testing.assert_array_equal(nodes, references[order][0])
        np.testing.assert_array_equal(weights, references[order][1])
        nodes[:] = np.nan
        weights[:] = np.nan


def test_high_order_adaptive_quadrature_matches_gaussian_marginal():
    rng = np.random.default_rng(94)
    data = pd.DataFrame({"y": rng.normal(size=12), "g": np.repeat(np.arange(3), 4)})
    matrices = build_model_matrices(parse_formula("y ~ 1 + (1 | g)"), data)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        deviance, beta, _ = adaptive_gh_deviance(np.array([0.7]), matrices, Gaussian(), nAGQ=400)
    covariance = np.eye(12) + 0.7**2 * (matrices.Z @ matrices.Z.T).toarray()
    residual = matrices.y - matrices.X @ beta
    expected = np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual)
    assert deviance == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("order", [0, 9, 25, 100, 200, 400])
def test_native_public_rules_do_not_share_mutable_results(order):
    native = pytest.importorskip("mixedlm._rust")
    expected = native.gauss_hermite(order)
    nodes, weights = native.gauss_hermite(order)
    nodes[:] = [np.nan] * order
    weights[:] = [-1] * order
    assert native.gauss_hermite(order) == expected
