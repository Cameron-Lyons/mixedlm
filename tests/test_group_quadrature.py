from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from mixedlm.estimation import laplace
from mixedlm.families import Binomial, Gaussian, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from scipy import integrate, sparse, stats


def make_matrices(kind="gaussian", pattern="mixed", n_groups=4, n_per_group=6):
    rng = np.random.default_rng(47)
    group = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.uniform(-1, 1, len(group))
    z = rng.uniform(0.5, 1.5, len(group))
    if pattern == "mixed":
        z[::3] = 0
    elif pattern == "all_zero":
        z[:] = 0
    elif pattern == "empty_group":
        z[group == n_groups - 1] = 0
    offset = np.linspace(-0.1, 0.3, len(group))
    eta = 0.4 + 0.2 * x + rng.normal(0, 0.3, n_groups)[group] * z + offset
    if kind == "gaussian":
        y, family = eta + rng.normal(0, 0.5, len(group)), Gaussian()
    elif kind == "poisson":
        y, family = rng.poisson(np.exp(eta)), Poisson()
    else:
        y, family = rng.binomial(1, 1 / (1 + np.exp(-eta))), Binomial()
    data = pd.DataFrame({"y": y, "x": x, "z": z, "g": group})
    matrices = build_model_matrices(
        parse_formula("y ~ x + (0 + z | g)"),
        data,
        weights=np.linspace(0.7, 1.4, len(group)),
        offset=offset,
    )
    return matrices, family


def evaluate(matrices, family, native=False, order=15, n_jobs=1, theta=0.7):
    if native:
        pytest.importorskip("mixedlm._rust")
        deviance, beta, u, _ = laplace.glmm_deviance_with_status(
            np.array([theta]), matrices, family, nAGQ=order
        )
        return deviance, beta, u
    return laplace.adaptive_gh_deviance(
        np.array([theta]), matrices, family, nAGQ=order, n_jobs=n_jobs
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("pattern", ["mixed", "all_zero", "empty_group", "nonzero"])
@pytest.mark.parametrize("theta", [0.0, 0.7, -1.2])
def test_gaussian_quadrature_matches_dense_marginal_calculation(native, pattern, theta):
    matrices, family = make_matrices(pattern=pattern)
    deviance, beta, u = evaluate(matrices, family, native, theta=theta)
    sqrt_w = np.sqrt(matrices.weights)
    wz = sqrt_w[:, None] * matrices.Z.toarray()
    covariance = np.eye(matrices.n_obs) + theta**2 * (wz @ wz.T)
    residual = sqrt_w * (matrices.y - matrices.X @ beta - matrices.offset)
    expected = np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual)
    assert deviance == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert np.isfinite(u).all()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["poisson", "binomial"])
@pytest.mark.parametrize("pattern", ["mixed", "empty_group", "all_zero"])
def test_quadrature_includes_fixed_rows_in_direct_integrals(native, kind, pattern):
    matrices, family = make_matrices(kind, pattern)
    deviance, beta, _ = evaluate(matrices, family, native, order=45, theta=0.4)
    fixed_eta = matrices.X @ beta + matrices.offset
    group_design = matrices.Z.toarray()
    included = np.zeros(matrices.n_obs, dtype=bool)
    expected = 0.0
    for g in range(matrices.n_random):
        rows = group_design[:, g] != 0
        included |= rows
        if not rows.any():
            continue

        def integrand(value, rows=rows, g=g):
            mu = family.clamp_mu(
                family.link.inverse(fixed_eta[rows] + group_design[rows, g] * 0.4 * value),
                eps=1e-10,
            )
            conditional = np.sum(
                family.deviance_resids(matrices.y[rows], mu, matrices.weights[rows])
            )
            return float(np.exp(-0.5 * conditional) * stats.norm.pdf(value))

        integral, error = integrate.quad(integrand, -10, 10, epsabs=1e-12, epsrel=1e-12)
        assert error < 1e-9
        expected -= 2 * np.log(integral)
    mu = family.clamp_mu(family.link.inverse(fixed_eta[~included]), eps=1e-10)
    expected += np.sum(
        family.deviance_resids(matrices.y[~included], mu, matrices.weights[~included])
    )
    assert deviance == pytest.approx(expected, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kind", ["gaussian", "poisson", "binomial"])
def test_explicit_sparse_zeros_match_implicit_zeros_without_mutating_inputs(native, kind):
    matrices, family = make_matrices(kind)
    dense = matrices.Z.toarray()
    groups = np.repeat(np.arange(4), 6)
    stored = sparse.coo_matrix(
        (dense[np.arange(len(groups)), groups], (np.arange(len(groups)), groups)), shape=dense.shape
    ).tocsc()
    assert stored.nnz > matrices.Z.nnz
    stored_data = stored.data.copy()
    stored.data.flags.writeable = False
    actual = evaluate(replace(matrices, Z=stored), family, native)
    expected = evaluate(matrices, family, native)
    np.testing.assert_array_equal(stored.data, stored_data)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["gaussian", "poisson", "binomial"])
def test_parallel_quadrature_preserves_shared_modes(kind):
    matrices, family = make_matrices(kind)
    state = laplace._pirls_state(matrices, family, np.array([0.7]))
    state.spherical.flags.writeable = False
    before = state.spherical.copy()
    with patch.object(laplace, "_pirls_state", return_value=state):
        serial = evaluate(matrices, family, n_jobs=1)
        parallel = evaluate(matrices, family, n_jobs=2)
    np.testing.assert_array_equal(state.spherical, before)
    for value, reference in zip(parallel, serial, strict=True):
        np.testing.assert_array_equal(value, reference)


def test_node_link_evaluations_only_visit_the_current_group():
    matrices, family = make_matrices(pattern="nonzero", n_groups=20, n_per_group=5)
    state = laplace._pirls_state(matrices, family, np.array([0.7]))
    original = family.link.inverse
    sizes = []

    def record(eta):
        sizes.append(len(eta))
        return original(eta)

    with (
        patch.object(laplace, "_pirls_state", return_value=state),
        patch.object(family.link, "inverse", side_effect=record),
    ):
        evaluate(matrices, family, order=7)
    assert sizes.count(matrices.n_obs) == 1
    assert sizes.count(5) == 20 * 7


@pytest.mark.parametrize("native", [False, True])
def test_overlapping_random_effect_rows_are_rejected(native):
    matrices, family = make_matrices(pattern="nonzero")
    z = matrices.Z.tolil()
    z[0, 1] = 0.5
    matrices = replace(matrices, Z=z.tocsc())
    with pytest.raises(
        ValueError, match="at most one nonzero random-effect coefficient per observation"
    ):
        evaluate(matrices, family, native)


def test_noncanonical_python_design_is_combined_without_mutation():
    matrices, family = make_matrices(pattern="nonzero")
    z = matrices.Z
    data = np.repeat(z.data / 2, 2)
    indices = np.repeat(z.indices, 2)
    indptr = z.indptr * 2
    duplicated = sparse.csc_matrix((data, indices, indptr), shape=z.shape)
    assert not duplicated.has_canonical_format
    original = (duplicated.data.copy(), duplicated.indices.copy(), duplicated.indptr.copy())
    actual = evaluate(replace(matrices, Z=duplicated), family)
    expected = evaluate(matrices, family)
    for current, previous in zip(
        (duplicated.data, duplicated.indices, duplicated.indptr), original, strict=True
    ):
        np.testing.assert_array_equal(current, previous)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-10)
