from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

try:
    from mixedlm._rust import (
        adaptive_gh_deviance,
        compute_zu,
        gauss_hermite,
        laplace_deviance,
        pirls,
        profiled_deviance,
        simulate_re_batch,
        sparse_cholesky_logdet,
        sparse_cholesky_solve,
        update_cholesky_factor,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not available")


class TestPyArrayLikeInputs:
    @pytest.fixture
    def simple_lmm_data(self):
        np.random.seed(42)
        n = 20
        n_groups = 4
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        z_dense = np.zeros((n, n_groups))
        for i, g in enumerate(groups):
            z_dense[i, g] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        return {
            "y": y,
            "x": x,
            "z_data": z_csc.data,
            "z_indices": z_csc.indices.astype(np.int64),
            "z_indptr": z_csc.indptr.astype(np.int64),
            "z_shape": z_csc.shape,
            "theta": np.array([1.0]),
            "weights": np.ones(n),
            "offset": np.zeros(n),
            "n_levels": [n_groups],
            "n_terms": [1],
            "correlated": [False],
        }

    def test_profiled_deviance_with_numpy_arrays(self, simple_lmm_data):
        d = simple_lmm_data
        result = profiled_deviance(
            d["theta"],
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_profiled_deviance_with_python_lists(self, simple_lmm_data):
        d = simple_lmm_data
        result = profiled_deviance(
            d["theta"].tolist(),
            d["y"].tolist(),
            d["x"].tolist(),
            d["z_data"].tolist(),
            d["z_indices"].tolist(),
            d["z_indptr"].tolist(),
            d["z_shape"],
            d["weights"].tolist(),
            d["offset"].tolist(),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_profiled_deviance_with_tuples(self, simple_lmm_data):
        d = simple_lmm_data
        result = profiled_deviance(
            tuple(d["theta"]),
            tuple(d["y"]),
            tuple(map(tuple, d["x"])),
            tuple(d["z_data"]),
            tuple(d["z_indices"]),
            tuple(d["z_indptr"]),
            d["z_shape"],
            tuple(d["weights"]),
            tuple(d["offset"]),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_profiled_deviance_list_equals_numpy(self, simple_lmm_data):
        d = simple_lmm_data
        result_numpy = profiled_deviance(
            d["theta"],
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        result_list = profiled_deviance(
            d["theta"].tolist(),
            d["y"].tolist(),
            d["x"].tolist(),
            d["z_data"].tolist(),
            d["z_indices"].tolist(),
            d["z_indptr"].tolist(),
            d["z_shape"],
            d["weights"].tolist(),
            d["offset"].tolist(),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        assert_allclose(result_numpy, result_list)

    def test_profiled_deviance_with_float32_converted(self, simple_lmm_data):
        d = simple_lmm_data
        result = profiled_deviance(
            np.asarray(d["theta"], dtype=np.float64),
            np.asarray(d["y"], dtype=np.float64),
            np.asarray(d["x"], dtype=np.float64),
            np.asarray(d["z_data"], dtype=np.float64),
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            np.asarray(d["weights"], dtype=np.float64),
            np.asarray(d["offset"], dtype=np.float64),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            True,
        )
        assert isinstance(result, float)

    def test_gauss_hermite_with_list_input(self):
        nodes, weights = gauss_hermite(5)
        assert len(nodes) == 5
        assert len(weights) == 5

    def test_compute_zu_with_lists(self):
        u = [1.0, 2.0]
        z_data = [1.0, 1.0]
        z_indices = [0, 1]
        z_indptr = [0, 1, 2]
        z_shape = (2, 2)
        n_obs = 2

        result = compute_zu(u, z_data, z_indices, z_indptr, z_shape, n_obs)
        assert len(result) == 2


class TestDirectRustFunctions:
    def test_gauss_hermite_basic(self):
        nodes, weights = gauss_hermite(5)
        assert len(nodes) == 5
        assert len(weights) == 5
        assert_allclose(sum(weights), np.sqrt(np.pi), rtol=1e-10)

    def test_gauss_hermite_n1(self):
        nodes, weights = gauss_hermite(1)
        assert len(nodes) == 1
        assert len(weights) == 1
        assert_allclose(nodes[0], 0.0, atol=1e-10)
        assert_allclose(weights[0], np.sqrt(np.pi), rtol=1e-10)

    def test_gauss_hermite_n2(self):
        nodes, weights = gauss_hermite(2)
        assert len(nodes) == 2
        assert_allclose(sorted(nodes), [-1 / np.sqrt(2), 1 / np.sqrt(2)], rtol=1e-10)

    def test_gauss_hermite_symmetry(self):
        nodes, weights = gauss_hermite(10)
        nodes = np.array(nodes)
        weights = np.array(weights)
        sorted_idx = np.argsort(nodes)
        nodes = nodes[sorted_idx]
        weights = weights[sorted_idx]
        assert_allclose(nodes, -nodes[::-1], atol=1e-10)
        assert_allclose(weights, weights[::-1], rtol=1e-10)

    def test_gauss_hermite_large_n(self):
        nodes, weights = gauss_hermite(50)
        assert len(nodes) == 50
        assert all(np.isfinite(nodes))
        assert all(np.isfinite(weights))

    def test_sparse_cholesky_solve_basic(self):
        A = sparse.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        b = np.array([[1.0], [2.0]])

        result = sparse_cholesky_solve(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
            b,
        )
        result = np.array(result)
        expected = np.linalg.solve(A.toarray(), b)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sparse_cholesky_solve_identity(self):
        A = sparse.csc_matrix(np.eye(3))
        b = np.array([[1.0], [2.0], [3.0]])

        result = sparse_cholesky_solve(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
            b,
        )
        result = np.array(result)
        assert_allclose(result, b, rtol=1e-10)

    def test_sparse_cholesky_solve_multiple_rhs(self):
        A = sparse.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        b = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = sparse_cholesky_solve(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
            b,
        )
        result = np.array(result)
        expected = np.linalg.solve(A.toarray(), b)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sparse_cholesky_logdet_basic(self):
        A = sparse.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))

        result = sparse_cholesky_logdet(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
        )
        expected = np.log(np.linalg.det(A.toarray()))
        assert_allclose(result, expected, rtol=1e-10)

    def test_sparse_cholesky_logdet_identity(self):
        A = sparse.csc_matrix(np.eye(5))

        result = sparse_cholesky_logdet(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
        )
        assert_allclose(result, 0.0, atol=1e-10)

    def test_sparse_cholesky_logdet_diagonal(self):
        diag = np.array([2.0, 3.0, 4.0])
        A = sparse.csc_matrix(np.diag(diag))

        result = sparse_cholesky_logdet(
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            A.shape,
        )
        expected = np.sum(np.log(diag))
        assert_allclose(result, expected, rtol=1e-10)

    def test_simulate_re_batch_basic(self):
        theta = np.array([1.0])
        sigma = 1.0
        n_levels = [5]
        n_terms = [1]
        correlated = [False]
        n_sim = 10

        result = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result = np.array(result)
        assert result.shape == (n_sim, 5)

    def test_simulate_re_batch_reproducible(self):
        theta = np.array([1.0])
        sigma = 1.0
        n_levels = [5]
        n_terms = [1]
        correlated = [False]
        n_sim = 10

        result1 = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result2 = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        assert_allclose(result1, result2)

    def test_simulate_re_batch_different_seeds(self):
        theta = np.array([1.0])
        sigma = 1.0
        n_levels = [5]
        n_terms = [1]
        correlated = [False]
        n_sim = 10

        result1 = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result2 = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=43)
        assert not np.allclose(result1, result2)

    def test_simulate_re_batch_correlated(self):
        theta = np.array([1.0, 0.5, 1.0])
        sigma = 1.0
        n_levels = [5]
        n_terms = [2]
        correlated = [True]
        n_sim = 100

        result = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result = np.array(result)
        assert result.shape == (n_sim, 10)

    def test_compute_zu_basic(self):
        u = np.array([1.0, 2.0])
        z_dense = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        z_csc = sparse.csc_matrix(z_dense)

        result = compute_zu(
            u,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            3,
        )
        result = np.array(result)
        expected = z_dense @ u
        assert_allclose(result, expected)

    def test_compute_zu_sparse(self):
        u = np.array([1.0, 2.0, 3.0, 4.0])
        z_dense = np.zeros((10, 4))
        z_dense[0, 0] = 1.0
        z_dense[1, 0] = 1.0
        z_dense[2, 1] = 1.0
        z_dense[3, 1] = 1.0
        z_dense[4, 2] = 1.0
        z_dense[5, 2] = 1.0
        z_dense[6, 3] = 1.0
        z_dense[7, 3] = 1.0
        z_dense[8, 3] = 1.0
        z_dense[9, 3] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        result = compute_zu(
            u,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            10,
        )
        result = np.array(result)
        expected = z_dense @ u
        assert_allclose(result, expected)


class TestGLMMFunctions:
    @pytest.fixture
    def simple_glmm_data(self):
        np.random.seed(42)
        n = 40
        n_groups = 4
        groups = np.repeat(np.arange(n_groups), n // n_groups)

        x = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.array([-0.5, 0.3])
        eta = x @ beta
        p = 1 / (1 + np.exp(-eta))
        y = (np.random.rand(n) < p).astype(float)

        z_dense = np.zeros((n, n_groups))
        for i, g in enumerate(groups):
            z_dense[i, g] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        return {
            "y": y,
            "x": x,
            "z_data": z_csc.data,
            "z_indices": z_csc.indices.astype(np.int64),
            "z_indptr": z_csc.indptr.astype(np.int64),
            "z_shape": z_csc.shape,
            "theta": np.array([0.5]),
            "weights": np.ones(n),
            "offset": np.zeros(n),
            "n_levels": [n_groups],
            "n_terms": [1],
            "correlated": [False],
        }

    def test_pirls_basic(self, simple_glmm_data):
        d = simple_glmm_data
        beta, u, deviance, converged = pirls(
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["theta"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "binomial",
            "logit",
        )
        assert len(beta) == 2
        assert len(u) == 4
        assert np.isfinite(deviance)
        assert isinstance(converged, bool)

    def test_pirls_with_lists(self, simple_glmm_data):
        d = simple_glmm_data
        beta, u, deviance, converged = pirls(
            d["y"].tolist(),
            d["x"].tolist(),
            d["z_data"].tolist(),
            d["z_indices"].tolist(),
            d["z_indptr"].tolist(),
            d["z_shape"],
            d["weights"].tolist(),
            d["offset"].tolist(),
            d["theta"].tolist(),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "binomial",
            "logit",
        )
        assert len(beta) == 2
        assert np.isfinite(deviance)

    def test_laplace_deviance_basic(self, simple_glmm_data):
        d = simple_glmm_data
        deviance, beta, u = laplace_deviance(
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["theta"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "binomial",
            "logit",
        )
        assert np.isfinite(deviance)
        assert len(beta) == 2
        assert len(u) == 4

    def test_adaptive_gh_deviance_basic(self, simple_glmm_data):
        d = simple_glmm_data
        deviance, beta, u = adaptive_gh_deviance(
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["theta"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "binomial",
            "logit",
            5,
        )
        assert np.isfinite(deviance)
        assert len(beta) == 2
        assert len(u) == 4

    def test_pirls_poisson(self, simple_glmm_data):
        d = simple_glmm_data
        d["y"] = np.random.poisson(3, len(d["y"])).astype(float)

        beta, u, deviance, converged = pirls(
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["theta"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "poisson",
            "log",
        )
        assert np.isfinite(deviance)

    def test_pirls_gaussian(self, simple_glmm_data):
        d = simple_glmm_data
        d["y"] = np.random.randn(len(d["y"]))

        beta, u, deviance, converged = pirls(
            d["y"],
            d["x"],
            d["z_data"],
            d["z_indices"],
            d["z_indptr"],
            d["z_shape"],
            d["weights"],
            d["offset"],
            d["theta"],
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            "gaussian",
            "identity",
        )
        assert np.isfinite(deviance)


class TestEdgeCasesAndErrors:
    def test_gauss_hermite_n0_returns_empty(self):
        nodes, weights = gauss_hermite(0)
        assert len(nodes) == 0
        assert len(weights) == 0

    def test_sparse_cholesky_not_positive_definite(self):
        A = sparse.csc_matrix(np.array([[-1.0, 0.0], [0.0, 1.0]]))
        b = np.array([[1.0], [1.0]])

        with pytest.raises(ValueError):
            sparse_cholesky_solve(
                A.data,
                A.indices.astype(np.int64),
                A.indptr.astype(np.int64),
                A.shape,
                b,
            )

    def test_sparse_cholesky_logdet_not_positive_definite(self):
        A = sparse.csc_matrix(np.array([[-1.0, 0.0], [0.0, 1.0]]))

        with pytest.raises(ValueError):
            sparse_cholesky_logdet(
                A.data,
                A.indices.astype(np.int64),
                A.indptr.astype(np.int64),
                A.shape,
            )

    def test_simulate_re_batch_n_sim_zero(self):
        theta = np.array([1.0])
        sigma = 1.0
        n_levels = [5]
        n_terms = [1]
        correlated = [False]
        n_sim = 0

        result = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result = np.array(result)
        assert result.size == 0

    def test_profiled_deviance_reml_vs_ml(self):
        np.random.seed(42)
        n = 20
        n_groups = 4
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        z_dense = np.zeros((n, n_groups))
        for i, g in enumerate(groups):
            z_dense[i, g] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        theta = np.array([1.0])
        weights = np.ones(n)
        offset = np.zeros(n)

        reml_deviance = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights,
            offset,
            [n_groups],
            [1],
            [False],
            True,
        )

        ml_deviance = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights,
            offset,
            [n_groups],
            [1],
            [False],
            False,
        )

        assert reml_deviance != ml_deviance
        assert np.isfinite(reml_deviance)
        assert np.isfinite(ml_deviance)

    def test_profiled_deviance_with_weights(self):
        np.random.seed(42)
        n = 20
        n_groups = 4
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        z_dense = np.zeros((n, n_groups))
        for i, g in enumerate(groups):
            z_dense[i, g] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        theta = np.array([1.0])
        weights_uniform = np.ones(n)
        weights_varied = np.random.rand(n) + 0.5
        offset = np.zeros(n)

        result_uniform = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights_uniform,
            offset,
            [n_groups],
            [1],
            [False],
            True,
        )

        result_varied = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights_varied,
            offset,
            [n_groups],
            [1],
            [False],
            True,
        )

        assert result_uniform != result_varied

    def test_profiled_deviance_with_offset(self):
        np.random.seed(42)
        n = 20
        n_groups = 4
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        z_dense = np.zeros((n, n_groups))
        for i, g in enumerate(groups):
            z_dense[i, g] = 1.0
        z_csc = sparse.csc_matrix(z_dense)

        theta = np.array([1.0])
        weights = np.ones(n)
        offset_zero = np.zeros(n)
        offset_nonzero = np.random.randn(n)

        result_zero = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights,
            offset_zero,
            [n_groups],
            [1],
            [False],
            True,
        )

        result_nonzero = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights,
            offset_nonzero,
            [n_groups],
            [1],
            [False],
            True,
        )

        assert result_zero != result_nonzero

    def test_update_cholesky_factor_basic(self):
        L = sparse.csc_matrix(np.array([[2.0, 0.0], [0.5, 1.5]]))
        theta = np.array([1.0, 0.0, 1.0])

        data, indices, indptr = update_cholesky_factor(
            L.data,
            L.indices.astype(np.int64),
            L.indptr.astype(np.int64),
            L.shape,
            theta,
        )
        assert len(data) > 0
        assert len(indices) > 0
        assert len(indptr) > 0


class TestNonContiguousArrays:
    def test_gauss_hermite_accepts_contiguous(self):
        nodes, weights = gauss_hermite(5)
        assert len(nodes) == 5

    def test_compute_zu_with_slice(self):
        u_full = np.array([0.0, 1.0, 0.0, 2.0])
        u = u_full[1::2]
        assert not u.flags["C_CONTIGUOUS"]

        z_dense = np.eye(2)
        z_csc = sparse.csc_matrix(z_dense)

        result = compute_zu(
            np.ascontiguousarray(u),
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            2,
        )
        assert_allclose(result, [1.0, 2.0])


class TestMultipleRandomEffects:
    def test_profiled_deviance_crossed_random_effects(self):
        np.random.seed(42)
        n = 30
        n_groups1 = 3
        n_groups2 = 5

        groups1 = np.tile(np.arange(n_groups1), n // n_groups1)
        groups2 = np.repeat(np.arange(n_groups2), n // n_groups2)

        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = x @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

        z1_dense = np.zeros((n, n_groups1))
        for i, g in enumerate(groups1):
            z1_dense[i, g] = 1.0

        z2_dense = np.zeros((n, n_groups2))
        for i, g in enumerate(groups2):
            z2_dense[i, g] = 1.0

        z_dense = np.hstack([z1_dense, z2_dense])
        z_csc = sparse.csc_matrix(z_dense)

        theta = np.array([1.0, 0.5])
        weights = np.ones(n)
        offset = np.zeros(n)

        result = profiled_deviance(
            theta,
            y,
            x,
            z_csc.data,
            z_csc.indices.astype(np.int64),
            z_csc.indptr.astype(np.int64),
            z_csc.shape,
            weights,
            offset,
            [n_groups1, n_groups2],
            [1, 1],
            [False, False],
            True,
        )
        assert np.isfinite(result)

    def test_simulate_re_batch_multiple_terms(self):
        theta = np.array([1.0, 0.5])
        sigma = 1.0
        n_levels = [5, 3]
        n_terms = [1, 1]
        correlated = [False, False]
        n_sim = 10

        result = simulate_re_batch(theta, sigma, n_levels, n_terms, correlated, n_sim, seed=42)
        result = np.array(result)
        assert result.shape == (n_sim, 8)
