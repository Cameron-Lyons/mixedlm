from __future__ import annotations

import numpy as np
import pytest
from mixedlm._rust import (
    LmmDesign,
    glmm_deviance,
    nlmm_deviance_with_status,
    simulate_re_batch,
    sparse_cholesky_logdet,
    sparse_cholesky_solve,
)
from mixedlm.nlme.models import SSasymp
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


def lmm_data(*groupings, slope=False):
    """Native LMM arguments with a random intercept, or a correlated slope, per grouping."""
    rng = np.random.default_rng(42)
    n = len(groupings[0])
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = x @ np.array([1.0, 0.5]) + rng.normal(scale=0.5, size=n)
    terms = x if slope else x[:, :1]
    width = terms.shape[1]
    # Each level's columns are adjacent, as in the model matrices.
    z = sparse.hstack(
        [
            sparse.csc_matrix(
                (
                    terms.ravel(),
                    (
                        np.repeat(np.arange(n), width),
                        (width * groups[:, None] + np.arange(width)).ravel(),
                    ),
                ),
                shape=(n, width * (groups.max() + 1)),
            )
            for groups in groupings
        ],
        format="csc",
    )
    return {
        "y": y,
        "x": x,
        "z_data": z.data,
        "z_indices": z.indices.astype(np.int64),
        "z_indptr": z.indptr.astype(np.int64),
        "z_shape": z.shape,
        "weights": np.ones(n),
        "offset": np.zeros(n),
        "n_levels": [int(groups.max()) + 1 for groups in groupings],
        "n_terms": [width] * len(groupings),
        "correlated": [slope] * len(groupings),
    }


def lmm_response(d):
    design = {name: value for name, value in d.items() if name != "y"}
    return LmmDesign(**design).with_response(d["y"])


def dense_profiled_deviance(d, theta, reml):
    """Profiled deviance from the dense observation covariance of scalar terms."""
    z = sparse.csc_matrix((d["z_data"], d["z_indices"], d["z_indptr"]), shape=d["z_shape"])
    scaled = z.toarray() * np.repeat(theta, d["n_levels"])
    covariance = np.diag(1 / d["weights"]) + scaled @ scaled.T
    x, residual = d["x"], d["y"] - d["offset"]
    information = x.T @ np.linalg.solve(covariance, x)
    beta = np.linalg.solve(information, x.T @ np.linalg.solve(covariance, residual))
    residual = residual - x @ beta
    df = len(residual) - (x.shape[1] if reml else 0)
    rss = residual @ np.linalg.solve(covariance, residual)
    deviance = df * (1 + np.log(2 * np.pi * rss / df)) + np.linalg.slogdet(covariance)[1]
    return deviance + (np.linalg.slogdet(information)[1] if reml else 0.0)


def as_lists(array):
    return array.tolist()


def as_tuples(array):
    return tuple(map(tuple, array)) if array.ndim == 2 else tuple(array.tolist())


class TestLmmDesign:
    @pytest.mark.parametrize("convert", [as_lists, as_tuples])
    def test_design_accepts_lists_and_tuples(self, convert):
        d = lmm_data(np.repeat(np.arange(4), 5))
        theta = np.array([1.0])
        expected = lmm_response(d).deviance_with_gradient(theta)
        converted = {
            name: convert(value) if isinstance(value, np.ndarray) else value
            for name, value in d.items()
        }
        actual = lmm_response(converted).deviance_with_gradient(convert(theta))
        assert actual[0] == expected[0]
        assert_array_equal(actual[1], expected[1])

    @pytest.mark.parametrize("reml", [False, True])
    @pytest.mark.parametrize("change", ["none", "weights", "offset", "crossed"])
    def test_deviance_matches_dense_likelihood(self, reml, change):
        if change == "crossed":
            d = lmm_data(np.tile(np.arange(3), 10), np.repeat(np.arange(5), 6))
            theta = np.array([1.0, 0.5])
        else:
            d = lmm_data(np.repeat(np.arange(4), 5))
            theta = np.array([1.0])
        rng = np.random.default_rng(7)
        if change == "weights":
            d["weights"] = rng.random(len(d["y"])) + 0.5
        elif change == "offset":
            d["offset"] = rng.normal(size=len(d["y"]))
        response = lmm_response(d)
        deviance = response.deviance(theta, reml)
        assert_allclose(deviance, dense_profiled_deviance(d, theta, reml), rtol=1e-12)
        assert deviance == response.evaluate(theta, reml)[0]

    @pytest.mark.parametrize("reml", [False, True])
    @pytest.mark.parametrize("slope", [False, True])
    def test_gradient_matches_deviance_and_central_differences(self, reml, slope):
        d = lmm_data(np.repeat(np.arange(5), 6), slope=slope)
        theta = np.array([1.0, 0.5, 0.8]) if slope else np.array([1.0])
        response = lmm_response(d)
        deviance, gradient = response.deviance_with_gradient(theta, reml)
        assert isinstance(deviance, float)
        assert isinstance(gradient, np.ndarray) and gradient.shape == theta.shape
        assert deviance == response.deviance(theta, reml)
        eps = 1e-6
        expected = [
            (response.deviance(theta + step, reml) - response.deviance(theta - step, reml))
            / (2 * eps)
            for step in eps * np.eye(len(theta))
        ]
        assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)


class TestDirectRustFunctions:
    def test_nlmm_weights_are_optional_and_effective(self):
        groups = np.repeat(np.arange(4), 8)
        x = np.tile(np.linspace(0.0, 5.0, 8), 4)
        phi = np.array([10.0, 0.5, -0.5])
        b = np.zeros((4, 1))
        theta = np.array([1.0])
        model = SSasymp()
        y = np.concatenate(
            [model.predict(phi + np.array([effect, 0.0, 0.0]), x[:8]) for effect in (-1, 0, 1, 2)]
        )
        args = (theta, y, x, groups, "ssasymp", phi, b, [0], 0.3)

        default = nlmm_deviance_with_status(*args)
        unit_weighted = nlmm_deviance_with_status(*args, np.ones(len(y)))
        weighted = nlmm_deviance_with_status(*args, np.linspace(0.25, 2.0, len(y)))

        assert default[0] == pytest.approx(unit_weighted[0], abs=1e-12)
        assert weighted[0] != pytest.approx(default[0])

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

    @staticmethod
    def _deviance(d, family="binomial", link="logit", n_agq=1, convert=np.asarray):
        return glmm_deviance(
            convert(d["y"]),
            convert(d["x"]),
            convert(d["z_data"]),
            convert(d["z_indices"]),
            convert(d["z_indptr"]),
            d["z_shape"],
            convert(d["weights"]),
            convert(d["offset"]),
            convert(d["theta"]),
            d["n_levels"],
            d["n_terms"],
            d["correlated"],
            family,
            link,
            n_agq,
        )

    def test_laplace_returns_mode_and_accepts_lists(self, simple_glmm_data):
        deviance, beta, u, converged = self._deviance(simple_glmm_data)
        assert converged
        assert np.isfinite(deviance)
        assert np.shape(beta) == (2,)
        assert np.shape(u) == (4,)

        from_lists = self._deviance(simple_glmm_data, convert=lambda value: value.tolist())
        assert from_lists[0] == deviance
        assert_array_equal(from_lists[1], beta)
        assert_array_equal(from_lists[2], u)

    def test_one_quadrature_node_is_the_laplace_approximation(self, simple_glmm_data):
        laplace, *_ = self._deviance(simple_glmm_data, n_agq=1)
        adaptive, beta, u, converged = self._deviance(simple_glmm_data, n_agq=5)
        assert converged
        assert np.isfinite(adaptive)
        assert adaptive == pytest.approx(laplace, rel=1e-2)
        assert adaptive != laplace

    def test_poisson_mode_reproduces_constant_mean(self, simple_glmm_data):
        d = simple_glmm_data
        d["y"] = np.full(len(d["y"]), 5.0)

        deviance, beta, u, converged = self._deviance(d, "poisson", "log")
        z = sparse.csc_matrix((d["z_data"], d["z_indices"], d["z_indptr"]), shape=d["z_shape"])
        fitted = np.exp(d["x"] @ np.asarray(beta) + z @ np.asarray(u) + d["offset"])

        assert converged
        assert np.isfinite(deviance)
        assert_allclose(fitted, 5.0, rtol=1e-6)

    def test_poisson_preserves_large_means(self, simple_glmm_data):
        d = simple_glmm_data
        rng = np.random.default_rng(123)
        eta = 3.0 + 0.3 * d["x"][:, 1]
        d["y"] = rng.poisson(np.exp(eta)).astype(float)

        deviance, beta, u, converged = self._deviance(d, "poisson", "log")

        assert converged
        assert np.isfinite(deviance)
        assert 2.5 < beta[0] < 3.5
        assert np.max(np.exp(d["x"] @ beta)) > 10

    def test_gaussian_identity_mode_solves_penalized_least_squares(self, simple_glmm_data):
        d = simple_glmm_data
        d["y"] = np.random.default_rng(7).normal(size=len(d["y"]))

        deviance, beta, u, converged = self._deviance(d, "gaussian", "identity")
        z = sparse.csc_matrix((d["z_data"], d["z_indices"], d["z_indptr"]), shape=d["z_shape"])
        lam = d["theta"][0] * np.eye(z.shape[1])
        design = np.hstack([d["x"], z.toarray() @ lam])
        penalty = np.diag([0.0] * d["x"].shape[1] + [1.0] * z.shape[1])
        solution = np.linalg.solve(design.T @ design + penalty, design.T @ d["y"])

        assert converged
        assert np.isfinite(deviance)
        assert_allclose(beta, solution[:2], rtol=1e-8, atol=1e-10)


class TestEdgeCasesAndErrors:
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


class TestMultipleRandomEffects:
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
