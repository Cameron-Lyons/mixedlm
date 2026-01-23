"""Tests for Z'WZ caching optimization."""

import numpy as np
from mixedlm import lFormula, lmer, load_sleepstudy
from mixedlm.estimation.reml import LMMOptimizer


class TestZTWZCache:
    def test_ztwz_cache_consistency(self):
        """Test that cached Z'WZ is computed correctly."""
        from mixedlm._rust import compute_ztwz
        from scipy import sparse

        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)
        matrices = parsed.matrices

        z_csc = matrices.Z.tocsc()
        z_data = np.ascontiguousarray(z_csc.data)
        z_indices = np.ascontiguousarray(z_csc.indices.astype(np.int64))
        z_indptr = np.ascontiguousarray(z_csc.indptr.astype(np.int64))
        z_shape = (z_csc.shape[0], z_csc.shape[1])
        weights = np.ascontiguousarray(matrices.weights)

        ztwz_rust = compute_ztwz(z_data, z_indices, z_indptr, z_shape, weights)
        q = z_shape[1]
        ztwz_rust_mat = ztwz_rust.reshape((q, q))

        sqrt_w = np.sqrt(weights)
        WZ = sparse.diags(sqrt_w, format="csc") @ matrices.Z
        ztwz_python = (WZ.T @ WZ).toarray()

        assert np.allclose(ztwz_rust_mat, ztwz_python, rtol=1e-12, atol=1e-12), (
            "Cached Z'WZ should match Python computation"
        )

    def test_ztwz_cache_with_lmer(self):
        """Test that lmer with caching produces valid results."""
        data = load_sleepstudy()
        model = lmer("Reaction ~ Days + (1 | Subject)", data)

        assert model.converged
        assert len(model.theta) == 1
        assert model.theta[0] >= 0

        beta = model.fixef()
        assert len(beta) == 2
        assert "(Intercept)" in beta
        assert "Days" in beta

    def test_ztwz_cache_deviance_consistency(self):
        """Test that cached deviance matches uncached Rust deviance."""
        from mixedlm._rust import profiled_deviance
        from mixedlm.estimation.reml import (
            _profiled_deviance_rust_cached,
            _RustMatrixCache,
        )

        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)
        matrices = parsed.matrices

        theta = np.array([0.9])

        # Compute deviance without caching (calling raw Rust function)
        z_csc = matrices.Z.tocsc()
        dev_uncached = profiled_deviance(
            theta,
            matrices.y,
            matrices.X,
            np.ascontiguousarray(z_csc.data),
            np.ascontiguousarray(z_csc.indices.astype(np.int64)),
            np.ascontiguousarray(z_csc.indptr.astype(np.int64)),
            (z_csc.shape[0], z_csc.shape[1]),
            matrices.weights,
            matrices.offset,
            [s.n_levels for s in matrices.random_structures],
            [s.n_terms for s in matrices.random_structures],
            [s.correlated for s in matrices.random_structures],
            True,
        )

        # Compute deviance with caching
        cache = _RustMatrixCache.from_matrices(matrices)
        dev_cached = _profiled_deviance_rust_cached(theta, cache, REML=True)

        assert np.abs(dev_cached - dev_uncached) < 1e-12, (
            f"Cached ({dev_cached}) and uncached ({dev_uncached}) "
            f"Rust deviances should match exactly"
        )

    def test_ztwz_cache_multiple_calls(self):
        """Test that cache works correctly across multiple deviance evaluations."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        optimizer = LMMOptimizer(parsed.matrices, REML=True, verbose=0, use_rust=True)

        theta1 = np.array([0.5])
        theta2 = np.array([1.0])
        theta3 = np.array([1.5])

        dev1a = optimizer.objective(theta1)
        dev2a = optimizer.objective(theta2)
        dev3a = optimizer.objective(theta3)

        dev1b = optimizer.objective(theta1)
        dev2b = optimizer.objective(theta2)
        dev3b = optimizer.objective(theta3)

        assert np.abs(dev1a - dev1b) < 1e-12
        assert np.abs(dev2a - dev2b) < 1e-12
        assert np.abs(dev3a - dev3b) < 1e-12
