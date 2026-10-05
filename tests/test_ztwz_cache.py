"""Tests for Z'WZ caching optimization."""

import numpy as np
from mixedlm import lFormula, lmer, load_sleepstudy
from mixedlm.estimation.reml import LMMOptimizer


class TestZTWZCache:
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
        """Test that the cached native deviance matches the Python profile."""
        from mixedlm.estimation.reml import (
            _profiled_deviance_rust_cached,
            _RustMatrixCache,
            profiled_deviance,
        )

        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)
        matrices = parsed.matrices

        theta = np.array([0.9])
        cache = _RustMatrixCache.from_matrices(matrices)
        dev_cached = _profiled_deviance_rust_cached(theta, cache, REML=True)
        dev_python = profiled_deviance(theta, matrices, REML=True)

        assert np.abs(dev_cached - dev_python) < 1e-9, (
            f"Cached native ({dev_cached}) and Python ({dev_python}) deviances should match"
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
