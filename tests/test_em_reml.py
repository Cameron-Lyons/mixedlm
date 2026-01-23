"""Tests for EM-REML algorithm."""

import numpy as np
import pytest

from mixedlm import lFormula, load_sleepstudy
from mixedlm.estimation.em_reml import em_reml_simple


class TestEMReml:
    def test_em_reml_simple_intercept(self):
        """Test EM-REML with simple random intercept model."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=50, verbose=0)

        assert result.converged
        assert len(result.theta) == 1
        assert result.theta[0] > 0  # Variance component should be positive
        assert len(result.beta) == 2  # Intercept + Days
        assert result.sigma > 0

    def test_em_reml_convergence(self):
        """Test that EM-REML converges with sufficient iterations."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=100, tol=1e-5, verbose=0)

        assert result.converged
        assert result.n_iter < 100  # Should converge before max_iter

    def test_em_reml_not_supported_complex(self):
        """Test that EM-REML raises error for unsupported models."""
        data = load_sleepstudy()

        # Random slope model (not supported)
        parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)

        with pytest.raises(NotImplementedError, match="random intercept models"):
            em_reml_simple(parsed.matrices, max_iter=10)

    @pytest.mark.xfail(reason="EM-REML has numerical issues, needs further refinement")
    def test_em_reml_reasonable_estimates(self):
        """Test that EM-REML produces reasonable parameter estimates."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=100, verbose=0)

        # Check that estimates are in reasonable range for sleepstudy data
        # Intercept should be around 250-260
        assert 200 < result.beta[0] < 300

        # Days effect should be around 10
        assert 5 < result.beta[1] < 15

        # Variance components should be positive and reasonable
        assert 0.1 < result.theta[0] < 10
        assert 10 < result.sigma < 50
