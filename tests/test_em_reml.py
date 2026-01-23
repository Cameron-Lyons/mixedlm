"""Tests for EM-REML algorithm."""

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
        assert result.theta[0] > 0
        assert len(result.beta) == 2
        assert result.sigma > 0

    def test_em_reml_convergence(self):
        """Test that EM-REML converges with sufficient iterations."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=100, tol=1e-5, verbose=0)

        assert result.converged
        assert result.n_iter < 100

    def test_em_reml_not_supported_complex(self):
        """Test that EM-REML raises error for unsupported models."""
        data = load_sleepstudy()

        parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)

        with pytest.raises(NotImplementedError, match="random intercept models"):
            em_reml_simple(parsed.matrices, max_iter=10)

    def test_em_reml_reasonable_estimates(self):
        """Test that EM-REML produces reasonable parameter estimates."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=100, verbose=0)

        assert 200 < result.beta[0] < 300

        assert 5 < result.beta[1] < 15

        assert 0.1 < result.theta[0] < 10
        assert 10 < result.sigma < 50

    def test_em_reml_avoids_singular_fits(self):
        """Test that EM-REML is more robust and avoids singular fits.

        This is one of the key advantages of EM-REML: it tends to avoid
        boundary solutions (theta=0) that direct optimization can converge to.
        """
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        em_result = em_reml_simple(parsed.matrices, max_iter=100, verbose=0)

        assert em_result.converged
        assert em_result.theta[0] > 0.1
        assert em_result.sigma > 0

        assert 200 < em_result.beta[0] < 300
        assert 5 < em_result.beta[1] < 15
