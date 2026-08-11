"""Tests for EM-REML algorithm."""

import warnings

import numpy as np
import pytest
from mixedlm import lFormula, lmer, lmerControl, load_penicillin, load_sleepstudy, set_cov_type
from mixedlm.estimation.em_reml import (
    _m_step_update_sigma,
    _sigma_to_theta,
    _weighted_crossproducts,
    em_reml_simple,
)
from mixedlm.matrices.design import RandomEffectStructure, build_model_matrices
from scipy import sparse


class TestEMReml:
    def test_weighted_crossproducts_match_dense_reference(self):
        rng = np.random.default_rng(20260803)
        X = rng.normal(size=(20, 3))
        Z_dense = rng.normal(size=(20, 8))
        Z_dense[np.abs(Z_dense) < 0.8] = 0.0
        Z = sparse.csc_matrix(Z_dense)
        y = rng.normal(size=20)
        weights = rng.uniform(0.25, 2.0, size=20)

        XtWX, XtWZ, ZtWZ, XtWy, ZtWy = _weighted_crossproducts(X, Z, y, weights)

        assert np.allclose(XtWX, X.T @ (weights[:, None] * X))
        assert np.allclose(XtWZ, X.T @ (weights[:, None] * Z_dense))
        assert np.allclose(ZtWZ, Z_dense.T @ (weights[:, None] * Z_dense))
        assert np.allclose(XtWy, X.T @ (weights * y))
        assert np.allclose(ZtWy, Z_dense.T @ (weights * y))

    def test_em_reml_keeps_random_design_sparse(self, monkeypatch):
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)
        random_design_shape = parsed.matrices.Z.shape
        original_toarray = sparse.csc_matrix.toarray

        def reject_random_design(self, *args, **kwargs):
            if self.shape == random_design_shape:
                raise AssertionError("EM-REML should not densify the random-effects design")
            return original_toarray(self, *args, **kwargs)

        monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_random_design)

        result = em_reml_simple(parsed.matrices, max_iter=3, min_iter_converge=10)
        assert result.n_iter == 3

    def test_converged_result_contains_latest_iteration(self):
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        result = em_reml_simple(
            parsed.matrices,
            max_iter=2,
            tol=2.0,
            min_iter_converge=0,
        )

        assert result.converged
        assert result.n_iter == 1
        assert np.isfinite(result.final_loglik)

    def test_max_iter_must_be_positive(self):
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (1 | Subject)", data)

        with pytest.raises(ValueError, match="max_iter must be at least 1"):
            em_reml_simple(parsed.matrices, max_iter=0)

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

    def test_em_reml_random_slope(self):
        """Test EM-REML with correlated random slope model."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=200, tol=1e-4, verbose=0)

        assert len(result.theta) == 3
        assert result.theta[0] > 0
        assert result.sigma > 0
        assert len(result.beta) == 2

    def test_em_reml_uncorrelated_slope(self):
        """Test EM-REML with uncorrelated random slope model."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (Days || Subject)", data)

        result = em_reml_simple(parsed.matrices, max_iter=200, tol=1e-4, verbose=0)

        assert len(result.theta) == 2
        assert all(t > 0 for t in result.theta)
        assert result.sigma > 0
        assert len(result.beta) == 2

    def test_em_reml_multiple_random_effects(self):
        """Test EM-REML with multiple random intercept terms."""
        data = load_penicillin()
        parsed = lFormula("diameter ~ 1 + (1 | plate) + (1 | sample)", data)

        result = em_reml_simple(parsed.matrices, max_iter=200, tol=1e-4, verbose=0)

        assert len(result.theta) == 2
        assert all(t >= 0 for t in result.theta)
        assert result.sigma > 0

    def test_em_reml_slope_reasonable_estimates(self):
        """Test that EM-REML slope estimates are in a reasonable range."""
        data = load_sleepstudy()
        parsed = lFormula("Reaction ~ Days + (Days | Subject)", data)

        em_result = em_reml_simple(parsed.matrices, max_iter=200, tol=1e-5, verbose=0)

        assert 200 < em_result.beta[0] < 300
        assert 5 < em_result.beta[1] < 15
        assert 10 < em_result.sigma < 50

        fit = lmer("Reaction ~ Days + (Days | Subject)", data)
        for i in range(len(em_result.beta)):
            assert abs(em_result.beta[i] - fit.beta[i]) / abs(fit.beta[i]) < 0.1

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

    def test_em_reml_single_group_level(self):
        """Test EM-REML with minimal number of group levels."""
        data = load_sleepstudy()
        subset = data[data["Subject"] == data["Subject"].iloc[0]].copy()
        subset["Group"] = "A"
        parsed = lFormula("Reaction ~ Days + (1 | Group)", subset)

        result = em_reml_simple(parsed.matrices, max_iter=50, verbose=0)

        assert len(result.theta) == 1
        assert result.sigma > 0

    def test_em_reml_compound_symmetry(self):
        """Test EM-REML with compound-symmetry covariance."""
        data = load_sleepstudy()
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "cs")
        matrices = build_model_matrices(formula, data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = em_reml_simple(matrices, max_iter=100)

        assert result.converged
        assert np.isfinite(result.final_loglik)
        assert len(result.theta) == 2
        assert result.theta[0] > 0
        assert -1 < result.theta[1] < 1
        assert result.sigma > 0

        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            fit = lmer(formula, data, control=lmerControl(em_init=True, em_maxiter=100))

        assert fit.converged
        assert len(fit.theta) == 2
        assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
        assert not any("EM initialization failed" in str(item.message) for item in fit_warnings)

    def test_compound_symmetry_m_step(self):
        """Test the exact constrained covariance update."""
        struct = RandomEffectStructure(
            grouping_factor="group",
            term_names=["a", "b", "c"],
            n_levels=2,
            n_terms=3,
            correlated=True,
            level_map={"A": 0, "B": 1},
            cov_type="cs",
        )
        u_block = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])
        var_u_block = np.eye(6) * 0.3

        sigma = _m_step_update_sigma(struct, u_block, var_u_block, variance_floor=1e-8)

        np.testing.assert_allclose(np.diag(sigma), np.repeat(89 / 30, 3))
        np.testing.assert_allclose(sigma[np.triu_indices(3, k=1)], np.repeat(5 / 3, 3))
        assert np.min(np.linalg.eigvalsh(sigma)) >= 1e-8

        theta = _sigma_to_theta([struct], [sigma], sigma2_e=2.0)
        np.testing.assert_allclose(theta, [np.sqrt(89 / 60), 50 / 89])

    def test_em_reml_ar1_not_supported(self):
        """Test that AR(1) still uses the explicit fallback path."""
        data = load_sleepstudy()
        formula = set_cov_type("Reaction ~ Days + (Days | Subject)", "ar1")
        matrices = build_model_matrices(formula, data)

        with pytest.raises(NotImplementedError, match="cov_type='ar1'"):
            em_reml_simple(matrices, max_iter=10)
