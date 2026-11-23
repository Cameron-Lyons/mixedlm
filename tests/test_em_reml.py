"""Tests for EM-REML algorithm."""

import warnings

import numpy as np
import pytest
from mixedlm import lFormula, lmer, lmerControl, load_penicillin, load_sleepstudy, set_cov_type
from mixedlm.estimation.em_reml import (
    _m_step_covariance,
    _sigma_to_theta,
    em_reml_simple,
)
from mixedlm.matrices.design import RandomEffectStructure, build_model_matrices
from scipy import linalg, optimize, sparse


class TestEMReml:
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

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("min_iter_converge", -1),
            ("tol", np.nan),
            ("variance_floor", 0),
            ("sigma2_u_init_min", -1),
            ("sigma2_u_init_scale", np.inf),
        ],
    )
    def test_invalid_controls(self, name, value):
        matrices = lFormula("Reaction ~ Days + (1 | Subject)", load_sleepstudy()).matrices
        with pytest.raises(ValueError, match=name):
            em_reml_simple(matrices, **{name: value})

    @pytest.mark.parametrize("slope", [0, 1])
    def test_rejects_rank_deficient_fixed_design(self, slope):
        matrices = lFormula("Reaction ~ Days + (1 | Subject)", load_sleepstudy()).matrices
        matrices.X[:, 1] = slope
        with pytest.raises(ValueError, match="full-rank fixed-effects"):
            em_reml_simple(matrices)

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

        U = u_block.reshape(2, 3)
        moment = (U.T @ U + var_u_block[:3, :3] + var_u_block[3:, 3:]) / 2
        sigma = _m_step_covariance(struct, moment, variance_floor=1e-8)

        np.testing.assert_allclose(np.diag(sigma), np.repeat(89 / 30, 3))
        np.testing.assert_allclose(sigma[np.triu_indices(3, k=1)], np.repeat(5 / 3, 3))
        assert np.min(np.linalg.eigvalsh(sigma)) >= 1e-8

        theta = _sigma_to_theta([struct], [sigma], sigma2_e=2.0)
        np.testing.assert_allclose(theta, [np.sqrt(89 / 60), 50 / 89])

    def test_em_reml_ar1_initialization(self):
        data = load_sleepstudy()
        data["Days2"] = (data["Days"] / 9) ** 2
        formula = set_cov_type("Reaction ~ Days + (Days + Days2 | Subject)", "ar1")
        matrices = build_model_matrices(formula, data)
        result = em_reml_simple(matrices, tol=1e-12, max_iter=1500)
        with warnings.catch_warnings(record=True) as caught:
            fit = lmer(formula, data, control=lmerControl(em_init=True))
        assert not any("EM initialization failed" in str(w.message) for w in caught)
        assert result.converged
        assert result.final_loglik == pytest.approx(float(fit.logLik()), abs=1e-6)
        np.testing.assert_allclose(result.theta, fit.theta, atol=1e-4)

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("kind", ["cs", "ar1"])
    def test_covariance_floor_preserves_unconstrained_eigenvalue(self, sign, kind):
        struct = RandomEffectStructure("g", ["a", "b"], 1, 2, True, {}, cov_type=kind)
        moment = np.array([[1, sign * 0.9], [sign * 0.9, 1]])
        covariance = _m_step_covariance(struct, moment, 0.8)
        np.testing.assert_allclose(covariance, [[1.35, sign * 0.55], [sign * 0.55, 1.35]])

    @pytest.mark.parametrize("sign", [-1, 1])
    @pytest.mark.parametrize("floor", [1e-8, 0.8])
    def test_ar1_m_step_matches_direct_optimization(self, sign, floor):
        q = 4
        distances = np.abs(np.arange(q)[:, None] - np.arange(q))
        moment = (sign * 0.7) ** distances + np.diag([0.2, 0, 0, 0.1])
        struct = RandomEffectStructure("g", list("abcd"), 1, q, True, {}, cov_type="ar1")
        covariance = _m_step_covariance(struct, moment, floor)

        def loss(parameters):
            scale, rho = parameters
            cov = scale * rho**distances
            return np.linalg.slogdet(cov)[1] + np.trace(linalg.solve(cov, moment))

        reference = optimize.minimize(
            loss,
            [1.5, 0.0],
            method="SLSQP",
            bounds=[(floor, None), (-0.999, 0.999)],
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda pars: (
                        np.linalg.eigvalsh(pars[0] * pars[1] ** distances)[0] - floor
                    ),
                }
            ],
            options={"ftol": 1e-12},
        )
        assert reference.success
        variance = covariance[0, 0]
        rho = covariance[0, 1] / variance
        np.testing.assert_allclose(covariance, variance * rho**distances)
        assert np.linalg.eigvalsh(covariance)[0] >= floor - 1e-12
        assert loss((variance, rho)) == pytest.approx(reference.fun, abs=1e-9)

    @pytest.mark.parametrize(
        "formula", ["Reaction ~ Days", "Reaction ~ 0", "Reaction ~ 0 + (1 | Subject)"]
    )
    def test_empty_design_blocks(self, formula):
        data = load_sleepstudy()
        result = em_reml_simple(lFormula(formula, data).matrices, tol=1e-12)
        fit = lmer(formula, data)
        assert result.final_loglik == pytest.approx(float(fit.logLik()), abs=1e-7)
        np.testing.assert_allclose(result.beta, fit.beta, atol=1e-6)

    def test_exact_fit_at_variance_floor(self):
        data = load_sleepstudy()
        data["Reaction"] = 0.0
        result = em_reml_simple(lFormula("Reaction ~ Days + (1 | Subject)", data).matrices)
        assert np.isfinite(result.final_loglik)
        assert result.sigma == pytest.approx(1e-4)


@pytest.mark.parametrize("kind", ["linear", "generalized"])
def test_modular_optimizers_use_em_start_unless_start_is_explicit(kind, monkeypatch):
    from unittest.mock import Mock

    from mixedlm import (
        families,
        glFormula,
        glmerControl,
        mkGlmerDevfun,
        mkLmerDevfun,
        optimizeGlmer,
        optimizeLmer,
    )

    data = load_sleepstudy()
    formula = "Reaction ~ Days + (1 | Subject)"
    if kind == "linear":
        devfun = mkLmerDevfun(
            lFormula(formula, data), control=lmerControl(em_init=True, em_maxiter=6)
        )
        optimize = optimizeLmer
    else:
        devfun = mkGlmerDevfun(
            glFormula(formula, data, family=families.Gaussian()),
            control=glmerControl(em_init=True, em_maxiter=6),
        )
        optimize = optimizeGlmer
    expected = em_reml_simple(devfun.parsed.matrices, max_iter=6)
    em = Mock(return_value=expected)
    monkeypatch.setattr("mixedlm.estimation.em_reml.em_reml_simple", em)
    np.testing.assert_array_equal(devfun.get_start(), expected.theta)
    em.reset_mock()
    result = optimize(devfun)
    em.assert_called_once_with(devfun.parsed.matrices, max_iter=6, verbose=0)
    em.reset_mock()
    explicit = optimize(devfun, start=expected.theta.copy())
    em.assert_not_called()
    np.testing.assert_allclose(result.theta, explicit.theta)
