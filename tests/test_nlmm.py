from __future__ import annotations

import numpy as np
import pytest
from mixedlm.estimation.nlmm import (
    NLMMOptimizationResult,
    NLMMOptimizer,
    _build_psi_matrix,
    nlmm_deviance,
    pnls_step,
)
from mixedlm.nlme.models import SSasymp, SSlogis, SSmicmen
from numpy.testing import assert_allclose


@pytest.fixture
def simple_nlmm_data():
    np.random.seed(42)
    n_groups = 5
    n_per_group = 10
    n = n_groups * n_per_group

    groups = np.repeat(np.arange(n_groups), n_per_group)
    x = np.tile(np.linspace(0, 5, n_per_group), n_groups)

    Asym_base = 10.0
    R0_base = 0.5
    lrc_base = -0.5

    group_asym = np.repeat(np.random.randn(n_groups) * 1.0, n_per_group)

    model = SSasymp()
    y = np.zeros(n)
    for i in range(n):
        params = np.array([Asym_base + group_asym[i], R0_base, lrc_base])
        y[i] = model.predict(params, np.array([x[i]]))[0]

    y += np.random.randn(n) * 0.3

    return x, y, groups


class TestBuildPsiMatrix:
    def test_empty_theta(self):
        Psi = _build_psi_matrix(np.array([]), n_random=2)
        assert Psi.shape == (2, 2)
        assert_allclose(Psi, np.eye(2))

    def test_diagonal_from_theta(self):
        theta = np.array([2.0, 0.0, 3.0])
        Psi = _build_psi_matrix(theta, n_random=2)
        assert Psi.shape == (2, 2)
        assert Psi[0, 0] == pytest.approx(4.0, abs=1e-6)
        assert Psi[1, 1] == pytest.approx(9.0, abs=1e-6)

    def test_symmetric(self):
        theta = np.array([1.0, 0.5, 1.0])
        Psi = _build_psi_matrix(theta, n_random=2)
        assert_allclose(Psi, Psi.T)

    def test_positive_semidefinite(self):
        theta = np.array([1.0, 0.5, 1.0])
        Psi = _build_psi_matrix(theta, n_random=2)
        eigvals = np.linalg.eigvalsh(Psi)
        assert np.all(eigvals >= -1e-10)


class TestPnlsStep:
    def test_output_shapes(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        n_groups = len(np.unique(groups))
        n_random = 1
        random_params = [0]

        phi = model.get_start(x, y)
        b = np.zeros((n_groups, n_random))
        Psi = np.eye(n_random)
        sigma = 1.0

        phi_new, b_new, sigma_new = pnls_step(
            y, x, groups, model, phi, b, Psi, sigma, random_params
        )

        assert phi_new.shape == phi.shape
        assert b_new.shape == b.shape
        assert sigma_new > 0

    def test_reduces_residuals(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        n_groups = len(np.unique(groups))
        n_random = 1
        random_params = [0]

        phi = model.get_start(x, y)
        b = np.zeros((n_groups, n_random))
        Psi = np.eye(n_random)
        sigma = np.std(y)

        phi_new, b_new, sigma_new = pnls_step(
            y, x, groups, model, phi, b, Psi, sigma, random_params
        )

        rss_before = 0.0
        rss_after = 0.0
        for g in range(n_groups):
            mask = groups == g
            pred_before = model.predict(phi, x[mask])
            params_after = phi_new.copy()
            params_after[random_params] += b_new[g, :]
            pred_after = model.predict(params_after, x[mask])
            rss_before += np.sum((y[mask] - pred_before) ** 2)
            rss_after += np.sum((y[mask] - pred_after) ** 2)

        assert rss_after <= rss_before * 1.1


class TestNlmmDeviance:
    def test_returns_valid_deviance(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        n_groups = len(np.unique(groups))
        n_random = 1
        random_params = [0]

        phi = model.get_start(x, y)
        b = np.zeros((n_groups, n_random))
        theta = np.array([1.0])
        sigma = np.std(y)

        dev, phi_new, b_new, sigma_new = nlmm_deviance(
            theta, y, x, groups, model, phi, b, random_params, sigma
        )

        assert np.isfinite(dev)
        assert dev > 0

    def test_updates_parameters(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        n_groups = len(np.unique(groups))
        n_random = 1
        random_params = [0]

        phi = model.get_start(x, y)
        b = np.zeros((n_groups, n_random))
        theta = np.array([1.0])
        sigma = np.std(y)

        dev, phi_new, b_new, sigma_new = nlmm_deviance(
            theta, y, x, groups, model, phi, b, random_params, sigma
        )

        assert phi_new.shape == phi.shape
        assert sigma_new > 0


class TestNLMMOptimizationResult:
    def test_dataclass_fields(self):
        result = NLMMOptimizationResult(
            phi=np.array([1.0, 2.0, 3.0]),
            theta=np.array([0.5]),
            sigma=1.2,
            b=np.zeros((5, 1)),
            deviance=100.0,
            converged=True,
            n_iter=50,
        )
        assert len(result.phi) == 3
        assert result.sigma == 1.2
        assert result.converged


class TestNLMMOptimizer:
    def test_initialization(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)

        assert optimizer.n_groups == len(np.unique(groups))
        assert optimizer.n_random == 1

    def test_get_start_theta(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        theta = optimizer.get_start_theta()

        assert len(theta) > 0
        assert np.all(np.diag(_build_psi_matrix(theta, 1)) > 0)

    def test_get_start_phi(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        phi = optimizer.get_start_phi()

        assert len(phi) == model.n_params

    def test_objective_returns_scalar(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        theta = optimizer.get_start_theta()
        dev = optimizer.objective(theta)

        assert np.isscalar(dev)
        assert np.isfinite(dev)

    def test_optimize_converges(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        result = optimizer.optimize(maxiter=50)

        assert isinstance(result, NLMMOptimizationResult)
        assert result.deviance > 0


class TestNLMMWithDifferentModels:
    def test_with_sslogis(self):
        np.random.seed(42)
        n_groups = 4
        n_per_group = 8
        n = n_groups * n_per_group

        groups = np.repeat(np.arange(n_groups), n_per_group)
        x = np.tile(np.linspace(-3, 3, n_per_group), n_groups)

        model = SSlogis()
        params = np.array([10.0, 0.0, 1.0])
        y = model.predict(params, x) + np.random.randn(n) * 0.5

        random_params = [0]
        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        result = optimizer.optimize(maxiter=30)

        assert np.isfinite(result.deviance)

    def test_with_ssmicmen(self):
        np.random.seed(42)
        n_groups = 4
        n_per_group = 8
        n = n_groups * n_per_group

        groups = np.repeat(np.arange(n_groups), n_per_group)
        x = np.tile(np.linspace(0.1, 5, n_per_group), n_groups)

        model = SSmicmen()
        params = np.array([10.0, 2.0])
        y = model.predict(params, x) + np.random.randn(n) * 0.3

        random_params = [0]
        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        result = optimizer.optimize(maxiter=30)

        assert np.isfinite(result.deviance)


class TestNLMMEdgeCases:
    def test_single_group(self):
        np.random.seed(42)
        n = 20
        groups = np.zeros(n, dtype=int)
        x = np.linspace(0, 5, n)

        model = SSasymp()
        params = np.array([10.0, 0.5, -0.5])
        y = model.predict(params, x) + np.random.randn(n) * 0.3

        random_params = [0]
        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        result = optimizer.optimize(maxiter=20)

        assert np.isfinite(result.deviance)

    def test_multiple_random_effects(self, simple_nlmm_data):
        x, y, groups = simple_nlmm_data
        model = SSasymp()
        random_params = [0, 1]

        optimizer = NLMMOptimizer(y, x, groups, model, random_params)
        assert optimizer.n_random == 2
        assert optimizer.n_theta == 3
