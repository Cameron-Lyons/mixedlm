from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    ICCResult,
    R2NakagawaResult,
    families,
    glmer,
    icc,
    lmer,
    lmerControl,
    r2_nakagawa,
)
from mixedlm.diagnostics.fit_metrics import _distribution_specific_variance
from mixedlm.estimation.nlmm import _build_psi_matrix
from mixedlm.families import (
    Binomial,
    Gamma,
    Gaussian,
    InverseGaussian,
    NegativeBinomial,
    Poisson,
    QuasiFamily,
)
from mixedlm.families.base import LogLink
from mixedlm.nlme import SSmicmen
from numpy.testing import assert_allclose
from scipy import linalg, sparse


@dataclass
class _FakeLinearModel:
    beta: np.ndarray
    sigma: float
    matrices: SimpleNamespace
    structures: list[SimpleNamespace]
    covariances: list[np.ndarray]

    def isLMM(self) -> bool:
        return True

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return False

    def _iter_random_cov_blocks(self, scale: float = 1.0):
        for structure, covariance in zip(self.structures, self.covariances, strict=True):
            yield structure, covariance * scale


@dataclass
class _FakeGeneralizedModel:
    beta: np.ndarray
    u: np.ndarray
    family: object
    matrices: SimpleNamespace
    structures: list[SimpleNamespace]
    covariances: list[np.ndarray]

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return True

    def isNLMM(self) -> bool:
        return False

    def _iter_random_cov_blocks(self, scale: float = 1.0):
        for structure, covariance in zip(self.structures, self.covariances, strict=True):
            yield structure, covariance * scale


@dataclass
class _FakeNonlinearModel:
    model: object
    phi: np.ndarray
    theta: np.ndarray
    sigma: float
    x: np.ndarray
    random_params: list[int]
    group_var: str = "subject"

    def weights(self, copy: bool = True) -> np.ndarray:
        return np.ones(len(self.x))

    def offset(self, copy: bool = True) -> np.ndarray:
        return np.zeros(len(self.x))

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return True


@pytest.fixture
def linear_model() -> _FakeLinearModel:
    x = np.arange(4, dtype=np.float64)
    X = np.column_stack([np.ones(4), x])
    Z = sparse.csc_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        )
    )
    structure = SimpleNamespace(
        grouping_factor="subject",
        n_levels=2,
        n_terms=2,
    )
    matrices = SimpleNamespace(X=X, Z=Z, offset=np.zeros(4), weights=np.ones(4))
    relative_covariance = np.array([[0.25, 0.05], [0.05, 0.125]])
    return _FakeLinearModel(
        beta=np.array([1.0, 2.0]),
        sigma=2.0,
        matrices=matrices,
        structures=[structure],
        covariances=[relative_covariance],
    )


def _fixed_only_model(predictions, weights=None) -> _FakeLinearModel:
    n_obs = len(predictions)
    matrices = SimpleNamespace(
        X=np.asarray(predictions, dtype=np.float64)[:, None],
        Z=sparse.csc_matrix((n_obs, 0)),
        offset=np.zeros(n_obs),
        weights=np.ones(n_obs) if weights is None else np.asarray(weights, dtype=np.float64),
    )
    return _FakeLinearModel(
        beta=np.array([1.0]), sigma=1.0, matrices=matrices, structures=[], covariances=[]
    )


def _decimal_residual(family, means, weights, approximation) -> float:
    """Average link-scale residual variance from the distribution formulas."""
    with localcontext() as context:
        context.prec = 500
        base = getattr(family, "base_family", family)
        dispersion = Decimal(str(getattr(family, "phi", 1)))
        contributions = []
        for mean, weight in zip(means, weights, strict=True):
            mu = Decimal(str(mean))
            if isinstance(base, Gaussian):
                variance = Decimal(1)
            elif isinstance(base, Poisson):
                variance = mu
            elif isinstance(base, Gamma):
                variance = mu**2
            elif isinstance(base, InverseGaussian):
                variance = mu**3
            elif isinstance(base, NegativeBinomial):
                variance = mu + mu**2 / Decimal(str(base.theta))
            else:
                raise AssertionError("Unknown oracle distribution")
            ratio = dispersion * variance / (Decimal(str(weight)) * mu**2)
            contributions.append((1 + ratio).ln() if approximation == "lognormal" else ratio)
        return float(sum(contributions) / len(contributions))


def _as_glmm(linear_model: _FakeLinearModel, family: object) -> _FakeGeneralizedModel:
    return _FakeGeneralizedModel(
        beta=np.array([-1.0, 0.2]),
        u=np.zeros(linear_model.matrices.Z.shape[1]),
        family=family,
        matrices=linear_model.matrices,
        structures=linear_model.structures,
        covariances=linear_model.covariances,
    )


class TestNakagawaR2:
    def test_linear_random_slope_decomposition(self, linear_model) -> None:
        result = r2_nakagawa(linear_model)

        fixed = np.var(np.array([1.0, 3.0, 5.0, 7.0]), ddof=1)
        random = np.mean(np.array([1.0, 1.9, 3.8, 6.7]))
        residual = 4.0
        total = fixed + random + residual

        assert isinstance(result, R2NakagawaResult)
        assert result.variance_fixed == pytest.approx(fixed)
        assert result.variance_random == pytest.approx(random)
        assert result.variance_residual == residual
        assert result.random_by_group == {"subject": pytest.approx(random)}
        assert result.marginal == pytest.approx(fixed / total)
        assert result.conditional == pytest.approx((fixed + random) / total)
        assert result.approximation == "gaussian"

    @pytest.mark.parametrize("sigma", [0.1, 0.3, 3.0, 123.456])
    def test_unit_weights_keep_exact_residual_variance(self, linear_model, sigma) -> None:
        linear_model.sigma = sigma

        assert r2_nakagawa(linear_model).variance_residual == sigma**2

    def test_fixed_offset_contributes_to_explained_variance(self, linear_model) -> None:
        linear_model.beta = np.array([2.0, 0.0])
        linear_model.matrices.offset = np.array([0.0, 1.0, 2.0, 3.0])

        result = r2_nakagawa(linear_model)

        assert result.variance_fixed == pytest.approx(np.var([2.0, 3.0, 4.0, 5.0], ddof=1))

    def test_no_random_effects_makes_conditional_equal_marginal(self, linear_model) -> None:
        linear_model.matrices.Z = sparse.csc_matrix((4, 0))
        linear_model.structures = []
        linear_model.covariances = []

        result = r2_nakagawa(linear_model)

        assert result.conditional == pytest.approx(result.marginal)
        assert result.variance_random == 0.0
        assert result.random_by_group == {}

    def test_crossed_groups_partition_random_variance(self, linear_model) -> None:
        item_structure = SimpleNamespace(grouping_factor="item", n_levels=2, n_terms=1)
        item_design = sparse.csc_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
        linear_model.matrices.Z = sparse.hstack(
            [linear_model.matrices.Z, item_design], format="csc"
        )
        linear_model.structures.append(item_structure)
        linear_model.covariances.append(np.array([[0.5]]))

        result = r2_nakagawa(linear_model)

        assert result.random_by_group["subject"] == pytest.approx(3.35)
        assert result.random_by_group["item"] == pytest.approx(2.0)
        assert sum(result.random_by_group.values()) == pytest.approx(result.variance_random)

    def test_poisson_lognormal_residual_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        result = r2_nakagawa(model)

        eta = model.matrices.X @ model.beta
        mu = np.exp(eta)
        expected = np.mean(np.log1p(1.0 / mu))
        assert result.variance_residual == pytest.approx(expected)
        assert result.approximation == "lognormal"

    def test_poisson_delta_residual_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        result = r2_nakagawa(model, approximation="delta")

        mu = np.exp(model.matrices.X @ model.beta)
        assert result.variance_residual == pytest.approx(np.mean(1.0 / mu))
        assert result.approximation == "delta"

    def test_binomial_uses_link_theoretical_variance(self, linear_model) -> None:
        model = _as_glmm(linear_model, Binomial())

        result = r2_nakagawa(model)

        assert result.variance_residual == pytest.approx(np.pi**2 / 3.0)
        assert result.approximation == "theoretical"

    def test_nonlinear_delta_random_variance(self) -> None:
        nonlinear = SSmicmen()
        x = np.array([0.5, 1.0, 2.0, 4.0])
        phi = np.array([10.0, 2.0])
        theta = np.array([0.2, 0.05, 0.1])
        model = _FakeNonlinearModel(
            model=nonlinear,
            phi=phi,
            theta=theta,
            sigma=0.5,
            x=x,
            random_params=[0, 1],
        )

        result = r2_nakagawa(model)

        gradient = nonlinear.gradient(phi, x)
        covariance = _build_psi_matrix(theta, 2) * model.sigma**2
        expected_random = np.mean(np.sum((gradient @ covariance) * gradient, axis=1))
        expected_fixed = np.var(nonlinear.predict(phi, x), ddof=1)
        assert result.variance_fixed == pytest.approx(expected_fixed)
        assert result.variance_random == pytest.approx(expected_random)
        assert result.variance_residual == pytest.approx(0.25)
        assert result.approximation == "gaussian-delta"


class TestICC:
    def test_adjusted_and_unadjusted_icc(self, linear_model) -> None:
        result = icc(linear_model)

        fixed = np.var(np.array([1.0, 3.0, 5.0, 7.0]), ddof=1)
        random = np.mean(np.array([1.0, 1.9, 3.8, 6.7]))
        residual = 4.0
        assert isinstance(result, ICCResult)
        assert result.adjusted == pytest.approx(random / (random + residual))
        assert result.unadjusted == pytest.approx(random / (fixed + random + residual))
        assert result.by_group["subject"] == pytest.approx(result.adjusted)
        assert result.by_group_unadjusted["subject"] == pytest.approx(result.unadjusted)

    def test_no_random_effects_returns_zero(self, linear_model) -> None:
        linear_model.matrices.Z = sparse.csc_matrix((4, 0))
        linear_model.structures = []
        linear_model.covariances = []

        result = icc(linear_model)

        assert result.adjusted == 0.0
        assert result.unadjusted == 0.0
        assert result.by_group == {}


@pytest.mark.installed_wheel
class TestVarianceOracles:
    """Variance components checked against the covariance of observed responses."""

    @pytest.mark.parametrize("use_rust", [False, True])
    def test_weighted_crossed_fit_matches_observation_covariance(self, use_rust) -> None:
        rng = np.random.default_rng(72)
        subject = np.repeat(np.arange(8), 8)
        item = np.tile(np.arange(8), 8)
        x = rng.normal(size=64)
        precision = np.exp(rng.normal(scale=0.6, size=64))
        y = (
            1.2
            + 0.7 * x
            + rng.normal(scale=0.8, size=8)[subject]
            + rng.normal(scale=0.4, size=8)[item]
            + rng.normal(scale=0.5, size=64) / np.sqrt(precision)
        )
        data = pd.DataFrame({"y": y, "x": x, "subject": subject, "item": item})
        model = lmer(
            "y ~ x + (1 | subject) + (1 | item)",
            data,
            weights=precision,
            control=lmerControl(use_rust=use_rust),
        )
        assert model.converged

        # Independent observation covariance: V = Z G Z' + sigma² W^-1.
        blocks = [
            np.kron(np.eye(structure.n_levels), covariance)
            for structure, covariance in model._iter_random_cov_blocks(scale=model.sigma**2)
        ]
        design = model.matrices.Z.toarray()
        random_covariance = design @ linalg.block_diag(*blocks) @ design.T
        fixed = np.var(model.matrices.X @ model.beta, ddof=1)
        random = np.trace(random_covariance) / len(data)
        residual = np.trace(np.diag(model.sigma**2 / precision)) / len(data)
        total = fixed + random + residual

        result = r2_nakagawa(model)
        correlation = icc(model)
        assert result.variance_residual == pytest.approx(residual)
        assert result.variance_random == pytest.approx(random)
        assert result.marginal == pytest.approx(fixed / total)
        assert result.conditional == pytest.approx((fixed + random) / total)
        assert correlation.adjusted == pytest.approx(random / (random + residual))
        assert sum(correlation.by_group.values()) == pytest.approx(correlation.adjusted)

        # The same response distribution can be parameterized with a common
        # precision multiplier, compensated by sigma and the relative covariance.
        scaled = replace(
            model,
            sigma=model.sigma * np.sqrt(16.0),
            theta=model.theta / np.sqrt(16.0),
            matrices=replace(model.matrices, weights=16.0 * model.matrices.weights),
        )
        assert r2_nakagawa(scaled).as_dict() == pytest.approx(result.as_dict())
        assert icc(scaled).as_dict() == pytest.approx(correlation.as_dict())

    def test_multi_term_random_slope_matches_observation_covariance(self) -> None:
        rng = np.random.default_rng(6)
        n_obs, n_terms, n_levels = 11, 5, 3
        term_design = rng.normal(size=(n_obs, n_terms))
        levels = np.arange(n_obs) % n_levels
        design = np.zeros((n_obs, n_levels * n_terms))
        for row, level in enumerate(levels):
            design[row, level * n_terms : (level + 1) * n_terms] = term_design[row]
        factor = rng.normal(size=(n_terms, n_terms))
        covariance = factor @ factor.T
        model = _FakeLinearModel(
            beta=np.array([0.0]),
            sigma=1.0,
            matrices=SimpleNamespace(
                X=np.ones((n_obs, 1)),
                Z=sparse.csc_matrix(design),
                offset=np.zeros(n_obs),
                weights=np.ones(n_obs),
            ),
            structures=[
                SimpleNamespace(grouping_factor="group", n_terms=n_terms, n_levels=n_levels)
            ],
            covariances=[covariance],
        )

        expected = np.trace(design @ np.kron(np.eye(n_levels), covariance) @ design.T) / n_obs
        assert r2_nakagawa(model).variance_random == pytest.approx(expected)

    def test_nonlinear_residual_component_honors_precision(self) -> None:
        class WeightedModel(_FakeNonlinearModel):
            def weights(self, copy=True):
                return np.array([0.25, 1.0, 4.0, 16.0])

        model = WeightedModel(
            model=SSmicmen(),
            phi=np.array([10.0, 2.0]),
            theta=np.array([0.2]),
            sigma=2.0,
            x=np.array([0.5, 1.0, 2.0, 4.0]),
            random_params=[0],
        )

        expected = (16.0 + 4.0 + 1.0 + 0.25) / 4
        assert r2_nakagawa(model).variance_residual == pytest.approx(expected)
        assert icc(model).variance_residual == pytest.approx(expected)

    def test_nonlinear_offsets_contribute_to_fixed_prediction_variance(self) -> None:
        class OffsetModel(_FakeNonlinearModel):
            def offset(self, copy=True):
                # Known offsets exactly cancel the population Michaelis-Menten curve.
                return -np.array([2.0, 10.0 / 3.0, 5.0, 20.0 / 3.0])

        model = OffsetModel(
            model=SSmicmen(),
            phi=np.array([10.0, 2.0]),
            theta=np.array([0.2]),
            sigma=2.0,
            x=np.array([0.5, 1.0, 2.0, 4.0]),
            random_params=[0],
        )

        result = r2_nakagawa(model)
        assert result.variance_fixed == pytest.approx(0.0, abs=1e-25)
        assert result.marginal == pytest.approx(0.0, abs=1e-25)
        assert result.variance_random > 0

    def test_large_common_baseline_keeps_small_fixed_variation(self) -> None:
        model = _fixed_only_model(1e15 + np.array([0.0, 0.25, 0.5, 0.75]))

        expected = (0.375**2 + 0.125**2 + 0.125**2 + 0.375**2) / 3
        assert r2_nakagawa(model).variance_fixed == pytest.approx(expected)


@pytest.mark.installed_wheel
class TestLogLinkResidualVariance:
    """Extreme GLMM variances verified against decimal distribution formulas."""

    @pytest.mark.parametrize(
        "family",
        [
            Gaussian(link="log"),
            Poisson(),
            Gamma(),
            InverseGaussian(),
            NegativeBinomial(theta=4),
            QuasiFamily(Gamma(), phi=2.5),
        ],
    )
    @pytest.mark.parametrize("mean", [1e-150, 0.7, 1e150, 1e200])
    @pytest.mark.parametrize("approximation", ["lognormal", "delta"])
    def test_builtin_log_variance_matches_high_precision_distribution(
        self, family, mean, approximation
    ) -> None:
        means = np.array([mean, 2 * mean])
        weights = np.array([2.0, 8.0])
        expected = _decimal_residual(family, means, weights, approximation)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual, used = _distribution_specific_variance(family, means, weights, approximation)

        assert used == approximation
        assert_allclose(actual, expected, rtol=2e-12, atol=0)

    def test_custom_variance_override_is_preserved(self) -> None:
        class CustomPoisson(Poisson):
            def variance(self, mu):
                return 7 * mu

        actual, used = _distribution_specific_variance(
            CustomPoisson(), np.array([2.0, 4.0]), np.array([1.0, 2.0]), "lognormal"
        )

        assert used == "lognormal"
        assert actual == pytest.approx((np.log1p(3.5) + np.log1p(0.875)) / 2)

    def test_custom_log_link_derivative_is_preserved(self) -> None:
        class ScaledLogLink(LogLink):
            def link(self, mu):
                return 2 * np.log(mu)

            def inverse(self, eta):
                return np.exp(eta / 2)

            def deriv(self, mu):
                return 2 / mu

        actual, used = _distribution_specific_variance(
            Poisson(link=ScaledLogLink()), np.array([2.0, 4.0]), np.array([1.0, 2.0]), "delta"
        )

        assert used == "delta"
        assert actual == pytest.approx((4 / 2 + 4 / (2 * 4)) / 2)


class TestFitMetricValidation:
    def test_rejects_unknown_approximation(self, linear_model) -> None:
        with pytest.raises(ValueError, match="Unknown approximation"):
            r2_nakagawa(linear_model, approximation="unknown")

    def test_rejects_theoretical_for_non_binomial(self, linear_model) -> None:
        model = _as_glmm(linear_model, Poisson())

        with pytest.raises(ValueError, match="only available for binomial"):
            r2_nakagawa(model, approximation="theoretical")

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted linear"):
            r2_nakagawa(object())

    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_predictions(self, invalid) -> None:
        with pytest.raises(ValueError, match="Fixed predictions must be finite"):
            r2_nakagawa(_fixed_only_model([1.0, invalid]))

    @pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_invalid_gaussian_precision(self, invalid) -> None:
        with pytest.raises(ValueError, match="weights must be finite and positive"):
            r2_nakagawa(_fixed_only_model([1.0, 1.0], weights=[1.0, invalid]))

    def test_rejects_unrepresentable_delta_variance(self) -> None:
        with pytest.raises(ValueError, match="Delta-method residual variance is not finite"):
            _distribution_specific_variance(
                Gaussian(link="log"), np.array([1e-200]), np.ones(1), "delta"
            )

    def test_result_helpers(self, linear_model) -> None:
        r2 = r2_nakagawa(linear_model)
        correlation = icc(linear_model)

        assert r2.as_dict()["marginal"] == r2.marginal
        assert correlation.as_dict()["adjusted"] == correlation.adjusted
        assert "marginal=" in str(r2)
        assert "adjusted=" in str(correlation)


class TestFittedModelIntegration:
    def test_linear_result(self) -> None:
        rng = np.random.default_rng(510)
        groups = np.repeat(np.arange(8), 8)
        x = rng.normal(size=len(groups))
        group_effect = rng.normal(scale=0.8, size=8)
        y = 1.0 + 0.7 * x + group_effect[groups] + rng.normal(scale=0.4, size=len(groups))
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        model = lmer("y ~ x + (1 | group)", data)

        r2 = r2_nakagawa(model)
        correlation = icc(model)

        assert 0.0 <= r2.marginal <= r2.conditional <= 1.0
        assert 0.0 <= correlation.unadjusted <= correlation.adjusted <= 1.0

    def test_generalized_result(self) -> None:
        rng = np.random.default_rng(511)
        groups = np.repeat(np.arange(10), 10)
        x = rng.normal(size=len(groups))
        eta = -0.6 + 0.5 * x + rng.normal(scale=1.0, size=10)[groups]
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        model = glmer("y ~ x + (1 | group)", data, family=families.Binomial())
        assert not model.isSingular()

        r2 = r2_nakagawa(model)
        correlation = icc(model)

        assert r2.approximation == "theoretical"
        assert 0.0 < r2.marginal < r2.conditional < 1.0
        assert 0.0 < correlation.unadjusted < correlation.adjusted < 1.0
