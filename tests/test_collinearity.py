from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mixedlm import CollinearityResult, check_collinearity, families, glmer, lmer, nlme, nlmer
from mixedlm.families import Poisson


@dataclass
class _FakeLinearModel:
    matrices: SimpleNamespace

    def isLMM(self) -> bool:
        return True

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return False


@dataclass
class _FakeGeneralizedModel:
    matrices: SimpleNamespace
    family: Any
    eta: np.ndarray

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return True

    def isNLMM(self) -> bool:
        return False

    def linear_predictor(self) -> np.ndarray:
        return self.eta


@dataclass
class _GradientModel:
    values: np.ndarray
    param_names: tuple[str, ...]

    def gradient(self, phi: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.values


@dataclass
class _FakeNonlinearModel:
    model: _GradientModel
    phi: np.ndarray
    x: np.ndarray
    _weights: np.ndarray | None = None

    def isLMM(self) -> bool:
        return False

    def isGLMM(self) -> bool:
        return False

    def isNLMM(self) -> bool:
        return True


def _linear_model(
    design: np.ndarray,
    names: list[str],
    weights: np.ndarray | None = None,
) -> _FakeLinearModel:
    if weights is None:
        weights = np.ones(design.shape[0])
    return _FakeLinearModel(
        SimpleNamespace(X=np.asarray(design), fixed_names=names, weights=np.asarray(weights))
    )


def _orthogonal_basis() -> tuple[np.ndarray, np.ndarray]:
    x = np.tile(np.array([-1.0, -1.0, 1.0, 1.0]), 4)
    z = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), 4)
    return x, z


class TestCollinearityValues:
    def test_orthogonal_predictors_have_unit_vif(self) -> None:
        x, z = _orthogonal_basis()
        design = np.column_stack([np.ones(len(x)), x, z])

        result = check_collinearity(_linear_model(design, ["(Intercept)", "x", "z"]))

        assert isinstance(result, CollinearityResult)
        assert result.terms == ("x", "z")
        np.testing.assert_allclose(result.gvif, 1.0)
        np.testing.assert_allclose(result.adjusted_gvif, 1.0)
        np.testing.assert_allclose(result.vif, 1.0)
        np.testing.assert_allclose(result.tolerance, 1.0)
        np.testing.assert_allclose(result.condition_indices, 1.0)
        assert result.condition_number == pytest.approx(1.0)
        assert result.severity == ("low", "low")
        assert result.rank == result.n_columns == 2

    def test_correlated_predictors_match_closed_form(self) -> None:
        x, z = _orthogonal_basis()
        correlation = 0.8
        correlated = correlation * x + np.sqrt(1.0 - correlation**2) * z
        design = np.column_stack([np.ones(len(x)), x, correlated])

        result = check_collinearity(_linear_model(design, ["(Intercept)", "x", "correlated"]))

        expected_vif = 1.0 / (1.0 - correlation**2)
        expected_condition = np.sqrt((1.0 + correlation) / (1.0 - correlation))
        np.testing.assert_allclose(result.vif, expected_vif)
        np.testing.assert_allclose(result.tolerance, 1.0 / expected_vif)
        assert result.condition_number == pytest.approx(expected_condition)

    def test_is_invariant_to_predictor_scale(self) -> None:
        x, z = _orthogonal_basis()
        design = np.column_stack([np.ones(len(x)), x * 1e-12, z * 1e12])

        result = check_collinearity(_linear_model(design, ["(Intercept)", "small", "large"]))

        assert result.terms == ("small", "large")
        np.testing.assert_allclose(result.vif, 1.0)
        assert result.rank == result.n_columns == 2

    def test_multicolumn_term_reports_gvif_adjustments(self) -> None:
        d1 = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), 8)
        d2 = np.tile(np.array([0.0, 0.0, 1.0, 0.0]), 8)
        x = np.linspace(-1.0, 1.0, len(d1)) + 0.4 * d1 - 0.2 * d2
        design = np.column_stack([np.ones(len(x)), d1, d2, x])
        model = _linear_model(design, ["(Intercept)", "factor[B]", "factor[C]", "x"])

        result = check_collinearity(model)

        assert result.terms == ("factor", "x")
        assert result.df.tolist() == [2, 1]
        assert result.adjusted_gvif[0] == pytest.approx(result.gvif[0] ** 0.25)
        assert result.vif[0] == pytest.approx(np.sqrt(result.gvif[0]))
        assert result.vif[1] == pytest.approx(result.gvif[1])
        assert np.all(result.gvif >= 1.0)

    def test_interaction_columns_are_grouped(self) -> None:
        x, z = _orthogonal_basis()
        third = x * z
        fourth = np.roll(third, 1)
        design = np.column_stack([np.ones(len(x)), x, z, third, fourth])
        names = ["(Intercept)", "x", "z", "factor[B]:x", "factor[C]:x"]

        result = check_collinearity(_linear_model(design, names))

        assert result.terms == ("x", "z", "factor:x")
        assert result.df.tolist() == [1, 1, 2]

    def test_prior_weights_are_used(self) -> None:
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        z = np.array([-1.0, 1.0, 0.5, 1.5, 1.0])
        weights = np.array([1.0, 1.0, 2.0, 4.0, 8.0])
        design = np.column_stack([np.ones(len(x)), x, z])

        result = check_collinearity(_linear_model(design, ["(Intercept)", "x", "z"], weights))

        x_centered = x - np.average(x, weights=weights)
        z_centered = z - np.average(z, weights=weights)
        correlation = np.sum(weights * x_centered * z_centered) / np.sqrt(
            np.sum(weights * x_centered**2) * np.sum(weights * z_centered**2)
        )
        expected = 1.0 / (1.0 - correlation**2)
        np.testing.assert_allclose(result.vif, expected)
        assert result.weighting == "prior weights"

    def test_generalized_models_use_working_weights(self) -> None:
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        z = np.array([-1.0, 1.0, 0.5, 1.5, 1.0])
        design = np.column_stack([np.ones(len(x)), x, z])
        eta = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        prior = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        matrices = SimpleNamespace(
            X=design,
            fixed_names=["(Intercept)", "x", "z"],
            weights=prior,
        )
        model = _FakeGeneralizedModel(matrices, Poisson(), eta)

        result = check_collinearity(model)

        weights = prior * np.exp(eta)
        x_centered = x - np.average(x, weights=weights)
        z_centered = z - np.average(z, weights=weights)
        correlation = np.sum(weights * x_centered * z_centered) / np.sqrt(
            np.sum(weights * x_centered**2) * np.sum(weights * z_centered**2)
        )
        expected = 1.0 / (1.0 - correlation**2)
        np.testing.assert_allclose(result.vif, expected)
        assert result.weighting == "prior × working weights"

    def test_nonlinear_models_use_local_gradient(self) -> None:
        x, z = _orthogonal_basis()
        correlation = 0.6
        correlated = correlation * x + np.sqrt(1.0 - correlation**2) * z
        gradient = np.column_stack([x, correlated])
        model = _FakeNonlinearModel(
            model=_GradientModel(gradient, ("Asym", "lrc")),
            phi=np.array([1.0, 2.0]),
            x=np.arange(len(x)),
        )

        result = check_collinearity(model)

        np.testing.assert_allclose(result.vif, 1.0 / (1.0 - correlation**2))
        assert result.terms == ("Asym", "lrc")
        assert result.weighting == "local gradient with prior weights"

    def test_singular_design_reports_infinite_diagnostics(self) -> None:
        x = np.linspace(-1.0, 1.0, 20)
        design = np.column_stack([np.ones(len(x)), x, 2.0 * x])

        result = check_collinearity(_linear_model(design, ["(Intercept)", "x", "duplicate"]))

        assert result.rank == 1
        assert result.n_columns == 2
        assert np.all(np.isinf(result.gvif))
        assert np.all(np.isinf(result.vif))
        assert np.all(result.tolerance == 0.0)
        assert np.isinf(result.condition_number)
        assert result.problematic() == ("x", "duplicate")

    def test_intercept_only_model_returns_empty_result(self) -> None:
        design = np.ones((20, 1))

        result = check_collinearity(_linear_model(design, ["(Intercept)"]))

        assert len(result) == 0
        assert result.terms == ()
        assert result.condition_number == 1.0
        assert result.rank == result.n_columns == 0

    def test_custom_thresholds_set_severity(self) -> None:
        x, z = _orthogonal_basis()
        correlation = 0.8
        correlated = correlation * x + np.sqrt(1.0 - correlation**2) * z
        design = np.column_stack([np.ones(len(x)), x, correlated])

        result = check_collinearity(
            _linear_model(design, ["(Intercept)", "x", "correlated"]),
            moderate=2.0,
            high=3.0,
        )

        assert result.severity == ("moderate", "moderate")
        assert result.problematic(2.5) == ("x", "correlated")

    def test_dataframe_and_string_views(self) -> None:
        x, z = _orthogonal_basis()
        result = check_collinearity(
            _linear_model(np.column_stack([np.ones(len(x)), x, z]), ["(Intercept)", "x", "z"])
        )
        frame = result.to_dataframe()

        assert list(frame.columns) == [
            "term",
            "df",
            "GVIF",
            "GVIF^(1/(2*df))",
            "VIF",
            "tolerance",
            "severity",
        ]
        assert frame["term"].tolist() == ["x", "z"]
        assert "Condition number:" in str(result)
        assert "Tolerance" in str(result)


class TestCollinearityValidation:
    @pytest.mark.parametrize(
        ("moderate", "high"),
        [(1.0, 10.0), (5.0, 5.0), (10.0, 5.0), (np.nan, 10.0), (5.0, np.inf)],
    )
    def test_rejects_invalid_thresholds(self, moderate, high) -> None:
        with pytest.raises(ValueError, match="Thresholds|thresholds"):
            check_collinearity(
                _linear_model(np.column_stack([np.ones(4), np.arange(4)]), ["i", "x"]),
                moderate=moderate,
                high=high,
            )

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted linear"):
            check_collinearity(object())

    @pytest.mark.parametrize("weights", [np.array([1.0, 0.0]), np.array([1.0, np.nan])])
    def test_rejects_invalid_weights(self, weights) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            check_collinearity(_linear_model(np.eye(2), ["x", "z"], weights))

    def test_rejects_weight_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="weights has length"):
            check_collinearity(_linear_model(np.eye(2), ["x", "z"], np.ones(3)))

    def test_rejects_name_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="columns but"):
            check_collinearity(_linear_model(np.eye(2), ["x"]))

    def test_rejects_nonfinite_design(self) -> None:
        with pytest.raises(ValueError, match="finite values"):
            check_collinearity(_linear_model(np.array([[1.0], [np.nan]]), ["x"]))

    def test_rejects_invalid_problematic_threshold(self) -> None:
        x, z = _orthogonal_basis()
        result = check_collinearity(_linear_model(np.column_stack([x, z]), ["x", "z"]))

        for threshold in (0.0, -1.0, np.nan):
            with pytest.raises(ValueError, match="threshold"):
                result.problematic(threshold)


class TestCollinearityIntegration:
    def test_fitted_linear_model(self) -> None:
        rng = np.random.default_rng(630)
        groups = np.repeat(np.arange(10), 8)
        x = rng.normal(size=len(groups))
        z = 0.85 * x + rng.normal(scale=0.3, size=len(groups))
        y = (
            1.0
            + x
            - 0.5 * z
            + rng.normal(scale=0.5, size=10)[groups]
            + rng.normal(scale=0.4, size=len(groups))
        )
        data = pd.DataFrame({"y": y, "x": x, "z": z, "group": groups.astype(str)})
        model = lmer("y ~ x + z + (1 | group)", data)

        result = check_collinearity(model)

        assert result.terms == ("x", "z")
        assert np.all(result.vif > 1.0)
        assert np.all(np.isfinite(result.vif))

    def test_fitted_generalized_model(self) -> None:
        rng = np.random.default_rng(631)
        groups = np.repeat(np.arange(10), 10)
        x = rng.normal(size=len(groups))
        z = 0.7 * x + rng.normal(scale=0.5, size=len(groups))
        eta = -0.5 + 0.6 * x - 0.2 * z + rng.normal(scale=0.2, size=10)[groups]
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
        data = pd.DataFrame({"y": y, "x": x, "z": z, "group": groups.astype(str)})
        model = glmer("y ~ x + z + (1 | group)", data, family=families.Binomial())

        result = check_collinearity(model)

        assert result.terms == ("x", "z")
        assert np.all(result.vif >= 1.0)
        assert np.all(np.isfinite(result.vif))

    def test_fitted_nonlinear_model(self) -> None:
        rng = np.random.default_rng(633)
        rows = []
        for group in range(6):
            asymptote = 10.0 + rng.normal(scale=0.5)
            for time in np.linspace(0.0, 5.0, 8):
                response = asymptote - (asymptote - 2.0) * np.exp(-np.exp(-0.5) * time)
                rows.append(
                    {
                        "subject": str(group),
                        "time": time,
                        "y": response + rng.normal(scale=0.2),
                    }
                )
        data = pd.DataFrame(rows)
        model = nlmer(
            nlme.SSasymp(),
            data,
            x_var="time",
            y_var="y",
            group_var="subject",
            random_params=["Asym"],
        )

        result = check_collinearity(model)

        assert result.terms == tuple(model.model.param_names)
        assert np.all(result.vif >= 1.0)
        assert np.all(np.isfinite(result.vif))
