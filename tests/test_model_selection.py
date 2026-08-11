from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mixedlm import ModelSelectionResult, lmer, model_selection
from mixedlm.families import NegativeBinomial, Poisson
from mixedlm.families.custom import QuasiFamily


@dataclass(frozen=True)
class _Formula:
    response: str
    label: str

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class _LogLik:
    value: float
    df: int
    nobs: int
    REML: bool = False


@dataclass
class _FakeModel:
    label: str
    ll: float
    k: int
    n: int = 100
    kind: str = "LMM"
    response: str = "y"
    REML: bool = False
    family: Any = None
    y: Any = None
    _weights: Any = None
    fixed_design: Any = None

    @property
    def formula(self) -> _Formula:
        return _Formula(self.response, self.label)

    def extractAIC(self) -> tuple[float, float]:
        return float(self.k), -2.0 * self.ll + 2.0 * self.k

    def logLik(self) -> _LogLik:
        return _LogLik(self.ll, self.k, self.n, self.REML)

    def BIC(self) -> float:
        return -2.0 * self.ll + self.k * np.log(self.n)

    def isLMM(self) -> bool:
        return self.kind == "LMM"

    def isGLMM(self) -> bool:
        return self.kind == "GLMM"

    def isNLMM(self) -> bool:
        return self.kind == "NLMM"


class _FakeNonlinearResult(_FakeModel):
    def logLik(self) -> float:
        return self.ll

    def nobs(self) -> int:
        return self.n


@pytest.fixture
def candidates() -> tuple[_FakeModel, _FakeModel, _FakeModel]:
    return (
        _FakeModel("y ~ 1 + (1 | g)", ll=-100.0, k=3),
        _FakeModel("y ~ x + (1 | g)", ll=-98.0, k=5),
        _FakeModel("y ~ x + z + (1 | g)", ll=-97.0, k=7),
    )


class TestModelSelectionRanking:
    def test_aicc_values_weights_and_order(self, candidates) -> None:
        result = model_selection(*candidates)

        expected_aic = np.array([206.0, 206.0, 208.0])
        expected_correction = np.array([24.0 / 96.0, 60.0 / 94.0, 112.0 / 92.0])
        expected_aicc = expected_aic + expected_correction
        expected_delta = expected_aicc - expected_aicc.min()
        expected_relative = np.exp(-0.5 * expected_delta)
        expected_weight = expected_relative / expected_relative.sum()

        assert isinstance(result, ModelSelectionResult)
        assert result.criterion == "AICc"
        assert result.likelihood == "ML"
        assert result.models == candidates
        assert result.best_model is candidates[0]
        assert result.best_name == "y ~ 1 + (1 | g)"
        np.testing.assert_allclose(result.aicc, expected_aicc)
        np.testing.assert_allclose(result.delta, expected_delta)
        np.testing.assert_allclose(result.relative_likelihood, expected_relative)
        np.testing.assert_allclose(result.weight, expected_weight)
        np.testing.assert_allclose(result.cumulative_weight, np.cumsum(expected_weight))
        assert result.cumulative_weight[-1] == pytest.approx(1.0)

    def test_ranks_by_requested_criterion(self, candidates) -> None:
        aic = model_selection(*candidates, criterion="aic")
        bic = model_selection(*candidates, criterion="BIC")

        assert aic.criterion == "AIC"
        assert aic.models[:2] == candidates[:2]
        assert aic.weight[0] == pytest.approx(aic.weight[1])
        assert bic.best_model is candidates[0]
        assert np.all(np.diff(bic.bic) >= 0.0)

    def test_dataframe_and_string_views(self, candidates) -> None:
        result = model_selection(*candidates, names=["base", "x", "x + z"])
        frame = result.to_dataframe()

        assert list(frame.columns) == [
            "model",
            "nobs",
            "df",
            "logLik",
            "AIC",
            "AICc",
            "BIC",
            "delta",
            "relative_likelihood",
            "weight",
            "cumulative_weight",
        ]
        assert frame["model"].tolist() == ["base", "x", "x + z"]
        assert len(result) == 3
        assert "Model selection by AICc (ML)" in str(result)
        assert "cum.weight" in str(result)

    def test_evidence_set_includes_threshold_crossing_model(self, candidates) -> None:
        result = model_selection(*candidates, names=["base", "x", "x + z"])
        threshold = float(result.cumulative_weight[0] + 1e-6)

        assert result.evidence_set(threshold) == candidates[:2]
        assert result.evidence_names(threshold) == ("base", "x")
        assert result.evidence_set(1.0) == candidates

    def test_duplicate_formula_names_are_disambiguated(self) -> None:
        first = _FakeModel("same formula", ll=-10.0, k=2)
        second = _FakeModel("same formula", ll=-11.0, k=2)

        result = model_selection(first, second, criterion="AIC")

        assert set(result.model_names) == {"same formula [1]", "same formula [2]"}

    def test_small_sample_marks_only_undefined_aicc_as_infinite(self) -> None:
        supported = _FakeModel("supported", ll=-20.0, k=2, n=8)
        saturated = _FakeModel("saturated", ll=-10.0, k=7, n=8)

        result = model_selection(saturated, supported)

        assert result.best_model is supported
        assert np.isfinite(result.aicc[0])
        assert np.isinf(result.aicc[1])
        assert result.weight.tolist() == [1.0, 0.0]

    def test_supports_nonlinear_result_interface(self) -> None:
        response = np.arange(20, dtype=np.float64)
        first = _FakeNonlinearResult("nonlinear one", ll=-30.0, k=5, n=20, kind="NLMM", y=response)
        second = _FakeNonlinearResult("nonlinear two", ll=-28.0, k=6, n=20, kind="NLMM", y=response)

        result = model_selection(first, second)

        assert result.likelihood == "ML"
        assert result.models == (first, second)


class TestModelSelectionValidation:
    def test_requires_multiple_models(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            model_selection(_FakeModel("one", ll=-10.0, k=2))

    @pytest.mark.parametrize("criterion", ["QAIC", "", "AIC-c"])
    def test_rejects_unknown_criterion(self, criterion) -> None:
        with pytest.raises(ValueError, match="criterion"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=2),
                _FakeModel("two", ll=-9.0, k=3),
                criterion=criterion,
            )

    def test_validates_names(self, candidates) -> None:
        with pytest.raises(ValueError, match="one label per model"):
            model_selection(*candidates, names="abc")
        with pytest.raises(ValueError, match="length"):
            model_selection(*candidates, names=["one"])
        with pytest.raises(ValueError, match="unique"):
            model_selection(*candidates, names=["same", "same", "third"])
        with pytest.raises(ValueError, match="empty"):
            model_selection(*candidates, names=["first", " ", "third"])

    def test_rejects_different_observation_counts(self) -> None:
        with pytest.raises(ValueError, match="observation counts"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=2, n=30),
                _FakeModel("two", ll=-9.0, k=3, n=31),
            )

    def test_rejects_different_model_kinds(self) -> None:
        with pytest.raises(ValueError, match="likelihood classes"):
            model_selection(
                _FakeModel("linear", ll=-10.0, k=2),
                _FakeModel("nonlinear", ll=-9.0, k=3, kind="NLMM"),
            )

    def test_rejects_different_responses(self) -> None:
        with pytest.raises(ValueError, match="different responses"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=2, response="y"),
                _FakeModel("two", ll=-9.0, k=3, response="z"),
            )

    def test_rejects_different_response_observations(self) -> None:
        with pytest.raises(ValueError, match="response observations"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=2, n=3, y=np.array([1.0, 2.0, 3.0])),
                _FakeModel("two", ll=-9.0, k=3, n=3, y=np.array([1.0, 2.0, 4.0])),
            )

    def test_rejects_different_prior_weights(self) -> None:
        response = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="prior weights"):
            model_selection(
                _FakeModel(
                    "one",
                    ll=-10.0,
                    k=2,
                    n=3,
                    y=response,
                    _weights=np.ones(3),
                ),
                _FakeModel(
                    "two",
                    ll=-9.0,
                    k=3,
                    n=3,
                    y=response,
                    _weights=np.array([1.0, 1.0, 2.0]),
                ),
            )

    def test_rejects_different_generalized_families(self) -> None:
        with pytest.raises(ValueError, match="same family"):
            model_selection(
                _FakeModel("poisson", ll=-10.0, k=2, kind="GLMM", family=Poisson()),
                _FakeModel(
                    "negative binomial",
                    ll=-9.0,
                    k=3,
                    kind="GLMM",
                    family=NegativeBinomial(),
                ),
            )

    def test_rejects_different_family_parameters(self) -> None:
        with pytest.raises(ValueError, match="family parameters"):
            model_selection(
                _FakeModel("theta one", ll=-10.0, k=2, kind="GLMM", family=NegativeBinomial(1.0)),
                _FakeModel("theta two", ll=-9.0, k=3, kind="GLMM", family=NegativeBinomial(2.0)),
            )

    def test_rejects_quasi_likelihood(self) -> None:
        quasi = QuasiFamily(Poisson(), phi=2.0)
        with pytest.raises(ValueError, match="quasi-likelihood"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=2, kind="GLMM", family=quasi),
                _FakeModel("two", ll=-9.0, k=3, kind="GLMM", family=quasi),
            )

    def test_reml_requires_explicit_opt_in(self) -> None:
        design = np.ones((100, 2))
        first = _FakeModel("one", ll=-10.0, k=2, REML=True, fixed_design=design)
        second = _FakeModel("two", ll=-9.0, k=3, REML=True, fixed_design=design)

        with pytest.raises(ValueError, match="REML criteria"):
            model_selection(first, second)

        result = model_selection(first, second, allow_reml=True)
        assert result.likelihood == "REML"

    def test_reml_requires_identical_fixed_designs(self) -> None:
        with pytest.raises(ValueError, match="identical fixed-effects designs"):
            model_selection(
                _FakeModel(
                    "one",
                    ll=-10.0,
                    k=2,
                    REML=True,
                    fixed_design=np.ones((100, 1)),
                ),
                _FakeModel(
                    "two",
                    ll=-9.0,
                    k=3,
                    REML=True,
                    fixed_design=np.ones((100, 2)),
                ),
                allow_reml=True,
            )

    def test_rejects_mixed_ml_and_reml(self) -> None:
        with pytest.raises(ValueError, match="ML and REML"):
            model_selection(
                _FakeModel("ml", ll=-10.0, k=2),
                _FakeModel("reml", ll=-9.0, k=3, REML=True),
                allow_reml=True,
            )

    def test_rejects_invalid_evidence_threshold(self, candidates) -> None:
        result = model_selection(*candidates)

        for threshold in (0.0, 1.1, np.nan):
            with pytest.raises(ValueError, match="threshold"):
                result.evidence_set(threshold)

    def test_rejects_all_undefined_aicc(self) -> None:
        with pytest.raises(ValueError, match="No model has a finite AICc"):
            model_selection(
                _FakeModel("one", ll=-10.0, k=5, n=6),
                _FakeModel("two", ll=-9.0, k=6, n=6),
            )

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError, match="fitted linear"):
            model_selection(object(), object())


class TestModelSelectionIntegration:
    def test_ranks_fitted_linear_models(self) -> None:
        rng = np.random.default_rng(620)
        groups = np.repeat(np.arange(10), 8)
        x = rng.normal(size=len(groups))
        y = (
            1.0
            + 0.8 * x
            + rng.normal(scale=0.5, size=10)[groups]
            + rng.normal(scale=0.4, size=len(groups))
        )
        data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
        intercept = lmer("y ~ 1 + (1 | group)", data, REML=False)
        slope = lmer("y ~ x + (1 | group)", data, REML=False)

        result = model_selection(intercept, slope, names=["intercept", "slope"])

        assert result.best_model is slope
        assert result.best_name == "slope"
        assert result.delta[0] == pytest.approx(0.0)
        assert result.weight[0] > 0.99
