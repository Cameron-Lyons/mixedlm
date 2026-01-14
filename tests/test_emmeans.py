from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.inference.emmeans import (
    ContrastResult,
    EmmeanResult,
    _adjust_pvalues,
    emmeans,
)
from numpy.testing import assert_allclose


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n_groups = 5
    n_per_group = 20
    n = n_groups * n_per_group

    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_per_group)
    treatment = np.tile(["A", "B"], n // 2)

    group_effects = np.repeat(np.random.randn(n_groups), n_per_group)
    treatment_effect = np.where(np.array(treatment) == "B", 2.0, 0.0)

    y = 5.0 + treatment_effect + group_effects + np.random.randn(n) * 0.5

    return pd.DataFrame({"y": y, "treatment": treatment, "group": groups})


@pytest.fixture
def lmer_result(simple_data):
    return lmer("y ~ treatment + (1|group)", simple_data)


class TestAdjustPvalues:
    def test_none_adjustment(self):
        p = np.array([0.01, 0.05, 0.10])
        adjusted = _adjust_pvalues(p, "none", 3, 100, None)
        assert_allclose(adjusted, p)

    def test_bonferroni(self):
        p = np.array([0.01, 0.05, 0.10])
        adjusted = _adjust_pvalues(p, "bonferroni", 3, 100, None)
        assert_allclose(adjusted, [0.03, 0.15, 0.30])

    def test_bonferroni_clipping(self):
        p = np.array([0.5, 0.7])
        adjusted = _adjust_pvalues(p, "bonferroni", 2, 100, None)
        assert np.all(adjusted <= 1.0)

    def test_holm(self):
        p = np.array([0.01, 0.04, 0.06])
        adjusted = _adjust_pvalues(p, "holm", 3, 100, None)
        assert adjusted[0] <= adjusted[1]
        assert adjusted[1] <= adjusted[2]

    def test_fdr(self):
        p = np.array([0.01, 0.04, 0.06])
        adjusted = _adjust_pvalues(p, "fdr", 3, 100, None)
        assert adjusted[0] <= p[0] * 3

    def test_tukey(self):
        p = np.array([0.01])
        t_ratio = np.array([3.0])
        adjusted = _adjust_pvalues(p, "tukey", 3, 100, t_ratio)
        assert len(adjusted) == 1
        assert 0 <= adjusted[0] <= 1


class TestEmmeanResult:
    def test_str_method(self):
        result = EmmeanResult(
            emmean=np.array([1.0, 2.0]),
            se=np.array([0.1, 0.2]),
            df=50.0,
            lower=np.array([0.8, 1.6]),
            upper=np.array([1.2, 2.4]),
            grid=pd.DataFrame({"treatment": ["A", "B"]}),
            level=0.95,
        )
        s = str(result)
        assert "Estimated Marginal Means" in s
        assert "emmean" in s

    def test_repr_method(self):
        result = EmmeanResult(
            emmean=np.array([1.0, 2.0]),
            se=np.array([0.1, 0.2]),
            df=50.0,
            lower=np.array([0.8, 1.6]),
            upper=np.array([1.2, 2.4]),
            grid=pd.DataFrame({"treatment": ["A", "B"]}),
            level=0.95,
        )
        r = repr(result)
        assert "EmmeanResult" in r
        assert "n=2" in r


class TestContrastResult:
    def test_str_method(self):
        result = ContrastResult(
            contrast=["A - B"],
            estimate=np.array([-1.0]),
            se=np.array([0.2]),
            df=50.0,
            t_ratio=np.array([-5.0]),
            p_value=np.array([0.001]),
            adjust="none",
        )
        s = str(result)
        assert "Pairwise Comparisons" in s
        assert "A - B" in s

    def test_repr_method(self):
        result = ContrastResult(
            contrast=["A - B"],
            estimate=np.array([-1.0]),
            se=np.array([0.2]),
            df=50.0,
            t_ratio=np.array([-5.0]),
            p_value=np.array([0.001]),
            adjust="none",
        )
        r = repr(result)
        assert "ContrastResult" in r


class TestEmmeans:
    def test_emmeans_basic(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        assert em.result.emmean is not None
        assert len(em.result.emmean) == 2

    def test_emmeans_grid(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        assert "treatment" in em.result.grid.columns
        assert set(em.result.grid["treatment"]) == {"A", "B"}

    def test_emmeans_confidence_intervals(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        assert np.all(em.result.lower < em.result.emmean)
        assert np.all(em.result.upper > em.result.emmean)

    def test_emmeans_str(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        s = str(em)
        assert "Estimated Marginal Means" in s

    def test_emmeans_repr(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        r = repr(em)
        assert "Emmeans" in r


class TestEmmeansPairs:
    def test_pairs_basic(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        pairs = em.pairs()
        assert len(pairs.contrast) == 1
        assert len(pairs.estimate) == 1

    def test_pairs_adjustment(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        pairs_none = em.pairs(adjust="none")
        pairs_bonf = em.pairs(adjust="bonferroni")
        assert pairs_bonf.adjust == "bonferroni"
        assert pairs_none.p_value[0] <= pairs_bonf.p_value[0]


class TestEmmeansContrast:
    def test_pairwise_contrast(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        contrast = em.contrast(method="pairwise")
        assert len(contrast.contrast) >= 1

    def test_trt_vs_ctrl(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        contrast = em.contrast(method="trt.vs.ctrl")
        assert len(contrast.contrast) == 1

    def test_custom_contrast(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        C = np.array([[1, -1]])
        contrast = em.contrast(method=C)
        assert len(contrast.contrast) == 1
        assert contrast.contrast[0] == "C1"

    def test_invalid_method_raises(self, lmer_result, simple_data):
        em = emmeans(lmer_result, "treatment")
        with pytest.raises(ValueError, match="Unknown contrast method"):
            em.contrast(method="invalid")


class TestEmmeansEdgeCases:
    def test_invalid_factor_raises(self, lmer_result, simple_data):
        with pytest.raises(ValueError, match="must be a factor"):
            emmeans(lmer_result, "nonexistent")

    def test_multiple_specs(self, simple_data):
        simple_data["factor2"] = np.where(np.random.rand(len(simple_data)) > 0.5, "X", "Y")
        result = lmer("y ~ treatment * factor2 + (1|group)", simple_data)
        em = emmeans(result, ["treatment", "factor2"])
        assert len(em.result.emmean) == 4

    def test_with_at_argument(self, simple_data):
        simple_data["factor2"] = np.where(np.random.rand(len(simple_data)) > 0.5, "X", "Y")
        result = lmer("y ~ treatment * factor2 + (1|group)", simple_data)
        em = emmeans(result, "treatment", at={"factor2": "X"})
        assert len(em.result.emmean) == 2
