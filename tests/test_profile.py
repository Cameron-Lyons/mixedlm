from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.inference.profile import (
    ProfileResult,
    as_dataframe,
    confint_profile,
    logProf,
    profile_lmer,
    sdProf,
    varianceProf,
)
from numpy.testing import assert_allclose


@pytest.fixture
def sleepstudy_data():
    np.random.seed(42)
    n_subjects = 10
    n_days = 10
    n = n_subjects * n_days

    subjects = np.repeat([f"S{i}" for i in range(n_subjects)], n_days)
    days = np.tile(np.arange(n_days), n_subjects)

    subject_intercepts = np.repeat(np.random.randn(n_subjects) * 25, n_days)
    subject_slopes = np.repeat(np.random.randn(n_subjects) * 5, n_days)

    y = 250 + 10 * days + subject_intercepts + subject_slopes * days + np.random.randn(n) * 30

    return pd.DataFrame({"Reaction": y, "Days": days, "Subject": subjects})


@pytest.fixture
def lmer_result(sleepstudy_data):
    return lmer("Reaction ~ Days + (1|Subject)", sleepstudy_data)


class TestProfileResult:
    def test_dataclass_fields(self):
        result = ProfileResult(
            parameter="test",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            level=0.95,
        )
        assert result.parameter == "test"
        assert len(result.values) == 3
        assert result.mle == 2.0

    def test_plot_method(self):
        matplotlib = pytest.importorskip("matplotlib")
        result = ProfileResult(
            parameter="test",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            level=0.95,
        )
        ax = result.plot()
        assert ax is not None
        matplotlib.pyplot.close("all")


class TestProfileLmer:
    def test_profile_single_param(self, lmer_result):
        profiles = profile_lmer(lmer_result, which=["Days"], n_points=10)
        assert "Days" in profiles
        assert len(profiles["Days"].values) == 10

    def test_profile_all_params(self, lmer_result):
        profiles = profile_lmer(lmer_result, n_points=10)
        assert "(Intercept)" in profiles
        assert "Days" in profiles

    def test_profile_zeta_at_mle(self, lmer_result):
        profiles = profile_lmer(lmer_result, which=["Days"], n_points=21)
        profile = profiles["Days"]
        closest_to_mle = np.argmin(np.abs(profile.values - profile.mle))
        assert abs(profile.zeta[closest_to_mle]) < 0.5

    def test_profile_ci_contains_mle(self, lmer_result):
        profiles = profile_lmer(lmer_result, which=["Days"], n_points=15)
        profile = profiles["Days"]
        assert profile.ci_lower < profile.mle
        assert profile.ci_upper > profile.mle

    def test_profile_level(self, lmer_result):
        profiles = profile_lmer(lmer_result, which=["Days"], level=0.90, n_points=10)
        assert profiles["Days"].level == 0.90


class TestLogProf:
    def test_log_transformation(self):
        original = ProfileResult(
            parameter="sigma",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            level=0.95,
        )
        log_profile = logProf(original)
        assert log_profile.parameter == "log(sigma)"
        assert_allclose(log_profile.mle, np.log(2.0))
        assert_allclose(log_profile.zeta, original.zeta)


class TestVarianceProf:
    def test_variance_transformation(self):
        original = ProfileResult(
            parameter="sigma",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            level=0.95,
        )
        var_profile = varianceProf(original)
        assert var_profile.parameter == "sigma²"
        assert var_profile.mle == 4.0
        assert_allclose(var_profile.values, [1.0, 4.0, 9.0])

    def test_variance_ci_swap(self):
        original = ProfileResult(
            parameter="sigma",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([1.0, 0.0, -1.0]),
            mle=2.0,
            ci_lower=-1.5,
            ci_upper=-2.5,
            level=0.95,
        )
        var_profile = varianceProf(original)
        assert var_profile.ci_lower <= var_profile.ci_upper


class TestSdProf:
    def test_sd_transformation(self):
        original = ProfileResult(
            parameter="var",
            values=np.array([1.0, 4.0, 9.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=4.0,
            ci_lower=1.0,
            ci_upper=9.0,
            level=0.95,
        )
        sd_profile = sdProf(original)
        assert sd_profile.parameter == "sqrt(var)"
        assert sd_profile.mle == 2.0
        assert_allclose(sd_profile.values, [1.0, 2.0, 3.0])


class TestAsDataframe:
    def test_single_profile(self):
        profile = ProfileResult(
            parameter="test",
            values=np.array([1.0, 2.0, 3.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=2.0,
            ci_lower=1.5,
            ci_upper=2.5,
            level=0.95,
        )
        df = as_dataframe(profile)
        assert len(df) == 3
        assert "parameter" in df.columns
        assert "value" in df.columns
        assert "zeta" in df.columns

    def test_multiple_profiles(self):
        profiles = {
            "param1": ProfileResult(
                parameter="param1",
                values=np.array([1.0, 2.0]),
                zeta=np.array([-1.0, 1.0]),
                mle=1.5,
                ci_lower=1.0,
                ci_upper=2.0,
                level=0.95,
            ),
            "param2": ProfileResult(
                parameter="param2",
                values=np.array([3.0, 4.0]),
                zeta=np.array([-0.5, 0.5]),
                mle=3.5,
                ci_lower=3.0,
                ci_upper=4.0,
                level=0.95,
            ),
        }
        df = as_dataframe(profiles)
        assert len(df) == 4


class TestConfintProfile:
    def test_confint_basic(self):
        profiles = {
            "param1": ProfileResult(
                parameter="param1",
                values=np.array([1.0, 2.0, 3.0]),
                zeta=np.array([-1.0, 0.0, 1.0]),
                mle=2.0,
                ci_lower=1.5,
                ci_upper=2.5,
                level=0.95,
            ),
        }
        ci = confint_profile(profiles)
        assert len(ci) == 1
        assert "parameter" in ci.columns
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_confint_custom_level(self):
        profiles = {
            "param1": ProfileResult(
                parameter="param1",
                values=np.linspace(-3, 3, 50),
                zeta=np.linspace(-3, 3, 50),
                mle=0.0,
                ci_lower=-2.0,
                ci_upper=2.0,
                level=0.95,
            ),
        }
        ci_90 = confint_profile(profiles, level=0.90)
        ci_95 = confint_profile(profiles, level=0.95)
        assert ci_90.iloc[0]["lower"] >= ci_95.iloc[0]["lower"]


class TestProfileIntegration:
    def test_full_workflow(self, lmer_result):
        profiles = profile_lmer(lmer_result, n_points=15)
        df = as_dataframe(profiles)
        ci = confint_profile(profiles)

        assert len(profiles) > 0
        assert len(df) > 0
        assert len(ci) == len(profiles)

    def test_parallel_profiling(self, lmer_result):
        profiles_serial = profile_lmer(lmer_result, which=["Days"], n_points=10, n_jobs=1)
        assert "Days" in profiles_serial
