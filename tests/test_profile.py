from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mixedlm import lmer
from mixedlm.inference.profile import (
    Profile2DResult,
    ProfileResult,
    as_dataframe,
    confint_profile,
    logProf,
    profile_lmer,
    sdProf,
    varianceProf,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from tests._datasets import SLEEPSTUDY


def constrained_zeta(value, mle, full_deviance):
    """Signed root ML deviance with the Days slope fixed through an offset."""
    days = SLEEPSTUDY["Days"].to_numpy()
    constrained = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, offset=value * days, REML=False)
    return np.sign(value - mle) * np.sqrt(max(constrained.deviance - full_deviance, 0.0))


@pytest.fixture(scope="module")
def ml_deviance():
    return lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False).deviance


class TestProfileLmer:
    def test_zeta_matches_constrained_ml_refits(self, sleepstudy_lmm, ml_deviance):
        profile = profile_lmer(sleepstudy_lmm, which=["Days"], n_points=5)["Days"]

        assert profile.mle == pytest.approx(sleepstudy_lmm.beta[1])
        for value, zeta in zip(profile.values, profile.zeta, strict=True):
            assert zeta == pytest.approx(
                constrained_zeta(value, profile.mle, ml_deviance), abs=1e-6
            )

    @pytest.mark.parametrize("level", [0.90, 0.95])
    def test_interval_ends_at_the_critical_zeta(self, sleepstudy_lmm, ml_deviance, level):
        profile = profile_lmer(sleepstudy_lmm, which=["Days"], level=level, n_points=15)["Days"]
        critical = stats.norm.isf((1 - level) / 2)

        assert profile.level == level
        for bound, sign in ((profile.ci_lower, -1), (profile.ci_upper, 1)):
            assert constrained_zeta(bound, profile.mle, ml_deviance) == pytest.approx(
                sign * critical, abs=1e-3
            )

    def test_method_matches_function(self, sleepstudy_lmm):
        method = sleepstudy_lmm.profile(which="Days", n_points=10)
        function = profile_lmer(sleepstudy_lmm, which=["Days"], n_points=10)

        assert list(method) == ["Days"]
        assert_allclose(method["Days"].values, function["Days"].values)
        assert_allclose(method["Days"].zeta, function["Days"].zeta)

    def test_default_profiles_every_fixed_effect(self, sleepstudy_lmm):
        profiles = profile_lmer(sleepstudy_lmm, n_points=10)

        assert list(profiles) == ["(Intercept)", "Days"]
        assert [len(profile.values) for profile in profiles.values()] == [10, 10]

    def test_glmer_result_profile_method(self, cbpp_glmm):
        profiles = cbpp_glmm.profile(which="(Intercept)", n_points=8)

        profile = profiles["(Intercept)"]
        assert set(profiles) == {"(Intercept)"}
        assert len(profile.values) == 8
        assert profile.mle == pytest.approx(cbpp_glmm.beta[0])
        assert profile.ci_lower < profile.mle < profile.ci_upper

    def test_glmer_result_profile_rejects_parallel_jobs(self, cbpp_glmm):
        with pytest.raises(ValueError, match="only for LmerResult"):
            cbpp_glmm.profile(n_jobs=2)


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

    def test_variance_of_a_negative_profile_reorders_the_bounds(self):
        original = ProfileResult(
            parameter="b",
            values=np.array([-3.0, -2.0, -1.0]),
            zeta=np.array([-1.0, 0.0, 1.0]),
            mle=-2.0,
            ci_lower=-2.5,
            ci_upper=-1.5,
            level=0.95,
        )
        var_profile = varianceProf(original)
        assert (var_profile.ci_lower, var_profile.ci_upper) == (2.25, 6.25)
        assert_allclose(var_profile.values, [9.0, 4.0, 1.0])
        assert_allclose(var_profile.zeta, original.zeta)


def _profile_through(values, mle, ci_lower, ci_upper):
    return ProfileResult(
        parameter="b",
        values=np.asarray(values, dtype=float),
        zeta=np.linspace(-1.0, 1.0, len(values)),
        mle=mle,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        level=0.95,
    )


@pytest.mark.parametrize(
    ("transform", "profile", "message"),
    [
        (logProf, _profile_through([-0.5, 0.5, 1.5], 0.5, -0.2, 1.2), "positive parameter"),
        (logProf, _profile_through([0.5, 1.0, 1.5], 1.0, 0.0, 1.4), "positive parameter"),
        (sdProf, _profile_through([0.5, 1.0, 1.5], 1.0, -0.1, 1.4), "nonnegative parameter"),
        (varianceProf, _profile_through([-0.5, 0.5, 1.5], 0.5, -0.2, 1.2), "change sign"),
    ],
)
def test_scale_transforms_reject_profiles_outside_their_domain(transform, profile, message):
    # Clamping a signed fixed-effect profile would silently invent values.
    with pytest.raises(ValueError, match=message):
        transform(profile)


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
    def test_full_workflow(self, sleepstudy_lmm):
        profiles = profile_lmer(sleepstudy_lmm, n_points=15)
        df = as_dataframe(profiles)
        ci = confint_profile(profiles)

        assert list(profiles) == ["(Intercept)", "Days"]
        assert df["parameter"].value_counts().to_dict() == {"(Intercept)": 15, "Days": 15}
        assert ci["parameter"].tolist() == list(profiles)
        assert_allclose(ci["estimate"], sleepstudy_lmm.beta)
        assert_allclose(ci["lower"], [profile.ci_lower for profile in profiles.values()])
        assert_allclose(ci["upper"], [profile.ci_upper for profile in profiles.values()])

    @staticmethod
    def assert_profiles_match(actual, expected):
        assert list(actual) == list(expected)
        for name, reference in expected.items():
            assert_allclose(actual[name].values, reference.values)
            assert_allclose(actual[name].zeta, reference.zeta, atol=1e-8)
            assert actual[name].ci_lower == pytest.approx(reference.ci_lower)
            assert actual[name].ci_upper == pytest.approx(reference.ci_upper)

    def test_parallel_profiling(self, sleepstudy_lmm):
        # One worker per profiled coefficient; a single coefficient runs serially.
        which = ["(Intercept)", "Days"]
        profiles_serial = profile_lmer(sleepstudy_lmm, which=which, n_points=10, n_jobs=1)
        profiles_parallel = profile_lmer(sleepstudy_lmm, which=which, n_points=10, n_jobs=2)

        self.assert_profiles_match(profiles_parallel, profiles_serial)

    def test_parallel_profiling_falls_back_when_process_pool_is_unavailable(
        self, sleepstudy_lmm, monkeypatch
    ):
        from mixedlm.inference import lmm_profile

        def unavailable(*args, **kwargs):
            raise PermissionError("process semaphores are unavailable")

        monkeypatch.setattr(lmm_profile, "process_pool", unavailable)

        which = ["(Intercept)", "Days"]
        expected = profile_lmer(sleepstudy_lmm, which=which, n_points=10, n_jobs=1)
        with pytest.warns(RuntimeWarning, match="falling back to serial execution"):
            actual = profile_lmer(sleepstudy_lmm, which=which, n_points=10, n_jobs=2)

        self.assert_profiles_match(actual, expected)


def test_profile_2d_plot_passes_1d_coordinates_to_contour(monkeypatch):
    # A supplied axis keeps plotting independent of matplotlib.
    def unexpected_meshgrid(*args, **kwargs):
        raise AssertionError("plotting should not materialize coordinate grids")

    monkeypatch.setattr(np, "meshgrid", unexpected_meshgrid)
    ax = MagicMock()
    profile = Profile2DResult(
        param1="a",
        param2="b",
        values1=np.array([0.0, 1.0, 2.0]),
        values2=np.array([0.0, 1.0, 2.0, 3.0]),
        zeta=np.zeros((3, 4)),
        mle1=1.0,
        mle2=2.0,
        level=0.95,
    )

    profile.plot(ax=ax, show_ci=False, show_mle=False)

    # Rows of zeta follow param1, so param2 runs along the x axis.
    x, y, z = ax.contour.call_args.args
    assert_array_equal(x, [0.0, 1.0, 2.0, 3.0])
    assert_array_equal(y, [0.0, 1.0, 2.0])
    assert z.shape == (3, 4)
