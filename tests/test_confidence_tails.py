from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from mixedlm import allEffects, bootCI, emmeans, ggpredict, tidy
from mixedlm.inference import linear_hypothesis
from mixedlm.inference.profile import ProfileResult, confint_profile, profile_glmer, profile_lmer
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.models.nlmer import NlmerResult
from mixedlm.power import _binomial_score_interval
from numpy.testing import assert_allclose
from scipy import stats

from tests.test_boot_ci import bootstrap_result as bootstrap_result
from tests.test_linear_hypothesis import glmm_result as glmm_result
from tests.test_linear_hypothesis import lmm_result as lmm_result
from tests.test_reporting import nlmm_model as nlmm_model

LEVELS = [0.95, np.nextafter(1.0, 0.0), np.nextafter(np.float32(1), np.float32(0))]


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_marginal_and_adjusted_prediction_intervals_preserve_tail_probability(request, kind, level):
    model = request.getfixturevalue(f"{kind}_result")
    marginal = emmeans(model, [], level=level, type="link").result
    effect = ggpredict(model, "x", at={"x": [0.0]}, level=level, type="link")
    df = model.df_residual() if kind == "lmm" else np.inf
    tail = (1.0 - float(level)) / 2.0
    for estimate, se, lower, upper in [
        (marginal.emmean, marginal.se, marginal.lower, marginal.upper),
        (effect.predicted, effect["std.error"], effect["conf.low"], effect["conf.high"]),
    ]:
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
        critical = (upper - estimate) / se
        assert_allclose(stats.t.sf(critical, df), tail, rtol=1e-9, atol=0)
        assert_allclose(estimate - lower, upper - estimate, rtol=1e-12)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_all_effect_grids_retain_finite_intervals(request, kind, level):
    model = request.getfixturevalue(f"{kind}_result")
    results = allEffects(model, n_points=3, level=level, type="link")
    for term, result in results.items():
        expected = ggpredict(model, term, n_points=3, level=level, type="link")
        assert np.isfinite(result[["conf.low", "conf.high"]]).all().all()
        assert_allclose(result[["conf.low", "conf.high"]], expected[["conf.low", "conf.high"]])


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("kind", ["lmm", "glmm", "nlmm"])
def test_wald_parameter_intervals_preserve_tail_probability(request, kind, level):
    model = request.getfixturevalue("nlmm_model" if kind == "nlmm" else f"{kind}_result")
    estimates = model.phi if kind == "nlmm" else model.beta
    se = np.sqrt(np.diag(model.vcov()))

    result = model.confint(level=level, method="Wald")

    bounds = np.array(list(result.values()))
    assert np.isfinite(bounds).all()
    critical = (bounds[:, 1] - estimates) / se
    assert_allclose(stats.norm.sf(critical), (1 - float(level)) / 2, rtol=1e-9, atol=0)


@pytest.mark.parametrize(
    "kind,interval", [("lmm", "confidence"), ("lmm", "prediction"), ("glmm", "confidence")]
)
@pytest.mark.parametrize("level", LEVELS)
def test_model_prediction_intervals_keep_finite_endpoints(request, kind, interval, level):
    model = request.getfixturevalue(f"{kind}_result")
    kwargs = {"type": "link"} if kind == "glmm" else {}

    result = model.predict(re_form="NA", interval=interval, level=level, **kwargs)

    assert np.isfinite(result.lower).all() and np.isfinite(result.upper).all()
    variance = result.se_fit**2
    if interval == "prediction":
        variance = variance + model.sigma**2 / model.matrices.weights
    critical = (result.upper - result.fit) / np.sqrt(variance)
    assert_allclose(stats.norm.sf(critical), (1 - float(level)) / 2, rtol=1e-9, atol=0)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("level", LEVELS)
def test_hypothesis_and_tidy_intervals_preserve_tail_probability(request, kind, level):
    model = request.getfixturevalue(f"{kind}_result")
    hypothesis = linear_hypothesis(model, {"x": 1.0}, level=level)
    df = model.df_residual() if kind == "lmm" else np.inf
    critical = (hypothesis.conf_high - hypothesis.estimate) / hypothesis.std_error
    assert_allclose(stats.t.sf(critical, df), (1 - float(level)) / 2, rtol=1e-9, atol=0)

    result = tidy(model, conf_int=True, conf_level=level, ddf_method="normal")
    critical = (result["conf.high"] - result.estimate) / result["std.error"]
    assert_allclose(stats.norm.sf(critical), (1 - float(level)) / 2, rtol=1e-9, atol=0)


@pytest.mark.parametrize("level", LEVELS)
def test_normal_bootstrap_interfaces_agree_at_extreme_confidence(request, level):
    bootstrap = request.getfixturevalue("bootstrap_result")
    result = bootstrap.ci(method="normal", level=level)
    table = bootCI(bootstrap, method="normal", level=level)

    for index, name in enumerate(bootstrap.fixed_names):
        samples = bootstrap.beta_samples[:, index]
        samples = samples[np.isfinite(samples)]
        center = 2 * bootstrap.original_beta[index] - samples.mean()
        lower, upper = result[name]
        assert np.isfinite([lower, upper]).all()
        critical = (upper - center) / samples.std(ddof=1)
        assert_allclose(stats.norm.sf(critical), (1 - float(level)) / 2, rtol=1e-9, atol=0)
        assert_allclose(table.loc[index, ["conf.low", "conf.high"]].astype(float), [lower, upper])


def test_degenerate_normal_bootstrap_interval_does_not_multiply_zero_by_infinity(request):
    bootstrap = request.getfixturevalue("bootstrap_result")
    bootstrap.beta_samples[:] = bootstrap.original_beta

    result = bootstrap.ci(method="normal", level=np.nextafter(1.0, 0.0))

    assert_allclose(list(result.values()), [[1, 1], [2, 2]])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_profile_builders_keep_extreme_confidence_endpoints_finite(request, kind):
    model = request.getfixturevalue(f"{kind}_result")
    profile = profile_lmer if kind == "lmm" else profile_glmer

    result = profile(model, which="x", n_points=7, level=np.nextafter(1.0, 0.0))["x"]

    assert np.isfinite([result.ci_lower, result.ci_upper]).all()
    assert result.ci_lower < result.mle < result.ci_upper


@pytest.mark.parametrize("level", LEVELS)
def test_profile_interval_extraction_and_plot_use_finite_cutoffs(level):
    critical = stats.norm.isf((1 - float(level)) / 2)
    original_critical = stats.norm.isf(0.025)
    profile = ProfileResult(
        "x",
        np.array([-3.0, 0.0, 3.0]),
        np.array([-3.0, 0.0, 3.0]),
        0.0,
        -original_critical,
        original_critical,
        0.95,
    )

    interval = confint_profile({"x": profile}, level=level)

    assert_allclose(interval[["lower", "upper"]], [[-critical, critical]])
    profile.level = level
    axes = MagicMock()
    profile.plot(ax=axes)
    assert_allclose(
        [call.args[0] for call in axes.axhline.call_args_list], [0.0, critical, -critical]
    )


def test_profile_interval_fallback_uses_the_stored_extreme_level(monkeypatch):
    level = np.nextafter(1.0, 0.0)
    original_critical = stats.norm.isf((1 - level) / 2)
    profile = ProfileResult(
        "x",
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        0.0,
        -original_critical,
        original_critical,
        level,
    )

    def unavailable_interpolation(*args, **kwargs):
        raise ValueError("interpolation unavailable")

    monkeypatch.setattr("scipy.interpolate.interp1d", unavailable_interpolation)
    interval = confint_profile({"x": profile}, level=0.9)

    critical = stats.norm.isf(0.05)
    assert_allclose(interval[["lower", "upper"]], [[-critical, critical]])


@pytest.mark.parametrize("kind", ["lmm", "nlmm"])
def test_tidy_student_reference_preserves_extreme_confidence(request, kind):
    model = request.getfixturevalue("lmm_result" if kind == "lmm" else "nlmm_model")
    level = np.nextafter(1.0, 0.0)
    table = tidy(model, conf_int=True, conf_level=level, ddf_method="Satterthwaite")

    assert np.isfinite(table.df).all()
    critical = (table["conf.high"] - table.estimate) / table["std.error"]
    assert_allclose(stats.t.sf(critical, table.df), (1 - level) / 2, rtol=1e-9, atol=0)


@pytest.mark.parametrize("alpha", [0.05, 1e-20, 1e-100])
@pytest.mark.parametrize("successes", [0, 17, 50])
def test_wilson_power_intervals_use_the_requested_significance_tail(alpha, successes):
    lower, upper = _binomial_score_interval(successes, 50, alpha)
    assert 0 <= lower < upper <= 1
    for bound in [lower, upper]:
        if bound in (0, 1):
            continue
        score = abs(successes / 50 - bound) / np.sqrt(bound * (1 - bound) / 50)
        assert_allclose(stats.norm.sf(score), alpha / 2, rtol=1e-9, atol=0)


_VALIDATED_CALLS = {
    "emmeans": lambda level: emmeans(object(), [], level=level),
    "effect": lambda level: ggpredict(object(), "x", level=level),
    "all_effects": lambda level: allEffects(object(), level=level),
    "lmm_profile": lambda level: profile_lmer(object(), which=[], level=level),
    "glmm_profile": lambda level: profile_glmer(object(), which=[], level=level),
    "profile_intervals": lambda level: confint_profile({}, level=level),
    "lmm_wald": lambda level: LmerResult.confint(object(), level=level),
    "glmm_wald": lambda level: GlmerResult.confint(object(), level=level),
    "nlmm_wald": lambda level: NlmerResult.confint(object(), level=level, method="Wald"),
    "lmm_predict": lambda level: LmerResult.predict(object(), level=level),
    "glmm_predict": lambda level: GlmerResult.predict(object(), level=level),
}


@pytest.mark.parametrize("api", _VALIDATED_CALLS)
@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, np.nan, np.inf, True, "0.95", [0.95]])
def test_invalid_confidence_is_rejected_before_model_work(api, level):
    with pytest.raises((TypeError, ValueError), match="level must be.*between 0 and 1"):
        _VALIDATED_CALLS[api](level)
