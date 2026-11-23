"""Profile intervals of standard deviations, correlations and sigma agree with lme4."""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer, load_sleepstudy, varianceProf
from mixedlm.estimation.joint_glmm import JointGLMMObjective
from mixedlm.estimation.reml import _build_lambda_blocks
from mixedlm.inference.variance_profile import sdcor_from_theta, variance_parameters
from numpy.testing import assert_allclose
from scipy import optimize, stats

from tests._datasets import CBPP, SLEEPSTUDY

# lme4 2.1.0: confint(lmer(Reaction ~ Days + (Days | Subject), sleepstudy)). Its
# profile interpolates a spline, so the endpoints agree to about 3e-4.
LME4_SLEEPSTUDY = {
    ".sig01": (14.38145599, 37.71583597),
    ".sig02": (3.80116385, 8.75338599),
    ".sig03": (-0.48150078, 0.68498637),
    ".sigma": (22.89826688, 28.85799652),
    "(Intercept)": (237.68069550, 265.12951462),
    "Days": (7.35865330, 13.57591873),
}


@pytest.fixture(scope="module")
def sleep_fit():
    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)


def test_profile_intervals_match_lme4(sleep_fit):
    intervals = sleep_fit.confint(parm=["theta_", "beta_"], method="profile")
    assert list(intervals) == list(LME4_SLEEPSTUDY)
    for name, expected in LME4_SLEEPSTUDY.items():
        assert_allclose(intervals[name], expected, atol=5e-4, err_msg=name)


def test_parameters_are_named_and_described_as_in_lme4(sleep_fit):
    parameters = sleep_fit.variance_parameters()
    # lme4 2: each block lists its SDs, then its correlations.
    assert [(p.name, p.label) for p in parameters] == [
        (".sig01", "sd_(Intercept)|Subject"),
        (".sig02", "sd_Days|Subject"),
        (".sig03", "cor_Days.(Intercept)|Subject"),
        (".sigma", "sigma"),
    ]
    assert parameters[2].description == "Correlation of Days and (Intercept) | Subject"
    assert parameters[1].description == "SD of Days | Subject"
    by_label = sleep_fit.confint(parm=["sd_Days|Subject", "sigma"], method="profile")
    assert list(by_label) == [".sig02", ".sigma"]
    assert_allclose(by_label[".sig02"], LME4_SLEEPSTUDY[".sig02"], atol=5e-4)


def test_selectors_and_default_parameters(sleep_fit):
    assert list(sleep_fit.confint()) == ["(Intercept)", "Days"]
    wald = sleep_fit.confint(parm=["theta_", "Days"])
    assert list(wald) == [".sig01", ".sig02", ".sig03", ".sigma", "Days"]
    assert all(np.isnan(wald[name]).all() for name in (".sig01", ".sigma"))
    profiles = sleep_fit.profile(which=[".sig02", "Days"], n_points=5)
    assert set(profiles) == {".sig02", "Days"}
    profile = profiles[".sig02"]
    assert np.all(np.diff(profile.zeta) > 0)
    assert profile.zeta[2] == 0.0
    assert_allclose(profile.zeta[[0, -1]], [-1.959964, 1.959964], atol=1e-5)
    variance = varianceProf(profile)
    assert_allclose(
        [variance.ci_lower, variance.ci_upper], np.square(LME4_SLEEPSTUDY[".sig02"]), rtol=1e-4
    )


def test_bootstrap_intervals_cover_the_estimates(sleep_fit):
    intervals = sleep_fit.confint(parm=["theta_", "Days"], method="boot", n_boot=60, seed=3)
    estimates = dict(
        zip(
            [p.name for p in sleep_fit.variance_parameters()],
            [
                *sdcor_from_theta(
                    sleep_fit.theta, sleep_fit.sigma, sleep_fit.matrices.random_structures
                ),
                sleep_fit.sigma,
            ],
            strict=True,
        )
    )
    for name, estimate in estimates.items():
        lower, upper = intervals[name]
        assert lower < estimate < upper, name
    assert "Days" in intervals


def test_parallel_profiles_match_serial(sleep_fit):
    serial = sleep_fit.profile(which="theta_", n_points=3)
    parallel = sleep_fit.profile(which="theta_", n_points=3, n_jobs=2)
    for name, profile in serial.items():
        assert_allclose(
            [parallel[name].ci_lower, parallel[name].ci_upper],
            [profile.ci_lower, profile.ci_upper],
            rtol=1e-7,
        )


def test_boundary_estimates_report_the_parameter_limit():
    rng = np.random.default_rng(2)
    data = pd.DataFrame(
        {"y": rng.standard_normal(60), "x": rng.standard_normal(60), "g": np.repeat(range(12), 5)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = lmer("y ~ x + (1 | g)", data)
    lower, upper = fit.confint(parm=".sig01", method="profile")[".sig01"]
    assert lower == 0.0
    assert upper > 0.0


def test_uncorrelated_terms_profile_each_standard_deviation():
    fit = lmer("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", SLEEPSTUDY)
    intervals = fit.confint(parm="theta_", method="profile")
    assert list(intervals) == [".sig01", ".sig02", ".sigma"]
    # lme4 2.1.0: confint(lmer(Reaction ~ Days + (1|Subject) + (0+Days|Subject), sleepstudy))
    assert_allclose(intervals[".sig01"], (15.25864870, 37.78647252), atol=5e-4)
    assert_allclose(intervals[".sig02"], (3.96407385, 8.76915922), atol=5e-4)
    assert_allclose(intervals[".sigma"], (22.88055465, 28.78759760), atol=5e-4)


def test_glmm_profiles_reach_the_likelihood_ratio_cutoff():
    fit = glmer(
        "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        CBPP,
        family=families.Binomial(),
    )
    assert [p.name for p in fit.variance_parameters()] == [".sig01"]
    lower, upper = fit.confint(parm=".sig01", method="profile")[".sig01"]
    objective = JointGLMMObjective(
        fit.matrices, fit.family, fit.nAGQ, pirls_maxiter=fit.pirls_maxiter, pirls_tol=fit.pirls_tol
    )

    def minimum(theta=None):
        if theta is None:
            start, unpack = np.r_[fit.theta, fit.beta], lambda p: p
        else:
            start, unpack = fit.beta, lambda b: np.r_[theta, b]
        fitted = optimize.minimize(
            lambda p: objective.evaluate(unpack(p))[0],
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000},
        )
        return fitted.fun

    best = minimum()
    cutoff = stats.chi2.isf(0.05, 1)
    for endpoint in (lower, upper):
        assert minimum(np.array([endpoint])) - best == pytest.approx(cutoff, abs=1e-4)
    with pytest.raises(ValueError, match="Unknown newparams"):
        fit.simulate(newparams={"sigma": 1.0})


@pytest.mark.parametrize(
    "structure",
    [
        SimpleNamespace(n_terms=3, correlated=True, cov_type="us", term_names=["a", "b", "c"]),
        SimpleNamespace(n_terms=2, correlated=False, cov_type="us", term_names=["a", "b"]),
        SimpleNamespace(n_terms=3, correlated=True, cov_type="cs", term_names=["a", "b", "c"]),
        SimpleNamespace(n_terms=3, correlated=True, cov_type="ar1", term_names=["a", "b", "c"]),
        SimpleNamespace(n_terms=1, correlated=True, cov_type="us", term_names=["a"]),
    ],
)
def test_standard_deviations_and_correlations_follow_the_covariance(structure):
    structure.grouping_factor = "g"
    structure.n_levels = 4
    structures = [structure]
    rng = np.random.default_rng(5)
    q = structure.n_terms
    if structure.cov_type in ("cs", "ar1"):
        theta = np.array([0.8, 0.3]) if q > 1 else np.array([0.8])
    elif structure.correlated:
        theta = rng.uniform(0.2, 1.0, q * (q + 1) // 2)
    else:
        theta = rng.uniform(0.2, 1.0, q)
    (factor,) = _build_lambda_blocks(theta, structures)
    covariance = 2.5**2 * factor @ factor.T
    sd = np.sqrt(np.diag(covariance))
    parameters = variance_parameters(structures, has_sigma=False)
    values = sdcor_from_theta(theta, 2.5, structures)
    assert len(values) == len(parameters)
    for parameter, value in zip(parameters, values, strict=True):
        if parameter.kind == "sd":
            assert value == pytest.approx(sd[structure.term_names.index(parameter.terms[0])])
        else:
            i, j = (structure.term_names.index(term) for term in parameter.terms)
            assert value == pytest.approx(covariance[i, j] / (sd[i] * sd[j]))


def test_tidy_reports_profile_limits_for_random_parameters(sleep_fit):
    table = sleep_fit.tidy(["fixed", "ran_pars"], conf_int=True, conf_method="profile")
    limits = dict(
        zip(table["term"], zip(table["conf.low"], table["conf.high"], strict=True), strict=True)
    )
    expected = {
        "(Intercept)": LME4_SLEEPSTUDY["(Intercept)"],
        "Days": LME4_SLEEPSTUDY["Days"],
        "sd__(Intercept)": LME4_SLEEPSTUDY[".sig01"],
        "cor__(Intercept).Days": LME4_SLEEPSTUDY[".sig03"],
        "sd__Days": LME4_SLEEPSTUDY[".sig02"],
        "sd__Observation": LME4_SLEEPSTUDY[".sigma"],
    }
    for term, interval in expected.items():
        assert_allclose(limits[term], interval, atol=5e-4, err_msg=term)
    wald = sleep_fit.tidy("ran_pars", conf_int=True)
    assert wald["conf.low"].isna().all()
    with pytest.raises(ValueError, match="Unknown conf_method"):
        sleep_fit.tidy(conf_int=True, conf_method="exact")


@pytest.mark.parametrize(
    ("levels", "expected"),
    [([18, 6], [0, 1]), ([6, 18], [1, 0]), ([10, 10, 5], [0, 1, 2]), ([5, 10, 10], [2, 1, 0])],
)
def test_blocks_are_numbered_in_lme4_order(levels, expected):
    from mixedlm.inference.variance_profile import block_order

    structures = [SimpleNamespace(n_levels=n) for n in levels]
    assert block_order(structures) == expected


def test_parameter_names_follow_lme4_term_order():
    data = SLEEPSTUDY.assign(batch=SLEEPSTUDY["Subject"].astype(str).str[-1])
    # batch has fewer levels than Subject, so lme4 numbers the Subject block first.
    fit = lmer("Reaction ~ Days + (1 | batch) + (Days | Subject)", data)
    described = [(p.name, p.group, p.kind) for p in fit.variance_parameters()]
    assert described == [
        (".sig01", "Subject", "sd"),
        (".sig02", "Subject", "sd"),
        (".sig03", "Subject", "cor"),
        (".sig04", "batch", "sd"),
        (".sigma", "Residual", "sigma"),
    ]
    values = sdcor_from_theta(fit.theta, fit.sigma, fit.matrices.random_structures)
    groups = fit.VarCorr().groups
    assert values[0] == pytest.approx(groups["Subject"].stddev["(Intercept)"])
    assert values[1] == pytest.approx(groups["Subject"].stddev["Days"])
    assert values[2] == pytest.approx(groups["Subject"].corr[1, 0])
    assert values[3] == pytest.approx(groups["batch"].stddev["(Intercept)"])
    table = fit.tidy("ran_pars", conf_int=True, conf_method="boot", n_boot=20, seed=1)
    assert table["conf.low"].notna().all()


def test_reordered_blocks_match_lme4():
    data = SLEEPSTUDY.assign(batch=SLEEPSTUDY["Subject"].astype(str).str[-1])
    fit = lmer("Reaction ~ Days + (1 | batch) + (Days | Subject)", data)
    intervals = fit.confint(parm="theta_", method="profile")
    # lme4 2.1.0: confint(lmer(Reaction ~ Days + (1 | batch) + (Days | Subject), data))
    expected = {
        ".sig01": (8.78035856, 36.31202237),
        ".sig02": (3.80116351, 8.75338807),
        # lme4's spline overshoots this endpoint: its 0.84506 raises the deviance
        # by 3.923, above the 3.841 cutoff, so the exact limit is 0.83505.
        ".sig03": (-0.54756481, 0.83505),
        ".sig04": (0.0, 32.71399120),
        ".sigma": (22.89826680, 28.85799627),
    }
    for name, interval in expected.items():
        assert_allclose(intervals[name], interval, atol=2e-3, err_msg=name)


def test_three_term_blocks_profile_over_valid_correlations():
    from mixedlm.inference.variance_profile import _LMMVarianceProfile

    rng = np.random.default_rng(7)
    data = SLEEPSTUDY.assign(D2=SLEEPSTUDY["Days"] ** 2 / 10 + rng.normal(size=len(SLEEPSTUDY)))
    fit = lmer("Reaction ~ Days + D2 + (Days + D2 | Subject)", data)
    intervals = fit.confint(parm="theta_", method="profile")
    assert len(intervals) == 7
    estimates = _LMMVarianceProfile(fit).estimates()
    for (lower, upper), estimate in zip(intervals.values(), estimates, strict=True):
        assert lower <= estimate <= upper
    # Constrained points satisfy their constraint exactly and keep valid factors.
    profile = _LMMVarianceProfile(fit)
    for index, value in [(3, 0.8), (5, -0.99), (1, 14.0)]:
        profile.start(index)
        rise = profile.profile_deviance(value) - profile.minimum
        x = profile.cache[value][1]
        constraint = profile._constraint(index, value)
        sigma = (
            constraint.sigma(x)
            if constraint.sigma is not None
            else profile.optimizer._evaluate_core(x).sigma
        )
        achieved = sdcor_from_theta(x, sigma, fit.matrices.random_structures)[index]
        assert achieved == pytest.approx(value, abs=1e-9)
        assert rise >= -1e-6


def test_profiles_leave_zero_variance_boundaries():
    # sd(b) is estimated as exactly zero; profiles of sd(a) must still re-open it.
    rng = np.random.default_rng(2)
    a = np.repeat(np.arange(24), 3)
    data = pd.DataFrame(
        {
            "y": rng.normal(scale=0.8, size=24)[a] + rng.normal(size=72),
            "a": [f"a{i:02d}" for i in a],
            "b": [f"b{i // 4}" for i in a],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = lmer("y ~ 1 + (1 | a) + (1 | b)", data, REML=False)
    assert fit.theta[1] == 0.0
    lower, _ = fit.confint(parm=".sig01", method="profile")[".sig01"]
    # A dense brute-force ML profile puts the limit at 0.3548060 (lme4 2.1.0: 0.354875).
    assert lower == pytest.approx(0.3548060, abs=2e-6)


def test_uncorrelated_terms_are_numbered_as_separate_blocks():
    rng = np.random.default_rng(5)
    n = 360
    g, h, x = rng.integers(0, 12, n), rng.integers(0, 30, n), rng.normal(size=n)
    y = (
        1
        + 0.3 * x
        + rng.normal(scale=0.5, size=12)[g]
        + rng.normal(scale=0.3, size=12)[g] * x
        + rng.normal(scale=0.8, size=30)[h]
        + rng.normal(scale=0.8, size=n)
    )
    data = pd.DataFrame(
        {"y": y, "x": x, "g": [f"g{i:02d}" for i in g], "h": [f"h{i:02d}" for i in h]}
    )
    fit = lmer("y ~ x + (1 + x || g) + (1 | h)", data)
    # lme4 2.1.0 expands || into (1|g) + (0+x|g) and orders the blocks h, g.x, g.
    assert [p.label for p in fit.variance_parameters()] == [
        "sd_(Intercept)|h",
        "sd_x|g",
        "sd_(Intercept)|g",
        "sigma",
    ]
    intervals = fit.confint(parm="theta_", method="profile")
    assert_allclose(intervals[".sig01"], (0.531293, 0.946751), atol=2e-6)
    assert_allclose(intervals[".sig03"], (0.258243, 0.671709), atol=5e-6)
    # lme4's spline gives 0.046444 for the lower limit; a dense brute-force
    # profile reaches the cutoff at 0.045350.
    assert_allclose(intervals[".sig02"], (0.045350, 0.317752), atol=2e-6)


def _nested(seed):
    rng = np.random.default_rng(seed)
    a = np.repeat(np.arange(24), 3)
    return pd.DataFrame(
        {
            "y": rng.normal(scale=0.8, size=24)[a] + rng.normal(size=72),
            "a": [f"a{i:02d}" for i in a],
            "b": [f"b{i // 4}" for i in a],
        }
    )


@pytest.mark.parametrize(
    ("seed", "optimizer", "lower"),
    # Lower limits of a dense brute-force ML profile.
    [(3, "auto", 0.515581), (2, "L-BFGS-B", 0.354806)],
)
def test_profiles_leave_near_zero_variances(seed, optimizer, lower):
    from mixedlm import lmerControl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = lmer(
            "y ~ 1 + (1 | a) + (1 | b)",
            _nested(seed),
            REML=False,
            control=lmerControl(optimizer=optimizer),
        )
    assert fit.confint(parm=".sig01", method="profile")[".sig01"][0] == pytest.approx(
        lower, abs=2e-6
    )


def test_profiles_of_middle_columns_do_not_collapse_onto_a_zero_factor():
    rng = np.random.default_rng(11)
    n = 360
    g, x, z = rng.integers(0, 15, n), rng.uniform(0, 4, n), rng.uniform(0, 2, n)
    w = rng.normal(size=n)
    f = rng.choice(["A", "B", "C"], n)
    rng.choice(["p", "q"], n)
    y = (
        1
        + 0.3 * x
        + 0.3 * x * z
        + 0.2 * w
        + (f == "B") * 0.5
        + rng.normal(size=15)[g]
        + rng.normal(size=15)[g] * x / 3
        + rng.normal(size=n)
    )
    data = pd.DataFrame({"y": y, "x": x, "z": z, "xz": x * z, "g": [f"g{i:02d}" for i in g]})
    middle = lmer("y ~ x + z + x:z + (1 + x:z + x | g)", data, REML=False)
    last = lmer("y ~ x + z + x:z + (1 + xz + x | g)", data, REML=False)
    assert middle.matrices.random_structures[0].term_names == ["(Intercept)", "x", "x:z"]
    first = middle.confint(parm="sd_x|g", method="profile")[".sig02"]
    second = last.confint(parm="sd_x|g", method="profile")[".sig03"]
    # A dense brute-force profile reaches the cutoff at 0.18971 (lme4 2.1.0: 0.18951).
    assert_allclose(first, second, atol=1e-6)
    assert first[0] == pytest.approx(0.189705, abs=1e-5)


def test_double_bar_and_separate_terms_give_the_same_profiles():
    rng = np.random.default_rng(11)
    n = 600
    g, h, k = rng.integers(0, 12, n), rng.integers(0, 30, n), rng.integers(0, 8, n)
    x, z, f = rng.normal(size=n), rng.normal(size=n), rng.integers(0, 3, n)
    y = (
        1
        + 0.3 * x
        + rng.normal(scale=0.5, size=12)[g]
        + rng.normal(scale=0.3, size=12)[g] * x
        + rng.normal(scale=0.4, size=12)[g] * z
        + rng.normal(scale=0.8, size=30)[h]
        + rng.normal(scale=0.5, size=8)[k]
        + 0.3 * f
        + rng.normal(scale=0.4, size=(12, 3))[g, f]
        + rng.normal(scale=0.8, size=n)
    )
    data = pd.DataFrame(
        {"y": y, "x": x, "z": z, "g": [f"g{i:02d}" for i in g], "h": [f"h{i:02d}" for i in h]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        intervals = [
            lmer(formula, data).confint(parm="sd_z|h", method="profile")[".sig02"]
            for formula in (
                "y ~ x + (1 + x || g) + (1 + z | h)",
                "y ~ x + (1 | g) + (0 + x | g) + (1 + z | h)",
            )
        ]
    # lme4 2.1.0 and a dense brute force give an upper limit of 0.180881.
    assert_allclose(intervals[0], intervals[1], atol=1e-6)
    assert intervals[0][1] == pytest.approx(0.180883, abs=1e-5)


def test_multi_column_numeric_terms_in_double_bars_stay_correlated_as_in_lme4():
    rng = np.random.default_rng(5)
    n = 360
    g, h, x = rng.integers(0, 12, n), rng.integers(0, 30, n), rng.normal(size=n)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n) + x,
            "x": x,
            "g": [f"g{i:02d}" for i in g],
            "h": [f"h{i:02d}" for i in h],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = lmer("y ~ x + (1 + poly(x, 2) || g) + (1 | h)", data)
    # As in lme4 2.1.0, (1 + poly(x, 2) || g) is (1 | g) + (0 + poly(x, 2) | g), and
    # confint() numbers the h block, then the poly(x, 2) block, then the g intercept.
    assert str(fit.formula) == "y ~ x + (1 || g) + (0 + poly(x, 2) | g) + (1 | h)"
    assert [p.label for p in fit.variance_parameters()] == [
        "sd_(Intercept)|h",
        "sd_poly(x, 2)1|g",
        "sd_poly(x, 2)2|g",
        "cor_poly(x, 2)2.poly(x, 2)1|g",
        "sd_(Intercept)|g",
        "sigma",
    ]


def test_double_bar_polynomials_fit_the_lme4_model():
    sleep = load_sleepstudy()
    fit = lmer("Reaction ~ Days + (1 + poly(Days, 2) || Subject)", sleep, REML=False)
    expanded = lmer(
        "Reaction ~ Days + (1 | Subject) + (0 + poly(Days, 2) | Subject)", sleep, REML=False
    )
    # lme4 2.1.0: 4 theta and logLik -874.5684.
    assert len(fit.theta) == 4
    assert fit.deviance == pytest.approx(expanded.deviance, rel=1e-8)
    assert fit.deviance == pytest.approx(2 * 874.5684, abs=2e-4)


def test_glmm_profiles_survive_points_where_full_newton_steps_overshoot():
    from mixedlm import load_cbpp
    from mixedlm.estimation.joint_glmm import JointGLMMObjective

    fit = glmer(
        "cbind(incidence, size - incidence) ~ 1 + (1 | herd)",
        load_cbpp(),
        family=families.Binomial(),
    )
    objective = JointGLMMObjective(fit.matrices, fit.family, fit.nAGQ, pirls_tol=fit.pirls_tol)
    # A full PIRLS step diverges here; lme4's step halving reaches the mode.
    _, _, _, converged = objective.evaluate(np.array([1.5, -3.0]))
    assert converged
    intervals = fit.confint(parm=["theta_", "beta_"], method="profile")
    # lme4 2.1.0: confint(glmer(..., cbpp, binomial)).
    assert_allclose(intervals[".sig01"], (0.50358841, 1.3242127), atol=2e-5)
    assert_allclose(intervals["(Intercept)"], (-2.57099779, -1.5556745), atol=2e-5)


def test_glmm_profile_nuisance_fits_accept_a_stationary_start(monkeypatch):
    from types import SimpleNamespace

    from mixedlm import load_cbpp
    from mixedlm.inference import glmm_profile

    fit = glmer(
        "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        load_cbpp(),
        family=families.Binomial(),
    )
    likelihood = glmm_profile._GLMMProfileLikelihood(fit, np.sqrt(np.diag(fit.vcov())))

    def stalled(objective, start, **_):
        return SimpleNamespace(
            success=False, x=np.asarray(start), fun=objective(start), message="ABNORMAL"
        )

    monkeypatch.setattr(glmm_profile, "run_optimizer", stalled)
    deviance, parameters = likelihood.fit(likelihood.start)
    assert deviance == pytest.approx(fit.deviance, rel=1e-8)
    assert_allclose(parameters, likelihood.start)
    away = likelihood.start.copy()
    away[1] += 0.5
    with pytest.raises(RuntimeError, match="nuisance optimization failed"):
        likelihood.fit(away)


def test_pirls_converges_with_large_counts_near_the_mode():
    rng = np.random.default_rng(1002)
    groups = np.repeat(np.arange(12), 30)
    x = rng.normal(size=360)
    effects = rng.normal(0, 1.0, 12)[groups]
    data = pd.DataFrame(
        {
            "y": rng.poisson(np.exp(14 + 0.3 * x + effects)).astype(float),
            "x": x,
            "a": [f"a{i}" for i in groups],
        }
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit = glmer("y ~ x + (1 | a)", data, family=families.Poisson())
    # Rounding in the deviance of counts near 1e7 must not stop the mode solve.
    assert fit.converged and fit.pirls_converged


def test_glmm_profile_fallback_does_not_accept_a_variance_boundary_below_the_optimum():
    from mixedlm.inference import glmm_profile

    # theta = 0 has no first-order slope, but the deviance falls inward.
    def objective(values):
        theta, beta = values
        return (beta - 1.0) ** 2 - theta**2 + theta**4

    bounds = [(0.0, None), (None, None)]
    assert not glmm_profile._GLMMProfileLikelihood._stationary(
        objective, np.array([0.0, 1.0]), bounds
    )
    assert glmm_profile._GLMMProfileLikelihood._stationary(
        lambda values: (values[1] - 1.0) ** 2 + values[0] ** 2, np.array([0.0, 1.0]), bounds
    )


def test_glmm_profiles_with_floored_working_weights_converge():
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(20), 10)
    x = rng.normal(size=200)
    effects = rng.normal(0, 0.7, 20)
    counts = rng.poisson(np.exp(1 + 0.5 * x + effects[groups]))
    rng.binomial(1, 0.5, size=200)
    data = pd.DataFrame(
        {
            "y": counts,
            "x": x,
            "g": [f"g{i:02d}" for i in groups],
            "w": np.exp(rng.uniform(np.log(1e-4), np.log(1e4), size=200)) * 1e-6,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = glmer("y ~ x + (1 | g)", data, family=families.Poisson(), weights="w")
    # Pre-change full steps and this solve give the same intervals.
    intervals = fit.confint(parm=["(Intercept)", "x"], method="profile")
    assert_allclose(intervals["(Intercept)"], (-4.8626, 3.1329), atol=2e-4)
    assert_allclose(intervals["x"], (-2.7923, 3.0478), atol=2e-4)


def test_singular_glmm_fits_reach_the_lower_boundary_optimum():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(15), 12)
    x, x2 = rng.normal(size=180), rng.normal(size=180)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.7 * x - 0.4 * x2)))),
            "x": x,
            "x2": x2,
            "g": [f"g{i:02d}" for i in groups],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = glmer("y ~ x + x2 + (1 + x | g)", data, family=families.Binomial())
    # lme4 2.1.0: -2logLik 226.980548, theta 0.2683446 0.7090361 0.
    assert fit.deviance == pytest.approx(226.980548, abs=1e-5)
    assert_allclose(fit.theta, [0.2683446, 0.7090361, 0.0], atol=1e-3)
    assert_allclose(fit.beta, [0.09650576, 0.6268355, -0.5213145], atol=1e-4)


def test_glmm_profiles_survive_saturated_newton_steps():
    data = pd.DataFrame(
        {
            "s": [96, 51, 153, 9, 100, 124, 89, 21, 153, 8, 34, 15, 179, 92, 108, 89],
            "n": [105, 54, 153, 9, 103, 133, 138, 21, 153, 8, 39, 15, 195, 101, 108, 93],
            "x": [
                0.5888349437484451,
                0.04211810294245316,
                -1.5709005193070567,
                1.0016547499330197,
                -0.09787618745319536,
                0.6198022056116699,
                1.8368321511859722,
                0.26842997179099165,
                -1.0744686765137588,
                -0.6809789689138194,
                1.0168262786662374,
                -1.4567682446941592,
                0.263589274161203,
                0.37526449952140556,
                -1.1503043956147625,
                0.16926391115467035,
            ],
            "g": [str(v) for v in np.repeat(np.arange(8), 2)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = glmer("cbind(s, n - s) ~ x + (1 | g)", data, family=families.Binomial())
    # lme4 2.1.0: (-2.1079, -1.2235), whose upper limit is spline-interpolated.
    assert_allclose(fit.confint(parm="x", method="profile")["x"], (-2.1079, -1.2235), atol=2e-3)


def test_singular_check_reports_the_kept_optimizer_without_foreign_options():
    from mixedlm import glmerControl

    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(15), 12)
    x, x2 = rng.normal(size=180), rng.normal(size=180)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.7 * x - 0.4 * x2)))),
            "x": x,
            "x2": x2,
            "g": [f"g{i:02d}" for i in groups],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = glmer(
            "y ~ x + x2 + (1 + x | g)",
            data,
            family=families.Binomial(),
            control=glmerControl(optCtrl={"final_tr_radius": 1e-6}),
        )
    assert fit.optimizer == "L-BFGS-B"
    assert not [w for w in caught if "Unknown solver options" in str(w.message)]
