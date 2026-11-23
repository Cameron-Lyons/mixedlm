"""Computed formula terms agree with fits on precomputed columns."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    anova_type3,
    cross_validate,
    emmeans,
    families,
    glmer,
    lmer,
)
from mixedlm.formula.parser import parse_formula, subbars, update_formula
from mixedlm.formula.terms import TransformTerm
from mixedlm.inference.drop1 import _droppable_fixed_terms
from mixedlm.matrices import build_model_matrices
from numpy.testing import assert_allclose

from tests._datasets import CBPP, SLEEPSTUDY

BINOMIAL = families.Binomial()


def _assert_same_fit(first, second, *, rtol=1e-6):
    assert_allclose(first.beta, second.beta, rtol=rtol, atol=1e-8)
    assert_allclose(first.theta, second.theta, rtol=1e-5, atol=1e-7)
    assert_allclose(first.deviance, second.deviance, rtol=1e-9)


@pytest.fixture(autouse=True)
def _quiet_singular_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# Parsing


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ (a + b + c)^2", "y ~ a + b + c + a:b + a:c + b:c"),
        ("y ~ (a + b)^3", "y ~ a + b + a:b"),
        ("y ~ (a + b + c)^2 - a:b", "y ~ a + b + c + a:c + b:c"),
        ("y ~ a * (b + c)", "y ~ a + b + c + a:b + a:c"),
        ("y ~ a / b", "y ~ a + a:b"),
        ("y ~ (a + b) / c", "y ~ a + b + a:b:c"),
        ("y ~ x + ((x + z)^2 | g)", "y ~ x + (1 + x + z + x:z | g)"),
    ],
)
def test_formula_algebra_follows_r(formula, expected):
    parsed = parse_formula(formula)
    assert str(parsed) == expected
    assert parse_formula(str(parsed)) == parsed


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("y ~ poly(x, 2)", "y ~ poly(x, degree = 2)"),
        ("y ~ scale(x)", "y ~ scale(x, center = TRUE, scale = T)"),
        ("y ~ poly(x, 2, raw = FALSE)", "y ~ poly(x, 2)"),
        ("y ~ C(a, sum)", "y ~ C(a, 'contr.sum')"),
        ("y ~ log(x, base = 2)", "y ~ log(x, 2)"),
        ("y ~ I(x^2)", "y ~ I(x**2)"),
    ],
)
def test_equivalent_spellings_are_the_same_term(first, second):
    assert parse_formula(first) == parse_formula(second)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ log(x + 1) + I(-x**2) + I((-x)**2) + I(a - (b - c)) + I(2**3**x)",
        "y ~ scale(log(x), scale = FALSE) + poly(x, 3, raw = TRUE)",
        'y ~ relevel(a, ref = "level \\"one\\"") + C(b, helmert) + factor(c)',
        "cbind(s, n - s) ~ x + offset(log(t)) + (log(x) || g)",
        "log(y) ~ sqrt(`a b`) + (1 | g)",
    ],
)
def test_computed_terms_round_trip(formula):
    parsed = parse_formula(formula)
    assert parse_formula(str(parsed)) == parsed


def test_computed_terms_track_source_columns():
    parsed = parse_formula(
        "cbind(s, n - s) ~ log(x + w) + poly(z, 2) + offset(log(t)) + (scale(v) | g)"
    )
    assert parsed.all_variables == {"s", "n", "x", "w", "z", "t", "v", "g"}
    assert parsed.fixed_variables == {"x", "w", "z"}
    assert parsed.offset_variables == {"t"}
    assert parsed.random_variables == {"v"}
    assert parsed.response_variables == {"s", "n"}
    poly = parsed.fixed.terms[1]
    assert isinstance(poly, TransformTerm)
    assert poly.column_names == ("poly(z, 2)1", "poly(z, 2)2")


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("y ~ x^2", r"use I\(x\*\*2\)"),
        ("y ~ log(scale(x))", "cannot be nested"),
        ("y ~ foo(x)", "Unknown formula function 'foo'"),
        ("scale(y) ~ x", "cannot transform a response"),
        ("y ~ x:cbind(a, b)", "only allowed as a response"),
        ("y ~ x:offset(t)", "only allowed as a separate additive term"),
        ("y ~ poly(x, 0)", "positive integer"),
        ("y ~ poly(x, 2, raw = yes)", "TRUE or FALSE"),
        ("y ~ C(a, nonsense)", "contrast name"),
        ("y ~ relevel(a)", "requires argument 'ref'"),
        ("y ~ log(x, base = 1)", "other than 1"),
        ("y ~ log(x, bse = 2)", "no argument 'bse'"),
        ("y ~ log(2)", "must refer to a data variable"),
        ("y ~ log('x')", "cannot use the constant"),
        ("cbind(s) ~ x", "takes 2 unnamed arguments"),
        ("y ~ offset(a, b)", "takes 1 unnamed argument"),
    ],
)
def test_invalid_computed_terms_raise_clear_errors(formula, message):
    with pytest.raises(ValueError, match=message):
        parse_formula(formula)


def test_formula_helpers_preserve_computed_terms():
    formula = parse_formula("cbind(s, n - s) ~ log(x) + offset(log(t)) + (scale(z) | g)")
    assert subbars(formula) == ("cbind(s, n - s) ~ log(x) + offset(log(t)) + g + g:scale(z)")
    removed = update_formula(formula, ". ~ . - log(x) - offset(log(t)) + poly(x, 2)")
    assert str(removed) == "cbind(s, n - s) ~ poly(x, 2) + (1 + scale(z) | g)"
    renamed = update_formula(formula, "log(y) ~ .")
    assert str(renamed) == "log(y) ~ log(x) + offset(log(t)) + (1 + scale(z) | g)"
    assert _droppable_fixed_terms(
        type("Model", (), {"formula": parse_formula("y ~ x + log(x) * z")})()
    ) == ["x", "log(x):z"]


# Predictors


def test_elementwise_transforms_match_precomputed_columns():
    data = SLEEPSTUDY.assign(D1=SLEEPSTUDY["Days"] + 1.0)
    fitted = lmer("Reaction ~ log(D1) + sqrt(Days) + (1 | Subject)", data)
    manual = lmer(
        "Reaction ~ a + b + (1 | Subject)",
        data.assign(a=np.log(data["D1"]), b=np.sqrt(data["Days"])),
    )
    _assert_same_fit(fitted, manual)
    assert fitted.matrices.fixed_names == ["(Intercept)", "log(D1)", "sqrt(Days)"]
    new = pd.DataFrame({"D1": [1.0, 8.0], "Days": [0.0, 7.0], "Subject": ["308", "309"]})
    expected = manual.predict(new.assign(a=np.log(new["D1"]), b=np.sqrt(new["Days"])))
    assert_allclose(fitted.predict(new), expected)


def test_scale_reuses_fitted_center_and_scale_for_new_data():
    days = SLEEPSTUDY["Days"]
    center, scale = days.mean(), days.std(ddof=1)
    fitted = lmer("Reaction ~ scale(Days) + (scale(Days) | Subject)", SLEEPSTUDY)
    manual = lmer("Reaction ~ s + (s | Subject)", SLEEPSTUDY.assign(s=(days - center) / scale))
    _assert_same_fit(fitted, manual, rtol=1e-5)
    assert fitted.matrices.transform_state["scale(Days)"] == pytest.approx((center, scale))

    # A single new row cannot be standardized by itself.
    new = pd.DataFrame({"Days": [9.0], "Subject": ["308"]})
    assert_allclose(
        fitted.predict(new), manual.predict(new.assign(s=(9.0 - center) / scale)), rtol=1e-6
    )


def test_scale_without_centering_uses_the_root_mean_square():
    data = SLEEPSTUDY.assign(D1=SLEEPSTUDY["Days"] + 1.0)
    fitted = lmer("Reaction ~ scale(D1, center = FALSE) + (1 | Subject)", data)
    rms = np.sqrt(np.sum(data["D1"] ** 2) / (len(data) - 1))
    manual = lmer("Reaction ~ s + (1 | Subject)", data.assign(s=data["D1"] / rms))
    _assert_same_fit(fitted, manual)


def test_orthogonal_polynomials_match_r():
    data = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    data["y"] = np.sin(data["x"])
    matrices = build_model_matrices(parse_formula("y ~ poly(x, 2)"), data)
    # R: poly(1:10, 2)
    expected_linear = (data["x"] - 5.5) / np.sqrt(82.5)
    quadratic = (data["x"] - 5.5) ** 2 - 8.25
    expected_quadratic = quadratic / np.sqrt(np.sum(quadratic**2))
    assert_allclose(matrices.X[:, 1], expected_linear, atol=1e-12)
    assert_allclose(matrices.X[:, 2], expected_quadratic, atol=1e-12)
    assert matrices.fixed_names == ["(Intercept)", "poly(x, 2)1", "poly(x, 2)2"]

    # New data uses the fitted basis: R's predict(poly(1:10, 2), 11).
    from mixedlm.matrices import build_fixed_matrix

    X, _ = build_fixed_matrix(
        parse_formula("y ~ poly(x, 2)"),
        pd.DataFrame({"x": [11.0]}),
        transform_state=matrices.transform_state,
    )
    assert_allclose(X[0, 1:], [5.5 / np.sqrt(82.5), 22.0 / np.sqrt(528.0)], atol=1e-12)


def test_polynomials_span_the_same_space_as_raw_powers():
    orthogonal = lmer("Reaction ~ poly(Days, 3) + (1 | Subject)", SLEEPSTUDY)
    raw = lmer("Reaction ~ poly(Days, 3, raw = TRUE) + (1 | Subject)", SLEEPSTUDY)
    powers = lmer("Reaction ~ Days + I(Days**2) + I(Days**3) + (1 | Subject)", SLEEPSTUDY)
    _assert_same_fit(raw, powers)
    assert_allclose(orthogonal.fitted(), powers.fitted(), rtol=1e-8)
    assert_allclose(orthogonal.predict(SLEEPSTUDY.iloc[:7]), powers.fitted()[:7], rtol=1e-8)
    table = anova_type3(orthogonal)
    assert table.num_df[table.terms.index("poly(Days, 3)")] == 3


def test_poly_requires_enough_unique_points():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="less than the number of unique points"):
        build_model_matrices(parse_formula("y ~ poly(x, 2)"), data)


def test_factor_codes_numeric_columns_with_fitted_levels():
    data = SLEEPSTUDY.assign(Days=SLEEPSTUDY["Days"].astype(int))
    fitted = lmer("Reaction ~ factor(Days) + (1 | Subject)", data)
    coded = lmer(
        "Reaction ~ D + (1 | Subject)",
        data.assign(D=data["Days"].astype(str).str.zfill(2)),
    )
    _assert_same_fit(fitted, coded)
    assert fitted.matrices.fixed_names[1] == "factor(Days).1"
    assert fitted.matrices.transform_state["factor(Days)"] == list(range(10))
    with pytest.raises(ValueError, match=r"New level\(s\) 12 in factor 'factor\(Days\)'"):
        fitted.predict(pd.DataFrame({"Days": [12], "Subject": ["308"]}))

    means = emmeans(fitted, "Days").result
    assert means.grid["Days"].tolist() == list(range(10))
    assert_allclose(means.emmean, emmeans(coded, "D").result.emmean)


def test_factors_and_factor_transforms_drop_unused_levels_as_lme4_does():
    data = SLEEPSTUDY.assign(D=pd.Categorical(SLEEPSTUDY["Days"].astype(int), categories=range(12)))
    for formula in ("Reaction ~ D", "Reaction ~ factor(D)", "Reaction ~ C(D)"):
        assert build_model_matrices(parse_formula(formula), data).X.shape[1] == 10
    # lme4 2.1.0: an unused first category is not the reference, and an unused
    # relevel() reference gives way to the first observed level.
    unused_first = SLEEPSTUDY.assign(
        D=pd.Categorical(SLEEPSTUDY["Days"].astype(int), categories=[11, *range(10)])
    )
    plain = build_model_matrices(parse_formula("Reaction ~ D + (0 + D | Subject)"), unused_first)
    assert plain.category_levels["D"] == list(range(10))
    assert plain.random_structures[0].term_names[0] == "D.0"
    shifted = build_model_matrices(parse_formula("Reaction ~ relevel(D, ref = 11)"), unused_first)
    assert shifted.transform_state["relevel(D, ref = 11)"] == list(range(10))


def test_factor_contrasts_and_reference_levels():
    data = SLEEPSTUDY.assign(D=SLEEPSTUDY["Days"].astype(int).astype(str))
    by_argument = lmer("Reaction ~ D + (1 | Subject)", data, contrasts={"D": "sum"})
    by_formula = lmer("Reaction ~ C(D, sum) + (1 | Subject)", data)
    by_label = lmer("Reaction ~ factor(D) + (1 | Subject)", data, contrasts={"factor(D)": "sum"})
    _assert_same_fit(by_formula, by_argument)
    _assert_same_fit(by_label, by_argument)

    releveled = lmer('Reaction ~ relevel(D, ref = "9") + (1 | Subject)', data)
    assert releveled.fixef()["(Intercept)"] == pytest.approx(
        data.loc[data["D"] == "9", "Reaction"].mean(), rel=0.05
    )
    with pytest.raises(ValueError, match="reference level '10'"):
        lmer('Reaction ~ relevel(D, ref = "10") + (1 | Subject)', data)


def test_factor_without_intercept_uses_every_level():
    data = SLEEPSTUDY.assign(D=SLEEPSTUDY["Days"].astype(int))
    matrices = build_model_matrices(parse_formula("Reaction ~ 0 + factor(D)"), data)
    assert matrices.X.shape[1] == 10
    assert matrices.fixed_names[0] == "factor(D).0"


def test_transforms_reject_domain_errors_and_drop_missing_sources():
    with pytest.raises(ValueError, match=r"'log\(Days\)' is not finite for 18 observation"):
        lmer("Reaction ~ log(Days) + (1 | Subject)", SLEEPSTUDY)
    with pytest.raises(ValueError, match="requires a numeric variable"):
        lmer("Reaction ~ log(Subject) + (1 | Subject)", SLEEPSTUDY)

    data = SLEEPSTUDY.copy()
    data.loc[[3, 40], "Days"] = np.nan
    fit = lmer("Reaction ~ log1p(Days) + (1 | Subject)", data)
    assert fit.nobs() == len(data) - 2


def test_polars_input_matches_pandas():
    pl = pytest.importorskip("polars")
    formula = "Reaction ~ scale(Days) + poly(Days, 2) + factor(Subject) + (1 | Subject)"
    pandas_fit = lmer(formula, SLEEPSTUDY)
    polars_fit = lmer(formula, pl.from_pandas(SLEEPSTUDY))
    _assert_same_fit(polars_fit, pandas_fit, rtol=1e-8)
    assert_allclose(
        polars_fit.predict(pl.from_pandas(SLEEPSTUDY.head(4))),
        pandas_fit.predict(SLEEPSTUDY.head(4)),
    )


def test_cross_validation_learns_transforms_from_training_folds():
    model = lmer("Reaction ~ scale(Days) + (1 | Subject)", SLEEPSTUDY)
    folds = [np.arange(0, 90), np.arange(90, 180)]
    result = cross_validate(
        model,
        cv=[(np.setdiff1d(np.arange(180), test), test) for test in folds],
        metrics=["mse"],
    )
    for test in folds:
        train = SLEEPSTUDY.drop(index=test)
        refit = lmer("Reaction ~ scale(Days) + (1 | Subject)", train.reset_index(drop=True))
        expected = refit.predict(SLEEPSTUDY.iloc[test], allow_new_levels=True)
        assert_allclose(result.predictions[test], expected, rtol=1e-6)


# Offsets


@pytest.fixture(scope="module")
def exposure_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(12), 15)
    exposure = rng.uniform(0.5, 4.0, groups.size)
    x = rng.standard_normal(groups.size)
    mu = exposure * np.exp(0.3 + 0.5 * x + rng.normal(0, 0.4, 12)[groups])
    return pd.DataFrame(
        {"y": rng.poisson(mu).astype(float), "x": x, "t": exposure, "g": groups.astype(str)}
    )


def test_formula_offsets_match_supplied_offsets(exposure_data):
    poisson = families.Poisson()
    in_formula = glmer("y ~ x + offset(log(t)) + (1 | g)", exposure_data, family=poisson)
    supplied = glmer(
        "y ~ x + (1 | g)", exposure_data, family=poisson, offset=np.log(exposure_data["t"])
    )
    _assert_same_fit(in_formula, supplied)
    assert_allclose(in_formula.offset(), np.log(exposure_data["t"]))

    new = exposure_data.iloc[:6]
    assert_allclose(
        in_formula.predict(new), supplied.predict(new, offset=np.log(new["t"])), rtol=1e-8
    )
    # A supplied prediction offset adds to the formula's terms.
    assert_allclose(
        in_formula.predict(new, offset=1.0, type="link"),
        supplied.predict(new, offset=np.log(new["t"]) + 1.0, type="link"),
        rtol=1e-8,
    )
    with pytest.raises(ValueError, match="missing offset variable"):
        in_formula.predict(new.drop(columns="t"))


def test_refits_add_formula_offsets_once(exposure_data):
    poisson = families.Poisson()
    model = glmer("y ~ x + offset(log(t)) + (1 | g)", exposure_data, family=poisson)
    _assert_same_fit(model.update(), model)
    without = model.update(". ~ . - offset(log(t))")
    plain = glmer("y ~ x + (1 | g)", exposure_data, family=poisson)
    _assert_same_fit(without, plain)

    both = glmer(
        "y ~ x + offset(log(t)) + (1 | g)", exposure_data, family=poisson, offset=np.full(180, 0.2)
    )
    assert_allclose(both.offset(), np.log(exposure_data["t"]) + 0.2)
    _assert_same_fit(both.update(), both)
    assert np.all(np.isfinite(both.drop1(exposure_data).aic))

    linear = lmer("log1p(y) ~ x + offset(0.5 * x) + (1 | g)", exposure_data)
    manual = lmer(
        "z ~ x + (1 | g)",
        exposure_data.assign(z=np.log1p(exposure_data["y"]) - 0.5 * exposure_data["x"]),
    )
    _assert_same_fit(linear, manual)


# Responses


def test_cbind_matches_successes_over_trials():
    expected = glmer("incidence / size ~ period + (1 | herd)", CBPP, family=BINOMIAL)
    expression = glmer(
        "cbind(incidence, size - incidence) ~ period + (1 | herd)", CBPP, family=BINOMIAL
    )
    columns = glmer(
        "cbind(incidence, failures) ~ period + (1 | herd)",
        CBPP.assign(failures=CBPP["size"] - CBPP["incidence"]),
        family=BINOMIAL,
    )
    for fit in (expression, columns):
        _assert_same_fit(fit, expected)
        assert_allclose(fit.logLik().value, expected.logLik().value)
    with pytest.raises(ValueError, match="require a binomial GLMM"):
        lmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", CBPP)


@pytest.mark.parametrize("failures", ["size - incidence", "failures"])
def test_cbind_refits_keep_trial_counts(failures):
    data = CBPP.assign(failures=CBPP["size"] - CBPP["incidence"])
    fit = glmer(f"cbind(incidence, {failures}) ~ period + (1 | herd)", data, family=BINOMIAL)
    counts = glmer("incidence / size ~ period + (1 | herd)", data, family=BINOMIAL)
    simulated = fit.simulate(nsim=1, seed=11).ravel()
    refit = fit.refit(simulated)
    expected = counts.refit(simulated)
    _assert_same_fit(refit, expected)
    assert_allclose(refit.matrices.trials, data["size"])
    _assert_same_fit(refit.update(), expected)


def test_transformed_responses():
    fitted = lmer("log(Reaction) ~ Days + (1 | Subject)", SLEEPSTUDY)
    manual = lmer("r ~ Days + (1 | Subject)", SLEEPSTUDY.assign(r=np.log(SLEEPSTUDY["Reaction"])))
    _assert_same_fit(fitted, manual)
    assert fitted.formula.response == "log(Reaction)"

    simulated = fitted.simulate(nsim=1, seed=5).ravel()
    refit = fitted.refit(simulated)
    _assert_same_fit(refit, manual.refit(simulated))
    _assert_same_fit(refit.update(), refit)

    with pytest.raises(ValueError, match="not finite"):
        lmer("log(Days) ~ Reaction + (1 | Subject)", SLEEPSTUDY)


def test_dot_in_a_new_response_stands_for_the_old_response():
    formula = parse_formula("y ~ x + (1 | g)")
    assert str(update_formula(formula, "log(.) ~ .")) == "log(y) ~ x + (1 | g)"
    assert str(update_formula(formula, "cbind(., n - .) ~ .")) == ("cbind(y, n - y) ~ x + (1 | g)")
    transformed = update_formula(formula, "log(.) ~ .")
    assert str(update_formula(transformed, "I(. * 2) ~ . - x")) == "I(log(y) * 2) ~ 1 + (1 | g)"
    with pytest.raises(ValueError, match="grouped response"):
        update_formula(parse_formula("s / n ~ x"), "log(.) ~ .")
    fit = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
    manual = lmer("log(Reaction) ~ Days + (1 | Subject)", SLEEPSTUDY)
    _assert_same_fit(fit.update("log(.) ~ ."), manual)


# Review fixes


@pytest.mark.parametrize(
    ("formula", "operator", "position"),
    [("y ~ x^2", "'^'", 5), ("y ~ x**2", "'**'", 5), ("y ~ a + b + x^3", "'^'", 13)],
)
def test_single_predictor_powers_report_the_written_operator(formula, operator, position):
    with pytest.raises(ValueError, match=rf"{re.escape(operator)} at position {position} "):
        parse_formula(formula)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ I(x / 5.)", "y ~ I(x / 5)"),
        ("y ~ I(x * 1.e3)", "y ~ I(x * 1000)"),
        ("y ~ log(x + 1.) + offset(2.*x)", "y ~ log(x + 1) + offset(2 * x)"),
        ("y ~ I(x / log(2)) + I(sqrt(2) * x)", "y ~ I(x / log(2)) + I(sqrt(2) * x)"),
        ("y ~ exp(-x / exp(1))", "y ~ exp(-x / exp(1))"),
    ],
)
def test_numbers_and_constant_subexpressions_follow_r(formula, expected):
    parsed = parse_formula(formula)
    assert str(parsed) == expected
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 4.0]})
    assert np.all(np.isfinite(build_model_matrices(parsed, data).X))
    with pytest.raises(ValueError, match="must refer to a data variable"):
        parse_formula("y ~ log(2)")


def test_terms_are_ordered_by_degree_as_in_r():
    formula = parse_formula("y ~ a*b*c + x:z + x + (1 + x:z + x | g)")
    assert str(formula) == ("y ~ a + b + c + x + a:b + a:c + b:c + x:z + a:b:c + (1 + x + x:z | g)")
    rng = np.random.default_rng(5)
    data = pd.DataFrame(
        {"y": rng.normal(size=120), "x": rng.normal(size=120), "z": rng.normal(size=120)}
    )
    data["g"] = np.repeat(np.arange(12), 10).astype(str)
    fit = lmer("y ~ x:z + x + (1 + x:z + x | g)", data)
    assert fit.matrices.fixed_names == ["(Intercept)", "x", "x:z"]
    # lme4 2.1.0: m@cnms$g is "(Intercept)" "x" "x:z", so .sig02 is sd(x).
    assert [p.label for p in fit.variance_parameters()[:3]] == [
        "sd_(Intercept)|g",
        "sd_x|g",
        "sd_x:z|g",
    ]


def test_formulas_pickled_before_offset_terms_still_work():
    formula = parse_formula("y ~ x + (1 | g)")
    del formula.__dict__["offsets"]  # as restored from a pickle without the field
    assert str(formula) == "y ~ x + (1 | g)"
    assert str(update_formula(formula, ". ~ . + z")) == "y ~ x + z + (1 | g)"


def test_scale_and_poly_learn_from_rows_before_missing_values_are_dropped():
    data = SLEEPSTUDY.copy()
    data.loc[[0, 5, 17, 40], "Reaction"] = np.nan
    fit = lmer("Reaction ~ scale(Days) + poly(Days, 2) + (1 | Subject)", data)
    days = SLEEPSTUDY["Days"]
    assert fit.matrices.transform_state["scale(Days)"] == pytest.approx(
        (days.mean(), days.std(ddof=1))
    )
    # lme4 2.1.0 on the same data, which also drops the aliased poly(Days, 2)1.
    assert_allclose(list(fit.fixef().values()), [298.37765821, 30.03273928, 41.21235438], rtol=1e-7)


def test_dot_responses_of_refits_use_the_refit_response():
    fit = lmer("log(Reaction) ~ Days + (1 | Subject)", SLEEPSTUDY)
    refit = fit.refit(fit.simulate(seed=2))
    manual = lmer("r ~ Days + (1 | Subject)", SLEEPSTUDY.assign(r=np.exp(refit.matrices.y)))
    _assert_same_fit(refit.update("exp(.) ~ ."), manual)


def test_simulated_formulas_include_offsets():
    from mixedlm import simulate_formula

    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=2000),
            "t": rng.uniform(1, 30, 2000),
            "g": rng.integers(0, 40, 2000).astype(str),
        }
    )
    formula = "y ~ x + offset(log(t)) + (1 | g)"
    simulated = simulate_formula(
        formula, data, family="poisson", beta={"(Intercept)": 0.0, "x": 0.5}, theta=[0.3], seed=3
    )
    fit = glmer(formula, simulated, family=families.Poisson())
    assert_allclose(fit.beta, [0.0, 0.5], atol=0.05)


def test_lmlist_subtracts_formula_offsets():
    from mixedlm import lmList

    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {"x": rng.normal(size=400), "o": rng.normal(size=400), "g": np.repeat(range(20), 20)}
    )
    data["y"] = 1 + 2 * data["x"] + data["o"] + rng.normal(scale=0.1, size=400)
    fits = lmList("y ~ x + offset(o) | g", data)
    assert_allclose(fits["coef"]["x"], 2.0, atol=0.1)
    assert_allclose(fits["coef"]["(Intercept)"], 1.0, atol=0.1)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x + offset(log(t)) + (1 | g)",
        "cbind(s, n - s) ~ x + (1 | g)",
        "cbind(s, f) ~ x + (1 | g)",
        "s / n ~ x + (1 | g)",
        "log(y) ~ x + (1 | g)",
    ],
)
def test_data_templates_cover_offsets_and_grouped_responses(formula):
    from mixedlm import glFormula, lFormula, mkDataTemplate, mkMinimalData

    for template in (mkDataTemplate(formula, nlevs={"g": 4}), mkMinimalData(formula)):
        if "cbind" in formula or "/" in formula:
            glFormula(formula, template, family=BINOMIAL)
        else:
            lFormula(formula, template)


@pytest.mark.parametrize(
    ("new", "error"),
    [
        (".~.+Days^2", "'^' at position 8 "),
        (".~Days**2", "'**' at position 6 "),
        ("Reaction~Days^2+(1|Subject)", "'^' at position 13 "),
        (". ~ . + 2", "number 2 at position 8;"),
    ],
)
def test_update_errors_report_positions_in_the_new_formula(new, error):
    fit = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)
    with pytest.raises(ValueError, match=re.escape(error)):
        fit.update(new)


def test_lmlist_errors_report_positions_in_the_given_formula():
    from mixedlm import lmList

    with pytest.raises(ValueError, match=re.escape("'^' at position 18 ")):
        lmList("   Reaction ~ Days^2 | Subject", SLEEPSTUDY)


@pytest.mark.parametrize(
    ("formula", "position"), [("y ~ x^²", 6), ("y ~ (a + b)^²", 12), ("y ~ (a + b)**²", 13)]
)
def test_non_ascii_digits_are_rejected_at_their_position(formula, position):
    with pytest.raises(ValueError, match=f"Unexpected character '²' at position {position}"):
        parse_formula(formula)


def test_numbers_with_trailing_dots_are_intercepts_and_powers():
    assert str(parse_formula("y ~ Days - 1. + (0. + Days | g) + (1. | g)")) == str(
        parse_formula("y ~ 0 + Days + (0 + Days | g) + (1 | g)")
    )
    assert str(parse_formula("y ~ (a + b)^2.")) == "y ~ a + b + a:b"
    assert str(update_formula(parse_formula("y ~ 0 + x"), ". ~ . + 1.")) == "y ~ x"
    fit = lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY).update(". ~ . - 1.")
    # lme4 2.1.0: fixef(lmer(Reaction ~ Days - 1. + (1 | Subject), sleepstudy))
    assert fit.fixef() == pytest.approx({"Days": 10.67193}, abs=1e-5)


@pytest.mark.parametrize("response", ["I(2 * 3)", "log(2)", "sqrt(4)", "I(1)"])
def test_constant_responses_are_rejected(response):
    message = f"The response {response} at position 0 must refer to a data column"
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_formula(f"{response} ~ x + (1 | g)")
    with pytest.raises(ValueError, match=re.escape(message)):
        lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY).update(f"{response} ~ .")


def test_constant_options_are_evaluated():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0, 5.0], "x": [1.0, 2.0, 4.0, 8.0]})
    for formula, expected, columns in [
        # The natural base is log()'s default, so it is dropped as R's default.
        ("y ~ log(x, base = exp(1))", "y ~ log(x)", [np.log(data.x)]),
        ("y ~ log(x, base = 2)", "y ~ log(x, base = 2)", [np.log2(data.x)]),
        ("y ~ log(x, base = 1/2)", "y ~ log(x, base = 0.5)", [-np.log2(data.x)]),
        ("y ~ poly(x, 1 + 1, raw = TRUE)", "y ~ poly(x, 2, raw = TRUE)", [data.x, data.x**2]),
    ]:
        parsed = parse_formula(formula)
        assert str(parsed) == expected
        assert_allclose(build_model_matrices(parsed, data).X[:, 1:], np.column_stack(columns))


def test_updates_of_refits_keep_the_refit_response():
    fit = lmer("log(Reaction) ~ Days + (1 | Subject)", SLEEPSTUDY, REML=False)
    simulated = fit.simulate(seed=3).ravel()
    refit = fit.refit(simulated)
    updated = refit.update()
    assert_allclose(updated.matrices.y, simulated)
    _assert_same_fit(updated.update(), refit)
    bigger = refit.update(". ~ . + I(Days**2)")
    _assert_same_fit(bigger.update(". ~ . - I(Days**2)"), refit)
    # The stored response stays internal.
    assert sorted(updated.model_frame().columns) == ["Days", "Reaction", "Subject"]
    cross_validate(updated, cv=3, random_state=1)


def test_computed_responses_ignore_columns_named_like_the_expression():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "a": rng.uniform(1, 3, 120),
            "b": rng.uniform(1, 2, 120),
            "x": rng.normal(size=120),
            "g": np.repeat(np.arange(12), 10).astype(str),
            "a + b": rng.uniform(1, 50, 120),
            "log(a)": rng.uniform(1, 50, 120),
        }
    )
    assert_allclose(lmer("log(a + b) ~ x + (1 | g)", data).matrices.y, np.log(data.a + data.b))
    assert_allclose(lmer("log(a) ~ x + (1 | g)", data).matrices.y, np.log(data.a))


def test_refits_reuse_transforms_learned_before_missing_rows_were_dropped():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 10, 150)
    x = rng.normal(size=150)
    z = rng.normal(size=150)
    y = 1 + x + 0.5 * z + rng.normal(size=150) + rng.normal(size=10)[g]
    data = pd.DataFrame({"y": y, "x": x, "z": z, "g": [f"g{i}" for i in g]})
    data.loc[[1, 5, 9], "y"] = np.nan
    data.loc[[20, 21, 22, 23], "z"] = np.nan
    fit = lmer("y ~ scale(x) + z + (scale(x) || g)", data)
    updated = fit.update()
    assert updated.matrices.transform_state == fit.matrices.transform_state
    _assert_same_fit(updated, fit)
    assert fit.update(REML=False).deviance == pytest.approx(fit.refitML().deviance, rel=1e-9)
    assert_allclose(fit.drop1(fit.model_frame()).lrt, fit.drop1(data).lrt, rtol=1e-6)


def test_non_finite_transforms_report_positions_after_missing_rows_are_removed():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=40) + 3,
            "x": rng.uniform(1, 2, 40),
            "g": [f"g{i % 5}" for i in range(40)],
        }
    )
    data.loc[[0, 1, 2, 3], "y"] = np.nan
    data.loc[10, "x"] = 0.0
    message = "first at position 6 among the rows evaluated after removing missing values"
    with pytest.raises(ValueError, match=message):
        lmer("y ~ log(x) + (1 | g)", data)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x + offset(log(x)) + (1 | g)",
        "y ~ log(x) + (1 | g)",
        "y ~ sqrt(x) + (1 | g)",
        "log(p / (1 - p)) ~ x + (1 | g)",
        "asin(sqrt(p)) ~ x + (1 | g)",
    ],
)
def test_data_templates_respect_transform_domains(formula):
    from mixedlm import lFormula, mkDataTemplate, mkMinimalData

    np.random.seed(0)
    for template in (mkDataTemplate(formula, nlevs={"g": 4}), mkMinimalData(formula)):
        lFormula(formula, template)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("cbind(size - incidence, incidence) ~ period + (1 | herd)", ["(Intercept)", "period"]),
        ("cbind(s, n - s) ~ log(n) + (1 | g)", ["(Intercept)", "log(n)"]),
        ("s / n ~ n + (1 | g)", ["(Intercept)", "n"]),
    ],
)
def test_grouped_binomial_templates_can_be_fitted(formula, expected):
    from mixedlm import mkDataTemplate, mkMinimalData

    np.random.seed(0)
    for template in (mkDataTemplate(formula), mkMinimalData(formula)):
        assert list(glmer(formula, template, family=BINOMIAL).fixef()) == expected


def test_template_columns_follow_the_formula():
    from mixedlm import mkMinimalData

    template = mkMinimalData("cbind(s, a + b + c) ~ x + (1 | g)")
    assert list(template.columns) == ["s", "a", "b", "c", "x", "g"]


def test_simulated_cbind_responses_keep_the_trials():
    from mixedlm import simulate_formula

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "s": np.full(2000, 5),
            "f": np.full(2000, 5),
            "n": np.full(2000, 10),
            "x": rng.standard_normal(2000),
            "g": np.repeat(np.arange(100), 20).astype(str),
        }
    )
    options = dict(beta={"(Intercept)": 2.0, "x": 0.0}, theta=[0.01], seed=1)
    for formula in ("cbind(s, f) ~ x + (1 | g)", "cbind(s, n - s) ~ x + (1 | g)"):
        simulated = simulate_formula(formula, data, family="binomial", **options)
        if "f)" in formula:
            assert_allclose(simulated["s"] + simulated["f"], 10)
        fit = glmer(formula, simulated, family=BINOMIAL)
        assert fit.fixef()["(Intercept)"] == pytest.approx(2.0, abs=0.1)
    with pytest.raises(ValueError, match="cannot update the failures f - 1"):
        simulate_formula("cbind(s, f - 1) ~ x + (1 | g)", data, family="binomial", **options)
    with pytest.raises(ValueError, match=re.escape("simulate the column behind log(y)")):
        simulate_formula("log(y) ~ x + (1 | g)", data.assign(y=1.0), **options)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        # lme4 2.1.0 colnames(lFormula(...)$X): PowerTerms builds (a + b)^2 as a, a:b, b.
        (
            "y ~ (a + b)^2 * (c + d)^2",
            "y ~ a + b + c + d + a:b + c:d + a:c + a:d + b:c + b:d"
            " + a:c:d + a:b:c + a:b:d + b:c:d + a:b:c:d",
        ),
        ("y ~ (a + b)^2:(c + a)", "y ~ a + a:c + a:b + b:c + a:b:c"),
    ],
)
def test_crossing_powers_follow_r_term_order(formula, expected):
    from mixedlm.formula.terms import format_term

    terms = [format_term(term) for term in parse_formula(formula).fixed.terms]
    assert terms == expected.removeprefix("y ~ ").split(" + ")


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x:z + z",
        "y ~ c:a + a + c",
        "y ~ a + b + b:a",
        "y ~ x:a + z:b + b + a",
        "y ~ c:b:a + a:c + b",
        "y ~ x + (1 + x:z + z | g)",
        "y ~ (a + b)^2 * (c + d)^2",
    ],
)
def test_printed_formulas_parse_to_the_same_terms(formula):
    parsed = parse_formula(formula)
    again = parse_formula(str(parsed))
    assert (again.fixed.terms, again.random) == (parsed.fixed.terms, parsed.random)


def test_printed_formulas_keep_written_interaction_order_as_r_does():
    # R's formula(fit) prints y ~ x:z + z; mixedlm prints it the same way.
    assert str(parse_formula("y ~ x:z + z")) == "y ~ x:z + z"
    assert str(parse_formula("y ~ a + b + b:a")) == "y ~ a + b + a:b"


def test_updates_relabel_interactions_as_r_update_does():
    # lme4 2.1.0: update(fit, . ~ . + c:a + a + c) has the term a:c.
    assert str(update_formula(parse_formula("y ~ x + (1 | g)"), ". ~ . + c:a + a + c")) == (
        "y ~ x + a + c + a:c + (1 | g)"
    )
    assert str(update_formula(parse_formula("y ~ x:a + a"), ". ~ . + z")) == "y ~ a + z + a:x"


@pytest.mark.parametrize(
    ("new", "expected"),
    [
        # As R's update(), '.' stands for the old terms wherever it appears.
        (". ~ z + .", "y ~ z + x + w + (1 | g)"),
        (". ~ .^2", "y ~ x + w + x:w + (1 | g)"),
        (". ~ . * z", "y ~ x + w + z + x:z + w:z + (1 | g)"),
        (". ~ (.)", "y ~ x + w + (1 | g)"),
        (". ~ . + z - .", "y ~ z"),
        (". ~ . + (1 | g)", "y ~ x + w + (1 | g)"),
    ],
)
def test_updates_substitute_the_old_formula_for_each_dot(new, expected):
    assert str(update_formula(parse_formula("y ~ x + w + (1 | g)"), new)) == expected


def test_updates_order_terms_as_r_update_does():
    # lme4 2.1.0: update(fit, . ~ k + .) on y ~ 0 + f codes k with indicators, then f.
    zero = update_formula(parse_formula("y ~ 0 + f + (1 | g)"), ". ~ k + .")
    assert str(zero) == "y ~ 0 + k + f + (1 | g)"
    for new in (". ~ .", ". ~ f:k + k + (1 | g)"):
        assert str(update_formula(parse_formula("y ~ f:k + k + (1 | g)"), new)) == (
            "y ~ k + k:f + (1 | g)"
        )


def test_formula_helmert_contrasts_are_r_contr_helmert():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {"a": np.tile(["a1", "a2", "a3"], 40), "g": np.repeat([f"g{i}" for i in range(12)], 10)}
    )
    data["y"] = data["a"].map({"a1": 0.0, "a2": 2.0, "a3": 5.0}) + rng.normal(size=120)
    # lme4 2.1.0: fixef(lmer(y ~ C(a, helmert) + (1 | g), d)).
    expected = [2.4145, 1.0969, 1.4792]
    formula_fit = lmer("y ~ C(a, helmert) + (1 | g)", data)
    named_fit = lmer("y ~ a + (1 | g)", data, contrasts={"a": "contr.helmert"})
    assert_allclose(formula_fit.beta, expected, atol=1e-4)
    assert_allclose(named_fit.beta, expected, atol=1e-4)
    unit = lmer("y ~ a + (1 | g)", data, contrasts={"a": "helmert"})
    assert_allclose(unit.beta[1:], np.array(expected[1:]) * np.sqrt([2.0, 6.0]), atol=1e-3)


@pytest.mark.parametrize(
    "formula", ["y ~ log(age - 18) + (1 | g)", "y ~ sqrt(-x) + (1 | g)", "y ~ log(1 - x) + (1 | g)"]
)
def test_data_templates_find_shifted_transform_domains(formula):
    from mixedlm import lFormula, mkDataTemplate, mkMinimalData

    np.random.seed(0)
    for template in (mkDataTemplate(formula, nlevs={"g": 4}), mkMinimalData(formula)):
        lFormula(formula, template)


def test_data_templates_warn_when_no_values_fit_the_domain():
    from mixedlm import mkMinimalData

    with pytest.warns(UserWarning, match="No template values keep every transform"):
        mkMinimalData("y ~ log(x - 1e6) + (1 | g)")


def test_binomial_factor_responses_drop_unused_levels():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": rng.normal(size=200), "g": np.repeat(np.arange(20), 10).astype(str)})
    outcome = np.where(rng.uniform(size=200) < 1 / (1 + np.exp(-(0.3 + data["x"]))), "yes", "no")
    declared = data.assign(y=pd.Categorical(outcome, categories=["maybe", "no", "yes"]))
    plain = data.assign(y=pd.Categorical(outcome, categories=["no", "yes"]))
    _assert_same_fit(
        glmer("y ~ x + (1 | g)", declared, family=BINOMIAL),
        glmer("y ~ x + (1 | g)", plain, family=BINOMIAL),
    )


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ log(x) + sqrt(-z) + (1 | g)",
        "y ~ I(x^0.5) + (1 | g)",
        "y ~ log(age - 18) + log(1 - p) + (1 | g)",
    ],
)
def test_data_templates_find_each_variables_domain(formula):
    from mixedlm import lFormula, mkMinimalData

    np.random.seed(0)
    lFormula(formula, mkMinimalData(formula))


@pytest.mark.parametrize(
    ("new", "expected"),
    [
        # R keeps the old '- 1' wherever '.' stands, unless a later 1 restores it.
        (". ~ (.)", "y ~ 0 + x + z + (1 | g)"),
        (". ~ .^2", "y ~ 0 + x + z + x:z + (1 | g)"),
        (". ~ .:w", "y ~ 0 + x:w + w:z + (1 | g)"),
        (". ~ 1 + .", "y ~ 0 + x + z + (1 | g)"),
        (". ~ . + 1", "y ~ x + z + (1 | g)"),
    ],
)
def test_updates_keep_the_old_intercept_removal_at_each_dot(new, expected):
    assert str(update_formula(parse_formula("y ~ 0 + x + z + (1 | g)"), new)) == expected


def test_updates_of_formulas_without_terms_and_their_bars():
    assert str(update_formula(parse_formula("y ~ 1"), ". ~ w + .")) == "y ~ w"
    assert str(update_formula(parse_formula("y ~ 1"), ". ~ (1 | g) + .")) == "y ~ 1 + (1 | g)"
    old = parse_formula("y ~ x + (1 | h) + (1 | g)")
    # As R's simplification does, each bar stays once at its first place.
    assert str(update_formula(old, ". ~ . + (1 | h)")) == "y ~ x + (1 | h) + (1 | g)"
    assert str(update_formula(old, ". ~ (1 | g) + .")) == "y ~ x + (1 | g) + (1 | h)"
    doubled = parse_formula("y ~ x + (1 | g) + (1 | g)")
    assert str(update_formula(doubled, ". ~ .")) == "y ~ x + (1 | g)"


@pytest.mark.parametrize(
    ("new", "error"),
    [
        (". ~ .:w +", "at position 9"),
        (". ~ .:w + (1 | g", "at position 16"),
        (". ~ . + w $", "'$' at position 10"),
        ("    . ~ . + w $", "'$' at position 14"),
        (". ~ . + `w", "starting at position 8"),
    ],
)
def test_update_errors_point_into_the_update_as_written(new, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        update_formula(parse_formula("y ~ x + (1 + x | g)"), new)


def test_printed_grouped_interactions_parse_back_to_the_same_terms():
    for text in ("y ~ (a + b):c + a:b", "y ~ (t + q + p):u + (t + r + q):p"):
        parsed = parse_formula(text)
        assert parse_formula(str(parsed)).fixed.terms == parsed.fixed.terms


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # R reverses the old '- 1' in a subtracted term and ignores it in calls.
        ("y ~ 0 + x + z + (1 | g)", ". ~ . + w - .", "y ~ w"),
        ("y ~ 0 + x*z + (1 | g)", ". ~ . - .:z", "y ~ x + (1 | g)"),
        ("y ~ 0 + x + (1 | g)", ". ~ w + (. | h)", "y ~ w + (0 + x | h) + (1 | g)"),
        # Inside a call the '.' is the old right-hand side as written, as in R.
        ("y ~ 0 + x + (1 | g)", ". ~ w + I(.^2)", "y ~ w + I((0 + x)**2) + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ .^2", "y ~ x + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ (.)^2 + w", "y ~ x + w + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ . - 1e-0", "y ~ 0 + x + (1 | g)"),
        ("y ~ 0 + x + (1 | g)", ". ~ . + 1e+0", "y ~ x + (1 | g)"),
    ],
)
def test_updates_read_dots_and_intercepts_as_r_does(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


def test_parenthesized_single_terms_cross_while_bare_powers_raise():
    assert str(parse_formula("y ~ (x)^2")) == "y ~ x"
    with pytest.raises(ValueError, match="crosses formula terms"):
        parse_formula("y ~ x^2")


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # R substitutes the old formula as written, including an explicit 1.
        (
            "y ~ a:b + c + (1 | g)",
            ". ~ (. + v)^2",
            "y ~ c + v + a:b + c:v + c:a:b + v:a:b + (1 | g)",
        ),
        ("y ~ z:a + a:x + (1 | g)", ". ~ .:w", "y ~ z:a:w + a:w:x + (1 | g)"),
        ("y ~ 1 + x", ". ~ . - .", "y ~ 0"),
        ("y ~ x", ". ~ . - .", "y ~ 1"),
    ],
)
def test_updates_substitute_the_old_formula_as_written(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


def test_formula_sources_do_not_affect_equality_or_survive_changes():
    first, second = parse_formula("y ~ x + z"), parse_formula("y ~ x+z")
    assert first == second and first.source != second.source
    # An update's result is R's simplified formula, so it drops the source.
    assert update_formula(first, ". ~ . + w").source is None


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # R reverses the old intercept term for a '.' subtracted inside parentheses.
        ("y ~ x - 1", ". ~ (k - .)", "y ~ k"),
        ("y ~ 1 + x", ". ~ (k - .)", "y ~ 0 + k"),
        ("y ~ 0 - c + a + (1 | g)", ". ~ . + (1 + . | h)", "y ~ 0 + a + (1 | g) + (0 + a | h)"),
    ],
)
def test_updates_read_nested_subtraction_and_bars_as_r_does(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


def test_updates_keep_the_written_intercept_after_covariance_changes():
    from mixedlm import nobars, set_cov_type

    formula = parse_formula("y ~ 1 + x + (1 + z | g)")
    for changed in (set_cov_type(formula, "cs"), set_cov_type(formula, "ar1")):
        assert str(update_formula(changed, ". ~ 0 + .")) == "y ~ x + (1 + z | g)"
    assert str(update_formula(nobars(formula), ". ~ 0 + .")) == "y ~ x"


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # lme4 2.1.0 lFormula()$reTrms$cnms of R's update() of the same formulas.
        ("y ~ 0 + x", ". ~ . + (. * z | g)", "y ~ 0 + x + (0 + x + z + x:z | g)"),
        ("y ~ 0 + x", ". ~ . + (.:z | g)", "y ~ 0 + x + (0 + x:z | g)"),
        ("y ~ 1 + x", ". ~ . + (0 + . | g)", "y ~ x + (1 + x | g)"),
        ("y ~ 1 + x", ". ~ . + (z - . | g)", "y ~ x + (0 + z | g)"),
        ("y ~ 0 + x", ". ~ . + (0 + z - . | g)", "y ~ 0 + x + (1 + z | g)"),
        ("y ~ 1 + x", ". ~ log(.)", "y ~ log(1 + x)"),
        ("y ~ x + 1", ". ~ . + I(.)", "y ~ x + I(x + 1)"),
    ],
)
def test_dots_in_bars_and_calls_follow_r(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # R 4.6.2 update(), with terms() and lme4's findbars() of the result.
        ("y ~ x + z + (1 | g) + (0 + x | h)", ". ~ (.) - (0 + x | h)", "y ~ x + z + (1 | g)"),
        (
            "y ~ x + z + (1 | g) + (0 + x | h)",
            ". ~ (. + w) - (0 + x | h) + (0 + x | k)",
            "y ~ x + z + w + (1 | g) + (0 + x | k)",
        ),
        ("y ~ x + (1 | g)", ". ~ (.) + (1 | h)", "y ~ x + (1 | g) + (1 | h)"),
        ("y ~ x + (1 | g)", ". ~ (1 | h) + (.)", "y ~ x + (1 | h) + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ (1 | k) + log(.)", "y ~ log(x) + (1 | k) + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ log(.) - (1 | g)", "y ~ log(x)"),
        ("y ~ x + (1 | g)", ". ~ (w + (.)) - (1 | g)", "y ~ w + x"),
        # A subtracted term removes them only when it keeps the '.' terms themselves.
        ("y ~ x + (1 | g)", ". ~ . - .:w", "y ~ x + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ . - .*w", "y ~ 1"),
        ("y ~ x + (1 | g)", ". ~ . - (.)^2", "y ~ 1"),
        # A subtraction within a term drops them from it.
        ("y ~ x + (1 | g)", ". ~ w - (z - .)", "y ~ w"),
        ("y ~ x + (1 | g)", ". ~ (1 | h) + (w - (.))", "y ~ w + (1 | h)"),
        # In a product they stay as they are, where R would cross them.
        ("y ~ x + (1 | g)", ". ~ .:w + (1 | h)", "y ~ x:w + (1 | g) + (1 | h)"),
        # A backticked `.` is the same name, as in R.
        ("y ~ x + (1 | g)", ". ~ `.` + w", "y ~ x + w + (1 | g)"),
        ("y ~ x + (1 | g)", ". ~ (1 | k) + `.` + w", "y ~ x + w + (1 | k) + (1 | g)"),
        ("y ~ x", ". ~ log(`.`)", "y ~ log(x)"),
        ("y ~ 0 + x", ". ~ w + (`.` | g)", "y ~ w + (0 + x | g)"),
    ],
)
def test_dots_within_terms_bring_the_old_random_terms_at_their_place(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


def test_refits_drop_random_terms_subtracted_after_a_parenthesized_dot():
    fit = lmer("Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)", SLEEPSTUDY, REML=False)
    for new in (". ~ . - (0 + Days | Subject)", ". ~ (.) - (0 + Days | Subject)"):
        updated = fit.update(new)
        assert str(updated.formula) == "Reaction ~ Days + (1 | Subject)"
        assert updated.npar() == 4


@pytest.mark.parametrize("old", ["y ~ x + (1 | s)", "y ~ 0 + x + (1 | s)"])
def test_update_errors_name_the_tokens_as_written(old):
    # The old intercept term inserted into a bar never shows in the error.
    with pytest.raises(ValueError, match=re.escape("got PIPE at position 13")):
        update_formula(parse_formula(old), ". ~ . + (. * | g)")
    with pytest.raises(ValueError, match=re.escape("PIPE at position 12")):
        update_formula(parse_formula(old), ". ~ . + ((. | g) + w")


@pytest.mark.parametrize(
    ("new", "error"),
    [
        # As R, which reads these as syntax errors, and not as calls of the name.
        (". ~ log .", "Expected '+' or '-' between terms, got IDENTIFIER at position 8"),
        (". ~ w + offset .", "got IDENTIFIER at position 15"),
        (". ~ log `.`", "got IDENTIFIER at position 8"),
        (". ~ w .", "got IDENTIFIER at position 6"),
        # Errors at a '.' name the '.' as written, not the text replacing it.
        (". ~ (z - .) .", "got IDENTIFIER at position 12"),
        (". ~ (.)^.", "Expected a positive integer power, got IDENTIFIER at position 8"),
        (". ~ (w . | h)", "got IDENTIFIER at position 7"),
        (". ~ (1 | .)", "'.' at position 9 stands for the old fixed terms, which cannot group"),
    ],
)
def test_updates_read_dots_after_names_as_syntax_errors(new, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        update_formula(parse_formula("y ~ x + z + (1 | g)"), new)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # R 4.6.2 update() and terms(): an old fixed part with no terms adds none.
        ("y ~ 1 + (1 | g)", ". ~ (.) + w", "y ~ w + (1 | g)"),
        ("y ~ (1 | g)", ". ~ (w + .)", "y ~ w + (1 | g)"),
        ("y ~ 1 + (1 | g)", ". ~ (z - .) + w", "y ~ 0 + z + w"),
        ("y ~ 0 + (1 | g)", ". ~ (z - .) + w", "y ~ z + w"),
        ("y ~ 1 + (1 | g)", ". ~ .*w", "y ~ w + (1 | g)"),
        ("y ~ 0 + (1 | g)", ". ~ .^2 + w", "y ~ 0 + w + (1 | g)"),
        ("y ~ -1 + (1 | g)", ". ~ w - (.)", "y ~ w"),
        ("y ~ (1 | g)", ". ~ (.) - (1 | g) + w", "y ~ w"),
    ],
)
def test_dots_within_terms_of_formulas_without_fixed_terms(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


def test_refits_of_null_models_accept_parenthesized_dots():
    null = lmer("Reaction ~ 1 + (1 | Subject)", SLEEPSTUDY, REML=False)
    for new in (". ~ (.) + Days", ". ~ (Days + .)"):
        assert str(null.update(new, data=SLEEPSTUDY).formula) == "Reaction ~ Days + (1 | Subject)"


@pytest.mark.parametrize(
    ("old", "formula"),
    [
        (None, "y ~ x + (0 | g)"),
        (None, "y ~ x + (1 - 1 | g)"),
        # The old '0' takes the bar's intercept, and the old formula has no terms.
        ("y ~ 0 + (1 | g)", ". ~ w + (. | h)"),
        ("y ~ x + (1 | g)", ". ~ . + (0 + z - z | h)"),
    ],
)
def test_random_terms_without_columns_are_errors_at_their_bar(old, formula):
    # lme4 cannot fit such a bar either.
    with pytest.raises(
        ValueError, match=re.escape("term at position 8 has no intercept or slopes")
    ):
        parse_formula(formula) if old is None else update_formula(parse_formula(old), formula)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # The as-written check reads '.' as one name, so it leaves bars to the update.
        ("y ~ x + z + (1 | g)", ". ~ . + (0 + .^2 - . | h)", "y ~ x + z + (1 | g) + (0 + x:z | h)"),
        ("y ~ x + z + (1 | g)", ". ~ . + (0 + .:. - . | h)", "y ~ x + z + (1 | g) + (0 + x:z | h)"),
        # With no old fixed terms a '.' adds none to a bar, and its intercept is the bar's.
        ("y ~ 1 + (1 | g)", ". ~ . + (1 + .:w | h)", "y ~ 1 + (1 | g) + (1 | h)"),
        ("y ~ (1 | g)", ". ~ . + (w*. | h)", "y ~ 1 + (1 | g) + (1 + w | h)"),
        ("y ~ 1 + (1 | g)", ". ~ . + (w + (. + v):z | h)", "y ~ 1 + (1 | g) + (1 + w + v:z | h)"),
        ("y ~ (1 | g)", ". ~ . + (0 + w + . | h)", "y ~ 1 + (1 | g) + (0 + w | h)"),
        ("y ~ (1 | g)", ". ~ . + (w - . | h)", "y ~ 1 + (1 | g) + (1 + w | h)"),
        ("y ~ (1 | g)", ". ~ . + (0 + . + w || h)", "y ~ 1 + (1 | g) + (0 + w || h)"),
        # Subtracting a bar with no columns removes nothing, as in R.
        ("y ~ x + (1 | g)", ". ~ . - (0 | g)", "y ~ x + (1 | g)"),
        # A one-sided update keeps the old response, as in R.
        ("y ~ x + (1 | g)", "~ . + w", "y ~ x + w + (1 | g)"),
    ],
)
def test_updates_check_bars_after_substituting_dots(old, new, expected):
    assert str(update_formula(parse_formula(old), new)) == expected


@pytest.mark.parametrize(("new", "position"), [("log(.~ .", 5), ("   log(y~ .", 8)])
def test_update_response_errors_point_at_the_tilde(new, position):
    with pytest.raises(ValueError, match=re.escape(f"got TokenType.TILDE at position {position}")):
        update_formula(parse_formula("y ~ x + (1 | g)"), new)
