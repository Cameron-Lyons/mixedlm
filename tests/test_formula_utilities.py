"""Formula string utilities rewrite and report lme4 formulas exactly."""

import pytest
from mixedlm import (
    dropOffset,
    expandDoubleVerts,
    getFixedFormulaStr,
    getNGroups,
    getRandomFormulaStr,
    getResponseName,
    lmer,
    load_sleepstudy,
    parse_formula,
)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ x + (1 + x || group)", "y ~ x + (1 | group) + (0 + x | group)"),
        ("y ~ x + (0 + x || group)", "y ~ x + (0 + x | group)"),
        (
            "y ~ x + (1 + x || g1) + (1 + z || g2)",
            "y ~ x + (1 | g1) + (0 + x | g1) + (1 | g2) + (0 + z | g2)",
        ),
        ("y ~ x + (1 | group)", "y ~ x + (1 | group)"),
        (
            "`response value` ~ x + (`random|slope` || `group id`)",
            "`response value` ~ x + (1 | `group id`) + (0 + `random|slope` | `group id`)",
        ),
    ],
)
def test_expandDoubleVerts_splits_each_double_bar_term(formula, expected):
    assert expandDoubleVerts(formula) == expected


def test_expanded_double_bar_fits_the_same_model():
    formula = "Reaction ~ Days + (Days || Subject)"
    data = load_sleepstudy()
    direct = lmer(formula, data)
    expanded = lmer(expandDoubleVerts(formula), data)

    assert expanded.deviance == pytest.approx(direct.deviance, rel=1e-10)
    assert expanded.theta == pytest.approx(direct.theta, rel=1e-6, abs=1e-8)
    assert getNGroups(expandDoubleVerts(formula)) == 1


def test_expanded_quoted_names_round_trip_through_the_parser():
    expanded = expandDoubleVerts("`response value` ~ x + (`random|slope` || `group id`)")

    assert parse_formula(expanded).response == "response value"


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ x + offset(a) + z + offset(b)", "y ~ x + z"),
        ("y ~ x + offset(exposure)", "y ~ x"),
        ("y ~ x + (1 | group)", "y ~ x + (1 | group)"),
        pytest.param(
            "y ~ x + offset(log(t)) + (1 | group)",
            "y ~ x + (1 | group)",
            marks=pytest.mark.xfail(
                strict=True,
                reason="dropOffset stops at the first ')' and returns 'y ~ x ) + (1 | group)'",
            ),
        ),
    ],
)
def test_dropOffset_removes_every_offset_term(formula, expected):
    assert dropOffset(formula) == expected


@pytest.mark.parametrize(
    ("formula", "response", "fixed", "random"),
    [
        ("y ~ x + z + (1 | group)", "y", "y ~ x + z", "(1 | group)"),
        (
            "Reaction ~ Days + (Days | Subject)",
            "Reaction",
            "Reaction ~ Days",
            "(1 + Days | Subject)",
        ),
        ("y ~ x + (1 | group) + (1 | subject)", "y", "y ~ x", "(1 | group) + (1 | subject)"),
        (
            "`response value` ~ `fixed + value` + (`random slope` | `group id`)",
            "response value",
            "`response value` ~ `fixed + value`",
            "(1 + `random slope` | `group id`)",
        ),
    ],
)
def test_formula_parts_are_reported_verbatim(formula, response, fixed, random):
    assert getResponseName(formula) == response
    assert getFixedFormulaStr(formula) == fixed
    assert getRandomFormulaStr(formula) == random


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ x", 0),
        ("y ~ x + (1 | group)", 1),
        ("y ~ x + (1 | group) + (1 | subject)", 2),
        ("y ~ x + (1 | group) + (x | group)", 1),
        ("y ~ x + (1 | a/b)", 2),
    ],
)
def test_getNGroups_counts_distinct_grouping_factors(formula, expected):
    assert getNGroups(formula) == expected
