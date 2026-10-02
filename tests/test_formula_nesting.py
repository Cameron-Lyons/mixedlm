"""Formula grouping agrees with independently constructed hierarchical designs."""

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glmer, lmer
from mixedlm.estimation import reml
from mixedlm.formula import parse_formula
from mixedlm.formula.parser import set_cov_type, update_formula
from mixedlm.matrices import build_model_matrices
from numpy.testing import assert_allclose, assert_array_equal


@pytest.mark.parametrize(
    "formula",
    [
        "y ~",
        "y ~ x +",
        "y ~ x -",
        "y ~ x x",
        "y ~ x ) + (1 | g)",
        "y ~ x / z + (1 | g)",
        "y ~ x ~ z",
        "y ~ x | g",
        "y ~ x + + z",
        "y ~ x + - z",
        "y ~ x *",
        "y ~ x :",
        "y ~ x * + z",
        "y ~ x + (1 x | g)",
        "y ~ x + (1 + | g)",
        "y ~ x + (| g)",
        "y ~ x + (1 | g/)",
        "y ~ x + (1 | g:)",
        "y ~ x + (1 | g) garbage",
        "y ~ x + (1 | g))",
    ],
)
def test_invalid_formula_never_silently_discards_terms(formula):
    with pytest.raises(ValueError, match="Expected.*position"):
        parse_formula(formula)


@pytest.mark.parametrize(
    "update", [". ~ . +", ". ~ . -", ". ~ . ++ x", ". ~ . + (1 | g", ". ~ . )"]
)
def test_invalid_formula_updates_are_rejected(update):
    with pytest.raises(ValueError):
        update_formula(parse_formula("y ~ x + (1 | g)"), update)


def test_three_level_nesting_expands_to_all_parent_factors():
    formula = parse_formula("y ~ x + (0 + x || school/classroom/student)")
    assert [term.grouping_factors for term in formula.random] == [
        ("school",),
        ("school", "classroom"),
        ("school", "classroom", "student"),
    ]
    assert all(not term.has_intercept and not term.correlated for term in formula.random)
    assert parse_formula(str(formula)) == formula

    updated = update_formula(formula, ". ~ . - (0 + x || school/classroom/student)")
    assert updated.random == ()


def test_joint_grouping_only_adds_one_factor_and_covariance_can_be_selected():
    formula = parse_formula("y ~ x + (x | school) + (x | school:classroom)")
    specified = set_cov_type(formula, {"school:classroom": "ar1"})
    assert [term.cov_type for term in specified.random] == ["us", "ar1"]
    assert specified.all_variables == {"y", "x", "school", "classroom"}
    assert parse_formula(str(formula)) == formula


@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_nested_covariance_matches_manual_parent_and_child_indicators(frame_type):
    data = pd.DataFrame(
        {
            "y": np.arange(8.0),
            "x": np.linspace(-1, 1, 8),
            "school": ["a/b", "a", "a/b", "a", "c", "c", "d", "d"],
            "classroom": ["c", "b/c", "c", "b/c", "x", "y", "x", "y"],
        }
    )
    # Distinct pairs a/b,c and a,b/c must not merge into the same child level.
    parent = pd.get_dummies(data["school"], dtype=float).to_numpy()
    child_codes, children = pd.factorize(pd.MultiIndex.from_frame(data[["school", "classroom"]]))
    child = np.eye(len(children))[child_codes]
    if frame_type == "polars":
        pl = pytest.importorskip("polars")
        data = pl.from_pandas(data)
    matrices = build_model_matrices(parse_formula("y ~ x + (1 | school/classroom)"), data)
    assert [structure.grouping_factor for structure in matrices.random_structures] == [
        "school",
        "school:classroom",
    ]
    assert [structure.n_levels for structure in matrices.random_structures] == [4, 6]
    design = matrices.Z.toarray()
    assert_array_equal(design @ design.T, parent @ parent.T + child @ child.T)


def _hierarchical_data():
    rng = np.random.default_rng(491)
    school = np.repeat(np.arange(6), 21)
    classroom = np.tile(np.repeat(np.arange(3), 7), 6)
    child = 3 * school + classroom
    x = rng.normal(size=len(child))
    offset = rng.normal(0, 0.15, len(child))
    weights = rng.uniform(0.5, 2, len(child))
    eta = 0.7 + 0.3 * x + rng.normal(0, 0.6, 6)[school] + rng.normal(0, 0.4, 18)[child]
    return pd.DataFrame(
        {
            "y": eta + offset + rng.normal(0, 0.3, len(child)) / np.sqrt(weights),
            "count": rng.poisson(np.exp(eta + offset)),
            "x": x,
            "school": school,
            "classroom": classroom,
            "child": child,
            "offset": offset,
            "weight": weights,
        }
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("restricted", [False, True])
def test_nested_lmm_matches_explicit_two_factor_fit(native, restricted, monkeypatch):
    if not native:
        monkeypatch.setattr(reml, "_HAS_RUST", False)
    data = _hierarchical_data()
    arguments = {"weights": "weight", "offset": "offset", "REML": restricted}
    nested = lmer("y ~ x + (1 | school/classroom)", data, **arguments)
    manual = lmer("y ~ x + (1 | school) + (1 | child)", data, **arguments)
    assert nested.converged and manual.converged
    assert_allclose(nested.beta, manual.beta, atol=1e-8)
    assert_allclose(nested.theta, manual.theta, atol=1e-6)
    assert nested.logLik().value == pytest.approx(manual.logLik().value, abs=1e-8)
    assert_allclose(nested.predict(data), manual.predict(data), atol=1e-7)
    assert_allclose(nested.predict(data, offset="offset"), nested.fitted(), atol=1e-8)

    new = data.iloc[[0]].copy()
    new["classroom"] = 999
    new["child"] = 999
    assert_allclose(
        nested.predict(new, allow_new_levels=True),
        manual.predict(new, allow_new_levels=True),
        atol=1e-7,
    )


def test_nested_glmm_matches_explicit_fit_and_conditional_predictions():
    data = _hierarchical_data()
    arguments = {"family": families.Poisson(), "nAGQ": 0, "offset": "offset"}
    nested = glmer("count ~ x + (1 | school/classroom)", data, **arguments)
    manual = glmer("count ~ x + (1 | school) + (1 | child)", data, **arguments)
    assert nested.converged and manual.converged
    assert_allclose(nested.beta, manual.beta, atol=1e-7)
    assert_allclose(nested.theta, manual.theta, atol=1e-6)
    assert nested.logLik().value == pytest.approx(manual.logLik().value, abs=1e-8)
    assert_allclose(nested.predict(data, offset="offset"), nested.fitted(), atol=1e-8)
    assert_allclose(nested.predict(data), manual.predict(data), atol=1e-7)
