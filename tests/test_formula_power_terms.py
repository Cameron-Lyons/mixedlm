import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.formula.parser import (
    getFixedFormulaStr,
    getRandomFormulaStr,
    parse_formula,
    subbars,
    update_formula,
)
from mixedlm.formula.terms import InteractionTerm, PowerTerm
from mixedlm.inference.drop1 import _droppable_fixed_terms
from mixedlm.matrices import build_model_matrices

from tests._lmer_data import SLEEPSTUDY


@pytest.fixture
def polynomial_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 1.5, 2.5, 4.0, 6.0, 8.5],
            "x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )


def test_power_terms_parse_roundtrip_and_track_source_variables() -> None:
    formula = parse_formula("y ~ x + I(x**2) + I(x**3) + (I(z**2) | g)")

    assert formula.fixed.terms[1:] == (PowerTerm("x", 2), PowerTerm("x", 3))
    assert formula.random[0].expr == (PowerTerm("z", 2),)
    assert formula.all_variables == {"y", "x", "z", "g"}
    assert str(parse_formula(str(formula))) == str(formula)


def test_power_terms_build_fixed_and_interaction_columns(
    polynomial_data: pd.DataFrame,
) -> None:
    formula = parse_formula("y ~ x + I(x**2) * z + (1 | g)")
    matrices = build_model_matrices(formula, polynomial_data)
    x = polynomial_data["x"].to_numpy()
    z = polynomial_data["z"].to_numpy()

    assert formula.fixed.terms[-1] == InteractionTerm((PowerTerm("x", 2), "z"))
    assert matrices.fixed_names == [
        "(Intercept)",
        "x",
        "I(x**2)",
        "z",
        "I(x**2):z",
    ]
    assert np.allclose(matrices.X[:, 2], x**2)
    assert np.allclose(matrices.X[:, 4], x**2 * z)


def test_power_term_preserves_full_rank_factor_without_intercept(
    polynomial_data: pd.DataFrame,
) -> None:
    matrices = build_model_matrices(
        parse_formula("y ~ 0 + I(x**2) + g + (1 | g)"),
        polynomial_data,
    )

    assert matrices.fixed_names == ["I(x**2)", "g.a", "g.b"]
    assert np.allclose(matrices.X[:, 0], polynomial_data["x"] ** 2)
    assert np.array_equal(matrices.X[:, 1:], np.repeat(np.eye(2), 3, axis=0))


def test_power_term_builds_random_slope(polynomial_data: pd.DataFrame) -> None:
    formula = parse_formula("y ~ x + (0 + I(x**2) | g)")
    matrices = build_model_matrices(formula, polynomial_data)

    assert matrices.random_structures[0].term_names == ["I(x**2)"]
    assert matrices.Z.shape == (len(polynomial_data), 2)
    assert np.allclose(matrices.Z.toarray()[:3, 0], polynomial_data["x"][:3] ** 2)
    assert np.allclose(matrices.Z.toarray()[3:, 1], polynomial_data["x"][3:] ** 2)


def test_power_term_fit_matches_precomputed_predictor() -> None:
    data = SLEEPSTUDY.copy()
    data["DaysSquared"] = data["Days"] ** 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model is singular")
        power_fit = lmer("Reaction ~ Days + I(Days**2) + (1 | Subject)", data)
        manual_fit = lmer("Reaction ~ Days + DaysSquared + (1 | Subject)", data)

    assert np.allclose(power_fit.beta, manual_fit.beta)
    assert power_fit.logLik().value == pytest.approx(manual_fit.logLik().value)


def test_formula_helpers_preserve_power_terms() -> None:
    formula = parse_formula("y ~ x + I(x**2) + (I(x**3) | g)")

    assert getFixedFormulaStr(formula) == "y ~ x + I(x**2)"
    assert getRandomFormulaStr(formula) == "(1 + I(x**3) | g)"
    assert subbars(formula) == "y ~ x + I(x**2) + g + g:I(x**3)"

    removed = update_formula(formula, ". ~ . - I(x**2)")
    assert PowerTerm("x", 2) not in removed.fixed.terms
    restored = update_formula(removed, ". ~ . + I(x**2)")
    assert PowerTerm("x", 2) in restored.fixed.terms


def test_drop1_marginality_distinguishes_power_factors() -> None:
    model = SimpleNamespace(formula=parse_formula("y ~ x + I(x**2) * z"))

    assert _droppable_fixed_terms(model) == ["x", "I(x**2):z"]


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ I(x)",
        "y ~ I(x**-2)",
        "y ~ I(x**2.5)",
    ],
)
def test_invalid_power_terms_raise_clear_parse_errors(formula: str) -> None:
    with pytest.raises(ValueError, match="Expected"):
        parse_formula(formula)
