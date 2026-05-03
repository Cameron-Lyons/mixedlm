# ruff: noqa: F401
import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    anova,
    families,
    findbars,
    glmer,
    glmerControl,
    is_mixed_formula,
    lmer,
    lmerControl,
    nlme,
    nlmer,
    nobars,
    parse_formula,
    subbars,
)
from mixedlm.formula.terms import InteractionTerm, VariableTerm
from mixedlm.matrices import build_model_matrices
from mixedlm.models.control import GlmerControl, LmerControl

from tests._lmer_data import CBPP, SLEEPSTUDY


class TestFormulaParser:
    def _fixed_term_labels(self, formula: str) -> list[str]:
        parsed = parse_formula(formula)
        labels = []
        for term in parsed.fixed.terms:
            if isinstance(term, VariableTerm):
                labels.append(term.name)
            elif isinstance(term, InteractionTerm):
                labels.append(":".join(term.variables))
        return labels

    def test_simple_random_intercept(self) -> None:
        formula = parse_formula("y ~ x + (1 | group)")
        assert formula.response == "y"
        assert formula.fixed.has_intercept
        assert len(formula.random) == 1
        assert formula.random[0].grouping == "group"
        assert formula.random[0].correlated

    def test_random_slope(self) -> None:
        formula = parse_formula("y ~ x + (x | group)")
        assert len(formula.random) == 1
        assert formula.random[0].has_intercept

    def test_uncorrelated_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (x || group)")
        assert len(formula.random) == 1
        assert not formula.random[0].correlated

    def test_nested_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (1 | group/subgroup)")
        assert len(formula.random) == 1
        assert formula.random[0].is_nested
        assert formula.random[0].grouping == ("group", "subgroup")

    def test_crossed_random_effects(self) -> None:
        formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
        assert len(formula.random) == 2

    def test_no_intercept(self) -> None:
        formula = parse_formula("y ~ 0 + x + (1 | group)")
        assert not formula.fixed.has_intercept

    def test_star_expands_to_main_effects_and_interaction(self) -> None:
        labels = self._fixed_term_labels("y ~ x1 * x2 + (1 | g)")
        assert labels == ["x1", "x2", "x1:x2"]

    def test_star_expansion_deduplicates_existing_main_effects(self) -> None:
        labels = self._fixed_term_labels("y ~ x1 + x1 * x2 + (1 | g)")
        assert labels == ["x1", "x2", "x1:x2"]

    def test_chained_star_expands_all_interactions(self) -> None:
        labels = self._fixed_term_labels("y ~ x1 * x2 * x3 + (1 | g)")
        assert labels == ["x1", "x2", "x1:x2", "x3", "x1:x3", "x2:x3", "x1:x2:x3"]

    def test_random_effect_star_expands_to_terms(self) -> None:
        formula = parse_formula("y ~ x1 + (x1 * x2 | g)")
        labels = []
        for term in formula.random[0].expr:
            if isinstance(term, VariableTerm):
                labels.append(term.name)
            elif isinstance(term, InteractionTerm):
                labels.append(":".join(term.variables))
        assert labels == ["x1", "x2", "x1:x2"]

    def test_random_effect_star_expansion_deduplicates_existing_terms(self) -> None:
        formula = parse_formula("y ~ x1 + (x1 + x1 * x2 | g)")
        labels = []
        for term in formula.random[0].expr:
            if isinstance(term, VariableTerm):
                labels.append(term.name)
            elif isinstance(term, InteractionTerm):
                labels.append(":".join(term.variables))
        assert labels == ["x1", "x2", "x1:x2"]


class TestModelMatrices:
    def test_fixed_matrix_with_intercept(self) -> None:
        formula = parse_formula("Reaction ~ Days + (1 | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_obs == 180
        assert matrices.n_fixed == 2
        assert matrices.fixed_names == ["(Intercept)", "Days"]
        assert matrices.X.shape == (180, 2)
        assert np.allclose(matrices.X[:, 0], 1.0)

    def test_random_matrix(self) -> None:
        formula = parse_formula("Reaction ~ Days + (1 | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_random == 18
        assert matrices.Z.shape == (180, 18)

    def test_random_slope_matrix(self) -> None:
        formula = parse_formula("Reaction ~ Days + (Days | Subject)")
        matrices = build_model_matrices(formula, SLEEPSTUDY)

        assert matrices.n_random == 36
        assert len(matrices.random_structures) == 1
        assert matrices.random_structures[0].n_terms == 2

    def test_star_fixed_matrix_includes_main_effect_columns(self) -> None:
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [2.0, 3.0, 4.0, 5.0],
                "g": ["a", "a", "b", "b"],
            }
        )
        formula = parse_formula("y ~ x1 * x2 + (1 | g)")
        matrices = build_model_matrices(formula, data)

        assert matrices.fixed_names == ["(Intercept)", "x1", "x2", "x1:x2"]
        assert np.allclose(matrices.X[:, 3], data["x1"].to_numpy() * data["x2"].to_numpy())
