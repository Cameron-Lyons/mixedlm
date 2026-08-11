import numpy as np
import pandas as pd
from mixedlm import (
    parse_formula,
)
from mixedlm.formula.terms import InteractionTerm, VariableTerm
from mixedlm.matrices import build_model_matrices
from mixedlm.utils.dataframe import concat_columns_as_string

from tests._lmer_data import SLEEPSTUDY


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

    def test_backticks_quote_names_and_formula_operators(self) -> None:
        formula = parse_formula(
            "`response value` ~ `fixed + value`:`second:value` + (`random slope` | `group/id`)"
        )

        assert formula.response == "response value"
        assert formula.fixed.terms == (InteractionTerm(("fixed + value", "second:value")),)
        assert formula.random[0].expr == (VariableTerm("random slope"),)
        assert formula.random[0].grouping == "group/id"
        assert parse_formula(str(formula)) == formula

    def test_backticks_support_nested_grouping_factors(self) -> None:
        formula = parse_formula("y ~ x + (1 | `school id`/`class/id`)")

        assert formula.random[0].grouping == ("school id", "class/id")
        assert str(formula) == "y ~ x + (1 | `school id`/`class/id`)"

    def test_quoted_random_slope_without_intercept_round_trips(self) -> None:
        formula = parse_formula("y ~ x + (0 + `random slope` | `group id`)")

        assert not formula.random[0].has_intercept
        assert str(formula) == "y ~ x + (0 + `random slope` | `group id`)"
        assert parse_formula(str(formula)) == formula

    def test_backticks_can_be_escaped(self) -> None:
        formula = parse_formula(r"`response\`name` ~ `fixed\`name`")

        assert formula.response == "response`name"
        assert formula.fixed.terms == (VariableTerm("fixed`name"),)
        assert parse_formula(str(formula)) == formula

    def test_unterminated_backtick_name_is_rejected(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "Unterminated quoted identifier"):
            parse_formula("`response ~ x")


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

    def test_no_intercept_categorical_fixed_effect_uses_all_levels(self) -> None:
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "treatment": ["a", "b", "c", "a", "b", "c"],
                "group": ["g1", "g1", "g1", "g2", "g2", "g2"],
            }
        )

        for formula in (
            "y ~ 0 + treatment + (1 | group)",
            "y ~ treatment - 1 + (1 | group)",
        ):
            matrices = build_model_matrices(parse_formula(formula), data)

            assert matrices.fixed_names == ["treatment.a", "treatment.b", "treatment.c"]
            assert np.array_equal(matrices.X, np.tile(np.eye(3), (2, 1)))

    def test_no_intercept_categorical_random_effect_uses_all_levels(self) -> None:
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "treatment": ["a", "b", "c", "a", "b", "c"],
                "group": ["g1", "g1", "g1", "g2", "g2", "g2"],
            }
        )

        matrices = build_model_matrices(
            parse_formula("y ~ 1 + (0 + treatment | group)"),
            data,
        )

        structure = matrices.random_structures[0]
        assert structure.term_names == ["treatment.a", "treatment.b", "treatment.c"]
        assert structure.n_terms == 3
        assert np.array_equal(matrices.Z.toarray(), np.eye(6))

    def test_categorical_effect_with_intercept_remains_reduced_rank(self) -> None:
        data = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "treatment": ["a", "b", "c", "a", "b", "c"],
                "group": ["g1", "g1", "g1", "g2", "g2", "g2"],
            }
        )

        matrices = build_model_matrices(
            parse_formula("y ~ treatment + (1 + treatment | group)"),
            data,
        )

        assert matrices.fixed_names == ["(Intercept)", "treatment.1", "treatment.2"]
        assert matrices.random_structures[0].term_names == [
            "(Intercept)",
            "treatment.1",
            "treatment.2",
        ]

    def test_nested_group_labels_preserve_mixed_values(self) -> None:
        data = pd.DataFrame(
            {
                "school": [1, 1, 2, 2],
                "classroom": pd.Categorical(["A", "B", "A", "B"]),
            }
        )

        labels = concat_columns_as_string(data, ["school", "classroom"])

        np.testing.assert_array_equal(labels, ["1/A", "1/B", "2/A", "2/B"])

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

    def test_quoted_names_build_model_matrices(self) -> None:
        data = pd.DataFrame(
            {
                "response value": [1.0, 2.0, 3.0, 4.0],
                "fixed + value": [0.0, 1.0, 2.0, 3.0],
                "group/id": ["a", "a", "b", "b"],
            }
        )
        formula = parse_formula("`response value` ~ `fixed + value` + (1 | `group/id`)")
        matrices = build_model_matrices(formula, data)

        assert np.array_equal(matrices.y, data["response value"].to_numpy())
        assert matrices.fixed_names == ["(Intercept)", "fixed + value"]
        assert matrices.random_structures[0].grouping_factor == "group/id"
        assert matrices.Z.shape == (4, 2)
