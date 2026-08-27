from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.modular import mkDataTemplate, mkMinimalData, mkParsTemplate


def test_balanced_template_deduplicates_repeated_grouping_factors() -> None:
    data = mkDataTemplate(
        "y ~ x + (1 | group) + (x || group)",
        nlevs={"group": 4},
        seed=42,
    )

    assert list(data.columns) == ["y", "x", "group"]
    assert data.shape == (4, 3)
    assert data["group"].nunique() == 4


def test_unbalanced_template_handles_unequal_group_counts() -> None:
    data = mkDataTemplate(
        "y ~ x + (1 | small) + (1 | large)",
        nlevs={"small": 2, "large": 100},
        balanced=False,
        seed=42,
    )

    assert data.shape == (204, 4)
    assert data["small"].nunique() == 2
    assert data["large"].nunique() == 100


def test_templates_support_nested_groups_interactions_and_grouped_responses() -> None:
    formula = "successes / trials ~ x * z + I(w**2) + (x || site/subject)"

    template = mkDataTemplate(
        formula,
        nlevs={"site": 2, "subject": 3},
        seed=42,
    )
    minimal = mkMinimalData(formula, n=8, seed=42)

    expected_columns = ["successes", "trials", "x", "z", "w", "site", "subject"]
    assert list(template.columns) == expected_columns
    assert list(minimal.columns) == expected_columns
    assert template.shape == (6, len(expected_columns))
    assert minimal.shape == (8, len(expected_columns))
    assert np.all(template["successes"] == 0)
    assert np.all(template["trials"] == 1)

    for data in (template, minimal):
        matrices = build_model_matrices(parse_formula(formula), data, grouped_binomial=True)
        assert matrices.n_obs == len(data)
        assert matrices.trials is not None


@pytest.mark.parametrize("factory", [mkDataTemplate, mkMinimalData])
def test_template_seeds_are_reproducible_and_isolate_global_random_state(factory) -> None:
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)

    first = factory("y ~ z + x + (w | group)", seed=99)
    second = factory("y ~ z + x + (w | group)", seed=99)
    observed = np.random.random(5)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("count", [0, -1])
def test_data_template_rejects_nonpositive_group_counts(count: int) -> None:
    with pytest.raises(ValueError, match=r"nlevs\['group'\] must be positive"):
        mkDataTemplate("y ~ (1 | group)", nlevs={"group": count})


@pytest.mark.parametrize("n", [0, -1])
def test_minimal_data_rejects_nonpositive_row_counts(n: int) -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        mkMinimalData("y ~ x + (1 | group)", n=n)


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_parameter_template_honors_structured_covariance(cov_type: str) -> None:
    from mixedlm import set_cov_type

    formula = set_cov_type("y ~ x + z + (x + z | group)", cov_type)
    data = mkMinimalData(str(formula), n=20, seed=42)

    template = mkParsTemplate(formula, data)

    assert template["theta"] == ["sd_common|group", "rho|group"]
    assert template["n_theta"] == 2
