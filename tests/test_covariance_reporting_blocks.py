from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, tidy
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg


@pytest.fixture(params=[LmerResult, GlmerResult], ids=["lmer", "glmer"])
def make_result(request):
    def make(formula_text="y ~ x + z + (x | group) + (0 + z | group)", theta=None):
        rng = np.random.default_rng(621)
        n = 30
        data = pd.DataFrame(
            {
                "y": np.arange(n) % 2,
                "x": rng.normal(size=n),
                "z": rng.normal(size=n),
                "w": rng.normal(size=n),
                "group": np.repeat(np.arange(6), 5),
                "group.1": np.repeat(np.arange(6), 5),
                "other": np.arange(n) % 5,
            }
        )
        formula = parse_formula(formula_text)
        matrices = build_model_matrices(formula, data)
        common = dict(
            formula=formula,
            matrices=matrices,
            theta=np.asarray([0.8, 0.2, 0.5, 0.4] if theta is None else theta),
            beta=np.zeros(matrices.n_fixed),
            u=np.zeros(matrices.n_random),
            deviance=0.0,
            converged=True,
            n_iter=0,
        )
        if request.param is LmerResult:
            return LmerResult(sigma=2.0, REML=True, **common)
        return GlmerResult(family=families.Binomial(), nAGQ=1, **common)

    return make


def _scale(result):
    return result.sigma**2 if isinstance(result, LmerResult) else 1.0


def test_varcorr_retains_all_random_terms_for_one_group(make_result):
    result = make_result()
    first_factor = np.array([[0.8, 0.0], [0.2, 0.5]])
    scale = _scale(result)

    report = result.VarCorr()

    assert list(report.groups) == ["group", "group.1"]
    assert report.groups["group"].term_names == ["(Intercept)", "x"]
    assert report.groups["group.1"].term_names == ["z"]
    assert_allclose(report.get_cov("group"), first_factor @ first_factor.T * scale)
    assert_allclose(report.get_cov("group.1"), [[0.4**2 * scale]])
    assert set(report.as_dict()["group"]) == {"(Intercept)", "x"}
    assert set(report.as_dict()["group.1"]) == {"z"}
    assert "group.1" in str(report)
    for name, group in report.groups.items():
        assert group.name == name
        assert group.grouping_factor == "group"
        assert_allclose(list(group.variance.values()), np.diag(group.cov))
        assert_allclose(list(group.stddev.values()), np.sqrt(np.diag(group.cov)))


def test_pca_combines_all_block_eigenvalues_for_group(make_result):
    result = make_result()
    first_factor = np.array([[0.8, 0.0], [0.2, 0.5]])
    covariance = linalg.block_diag(first_factor @ first_factor.T, [[0.4**2]]) * _scale(result)
    expected = np.linalg.eigvalsh(covariance)[::-1]

    pca = result.rePCA()

    assert list(pca.groups) == ["group"]
    assert pca["group"].n_terms == 3
    assert_allclose(pca["group"].sdev, np.sqrt(expected))
    assert_allclose(pca["group"].proportion, expected / expected.sum())
    assert_allclose(pca["group"].cumulative, np.cumsum(expected / expected.sum()))


def test_zero_earlier_block_is_visible_to_pca_singularity(make_result):
    result = make_result(theta=[0.0, 0.0, 0.0, 0.4])

    pca = result.rePCA()

    assert pca.is_singular()["group"]
    assert pca["group"].n_terms == 3
    assert_array_equal(pca["group"].proportion, [1.0, 0.0, 0.0])
    assert_array_equal(result.VarCorr().groups["group"].corr, np.eye(2))


def test_all_zero_blocks_have_finite_pca_proportions(make_result):
    result = make_result(theta=np.zeros(4))

    pca = result.rePCA()["group"]

    assert_array_equal(pca.sdev, np.zeros(3))
    assert_array_equal(pca.proportion, np.zeros(3))
    assert_array_equal(pca.cumulative, np.zeros(3))


def test_generated_names_preserve_existing_group_names(make_result):
    result = make_result(
        "y ~ x + z + (1 | group) + (0 + x | group) + (1 | `group.1`) + (0 + z | group)",
        theta=[0.4, 0.5, 0.6, 0.7],
    )

    report = result.VarCorr()

    assert list(report.groups) == ["group", "group.2", "group.1", "group.3"]
    assert report.groups["group.1"].term_names == ["(Intercept)"]
    assert report.groups["group.1"].grouping_factor == "group.1"
    assert report.groups["group.2"].grouping_factor == "group"
    assert_allclose(report.groups["group.1"].cov, [[0.6**2 * _scale(result)]])
    assert list(result.rePCA().groups) == ["group", "group.1"]
    assert result.rePCA()["group"].n_terms == 3
    assert result.rePCA()["group.1"].n_terms == 1


def test_nonadjacent_blocks_are_grouped_without_mixing_factors(make_result):
    result = make_result(
        "y ~ x + (1 | group) + (1 | other) + (0 + x | group)", theta=[0.4, 0.7, 0.5]
    )

    assert list(result.VarCorr().groups) == ["group", "other", "group.1"]
    pca = result.rePCA()
    assert list(pca.groups) == ["group", "other"]
    assert_allclose(pca["group"].sdev, np.array([0.5, 0.4]) * np.sqrt(_scale(result)))
    assert_allclose(pca["other"].sdev, [0.7 * np.sqrt(_scale(result))])


@pytest.mark.parametrize("cov_type", ["cs", "ar1"])
def test_structured_blocks_keep_covariance_and_all_pca_terms(make_result, cov_type):
    result = make_result("y ~ x + z + (x + z | group) + (0 + w | group)", [0.9, 0.3, 0.4])
    result.matrices.random_structures[0].cov_type = cov_type
    correlation = (
        np.full((3, 3), 0.3)
        if cov_type == "cs"
        else 0.3 ** np.abs(np.arange(3)[:, None] - np.arange(3))
    )
    np.fill_diagonal(correlation, 1.0)
    covariance = linalg.block_diag(0.9**2 * correlation, [[0.4**2]]) * _scale(result)

    report = result.VarCorr()
    pca = result.rePCA()["group"]

    assert list(report.groups) == ["group", "group.1"]
    assert_allclose(report.groups["group"].corr, correlation)
    assert_allclose(report.groups["group.1"].cov, [[0.4**2 * _scale(result)]])
    assert pca.n_terms == 4
    assert_allclose(pca.sdev, np.sqrt(np.linalg.eigvalsh(covariance)[::-1]))


def test_tidy_includes_parameters_from_each_covariance_block(make_result):
    result = make_result()

    table = tidy(result, effects="ran_pars")

    group_rows = table.loc[table["group"] == "group"]
    assert set(group_rows["term"]) == {"sd__(Intercept)", "sd__x", "cor__(Intercept).x"}
    second = table.loc[table["group"] == "group.1"]
    assert second["term"].tolist() == ["sd__z"]
    assert_allclose(second["estimate"], [0.4 * np.sqrt(_scale(result))])


def test_duplicate_coefficient_names_stay_in_separate_blocks(make_result):
    result = make_result("y ~ x + (1 | group) + (1 | group)", [0.4, 0.7])

    report = result.VarCorr()

    assert len(result.formula.random) == len(result.matrices.random_structures) == 2
    assert list(report.groups) == ["group", "group.1"]
    assert report.groups["group"].term_names == report.groups["group.1"].term_names
    assert_allclose(report.groups["group"].cov, [[0.4**2 * _scale(result)]])
    assert_allclose(report.groups["group.1"].cov, [[0.7**2 * _scale(result)]])
    assert_allclose(result.rePCA()["group"].sdev, np.array([0.7, 0.4]) * np.sqrt(_scale(result)))


def test_reports_do_not_expand_group_level_or_combined_covariance(make_result, monkeypatch):
    result = make_result()
    for structure in result.matrices.random_structures:
        structure.n_levels = 100_000

    def reject_assembly(*args, **kwargs):
        raise AssertionError("reporting should operate on individual covariance blocks")

    from mixedlm.estimation import reml

    monkeypatch.setattr(reml, "_build_lambda", reject_assembly)
    monkeypatch.setattr(linalg, "block_diag", reject_assembly)

    assert len(result.VarCorr().groups) == 2
    assert result.rePCA()["group"].n_terms == 3
