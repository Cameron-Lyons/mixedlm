from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer, parse_formula
from mixedlm.families import Poisson
from mixedlm.inference.emmeans import emmeans
from mixedlm.matrices import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import stats


def _model(kind="lmm", backend="pandas", contrasts=None, treatments=("C", "A", "B")):
    data = pd.DataFrame(
        itertools.product(treatments, ["west", "east"], [-1.0, 0.5, 2.0], [-2.0, 0.5, 1.5]),
        columns=["treatment", "site", "x", "z"],
    )
    data["treatment"] = pd.Categorical(data.treatment, categories=treatments, ordered=True)
    data["g"] = np.arange(len(data)) % 6
    data["y"] = 2.0 + data.x**2 + (data.treatment == "A") + np.sin(np.arange(len(data)))
    if backend == "polars":
        pl = pytest.importorskip("polars")
        data = pl.DataFrame(data.to_dict("list")).with_columns(
            pl.col("treatment").cast(pl.Enum(["C", "A", "B"]))
        )
    formula = parse_formula(
        "y ~ treatment * site + x + I(x**2) + treatment:x + z + I(z**2) + (1 | g)"
    )
    matrices = build_model_matrices(formula, data, contrasts=contrasts)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.6]),
        beta=np.linspace(-0.2, 0.3, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    return (
        LmerResult(**common, sigma=1.3, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=Poisson(), nAGQ=1)
    )


def _direct_rows(model, result_grid, at=None):
    values = {
        "treatment": ["C", "A", "B"],
        "site": ["east", "west"],
        "x": [0.5],
        "z": [0.0],
    }
    values.update({} if at is None else at)
    coefficients = []
    for _, row in result_grid.iterrows():
        reference = {
            name: [row[name]] if name in row else levels for name, levels in values.items()
        }
        grid = pd.DataFrame(itertools.product(*reference.values()), columns=list(reference))
        coefficients.append(model._prediction_fixed_matrix(grid).mean(axis=0))
    return np.array(coefficients)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("container", [list, tuple, np.array, pd.Index, pd.Series])
def test_all_numeric_reference_values_are_averaged(kind, backend, container):
    model = _model(kind, backend)
    at = {"x": [-2.0, 1.0, 3.0], "z": [-1.0, 2.0]}
    means = emmeans(
        model, "treatment", at={name: container(value) for name, value in at.items()}, type="link"
    )
    expected = _direct_rows(model, means.result.grid, at)

    assert_allclose(means._L, expected, atol=1e-14)
    assert_allclose(means.result.emmean, expected @ model.beta, atol=1e-14)
    covariance = expected @ model.vcov() @ expected.T
    assert_allclose(means.result.se**2, np.diag(covariance), atol=1e-14)
    assert means.result.grid.treatment.tolist() == ["C", "A", "B"]


@pytest.mark.parametrize("specs", [["treatment", "x"], ["x", "treatment"], ["x"]])
def test_numeric_specs_keep_each_requested_value(specs):
    model = _model()
    at = {"x": [3.0, -1.0, 0.5], "z": [-2.0, 2.0]}

    means = emmeans(model, specs, at=at)

    expected_grid = pd.DataFrame(
        itertools.product(
            *(["C", "A", "B"] if name == "treatment" else at[name] for name in specs)
        ),
        columns=specs,
    )
    assert_frame_equal(means.result.grid, expected_grid)
    assert_allclose(means._L, _direct_rows(model, expected_grid, at), atol=1e-14)
    assert (
        len(means.pairs(adjust="none").estimate)
        == len(expected_grid) * (len(expected_grid) - 1) // 2
    )


def test_numeric_spec_defaults_to_reduced_covariate():
    means = emmeans(_model(), "x")

    assert_frame_equal(means.result.grid, pd.DataFrame({"x": [0.5]}))


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("method", ["pairwise", "trt.vs.ctrl", "custom"])
@pytest.mark.parametrize("adjust", ["none", "bonferroni", "holm", "tukey"])
def test_grouped_contrasts_match_separate_reference_grids(kind, method, adjust):
    model = _model(kind)
    values = {"site": ["west", "east"], "x": [2.0, -1.0]}
    means = emmeans(model, "treatment", by=["site", "x"], at=values)
    coefficients = np.array([[-1.0, 0.5, 0.5], [1.0, -1.0, 0.0]])

    def compare(means):
        if method == "pairwise":
            return means.pairs(adjust=adjust)
        return means.contrast(coefficients if method == "custom" else method, adjust=adjust)

    actual = compare(means)
    assert actual.grid.columns.tolist() == ["site", "x"]
    offset = 0
    for site, x in itertools.product(values["site"], values["x"]):
        expected = compare(emmeans(model, "treatment", at={"site": site, "x": x}))
        stop = offset + len(expected.estimate)
        for field in ("estimate", "se", "t_ratio", "p_value"):
            assert_allclose(
                getattr(actual, field)[offset:stop], getattr(expected, field), atol=1e-13
            )
        assert actual.grid.iloc[offset:stop].site.tolist() == [site] * (stop - offset)
        assert actual.grid.iloc[offset:stop].x.tolist() == [x] * (stop - offset)
        assert actual.contrast[offset:stop] == [
            f"{label} | site={site}, x={x}" for label in expected.contrast
        ]
        offset = stop
    assert offset == len(actual.estimate)
    assert "site=west" in str(actual)
    assert "site" in str(means)


@pytest.mark.parametrize("adjust", ["bonferroni", "tukey"])
def test_group_adjustment_uses_only_its_three_means(adjust):
    means = emmeans(_model(), "treatment", by="site")
    actual = means.pairs(adjust=adjust)
    expected_rows = _direct_rows(_model(), means.result.grid)
    left = [0, 0, 2, 1, 1, 3]
    right = [2, 4, 4, 3, 5, 5]
    differences = expected_rows[left] - expected_rows[right]
    covariance = differences @ means._vcov @ differences.T
    expected_t = differences @ means._beta / np.sqrt(np.diag(covariance))
    expected_p = (
        np.minimum(6 * stats.t.sf(np.abs(expected_t), means._df), 1.0)
        if adjust == "bonferroni"
        else stats.studentized_range.sf(np.sqrt(2) * np.abs(expected_t), 3, means._df)
    )

    assert_allclose(actual.estimate, differences @ means._beta, atol=1e-13)
    assert_allclose(actual.p_value, expected_p, atol=1e-13)


def test_by_also_in_specs_is_not_compared_between_groups():
    model = _model()
    explicit = emmeans(model, ["treatment", "site"], by="site")
    implicit = emmeans(model, "treatment", by="site")

    assert_frame_equal(explicit.result.grid, implicit.result.grid)
    assert explicit.pairs().contrast == implicit.pairs().contrast
    assert_allclose(explicit.pairs().estimate, implicit.pairs().estimate)


def test_grouping_positional_and_legacy_keyword_are_preserved():
    model = _model()
    expected = emmeans(model, "treatment", by="site")

    for actual in (emmeans(model, "treatment", "site"), emmeans(model, "treatment", _by="site")):
        assert_frame_equal(actual.result.grid, expected.result.grid)
        assert_allclose(actual._L, expected._L)
        assert actual.pairs().contrast == expected.pairs().contrast


def test_no_grouping_preserves_ordinary_contrast_labels():
    means = emmeans(_model(), "treatment", by=[])
    pairs = means.pairs()

    assert pairs.contrast == ["C - A", "C - B", "A - B"]
    assert pairs.grid is None


@pytest.mark.parametrize("contrasts", [{"treatment": "sum"}, {"treatment": "helmert"}])
def test_grouped_grid_preserves_fitted_contrasts(contrasts):
    model = _model(contrasts=contrasts)
    means = emmeans(model, "treatment", by="x", at={"x": [-1.0, 2.0]})

    assert_allclose(means._L, _direct_rows(model, means.result.grid), atol=1e-14)


def test_at_overrides_skip_covariate_reducer():
    def unexpected_reducer(column):
        raise AssertionError("overridden covariates need no reduction")

    means = emmeans(
        _model(), "treatment", at={"x": np.array(1.0), "z": -2.0}, cov_reduce=unexpected_reducer
    )

    assert len(means.result.emmean) == 3


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_custom_covariate_reducer_is_used_for_each_unspecified_numeric_variable(backend):
    calls = []

    def maximum(column):
        calls.append(column.name)
        return column.max()

    model = _model(backend=backend)
    means = emmeans(model, "treatment", cov_reduce=maximum)

    assert calls == ["x", "z"]
    assert_allclose(means._L, _direct_rows(model, means.result.grid, {"x": [2.0], "z": [1.5]}))


def test_empty_specs_returns_grand_mean():
    model = _model()
    means = emmeans(model, [], at={"x": [-1.0, 2.0]})

    assert means.result.grid.shape == (1, 0)
    assert_allclose(means._L, _direct_rows(model, means.result.grid, {"x": [-1.0, 2.0]}))


def test_grouped_generalized_means_preserve_response_transformation():
    model = _model("glmm")
    kwargs = dict(specs="treatment", by="x", at={"x": [-1.0, 2.0]}, level=0.9)
    link = emmeans(model, type="link", **kwargs)
    response = emmeans(model, type="response", **kwargs)

    assert_frame_equal(link.result.grid, response.result.grid)
    assert_allclose(response.result.emmean, np.exp(link.result.emmean))
    assert_allclose(response.result.se, np.exp(link.result.emmean) * link.result.se)
    assert_allclose(response.result.lower, np.exp(link.result.lower))
    assert_allclose(response.result.upper, np.exp(link.result.upper))
    assert_allclose(response.pairs().estimate, link.pairs().estimate)


@pytest.mark.parametrize("levels", [(3, 1, 2), (1, "second", 3)])
def test_categorical_reference_levels_preserve_values_and_order(levels):
    model = _model(treatments=levels)
    means = emmeans(model, "treatment", at={"site": "west", "x": 1.0, "z": 0.0})

    assert means.result.grid.treatment.tolist() == list(levels)
    expected = model.predict(means.result.grid.assign(site="west", x=1.0, z=0.0), re_form="NA")
    assert_allclose(means.result.emmean, expected)


def test_group_metadata_preserves_large_integer_levels_with_numeric_covariates():
    levels = (2**53 + 1, 2**53 + 3, 2**53 + 5)
    means = emmeans(_model(treatments=levels), "site", by=["treatment", "x"], at={"x": [-1.0, 2.0]})

    comparisons = means.pairs(adjust="none")

    assert comparisons.grid.treatment.tolist() == [value for value in levels for _ in range(2)]
    assert comparisons.contrast[0].endswith(f"treatment={levels[0]}, x=-1.0")


@pytest.mark.parametrize("at", [[1.0, 2.0], {1: 2.0}])
def test_invalid_reference_mapping_types_raise(at):
    with pytest.raises(TypeError, match="at"):
        emmeans(_model(), "treatment", at=at)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"at": {"missing": 1}}, "Unknown fixed-effect predictor"),
        ({"at": {"g": 1}}, "Unknown fixed-effect predictor"),
        ({"at": {"x": []}}, "nonempty"),
        ({"at": {"x": [[1.0, 2.0]]}}, "1-D"),
        ({"at": {"x": [0.0, np.nan]}}, "missing"),
        ({"at": {"x": [np.inf]}}, "finite"),
        ({"at": {"x": [1.0, 1.0]}}, "distinct"),
        ({"at": {"x": ["text"]}}, "numeric"),
        ({"at": {"treatment": []}}, "nonempty"),
        ({"at": {"treatment": ["A", None]}}, "missing"),
        ({"at": {"treatment": ["A", "A"]}}, "distinct"),
        ({"at": {"treatment": ["unknown"]}}, "New level"),
        ({"by": "missing"}, "must name a fixed-effect predictor"),
        ({"by": ["site", "site"]}, "duplicate"),
        ({"by": "site", "_by": "site"}, "only one"),
        ({"level": 0.0}, "level"),
        ({"level": 1.0}, "level"),
        ({"level": np.nan}, "level"),
    ],
)
def test_invalid_grid_settings_fail_before_covariance_work(kwargs, message, monkeypatch):
    model = _model()

    def unexpected_covariance():
        raise AssertionError("invalid grid settings must fail before covariance work")

    monkeypatch.setattr(model, "vcov", unexpected_covariance)
    with pytest.raises(ValueError, match=message):
        emmeans(model, "treatment", **kwargs)


@pytest.mark.parametrize("specs", [None, 3, ["treatment", 1]])
def test_invalid_specification_types_raise(specs):
    with pytest.raises(TypeError, match="specs"):
        emmeans(_model(), specs)


def test_duplicate_specs_raise():
    with pytest.raises(ValueError, match="duplicate"):
        emmeans(_model(), ["treatment", "treatment"])


def test_group_with_one_mean_cannot_produce_pairwise_comparisons():
    means = emmeans(_model(), "site", by="site")

    with pytest.raises(ValueError, match="at least 2 levels"):
        means.pairs()

    control = means.contrast("trt.vs.ctrl")
    assert control.estimate.shape == (0,)
    assert control.grid.shape == (0, 1)


def test_grouped_comparisons_share_design_and_covariance_setup(monkeypatch):
    model = _model()
    calls = {"design": 0, "covariance": 0}
    original_design = model._prediction_fixed_matrix
    original_covariance = model.vcov

    def design(grid):
        calls["design"] += 1
        return original_design(grid)

    def covariance():
        calls["covariance"] += 1
        return original_covariance()

    monkeypatch.setattr(model, "_prediction_fixed_matrix", design)
    monkeypatch.setattr(model, "vcov", covariance)
    means = emmeans(model, "treatment", by="x", at={"x": np.linspace(-1, 2, 16)})
    result = means.pairs(adjust="none")

    assert len(result.estimate) == 16 * 3
    assert calls == {"design": 1, "covariance": 1}


def test_fitted_model_conditional_means_equal_predictions():
    model = _model()
    data = model.model_frame()
    fitted = lmer(str(model.formula), data)
    means = emmeans(fitted, ["treatment", "site"], by="x", at={"x": [-1.0, 2.0], "z": 0.0})
    prediction_grid = means.result.grid.assign(z=0.0)

    assert_allclose(means.result.emmean, fitted.predict(prediction_grid, re_form="NA"))
    assert_array_equal(means.result.grid.x.unique(), [-1.0, 2.0])
