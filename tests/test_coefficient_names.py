import numpy as np
import pandas as pd
import pytest
from mixedlm import bootCI, families, lmList, parse_formula, tidy
from mixedlm.inference import ddf
from mixedlm.inference.bootstrap import BootstrapResult
from mixedlm.inference.ddf import DenomDFResult, pvalues_with_ddf
from mixedlm.inference.hypothesis import linear_hypothesis
from mixedlm.inference.profile import profile_glmer, profile_lmer, slice2D
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult
from mixedlm.utils import _format_pvalue
from scipy import stats


def _model(kind="lmm", collision=True):
    rng = np.random.default_rng(251)
    data = pd.DataFrame(
        {
            "a": pd.Categorical(np.arange(120) % 3),
            "a.1": rng.normal(size=120),
            "x": rng.normal(size=120),
            "g": np.arange(120) % 10,
            "y": rng.normal(size=120),
        }
    )
    rhs = "a + `a.1`" if collision else "a + x"
    formula = parse_formula(f"y ~ {rhs} + (1 | g)")
    matrices = build_model_matrices(formula, data)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.3]),
        beta=np.array([0.25, 1.1, -0.6, 2.0]),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    result = (
        LmerResult(**common, sigma=0.7, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    result.vcov = lambda: np.diag([0.01, 0.04, 0.09, 0.16])
    return result, data


def _patch_df(monkeypatch, model, method):
    calls = []
    dfs = np.array([7.0, 11.0, 19.0, 37.0])

    def calculate(result):
        calls.append(result)
        return DenomDFResult(dfs.copy(), method, list(result.matrices.fixed_names))

    monkeypatch.setattr(
        ddf, "satterthwaite_df" if method == "Satterthwaite" else "kenward_roger_df", calculate
    )
    return dfs, calls


@pytest.mark.parametrize("method", ["Satterthwaite", "Kenward-Roger", "normal", None])
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_tidy_preserves_every_coefficient_and_its_uncertainty(monkeypatch, kind, method):
    model, _ = _model(kind)
    dfs = np.full(4, np.nan)
    calls = []
    if kind == "lmm" and method not in (None, "normal"):
        dfs, calls = _patch_df(monkeypatch, model, method)
    table = tidy(model, conf_int=True, conf_level=0.9, ddf_method=method)
    assert table["term"].tolist() == model.matrices.fixed_names
    np.testing.assert_array_equal(table["estimate"], model.beta)
    se = np.array([0.1, 0.2, 0.3, 0.4])
    statistic = model.beta / se
    np.testing.assert_allclose(table["std.error"], se)
    np.testing.assert_array_equal(table["df"], dfs)
    if np.isfinite(dfs).all():
        expected_p = 2 * stats.t.sf(abs(statistic), dfs)
        critical = stats.t.ppf(0.95, dfs)
        assert len(calls) == 1
    else:
        expected_p = 2 * stats.norm.sf(abs(statistic))
        critical = stats.norm.ppf(0.95)
    np.testing.assert_allclose(table["p.value"], expected_p)
    np.testing.assert_allclose(table["conf.low"], model.beta - critical * se)
    np.testing.assert_allclose(table["conf.high"], model.beta + critical * se)
    assert table.iloc[1]["estimate"] != table.iloc[3]["estimate"]


@pytest.mark.parametrize("method", ["Satterthwaite", "Kenward-Roger"])
@pytest.mark.parametrize("collision", [False, True])
def test_summary_reuses_one_df_calculation_and_keeps_pvalues_positional(
    monkeypatch, method, collision
):
    model, _ = _model(collision=collision)
    dfs, calls = _patch_df(monkeypatch, model, method)
    summary = model.summary(ddf_method=method)
    assert len(calls) == 1
    expected = 2 * stats.t.sf(np.abs(model.beta / [0.1, 0.2, 0.3, 0.4]), dfs)
    rows = [line for line in summary.splitlines() if line.startswith("a.1 ")]
    assert len(rows) == (2 if collision else 1)
    indices = [1, 3] if collision else [1]
    for row, index in zip(rows, indices, strict=True):
        assert _format_pvalue(expected[index]) in row
        assert f"{dfs[index]:.2f}" in row


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_dictionary_coefficients_reject_ambiguous_names(kind):
    model, _ = _model(kind)
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'.*tidy"):
        model.fixef()
    # The positional output remains complete and does not call fixef().
    np.testing.assert_array_equal(model.tidy(ddf_method="normal")["estimate"], model.beta)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("form", ["mapping", "nested", "sequence", "dataframe"])
def test_named_hypotheses_reject_only_the_ambiguous_selection(kind, form):
    model, _ = _model(kind)

    def specification(name):
        return {
            "mapping": {name: 1.0},
            "nested": {"test": {name: 1.0}},
            "sequence": [{name: 1.0}],
            "dataframe": pd.DataFrame({name: [1.0]}),
        }[form]

    with pytest.raises(
        ValueError, match="Ambiguous coefficient name 'a.1'.*numeric constraint matrix"
    ):
        linear_hypothesis(model, specification("a.1"))
    selected = linear_hypothesis(model, specification("a.2"))
    np.testing.assert_array_equal(selected.constraints, [[0, 0, 1, 0]])
    np.testing.assert_allclose(selected.estimate, [model.beta[2]])
    positional = linear_hypothesis(model, np.eye(4)[[1, 3]], labels=["factor", "numeric"])
    np.testing.assert_array_equal(positional.estimate, model.beta[[1, 3]])
    np.testing.assert_allclose(positional.std_error, [0.2, 0.4])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("method", ["Wald", "profile", "boot"])
@pytest.mark.parametrize("selection", [None, "a.1", ["a.1", "a.2"]])
def test_named_intervals_reject_ambiguity_before_covariance_or_refitting(
    monkeypatch, kind, method, selection
):
    model, _ = _model(kind)

    def unexpected(*args, **kwargs):
        raise AssertionError("Ambiguous names need no inference work")

    monkeypatch.setattr(model, "vcov", unexpected)
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'"):
        model.confint(parm=selection, method=method, n_boot=2)


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_unambiguous_wald_selection_is_usable_with_other_repeated_names(kind):
    model, _ = _model(kind)
    ci = model.confint("a.2", level=0.9)
    critical = stats.norm.ppf(0.95)
    np.testing.assert_allclose(ci["a.2"], [-0.6 - 0.3 * critical, -0.6 + 0.3 * critical])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
@pytest.mark.parametrize("selection", [None, "a.1", ["a.1"]])
def test_direct_profile_rejects_ambiguity_before_work(monkeypatch, kind, selection):
    model, _ = _model(kind)
    model.vcov = lambda: pytest.fail("No covariance needed for ambiguous profile")
    function = profile_lmer if kind == "lmm" else profile_glmer
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'"):
        function(model, which=selection)


def test_named_slice_rejects_ambiguous_axis():
    model, _ = _model()
    model.vcov = lambda: pytest.fail("No covariance needed for ambiguous slice")
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'"):
        slice2D(model, "a.1", "a.2", n_points=3)


def test_degrees_of_freedom_keep_positional_access_but_reject_ambiguous_keys():
    result = DenomDFResult(np.array([4.0, 12.0, 7.0]), "test", ["x", "x", "z"])
    np.testing.assert_array_equal(result.df, [4, 12, 7])
    assert result["z"] == 7
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'x'"):
        result["x"]
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'x'"):
        result.as_dict()


@pytest.mark.parametrize("method", ["Satterthwaite", "Kenward-Roger"])
def test_named_pvalues_reject_ambiguity_before_calculating_df(monkeypatch, method):
    model, _ = _model()

    def unexpected(*args, **kwargs):
        raise AssertionError("No degrees of freedom needed for an ambiguous dictionary")

    monkeypatch.setattr(ddf, "satterthwaite_df", unexpected)
    monkeypatch.setattr(ddf, "kenward_roger_df", unexpected)
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'"):
        pvalues_with_ddf(model, method)


@pytest.mark.parametrize("zero_se", [False, True])
def test_vector_pvalues_match_individual_tails(monkeypatch, zero_se):
    model, _ = _model(collision=False)
    if zero_se:
        model.vcov = lambda: np.diag([0.0, 0.04, 0.09, 0.16])
    dfs, calls = _patch_df(monkeypatch, model, "Satterthwaite")
    result = pvalues_with_ddf(model)
    assert len(calls) == 1
    for index, name in enumerate(model.matrices.fixed_names):
        se = np.sqrt(model.vcov()[index, index])
        expected_t = model.beta[index] / se if se > 0 else np.nan
        np.testing.assert_allclose(
            result[name],
            [model.beta[index], expected_t, 2 * stats.t.sf(abs(expected_t), dfs[index])],
            equal_nan=True,
        )


@pytest.mark.parametrize("method", ["percentile", "basic", "normal"])
def test_bootstrap_tables_preserve_duplicate_names_when_dictionaries_cannot(method):
    samples = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 5.0], [4.0, 7.0]])
    result = BootstrapResult(
        n_boot=4,
        beta_samples=samples,
        theta_samples=np.empty((4, 0)),
        sigma_samples=None,
        fixed_names=["x", "x"],
        original_beta=np.array([1.0, 4.0]),
        original_theta=np.empty(0),
        original_sigma=None,
        n_failed=0,
    )
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'x'.*bootCI"):
        result.ci(method=method)
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'x'.*bootCI"):
        result.se()
    table = bootCI(result, method=method)
    assert table["parameter"].tolist() == ["x", "x"]
    np.testing.assert_array_equal(table["estimate"], [1, 4])
    np.testing.assert_allclose(table["std.error"], samples.std(axis=0, ddof=1))
    assert result.summary().count("x ") == 2


def test_lmlist_rejects_unrepresentable_named_coefficients():
    _, data = _model()
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'a.1'.*lmList"):
        lmList("y ~ a + `a.1` | g", data)


def test_tidy_checks_coefficient_and_df_lengths(monkeypatch):
    model, _ = _model()
    model.beta = np.ones(3)
    with pytest.raises(ValueError, match="Fixed estimates have shape"):
        tidy(model, ddf_method="normal")
    model.beta = np.ones(4)
    monkeypatch.setattr(
        ddf, "satterthwaite_df", lambda result: DenomDFResult(np.ones(3), "test", ["x"] * 3)
    )
    with pytest.raises(ValueError, match="degrees of freedom must have shape"):
        tidy(model)


def test_nonlinear_tables_preserve_duplicate_parameter_names():
    from types import SimpleNamespace

    from mixedlm.models.nlmer import NlmerResult

    model = NlmerResult(
        model=SimpleNamespace(param_names=["same", "same"]),
        group_var="g",
        phi=np.array([1.0, 3.0]),
        theta=np.empty(0),
        sigma=1.0,
        b=np.empty((2, 0)),
        random_params=[],
        deviance=0.0,
        converged=True,
        n_iter=0,
        x=np.arange(12.0),
        y=np.ones(12),
        groups=np.arange(12) % 2,
        group_levels=["a", "b"],
    )
    model.vcov = lambda: np.diag([0.04, 0.25])
    with pytest.raises(ValueError, match="Ambiguous coefficient name 'same'"):
        model.fixef()
    for method in ["Wald", "boot"]:
        with pytest.raises(ValueError, match="Ambiguous coefficient name 'same'"):
            model.confint(method=method, n_boot=1)
    table = tidy(model, conf_int=True)
    assert table["term"].tolist() == ["same", "same"]
    np.testing.assert_array_equal(table["estimate"], model.phi)
    np.testing.assert_allclose(table["std.error"], [0.2, 0.5])


@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_actual_fit_reports_every_coefficient_with_repeated_names(kind):
    from mixedlm import glmer, glmerControl, lmer, lmerControl

    _, data = _model(kind)
    rng = np.random.default_rng(28)
    linear = (
        0.4
        + 0.7 * (data["a"].astype(int) == 1)
        - 0.2 * (data["a"].astype(int) == 2)
        + 0.3 * data["a.1"]
    )
    data["y"] = (
        linear + rng.normal(scale=0.4, size=len(data))
        if kind == "lmm"
        else rng.poisson(np.exp(linear))
    )
    if kind == "lmm":
        model = lmer("y ~ a + `a.1` + (1 | g)", data, control=lmerControl(check_singular=False))
    else:
        model = glmer(
            "y ~ a + `a.1` + (1 | g)",
            data,
            family=families.Poisson(),
            control=glmerControl(check_singular=False),
        )
    table = tidy(model, conf_int=True)
    assert len(table) == model.matrices.n_fixed == 4
    assert table["term"].tolist().count("a.1") == 2
    np.testing.assert_array_equal(table["estimate"], model.beta)
    np.testing.assert_allclose(table["std.error"], np.sqrt(np.diag(model.vcov())))
