import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    coef,
    families,
    fixef,
    getME,
    glmer,
    glmer_nb,
    lmer,
    load_cake,
    load_cbpp,
    load_dyestuff,
    load_pastes,
    load_sleepstudy,
    ranef,
)
from mixedlm.inference.ddf import kenward_roger_df, pvalues_with_ddf, satterthwaite_df
from mixedlm.models.control import lmerControl
from mixedlm.utils.contrasts import contr_helmert, contr_poly, contr_sum, contr_treatment
from mixedlm.utils.lme4_compat import (
    DevComp,
    VarCorr,
    checkConv,
    convergence_ok,
    devcomp,
    dummy,
    factorize,
    fortify,
    isNested,
    mkMerMod,
    ngrps,
    pvalues,
    scale_vcov,
    sigma,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats


@pytest.fixture(scope="module")
def lmer_model():
    sleepstudy = load_sleepstudy()
    return lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)


@pytest.fixture(scope="module")
def glmer_model():
    rng = np.random.default_rng(123)
    n = 100
    n_groups = 15
    group = np.repeat(np.arange(n_groups), n // n_groups + 1)[:n]
    x = rng.normal(size=n)
    re = rng.normal(scale=0.5, size=n_groups)
    eta = -1 + 0.5 * x + re[group]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    data = pd.DataFrame({"y": y, "x": x, "group": [f"h{g}" for g in group]})
    return glmer("y ~ x + (1 | group)", data, family=families.Binomial())


class TestAccessorFunctions:
    @pytest.mark.parametrize("fixture", ["lmer_model", "glmer_model"])
    def test_functions_return_the_model_methods(self, fixture, request) -> None:
        model = request.getfixturevalue(fixture)

        assert sigma(model) == (model.sigma if fixture == "lmer_model" else 1.0)
        assert ngrps(model) == model.ngrps()
        assert fixef(model) == model.fixef()
        assert str(VarCorr(model)) == str(model.VarCorr())
        for function, method in [(ranef, model.ranef), (coef, model.coef)]:
            actual, expected = function(model), method()
            assert actual.keys() == expected.keys()
            for group, terms in expected.items():
                assert actual[group].keys() == terms.keys()
                for term, values in terms.items():
                    assert_array_equal(actual[group][term], values)
        for name in ("X", "y", "theta", "beta", "u"):
            assert_array_equal(getME(model, name), model.getME(name))
        assert_array_equal(getME(model, "Z").toarray(), model.getME("Z").toarray())

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason="ranef(model, condVar=True) returns only the modes and drops condVar",
    )
    @pytest.mark.parametrize("fixture", ["lmer_model", "glmer_model"])
    def test_ranef_function_keeps_conditional_variances(self, fixture, request) -> None:
        model = request.getfixturevalue(fixture)
        expected = model.ranef(condVar=True)
        actual = ranef(model, condVar=True)

        for group, terms in expected.condVar.items():
            for term, variances in terms.items():
                assert_array_equal(actual[group][term], expected[group][term])
                assert_array_equal(actual.condVar[group][term], variances)

    def test_ngrps_counts_levels_of_each_grouping_factor(self, lmer_model, glmer_model) -> None:
        assert ngrps(lmer_model) == {"Subject": 18}
        assert ngrps(glmer_model) == {"group": 15}

    def test_root_accessors_are_canonical_functions(self) -> None:
        import mixedlm.utils.lme4_compat as compat

        assert coef is compat.coef
        assert fixef is compat.fixef
        assert getME is compat.getME
        assert ranef is compat.ranef

    def test_getME_invalid(self, lmer_model) -> None:
        with pytest.raises(ValueError, match="Unknown component name"):
            getME(lmer_model, "invalid_component")


class TestPvalues:
    @pytest.fixture
    def lmer_model(self):
        sleepstudy = load_sleepstudy()
        return lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

    def test_pvalues_normal(self, lmer_model) -> None:
        z = lmer_model.beta / np.sqrt(np.diag(lmer_model.vcov()))
        expected = dict(zip(lmer_model.fixef(), 2 * stats.norm.sf(np.abs(z)), strict=True))

        assert pvalues(lmer_model, method="normal") == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        ("method", "canonical"),
        [
            ("Satterthwaite", "Satterthwaite"),
            ("satt", "Satterthwaite"),
            ("KR", "Kenward-Roger"),
            ("kenward_roger", "Kenward-Roger"),
        ],
    )
    def test_pvalues_matches_canonical_ddf(self, lmer_model, method, canonical) -> None:
        expected = {
            name: values[2]
            for name, values in pvalues_with_ddf(lmer_model, method=canonical).items()
        }

        assert pvalues(lmer_model, method=method) == pytest.approx(expected)

    def test_denominator_df_methods_report_canonical_names(self, lmer_model) -> None:
        assert satterthwaite_df(lmer_model).method == "Satterthwaite"
        assert kenward_roger_df(lmer_model).method == "Kenward-Roger"

    def test_pvalues_invalid_method(self, lmer_model) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            pvalues(lmer_model, method="invalid")

    def test_glmm_pvalues_validate_the_method(self) -> None:
        data = load_cbpp()
        model = glmer("incidence / size ~ period + (1 | herd)", data, family=families.Binomial())
        z = model.beta / np.sqrt(np.diag(model.vcov()))
        expected = dict(zip(model.fixef(), 2 * stats.norm.sf(np.abs(z)), strict=True))

        with pytest.raises(ValueError, match="Unknown method 'bogus'"):
            pvalues(model, method="bogus")
        for method in ("Satterthwaite", "KR", "normal"):
            assert pvalues(model, method=method) == pytest.approx(expected)


@pytest.fixture(scope="module")
def converged_model():
    return lmer(
        "Reaction ~ Days + (Days | Subject)",
        load_sleepstudy(),
        control=lmerControl(optimizer="COBYQA"),
    )


class TestConvergenceFunctions:
    def test_checkConv_reports_the_fit_metadata(self, converged_model) -> None:
        info = checkConv(converged_model)

        assert info.converged and not info.is_singular
        assert info.optimizer == converged_model.optimizer == "COBYQA"
        assert info.iterations == converged_model.n_iter > 0
        assert info.messages == []
        assert convergence_ok(converged_model)

    def test_checkConv_reports_the_optimizer_message(self) -> None:
        stalled = lmer(
            "Reaction ~ Days + (Days | Subject)",
            load_sleepstudy(),
            control=lmerControl(optimizer="COBYQA", maxiter=2, check_conv=False),
        )
        info = checkConv(stalled)

        assert not info.converged
        assert info.iterations == 2
        assert info.messages == [f"Optimizer did not report convergence: {stalled.message}"]
        assert not convergence_ok(stalled)

    def test_singularity_tolerance_controls_convergence_ok(self, converged_model) -> None:
        info = checkConv(converged_model, tol=1e3)

        assert info.is_singular
        assert info.messages == ["Model is singular (boundary fit) at tolerance 1000.0"]
        assert not convergence_ok(converged_model, tol=1e3)


class TestFortify:
    def test_fortify_basic(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
        fortified = fortify(result, sleepstudy)

        assert isinstance(fortified, pd.DataFrame)
        assert len(fortified) == len(sleepstudy)
        assert_allclose(fortified[".fitted"], result.fitted())
        assert_allclose(fortified[".resid"], result.residuals())
        assert_allclose(fortified[".fixed"], result.matrices.X @ result.beta)

    def test_fortify_fitted_plus_resid(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
        fortified = fortify(result, sleepstudy)

        reconstructed = fortified[".fitted"] + fortified[".resid"]
        assert np.allclose(reconstructed, sleepstudy["Reaction"], rtol=1e-10)

    @pytest.mark.parametrize("na_action", ["omit", "exclude"])
    def test_fortify_aligns_rows_dropped_for_missing_values(self, na_action) -> None:
        data = load_sleepstudy()
        data.loc[[0, 90], "Reaction"] = np.nan
        result = lmer("Reaction ~ Days + (Days | Subject)", data, na_action=na_action)
        fortified = fortify(result, data)

        assert len(fortified) == len(data)
        dropped = data["Reaction"].isna()
        assert fortified.loc[dropped, [".fitted", ".resid", ".fixed"]].isna().all().all()
        kept = fortified[~dropped]
        assert_allclose(kept["Reaction"] - kept[".fitted"], kept[".resid"], atol=1e-9)
        assert_allclose(kept[".fitted"], result.fitted(na_expand=False))
        assert len(fortify(result)) == result.nobs()

    def test_fortify_rejects_data_of_another_length(self) -> None:
        data = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", data)

        with pytest.raises(ValueError, match="data has 179 rows"):
            fortify(result, data.iloc[1:])

    def test_fortify_population_fit_excludes_random_effects(self) -> None:
        data = load_sleepstudy()
        offset = np.linspace(-1.0, 1.0, len(data))
        result = lmer("Reaction ~ Days + (Days | Subject)", data, offset=offset)
        conditional = fortify(result, data)
        population = fortify(result, data, include_re=False)
        fixed = result.matrices.X @ result.beta + offset

        assert_allclose(population[".fitted"], fixed)
        assert_allclose(conditional[".fixed"], fixed)
        assert not np.allclose(conditional[".fitted"], population[".fitted"])

    def test_fortify_glmm_uses_the_response_scale(self) -> None:
        data = load_cbpp()
        result = glmer("incidence / size ~ period + (1 | herd)", data, family=families.Binomial())
        conditional = fortify(result, data)
        population = fortify(result, data, include_re=False)

        assert_allclose(conditional[".fitted"], result.fitted(type="response"))
        assert_allclose(conditional[".mu"], conditional[".fitted"])
        assert_allclose(population[".fitted"], result.family.link.inverse(conditional[".fixed"]))
        assert_allclose(population[".mu"], conditional[".mu"])


class TestDevcomp:
    def test_devcomp_matches_getme_for_lmm(self) -> None:
        result = lmer("Reaction ~ Days + (1 | Subject)", load_sleepstudy())
        dc = devcomp(result)
        expected = result.getME("devcomp")

        assert isinstance(dc, DevComp)
        assert dc.cmp.keys() == expected["cmp"].keys()
        assert_allclose(list(dc.cmp.values()), list(expected["cmp"].values()), equal_nan=True)
        assert dc.dims == expected["dims"]
        assert (dc.dims["n"], dc.dims["p"], dc.dims["q"], dc.dims["ngrps"]) == (180, 2, 18, 1)
        assert dc.cmp["REML"] == result.deviance
        assert "pwrss" in str(dc)

    def test_devcomp_rejects_nonlinear_models(self) -> None:
        from tests._nlmm_models import fit_nlme

        with pytest.raises(TypeError, match="devcomp\\(\\) is not available for NlmerResult"):
            devcomp(fit_nlme())


class TestDummy:
    X = np.array(["b", "c", "a", "b", "c", "a", "c"])
    CODES = np.array([1, 2, 0, 1, 2, 0, 2])

    @pytest.mark.parametrize(
        ("contrasts", "matrix"),
        [
            ("treatment", contr_treatment(3)),
            ("sum", contr_sum(3)),
            ("helmert", contr_helmert(3)),
            ("poly", contr_poly(3)),
        ],
    )
    def test_rows_match_the_contrast_matrix(self, contrasts, matrix) -> None:
        assert_allclose(dummy(self.X, contrasts=contrasts), matrix[self.CODES])

    def test_poly_contrasts_are_orthonormal(self) -> None:
        coded = dummy(["low", "mid", "high"], contrasts="poly")
        # Levels sort as high < low < mid; each row codes its level.
        expected = contr_poly(3)[[1, 2, 0]]

        assert_allclose(coded, expected)
        assert_allclose(contr_poly(3).T @ contr_poly(3), np.eye(2), atol=1e-12)

    def test_ordered_categorical_keeps_level_order(self) -> None:
        levels = ["low", "mid", "high"]
        x = pd.Categorical(["high", "low", "mid", "low"], categories=levels, ordered=True)

        assert_allclose(dummy(x, contrasts="poly"), contr_poly(3)[[2, 0, 1, 0]])
        assert_allclose(dummy(pd.Series(x), base="mid"), [[0, 1], [1, 0], [0, 0], [1, 0]])

    @pytest.mark.parametrize("base", ["b", 1, -2])
    def test_base_level_by_name_or_index(self, base) -> None:
        expected = np.column_stack([self.X == "a", self.X == "c"]).astype(float)

        assert_allclose(dummy(self.X, base=base), expected)

    @pytest.mark.parametrize(
        ("base", "message"),
        [(3, "base index 3 is out of range"), (-4, "base index -4"), ("z", "'z' is not a level")],
    )
    def test_invalid_base_raises(self, base, message) -> None:
        with pytest.raises(ValueError, match=message):
            dummy(self.X, base=base)

    def test_unknown_contrast_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown contrast type"):
            dummy(self.X, contrasts="bogus")


class TestScaleVcov:
    def test_scale_only_predictor_length(self) -> None:
        vcov = np.array([[1.0, 0.1], [0.1, 0.5]])
        adjusted = scale_vcov(vcov, scale=np.array([2.0]))
        expected = np.array([[1.0, 0.2], [0.2, 2.0]])
        assert np.allclose(adjusted, expected)

    def test_center_and_scale_adjust_intercept(self) -> None:
        vcov = np.array([[1.0, 0.1], [0.1, 0.5]])
        center = np.array([3.0])
        scale = np.array([2.0])

        adjusted = scale_vcov(vcov, center=center, scale=scale)
        jacobian = np.array([[1.0, -6.0], [0.0, 2.0]])
        expected = jacobian @ vcov @ jacobian.T

        assert np.allclose(adjusted, expected)

    def test_invalid_vector_length_raises(self) -> None:
        vcov = np.eye(3)
        with pytest.raises(ValueError, match="must be 2 or 3"):
            scale_vcov(vcov, center=np.array([1.0]))


class TestFactorize:
    def test_factorize_basic(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "x"], "b": [1, 2, 3]})
        result = factorize(df, columns=["a"])

        assert result["a"].dtype.name == "category"
        assert result["b"].dtype == np.int64

    def test_factorize_all_object_columns(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "c": [1, 2]})
        result = factorize(df)

        assert result["a"].dtype.name == "category"
        assert result["b"].dtype.name == "category"
        assert result["c"].dtype == np.int64

    def test_factorize_no_modification_inplace(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"]})
        original_dtype = df["a"].dtype
        result = factorize(df, columns=["a"])

        assert df["a"].dtype == original_dtype
        assert result["a"].dtype.name == "category"


class TestMkMerMod:
    def test_mkmermod_copy(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        new_model = mkMerMod(result)

        assert new_model is not result
        assert_array_equal(new_model.theta, result.theta)
        assert_array_equal(new_model.beta, result.beta)
        assert new_model.deviance == result.deviance

    def test_mkmermod_new_theta(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        new_theta = result.theta * 1.1
        new_model = mkMerMod(result, theta=new_theta)

        assert np.allclose(new_model.theta, new_theta)

    def test_mkmermod_new_beta(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        new_beta = result.beta * 1.1
        new_model = mkMerMod(result, beta=new_beta)

        assert np.allclose(new_model.beta, new_beta)


class TestGlmerNb:
    def test_glmer_nb_is_glmer_with_a_unit_theta_negative_binomial(self) -> None:
        rng = np.random.default_rng(42)
        group = np.repeat(np.arange(10), 10)
        x = rng.normal(size=len(group))
        mu = np.exp(1 + 0.5 * x + rng.normal(scale=0.5, size=10)[group])
        data = pd.DataFrame({"y": rng.poisson(mu), "x": x, "group": [f"g{g}" for g in group]})

        result = glmer_nb("y ~ x + (1 | group)", data)
        expected = glmer("y ~ x + (1 | group)", data, family=families.NegativeBinomial(theta=1.0))

        assert result.converged
        assert result.family.theta == 1.0
        assert_array_equal(result.beta, expected.beta)
        assert_array_equal(result.theta, expected.theta)
        assert result.deviance == expected.deviance


class TestIsNested:
    def test_nested_students_in_schools(self) -> None:
        students = ["s1", "s2", "s3", "s4", "s5", "s6"]
        schools = ["A", "A", "A", "B", "B", "B"]
        assert isNested(students, schools) is True

    def test_not_nested_crossed(self) -> None:
        factor1 = ["a", "b", "a", "b", "a", "b"]
        factor2 = ["X", "X", "Y", "Y", "Z", "Z"]
        assert isNested(factor1, factor2) is False

    def test_nested_with_numpy_arrays(self) -> None:
        factor1 = np.array([1, 2, 3, 4, 5, 6])
        factor2 = np.array(["A", "A", "B", "B", "C", "C"])
        assert isNested(factor1, factor2) is True

    def test_nested_with_pandas_series(self) -> None:
        df = pd.DataFrame({"student": ["s1", "s2", "s3", "s4"], "school": ["A", "A", "B", "B"]})
        assert isNested(df["student"], df["school"]) is True

    def test_nested_identical_factors(self) -> None:
        factor = ["A", "B", "C", "A", "B", "C"]
        assert isNested(factor, factor) is True

    def test_nested_with_real_data(self) -> None:
        sleepstudy = load_sleepstudy()
        assert isNested(sleepstudy["Days"], sleepstudy["Subject"]) is False


class TestBalancedDesigns:
    """Balanced designs have closed-form REML estimates: the ANOVA moment estimators."""

    @staticmethod
    def _means(data, response, by):
        return data.groupby(by)[response].transform("mean").to_numpy()

    def test_dyestuff_one_way_classification(self) -> None:
        data = load_dyestuff()
        result = lmer("Yield ~ 1 + (1 | Batch)", data)
        y = data["Yield"].to_numpy()
        batch = self._means(data, "Yield", "Batch")
        within = np.sum((y - batch) ** 2) / (30 - 6)
        between = np.sum((batch - y.mean()) ** 2) / (6 - 1)

        assert result.sigma**2 == pytest.approx(within, rel=1e-6)
        assert (result.theta[0] * result.sigma) ** 2 == pytest.approx(
            (between - within) / 5, rel=1e-6
        )
        assert_allclose(result.beta, [y.mean()], rtol=1e-12)
        assert_allclose(np.sqrt(result.vcov()), [[np.sqrt(between / 30)]], rtol=1e-6)
        # Published lme4 REML criterion at convergence.
        assert result.deviance == pytest.approx(319.6543, abs=1e-4)

    def test_pastes_nested_classification(self) -> None:
        data = load_pastes()
        result = lmer("strength ~ 1 + (1 | batch/cask)", data)
        y = data["strength"].to_numpy()
        cask = self._means(data, "strength", ["batch", "cask"])
        batch = self._means(data, "strength", "batch")
        within = np.sum((y - cask) ** 2) / (60 - 30)
        casks = np.sum((cask - batch) ** 2) / (30 - 10)
        batches = np.sum((batch - y.mean()) ** 2) / (10 - 1)
        variances = {
            structure.grouping_factor: (value * result.sigma) ** 2
            for structure, value in zip(
                result.matrices.random_structures, result.theta, strict=True
            )
        }

        assert result.sigma**2 == pytest.approx(within, rel=1e-6)
        assert variances["batch:cask"] == pytest.approx((casks - within) / 2, rel=1e-6)
        assert variances["batch"] == pytest.approx((batches - casks) / 6, rel=1e-5)
        assert_allclose(result.beta, [y.mean()], rtol=1e-12)
        assert_allclose(np.sqrt(result.vcov()), [[np.sqrt(batches / 60)]], rtol=1e-5)

    def test_cake_additive_split_plot(self) -> None:
        data = load_cake()
        result = lmer("angle ~ recipe + temperature + (1 | replicate)", data)
        y = data["angle"].to_numpy(dtype=float)
        recipe = self._means(data, "angle", "recipe")
        temperature = self._means(data, "angle", "temperature")
        replicate = self._means(data, "angle", "replicate")
        # Every replicate sees every recipe and temperature once, so the effects are orthogonal.
        residual = y - recipe - temperature - replicate + 2 * y.mean()
        error = np.sum(residual**2) / (270 - 1 - 2 - 5 - 14)
        replicates = np.sum((replicate - y.mean()) ** 2) / (15 - 1)

        assert result.sigma**2 == pytest.approx(error, rel=1e-6)
        assert (result.theta[0] * result.sigma) ** 2 == pytest.approx(
            (replicates - error) / 18, rel=1e-5
        )
        assert_allclose(result.matrices.X @ result.beta, recipe + temperature - y.mean(), rtol=1e-9)
