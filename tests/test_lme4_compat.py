import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    families,
    glmer,
    glmer_nb,
    lmer,
    load_arabidopsis,
    load_cake,
    load_cbpp,
    load_dyestuff,
    load_dyestuff2,
    load_grouseticks,
    load_insteval,
    load_pastes,
    load_penicillin,
    load_sleepstudy,
    load_verbagg,
)
from mixedlm.inference.bootstrap import bootMer
from mixedlm.utils.lme4_compat import (
    ConvergenceInfo,
    DevComp,
    GHrule,
    VarCorr,
    checkConv,
    coef,
    convergence_ok,
    devcomp,
    factorize,
    fixef,
    fortify,
    getME,
    isNested,
    lmList,
    mkMerMod,
    ngrps,
    pvalues,
    ranef,
    scale_vcov,
    sigma,
    vcconv,
)


class TestAccessorFunctions:
    @pytest.fixture
    def lmer_model(self):
        sleepstudy = load_sleepstudy()
        return lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

    @pytest.fixture
    def glmer_model(self):
        np.random.seed(123)
        n = 100
        n_groups = 15
        group = np.repeat(np.arange(n_groups), n // n_groups + 1)[:n]
        x = np.random.randn(n)
        re = np.random.randn(n_groups) * 0.5
        eta = -1 + 0.5 * x + re[group]
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p)
        data = pd.DataFrame({"y": y, "x": x, "group": [f"h{g}" for g in group]})
        return glmer("y ~ x + (1 | group)", data, family=families.Binomial())

    def test_sigma_lmer(self, lmer_model) -> None:
        s = sigma(lmer_model)
        assert isinstance(s, float)
        assert s > 0
        assert s == lmer_model.sigma

    def test_sigma_glmer(self, glmer_model) -> None:
        s = sigma(glmer_model)
        assert s == 1.0

    def test_ngrps_lmer(self, lmer_model) -> None:
        groups = ngrps(lmer_model)
        assert isinstance(groups, dict)
        assert "Subject" in groups
        assert groups["Subject"] == 18

    def test_ngrps_glmer(self, glmer_model) -> None:
        groups = ngrps(glmer_model)
        assert "group" in groups
        assert groups["group"] == 15

    def test_fixef_lmer(self, lmer_model) -> None:
        fe = fixef(lmer_model)
        assert isinstance(fe, dict)
        assert "(Intercept)" in fe
        assert "Days" in fe
        assert fe == lmer_model.fixef()

    def test_ranef_lmer(self, lmer_model) -> None:
        re = ranef(lmer_model)
        assert isinstance(re, dict)
        assert "Subject" in re
        assert "(Intercept)" in re["Subject"]
        assert "Days" in re["Subject"]

    def test_ranef_condvar(self, lmer_model) -> None:
        re = ranef(lmer_model, condVar=True)
        assert isinstance(re, dict)
        assert "Subject" in re

    def test_VarCorr_lmer(self, lmer_model) -> None:
        vc = VarCorr(lmer_model)
        assert vc is not None

    def test_getME_theta(self, lmer_model) -> None:
        theta = getME(lmer_model, "theta")
        assert theta is not None
        assert len(theta) == 3

    def test_getME_beta(self, lmer_model) -> None:
        beta = getME(lmer_model, "beta")
        assert beta is not None
        assert len(beta) == 2

    def test_getME_sigma(self, lmer_model) -> None:
        s = getME(lmer_model, "sigma")
        assert s == lmer_model.sigma

    def test_getME_X(self, lmer_model) -> None:
        X = getME(lmer_model, "X")
        assert X is not None
        assert X.shape[0] == 180
        assert X.shape[1] == 2

    def test_getME_Z(self, lmer_model) -> None:
        Z = getME(lmer_model, "Z")
        assert Z is not None
        assert Z.shape[0] == 180

    def test_getME_invalid(self, lmer_model) -> None:
        with pytest.raises(ValueError, match="Unknown component name"):
            getME(lmer_model, "invalid_component")

    def test_coef_lmer(self, lmer_model) -> None:
        c = coef(lmer_model)
        assert isinstance(c, dict)
        assert "Subject" in c
        assert len(c["Subject"]["(Intercept)"]) == 18
        assert len(c["Subject"]["Days"]) == 18


class TestPvalues:
    @pytest.fixture
    def lmer_model(self):
        sleepstudy = load_sleepstudy()
        return lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

    def test_pvalues_satterthwaite(self, lmer_model) -> None:
        pvals = pvalues(lmer_model, method="Satterthwaite")
        assert isinstance(pvals, dict)
        assert "(Intercept)" in pvals
        assert "Days" in pvals
        assert all(0 <= p <= 1 for p in pvals.values())

    def test_pvalues_normal(self, lmer_model) -> None:
        pvals = pvalues(lmer_model, method="normal")
        assert isinstance(pvals, dict)
        assert "(Intercept)" in pvals
        assert "Days" in pvals
        assert all(0 <= p <= 1 for p in pvals.values())

    def test_pvalues_invalid_method(self, lmer_model) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            pvalues(lmer_model, method="invalid")


class TestLmList:
    def test_lmlist_basic(self) -> None:
        sleepstudy = load_sleepstudy()
        results = lmList("Reaction ~ Days", sleepstudy, group="Subject")

        assert isinstance(results, dict)
        assert "fits" in results
        assert "coef" in results
        assert len(results["fits"]) == 18
        for _subj, fit in results["fits"].items():
            assert "coef" in fit
            assert len(fit["coef"]) == 2

    def test_lmlist_coef_dataframe(self) -> None:
        sleepstudy = load_sleepstudy()
        results = lmList("Reaction ~ Days", sleepstudy, group="Subject")

        assert isinstance(results["coef"], pd.DataFrame)
        assert "(Intercept)" in results["coef"].columns
        assert "Days" in results["coef"].columns
        assert len(results["coef"]) == 18

    def test_lmlist_pooled(self) -> None:
        sleepstudy = load_sleepstudy()
        results = lmList("Reaction ~ Days", sleepstudy, group="Subject", pool=True)

        assert "pooled" in results
        assert "coef" in results["pooled"]

    def test_lmlist_not_pooled(self) -> None:
        sleepstudy = load_sleepstudy()
        results = lmList("Reaction ~ Days", sleepstudy, group="Subject", pool=False)

        assert "pooled" not in results


class TestConvergenceFunctions:
    @pytest.fixture
    def converged_model(self):
        sleepstudy = load_sleepstudy()
        return lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

    def test_checkConv_returns_info(self, converged_model) -> None:
        info = checkConv(converged_model)
        assert isinstance(info, ConvergenceInfo)
        assert hasattr(info, "converged")
        assert hasattr(info, "gradient_norm")
        assert hasattr(info, "messages")
        assert hasattr(info, "is_singular")

    def test_checkConv_grad_tol(self, converged_model) -> None:
        info = checkConv(converged_model, grad_tol=1e-3)
        assert isinstance(info.converged, bool)

    def test_convergence_ok_converged_model(self, converged_model) -> None:
        ok = convergence_ok(converged_model)
        assert isinstance(ok, bool)

    def test_convergence_ok_strict_tol(self, converged_model) -> None:
        ok = convergence_ok(converged_model, tol=1e-10)
        assert isinstance(ok, bool)


class TestFortify:
    def test_fortify_basic(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
        fortified = fortify(result, sleepstudy)

        assert isinstance(fortified, pd.DataFrame)
        assert ".fitted" in fortified.columns
        assert ".resid" in fortified.columns
        assert len(fortified) == len(sleepstudy)

    def test_fortify_fixed_column(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)
        fortified = fortify(result, sleepstudy)

        assert ".fixed" in fortified.columns

    def test_fortify_fitted_plus_resid(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
        fortified = fortify(result, sleepstudy)

        reconstructed = fortified[".fitted"] + fortified[".resid"]
        assert np.allclose(reconstructed, sleepstudy["Reaction"], rtol=1e-10)


class TestDevcomp:
    def test_devcomp_basic(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
        dc = devcomp(result)

        assert isinstance(dc, DevComp)
        assert hasattr(dc, "cmp")
        assert hasattr(dc, "dims")

    def test_devcomp_cmp_keys(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
        dc = devcomp(result)

        assert "dev" in dc.cmp
        assert "logLik" in dc.cmp
        assert "wrss" in dc.cmp

    def test_devcomp_dims_keys(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)
        dc = devcomp(result)

        assert "n" in dc.dims
        assert "p" in dc.dims
        assert "q" in dc.dims
        assert dc.dims["n"] == 180
        assert dc.dims["p"] == 2


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


class TestVcconv:
    def test_vcconv_to_sdcorr(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        converted = vcconv(
            result.theta,
            result.matrices.random_structures,
            sigma=result.sigma,
            to="sdcorr",
        )

        assert isinstance(converted, dict)
        assert "Subject" in converted
        assert "sd" in converted["Subject"]
        assert "corr" in converted["Subject"]

    def test_vcconv_to_varcov(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        converted = vcconv(
            result.theta,
            result.matrices.random_structures,
            sigma=result.sigma,
            to="varcov",
        )

        assert isinstance(converted, dict)
        assert "Subject" in converted
        assert "var" in converted["Subject"]
        assert "cov" in converted["Subject"]

    def test_vcconv_to_theta(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy)

        converted = vcconv(
            result.theta,
            result.matrices.random_structures,
            sigma=result.sigma,
            to="theta",
        )

        assert isinstance(converted, dict)
        assert "Subject" in converted
        assert "theta" in converted["Subject"]


class TestGHrule:
    def test_ghrule_as_matrix(self) -> None:
        rule = GHrule(5, asMatrix=True)
        assert isinstance(rule, np.ndarray)
        assert rule.shape == (5, 2)

    def test_ghrule_as_dict(self) -> None:
        rule = GHrule(5, asMatrix=False)
        assert isinstance(rule, dict)
        assert "nodes" in rule
        assert "weights" in rule
        assert len(rule["nodes"]) == 5
        assert len(rule["weights"]) == 5

    def test_ghrule_weights_positive(self) -> None:
        rule = GHrule(10, asMatrix=False)
        assert all(w > 0 for w in rule["weights"])

    def test_ghrule_nodes_symmetric(self) -> None:
        rule = GHrule(7, asMatrix=False)
        assert np.isclose(rule["nodes"][3], 0.0, atol=1e-10)
        assert np.allclose(rule["nodes"][:3], -rule["nodes"][6:3:-1])


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
    def test_glmer_nb_basic(self) -> None:
        np.random.seed(42)
        n = 100
        n_groups = 10
        group = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.random.randn(n)
        re = np.random.randn(n_groups) * 0.5
        mu = np.exp(1 + 0.5 * x + re[group])
        y = np.random.poisson(mu)

        data = pd.DataFrame({"y": y, "x": x, "group": [f"g{g}" for g in group]})

        result = glmer_nb("y ~ x + (1 | group)", data)
        assert result is not None
        assert result.converged


class TestBootMer:
    def test_bootmer_basic(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

        boot = bootMer(result, nsim=10, seed=42)

        assert boot is not None
        assert boot.n_boot == 10
        assert boot.beta_samples.shape == (10, 2)

    def test_bootmer_theta_samples(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

        boot = bootMer(result, nsim=10, seed=42)

        assert boot.theta_samples is not None
        assert boot.theta_samples.shape[0] == 10

    def test_bootmer_parametric(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

        boot = bootMer(result, nsim=5, seed=42, bootstrap_type="parametric")
        assert boot.n_boot == 5

    def test_bootmer_ci(self) -> None:
        sleepstudy = load_sleepstudy()
        result = lmer("Reaction ~ Days + (1 | Subject)", sleepstudy)

        boot = bootMer(result, nsim=20, seed=42)
        ci = boot.ci(level=0.95)

        assert "(Intercept)" in ci
        assert "Days" in ci
        assert ci["(Intercept)"][0] < ci["(Intercept)"][1]


class TestDatasets:
    def test_load_sleepstudy(self) -> None:
        data = load_sleepstudy()
        assert isinstance(data, pd.DataFrame)
        assert "Reaction" in data.columns
        assert "Days" in data.columns
        assert "Subject" in data.columns
        assert len(data) == 180

    def test_load_cbpp(self) -> None:
        data = load_cbpp()
        assert isinstance(data, pd.DataFrame)
        assert "incidence" in data.columns
        assert "size" in data.columns
        assert "herd" in data.columns

    def test_load_dyestuff(self) -> None:
        data = load_dyestuff()
        assert isinstance(data, pd.DataFrame)
        assert "Yield" in data.columns
        assert "Batch" in data.columns
        assert len(data) == 30
        assert data["Batch"].nunique() == 6

    def test_load_dyestuff2(self) -> None:
        data = load_dyestuff2()
        assert isinstance(data, pd.DataFrame)
        assert "Yield" in data.columns
        assert "Batch" in data.columns
        assert len(data) == 30

    def test_load_penicillin(self) -> None:
        data = load_penicillin()
        assert isinstance(data, pd.DataFrame)
        assert "diameter" in data.columns
        assert "plate" in data.columns
        assert "sample" in data.columns
        assert len(data) == 144

    def test_load_cake(self) -> None:
        data = load_cake()
        assert isinstance(data, pd.DataFrame)
        assert "angle" in data.columns
        assert "recipe" in data.columns
        assert "replicate" in data.columns
        assert "temperature" in data.columns
        assert len(data) == 270

    def test_load_pastes(self) -> None:
        data = load_pastes()
        assert isinstance(data, pd.DataFrame)
        assert "strength" in data.columns
        assert "batch" in data.columns
        assert "cask" in data.columns
        assert len(data) == 60

    def test_load_insteval(self) -> None:
        data = load_insteval()
        assert isinstance(data, pd.DataFrame)
        assert "y" in data.columns
        assert "s" in data.columns
        assert "d" in data.columns
        assert len(data) >= 100

    def test_load_arabidopsis(self) -> None:
        data = load_arabidopsis()
        assert isinstance(data, pd.DataFrame)
        assert "total_fruits" in data.columns
        assert "gen" in data.columns
        assert len(data) > 0

    def test_load_grouseticks(self) -> None:
        data = load_grouseticks()
        assert isinstance(data, pd.DataFrame)
        assert "cTICKS" in data.columns
        assert "BROOD" in data.columns
        assert "LOCATION" in data.columns
        assert len(data) > 0

    def test_load_verbagg(self) -> None:
        data = load_verbagg()
        assert isinstance(data, pd.DataFrame)
        assert "r2" in data.columns
        assert "id" in data.columns
        assert "item" in data.columns
        assert len(data) > 0


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


class TestDatasetsUsability:
    def test_dyestuff_lmer(self) -> None:
        data = load_dyestuff()
        result = lmer("Yield ~ 1 + (1 | Batch)", data)
        assert result.converged

    def test_penicillin_crossed_random(self) -> None:
        data = load_penicillin()
        result = lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", data)
        assert result.converged
        assert "plate" in result.ngrps()
        assert "sample" in result.ngrps()

    def test_cake_split_plot(self) -> None:
        data = load_cake()
        result = lmer("angle ~ recipe + temperature + (1 | replicate)", data)
        assert result.converged

    def test_pastes_nested(self) -> None:
        data = load_pastes()
        result = lmer("strength ~ 1 + (1 | batch/cask)", data)
        assert result.converged
