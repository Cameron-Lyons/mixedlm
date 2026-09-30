import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, emmeans, ggpredict, lmer
from mixedlm._rust import (
    SparseCholeskySymbolic,
    simulate_re_batch,
    sparse_cholesky_logdet,
    sparse_cholesky_solve,
)
from mixedlm.estimation.reml import LMMOptimizer, profiled_deviance
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices
from mixedlm.utils.variance import cov2sdcor, sdcor2cov
from scipy import sparse


@pytest.fixture
def sleepstudy_data():
    np.random.seed(42)
    n_subjects = 18
    n_days = 10
    n_obs = n_subjects * n_days

    subject_ids = np.repeat(np.arange(n_subjects), n_days)
    days = np.tile(np.arange(n_days), n_subjects)

    intercept = 250
    slope = 10
    subject_intercepts = np.random.normal(0, 25, n_subjects)
    subject_slopes = np.random.normal(0, 6, n_subjects)
    noise = np.random.normal(0, 30, n_obs)

    reaction = (
        intercept
        + slope * days
        + subject_intercepts[subject_ids]
        + subject_slopes[subject_ids] * days
        + noise
    )

    return pd.DataFrame(
        {"Reaction": reaction, "Days": days, "Subject": [f"S{i}" for i in subject_ids]}
    )


@pytest.fixture
def large_data():
    np.random.seed(42)
    n_groups = 100
    obs_per_group = 50
    n_obs = n_groups * obs_per_group

    group_ids = np.repeat(np.arange(n_groups), obs_per_group)
    x = np.random.normal(0, 1, n_obs)
    group_effects = np.random.normal(0, 1, n_groups)
    y = 5 + 2 * x + group_effects[group_ids] + np.random.normal(0, 0.5, n_obs)

    return pd.DataFrame({"y": y, "x": x, "group": [f"g{i}" for i in group_ids]})


@pytest.fixture
def large_crossed_sparse_data():
    rng = np.random.default_rng(123)
    n_obs = 5_000
    n_group1 = 500
    n_group2 = 400

    group1_ids = np.arange(n_obs) % n_group1
    group2_ids = (np.arange(n_obs) * 11) % n_group2
    x = rng.normal(size=n_obs)
    group1_effects = rng.normal(scale=1.0, size=n_group1)
    group2_effects = rng.normal(scale=0.6, size=n_group2)
    y = 2.0 + 0.5 * x + group1_effects[group1_ids] + group2_effects[group2_ids]
    y += rng.normal(scale=0.25, size=n_obs)

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "group1": [f"g1_{i}" for i in group1_ids],
            "group2": [f"g2_{i}" for i in group2_ids],
        }
    )


@pytest.fixture
def covariance_data():
    rng = np.random.default_rng(321)
    q = 400
    sd = rng.uniform(0.5, 2.0, size=q)
    corr = np.full((q, q), 0.01)
    np.fill_diagonal(corr, 1.0)
    cov = corr * sd[:, np.newaxis] * sd[np.newaxis, :]
    return sd, corr, cov


@pytest.fixture
def large_nested_sparse_data():
    n_obs = 50_000
    row_ids = np.arange(n_obs)
    return pd.DataFrame(
        {
            "y": row_ids.astype(float),
            "district": row_ids % 100,
            "school": row_ids % 1_000,
        }
    )


@pytest.fixture
def sparse_spd_system():
    size = 2_000
    offdiag = np.full(size - 1, -1.0)
    matrix = sparse.diags(
        (offdiag, np.full(size, 4.0), offdiag),
        offsets=(-1, 0, 1),
        format="csc",
    )
    rng = np.random.default_rng(42)
    rhs = rng.standard_normal((size, 16))
    return (
        matrix.data.astype(np.float64),
        matrix.indices.astype(np.int64),
        matrix.indptr.astype(np.int64),
        matrix.shape,
        rhs,
    )


@pytest.mark.benchmark(group="lmer")
def test_benchmark_lmer_simple(benchmark, sleepstudy_data):
    def fit_model():
        return lmer("Reaction ~ Days + (1 | Subject)", data=sleepstudy_data)

    benchmark(fit_model)


@pytest.mark.benchmark(group="contrast-encoding")
@pytest.mark.parametrize("with_unknown", [False, True])
@pytest.mark.parametrize("n_obs", [128, 100_000])
def test_benchmark_categorical_contrast_encoding(benchmark, with_unknown, n_obs):
    from mixedlm.utils.contrasts import apply_contrasts_array

    categories = [f"c{i}" for i in range(40)]
    codes = np.arange(n_obs) % len(categories)
    values = np.asarray(categories, dtype=object)[codes]
    contrasts = np.arange(40 * 39, dtype=np.float64).reshape(40, 39)
    expected = contrasts[codes]
    if with_unknown:
        values[::31] = None
        expected[::31] = np.nan

    columns, names = benchmark(apply_contrasts_array, values, "factor", contrasts, categories)

    assert len(names) == contrasts.shape[1]
    np.testing.assert_array_equal(np.column_stack(columns), expected)


@pytest.mark.benchmark(group="em-reml")
def test_benchmark_em_reml_iterations(benchmark, large_crossed_sparse_data):
    from mixedlm.estimation.em_reml import em_reml_simple

    matrices = build_model_matrices(
        parse_formula("y ~ x + (1 | group1) + (1 | group2)"), large_crossed_sparse_data
    )
    result = benchmark(em_reml_simple, matrices, max_iter=3, min_iter_converge=10)

    assert result.n_iter == 3
    assert np.isfinite(result.final_loglik)


@pytest.fixture(scope="module")
def marginal_mean_model():
    from mixedlm.models.lmer import LmerResult

    n_levels, n_nuisance = 256, 8
    data = pd.DataFrame(
        {
            "treatment": np.repeat([f"L{i}" for i in range(n_levels)], n_nuisance),
            "nuisance": np.tile([f"N{i}" for i in range(n_nuisance)], n_levels),
            "group": (np.arange(n_levels * n_nuisance) % 32).astype(str),
            "y": np.ones(n_levels * n_nuisance),
        }
    )
    formula = parse_formula("y ~ treatment + nuisance + (1 | group)")
    matrices = build_model_matrices(formula, data)
    return LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=np.linspace(-0.2, 0.5, matrices.n_fixed),
        sigma=0.7,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )


@pytest.mark.benchmark(group="marginal-means")
@pytest.mark.parametrize("kind", ["grid", "pairs"])
def test_benchmark_large_marginal_means(benchmark, marginal_mean_model, kind):
    from mixedlm.inference.emmeans import emmeans

    means = emmeans(marginal_mean_model, "treatment")
    if kind == "grid":
        actual = benchmark(emmeans, marginal_mean_model, "treatment")
        np.testing.assert_allclose(actual.result.emmean, means.result.emmean)
        np.testing.assert_allclose(actual.result.se, means.result.se)
    else:
        actual = benchmark(means.pairs, adjust="none")
        left, right = np.triu_indices(len(means.result.emmean), k=1)
        expected = means.result.emmean[left] - means.result.emmean[right]
        np.testing.assert_allclose(actual.estimate, expected, atol=1e-14)
        assert np.all(np.isfinite(actual.se))


@pytest.mark.benchmark(group="lmer")
def test_benchmark_lmer_random_slope(benchmark, sleepstudy_data):
    def fit_model():
        return lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy_data)

    benchmark(fit_model)


@pytest.mark.benchmark(group="denominator-df")
def test_benchmark_repeated_ddf_many_coefficients(benchmark):
    from mixedlm.inference.ddf import satterthwaite_df
    from mixedlm.models.control import LmerControl

    rng = np.random.default_rng(147)
    n_levels, n_groups = 128, 16
    level = np.tile(np.arange(n_levels), n_groups)
    group = np.repeat(np.arange(n_groups), n_levels)
    y = rng.normal(size=n_levels)[level] + rng.normal(size=n_groups)[group]
    y += rng.normal(size=len(level))
    data = pd.DataFrame({"y": y, "treatment": pd.Categorical(level), "group": group.astype(str)})
    model = lmer(
        "y ~ 0 + treatment + (1 | group)",
        data,
        control=LmerControl(use_rust=False, em_init=False),
    )
    expected = satterthwaite_df(model)

    actual = benchmark(satterthwaite_df, model)

    np.testing.assert_allclose(actual.df, expected.df)
    assert len(actual.df) == n_levels


@pytest.mark.benchmark(group="lmer-large")
def test_benchmark_lmer_large_data(benchmark, large_data):
    def fit_model():
        return lmer("y ~ x + (1 | group)", data=large_data)

    benchmark(fit_model)


@pytest.mark.benchmark(group="denominator-df")
def test_benchmark_satterthwaite_df(benchmark, large_data):
    from mixedlm.inference.ddf import clear_vcov_grad_cache, satterthwaite_df

    model = lmer("y ~ x + (x | group)", data=large_data)
    model.vcov()

    def compute_df():
        clear_vcov_grad_cache()
        return satterthwaite_df(model)

    result = benchmark(compute_df)

    assert result.df.shape == (2,)
    assert np.all((result.df >= 1) & (result.df <= len(large_data) - 2))


@pytest.mark.benchmark(group="lmm-crossproducts")
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_benchmark_python_lmm_objective(benchmark, large_data, cached):
    matrices = build_model_matrices(parse_formula("y ~ x + (1 | group)"), large_data)
    theta = np.array([1.0])
    optimizer = LMMOptimizer(matrices, use_rust=False)
    expected = profiled_deviance(theta, matrices)
    optimizer.objective(theta)

    if cached:
        result = benchmark(optimizer.objective, theta)
    else:
        result = benchmark(profiled_deviance, theta, matrices)

    assert result == pytest.approx(expected, abs=1e-10)


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_crossed_sparse_design_build(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")

    def build_design():
        return build_model_matrices(formula, large_crossed_sparse_data)

    matrices = benchmark(build_design)
    assert matrices.Z.nnz == 2 * len(large_crossed_sparse_data)


@pytest.mark.benchmark(group="prediction-uncertainty")
def test_benchmark_repeated_prediction_uncertainty(benchmark, large_crossed_sparse_data):
    from mixedlm.models.lmer import LmerResult

    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, large_crossed_sparse_data)
    result = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([1.0, 0.6]),
        beta=np.array([2.0, 0.5]),
        sigma=0.25,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )
    newdata = large_crossed_sparse_data.iloc[:20]
    expected = result.predict(newdata, se_fit=True)

    actual = benchmark(result.predict, newdata, se_fit=True)

    np.testing.assert_allclose(actual.se_fit, expected.se_fit)


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_nested_sparse_design_build(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1/group2)")

    def build_design():
        return build_model_matrices(formula, large_crossed_sparse_data)

    matrices = benchmark(build_design)
    assert matrices.Z.nnz == len(large_crossed_sparse_data)


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_crossed_sparse_adaptive_start(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, large_crossed_sparse_data)
    optimizer = LMMOptimizer(matrices, use_rust=False)

    theta = benchmark(optimizer.get_start_theta)
    assert theta.shape == (2,)


@pytest.mark.benchmark(group="sparse-likelihood")
def test_benchmark_sparse_python_likelihood(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, large_crossed_sparse_data)
    optimizer = LMMOptimizer(matrices, use_rust=False)
    theta = np.array([0.8, 0.5])
    expected = optimizer.objective(theta)

    actual = benchmark(optimizer.objective, theta)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected)


@pytest.mark.benchmark(group="covariance-conversion")
def test_benchmark_sdcor2cov(benchmark, covariance_data):
    sd, corr, expected = covariance_data

    cov = benchmark(sdcor2cov, sd, corr)

    np.testing.assert_allclose(cov, expected)


@pytest.mark.benchmark(group="categorical-prediction")
def test_benchmark_categorical_prediction(benchmark):
    from mixedlm.models.lmer import LmerResult

    n_obs = 100_000
    data = pd.DataFrame(
        {
            "y": np.ones(n_obs),
            "category": np.take([f"c{i}" for i in range(20)], np.arange(n_obs) % 20),
            "group": np.arange(n_obs) % 100,
        }
    )
    formula = parse_formula("y ~ category + (1 | group)")
    matrices = build_model_matrices(formula, data)
    result = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8]),
        beta=np.linspace(0.1, 0.3, matrices.n_fixed),
        sigma=0.6,
        u=np.linspace(-0.5, 0.5, matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )

    predicted = benchmark(result.predict, data)

    np.testing.assert_allclose(predicted, matrices.X @ result.beta + matrices.Z @ result.u)


@pytest.mark.benchmark(group="covariance-factor")
@pytest.mark.parametrize("n_levels", [100, 10_000])
def test_benchmark_covariance_factor(benchmark, n_levels):
    from mixedlm.estimation.reml import _build_lambda
    from mixedlm.matrices.design import RandomEffectStructure

    structure = RandomEffectStructure(
        grouping_factor="group",
        term_names=["intercept", "x", "z"],
        n_levels=n_levels,
        n_terms=3,
        correlated=True,
        level_map={},
    )
    theta = np.array([1.0, -0.2, 0.7, 0.0, 0.3, 0.4])

    factor = benchmark(_build_lambda, theta, [structure])

    assert factor.shape == (3 * n_levels, 3 * n_levels)
    assert factor.nnz == 5 * n_levels


@pytest.mark.benchmark(group="covariance-conversion")
def test_benchmark_cov2sdcor(benchmark, covariance_data):
    expected_sd, expected_corr, cov = covariance_data

    sd, corr = benchmark(cov2sdcor, cov)

    np.testing.assert_allclose(sd, expected_sd)
    np.testing.assert_allclose(corr, expected_corr)


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_district_school_sparse_design_build(benchmark, large_nested_sparse_data):
    formula = parse_formula("y ~ 1 + (1 | district/school)")

    matrices = benchmark(build_model_matrices, formula, large_nested_sparse_data)
    assert matrices.Z.nnz == len(large_nested_sparse_data)


@pytest.mark.benchmark(group="conditional-variance")
def test_benchmark_glmm_conditional_variance(benchmark):
    from dataclasses import replace

    from mixedlm import condVar
    from mixedlm.families import Poisson
    from mixedlm.models.glmer import GlmerResult

    n_groups = 1_000
    n_obs = 5 * n_groups
    data = pd.DataFrame({"y": np.ones(n_obs), "group": np.arange(n_obs) % n_groups})
    formula = parse_formula("y ~ 1 + (1 | group)")
    matrices = build_model_matrices(formula, data)
    result = GlmerResult(
        formula=formula,
        matrices=matrices,
        family=Poisson(),
        theta=np.array([0.8]),
        beta=np.array([0.3]),
        u=np.zeros(n_groups),
        deviance=0.0,
        converged=True,
        n_iter=0,
        nAGQ=1,
    )

    def compute_condvar():
        return condVar(replace(result))

    actual = benchmark(compute_condvar)

    expected = 0.8**2 / (1.0 + 5 * np.exp(0.3) * 0.8**2)
    np.testing.assert_allclose(actual["group"]["(Intercept)"], expected)


@pytest.mark.benchmark(group="rust-sparse-cholesky")
def test_benchmark_sparse_cholesky_solve(benchmark, sparse_spd_system):
    data, indices, indptr, shape, rhs = sparse_spd_system
    result = benchmark(sparse_cholesky_solve, data, indices, indptr, shape, rhs)
    assert np.asarray(result).shape == rhs.shape


@pytest.mark.benchmark(group="rust-sparse-cholesky")
def test_benchmark_sparse_cholesky_logdet(benchmark, sparse_spd_system):
    data, indices, indptr, shape, _rhs = sparse_spd_system
    result = benchmark(sparse_cholesky_logdet, data, indices, indptr, shape)
    assert np.isfinite(result)


@pytest.mark.benchmark(group="rust-sparse-symbolic-cache")
def test_benchmark_sparse_symbolic_refactor(benchmark, sparse_spd_system):
    data, indices, indptr, shape, rhs = sparse_spd_system
    symbolic = SparseCholeskySymbolic(indices, indptr, shape[0])

    numeric = benchmark(symbolic.factor, data)
    result = numeric.solve(rhs[:, :1])
    assert np.asarray(result).shape == (shape[0], 1)


@pytest.mark.benchmark(group="rust-random-effect-simulation")
def test_benchmark_random_effect_simulation(benchmark):
    result = benchmark(
        simulate_re_batch,
        np.array([1.0, 0.25, 0.75]),
        1.0,
        [200_000],
        [2],
        [True],
        1,
        seed=42,
    )
    assert np.asarray(result).shape == (1, 400_000)


@pytest.mark.benchmark(group="multiplicity-adjustment")
@pytest.mark.parametrize("method", ["holm", "fdr"])
def test_benchmark_large_pvalue_adjustment(benchmark, method):
    from mixedlm.inference.emmeans import _adjust_pvalues

    p = np.random.default_rng(120).uniform(size=100_000) ** 8
    adjusted = benchmark(_adjust_pvalues, p, method, len(p), 100.0)

    assert adjusted.shape == p.shape
    assert np.all(adjusted >= p - 1e-15)
    assert np.all(adjusted <= 1.0)


@pytest.mark.benchmark(group="emmeans-grouped")
def test_benchmark_grouped_emmeans(benchmark):
    from mixedlm.inference.emmeans import emmeans
    from mixedlm.models.lmer import LmerResult

    data = pd.DataFrame(
        {
            "y": np.ones(1024),
            "treatment": np.tile([f"L{i:02}" for i in range(16)], 64),
            "x": np.repeat(np.tile([-1.0, 1.0], 32), 16),
            "group": np.repeat(np.arange(32), 32),
        }
    )
    formula = parse_formula("y ~ treatment * x + (1 | group)")
    matrices = build_model_matrices(formula, data)
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.6]),
        beta=np.linspace(-0.2, 0.3, matrices.n_fixed),
        sigma=1.0,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )
    model.vcov()

    def compare():
        return emmeans(model, "treatment", by="x", at={"x": np.linspace(-1, 1, 16)}).pairs(
            adjust="none"
        )

    result = benchmark(compare)

    assert result.estimate.shape == (16 * 120,)
    assert result.grid.x.nunique() == 16
    assert np.all(np.isfinite(result.se))


@pytest.mark.benchmark(group="leverage")
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_benchmark_large_leverage(benchmark, large_crossed_sparse_data, kind):
    from mixedlm.families import Poisson
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, large_crossed_sparse_data)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([1.0, 0.6]),
        beta=np.array([0.2, 0.1]),
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    result = (
        LmerResult(**common, sigma=0.25, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=Poisson(), nAGQ=1)
    )
    result.vcov()

    def compute_leverage():
        result.__dict__.pop("_hat_values", None)
        return result.hatvalues()

    values = benchmark(compute_leverage)

    assert values.shape == (len(large_crossed_sparse_data),)
    assert np.all((values >= 0) & (values < 1))


@pytest.mark.benchmark(group="result-projection")
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_benchmark_large_result_covariance(benchmark, kind):
    from dataclasses import replace

    from mixedlm.families import Poisson
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    n_groups = 2_048
    n_obs = 4 * n_groups
    data = pd.DataFrame({"y": np.ones(n_obs), "group": np.arange(n_obs) % n_groups})
    formula = parse_formula("y ~ 1 + (1 | group)")
    matrices = build_model_matrices(formula, data)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8]),
        beta=np.array([0.3]),
        u=np.zeros(n_groups),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    result = (
        LmerResult(**common, sigma=0.7, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=Poisson(), nAGQ=1)
    )
    actual = benchmark(lambda: replace(result).vcov())

    weight = 1.0 if kind == "lmm" else np.exp(0.3)
    scale = 0.7**2 if kind == "lmm" else 1.0
    expected = scale * (1.0 + 4 * weight * 0.8**2) / (n_obs * weight)
    np.testing.assert_allclose(actual, [[expected]])


@pytest.mark.benchmark(group="result-profile")
@pytest.mark.parametrize("dimension", [1, 2])
def test_benchmark_large_fixed_effect_profile(benchmark, dimension):
    from dataclasses import replace

    from mixedlm.inference.profile import profile_lmer, slice2D
    from mixedlm.models.lmer import LmerResult

    n_groups = 1_024
    n_obs = 4 * n_groups
    rng = np.random.default_rng(301)
    x = np.tile([-1.0, -0.3, 0.2, 1.2], n_groups)
    data = pd.DataFrame(
        {
            "y": 1.0 + 0.4 * x + rng.normal(size=n_obs),
            "x": x,
            "group": np.repeat(np.arange(n_groups), 4),
        }
    )
    formula = parse_formula("y ~ x + (1 | group)")
    matrices = build_model_matrices(formula, data)
    result = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8]),
        beta=np.array([1.0, 0.4]),
        sigma=0.7,
        u=np.zeros(n_groups),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )

    def compute_profile():
        fresh = replace(result)
        if dimension == 1:
            return profile_lmer(fresh, which="x", n_points=9)["x"]
        return slice2D(fresh, "(Intercept)", "x", n_points=9)

    actual = benchmark(compute_profile)

    assert actual.zeta.shape == ((9,) if dimension == 1 else (9, 9))
    assert np.all(np.isfinite(actual.zeta))


@pytest.mark.benchmark(group="ddf-information")
@pytest.mark.parametrize("cached", [False, True])
def test_benchmark_large_ddf_information(benchmark, large_crossed_sparse_data, cached):
    from mixedlm.inference.ddf import _weighted_crossproducts, _xt_vinv_x_from_theta
    from mixedlm.models.lmer import LmerResult

    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(
        formula,
        large_crossed_sparse_data,
        weights=np.linspace(0.4, 2.0, len(large_crossed_sparse_data)),
    )
    result = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.8, 0.5]),
        beta=np.array([0.2, 0.1]),
        sigma=0.7,
        u=np.zeros(matrices.n_random),
        deviance=0.0,
        REML=True,
        converged=True,
        n_iter=0,
    )
    expected = np.linalg.inv(result.vcov())
    crossproducts = _weighted_crossproducts(result) if cached else None

    actual = benchmark(_xt_vinv_x_from_theta, result, result.theta, crossproducts)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.benchmark(group="adjusted-effects")
def test_benchmark_adjusted_effect_grids(benchmark):
    from mixedlm.models.lmer import LmerResult

    rng = np.random.default_rng(642)
    n_rows = 20_000
    data = pd.DataFrame({f"x{i}": rng.normal(size=n_rows) for i in range(8)})
    data["treatment"] = pd.Categorical(
        np.resize(["C", "A", "B"], n_rows), categories=["C", "A", "B"]
    )
    data["g"] = np.arange(n_rows) % 8
    data["y"] = rng.normal(size=n_rows)
    formula = parse_formula(
        "y ~ treatment + " + " + ".join(f"x{i}" for i in range(8)) + " + (1 | g)"
    )
    matrices = build_model_matrices(formula, data, contrasts={"treatment": "sum"})
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.6]),
        beta=np.linspace(-0.4, 0.5, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        sigma=1.0,
        REML=True,
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model.vcov()

    result = benchmark(allEffects, model, n_points=5, contrasts=model.matrices.contrasts)

    assert list(result) == ["treatment", *(f"x{i}" for i in range(8))]
    for name, frame in result.items():
        grid = frame[[name]].copy()
        for i in range(8):
            if f"x{i}" != name:
                grid[f"x{i}"] = data[f"x{i}"].mean()
        if name != "treatment":
            grid["treatment"] = "C"
        expected = model.predict(grid, re_form="~0")
        np.testing.assert_allclose(frame.predicted, expected, atol=1e-12)


@pytest.mark.benchmark(group="polars-effect-grid")
def test_benchmark_polars_effect_grid(benchmark):
    from mixedlm.models.lmer import LmerResult

    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(983)
    n_rows = 50_000
    columns = {f"x{i}": rng.normal(size=n_rows) for i in range(6)}
    data = pl.DataFrame(
        {
            **columns,
            "treatment": np.resize(["C", "A", "B"], n_rows),
            "g": np.arange(n_rows) % 8,
            "y": rng.normal(size=n_rows),
        }
    ).with_columns(pl.col("treatment").cast(pl.Enum(["C", "A", "B"])))
    formula = parse_formula("y ~ treatment + " + " + ".join(columns) + " + (1 | g)")
    matrices = build_model_matrices(formula, data, contrasts={"treatment": "sum"})
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.6]),
        beta=np.linspace(-0.4, 0.5, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        sigma=1.0,
        REML=True,
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model.vcov()

    result = benchmark(ggpredict, model, "treatment", contrasts=model.matrices.contrasts)

    grid = pd.DataFrame({"treatment": ["C", "A", "B"]})
    for name, values in columns.items():
        grid[name] = values.mean()
    np.testing.assert_allclose(result.predicted, model.predict(grid, re_form="~0"), atol=1e-12)


@pytest.mark.benchmark(group="marginal-reference-grid")
def test_benchmark_streamed_marginal_reference_grid(benchmark):
    from mixedlm.models.lmer import LmerResult

    rng = np.random.default_rng(713)
    n_rows = 500
    data = pd.DataFrame({f"f{i}": rng.integers(0, 7, n_rows).astype(str) for i in range(6)})
    for name in data:
        data[name] = pd.Categorical(data[name], categories=list(map(str, range(7))))
    data["treatment"] = pd.Categorical(np.resize(["C", "A", "B"], n_rows))
    data["g"] = np.arange(n_rows) % 8
    data["y"] = rng.normal(size=n_rows)
    formula = parse_formula(
        "y ~ treatment + " + " + ".join(f"f{i}" for i in range(6)) + " + (1 | g)"
    )
    matrices = build_model_matrices(formula, data)
    model = LmerResult(
        formula=formula,
        matrices=matrices,
        theta=np.array([0.5]),
        beta=np.linspace(-0.3, 0.4, matrices.n_fixed),
        u=np.zeros(matrices.n_random),
        sigma=0.8,
        REML=True,
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model.vcov()

    result = benchmark(emmeans, model, "treatment")

    # Cycling all nuisance factors together gives the same equal-weight mean
    # for this additive model without constructing the full 3 * 7**6 grid.
    reference = pd.DataFrame({f"f{i}": list(map(str, range(7))) * 3 for i in range(6)})
    reference["treatment"] = np.repeat(["A", "B", "C"], 7)
    expected = model.predict(reference, re_form="~0").reshape(3, 7).mean(axis=1)
    np.testing.assert_allclose(result.result.emmean, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.benchmark(group="contrast-confidence")
@pytest.mark.parametrize("adjust", ["none", "tukey"])
def test_benchmark_contrast_confidence_intervals(benchmark, adjust):
    from mixedlm.inference.emmeans import ContrastResult

    n_means = 16
    n_comparisons = n_means * (n_means - 1) // 2
    result = ContrastResult(
        contrast=[f"C{i}" for i in range(n_comparisons)],
        estimate=np.linspace(-2.0, 2.0, n_comparisons),
        se=np.linspace(0.2, 0.5, n_comparisons),
        df=80.0,
        t_ratio=np.zeros(n_comparisons),
        p_value=np.ones(n_comparisons),
        adjust=adjust,
        _families=((0, n_comparisons, n_means),),
    )
    result.confint()

    intervals = benchmark(result.confint)

    assert intervals.shape == (n_comparisons, 6)
    assert np.all(intervals.lower < result.estimate)
    assert np.all(intervals.upper > result.estimate)
    np.testing.assert_allclose((intervals.lower + intervals.upper) / 2, result.estimate)


@pytest.mark.benchmark(group="custom-contrast-validation")
@pytest.mark.parametrize("kind", ["general", "pairwise"])
def test_benchmark_custom_contrast_validation(benchmark, kind):
    from mixedlm.inference.emmeans import EmmeanResult, Emmeans

    rng = np.random.default_rng(929)
    n_means, n_contrasts = 64, 2048
    coefficients = rng.normal(size=(n_means, 24))
    beta = rng.normal(size=24)
    values = coefficients @ beta
    zeros = np.zeros(n_means)
    means = Emmeans(
        EmmeanResult(
            values, zeros, 80.0, zeros, zeros, pd.DataFrame({"treatment": range(n_means)}), 0.95
        ),
        coefficients,
        np.eye(24),
        beta,
        80.0,
        ["treatment"],
        [list(range(n_means))],
    )
    if kind == "general":
        custom = rng.normal(size=(n_contrasts, n_means))
    else:
        custom = np.zeros((n_contrasts, n_means))
        left = np.arange(n_contrasts) % n_means
        custom[np.arange(n_contrasts), left] = 1.0
        custom[np.arange(n_contrasts), (left + 1) % n_means] = -1.0

    result = benchmark(means.contrast, custom, adjust="none")

    expected_coefficients = custom @ coefficients
    np.testing.assert_allclose(
        result.estimate, expected_coefficients @ beta, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result.se, np.linalg.norm(expected_coefficients, axis=1), rtol=1e-12)


@pytest.mark.benchmark(group="custom-contrast-scaling")
@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
def test_benchmark_scaled_custom_contrasts(benchmark, scale):
    from tests.test_emmeans import _synthetic_emmeans

    means = _synthetic_emmeans(n_levels=32, n_beta=16)
    coefficients = np.random.default_rng(423).normal(size=(512, 32))
    projected = coefficients @ means._L
    estimate = projected @ means._beta
    se = np.sqrt(np.einsum("ij,ij->i", projected @ means._vcov, projected))

    result = benchmark(means.contrast, coefficients * scale, adjust="none")

    np.testing.assert_allclose(result.estimate / scale, estimate, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.se / scale, se, rtol=1e-12)
    np.testing.assert_allclose(result.t_ratio, estimate / se, rtol=1e-12, atol=1e-12)


@pytest.mark.benchmark(group="adjusted-effects")
def test_benchmark_large_adjusted_effect_grid(benchmark):
    rng = np.random.default_rng(20260930)
    n = 512
    x = rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(-0.5, 0.5, n)
    group = np.repeat(np.arange(32), 16)
    treatment = np.arange(n) % 8
    y = 1 + x + z + 0.2 * x * z + treatment / 8
    y += rng.normal(scale=0.4, size=32)[group] + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame(
        {"y": y, "x": x, "z": z, "g": group, "treatment": pd.Categorical(treatment)}
    )
    model = lmer("y ~ treatment * x * z + (1 | g)", frame)
    model.vcov()

    actual = benchmark(ggpredict, model, ["x", "z", "treatment"], n_points=64)

    assert len(actual) == 64 * 64 * 8
    sample = actual.iloc[[0, len(actual) // 2, -1]]
    np.testing.assert_allclose(sample.predicted, model.predict(sample, re_form="NA"))
    assert np.isfinite(actual["std.error"]).all()
    assert np.all(actual["conf.low"] <= actual.predicted)
    assert np.all(actual["conf.high"] >= actual.predicted)


@pytest.mark.benchmark(group="formula-encoding")
@pytest.mark.parametrize("kind", ["categorical", "numeric", "small", "deep"])
def test_benchmark_repeated_formula_factors(benchmark, kind):
    from mixedlm.matrices.design import build_fixed_matrix

    n = 16 if kind == "small" else 50_000
    rng = np.random.default_rng(721)
    data = pd.DataFrame(
        {
            "a": pd.Categorical(rng.integers(0, 5, n), categories=range(5)),
            "b": pd.Categorical(rng.integers(0, 5, n), categories=range(5)),
            "c": pd.Categorical(rng.integers(0, 5, n), categories=range(5)),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    if kind == "deep":
        data = pd.DataFrame(
            {name: pd.Categorical(rng.integers(0, 3, n), categories=range(3)) for name in "abcdefg"}
        )
    rhs = {
        "categorical": "a*b*c",
        "numeric": "x*z + I(x**2)*I(z**3)",
        "small": "x",
        "deep": "a:b:c:d:e:f:g",
    }
    formula = parse_formula(f"y ~ {rhs[kind]}")
    matrix, names = benchmark(build_fixed_matrix, formula, data)
    assert matrix.shape == (n, {"categorical": 125, "numeric": 7, "small": 2, "deep": 129}[kind])
    assert len(names) == matrix.shape[1]
    np.testing.assert_array_equal(matrix[:, 0], np.ones(n))
    if kind == "numeric":
        np.testing.assert_array_equal(
            matrix[:, -1], data["x"].to_numpy() ** 2 * data["z"].to_numpy() ** 3
        )


@pytest.mark.benchmark(group="prediction-column-alignment")
@pytest.mark.parametrize("kind", ["lmm", "glmm"])
def test_benchmark_wide_numeric_prediction(benchmark, kind):
    from mixedlm import families
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    rng = np.random.default_rng(245)
    data = pd.DataFrame(
        rng.normal(scale=0.1, size=(50_000, 64)), columns=[f"x{i}" for i in range(64)]
    )
    formula = parse_formula("y ~ " + " + ".join(data.columns))
    training = data.iloc[:256].copy()
    training["y"] = 1.0
    matrices = build_model_matrices(formula, training)
    common = dict(
        formula=formula,
        matrices=matrices,
        theta=np.empty(0),
        beta=np.ones(65),
        u=np.empty(0),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    model = (
        LmerResult(**common, sigma=0.7, REML=True)
        if kind == "lmm"
        else GlmerResult(**common, family=families.Poisson(), nAGQ=1)
    )
    predicted = benchmark(model.predict, data, re_form="NA")
    expected = 1 + data.to_numpy().sum(axis=1)
    if kind == "glmm":
        expected = np.exp(expected)
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.benchmark(group="coefficient-reporting")
@pytest.mark.parametrize("method", ["Satterthwaite", "Kenward-Roger"])
def test_benchmark_wide_lmm_summary(benchmark, method):
    from mixedlm import lmerControl

    rng = np.random.default_rng(893)
    n, p = 960, 32
    data = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    rhs = " + ".join(data.columns)
    groups = np.arange(n) % 40
    data["group"] = groups
    data["y"] = (
        1
        + data["x0"] * 0.4
        + rng.normal(scale=0.8, size=40)[groups]
        + rng.normal(scale=0.7, size=n)
    )
    model = lmer(f"y ~ {rhs} + (1 | group)", data, control=lmerControl(check_singular=False))
    report = benchmark(model.summary, ddf_method=method)
    assert "Fixed effects:" in report
    assert "x31" in report
