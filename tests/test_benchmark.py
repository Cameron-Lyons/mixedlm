import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
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


@pytest.mark.benchmark(group="lmer")
def test_benchmark_lmer_random_slope(benchmark, sleepstudy_data):
    def fit_model():
        return lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy_data)

    benchmark(fit_model)


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
