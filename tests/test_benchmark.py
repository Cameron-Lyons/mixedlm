import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer
from mixedlm.estimation.reml import LMMOptimizer
from mixedlm.formula.parser import parse_formula
from mixedlm.matrices.design import build_model_matrices


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


@pytest.mark.benchmark(group="lmer")
def test_benchmark_lmer_simple(benchmark, sleepstudy_data):
    def fit_model():
        return lmer("Reaction ~ Days + (1 | Subject)", data=sleepstudy_data)

    benchmark(fit_model)


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


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_crossed_sparse_design_build(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")

    def build_design():
        return build_model_matrices(formula, large_crossed_sparse_data)

    matrices = benchmark(build_design)
    assert matrices.Z.nnz == 2 * len(large_crossed_sparse_data)


@pytest.mark.benchmark(group="sparse-design")
def test_benchmark_large_crossed_sparse_adaptive_start(benchmark, large_crossed_sparse_data):
    formula = parse_formula("y ~ x + (1 | group1) + (1 | group2)")
    matrices = build_model_matrices(formula, large_crossed_sparse_data)
    optimizer = LMMOptimizer(matrices, use_rust=False)

    theta = benchmark(optimizer.get_start_theta)
    assert theta.shape == (2,)
