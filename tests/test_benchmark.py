import numpy as np
import pandas as pd
import pytest
from mixedlm import lmer


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
