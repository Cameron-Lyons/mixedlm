import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from mixedlm import lmer


@st.composite
def random_lmm_data(draw):
    n_groups = draw(st.integers(min_value=3, max_value=10))
    obs_per_group = draw(st.integers(min_value=5, max_value=20))
    n_obs = n_groups * obs_per_group

    group_ids = np.repeat(np.arange(n_groups), obs_per_group)

    x = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )

    noise = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs,),
            elements=st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        )
    )

    intercept = 5.0
    slope = 2.0
    group_effects = np.random.default_rng(42).normal(0, 1, n_groups)
    y = intercept + slope * x + group_effects[group_ids] + noise

    df = pd.DataFrame({"y": y, "x": x, "group": [f"g{i}" for i in group_ids]})

    return df


@given(data=random_lmm_data())
@settings(max_examples=20, deadline=None)
def test_lmer_fits_random_data(data):
    assume(len(data) >= 10)
    assume(data["y"].std() > 0.01)
    assume(data["x"].std() > 0.01)

    model = lmer("y ~ x + (1 | group)", data=data)

    assert model is not None
    assert hasattr(model, "fe_params")
    assert hasattr(model, "re_params")
    assert len(model.fe_params) == 2


@given(data=random_lmm_data())
@settings(max_examples=10, deadline=None)
def test_lmer_residuals_reasonable(data):
    assume(len(data) >= 10)
    assume(data["y"].std() > 0.01)
    assume(data["x"].std() > 0.01)

    model = lmer("y ~ x + (1 | group)", data=data)
    residuals = model.resid

    assert len(residuals) == len(data)
    assert np.isfinite(residuals).all()


@given(data=random_lmm_data())
@settings(max_examples=10, deadline=None)
def test_lmer_fitted_values_finite(data):
    assume(len(data) >= 10)
    assume(data["y"].std() > 0.01)
    assume(data["x"].std() > 0.01)

    model = lmer("y ~ x + (1 | group)", data=data)
    fitted = model.fittedvalues

    assert len(fitted) == len(data)
    assert np.isfinite(fitted).all()


@given(
    intercept=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    slope=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10, deadline=None)
def test_lmer_recovers_known_parameters(intercept, slope):
    assume(abs(slope) > 0.1)

    np.random.seed(42)
    n_groups = 5
    obs_per_group = 50
    n_obs = n_groups * obs_per_group

    group_ids = np.repeat(np.arange(n_groups), obs_per_group)
    x = np.random.normal(0, 1, n_obs)
    group_effects = np.random.normal(0, 0.5, n_groups)
    noise = np.random.normal(0, 0.1, n_obs)
    y = intercept + slope * x + group_effects[group_ids] + noise

    df = pd.DataFrame({"y": y, "x": x, "group": [f"g{i}" for i in group_ids]})

    model = lmer("y ~ x + (1 | group)", data=df)

    est_intercept = model.fe_params[0]
    est_slope = model.fe_params[1]

    assert abs(est_intercept - intercept) < 1.0
    assert abs(est_slope - slope) < 0.5
