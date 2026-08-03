from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import allEffects, families, ggpredict, glmer, lmer, lmerControl


@pytest.fixture(scope="module")
def effect_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    groups = np.repeat([f"g{i}" for i in range(8)], 20)
    x = np.tile(np.linspace(-1.0, 1.0, 20), 8)
    treatment = np.tile(np.repeat(["control", "active"], 10), 8)
    group_effect = np.repeat(rng.normal(scale=0.4, size=8), 20)
    y = (
        2.0
        + 1.5 * x
        + 0.8 * (treatment == "active")
        + 0.5 * x * (treatment == "active")
        + group_effect
        + rng.normal(scale=0.2, size=len(x))
    )
    return pd.DataFrame({"y": y, "x": x, "treatment": treatment, "group": groups})


@pytest.fixture(scope="module")
def lmm_result(effect_data: pd.DataFrame):
    return lmer("y ~ x * treatment + (1 | group)", effect_data)


def test_ggpredict_numeric_grid_matches_fixed_effects(lmm_result) -> None:
    result = ggpredict(lmm_result, "x", at={"x": [-1.0, 0.0, 1.0]})

    beta = lmm_result.fixef()
    expected = beta["(Intercept)"] + beta["x"] * result["x"].to_numpy()
    np.testing.assert_allclose(result["predicted"], expected)
    assert result.columns.tolist() == [
        "x",
        "predicted",
        "std.error",
        "conf.low",
        "conf.high",
    ]
    assert np.all(result["conf.low"] <= result["predicted"])
    assert np.all(result["predicted"] <= result["conf.high"])


def test_ggpredict_builds_interaction_grid_in_one_result(lmm_result) -> None:
    result = ggpredict(
        lmm_result,
        ["x", "treatment"],
        at={"x": [-1.0, 1.0]},
    )

    assert len(result) == 4
    assert set(result["treatment"].astype(str)) == {"active", "control"}
    assert result.groupby("treatment", observed=True)["predicted"].nunique().eq(2).all()


def test_ggpredict_uses_requested_conditioning_value(lmm_result) -> None:
    control = ggpredict(lmm_result, "x", at={"x": [0.5], "treatment": "control"})
    active = ggpredict(lmm_result, "x", at={"x": [0.5], "treatment": "active"})

    assert active.loc[0, "predicted"] != control.loc[0, "predicted"]


def test_ggpredict_glmm_response_intervals_transform_link_scale() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat([f"g{i}" for i in range(10)], 20)
    x = np.tile(np.linspace(-1.5, 1.5, 20), 10)
    probabilities = 1.0 / (1.0 + np.exp(-(-0.4 + 1.2 * x)))
    y = rng.binomial(1, probabilities)
    data = pd.DataFrame({"y": y, "x": x, "group": groups})
    model = glmer("y ~ x + (1 | group)", data, family=families.Binomial())

    response = ggpredict(model, "x", at={"x": [-1.0, 0.0, 1.0]})
    link = ggpredict(model, "x", at={"x": [-1.0, 0.0, 1.0]}, type="link")

    np.testing.assert_allclose(response["predicted"], model.family.link.inverse(link["predicted"]))
    np.testing.assert_allclose(response["conf.low"], model.family.link.inverse(link["conf.low"]))
    np.testing.assert_allclose(response["conf.high"], model.family.link.inverse(link["conf.high"]))
    assert response["predicted"].between(0.0, 1.0).all()


def test_all_effects_returns_every_fixed_variable(lmm_result) -> None:
    effects = allEffects(lmm_result, n_points=5)

    assert list(effects) == ["x", "treatment"]
    assert len(effects["x"]) == 5
    assert len(effects["treatment"]) == 2


def test_ggpredict_honors_nondefault_contrasts(effect_data: pd.DataFrame) -> None:
    model = lmer(
        "y ~ treatment + (1 | group)",
        effect_data,
        contrasts={"treatment": "sum"},
        control=lmerControl(check_singular=False),
    )
    result = ggpredict(model, "treatment", contrasts={"treatment": "sum"})

    beta = model.beta
    expected = np.where(
        result["treatment"].astype(str) == "active",
        beta[0] + beta[1],
        beta[0] - beta[1],
    )
    np.testing.assert_allclose(
        result["predicted"],
        expected,
    )


def test_ggpredict_accepts_polars_backed_model(effect_data: pd.DataFrame) -> None:
    polars = pytest.importorskip("polars")
    model = lmer(
        "y ~ x + (1 | group)",
        polars.from_pandas(effect_data[["y", "x", "group"]]),
        control=lmerControl(check_singular=False),
    )

    result = ggpredict(model, "x", at={"x": [-1.0, 0.0, 1.0]})

    assert isinstance(result, pd.DataFrame)
    assert result["x"].tolist() == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"terms": "missing"}, "Unknown fixed-effect"),
        ({"terms": []}, "at least one"),
        ({"terms": "x", "type": "invalid"}, "response.*link"),
        ({"terms": "x", "level": 1.0}, "between 0 and 1"),
        ({"terms": "x", "n_points": 1}, "at least 2"),
        ({"terms": "x", "at": {"x": []}}, "cannot be empty"),
        ({"terms": "x", "at": {"x": [np.inf]}}, "must be finite"),
        ({"terms": "treatment", "at": {"treatment": "missing"}}, "Unknown level"),
    ],
)
def test_ggpredict_validates_inputs(lmm_result, kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ggpredict(lmm_result, **kwargs)
