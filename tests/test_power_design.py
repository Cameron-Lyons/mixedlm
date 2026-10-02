"""Power curves must change the simulated study, not just their x-axis labels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from mixedlm import families, glFormula, glmer, glmerControl, lmer
from mixedlm.power import extend, powerCurve, powerSim
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def pilot():
    rng = np.random.default_rng(926)
    groups = np.concatenate([np.repeat(group, 4 + group % 3) for group in range(10)])
    x = rng.normal(size=len(groups))
    offset = np.linspace(-1.2, 1.3, len(groups))
    weights = np.linspace(0.7, 1.6, len(groups))
    y = 1.0 + 0.6 * x + offset + rng.normal(size=10)[groups]
    y += rng.normal(scale=0.5, size=len(groups)) / np.sqrt(weights)
    data = pd.DataFrame({"group": groups, "x": x, "y": y})
    return lmer("y ~ x + (1 | group)", data, weights=weights, offset=offset)


@pytest.mark.parametrize("along", ["n_groups", "group"])
def test_group_curve_changes_real_refit_designs(pilot, along):
    observed = []

    def test(fitted):
        observed.append((fitted.ngrps()["group"], fitted.matrices.n_obs))
        return fitted.ngrps()["group"] >= 10

    curve = powerCurve(pilot, test=test, along=along, values=[6, 10, 14], nsim=2, seed=804)

    assert curve.powers == [0.0, 1.0, 1.0]
    assert observed == [(6, 30)] * 2 + [(10, 49)] * 2 + [(14, 68)] * 2
    assert [point.n_obs for point in curve.results] == [30, 49, 68]
    assert [point.n_groups for point in curve.results] == [6, 10, 14]
    assert [point.n_simulations for point in curve.results] == [2, 2, 2]
    assert [point.n_failed for point in curve.results] == [0, 0, 0]


def test_within_curve_resizes_each_unbalanced_group(pilot):
    sizes = []

    def test(fitted):
        counts = fitted.model_frame().groupby("group").size().to_numpy()
        sizes.append(counts)
        return bool(np.all(counts == 8))

    curve = powerCurve(pilot, test=test, along="within", values=[3, 8], nsim=2, seed=143)

    assert curve.powers == [0.0, 1.0]
    assert [point.n_obs for point in curve.results] == [30, 80]
    for counts, expected in zip(sizes, [3, 3, 8, 8], strict=True):
        assert_array_equal(counts, np.full(10, expected))


def test_resize_preserves_generating_parameters_and_row_metadata(pilot, monkeypatch):
    original_frame = pilot.model_frame()
    original_X = pilot.matrices.X.copy()
    original_theta = pilot.theta.copy()
    original_beta = pilot.beta.copy()
    snapshots = []
    simulate = type(pilot).simulate

    def capture(self, *args, **kwargs):
        snapshots.append(self)
        return simulate(self, *args, **kwargs)

    monkeypatch.setattr(type(pilot), "simulate", capture)
    powerCurve(pilot, test=lambda fitted: True, values=[6, 14], nsim=1, seed=73)

    row_maps = [np.arange(30), np.concatenate([np.arange(49), np.arange(19)])]
    for resized, rows in zip(snapshots, row_maps, strict=True):
        assert_allclose(resized.beta, original_beta, rtol=0, atol=0)
        assert_allclose(resized.theta, original_theta, rtol=0, atol=0)
        assert resized.sigma == pilot.sigma
        assert resized.REML == pilot.REML
        assert_array_equal(resized.matrices.X, original_X[rows])
        assert_array_equal(resized.matrices.weights, pilot.matrices.weights[rows])
        assert_array_equal(resized.matrices.offset, pilot.matrices.offset[rows])
        assert resized.matrices.Z.shape == (len(rows), resized.ngrps()["group"])
        assert_allclose(np.asarray(resized.matrices.Z.sum(axis=1)).ravel(), 1.0)
    pd.testing.assert_frame_equal(pilot.model_frame(), original_frame)
    assert_array_equal(pilot.matrices.X, original_X)
    assert_array_equal(pilot.beta, original_beta)
    assert_array_equal(pilot.theta, original_theta)


def test_named_coefficient_curve_sets_absolute_effects(pilot):
    curve = powerCurve(
        pilot,
        test=lambda fitted: bool(fitted.beta[1] > 0),
        along="x",
        values=[-5.0, 5.0],
        nsim=3,
        seed=42,
    )

    assert curve.powers == [0.0, 1.0]
    # Callable decisions do not claim a particular tested coefficient.
    assert all(point.effect_size is None for point in curve.results)
    named = powerCurve(pilot, test="x", along="x", values=[-5.0, 5.0], nsim=1, seed=42)
    assert [point.effect_size for point in named.results] == [-5.0, 5.0]


def test_multiplier_curve_uses_first_nonintercept_without_an_intercept(pilot):
    frame = pilot.model_frame().assign(z=np.tile([-1.0, 1.0], 25)[:49])
    model = lmer("y ~ 0 + x + z + (1 | group)", frame)
    curve = powerCurve(model, along="effect_size", values=[0.0, 2.0], nsim=1, seed=417)

    assert [point.effect_size for point in curve.results] == [0.0, 2.0 * model.beta[0]]


def test_grouped_binomial_curve_preserves_trials_and_prior_weights(monkeypatch):
    rng = np.random.default_rng(323)
    group = np.repeat(np.arange(8), 5)
    x = rng.normal(size=40)
    trials = np.tile([8, 12, 16, 20, 24], 8)
    offset = np.linspace(-0.2, 0.4, 40)
    prior = np.linspace(0.8, 1.2, 40)
    eta = -0.6 + 0.3 * x + offset + rng.normal(scale=0.5, size=8)[group]
    data = pd.DataFrame(
        {
            "successes": rng.binomial(trials, 1 / (1 + np.exp(-eta))),
            "trials": trials,
            "x": x,
            "group": group,
        }
    )
    model = glmer(
        "successes / trials ~ x + (1 | group)",
        data,
        family=families.Binomial(),
        weights=prior,
        offset=offset,
        nAGQ=0,
    )
    captured = []
    simulate = type(model).simulate

    def capture(self, *args, **kwargs):
        response = simulate(self, *args, **kwargs)
        captured.append((self, response))
        return response

    monkeypatch.setattr(type(model), "simulate", capture)
    curve = powerCurve(model, test=lambda fitted: True, values=[4, 10], nsim=1, seed=227)

    assert [point.n_obs for point in curve.results] == [20, 50]
    for (resized, response), rows in zip(
        captured, [np.arange(20), np.concatenate([np.arange(40), np.arange(10)])], strict=True
    ):
        assert_array_equal(resized.matrices.trials, trials[rows])
        assert_allclose(resized.matrices.weights, (prior * trials)[rows])
        assert_array_equal(resized.matrices.offset, offset[rows])
        assert np.all(response >= 0)
        assert np.all(response <= trials[rows])
        assert_array_equal(response, np.floor(response))
        assert resized.nAGQ == 0
    assert [point.n_simulations for point in curve.results] == [1, 1]


def test_curve_accepts_polars_model_frame(pilot):
    pl = pytest.importorskip("polars")
    data = pl.DataFrame(
        {name: pilot.model_frame()[name].to_numpy() for name in pilot.model_frame().columns}
    )
    model = lmer("y ~ x + (1 | group)", data)
    curve = powerCurve(model, test=lambda fitted: True, values=[6, 14], nsim=1, seed=535)
    assert [point.n_obs for point in curve.results] == [30, 68]
    assert [point.n_groups for point in curve.results] == [6, 14]


def test_extend_retains_polars_enum_success_and_predictor_contrast_order():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(752)
    group = np.repeat(np.arange(10), 10)
    treatment = np.tile(["A", "B"], 50)
    eta = -0.4 + 0.8 * (treatment == "A") + rng.normal(scale=0.6, size=10)[group]
    binary = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    data = pl.DataFrame(
        {
            "outcome": np.where(binary == 1, "Y", "N"),
            "treatment": treatment,
            "group": group,
        }
    ).with_columns(
        pl.col("outcome").cast(pl.Enum(["Y", "N"])),
        pl.col("treatment").cast(pl.Enum(["B", "A"])),
    )
    formula = "outcome ~ treatment + (1 | group)"
    control = glmerControl(check_conv=False, check_singular=False)
    pilot = glmer(formula, data, control=control)
    extended = extend(pilot, along="within", n=14)

    assert extended["outcome"].cat.categories.tolist() == ["Y", "N"]
    assert extended["treatment"].cat.categories.tolist() == ["B", "A"]
    encoded = glFormula(formula, extended).matrices
    assert encoded.response_levels == ("Y", "N")
    assert_array_equal(encoded.X[:, 1], extended["treatment"].eq("A").astype(float))
    numeric = extended.copy()
    numeric["outcome"] = extended["outcome"].eq("N").astype(float)
    factor_fit = glmer(formula, extended, control=control)
    numeric_fit = glmer(formula, numeric, control=control)
    assert_allclose(factor_fit.beta, numeric_fit.beta, rtol=1e-8, atol=1e-8)
    assert_allclose(factor_fit.theta, numeric_fit.theta, rtol=1e-8, atol=1e-8)
    assert_allclose(factor_fit.predict(extended), numeric_fit.predict(numeric))


def test_extend_retains_trained_polars_categorical_response_levels():
    pl = pytest.importorskip("polars")
    # A categorical response may share a pool with unused treatment labels.
    # The fitted response schema, rather than the whole pool, defines success.
    with pl.StringCache():
        data = pl.DataFrame(
            {
                "outcome": np.tile(["Y", "N", "Y", "N"], 6),
                "treatment": np.tile(["B", "A", "A", "B"], 6),
                "group": np.repeat(np.arange(6), 4),
            }
        ).with_columns(pl.col(["outcome", "treatment"]).cast(pl.Categorical))
        pilot = glmer(
            "outcome ~ treatment + (1 | group)",
            data,
            control=glmerControl(check_conv=False, check_singular=False),
        )
        result = extend(pilot, along="within", n=6)
    assert result["outcome"].cat.categories.tolist() == list(pilot.matrices.response_levels)
    assert (
        result["treatment"].cat.categories.tolist() == pilot.matrices.category_levels["treatment"]
    )


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("converged", False),
        ("beta", np.array([np.nan, 1.0])),
        ("theta", np.array([np.inf])),
        ("sigma", 0.0),
        ("deviance", np.inf),
    ],
)
def test_invalid_refits_are_excluded_before_custom_test(pilot, monkeypatch, attribute, value):
    calls = []
    invalid = replace(pilot, **{attribute: value})
    monkeypatch.setattr(pilot, "refit", lambda response: invalid)
    with pytest.warns(RuntimeWarning, match="All 2 power simulations failed"):
        result = powerSim(pilot, test=lambda fitted: calls.append(fitted) or True, nsim=2, seed=91)
    assert calls == []
    assert result.n_simulations == 0
    assert result.n_successes == 0
    assert result.n_failed == 2
    assert np.isnan(result.power)


def test_nonboolean_test_result_is_a_failed_simulation(pilot, monkeypatch):
    monkeypatch.setattr(pilot, "refit", lambda response: pilot)
    with pytest.warns(RuntimeWarning, match="boolean significance decision"):
        result = powerSim(pilot, test=lambda fitted: np.nan, nsim=2, seed=42)
    assert result.n_failed == 2
    assert result.n_simulations == 0


@pytest.mark.parametrize(
    "along,values",
    [
        ("unknown", [1]),
        ("n_groups", [True]),
        ("n_groups", [1.5]),
        ("within", [0]),
        ("effect_size", [np.inf]),
        ("x", [1.0, np.nan]),
        ("x", []),
    ],
)
def test_curve_rejects_invalid_points_before_simulation(pilot, monkeypatch, along, values):
    def should_not_simulate(*args, **kwargs):
        raise AssertionError("invalid curve must be rejected before simulating")

    monkeypatch.setattr(type(pilot), "simulate", should_not_simulate)
    with pytest.raises(ValueError):
        powerCurve(pilot, along=along, values=values, nsim=1)


def test_named_coefficient_curve_defaults_to_testing_that_coefficient(pilot):
    data = pilot.model_frame().assign(z=np.random.default_rng(934).normal(size=49))
    model = lmer("y ~ x + z + (1 | group)", data)
    curve = powerCurve(model, along="z", values=[0.0, 5.0], nsim=2, seed=524)
    assert [point.effect_size for point in curve.results] == [0.0, 5.0]
    assert curve.powers[1] == 1.0


def test_named_group_curve_reports_the_varied_crossed_factor(pilot):
    data = pilot.model_frame().assign(batch=np.tile(np.arange(4), 13)[:49])
    model = lmer("y ~ x + (1 | group) + (1 | batch)", data)
    curve = powerCurve(
        model, test=lambda fitted: True, along="batch", values=[2, 6], nsim=1, seed=935
    )
    assert [point.n_groups for point in curve.results] == [2, 6]
    assert [point.n_obs for point in curve.results] == [25, 74]


@pytest.mark.parametrize(
    "labels",
    [
        [float(2**54) + 4 * i for i in range(10)],
        [np.iinfo(np.int64).max - i for i in range(10)],
        [np.iinfo(np.uint64).max - i for i in range(10)],
        [float(np.finfo(float).max) - i * 1e307 for i in range(10)],
    ],
)
def test_group_extension_handles_extreme_numeric_labels(pilot, labels):
    from mixedlm.power import extend

    data = pilot.model_frame()
    data["group"] = data["group"].map(dict(enumerate(labels)))
    # Only the template labels change; the original model identifies the factor.
    result = extend(pilot, along="group", n=12, data=data)
    assert result["group"].nunique() == 12
    assert len(result) == len(data) + 9
    assert set(data["group"]).issubset(set(result["group"]))
    assert all(np.isfinite(value) for value in result["group"])
