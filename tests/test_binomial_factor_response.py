"""Binomial factor models must agree with explicitly coded numeric responses."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from mixedlm import cross_validate, families, glFormula, glmer, glmerControl, load_verbagg
from mixedlm.inference.cross_validation import _fit_fold
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(614)
    group = np.repeat(np.arange(10), 10)
    x = rng.normal(size=len(group))
    z = rng.normal(size=len(group))
    offset = 0.2 * np.sin(x)
    weights = rng.uniform(0.8, 1.4, size=len(group))
    eta = -0.2 + 0.7 * x + offset + rng.normal(scale=0.7, size=10)[group]
    binary = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    frame = pd.DataFrame(
        {
            "outcome": np.where(binary == 1, "Y", "N"),
            "binary": binary,
            "x": x,
            "z": z,
            "group": group,
        }
    )
    return frame, weights, offset


def _factor_frame(frame, backend):
    levels = ("Y", "N") if backend.endswith("reverse") else ("N", "Y")
    if backend.startswith("pandas"):
        result = frame.copy()
        if backend != "pandas_string":
            result["outcome"] = pd.Categorical(result["outcome"], categories=levels)
        return result, levels
    pl = pytest.importorskip("polars")
    result = pl.DataFrame(frame.to_dict(orient="list"))
    if backend == "polars_categorical":
        result = result.with_columns(pl.col("outcome").cast(pl.Categorical))
        declared = result["outcome"].cat.get_categories().to_list()
        levels = tuple(level for level in declared if level in ("N", "Y"))
    elif backend != "polars_string":
        result = result.with_columns(pl.col("outcome").cast(pl.Enum(list(levels))))
    return result, levels


def _fit(formula, frame, *, weights=None, offset=None):
    return glmer(
        formula,
        frame,
        family=families.Binomial(),
        weights=weights,
        offset=offset,
        control=glmerControl(check_singular=False, check_conv=False, check_nlev_gtreq_5="ignore"),
    )


@pytest.mark.parametrize(
    "backend",
    [
        "pandas_string",
        "pandas_factor",
        "pandas_reverse",
        "polars_string",
        "polars_categorical",
        "polars_reverse",
    ],
)
def test_factor_fit_refit_update_and_cv_match_numeric_encoding(study, backend):
    frame, weights, offset = study
    factor_data, levels = _factor_frame(frame, backend)
    numeric_data = frame.copy()
    numeric_data["outcome"] = (frame["outcome"] == levels[1]).astype(float)
    factor = _fit("outcome ~ x + (1 | group)", factor_data, weights=weights, offset=offset)
    numeric = _fit("outcome ~ x + (1 | group)", numeric_data, weights=weights, offset=offset)

    assert factor.matrices.response_levels == levels
    assert numeric.matrices.response_levels is None
    assert_array_equal(factor.matrices.y, numeric_data["outcome"])
    assert_allclose(factor.beta, numeric.beta, rtol=1e-9, atol=1e-9)
    assert_allclose(factor.theta, numeric.theta, rtol=1e-9, atol=1e-9)
    assert factor.deviance == pytest.approx(numeric.deviance)
    assert_allclose(factor.predict(factor_data), numeric.predict(numeric_data))

    # Simulations and numeric refits already use 0/1, even with reversed levels.
    simulated = numeric.simulate(seed=427)
    labels = np.asarray(levels)[simulated.astype(int)]
    expected_refit = numeric.refit(simulated)
    factor_numeric_refit = factor.refit(simulated)
    factor_label_refit = factor.refit(labels)
    assert_allclose(factor_numeric_refit.beta, expected_refit.beta)
    assert_allclose(factor_label_refit.beta, expected_refit.beta)
    assert_allclose(factor_label_refit.theta, expected_refit.theta)
    assert_array_equal(factor_label_refit.matrices.y, simulated)
    assert factor_label_refit.matrices.response_levels == levels

    # Formula updates retain the success definition, including explicit reversal.
    factor_update = factor.update(". ~ . + z", data=factor_data)
    numeric_update = numeric.update(". ~ . + z", data=numeric_data)
    assert factor_update.matrices.response_levels == levels
    assert_allclose(factor_update.beta, numeric_update.beta, rtol=1e-9, atol=1e-9)
    assert_allclose(factor_update.theta, numeric_update.theta, rtol=1e-9, atol=1e-9)

    options = {
        "cv": 2,
        "group": "group",
        "random_state": 99,
        "metrics": ["mse", "deviance"],
        "fit_kwargs": {"control": glmerControl(check_conv=False, check_singular=False)},
    }
    factor_cv = cross_validate(factor, **options)
    numeric_cv = cross_validate(numeric, **options)
    assert_allclose(factor_cv.predictions, numeric_cv.predictions, rtol=1e-8, atol=1e-8)
    assert factor_cv.scores == pytest.approx(numeric_cv.scores)

    # The stored frame remains current after refit, and pickle retains the schema.
    restored = pickle.loads(pickle.dumps(factor_label_refit))
    assert restored.matrices.response_levels == levels
    restored_update = restored.update()
    assert_array_equal(restored_update.matrices.y, simulated)
    assert_allclose(restored_update.beta, expected_refit.beta, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pandas_string", "polars_string"])
@pytest.mark.parametrize("success", [False, True])
def test_one_class_subsets_retain_original_two_level_response(study, backend, success):
    frame, weights, offset = study
    factor_data, levels = _factor_frame(frame, backend)
    fitted = _fit("outcome ~ x + (1 | group)", factor_data, weights=weights, offset=offset)
    selected = np.flatnonzero(frame["binary"].to_numpy() == success)
    subset = (
        factor_data.iloc[selected]
        if backend.startswith("pandas")
        else factor_data[selected.tolist()]
    )
    trained = _fit_fold(
        fitted,
        subset,
        weights[selected],
        offset[selected],
        {"control": glmerControl(check_conv=False, check_singular=False)},
    )
    numeric_subset = frame.iloc[selected].assign(outcome=float(success))
    numeric = _fit(
        "outcome ~ x + (1 | group)",
        numeric_subset,
        weights=weights[selected],
        offset=offset[selected],
    )
    assert trained.matrices.response_levels == levels
    assert_array_equal(trained.matrices.y, np.full(len(selected), float(success)))
    assert_allclose(trained.beta, numeric.beta)
    assert_allclose(trained.predict(subset), numeric.predict(numeric_subset))
    assert trained.converged == numeric.converged
    assert trained.pirls_converged == numeric.pirls_converged

    updated = fitted.update(
        data=subset,
        weights=weights[selected],
        offset=offset[selected],
        control=glmerControl(check_conv=False, check_singular=False),
    )
    assert updated.matrices.response_levels == levels
    assert_array_equal(updated.matrices.y, trained.matrices.y)
    assert_allclose(updated.beta, numeric.beta)


@pytest.mark.parametrize("backend", ["pandas_string", "polars_string"])
def test_cross_validation_with_one_class_training_folds_matches_numeric_response(backend):
    frame = pd.DataFrame(
        {
            "outcome": np.repeat(["N", "Y"], 50),
            "group": np.repeat(np.arange(10), 10),
        }
    )
    factor_data, _ = _factor_frame(frame, backend)
    numeric_data = frame.assign(outcome=np.repeat([0.0, 1.0], 50))
    factor = _fit("outcome ~ 1 + (1 | group)", factor_data)
    numeric = _fit("outcome ~ 1 + (1 | group)", numeric_data)
    options = {
        "cv": 2,
        "shuffle": False,
        "metrics": "mse",
        "fit_kwargs": {
            "control": glmerControl(check_conv=False, check_singular=False),
        },
    }
    factor_cv = cross_validate(factor, **options)
    numeric_cv = cross_validate(numeric, **options)
    assert_allclose(factor_cv.predictions, numeric_cv.predictions)
    assert factor_cv.scores == pytest.approx(numeric_cv.scores)


def test_numeric_factor_labels_and_numeric_refits_have_distinct_meanings(study):
    frame, weights, offset = study
    factor_data = frame.copy()
    factor_data["outcome"] = pd.Categorical(frame["binary"], categories=[1, 0])
    numeric_data = frame.assign(outcome=1.0 - frame["binary"])
    factor = _fit("outcome ~ x + (1 | group)", factor_data, weights=weights, offset=offset)
    numeric = _fit("outcome ~ x + (1 | group)", numeric_data, weights=weights, offset=offset)
    assert factor.matrices.response_levels == (1, 0)
    assert_array_equal(factor.matrices.y, numeric_data["outcome"])
    assert_allclose(factor.beta, numeric.beta)

    response = numeric.simulate(seed=610)
    categorical_response = pd.Categorical(1 - response, categories=[1, 0])
    encoded_refit = factor.refit(response)
    labeled_refit = factor.refit(categorical_response)
    expected = numeric.refit(response)
    assert_array_equal(encoded_refit.matrices.y, response)
    assert_array_equal(labeled_refit.matrices.y, response)
    assert_allclose(labeled_refit.beta, expected.beta)
    assert_allclose(encoded_refit.beta, expected.beta)
    updated = labeled_refit.update()
    assert_array_equal(updated.matrices.y, response)


def test_canonical_verbagg_factor_response_matches_manual_numeric_fit():
    frame = load_verbagg()
    matrices = glFormula("r2 ~ Anger + Gender + (1 | id)", frame).matrices
    assert matrices.response_levels == ("N", "Y")
    assert matrices.n_obs == 7584
    assert_array_equal(matrices.y, frame["r2"].eq("Y").astype(float))
    subset = frame.loc[frame["id"].isin(frame["id"].unique()[:10])].copy()
    numeric = subset.assign(r2=subset["r2"].eq("Y").astype(float))
    factor_fit = _fit("r2 ~ Anger + Gender + (1 | id)", subset)
    numeric_fit = _fit("r2 ~ Anger + Gender + (1 | id)", numeric)
    assert_allclose(factor_fit.beta, numeric_fit.beta, rtol=1e-9, atol=1e-9)
    assert_allclose(factor_fit.theta, numeric_fit.theta, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("values", [["N", "N"], ["N", "Y", "maybe"]])
def test_untyped_factor_response_requires_exactly_two_levels(values):
    frame = pd.DataFrame({"y": values, "g": np.arange(len(values))})
    with pytest.raises(ValueError, match="exactly two levels"):
        glFormula("y ~ 1 + (1 | g)", frame)


def test_numeric_boolean_and_proportion_responses_keep_original_values():
    for values in (
        [0.0, 0.2, 0.8, 1.0],
        [True, False, True, False],
        pd.Series([0.0, 0.2, 0.8, 1.0], dtype=object),
    ):
        frame = pd.DataFrame({"y": values, "g": [0, 0, 1, 1]})
        matrices = glFormula("y ~ 1 + (1 | g)", frame).matrices
        assert matrices.response_levels is None
        assert_array_equal(matrices.y, np.asarray(values, dtype=float))


def test_factor_responses_require_binomial_family(study):
    frame, _, _ = study
    with pytest.raises(ValueError, match="only supported for binomial"):
        glFormula("outcome ~ x + (1 | group)", frame, family=families.Poisson())


def test_polars_categorical_response_ignores_unrelated_shared_pool_levels():
    pl = pytest.importorskip("polars")
    with pl.StringCache():
        frame = pl.DataFrame(
            {
                "outcome": ["Y", "N", "Y", "N"],
                "treatment": ["low", "high", "low", "high"],
                "group": [0, 0, 1, 1],
            }
        ).with_columns(pl.col(["outcome", "treatment"]).cast(pl.Categorical))
        declared = frame["outcome"].cat.get_categories().to_list()
        expected_levels = tuple(level for level in declared if level in ("N", "Y"))
        matrices = glFormula("outcome ~ treatment + (1 | group)", frame).matrices
    assert matrices.response_levels == expected_levels
    assert_array_equal(
        matrices.y, (np.asarray(frame["outcome"].to_list()) == expected_levels[1]).astype(float)
    )


def test_factor_response_missing_rows_align_weights_and_offsets(study):
    frame, weights, offset = study
    frame = frame.copy()
    frame["outcome"] = pd.Categorical(frame["outcome"], categories=["Y", "N"])
    frame.loc[0, "outcome"] = np.nan
    result = glFormula(
        "outcome ~ x + (1 | group)", frame, weights=weights, offset=offset, na_action="omit"
    )
    assert result.matrices.response_levels == ("Y", "N")
    assert_array_equal(result.matrices.y, frame["outcome"].iloc[1:].eq("N").astype(float))
    assert_array_equal(result.matrices.weights, weights[1:])
    assert_array_equal(result.matrices.offset, offset[1:])
    with pytest.raises(ValueError, match="missing|NA"):
        glFormula("outcome ~ x + (1 | group)", frame, na_action="fail")


def test_unknown_refit_or_update_labels_are_rejected(study):
    frame, weights, offset = study
    model = _fit("outcome ~ x + (1 | group)", frame, weights=weights, offset=offset)
    labels = frame["outcome"].to_numpy().copy()
    labels[0] = "unknown"
    with pytest.raises(ValueError, match="unknown binomial response"):
        model.refit(labels)
    with pytest.raises(ValueError, match="unknown binomial response"):
        model.update(data=frame.assign(outcome=labels))
