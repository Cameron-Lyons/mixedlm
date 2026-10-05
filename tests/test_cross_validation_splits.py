"""Prescribed partitions must preserve out-of-fold and positional-fit semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import (
    CrossValidationFold,
    cross_validate,
    glmer,
    glmerControl,
    lmer,
    lmerControl,
    make_folds,
)
from mixedlm.inference import cross_validation as cv_module
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, xlogy

from tests._lmer_data import CBPP


@pytest.fixture(scope="module")
def fitted_study():
    rng = np.random.default_rng(883)
    group = np.repeat(np.arange(12), 8)
    x = rng.normal(size=len(group))
    z = rng.normal(size=len(group))
    offset = 0.25 * np.sin(x)
    weights = rng.uniform(0.7, 1.8, size=len(group))
    intercepts = rng.normal(scale=0.8, size=12)
    slopes = rng.normal(scale=0.5, size=12)
    # Rounded outcomes deliberately contain ties, so response matching alone
    # cannot establish row alignment for inherited offsets and prior weights.
    y = np.round(1.0 + 0.7 * x + intercepts[group] + slopes[group] * z + offset)
    data = pd.DataFrame({"y": y, "x": x, "z": z, "group": group.astype(str)})
    control = lmerControl(check_singular=False, check_nlev_gtreq_5="ignore")
    model = lmer(
        "y ~ x + z + (z | group)",
        data,
        weights=weights,
        offset=offset,
        REML=False,
        control=control,
    )
    return model, data, control


def test_prescribed_buffered_group_folds_match_independent_weighted_refits(fitted_study):
    model, data, control = fitted_study
    groups = data["group"].astype(int).to_numpy()
    splits = []
    for fold in range(3):
        held_out = np.arange(4 * fold, 4 * fold + 4)
        # Also exclude the next group from training. Custom folds need not use
        # every available training row, as in buffered spatial/time-block CV.
        excluded = np.append(held_out, (4 * fold + 4) % 12)
        splits.append(
            (np.flatnonzero(~np.isin(groups, excluded)), np.flatnonzero(np.isin(groups, held_out)))
        )
    result = cross_validate(
        model,
        cv=iter(splits),
        group="group",
        metrics=["mse", "mae"],
        n_jobs=2,
        fit_kwargs={"control": control},
    )

    expected = np.empty(len(data))
    for number, (train, test) in enumerate(splits):
        refitted = lmer(
            str(model.formula),
            data.iloc[train],
            weights=model.weights()[train],
            offset=model.offset()[train],
            REML=False,
            control=control,
        )
        expected[test] = (
            refitted.beta[0]
            + refitted.beta[1] * data["x"].to_numpy()[test]
            + refitted.beta[2] * data["z"].to_numpy()[test]
            + model.offset()[test]
        )
        assert_array_equal(result.folds[number].train_indices, train)
        assert_array_equal(result.folds[number].test_indices, test)
        assert_array_equal(result.fold_ids[test], np.full(len(test), number))
    assert_allclose(result.predictions, expected, rtol=1e-8, atol=1e-8)
    errors = data["y"].to_numpy() - expected
    assert result["mse"] == pytest.approx(np.average(errors**2, weights=model.weights()))
    assert result["mae"] == pytest.approx(np.average(np.abs(errors), weights=model.weights()))
    assert result.fold_scores["n_train"].tolist() == [56, 56, 56]


def test_generated_fold_objects_can_be_reused_without_advancing_random_stream(fitted_study):
    model, _, control = fitted_study
    folds = make_folds(model.nobs(), cv=2, random_state=91)
    expected = cross_validate(
        model, cv=2, random_state=91, metrics="mse", fit_kwargs={"control": control}
    )
    rng = np.random.default_rng(82)
    supplied = cross_validate(
        model, cv=folds, random_state=rng, metrics="mse", fit_kwargs={"control": control}
    )
    assert_allclose(supplied.predictions, expected.predictions, rtol=1e-8, atol=1e-8)
    assert supplied.scores == pytest.approx(expected.scores)
    assert rng.random() == np.random.default_rng(82).random()


def test_unsorted_explicit_binomial_splits_match_proportion_and_offset_refits():
    data = CBPP.assign(proportion=CBPP["incidence"] / CBPP["size"])
    weights = np.linspace(0.8, 1.6, len(data))
    offset = np.linspace(-0.2, 0.3, len(data))
    control = glmerControl(check_singular=False, check_nlev_gtreq_5="ignore")
    model = glmer(
        "incidence / size ~ period + (1 | herd)",
        data,
        weights=weights,
        offset=offset,
        control=control,
    )
    original_folds = make_folds(len(data), cv=2, groups=data["herd"], random_state=71)
    splits = [(fold.train_indices[::-1], fold.test_indices[::-1]) for fold in original_folds]
    actual = cross_validate(model, cv=splits, group="herd", fit_kwargs={"control": control})
    expected = np.empty(len(data))
    effective_weights = weights * data["size"].to_numpy()
    for train, test in splits:
        fitted = glmer(
            "proportion ~ period + (1 | herd)",
            data.iloc[train],
            weights=effective_weights[train],
            offset=offset[train],
            control=control,
        )
        period = data["period"].to_numpy()[test]
        design = np.column_stack([np.ones(len(test)), *(period == str(i) for i in (2, 3, 4))])
        expected[test] = expit(design @ fitted.beta + offset[test])
    assert_allclose(actual.predictions, expected, rtol=1e-7, atol=1e-8)
    observed = data["proportion"].to_numpy()
    terms = xlogy(observed, observed / expected) + xlogy(
        1 - observed, (1 - observed) / (1 - expected)
    )
    expected_deviance = 2 * np.average(terms, weights=effective_weights)
    assert actual["deviance"] == pytest.approx(expected_deviance, rel=1e-7)


@pytest.mark.parametrize(
    ("splits", "error", "match"),
    [
        ([], ValueError, "at least two"),
        ([([1, 2], [0])], ValueError, "at least two"),
        ([([2, 3], [0]), ([0, 1], [2, 3])], ValueError, "cover every observation"),
        ([([2, 3], [0, 1]), ([0], [1, 2, 3])], ValueError, "must not overlap"),
        ([([0, 2, 3], [0, 1]), ([0, 1], [2, 3])], ValueError, "must be disjoint"),
        ([([2, 3], [0, 0, 1]), ([0, 1], [2, 3])], ValueError, "duplicate rows"),
        ([([2, 2, 3], [0, 1]), ([0, 1], [2, 3])], ValueError, "duplicate rows"),
        ([([2, 3], [-1, 0, 1]), ([0, 1], [2, 3])], ValueError, "outside"),
        ([([2, 4], [0, 1]), ([0, 1], [2, 3])], ValueError, "outside"),
        ([([2, 3], [0.0, 1.0]), ([0, 1], [2, 3])], TypeError, "integer row"),
        ([([2, 3], [False, True]), ([0, 1], [2, 3])], TypeError, "integer row"),
        ([([], [0, 1]), ([0, 1], [2, 3])], ValueError, "nonempty 1D"),
        ([([2, 3], [[0, 1]]), ([0, 1], [2, 3])], ValueError, "nonempty 1D"),
        ([([0], [1], [2])], ValueError, "each cv split"),
        ("2", TypeError, "integer or an iterable"),
        (2.0, TypeError, "integer or an iterable"),
    ],
)
def test_explicit_fold_validation_rejects_invalid_partitions(splits, error, match):
    with pytest.raises(error, match=match):
        cv_module._explicit_folds(splits, 4, groups=None)


def test_explicit_fold_validation_does_not_cast_masked_or_overflowing_positions():
    masked = np.ma.array([0, 1], mask=[False, True])
    with pytest.raises(ValueError, match="masked"):
        cv_module._explicit_folds([([2, 3], masked), ([0, 1], [2, 3])], 4, None)
    huge = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(ValueError, match="outside"):
        cv_module._explicit_folds([([2, 3], huge), ([0, 1], [2, 3])], 4, None)


def test_explicit_folds_snapshot_arrays_and_number_in_iteration_order():
    train = np.array([2, 3])
    test = np.array([1, 0])
    folds = cv_module._explicit_folds(
        [CrossValidationFold(18, train, test), CrossValidationFold(18, test, train)], 4, None
    )
    train[:] = 0
    test[:] = 2
    assert [fold.fold for fold in folds] == [0, 1]
    assert_array_equal(folds[0].train_indices, [2, 3])
    assert_array_equal(folds[0].test_indices, [1, 0])
    with pytest.raises(ValueError, match="read-only"):
        folds[0].test_indices[0] = 0


def test_prescribed_group_folds_reject_training_leakage_before_fitting(fitted_study, monkeypatch):
    model, _, _ = fitted_study
    folds = make_folds(model.nobs(), cv=2, shuffle=False)
    # Move one row between test folds, splitting both affected groups.
    left, right = (fold.test_indices.copy() for fold in folds)
    left[0], right[0] = right[0], left[0]
    splits = [(np.setdiff1d(np.arange(model.nobs()), test), test) for test in (left, right)]
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match="train and test groups must be disjoint"):
        cross_validate(model, cv=splits, group="group")


def test_group_cannot_be_split_across_tests_even_when_excluded_from_training():
    splits = [([2], [0]), ([2], [1]), ([0, 1], [2, 3])]
    with pytest.raises(ValueError, match="whole group in a single fold"):
        cv_module._explicit_folds(splits, 4, np.array(["a", "a", "b", "b"]))


@pytest.mark.parametrize("column", ["x", "z", "group"])
def test_supplied_data_cannot_change_modeled_values_with_unchanged_responses(
    fitted_study, monkeypatch, column
):
    model, data, _ = fitted_study
    altered = data.copy()
    altered.loc[0, column] = "other-group" if column == "group" else altered.loc[0, column] + 1.0
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match=f"column '{column}'.*not aligned"):
        cross_validate(model, altered, cv=2)


def test_equal_response_row_swaps_cannot_misalign_weights_and_offsets(fitted_study, monkeypatch):
    model, data, _ = fitted_study
    indices = np.flatnonzero(data["y"].to_numpy() == data["y"].iloc[0])
    first, second = int(indices[0]), int(indices[-1])
    order = np.arange(len(data))
    order[first], order[second] = order[second], order[first]
    shuffled = data.iloc[order].reset_index(drop=True)
    assert_array_equal(shuffled["y"], data["y"])
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match="column .*not aligned"):
        cross_validate(model, shuffled, cv=2)


def test_supplied_categorical_metadata_cannot_change_fitted_contrast_basis(monkeypatch):
    rng = np.random.default_rng(601)
    group = np.repeat(np.arange(8), 8)
    condition = np.tile(["A", "B"], 32)
    y = 1 + 0.8 * (condition == "B") + rng.normal(size=64)
    data = pd.DataFrame(
        {"y": y, "condition": pd.Categorical(condition, categories=["B", "A"]), "group": group}
    )
    fitted = lmer("y ~ condition + (1 | group)", data, control=lmerControl(check_singular=False))
    altered = data.copy()
    altered["condition"] = altered["condition"].cat.reorder_categories(["A", "B"])
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match="categorical encoding"):
        cross_validate(fitted, altered, cv=2)


def test_supplied_data_can_add_external_fold_groups_and_change_dataframe_index(fitted_study):
    model, data, control = fitted_study
    supplied = data.assign(external_group=data["group"])
    supplied.index = np.arange(len(data)) * 7 + 19
    expected = cross_validate(
        model, cv=2, group="group", random_state=67, fit_kwargs={"control": control}
    )
    actual = cross_validate(
        model,
        supplied,
        cv=expected.folds,
        group="external_group",
        fit_kwargs={"control": control},
    )
    assert_allclose(actual.predictions, expected.predictions, rtol=1e-8, atol=1e-8)
    assert actual.scores == pytest.approx(expected.scores)


def test_missing_external_groups_are_rejected_before_prescribed_refits(fitted_study, monkeypatch):
    model, data, _ = fitted_study
    supplied = data.assign(external_group=data["group"])
    supplied.loc[0, "external_group"] = None
    folds = make_folds(model.nobs(), cv=2, shuffle=False)
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match="groups cannot contain missing"):
        cross_validate(model, supplied, cv=folds, group="external_group")


def test_missing_modeled_predictors_are_rejected_before_prescribed_refits(
    fitted_study, monkeypatch
):
    model, data, _ = fitted_study
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))
    with pytest.raises(ValueError, match="missing modeled columns.*x"):
        cross_validate(model, data.drop(columns="x"), cv=make_folds(model.nobs(), cv=2))


def test_results_without_source_frames_validate_encoded_observation_alignment(
    fitted_study, monkeypatch
):
    model, data, _ = fitted_study
    monkeypatch.setattr(model.matrices, "frame", None)
    cv_module._validate_predictor_alignment(model, data)
    changed_predictor = data.copy()
    changed_predictor.loc[0, "x"] += 1.0
    with pytest.raises(ValueError, match="fixed-effect values.*not aligned"):
        cv_module._validate_predictor_alignment(model, changed_predictor)
    changed_group = data.copy()
    changed_group.loc[0, "group"] = changed_group.loc[8, "group"]
    with pytest.raises(ValueError, match="random-effect values.*not aligned"):
        cv_module._validate_predictor_alignment(model, changed_group)
