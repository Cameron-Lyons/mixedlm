from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import cross_validate, families, glmer, lmer, lmerControl
from mixedlm.inference.cross_validation import (
    CrossValidationResult,
    make_folds,
    weighted_mae,
    weighted_mse,
    weighted_r2,
    weighted_rmse,
)
from numpy.testing import assert_allclose, assert_array_equal

from tests._lmer_data import CBPP, SLEEPSTUDY


def test_case_folds_are_exhaustive_balanced_and_reproducible() -> None:
    folds = make_folds(23, cv=5, random_state=42)
    repeated = make_folds(23, cv=5, random_state=42)

    test_rows = np.concatenate([fold.test_indices for fold in folds])
    sizes = [len(fold.test_indices) for fold in folds]

    assert_array_equal(np.sort(test_rows), np.arange(23))
    assert max(sizes) - min(sizes) <= 1
    for fold, same_fold in zip(folds, repeated, strict=True):
        assert_array_equal(fold.test_indices, same_fold.test_indices)
        assert len(np.intersect1d(fold.train_indices, fold.test_indices)) == 0


def test_group_folds_keep_groups_intact_and_balance_observations() -> None:
    group_sizes = [17, 11, 9, 7, 6, 5, 4, 3]
    groups = np.concatenate([np.repeat(f"g{i}", size) for i, size in enumerate(group_sizes)])
    folds = make_folds(len(groups), cv=3, groups=groups, random_state=7)

    held_out_groups: list[str] = []
    test_sizes = []
    for fold in folds:
        train_groups = set(groups[fold.train_indices])
        test_groups = set(groups[fold.test_indices])
        assert train_groups.isdisjoint(test_groups)
        held_out_groups.extend(test_groups)
        test_sizes.append(len(fold.test_indices))

    assert sorted(held_out_groups) == sorted(set(groups))
    assert max(test_sizes) - min(test_sizes) <= max(group_sizes)


@pytest.mark.parametrize(
    ("n_samples", "cv", "match"),
    [(1, 2, "n_samples"), (5, 1, "cv"), (5, 6, "cannot exceed")],
)
def test_invalid_case_fold_sizes_are_rejected(n_samples: int, cv: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_folds(n_samples, cv=cv)


def test_invalid_group_folds_are_rejected() -> None:
    with pytest.raises(ValueError, match="one value per observation"):
        make_folds(4, cv=2, groups=["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        make_folds(4, cv=2, groups=["a", "b", None, "c"])
    with pytest.raises(ValueError, match="unique groups"):
        make_folds(4, cv=3, groups=["a", "a", "b", "b"])


def test_weighted_scores_match_direct_calculations() -> None:
    observed = np.array([1.0, 2.0, 5.0, 8.0])
    predicted = np.array([1.5, 1.0, 6.0, 7.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    errors = observed - predicted

    expected_rmse = np.sqrt(np.average(np.square(errors), weights=weights))
    expected_mse = np.average(np.square(errors), weights=weights)
    expected_mae = np.average(np.abs(errors), weights=weights)
    mean = np.average(observed, weights=weights)
    expected_r2 = 1 - np.sum(weights * np.square(errors)) / np.sum(
        weights * np.square(observed - mean)
    )

    assert weighted_mse(observed, predicted, weights) == pytest.approx(expected_mse)
    assert weighted_rmse(observed, predicted, weights) == pytest.approx(expected_rmse)
    assert weighted_mae(observed, predicted, weights) == pytest.approx(expected_mae)
    assert weighted_r2(observed, predicted, weights) == pytest.approx(expected_r2)


def test_weighted_r2_handles_constant_responses() -> None:
    observed = np.ones(4)
    assert weighted_r2(observed, observed) == 1
    assert weighted_r2(observed, np.zeros(4)) == 0


def test_score_validation_rejects_bad_arrays() -> None:
    with pytest.raises(ValueError, match="aligned"):
        weighted_rmse(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        weighted_mae(np.ones(2), np.ones(2), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        weighted_r2(np.array([1.0, np.nan]), np.ones(2))


@pytest.fixture(scope="module")
def weighted_lmm() -> tuple[object, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    n_groups = 12
    n_per_group = 8
    group_index = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.normal(size=len(group_index))
    group_effect = rng.normal(scale=0.7, size=n_groups)
    offset = 0.15 * np.sin(x)
    weights = rng.uniform(0.5, 2.0, size=len(x))
    y = 2.0 + 1.4 * x + group_effect[group_index] + offset + rng.normal(scale=0.35, size=len(x))
    data = pd.DataFrame({"y": y, "x": x, "group": group_index.astype(str)})
    model = lmer(
        "y ~ x + (1 | group)",
        data,
        weights=weights,
        offset=offset,
        REML=False,
        control=lmerControl(check_singular=False),
    )
    return model, data


def median_absolute_error(y_true, y_pred, weights) -> float:
    del weights
    return float(np.median(np.abs(y_true - y_pred)))


def test_grouped_lmm_cross_validation_returns_aligned_predictions(weighted_lmm) -> None:
    model, data = weighted_lmm
    result = cross_validate(
        model,
        cv=3,
        group="group",
        metrics=["rmse", "mae", "r2", median_absolute_error],
        random_state=11,
        fit_kwargs={"control": lmerControl(check_singular=False)},
    )

    assert isinstance(result, CrossValidationResult)
    assert result.n_folds == 3
    assert result.group == "group"
    assert result.metric_names == ("rmse", "mae", "r2", "median_absolute_error")
    assert result.predictions.shape == (len(data),)
    assert np.all(np.isfinite(result.predictions))
    assert_array_equal(np.sort(np.unique(result.fold_ids)), [0, 1, 2])
    assert result["rmse"] > 0
    assert result.fold_scores["n_test"].sum() == len(data)
    assert result.any_singular == bool(result.fold_scores["singular"].any())
    assert result.all_converged == bool(result.fold_scores["converged"].all())
    assert result.summary()["metric"].tolist() == list(result.metric_names)
    assert "grouped by 'group'" in str(result)

    group_values = data["group"].to_numpy()
    for fold in result.folds:
        assert set(group_values[fold.train_indices]).isdisjoint(group_values[fold.test_indices])


def test_case_level_lmm_cross_validation(weighted_lmm) -> None:
    model, data = weighted_lmm
    fit_kwargs = {"control": lmerControl(check_singular=False)}
    result = cross_validate(
        model,
        data,
        cv=2,
        metrics=["mse", "rmse"],
        random_state=3,
        n_jobs=2,
        fit_kwargs=fit_kwargs,
    )
    serial = cross_validate(
        model,
        data,
        cv=2,
        metrics=["mse", "rmse"],
        random_state=3,
        n_jobs=1,
        fit_kwargs=fit_kwargs,
    )

    assert result.group is None
    assert result.metric_names == ("mse", "rmse")
    assert result["rmse"] == pytest.approx(np.sqrt(result["mse"]))
    assert result.fold_scores["n_test"].tolist() == [len(data) // 2, len(data) // 2]
    assert_allclose(result.predictions, serial.predictions)
    assert_allclose(result.fold_scores[["mse", "rmse"]], serial.fold_scores[["mse", "rmse"]])
    assert "case-level" in str(result)


@pytest.fixture(scope="module")
def grouped_binomial_model():
    data = CBPP.copy()
    data["y"] = data["incidence"] / data["size"]
    model = glmer(
        "y ~ period + (1 | herd)",
        data,
        family=families.Binomial(),
        weights=data["size"].to_numpy(),
    )
    return model


def test_grouped_glmm_cross_validation_includes_deviance(grouped_binomial_model) -> None:
    result = cross_validate(
        grouped_binomial_model,
        cv=3,
        group="herd",
        random_state=19,
    )

    assert result.metric_names == ("rmse", "deviance")
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["deviance"])
    assert result["deviance"] >= 0
    assert result.fold_scores["n_test"].sum() == grouped_binomial_model.nobs()
    assert "singular" in result.fold_scores


def test_polars_model_frame_cross_validation() -> None:
    pl = pytest.importorskip("polars")
    data = pl.DataFrame(SLEEPSTUDY.to_dict(orient="list"))
    control = lmerControl(check_singular=False)
    model = lmer("Reaction ~ Days + (1 | Subject)", data, REML=False, control=control)

    result = cross_validate(
        model,
        cv=2,
        group="Subject",
        random_state=5,
        fit_kwargs={"control": control},
    )

    assert result.n_folds == 2
    assert result.fold_scores["n_test"].sum() == len(SLEEPSTUDY)
    assert np.all(np.isfinite(result.predictions))


def test_cross_validation_input_errors(weighted_lmm) -> None:
    model, data = weighted_lmm
    with pytest.raises(ValueError, match="group column"):
        cross_validate(model, cv=2, group="missing")
    with pytest.raises(ValueError, match="same aligned observations"):
        cross_validate(model, data.iloc[:-1], cv=2)
    with pytest.raises(ValueError, match="generalized linear mixed model"):
        cross_validate(model, cv=2, metrics="deviance")
    with pytest.raises(ValueError, match="reserved arguments"):
        cross_validate(model, cv=2, fit_kwargs={"weights": np.ones(len(data))})
    with pytest.raises(ValueError, match="unknown metric"):
        cross_validate(model, cv=2, metrics="not_a_metric")
    with pytest.raises(ValueError, match="n_jobs"):
        cross_validate(model, cv=2, n_jobs=0)
    shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)
    with pytest.raises(ValueError, match="response values"):
        cross_validate(model, shuffled, cv=2)


def test_root_exports_are_available() -> None:
    import mixedlm as mlm

    assert mlm.cross_validate is cross_validate
    assert mlm.make_folds is make_folds
    assert mlm.weighted_mse is weighted_mse
    assert_allclose(mlm.weighted_rmse(np.array([0, 1]), np.array([0, 2])), np.sqrt(0.5))
