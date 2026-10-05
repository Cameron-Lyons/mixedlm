from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mixedlm import cross_validate, families, glmer, glmerControl, lmer, lmerControl
from mixedlm.inference import cross_validation as cv_module
from mixedlm.inference.cross_validation import (
    CrossValidationFold,
    CrossValidationResult,
    make_folds,
    weighted_mae,
    weighted_mse,
    weighted_r2,
    weighted_rmse,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, xlogy

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


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_weighted_r2_does_not_classify_tiny_constant_errors_as_perfect(scale: float) -> None:
    observed = np.full(4, scale)
    predicted = observed * (1.0 + 1e-8)

    assert weighted_r2(observed, predicted) == 0


def test_weighted_r2_reports_unrepresentable_error_ratio_without_dividing_by_zero() -> None:
    assert weighted_r2(np.array([1e-200, 2e-200]), np.ones(2)) == float("-inf")


@pytest.mark.parametrize(
    ("prediction", "weights", "expected"),
    [
        (np.array([2.0, 2.0, 2.0]), None, 0.0),
        (np.array([2.0, 4.0, 6.0]), None, -0.5),
        (np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]), -0.2),
        (np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 3.0]), -0.8),
    ],
)
def test_weighted_r2_preserves_representable_differences_at_large_baselines(
    prediction, weights, expected
) -> None:
    observed = np.array([0.0, 2.0, 4.0])
    assert weighted_r2(observed, prediction, weights) == pytest.approx(expected, abs=1e-14)
    assert weighted_r2(1e16 + observed, 1e16 + prediction, weights) == pytest.approx(
        expected, abs=1e-14
    )


def test_weighted_r2_supports_finite_values_whose_range_overflows() -> None:
    observed = np.array([-1e308, 0.0, 1e308])

    assert weighted_r2(observed, np.zeros(3)) == pytest.approx(0.0)
    assert weighted_r2(observed, observed) == 1.0


@pytest.mark.parametrize("response_scale", [1e-150, 1e-10, 1.0, 1e150])
@pytest.mark.parametrize("weight_scale", [1e-150, 1.0, 1e150])
def test_weighted_r2_is_invariant_to_response_and_weight_units(
    response_scale: float, weight_scale: float
) -> None:
    observed = np.array([1.0, 2.0, 5.0, 8.0])
    predicted = np.array([1.5, 1.0, 6.0, 7.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    # Exact weighted sums: weighted response mean = 5.2, SSE = 9.25,
    # and centered sum of squares = 69.6. Changing units cannot change R2.
    expected = 1.0 - 9.25 / 69.6

    assert weighted_r2(
        response_scale * observed, response_scale * predicted, weight_scale * weights
    ) == pytest.approx(expected)


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


@pytest.mark.parametrize("group", [None, "group"], ids=["conditional", "new-groups"])
def test_lmm_cross_validation_matches_independent_weighted_offset_refits(
    weighted_lmm, group: str | None
) -> None:
    model, data = weighted_lmm
    control = lmerControl(check_singular=False)
    result = cross_validate(
        model,
        cv=2,
        group=group,
        random_state=61,
        metrics=["mse", "mae"],
        fit_kwargs={"control": control},
    )
    expected = np.empty(len(data))
    weights = model.weights()
    offsets = model.offset()
    for fold in result.folds:
        trained = lmer(
            str(model.formula),
            data.iloc[fold.train_indices],
            weights=weights[fold.train_indices],
            offset=offsets[fold.train_indices],
            REML=False,
            control=control,
        )
        held_out = data.iloc[fold.test_indices]
        means = (
            trained.beta[0]
            + trained.beta[1] * held_out["x"].to_numpy()
            + offsets[fold.test_indices]
        )
        if group is None:
            structure = trained.matrices.random_structures[0]
            random_intercepts = trained.ranef()["group"]["(Intercept)"]
            means += np.array(
                [
                    random_intercepts[structure.level_map[level]]
                    if level in structure.level_map
                    else 0.0
                    for level in held_out["group"]
                ]
            )
        expected[fold.test_indices] = means

    assert_allclose(result.predictions, expected, rtol=1e-8, atol=1e-8)
    errors = data["y"].to_numpy() - expected
    assert result["mse"] == pytest.approx(np.sum(weights * errors**2) / np.sum(weights))
    assert result["mae"] == pytest.approx(np.sum(weights * np.abs(errors)) / np.sum(weights))


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


def test_grouped_binomial_count_syntax_matches_manual_proportion_cross_validation() -> None:
    data = CBPP.assign(proportion=CBPP["incidence"] / CBPP["size"])
    prior_weights = np.linspace(0.7, 1.6, len(data))
    effective_weights = prior_weights * data["size"].to_numpy()
    grouped = glmer("incidence / size ~ period + (1 | herd)", data, weights=prior_weights)
    manual = glmer("proportion ~ period + (1 | herd)", data, weights=effective_weights)
    grouped_cv = cross_validate(grouped, cv=2, group="herd", random_state=109)
    manual_cv = cross_validate(manual, cv=2, group="herd", random_state=109)

    assert_allclose(grouped_cv.predictions, manual_cv.predictions, rtol=1e-8, atol=1e-8)
    assert grouped_cv.scores == pytest.approx(manual_cv.scores)
    assert_allclose(
        grouped_cv.fold_scores[["rmse", "deviance"]],
        manual_cv.fold_scores[["rmse", "deviance"]],
        rtol=1e-8,
        atol=1e-8,
    )
    observed = data["proportion"].to_numpy()
    predicted = grouped_cv.predictions
    binomial_terms = np.zeros(len(observed))
    positive = observed > 0
    below_one = observed < 1
    binomial_terms[positive] += observed[positive] * np.log(
        observed[positive] / predicted[positive]
    )
    binomial_terms[below_one] += (1 - observed[below_one]) * np.log(
        (1 - observed[below_one]) / (1 - predicted[below_one])
    )
    expected_deviance = 2 * np.sum(effective_weights * binomial_terms) / np.sum(effective_weights)
    assert grouped_cv["deviance"] == pytest.approx(expected_deviance)

    # Identical proportions with different denominators represent different
    # observations and cannot inherit the original trial weights.
    changed_trials = data.copy()
    changed_trials.loc[0, ["incidence", "size"]] *= 2
    with pytest.raises(ValueError, match="trial counts.*aligned"):
        cross_validate(grouped, changed_trials, cv=2)


def test_glmm_cross_validation_applies_exposure_before_inverse_link() -> None:
    rng = np.random.default_rng(289)
    groups = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(groups))
    exposure = rng.uniform(0.5, 5.0, size=len(groups))
    offset = np.log(exposure)
    group_effects = rng.normal(scale=0.4, size=8)
    y = rng.poisson(np.exp(0.5 + 0.35 * x + group_effects[groups] + offset))
    data = pd.DataFrame({"y": y, "x": x, "group": groups.astype(str)})
    control = glmerControl(check_singular=False, check_nlev_gtreq_5="ignore")
    model = glmer("y ~ x + (1 | group)", data, families.Poisson(), offset=offset, control=control)
    result = cross_validate(
        model, cv=2, group="group", random_state=53, fit_kwargs={"control": control}
    )
    expected = np.empty(len(data))
    for fold in result.folds:
        trained = glmer(
            str(model.formula),
            data.iloc[fold.train_indices],
            families.Poisson(),
            offset=offset[fold.train_indices],
            control=control,
        )
        expected[fold.test_indices] = exposure[fold.test_indices] * np.exp(
            trained.beta[0] + trained.beta[1] * x[fold.test_indices]
        )

    assert_allclose(result.predictions, expected, rtol=1e-8, atol=1e-8)
    # Compute the Poisson deviance directly, with the zero-count convention.
    positive = y > 0
    contribution = expected - y
    contribution[positive] += y[positive] * np.log(y[positive] / expected[positive])
    assert result["deviance"] == pytest.approx(2.0 * np.mean(contribution))


@pytest.mark.parametrize("name", ["fold", "n_train", "n_test", "converged", "singular"])
def test_custom_metric_cannot_replace_fold_metadata(weighted_lmm, name: str) -> None:
    def scorer(y_true, y_pred, weights) -> float:
        return 1.0

    scorer.__name__ = name
    model, _ = weighted_lmm
    with pytest.raises(ValueError, match="reserved.*fold metadata"):
        cross_validate(model, cv=2, metrics=scorer)


@pytest.mark.parametrize("generalized", [False, True], ids=["lmer", "glmer"])
def test_cross_validation_preserves_categorical_random_slope_contrasts(
    generalized: bool,
) -> None:
    rng = np.random.default_rng(489)
    groups = np.repeat(np.arange(12), 8)
    coded_condition = np.tile([-1.0, 1.0], len(groups) // 2)
    intercepts = rng.normal(scale=0.6, size=12)
    slopes = rng.normal(scale=0.3, size=12)
    eta = 1.0 + 0.5 * coded_condition + intercepts[groups] + slopes[groups] * coded_condition
    y = rng.poisson(np.exp(eta)) if generalized else eta + rng.normal(scale=0.3, size=len(groups))
    data = pd.DataFrame(
        {
            "y": y,
            "condition": np.where(coded_condition < 0, "A", "B"),
            "group": groups.astype(str),
        }
    )
    contrasts = {"condition": np.array([[-1.0], [1.0]])}
    formula = "y ~ condition + (condition || group)"
    control = (
        glmerControl(check_singular=False) if generalized else lmerControl(check_singular=False)
    )
    fit_options = (
        {"family": families.Poisson(), "control": control}
        if generalized
        else {"REML": False, "control": control}
    )
    fit = glmer if generalized else lmer
    model = fit(formula, data, contrasts=contrasts, **fit_options)
    result = cross_validate(
        model, cv=2, random_state=88, metrics="mse", fit_kwargs={"control": control}
    )
    expected = np.empty(len(data))
    for fold in result.folds:
        trained = fit(formula, data.iloc[fold.train_indices], contrasts=contrasts, **fit_options)
        expected[fold.test_indices] = trained.predict(
            data.iloc[fold.test_indices], allow_new_levels=True
        )

    # The diagonal random covariance is defined in the fitted contrast basis.
    # Switching to treatment contrasts during refits changes the actual model.
    assert_allclose(result.predictions, expected, rtol=1e-8, atol=1e-8)


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


@pytest.fixture
def forbid_refits(monkeypatch):
    """Fail if cross-validation starts a refit before rejecting its inputs."""
    monkeypatch.setattr(cv_module, "_fit_fold", lambda *args: pytest.fail("fit before validation"))


@pytest.mark.installed_wheel
class TestPrescribedFolds:
    """Prescribed partitions must preserve out-of-fold and positional-fit semantics."""

    def test_prescribed_buffered_group_folds_match_independent_weighted_refits(
        self, fitted_study
    ) -> None:
        model, data, control = fitted_study
        groups = data["group"].astype(int).to_numpy()
        splits = []
        for fold in range(3):
            held_out = np.arange(4 * fold, 4 * fold + 4)
            # Also exclude the next group from training. Custom folds need not use
            # every available training row, as in buffered spatial/time-block CV.
            excluded = np.append(held_out, (4 * fold + 4) % 12)
            splits.append(
                (
                    np.flatnonzero(~np.isin(groups, excluded)),
                    np.flatnonzero(np.isin(groups, held_out)),
                )
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

    def test_generated_fold_objects_can_be_reused_without_advancing_random_stream(
        self, fitted_study
    ) -> None:
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

    def test_unsorted_explicit_binomial_splits_match_proportion_and_offset_refits(self) -> None:
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
    def test_explicit_fold_validation_rejects_invalid_partitions(
        self, splits, error, match
    ) -> None:
        with pytest.raises(error, match=match):
            cv_module._explicit_folds(splits, 4, groups=None)

    def test_explicit_fold_validation_does_not_cast_masked_or_overflowing_positions(self) -> None:
        masked = np.ma.array([0, 1], mask=[False, True])
        with pytest.raises(ValueError, match="masked"):
            cv_module._explicit_folds([([2, 3], masked), ([0, 1], [2, 3])], 4, None)
        huge = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
        with pytest.raises(ValueError, match="outside"):
            cv_module._explicit_folds([([2, 3], huge), ([0, 1], [2, 3])], 4, None)

    def test_explicit_folds_snapshot_arrays_and_number_in_iteration_order(self) -> None:
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

    def test_prescribed_group_folds_reject_training_leakage_before_fitting(
        self, fitted_study, forbid_refits
    ) -> None:
        model, _, _ = fitted_study
        folds = make_folds(model.nobs(), cv=2, shuffle=False)
        # Move one row between test folds, splitting both affected groups.
        left, right = (fold.test_indices.copy() for fold in folds)
        left[0], right[0] = right[0], left[0]
        splits = [(np.setdiff1d(np.arange(model.nobs()), test), test) for test in (left, right)]
        with pytest.raises(ValueError, match="train and test groups must be disjoint"):
            cross_validate(model, cv=splits, group="group")

    def test_group_cannot_be_split_across_tests_even_when_excluded_from_training(self) -> None:
        splits = [([2], [0]), ([2], [1]), ([0, 1], [2, 3])]
        with pytest.raises(ValueError, match="whole group in a single fold"):
            cv_module._explicit_folds(splits, 4, np.array(["a", "a", "b", "b"]))

    @pytest.mark.parametrize("column", ["x", "z", "group"])
    def test_supplied_data_cannot_change_modeled_values_with_unchanged_responses(
        self, fitted_study, forbid_refits, column
    ) -> None:
        model, data, _ = fitted_study
        altered = data.copy()
        altered.loc[0, column] = (
            "other-group" if column == "group" else altered.loc[0, column] + 1.0
        )
        with pytest.raises(ValueError, match=f"column '{column}'.*not aligned"):
            cross_validate(model, altered, cv=2)

    def test_equal_response_row_swaps_cannot_misalign_weights_and_offsets(
        self, fitted_study, forbid_refits
    ) -> None:
        model, data, _ = fitted_study
        indices = np.flatnonzero(data["y"].to_numpy() == data["y"].iloc[0])
        first, second = int(indices[0]), int(indices[-1])
        order = np.arange(len(data))
        order[first], order[second] = order[second], order[first]
        shuffled = data.iloc[order].reset_index(drop=True)
        assert_array_equal(shuffled["y"], data["y"])
        with pytest.raises(ValueError, match="column .*not aligned"):
            cross_validate(model, shuffled, cv=2)

    def test_supplied_categorical_metadata_cannot_change_fitted_contrast_basis(
        self, forbid_refits
    ) -> None:
        rng = np.random.default_rng(601)
        group = np.repeat(np.arange(8), 8)
        condition = np.tile(["A", "B"], 32)
        y = 1 + 0.8 * (condition == "B") + rng.normal(size=64)
        data = pd.DataFrame(
            {"y": y, "condition": pd.Categorical(condition, categories=["B", "A"]), "group": group}
        )
        fitted = lmer(
            "y ~ condition + (1 | group)", data, control=lmerControl(check_singular=False)
        )
        altered = data.copy()
        altered["condition"] = altered["condition"].cat.reorder_categories(["A", "B"])
        with pytest.raises(ValueError, match="categorical encoding"):
            cross_validate(fitted, altered, cv=2)

    def test_supplied_data_can_add_external_fold_groups_and_change_dataframe_index(
        self, fitted_study
    ) -> None:
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

    def test_missing_external_groups_are_rejected_before_prescribed_refits(
        self, fitted_study, forbid_refits
    ) -> None:
        model, data, _ = fitted_study
        supplied = data.assign(external_group=data["group"])
        supplied.loc[0, "external_group"] = None
        folds = make_folds(model.nobs(), cv=2, shuffle=False)
        with pytest.raises(ValueError, match="groups cannot contain missing"):
            cross_validate(model, supplied, cv=folds, group="external_group")

    def test_missing_modeled_predictors_are_rejected_before_prescribed_refits(
        self, fitted_study, forbid_refits
    ) -> None:
        model, data, _ = fitted_study
        with pytest.raises(ValueError, match="missing modeled columns.*x"):
            cross_validate(model, data.drop(columns="x"), cv=make_folds(model.nobs(), cv=2))

    def test_results_without_source_frames_validate_encoded_observation_alignment(
        self, fitted_study, monkeypatch
    ) -> None:
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
