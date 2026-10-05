"""Case-level and grouped cross-validation for mixed-effects models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from mixedlm.matrices.design import _build_response, _restore_binomial_factor
from mixedlm.utils.dataframe import (
    dataframe_length,
    ensure_dataframe,
    get_categories,
    get_column_numpy,
    get_columns,
    is_categorical_or_string,
)

if TYPE_CHECKING:
    from mixedlm.families import Family
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

MetricFunction: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], float
]
MetricSpec: TypeAlias = str | MetricFunction

_BUILTIN_METRICS = {"mse", "rmse", "mae", "r2", "deviance"}
_RESERVED_FIT_ARGUMENTS = {"data", "family", "formula", "na_action", "offset", "weights"}
_FOLD_METADATA = {"fold", "n_train", "n_test", "converged", "singular"}


@dataclass(frozen=True)
class CrossValidationFold:
    """Train and test row positions for one cross-validation fold."""

    fold: int
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]


FoldSpec: TypeAlias = CrossValidationFold | tuple[ArrayLike, ArrayLike]


@dataclass
class CrossValidationResult:
    """Out-of-fold predictions and scores from mixed-model cross-validation."""

    scores: dict[str, float]
    fold_scores: pd.DataFrame
    predictions: NDArray[np.float64]
    fold_ids: NDArray[np.int64]
    folds: tuple[CrossValidationFold, ...]
    group: str | None
    metric_names: tuple[str, ...]

    @property
    def all_converged(self) -> bool:
        """Whether every fold fit reported successful convergence."""
        return bool(self.fold_scores["converged"].all())

    @property
    def any_singular(self) -> bool:
        """Whether any fold fit is on a random-effects boundary."""
        return bool(self.fold_scores["singular"].any())

    @property
    def n_folds(self) -> int:
        """Number of fitted folds."""
        return len(self.folds)

    def __getitem__(self, metric: str) -> float:
        return self.scores[metric]

    def summary(self) -> pd.DataFrame:
        """Return overall and between-fold summaries for every metric."""
        rows: list[dict[str, float | str]] = []
        for metric in self.metric_names:
            values = self.fold_scores[metric].to_numpy(dtype=np.float64)
            mean, standard_deviation = _fold_score_moments(values)
            rows.append(
                {
                    "metric": metric,
                    "overall": self.scores[metric],
                    "fold_mean": mean,
                    "fold_std": standard_deviation,
                    "fold_min": float(np.min(values)),
                    "fold_max": float(np.max(values)),
                }
            )
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        mode = f"grouped by '{self.group}'" if self.group is not None else "case-level"
        lines = [f"{self.n_folds}-fold cross-validation ({mode})", ""]
        lines.append(f"  {'Metric':<14} {'Overall':>12} {'Fold mean':>12} {'Fold SD':>12}")
        for row in self.summary().itertuples(index=False):
            lines.append(
                f"  {row.metric:<14} {row.overall:>12.5g} "
                f"{row.fold_mean:>12.5g} {row.fold_std:>12.5g}"
            )
        converged = int(self.fold_scores["converged"].sum())
        singular = int(self.fold_scores["singular"].sum())
        lines.extend(
            [
                "",
                f"Converged folds: {converged}/{self.n_folds}",
                f"Singular folds: {singular}/{self.n_folds}",
            ]
        )
        return "\n".join(lines)


def _validate_score_inputs(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    for values in (y_true, y_pred, weights):
        if values is not None and (np.iscomplexobj(values) or np.ma.is_masked(values)):
            raise ValueError(
                "observed values, predictions, and weights must be unmasked real values"
            )
    observed = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if weights is None:
        score_weights = np.ones_like(observed)
    else:
        score_weights = np.asarray(weights, dtype=np.float64)

    if (
        observed.ndim != 1
        or predicted.shape != observed.shape
        or score_weights.shape != observed.shape
    ):
        raise ValueError("observed values, predictions, and weights must be aligned 1D arrays")
    if observed.size == 0:
        raise ValueError("scoring requires at least one observation")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("observed values and predictions must be finite")
    if not np.all(np.isfinite(score_weights)) or np.any(score_weights <= 0):
        raise ValueError("score weights must be finite and strictly positive")
    return observed, predicted, score_weights


def _difference_parts(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Retain finite differences even when subtraction exceeds float64 range."""
    with np.errstate(over="ignore"):
        difference = observed - predicted
    overflow = np.isinf(difference)
    if np.any(overflow):
        difference[overflow] = np.ldexp(observed[overflow], -1) - np.ldexp(predicted[overflow], -1)
    mantissa, exponent = np.frexp(difference)
    exponent[overflow] += 1
    return mantissa, exponent


def _sum_parts(values: NDArray[np.float64], exponents: NDArray[np.int32]) -> tuple[float, int]:
    """Sum binary-scaled terms without restoring their potentially extreme units."""
    active = values != 0.0
    if not np.any(active):
        return 0.0, 0
    exponent = int(np.max(exponents[active]))
    with np.errstate(under="ignore"):
        scaled = np.ldexp(values, exponents - exponent)
    return float(np.sum(scaled)), exponent


def _weighted_power_sum(
    values: tuple[NDArray[np.float64], NDArray[np.int32]],
    weights: tuple[NDArray[np.float64], NDArray[np.int32]],
    power: int,
) -> tuple[float, int]:
    mantissa, exponent = values
    weight_mantissa, weight_exponent = weights
    return _sum_parts(
        np.abs(mantissa) ** power * weight_mantissa, power * exponent + weight_exponent
    )


def _parts_ratio(
    numerator: tuple[float, int], denominator: tuple[float, int], *, root: bool = False
) -> float:
    value, shift = np.frexp(numerator[0] / denominator[0])
    exponent = int(shift) + numerator[1] - denominator[1]
    if root:
        value = np.sqrt(np.ldexp(value, exponent % 2))
        exponent //= 2
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(value, exponent))


def _fold_score_moments(values: NDArray[np.float64]) -> tuple[float, float]:
    """Summarize finite fold scores without squaring or summing their units."""
    if len(values) == 1:
        return float(values[0]), 0.0
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0
    with np.errstate(under="ignore"):
        mean = float(np.mean(values / scale)) * scale
    if np.all(values == values[0]):
        return mean, 0.0

    # Center before scaling to retain nearby values at large baselines.
    centered, exponents = _difference_parts(values, np.broadcast_to(values[0], values.shape))
    exponent = int(np.max(exponents[centered != 0.0]))
    with np.errstate(under="ignore"):
        scaled_centered = np.ldexp(centered, exponents - exponent)
    standard_deviation = float(np.std(scaled_centered, ddof=1))
    with np.errstate(over="ignore", under="ignore"):
        return mean, float(np.ldexp(standard_deviation, exponent))


def _weighted_error_scores(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    weights: NDArray[np.float64],
    names: Sequence[str],
) -> dict[str, float]:
    # Ordinary-sized inputs can use NumPy's direct reduction. These bounds
    # leave ample exponent headroom for squared errors, weighted sums, and
    # their ratio, even for arrays far larger than can fit in memory.
    with np.errstate(over="ignore"):
        difference = observed - predicted
    largest_error = np.max(np.abs(difference))
    if largest_error == 0.0:
        return dict.fromkeys(names, 0.0)
    squared = "mse" in names or "rmse" in names
    scores = {}
    if 1e-60 <= largest_error <= 1e60 and np.min(weights) >= 1e-60 and np.max(weights) <= 1e60:
        weight_sum = float(np.sum(weights))
        with np.errstate(under="ignore"):
            if squared:
                value = float(np.sum(np.square(difference) * weights)) / weight_sum
                if "mse" in names:
                    scores["mse"] = value
                if "rmse" in names:
                    scores["rmse"] = float(np.sqrt(value))
            if "mae" in names:
                scores["mae"] = float(np.sum(np.abs(difference) * weights)) / weight_sum
        return scores
    weight_parts = np.frexp(weights)
    weight_sum_parts = _sum_parts(*weight_parts)
    error_parts = _difference_parts(observed, predicted)
    if squared:
        total = _weighted_power_sum(error_parts, weight_parts, 2)
        if "mse" in names:
            scores["mse"] = _parts_ratio(total, weight_sum_parts)
        if "rmse" in names:
            scores["rmse"] = _parts_ratio(total, weight_sum_parts, root=True)
    if "mae" in names:
        total = _weighted_power_sum(error_parts, weight_parts, 1)
        scores["mae"] = _parts_ratio(total, weight_sum_parts)
    return scores


def weighted_mse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute mean-squared error with optional positive weights.

    Intermediate products are scaled to avoid overflow and underflow. An
    unrepresentable final score returns infinity or zero, respectively.
    """
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    return _weighted_error_scores(observed, predicted, score_weights, ("mse",))["mse"]


def weighted_rmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute root-mean-squared error without first forming the squared score."""
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    return _weighted_error_scores(observed, predicted, score_weights, ("rmse",))["rmse"]


def weighted_mae(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute mean absolute error with optional positive weights."""
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    return _weighted_error_scores(observed, predicted, score_weights, ("mae",))["mae"]


def weighted_r2(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute weighted coefficient of determination, independent of unit scales.

    Constant responses return one for exact predictions and zero otherwise.
    """
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    if np.all(observed == observed[0]):
        return 1.0 if np.array_equal(observed, predicted) else 0.0

    # Center at a heavily weighted observation before finding the mean, so
    # large baselines do not erase nearby representable differences. Keep
    # products in binary parts: normalizing weights alone can drop a tiny
    # weight whose large residual contributes substantially to the score.
    anchor = observed[int(np.argmax(score_weights))]
    with np.errstate(over="ignore"):
        centered_values = observed - anchor
        residual_values = observed - predicted
    largest_center = np.max(np.abs(centered_values))
    if (
        1e-60 <= largest_center <= 1e60
        and np.max(np.abs(residual_values)) <= 1e60
        and np.min(score_weights) >= 1e-60
        and np.max(score_weights) <= 1e60
    ):
        with np.errstate(under="ignore"):
            mean_value = np.average(centered_values, weights=score_weights)
            total_value = float(np.sum(score_weights * np.square(centered_values - mean_value)))
            residual_value = float(np.sum(score_weights * np.square(residual_values)))
        return 1 - residual_value / total_value
    centered, centered_exponent = _difference_parts(
        observed, np.broadcast_to(anchor, observed.shape)
    )
    weight_parts = np.frexp(score_weights)
    mean_sum, mean_power = _sum_parts(
        centered * weight_parts[0], centered_exponent + weight_parts[1]
    )
    weight_sum, weight_power = _sum_parts(*weight_parts)
    mean = mean_sum / weight_sum
    mean_exponent = mean_power - weight_power
    common_exponent = centered_exponent
    if mean != 0.0:
        common_exponent = np.maximum(centered_exponent, mean_exponent)
    with np.errstate(under="ignore"):
        deviations = np.ldexp(centered, centered_exponent - common_exponent) - np.ldexp(
            mean, mean_exponent - common_exponent
        )
    deviations, deviation_shift = np.frexp(deviations)
    total = _weighted_power_sum((deviations, common_exponent + deviation_shift), weight_parts, 2)
    residual = _weighted_power_sum(_difference_parts(observed, predicted), weight_parts, 2)
    if total[0] == 0.0:
        return 1.0 if residual[0] == 0.0 else float("-inf")
    return 1 - _parts_ratio(residual, total)


def _random_generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def make_folds(
    n_samples: int,
    cv: int = 5,
    *,
    groups: Any | None = None,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> tuple[CrossValidationFold, ...]:
    """Construct exhaustive case-level or observation-balanced grouped folds.

    When ``groups`` is supplied, every group is assigned to exactly one test
    fold. Groups are greedily assigned by decreasing size to the currently
    smallest fold, keeping observation counts balanced without scikit-learn.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be an integer")
    if isinstance(cv, bool) or not isinstance(cv, (int, np.integer)):
        raise TypeError("cv must be an integer")
    n_samples = int(n_samples)
    cv = int(cv)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if cv < 2:
        raise ValueError("cv must be at least 2")

    rng = _random_generator(random_state)
    if groups is None:
        if cv > n_samples:
            raise ValueError("cv cannot exceed the number of observations")
        order = np.arange(n_samples, dtype=np.intp)
        if shuffle:
            rng.shuffle(order)
        test_folds = [
            np.sort(chunk).astype(np.intp, copy=False) for chunk in np.array_split(order, cv)
        ]
    else:
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or len(group_values) != n_samples:
            raise ValueError("groups must be a 1D array with one value per observation")
        if bool(np.asarray(pd.isna(group_values)).any()):
            raise ValueError("groups cannot contain missing values")

        codes, unique_groups = pd.factorize(group_values, sort=False)
        n_groups = len(unique_groups)
        if cv > n_groups:
            raise ValueError("cv cannot exceed the number of unique groups")

        counts = np.bincount(codes, minlength=n_groups)
        group_order = np.arange(n_groups, dtype=np.intp)
        if shuffle:
            rng.shuffle(group_order)
        group_order = group_order[np.argsort(-counts[group_order], kind="stable")]

        fold_sizes = np.zeros(cv, dtype=np.int64)
        group_fold = np.empty(n_groups, dtype=np.int64)
        for group_code in group_order:
            fold = int(np.argmin(fold_sizes))
            group_fold[group_code] = fold
            fold_sizes[fold] += counts[group_code]
        row_folds = group_fold[codes]
        test_folds = [np.flatnonzero(row_folds == fold) for fold in range(cv)]

    all_rows = np.arange(n_samples, dtype=np.intp)
    folds: list[CrossValidationFold] = []
    for fold, test_indices in enumerate(test_folds):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_indices] = False
        folds.append(
            CrossValidationFold(
                fold=fold,
                train_indices=all_rows[train_mask],
                test_indices=np.asarray(test_indices, dtype=np.intp),
            )
        )
    return tuple(folds)


def _take_rows(data: Any, indices: NDArray[np.intp]) -> Any:
    if type(data).__module__.startswith("pandas"):
        return data.iloc[indices].copy()
    return data[indices.tolist()]


def _fold_indices(values: ArrayLike, n_samples: int, fold: int, kind: str) -> NDArray[np.intp]:
    """Validate positional indices before any potentially lossy integer cast."""
    if np.ma.is_masked(values):
        raise ValueError(f"fold {fold} {kind} indices must not contain masked values")
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"fold {fold} {kind} indices must be a nonempty 1D array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"fold {fold} {kind} indices must be integer row positions")
    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError(f"fold {fold} {kind} indices are outside the observation range")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"fold {fold} {kind} indices must not contain duplicate rows")
    result = np.array(indices, dtype=np.intp, copy=True)
    result.setflags(write=False)
    return result


def _explicit_folds(
    cv: Iterable[FoldSpec], n_samples: int, groups: Any | None
) -> tuple[CrossValidationFold, ...]:
    """Require one honest held-out prediction per observation before fitting."""
    if isinstance(cv, (str, bytes)):
        raise TypeError("cv must be an integer or an iterable of train/test splits")
    try:
        supplied = iter(cv)
    except TypeError:
        raise TypeError("cv must be an integer or an iterable of train/test splits") from None

    group_codes = None
    if groups is not None:
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or len(group_values) != n_samples:
            raise ValueError("groups must be a 1D array with one value per observation")
        if bool(np.asarray(pd.isna(group_values)).any()):
            raise ValueError("groups cannot contain missing values")
        group_codes, _ = pd.factorize(group_values, sort=False)

    membership = np.full(n_samples, -1, dtype=np.int64)
    folds: list[CrossValidationFold] = []
    for fold_number, split in enumerate(supplied):
        train: ArrayLike
        test: ArrayLike
        if isinstance(split, CrossValidationFold):
            train, test = split.train_indices, split.test_indices
        else:
            try:
                train, test = split
            except (TypeError, ValueError):
                raise ValueError("each cv split must contain train and test row indices") from None
        train_indices = _fold_indices(train, n_samples, fold_number, "train")
        test_indices = _fold_indices(test, n_samples, fold_number, "test")
        if np.intersect1d(train_indices, test_indices).size:
            raise ValueError(f"fold {fold_number} train and test rows must be disjoint")
        if np.any(membership[test_indices] >= 0):
            raise ValueError("cv test splits must not overlap; each observation is held out once")
        if (
            group_codes is not None
            and np.intersect1d(group_codes[train_indices], group_codes[test_indices]).size
        ):
            raise ValueError(f"fold {fold_number} train and test groups must be disjoint")
        membership[test_indices] = fold_number
        folds.append(CrossValidationFold(fold_number, train_indices, test_indices))

    if len(folds) < 2:
        raise ValueError("cv must contain at least two train/test splits")
    if np.any(membership < 0):
        raise ValueError("cv test splits must cover every observation exactly once")
    if group_codes is not None:
        first_fold = np.full(int(group_codes.max()) + 1, len(folds), dtype=np.int64)
        last_fold = np.full_like(first_fold, -1)
        np.minimum.at(first_fold, group_codes, membership)
        np.maximum.at(last_fold, group_codes, membership)
        if np.any(first_fold != last_fold):
            raise ValueError("cv test splits must hold out each whole group in a single fold")
    return tuple(folds)


def _validate_predictor_alignment(model: LmerResult | GlmerResult, frame: Any) -> None:
    """Prevent response ties from hiding changes to positional fit inputs."""
    predictors = model.formula.fixed_variables | model.formula.random_variables
    required = predictors | model.formula.grouping_factors
    columns = set(get_columns(frame))
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"cross-validation data is missing modeled columns: {missing}")

    stored = model.matrices.frame
    if stored is None:
        # Serialized or manually constructed results can lack their source frame.
        # In that case the fitted designs still establish observation alignment.
        if not np.array_equal(model._prediction_fixed_matrix(frame), model.matrices.X):
            raise ValueError("data fixed-effect values are not aligned with the fitted model")
        random_design, _ = model._prediction_random_matrix(frame, allow_new_levels=False)
        if (random_design != model.matrices.Z).nnz:
            raise ValueError("data random-effect values are not aligned with the fitted model")
    else:
        for name in sorted(required):
            if not np.array_equal(get_column_numpy(frame, name), get_column_numpy(stored, name)):
                raise ValueError(f"data column '{name}' is not aligned with the fitted model")

    for name in sorted(predictors):
        fitted_levels = model.matrices.category_levels.get(name)
        categorical = is_categorical_or_string(frame, name)
        if categorical != (fitted_levels is not None) or (
            fitted_levels is not None and get_categories(frame, name) != fitted_levels
        ):
            raise ValueError(
                f"data categorical encoding for '{name}' differs from the fitted model"
            )


def _stored_model_frame(model: LmerResult | GlmerResult) -> Any:
    frame = model.matrices.frame
    if frame is None:
        return model.model_frame()
    if hasattr(frame, "copy"):
        return frame.copy()
    if hasattr(frame, "clone"):
        return frame.clone()
    raise TypeError(f"unsupported stored model frame type: {type(frame).__name__}")


def _resolve_metrics(
    metrics: MetricSpec | Sequence[MetricSpec] | None,
    *,
    is_glmm: bool,
) -> tuple[tuple[str, MetricSpec], ...]:
    if metrics is None:
        requested: list[MetricSpec] = ["rmse", "deviance"] if is_glmm else ["rmse", "mae", "r2"]
    elif isinstance(metrics, str) or callable(metrics):
        requested = [metrics]
    else:
        requested = list(metrics)
    if not requested:
        raise ValueError("metrics must contain at least one scorer")

    resolved: list[tuple[str, MetricSpec]] = []
    seen: set[str] = set()
    for metric in requested:
        if isinstance(metric, str):
            name = metric.lower().replace("-", "_")
            if name == "r2_score":
                name = "r2"
            if name not in _BUILTIN_METRICS:
                choices = ", ".join(sorted(_BUILTIN_METRICS))
                raise ValueError(f"unknown metric '{metric}'; choose from: {choices}")
            if name == "deviance" and not is_glmm:
                raise ValueError("deviance scoring requires a generalized linear mixed model")
            scorer: MetricSpec = name
        elif callable(metric):
            name = getattr(metric, "__name__", "custom_metric")
            scorer = metric
        else:
            raise TypeError("each metric must be a supported name or callable")
        if name in _FOLD_METADATA:
            raise ValueError(f"metric name '{name}' is reserved for fold metadata")
        if name in seen:
            raise ValueError(f"duplicate metric name: {name}")
        seen.add(name)
        resolved.append((name, scorer))
    return tuple(resolved)


def _mean_deviance(
    family: Family,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    contributions = np.asarray(family.deviance_resids(y_true, y_pred, weights), dtype=np.float64)
    if contributions.shape != y_true.shape or not np.all(np.isfinite(contributions)):
        raise ValueError(
            "family deviance contributions must be finite and aligned with observations"
        )
    return float(np.sum(contributions) / np.sum(weights))


def _score_metrics(
    resolved: tuple[tuple[str, MetricSpec], ...],
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    weights: NDArray[np.float64],
    family: Family | None,
) -> dict[str, float]:
    error_names = [
        name
        for name, scorer in resolved
        if isinstance(scorer, str) and scorer in {"mse", "rmse", "mae"}
    ]
    error_scores = (
        _weighted_error_scores(*_validate_score_inputs(y_true, y_pred, weights), error_names)
        if error_names
        else {}
    )
    scores: dict[str, float] = {}
    for name, scorer in resolved:
        if name in error_scores:
            value = error_scores[name]
        elif scorer == "r2":
            value = weighted_r2(y_true, y_pred, weights)
        elif scorer == "deviance":
            assert family is not None
            value = _mean_deviance(family, y_true, y_pred, weights)
        else:
            assert callable(scorer)
            value = float(scorer(y_true, y_pred, weights))
        if not np.isfinite(value):
            raise ValueError(f"metric '{name}' returned a non-finite value")
        scores[name] = value
    return scores


def _fit_fold(
    model: LmerResult | GlmerResult,
    train_data: Any,
    train_weights: NDArray[np.float64],
    train_offset: NDArray[np.float64],
    fit_kwargs: dict[str, Any],
) -> LmerResult | GlmerResult:
    from mixedlm.models.glmer import glmer
    from mixedlm.models.lmer import LmerResult, lmer

    formula = str(model.formula)
    if isinstance(model, LmerResult):
        kwargs: dict[str, Any] = {"REML": model.REML, "contrasts": model.matrices.contrasts}
        kwargs.update(fit_kwargs)
        return lmer(
            formula,
            train_data,
            weights=train_weights,
            offset=train_offset,
            na_action="fail",
            **kwargs,
        )

    kwargs = {
        "nAGQ": model.nAGQ,
        "control": model._refit_control(),
        "contrasts": model.matrices.contrasts,
    }
    kwargs.update(fit_kwargs)
    train_data = _restore_binomial_factor(
        train_data, model.formula.response, model.matrices.response_levels
    )
    return glmer(
        formula,
        train_data,
        family=deepcopy(model.family),
        weights=train_weights,
        offset=train_offset,
        na_action="fail",
        **kwargs,
    )


def cross_validate(
    model: LmerResult | GlmerResult,
    data: Any | None = None,
    *,
    cv: int | Iterable[FoldSpec] = 5,
    group: str | None = None,
    metrics: MetricSpec | Sequence[MetricSpec] | None = None,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
    re_form: str | None = "auto",
    n_jobs: int = 1,
    fit_kwargs: dict[str, Any] | None = None,
) -> CrossValidationResult:
    """Refit and score an LMM or GLMM on exhaustive out-of-fold predictions.

    Case-level folds retain conditional random-effect predictions for group
    levels seen in training. When ``group`` is supplied, whole groups are held
    out and ``re_form='auto'`` uses fixed-effect predictions, matching
    generalization to unseen clusters.

    Custom metric callables receive ``(y_true, y_pred, weights)`` arrays and
    must return one finite scalar. Original weights are preserved for refits
    and scoring; original offsets are preserved for refits and held-out
    predictions. Set ``n_jobs`` above one to fit independent folds concurrently
    with threads. Categorical contrast coding and grouped binomial trial counts
    are retained from the fitted model.

    ``cv`` accepts a fold count or an iterable of ``(train_indices,
    test_indices)`` pairs or :class:`CrossValidationFold` objects. Explicit
    indices are zero-based row positions. Their test sets must partition all
    fitted observations; train sets may exclude additional rows, for example
    a buffer around a held-out time block. Train/test overlap and group leakage
    are rejected before fitting. ``shuffle`` and ``random_state`` apply only
    to generated folds. Supplied data must preserve modeled columns, row order,
    and categorical encoding so the original weights and offsets stay aligned.
    """
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult

    if not isinstance(model, LmerResult | GlmerResult):
        raise TypeError("cross_validate supports fitted linear and generalized linear mixed models")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be an integer")
    n_jobs = int(n_jobs)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")

    frame = _stored_model_frame(model) if data is None else ensure_dataframe(data)
    n_samples = dataframe_length(frame)
    if n_samples != model.nobs():
        raise ValueError(
            "data must contain the same aligned observations used by the fitted model; "
            "omit data to use the stored clean model frame"
        )
    if group is not None and group not in get_columns(frame):
        raise ValueError(f"group column '{group}' is not present in the cross-validation data")

    is_glmm = isinstance(model, GlmerResult)
    from mixedlm.families.binomial import Binomial

    is_binomial = isinstance(model, GlmerResult) and isinstance(model.family, Binomial)
    y: NDArray[np.float64] = np.asarray(model.matrices.y, dtype=np.float64)
    raw_response, trials = _build_response(
        model.formula,
        frame,
        grouped_binomial=is_binomial,
        response_levels=model.matrices.response_levels,
    )
    response = np.asarray(raw_response, dtype=np.float64)
    if response.shape != y.shape or not np.array_equal(response, y):
        raise ValueError("data response values are not aligned with the fitted model")
    if trials is not None and (
        model.matrices.trials is None or not np.array_equal(trials, model.matrices.trials)
    ):
        raise ValueError("data binomial trial counts are not aligned with the fitted model")
    if data is not None:
        _validate_predictor_alignment(model, frame)

    fit_options = {} if fit_kwargs is None else dict(fit_kwargs)
    conflicts = sorted(_RESERVED_FIT_ARGUMENTS.intersection(fit_options))
    if conflicts:
        raise ValueError(f"fit_kwargs cannot override reserved arguments: {', '.join(conflicts)}")

    resolved_metrics = _resolve_metrics(metrics, is_glmm=is_glmm)
    group_values = None if group is None else get_column_numpy(frame, group)
    if isinstance(cv, (int, np.integer)):
        folds = make_folds(
            n_samples,
            cv,
            groups=group_values,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        folds = _explicit_folds(cv, n_samples, group_values)

    weights: NDArray[np.float64] = np.asarray(model.weights(), dtype=np.float64)
    # Grouped binomial matrices store prior weights multiplied by trial counts.
    # Refitting the count formula applies that multiplication again, so pass
    # the original prior weights while retaining effective weights for scores.
    fit_weights = weights if trials is None else weights / trials
    offset: NDArray[np.float64] = np.asarray(model.offset(), dtype=np.float64)
    family = model.family if isinstance(model, GlmerResult) else None
    predictions = np.full(n_samples, np.nan, dtype=np.float64)
    fold_ids = np.full(n_samples, -1, dtype=np.int64)
    prediction_re_form = ("~0" if group is not None else None) if re_form == "auto" else re_form

    def fit_and_predict(
        fold: CrossValidationFold,
    ) -> tuple[CrossValidationFold, NDArray[np.float64], bool, bool]:
        train_data = _take_rows(frame, fold.train_indices)
        test_data = _take_rows(frame, fold.test_indices)
        fold_model = _fit_fold(
            model,
            train_data,
            fit_weights[fold.train_indices],
            offset[fold.train_indices],
            fit_options,
        )
        if isinstance(fold_model, GlmerResult):
            raw_predictions = fold_model.predict(
                test_data,
                type="response",
                re_form=prediction_re_form,
                allow_new_levels=True,
                offset=offset[fold.test_indices],
            )
        else:
            raw_predictions = fold_model.predict(
                test_data,
                re_form=prediction_re_form,
                allow_new_levels=True,
                offset=offset[fold.test_indices],
            )
        fold_predictions: NDArray[np.float64] = np.asarray(raw_predictions, dtype=np.float64)
        if fold_predictions.shape != fold.test_indices.shape:
            raise ValueError("fold predictions are not aligned with held-out observations")
        return (
            fold,
            fold_predictions,
            bool(fold_model.converged),
            bool(fold_model.isSingular()),
        )

    if n_jobs == 1:
        fitted_folds = [fit_and_predict(fold) for fold in folds]
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(folds))) as executor:
            fitted_folds = list(executor.map(fit_and_predict, folds))

    records: list[dict[str, float | int | bool]] = []
    for fold, fold_predictions, converged, singular in fitted_folds:
        predictions[fold.test_indices] = fold_predictions
        fold_ids[fold.test_indices] = fold.fold
        fold_metrics = _score_metrics(
            resolved_metrics,
            y[fold.test_indices],
            fold_predictions,
            weights[fold.test_indices],
            family,
        )
        record: dict[str, float | int | bool] = {
            "fold": fold.fold,
            "n_train": len(fold.train_indices),
            "n_test": len(fold.test_indices),
            "converged": converged,
            "singular": singular,
        }
        record.update(fold_metrics)
        records.append(record)

    if not np.all(np.isfinite(predictions)) or np.any(fold_ids < 0):
        raise RuntimeError("cross-validation did not produce one finite prediction per observation")
    overall_scores = _score_metrics(resolved_metrics, y, predictions, weights, family)
    return CrossValidationResult(
        scores=overall_scores,
        fold_scores=pd.DataFrame.from_records(records),
        predictions=predictions,
        fold_ids=fold_ids,
        folds=folds,
        group=group,
        metric_names=tuple(name for name, _ in resolved_metrics),
    )


__all__ = [
    "CrossValidationFold",
    "CrossValidationResult",
    "cross_validate",
    "make_folds",
    "weighted_mae",
    "weighted_mse",
    "weighted_r2",
    "weighted_rmse",
]
