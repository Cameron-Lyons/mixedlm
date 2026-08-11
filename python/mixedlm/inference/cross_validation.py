"""Case-level and grouped cross-validation for mixed-effects models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mixedlm.utils.dataframe import (
    dataframe_length,
    ensure_dataframe,
    get_column_numpy,
    get_columns,
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


@dataclass(frozen=True)
class CrossValidationFold:
    """Train and test row positions for one cross-validation fold."""

    fold: int
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]


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
            rows.append(
                {
                    "metric": metric,
                    "overall": self.scores[metric],
                    "fold_mean": float(np.mean(values)),
                    "fold_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
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


def weighted_mse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute mean-squared error with optional positive weights."""
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    return float(np.average(np.square(observed - predicted), weights=score_weights))


def weighted_rmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute root-mean-squared error with optional positive weights."""
    return float(np.sqrt(weighted_mse(y_true, y_pred, weights)))


def weighted_mae(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute mean absolute error with optional positive weights."""
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    return float(np.average(np.abs(observed - predicted), weights=score_weights))


def weighted_r2(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    """Compute weighted coefficient of determination."""
    observed, predicted, score_weights = _validate_score_inputs(y_true, y_pred, weights)
    mean = float(np.average(observed, weights=score_weights))
    residual_sum = float(np.sum(score_weights * np.square(observed - predicted)))
    total_sum = float(np.sum(score_weights * np.square(observed - mean)))
    if total_sum <= np.finfo(np.float64).eps:
        return 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
    return 1 - residual_sum / total_sum


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
    scores: dict[str, float] = {}
    for name, scorer in resolved:
        if scorer == "rmse":
            value = weighted_rmse(y_true, y_pred, weights)
        elif scorer == "mse":
            value = weighted_mse(y_true, y_pred, weights)
        elif scorer == "mae":
            value = weighted_mae(y_true, y_pred, weights)
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
        kwargs: dict[str, Any] = {"REML": model.REML}
        kwargs.update(fit_kwargs)
        return lmer(
            formula,
            train_data,
            weights=train_weights,
            offset=train_offset,
            na_action="fail",
            **kwargs,
        )

    kwargs = {"nAGQ": model.nAGQ}
    kwargs.update(fit_kwargs)
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
    cv: int = 5,
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
    must return one finite scalar. Original model weights and offsets are
    subset and preserved for every refit. Set ``n_jobs`` above one to fit
    independent folds concurrently with threads.
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

    y: NDArray[np.float64] = np.asarray(model.matrices.y, dtype=np.float64)
    response = np.asarray(
        get_column_numpy(frame, model.formula.response, dtype=float), dtype=np.float64
    )
    if response.shape != y.shape or not np.array_equal(response, y):
        raise ValueError("data response values are not aligned with the fitted model")

    fit_options = {} if fit_kwargs is None else dict(fit_kwargs)
    conflicts = sorted(_RESERVED_FIT_ARGUMENTS.intersection(fit_options))
    if conflicts:
        raise ValueError(f"fit_kwargs cannot override reserved arguments: {', '.join(conflicts)}")

    is_glmm = isinstance(model, GlmerResult)
    resolved_metrics = _resolve_metrics(metrics, is_glmm=is_glmm)
    group_values = None if group is None else get_column_numpy(frame, group)
    folds = make_folds(
        n_samples,
        cv,
        groups=group_values,
        shuffle=shuffle,
        random_state=random_state,
    )

    weights: NDArray[np.float64] = np.asarray(model.weights(), dtype=np.float64)
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
            weights[fold.train_indices],
            offset[fold.train_indices],
            fit_options,
        )
        if isinstance(fold_model, GlmerResult):
            raw_predictions = fold_model.predict(
                test_data,
                type="response",
                re_form=prediction_re_form,
                allow_new_levels=True,
            )
        else:
            raw_predictions = fold_model.predict(
                test_data,
                re_form=prediction_re_form,
                allow_new_levels=True,
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
