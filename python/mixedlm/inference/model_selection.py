from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ModelSelectionResult:
    """Information-criterion ranking for a collection of fitted models."""

    models: tuple[Any, ...]
    model_names: tuple[str, ...]
    criterion: str
    likelihood: str
    nobs: NDArray[np.int64]
    df: NDArray[np.int64]
    loglik: NDArray[np.float64]
    aic: NDArray[np.float64]
    aicc: NDArray[np.float64]
    bic: NDArray[np.float64]
    delta: NDArray[np.float64]
    relative_likelihood: NDArray[np.float64]
    weight: NDArray[np.float64]
    cumulative_weight: NDArray[np.float64]

    @property
    def best_model(self) -> Any:
        """Return the best-supported fitted model under the selected criterion."""
        return self.models[0]

    @property
    def best_name(self) -> str:
        """Return the display name of the best-supported model."""
        return self.model_names[0]

    def evidence_set(self, threshold: float = 0.95) -> tuple[Any, ...]:
        """Return the smallest leading set whose cumulative weight reaches ``threshold``."""
        end = self._evidence_set_end(threshold)
        return self.models[:end]

    def evidence_names(self, threshold: float = 0.95) -> tuple[str, ...]:
        """Return names in the smallest leading evidence set."""
        end = self._evidence_set_end(threshold)
        return self.model_names[:end]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the ranked comparison as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "model": self.model_names,
                "nobs": self.nobs,
                "df": self.df,
                "logLik": self.loglik,
                "AIC": self.aic,
                "AICc": self.aicc,
                "BIC": self.bic,
                "delta": self.delta,
                "relative_likelihood": self.relative_likelihood,
                "weight": self.weight,
                "cumulative_weight": self.cumulative_weight,
            }
        )

    def __len__(self) -> int:
        return len(self.models)

    def __str__(self) -> str:
        lines = [f"Model selection by {self.criterion} ({self.likelihood})", ""]
        lines.append(
            f"{'Model':<24} {'df':>4} {'logLik':>10} {self.criterion:>10} "
            f"{'delta':>9} {'weight':>9} {'cum.weight':>11}"
        )
        lines.append("-" * 84)
        values = self._criterion_values()
        for i, name in enumerate(self.model_names):
            lines.append(
                f"{name[:24]:<24} {self.df[i]:>4d} {self.loglik[i]:>10.3f} "
                f"{values[i]:>10.3f} {self.delta[i]:>9.3f} {self.weight[i]:>9.3f} "
                f"{self.cumulative_weight[i]:>11.3f}"
            )
        return "\n".join(lines)

    def _criterion_values(self) -> NDArray[np.float64]:
        if self.criterion == "AIC":
            return self.aic
        if self.criterion == "AICc":
            return self.aicc
        return self.bic

    def _evidence_set_end(self, threshold: float) -> int:
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be finite and in the interval (0, 1]")
        index = int(np.searchsorted(self.cumulative_weight, threshold, side="left"))
        return min(index + 1, len(self))


@dataclass(frozen=True)
class _ModelStats:
    model: Any
    name: str
    kind: str
    response: str | None
    family_signature: tuple[Any, ...] | None
    response_values: NDArray[np.float64] | None
    prior_weights: NDArray[np.float64] | None
    fixed_design: tuple[NDArray[np.float64], NDArray[np.float64]] | None
    reml: bool
    nobs: int
    df: int
    loglik: float
    aic: float
    bic: float


def model_selection(
    *models: Any,
    names: Sequence[str] | None = None,
    criterion: str = "AICc",
    allow_reml: bool = False,
) -> ModelSelectionResult:
    """Rank fitted mixed models by AIC, small-sample AIC, or BIC.

    The returned models are ordered from strongest to weakest support. Delta
    values, relative likelihoods, normalized weights, and cumulative weights
    are computed from the selected criterion in one vectorized pass.

    Parameters
    ----------
    *models : fitted mixed models
        Two or more linear, generalized, or nonlinear mixed-model results.
        Models must use the same response, likelihood family, and observations.
    names : sequence of str, optional
        Display names. Formula strings are used when available.
    criterion : {"AIC", "AICc", "BIC"}, default "AICc"
        Ranking criterion. AICc is infinite when ``n <= df + 1``.
    allow_reml : bool, default False
        Permit comparisons where every model uses REML. REML criteria are only
        comparable when the fixed-effects specification is unchanged.

    Returns
    -------
    ModelSelectionResult
        Ranked models and information-criterion evidence measures.
    """
    if len(models) < 2:
        raise ValueError("model_selection requires at least two fitted models")

    selected_criterion = _normalize_criterion(criterion)
    model_names = _resolve_names(models, names)
    stats = tuple(
        _extract_model_stats(model, name) for model, name in zip(models, model_names, strict=True)
    )
    _validate_comparability(stats, allow_reml=allow_reml)

    nobs = np.fromiter((item.nobs for item in stats), dtype=np.int64, count=len(stats))
    df = np.fromiter((item.df for item in stats), dtype=np.int64, count=len(stats))
    loglik = np.fromiter((item.loglik for item in stats), dtype=np.float64, count=len(stats))
    aic = np.fromiter((item.aic for item in stats), dtype=np.float64, count=len(stats))
    bic = np.fromiter((item.bic for item in stats), dtype=np.float64, count=len(stats))

    denominator = nobs - df - 1
    correction = np.full(len(stats), np.inf, dtype=np.float64)
    valid = denominator > 0
    correction[valid] = 2.0 * df[valid] * (df[valid] + 1) / denominator[valid]
    aicc = aic + correction

    if selected_criterion == "AIC":
        values = aic
    elif selected_criterion == "AICc":
        values = aicc
    else:
        values = bic

    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(
            f"No model has a finite {selected_criterion}; use AIC or BIC for this sample size"
        )

    best = float(np.min(values[finite]))
    delta = values - best
    relative_likelihood = np.zeros(len(stats), dtype=np.float64)
    relative_likelihood[finite] = np.exp(-0.5 * delta[finite])
    weight = relative_likelihood / np.sum(relative_likelihood)
    order = np.argsort(values, kind="stable")

    sorted_weight = weight[order]
    return ModelSelectionResult(
        models=tuple(stats[i].model for i in order),
        model_names=tuple(stats[i].name for i in order),
        criterion=selected_criterion,
        likelihood="REML" if stats[0].reml else "ML",
        nobs=nobs[order],
        df=df[order],
        loglik=loglik[order],
        aic=aic[order],
        aicc=aicc[order],
        bic=bic[order],
        delta=delta[order],
        relative_likelihood=relative_likelihood[order],
        weight=sorted_weight,
        cumulative_weight=np.cumsum(sorted_weight),
    )


def _extract_model_stats(model: Any, name: str) -> _ModelStats:
    kind = _model_kind(model)
    try:
        df_value, aic_value = model.extractAIC()
        loglik_result = model.logLik()
        bic_value = model.BIC()
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "Each input must be a fitted mixed model exposing extractAIC(), logLik(), and BIC()"
        ) from exc

    nobs_value = getattr(loglik_result, "nobs", None)
    if nobs_value is None:
        nobs_method = getattr(model, "nobs", None)
        if callable(nobs_method):
            nobs_value = nobs_method()
        elif hasattr(model, "matrices"):
            nobs_value = model.matrices.n_obs
        elif hasattr(model, "y"):
            nobs_value = len(model.y)
        else:
            raise TypeError("Could not determine the number of observations for a model")

    loglik_value = getattr(loglik_result, "value", loglik_result)
    try:
        nobs = int(nobs_value)
        df = int(df_value)
        loglik = float(loglik_value)
        aic = float(aic_value)
        bic = float(bic_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Model information criteria must be finite numeric scalars") from exc

    if nobs <= 0:
        raise ValueError("Model observation counts must be positive")
    if df <= 0 or df != float(df_value):
        raise ValueError("Model parameter counts must be positive integers")
    if not np.all(np.isfinite([loglik, aic, bic])):
        raise ValueError("Model log-likelihood, AIC, and BIC must be finite")

    response_values, prior_weights = _comparison_data(model)
    if response_values is not None and len(response_values) != nobs:
        raise ValueError(
            f"Model response has length {len(response_values)}, expected {nobs} observations"
        )
    if prior_weights is not None and len(prior_weights) != nobs:
        raise ValueError(f"Model weights have length {len(prior_weights)}, expected {nobs}")
    reml = bool(getattr(loglik_result, "REML", getattr(model, "REML", False)))
    return _ModelStats(
        model=model,
        name=name,
        kind=kind,
        response=_response_name(model),
        family_signature=_family_signature(model) if kind == "GLMM" else None,
        response_values=response_values,
        prior_weights=prior_weights,
        fixed_design=_fixed_design(model),
        reml=reml,
        nobs=nobs,
        df=df,
        loglik=loglik,
        aic=aic,
        bic=bic,
    )


def _validate_comparability(stats: tuple[_ModelStats, ...], allow_reml: bool) -> None:
    nobs = {item.nobs for item in stats}
    if len(nobs) != 1:
        raise ValueError(f"Models have different observation counts: {sorted(nobs)}")

    kinds = {item.kind for item in stats}
    if len(kinds) != 1:
        raise ValueError(f"Models use different likelihood classes: {sorted(kinds)}")

    responses = {item.response for item in stats if item.response is not None}
    if len(responses) > 1:
        raise ValueError(f"Models use different responses: {sorted(responses)}")

    _validate_equal_arrays(stats, "response_values", "response observations")
    _validate_equal_arrays(stats, "prior_weights", "prior weights")

    family_signatures = {
        item.family_signature for item in stats if item.family_signature is not None
    }
    if len(family_signatures) > 1:
        raise ValueError("Generalized models must use the same family, link, and family parameters")

    reml_values = {item.reml for item in stats}
    if len(reml_values) > 1:
        raise ValueError("ML and REML likelihood criteria cannot be compared")
    if True in reml_values and not allow_reml:
        raise ValueError(
            "REML criteria are not generally comparable across fixed-effects models; "
            "fit with REML=False or set allow_reml=True when fixed effects are identical"
        )
    if True in reml_values:
        designs = [item.fixed_design for item in stats]
        if any(design is None for design in designs):
            raise ValueError("Cannot verify identical fixed-effects designs for REML comparison")
        reference = designs[0]
        assert reference is not None
        reference_x, reference_offset = reference
        for design in designs[1:]:
            assert design is not None
            x, offset = design
            if not _arrays_equal(reference_x, x) or not _arrays_equal(reference_offset, offset):
                raise ValueError("REML model selection requires identical fixed-effects designs")


def _model_kind(model: Any) -> str:
    for method, label in (("isLMM", "LMM"), ("isGLMM", "GLMM"), ("isNLMM", "NLMM")):
        flag = getattr(model, method, None)
        if callable(flag) and flag():
            return label
    raise TypeError("Expected a fitted linear, generalized, or nonlinear mixed model")


def _response_name(model: Any) -> str | None:
    formula = getattr(model, "formula", None)
    response = getattr(formula, "response", None)
    if response is not None:
        return str(response)
    response = getattr(model, "_y_var", None)
    return str(response) if response is not None else None


def _comparison_data(
    model: Any,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    matrices = getattr(model, "matrices", None)
    if matrices is not None and hasattr(matrices, "y"):
        response = np.asarray(matrices.y, dtype=np.float64).reshape(-1)
        weights = np.asarray(matrices.weights, dtype=np.float64).reshape(-1)
        return response, weights

    response_value = getattr(model, "y", None)
    if response_value is None:
        return None, None
    response = np.asarray(response_value, dtype=np.float64).reshape(-1)
    weights_value = getattr(model, "_weights", None)
    weights = (
        np.ones(len(response), dtype=np.float64)
        if weights_value is None
        else np.asarray(weights_value, dtype=np.float64).reshape(-1)
    )
    return response, weights


def _fixed_design(
    model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    matrices = getattr(model, "matrices", None)
    if matrices is not None and hasattr(matrices, "X"):
        x = np.asarray(matrices.X, dtype=np.float64)
        offset = np.asarray(matrices.offset, dtype=np.float64).reshape(-1)
        return x, offset

    design = getattr(model, "fixed_design", None)
    if design is None:
        return None
    x = np.asarray(design, dtype=np.float64)
    return x, np.zeros(x.shape[0], dtype=np.float64)


def _validate_equal_arrays(
    stats: tuple[_ModelStats, ...],
    attribute: str,
    label: str,
) -> None:
    values = [getattr(item, attribute) for item in stats]
    available = [value for value in values if value is not None]
    if not available:
        return
    if len(available) != len(values):
        raise ValueError(f"Could not verify matching {label} for every model")
    reference = available[0]
    if any(not _arrays_equal(reference, value) for value in available[1:]):
        raise ValueError(f"Models use different {label}")


def _arrays_equal(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=True))


def _family_signature(model: Any) -> tuple[Any, ...]:
    from mixedlm.families.custom import QuasiFamily

    family = model.family
    if isinstance(family, QuasiFamily):
        raise ValueError("Information criteria are not defined for quasi-likelihood models")
    parameters = tuple(
        sorted(
            (key, _signature_value(value)) for key, value in vars(family).items() if key != "link"
        )
    )
    return type(family), type(family.link), parameters


def _signature_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def _normalize_criterion(criterion: str) -> str:
    normalized = criterion.strip().upper().replace("_", "")
    if normalized == "AICC":
        return "AICc"
    if normalized in ("AIC", "BIC"):
        return normalized
    raise ValueError("criterion must be 'AIC', 'AICc', or 'BIC'")


def _resolve_names(models: tuple[Any, ...], names: Sequence[str] | None) -> tuple[str, ...]:
    if names is not None:
        if isinstance(names, (str, bytes)):
            raise ValueError("names must be a sequence with one label per model")
        if len(names) != len(models):
            raise ValueError(f"names has length {len(names)}, expected {len(models)}")
        resolved = tuple(str(name).strip() for name in names)
        if any(not name for name in resolved):
            raise ValueError("Model names must not be empty")
        if len(set(resolved)) != len(resolved):
            raise ValueError("Model names must be unique")
        return resolved

    bases = []
    for index, model in enumerate(models, start=1):
        formula = getattr(model, "formula", None)
        bases.append(str(formula) if formula is not None else f"Model {index}")

    totals: dict[str, int] = {}
    for base in bases:
        totals[base] = totals.get(base, 0) + 1

    counts: dict[str, int] = {}
    generated: list[str] = []
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
        occurrence = counts[base]
        generated.append(base if totals[base] == 1 else f"{base} [{occurrence}]")
    return tuple(generated)
