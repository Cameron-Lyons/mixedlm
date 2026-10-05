from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mixedlm.estimation.reml import DevianceComponents
    from mixedlm.inference.allfit import AllFitResult
    from mixedlm.inference.bootstrap import BootstrapResult
    from mixedlm.inference.drop1 import Drop1Result
    from mixedlm.inference.profile_types import ProfileResult
    from mixedlm.models.control import LmerControl
    from mixedlm.utils.random import RandomSeed

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices
from mixedlm.models.lmer_types import LogLik as LogLik
from mixedlm.models.lmer_types import ModelTerms as ModelTerms
from mixedlm.models.lmer_types import PredictResult as PredictResult
from mixedlm.models.lmer_types import RanefResult as RanefResult
from mixedlm.models.lmer_types import RePCA as RePCA
from mixedlm.models.lmer_types import RePCAGroup as RePCAGroup
from mixedlm.models.lmer_types import VarCorrGroup as VarCorrGroup
from mixedlm.models.result_mixin import MerResultMixin
from mixedlm.models.shared_utils import (
    _RandomEffectFactor,
    dense_quadratic_form_diagonal,
    symmetric_inverse,
)
from mixedlm.utils import _format_pvalue, _get_signif_code
from mixedlm.utils.validation import _validate_confidence_level


@dataclass
class _WeightedProjection:
    """Reusable weighted mixed-model projection factors."""

    sqrt_weights: NDArray[np.float64]
    weighted_X: NDArray[np.float64]
    weighted_Z: sparse.csc_matrix
    lambda_matrix: sparse.csc_matrix | None
    random_factor: _RandomEffectFactor | None
    spherical_cross: NDArray[np.float64]
    random_fixed_map: NDArray[np.float64]
    XtVinvX: NDArray[np.float64]

    @property
    def L_V(self) -> NDArray[np.float64] | None:
        return self.random_factor.cholesky if self.random_factor is not None else None

    @cached_property
    def RZX(self) -> NDArray[np.float64]:
        if self.L_V is None:
            return self.spherical_cross.copy()
        return linalg.solve_triangular(self.L_V, self.spherical_cross, lower=True)


@dataclass
class VarCorr:
    groups: dict[str, VarCorrGroup]
    residual: float

    def __str__(self) -> str:
        lines = ["Random effects:"]
        lines.append(f" {'Groups':<11} {'Name':<12} {'Variance':>10} {'Std.Dev.':>10} {'Corr':>6}")
        for group_name, group in self.groups.items():
            for i, term in enumerate(group.term_names):
                grp = group_name if i == 0 else ""
                var = group.variance[term]
                sd = group.stddev[term]
                if i == 0 or group.corr is None:
                    lines.append(f" {grp:<11} {term:<12} {var:>10.4f} {sd:>10.4f}")
                else:
                    corr_vals = " ".join(f"{group.corr[i, j]:>6.2f}" for j in range(i))
                    lines.append(f" {grp:<11} {term:<12} {var:>10.4f} {sd:>10.4f} {corr_vals}")
        resid_sd = np.sqrt(self.residual)
        lines.append(f" {'Residual':<11} {'':<12} {self.residual:>10.4f} {resid_sd:>10.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_groups = len(self.groups)
        return f"VarCorr({n_groups} groups, residual={self.residual:.4f})"

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {name: group.variance for name, group in self.groups.items()}

    def get_cov(self, group: str) -> NDArray[np.floating]:
        return self.groups[group].cov

    def get_corr(self, group: str) -> NDArray[np.floating] | None:
        return self.groups[group].corr


def _warn_deprecated_alias(name: str, replacement: str) -> None:
    warnings.warn(
        f"LmerResult.{name} is deprecated and will be removed in a future release; "
        f"use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass
class LmerResult(MerResultMixin):
    _IS_LMM: ClassVar[bool] = True
    _HAS_SIGMA: ClassVar[bool] = True
    formula: Formula
    matrices: ModelMatrices
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    sigma: float
    u: NDArray[np.floating]
    deviance: float
    REML: bool
    converged: bool
    n_iter: int
    gradient_norm: float | None = None
    at_boundary: bool = False
    message: str = ""
    function_evals: int = 0
    optimizer: str = ""
    # The fitting controls, reused by formula-based refits such as drop1 and allFit.
    control: LmerControl | None = None

    @property
    def fe_params(self) -> NDArray[np.floating]:
        """Deprecated alias for ``beta``."""
        _warn_deprecated_alias("fe_params", "beta")
        return self.beta

    @property
    def re_params(self) -> NDArray[np.floating]:
        """Deprecated alias for ``theta``."""
        _warn_deprecated_alias("re_params", "theta")
        return self.theta

    @property
    def resid(self) -> NDArray[np.floating]:
        """Deprecated alias for ``residuals()``."""
        _warn_deprecated_alias("resid", "residuals()")
        return self.residuals()

    @property
    def fittedvalues(self) -> NDArray[np.floating]:
        """Deprecated alias for ``fitted()``."""
        _warn_deprecated_alias("fittedvalues", "fitted()")
        return self.fitted()

    @cached_property
    def _weighted_projection(self) -> _WeightedProjection:
        """Factor the weighted penalized least-squares system once."""
        sqrt_weights = np.sqrt(self.matrices.weights)
        weighted_X = sqrt_weights[:, None] * self.matrices.X
        weighted_Z = self.matrices.Z.multiply(sqrt_weights[:, None]).tocsc()
        q = self.matrices.n_random

        if q == 0:
            information = weighted_X.T @ weighted_X
            return _WeightedProjection(
                sqrt_weights=sqrt_weights,
                weighted_X=weighted_X,
                weighted_Z=weighted_Z,
                lambda_matrix=None,
                random_factor=None,
                spherical_cross=np.zeros((0, self.matrices.n_fixed), dtype=np.float64),
                random_fixed_map=np.zeros((0, self.matrices.n_fixed), dtype=np.float64),
                XtVinvX=np.asarray(information, dtype=np.float64),
            )

        lambda_matrix = _build_lambda(self.theta, self.matrices.random_structures)
        ZtWZ = weighted_Z.T @ weighted_Z
        V_factor = lambda_matrix.T @ ZtWZ @ lambda_matrix + sparse.eye(q, format="csc")
        random_factor = _RandomEffectFactor(V_factor)

        ZtWX = weighted_Z.T @ weighted_X
        spherical_cross = np.asarray(lambda_matrix.T @ ZtWX)
        random_fixed_map, correction = random_factor.solve_with_crossproduct(spherical_cross)
        information = weighted_X.T @ weighted_X - correction
        information = (information + information.T) / 2.0

        return _WeightedProjection(
            sqrt_weights=sqrt_weights,
            weighted_X=weighted_X,
            weighted_Z=weighted_Z,
            lambda_matrix=lambda_matrix,
            random_factor=random_factor,
            spherical_cross=spherical_cross,
            random_fixed_map=random_fixed_map,
            XtVinvX=np.asarray(information, dtype=np.float64),
        )

    def _compute_condVar(
        self, include_cov: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        from mixedlm.utils.variance import _conditional_variance_blocks

        q = self.matrices.n_random
        if q == 0:
            return {}

        # Reuse the factor cached for vcov and prediction intervals.
        projection = self._weighted_projection
        return _conditional_variance_blocks(
            projection.random_factor,
            projection.lambda_matrix,
            self.matrices.random_structures,
            scale=self.sigma**2,
            include_cov=include_cov,
        )

    @cached_property
    def _fitted_values(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def fitted(self, na_expand: bool = True) -> NDArray[np.floating]:
        """Get fitted values.

        Parameters
        ----------
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Fitted values.
        """
        values = self._fitted_values
        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(values)
        return values

    def residuals(self, type: str = "response", na_expand: bool = True) -> NDArray[np.floating]:
        """Get residuals.

        Parameters
        ----------
        type : str, default "response"
            Type of residuals: "response" or "pearson".
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Residuals.
        """
        fitted = self._fitted_values
        if type == "response":
            resid = self.matrices.y - fitted
        elif type == "pearson":
            resid = np.sqrt(self.matrices.weights) * (self.matrices.y - fitted) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(resid)
        return resid

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        re_form: str | None = None,
        allow_new_levels: bool = False,
        se_fit: bool = False,
        interval: str = "none",
        level: float = 0.95,
        offset: ArrayLike | str | None = None,
        weights: ArrayLike | str | None = None,
    ) -> NDArray[np.floating] | PredictResult:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        newdata : pandas or Polars DataFrame or LazyFrame, optional
            New data for prediction. If None, returns fitted values. Lazy queries
            are projected to prediction columns and collected once per call.
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" for fixed effects only.
        allow_new_levels : bool, default False
            Allow new levels in grouping factors (predicts with RE=0).
        se_fit : bool, default False
            If True, return standard errors of predictions.
        interval : str, default "none"
            Type of interval: "none", "confidence", or "prediction".
        level : float, default 0.95
            Confidence level for intervals.
        offset : array-like, scalar, or str, optional
            Offset for new-data predictions. A string selects a column from
            ``newdata``. Scalars are broadcast to every row.
        weights : array-like, scalar, or str, optional
            Positive finite residual precision weights for new-data prediction
            intervals. A string selects a column from ``newdata``; scalars are
            broadcast to every row. Requires ``newdata`` and
            ``interval="prediction"``. Residual variance is ``sigma**2 / weights``;
            omitted weights default to one. These weights use the same scale as
            the fitted prior weights and do not change mean standard errors.

        Returns
        -------
        NDArray or PredictResult
            Predictions. Returns PredictResult if se_fit=True or interval!="none".

        Notes
        -----
        Prediction uncertainty uses the prior weights from the fitted model.
        In-sample prediction intervals add residual variance ``sigma**2 / weight``;
        new-data prediction intervals use the supplied ``weights``, defaulting to one.
        """
        valid_intervals = ("none", "confidence", "prediction")
        if interval not in valid_intervals:
            raise ValueError(
                f"Unknown interval type: {interval}. Use 'none', 'confidence', or 'prediction'."
            )
        level = _validate_confidence_level(level)

        prediction_weights: float | NDArray[np.floating] = 1.0
        if weights is not None:
            if newdata is None:
                raise ValueError("Prediction weights can only be supplied with newdata.")
            if interval != "prediction":
                raise ValueError("Prediction weights require interval='prediction'.")

        include_re = re_form != "NA" and re_form != "~0"
        if newdata is not None:
            newdata = self._prepare_prediction_data(
                newdata,
                include_re=include_re,
                extra_columns=tuple(value for value in (offset, weights) if isinstance(value, str)),
            )
        if weights is not None:
            prediction_weights = self._prediction_vector(
                newdata, weights, name="weights", default=1.0
            )
            if np.any(prediction_weights <= 0):
                raise ValueError("Prediction weights must be strictly positive.")
        random_design: tuple[sparse.csr_matrix, NDArray[np.floating]] | None = None

        if newdata is None:
            if offset is not None:
                raise ValueError("Prediction offset can only be supplied with newdata.")
            if include_re:
                pred = self._fitted_values.copy()
            else:
                pred = self.matrices.X @ self.beta + self.matrices.offset
            if not se_fit and interval == "none":
                return pred
            X = self.matrices.X
        else:
            prediction_offset = self._prediction_offset(newdata, offset)
            X = self._prediction_fixed_matrix(newdata)
            pred = X @ self.beta + prediction_offset

            if include_re:
                if se_fit or interval != "none":
                    random_design = self._prediction_random_matrix(
                        newdata, allow_new_levels, scale=self.sigma
                    )
                    pred += random_design[0] @ self.u
                else:
                    pred += self._random_effect_prediction_contrib(
                        newdata, allow_new_levels, self.u
                    )

        if not se_fit and interval == "none":
            return pred

        var_fit = self._compute_prediction_variance(
            X,
            random_design,
            include_re=include_re,
        )
        se = np.sqrt(var_fit)

        if interval == "none":
            return PredictResult(fit=pred, se_fit=se, interval="none", level=level)

        from scipy import stats

        z_crit = stats.norm.isf((1 - level) / 2)

        if interval == "confidence":
            lower = pred - z_crit * se
            upper = pred + z_crit * se
            return PredictResult(
                fit=pred, se_fit=se, lower=lower, upper=upper, interval="confidence", level=level
            )
        elif interval == "prediction":
            residual_var: float | NDArray[np.floating]
            if newdata is None:
                residual_var = self.sigma**2 / self.matrices.weights
            else:
                residual_var = self.sigma**2 / prediction_weights
            var_pred = var_fit + residual_var
            se_pred = np.sqrt(var_pred)
            lower = pred - z_crit * se_pred
            upper = pred + z_crit * se_pred
            return PredictResult(
                fit=pred,
                se_fit=se,
                lower=lower,
                upper=upper,
                interval="prediction",
                level=level,
            )
        raise AssertionError("interval validation should make this branch unreachable")

    def _compute_prediction_variance(
        self,
        X: NDArray[np.floating],
        random_design: tuple[sparse.csr_matrix, NDArray[np.floating]] | None,
        *,
        include_re: bool,
    ) -> NDArray[np.floating]:
        """Compute pointwise mixed-model mean-prediction variance."""
        q = self.matrices.n_random
        if not include_re or q == 0:
            vcov_beta = self.vcov()
            return np.maximum(dense_quadratic_form_diagonal(X, vcov_beta), 0.0)

        Z_pred: sparse.csr_matrix
        prior_var: NDArray[np.floating]
        if random_design is None:
            Z_pred = self.matrices.Z.tocsr()
            prior_var = np.zeros(X.shape[0], dtype=np.float64)
        else:
            Z_pred, prior_var = random_design

        projection = self._weighted_projection
        assert projection.lambda_matrix is not None and projection.random_factor is not None
        vcov_beta = self.vcov()

        transformed_Z = (Z_pred @ projection.lambda_matrix).tocsr()
        adjusted_X = X - np.asarray(transformed_Z @ projection.random_fixed_map)
        var_fixed = dense_quadratic_form_diagonal(adjusted_X, vcov_beta)
        var_random = self.sigma**2 * projection.random_factor.quadratic_diagonal(transformed_Z)

        return np.maximum(var_fixed + var_random + prior_var, 0.0)

    def vcov(self) -> NDArray[np.floating]:
        if self.matrices.n_fixed == 0:
            return np.empty((0, 0), dtype=np.float64)
        information_inv = symmetric_inverse(self._weighted_projection.XtVinvX)
        return self.sigma**2 * information_inv

    @cached_property
    def _hat_values(self) -> NDArray[np.float64]:
        projection = self._weighted_projection
        information_inv = symmetric_inverse(projection.XtVinvX)

        if projection.lambda_matrix is None:
            projected_X = projection.weighted_X
            h_random = np.zeros(self.matrices.n_obs, dtype=np.float64)
        else:
            assert projection.random_factor is not None
            B = projection.weighted_Z @ projection.lambda_matrix
            h_random = projection.random_factor.quadratic_diagonal(B)
            projected_X = projection.weighted_X - B @ projection.random_fixed_map

        h_fixed = np.einsum("ij,ij->i", projected_X @ information_inv, projected_X)
        return np.clip(h_fixed + h_random, 0, 1 - 1e-10)

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values (hatvalues)
            - 'cooks_d': Cook's distance
            - 'std_resid': Standardized residuals
            - 'student_resid': Studentized residuals (leave-one-out scale)

        Notes
        -----
        Residuals include square-root prior weights. Large hat values flag
        high-leverage points, large residuals flag outliers and large Cook's
        distances flag influential observations. ``mixedlm.diagnostics.influence``
        returns the same quantities with DFBETAS and DFFITS.
        """
        from mixedlm.diagnostics.influence import influence

        diagnostics = influence(self)
        resid = diagnostics.residuals
        one_minus_h = 1 - np.clip(diagnostics.hat_values, 0, 1 - 1e-10)
        df = self.matrices.n_obs - self.matrices.n_fixed
        loo_var = (np.sum(resid**2) - resid**2 / one_minus_h) / (df - 1)

        return {
            "hat": diagnostics.hat_values,
            "cooks_d": diagnostics.cooks_distance,
            "std_resid": resid / (self.sigma * np.sqrt(one_minus_h)),
            "student_resid": resid / np.sqrt(np.maximum(loo_var, 1e-10) * one_minus_h),
        }

    def VarCorr(self) -> VarCorr:
        return VarCorr(groups=self._varcorr_groups(scale=self.sigma**2), residual=self.sigma**2)

    def _getme_components(self) -> dict[str, Callable[[], Any]]:
        return {
            **super()._getme_components(),
            "sigma": lambda: self.sigma,
            "REML": lambda: self.REML,
        }

    def _compute_RZX(self) -> NDArray[np.floating]:
        """Compute RZX, the cross-term in the mixed model equations."""
        return self._weighted_projection.RZX.copy()

    def _compute_RX(self) -> NDArray[np.floating]:
        """Compute the upper Cholesky factor of the fixed-effect information."""
        return linalg.cholesky(self._weighted_projection.XtVinvX, lower=False)

    def _devcomp_cmp(self) -> dict[str, float]:
        projection = self._weighted_projection
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        u = self._spherical_u()
        resid = self.residuals(na_expand=False)
        wrss = float(np.dot(self.matrices.weights * resid, resid))
        ussq = float(np.dot(u, u))
        pwrss = wrss + ussq
        deviance = float(self.deviance)
        return {
            "ldL2": 0.0 if projection.random_factor is None else projection.random_factor.logdet,
            "ldRX2": float(np.linalg.slogdet(projection.XtVinvX)[1]),
            "wrss": wrss,
            "ussq": ussq,
            "pwrss": pwrss,
            "drsum": np.nan,
            "REML": deviance if self.REML else np.nan,
            "dev": np.nan if self.REML else deviance,
            "sigmaML": float(np.sqrt(pwrss / n)),
            "sigmaREML": float(np.sqrt(pwrss / (n - p))),
        }

    def get_deviance_components(self) -> DevianceComponents:
        """Get detailed deviance components for the fitted model.

        Returns a DevianceComponents object containing the breakdown of
        deviance into its constituent parts, including log-determinants,
        weighted RSS, and random effect penalty terms.

        Returns
        -------
        DevianceComponents
            Object with fields:
            - total: Total deviance
            - ldL2: 2 * log|L| (log-determinant of L)
            - ldRX2: 2 * log|RX| (REML adjustment)
            - wrss: Weighted residual sum of squares
            - ussq: Sum of squared random effects (u'u)
            - pwrss: Penalized WRSS (wrss + ussq)
            - sigma2: Residual variance estimate
            - REML: Whether REML estimation was used

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> dc = result.get_deviance_components()
        >>> print(dc)
        Deviance Components:
          Total deviance:     ...
          log|L|^2 (ldL2):    ...
          ...
        """
        from mixedlm.estimation.reml import profiled_deviance_components

        return profiled_deviance_components(self.theta, self.matrices, self.REML)

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        REML: bool | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        **kwargs,
    ) -> LmerResult:
        """Update and re-fit the model with modified arguments.

        This method allows updating the model formula, data, or other arguments
        and refitting. It's similar to R's update() function.

        Parameters
        ----------
        formula : str, optional
            New formula. If None, uses the original formula.
            Use "." to refer to the original formula components:
            - ". ~ . + newvar" adds a fixed effect
            - ". ~ . - oldvar" removes a fixed effect
        data : DataFrame, optional
            New data. If None, uses the original data (must be stored).
        REML : bool, optional
            Whether to use REML. If None, uses the original setting.
        weights : array-like, optional
            New weights. If None, uses the original weights.
        offset : array-like, optional
            New offset. If None, uses the original offset.
        **kwargs
            Additional arguments passed to lmer(). The fitted control settings
            are reused unless ``control`` is given.

        Returns
        -------
        LmerResult
            New fitted model result.

        Raises
        ------
        ValueError
            If data is needed but not available.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> # Add another fixed effect
        >>> result2 = result.update(". ~ . + z")
        >>> # Change to ML estimation
        >>> result3 = result.update(REML=False)
        >>> # Fit with new data
        >>> result4 = result.update(data=new_data)
        """

        if data is None:
            if self.matrices.frame is not None:
                data = self.matrices.frame
            else:
                raise ValueError(
                    "No data available. Either provide data or ensure model_frame was stored."
                )

        new_formula = str(self.formula) if formula is None else self._update_formula(formula)

        if REML is None:
            REML = self.REML

        data_size_changed = len(data) != self.matrices.n_obs

        if weights is None and not data_size_changed:
            weights = self.matrices.weights
        if offset is None and not data_size_changed:
            offset = self.matrices.offset

        kwargs.setdefault("control", self._refit_control())

        return lmer(new_formula, data, REML=REML, weights=weights, offset=offset, **kwargs)

    def logLik(self) -> LogLik:
        return LogLik(
            value=-0.5 * self.deviance, df=self.npar(), nobs=self.matrices.n_obs, REML=self.REML
        )

    def get_deviance(self) -> float:
        """Get the deviance of the fitted model.

        For linear mixed models, this returns the profiled deviance
        (or REML criterion if REML=True) which is minimized during fitting.

        Returns
        -------
        float
            The deviance value.

        See Also
        --------
        REMLcrit : Get the REML criterion value.
        logLik : Get the log-likelihood.
        """
        return self.deviance

    def REMLcrit(self) -> float:
        """Get the REML criterion value.

        Returns the REML criterion if the model was fit with REML=True,
        otherwise returns the ML deviance. This is the objective function
        value that was minimized during model fitting.

        Returns
        -------
        float
            The REML criterion (if REML=True) or ML deviance (if REML=False).

        Notes
        -----
        The REML criterion is related to the restricted log-likelihood by:
            REML_crit = -2 * log(L_REML) + constant

        For ML estimation, this returns the same value as `get_deviance()`.

        See Also
        --------
        get_deviance : Get the deviance value.
        logLik : Get the log-likelihood.
        isREML : Check if the model was fit with REML.
        """
        return self.deviance

    def as_function(
        self,
        type: str = "deviance",
    ) -> object:
        """Return the model's objective function.

        Parameters
        ----------
        type : str, default "deviance"
            Type of function to return:
            - "deviance": returns the deviance function
            - "predict": returns a prediction function

        Returns
        -------
        callable
            The requested function.

        Examples
        --------
        >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
        >>> devfun = result.as_function("deviance")
        >>> devfun(result.theta)  # Should equal result.deviance
        """
        if type == "deviance":
            optimizer = LMMOptimizer(
                self.matrices,
                REML=self.REML,
                verbose=0,
            )
            return optimizer.objective
        elif type == "predict":

            def predict_fn(X: NDArray[np.floating]) -> NDArray[np.floating]:
                return X @ self.beta

            return predict_fn
        else:
            raise ValueError(f"Unknown type: {type}. Use 'deviance' or 'predict'.")

    def profile(
        self,
        which: str | list[str] | None = None,
        n_points: int = 20,
        level: float = 0.95,
        n_jobs: int = 1,
    ) -> dict[str, ProfileResult]:
        """Compute fixed-effect likelihood profiles for this fitted model."""
        from mixedlm.inference.profile import profile_lmer

        return profile_lmer(self, which=which, n_points=n_points, level=level, n_jobs=n_jobs)

    def _bootstrap(self, n_boot: int, seed: RandomSeed) -> BootstrapResult:
        from mixedlm.inference.bootstrap import bootstrap_lmer

        return bootstrap_lmer(self, n_boot=n_boot, seed=seed)

    def _simulate_from_eta(self, eta: NDArray[np.floating], rng: Any) -> NDArray[np.floating]:
        """Add residual noise with variance ``sigma**2 / weights`` along the first axis."""
        scale = self.sigma / np.sqrt(self.matrices.weights)
        return eta + (rng.standard_normal(eta.shape[::-1]) * scale).T

    def _refit_from_matrices(
        self,
        matrices: ModelMatrices,
        reml: bool | None = None,
        **kwargs,
    ) -> LmerResult:
        reml = self.REML if reml is None else reml
        optimizer = LMMOptimizer(
            matrices,
            REML=reml,
            verbose=0,
            use_rust=None if self.control is None else self.control.use_rust,
        )

        start = kwargs.pop("start", self.theta)
        kwargs.setdefault("method", "auto")
        opt_result = optimizer.optimize(start=start, **kwargs)

        return LmerResult(
            formula=self.formula,
            matrices=matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=reml,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            gradient_norm=opt_result.gradient_norm,
            at_boundary=opt_result.at_boundary,
            message=opt_result.message,
            function_evals=opt_result.function_evals,
            optimizer=opt_result.optimizer or kwargs["method"],
            control=self.control,
        )

    def _refit_control(self) -> LmerControl:
        """Return the fitting controls for formula-based refits, or the defaults."""
        from mixedlm.models.control import LmerControl

        return self.control if self.control is not None else LmerControl()

    def refitML(self, **kwargs) -> LmerResult:
        """Refit the model using ML instead of REML.

        This method refits a REML model using maximum likelihood estimation.
        This is useful for likelihood ratio tests comparing models with
        different fixed effects, where REML-based comparisons are not valid.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).

        Returns
        -------
        LmerResult
            New fitted model result with REML=False.
            If the model was already fit with ML (REML=False), returns self.

        Notes
        -----
        Likelihood ratio tests for comparing models with different fixed
        effects should use ML estimation, not REML. This is because REML
        estimates the variance components after profiling out the fixed
        effects, making the REML likelihoods not comparable when fixed
        effects differ.

        Examples
        --------
        >>> # Fit two models with REML
        >>> m1 = lmer("y ~ x1 + (1|group)", data)
        >>> m2 = lmer("y ~ x1 + x2 + (1|group)", data)
        >>> # Refit with ML for valid LRT comparison
        >>> m1_ml = m1.refitML()
        >>> m2_ml = m2.refitML()
        >>> # Now can compare likelihoods
        >>> from scipy import stats
        >>> lr_stat = -2 * (m1_ml.logLik().value - m2_ml.logLik().value)
        >>> p_value = stats.chi2.sf(lr_stat, df=1)

        See Also
        --------
        refit : Refit with a new response vector.
        update : Update and refit with modified arguments.
        isREML : Check if the model was fit with REML.
        """
        if not self.REML:
            return self

        matrices = self._clone_matrices_with_response_base(self.matrices.y)
        return self._refit_from_matrices(matrices, reml=False, **kwargs)

    def drop1(self, data: pd.DataFrame, test: str = "Chisq", n_jobs: int = 1) -> Drop1Result:
        from mixedlm.inference.drop1 import drop1_lmer

        return drop1_lmer(self, data, test=test, n_jobs=n_jobs)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> AllFitResult:
        from mixedlm.inference.allfit import allfit_lmer

        return allfit_lmer(self, data, optimizers=optimizers, n_jobs=n_jobs, verbose=verbose)

    def summary(self, ddf_method: str | None = "Satterthwaite") -> str:
        """Generate summary of linear mixed model fit.

        Parameters
        ----------
        ddf_method : str, optional
            Method for computing denominator degrees of freedom and p-values.
            Options: "Satterthwaite", "Kenward-Roger", None (no p-values).
            Default is "Satterthwaite".

        Returns
        -------
        str
            Summary string formatted like lme4/lmerTest output.
        """
        lines = []
        lines.append("Linear mixed model fit by " + ("REML" if self.REML else "ML"))
        lines.append(f"Formula: {self.formula}")
        lines.append("")

        lines.append(str(self.VarCorr()))
        lines.append(f"Number of obs: {self.matrices.n_obs}")
        for struct in self.matrices.random_structures:
            lines.append(f"  groups:  {struct.grouping_factor}, {struct.n_levels}")
        lines.append("")

        lines.append("Fixed effects:")
        vcov = self.vcov()
        se = np.sqrt(np.diag(vcov))

        if ddf_method is not None:
            from scipy import stats

            from mixedlm.inference.ddf import kenward_roger_df, satterthwaite_df

            if ddf_method == "Satterthwaite":
                ddf_result = satterthwaite_df(self)
            elif ddf_method == "Kenward-Roger":
                ddf_result = kenward_roger_df(self)
            else:
                raise ValueError(
                    f"Unknown ddf_method: {ddf_method}. "
                    "Use 'Satterthwaite', 'Kenward-Roger', or None."
                )

            statistics = np.divide(
                self.beta, se, out=np.full_like(self.beta, np.nan, dtype=np.float64), where=se > 0
            )
            p_values = 2 * stats.t.sf(np.abs(statistics), ddf_result.df)

            lines.append(
                f"{'':12} {'Estimate':>10}  {'Std.Error':>10}  {'df':>8}  "
                f"{'t value':>8}  {'Pr(>|t|)':>10}"
            )
            for i, name in enumerate(self.matrices.fixed_names):
                t_val = statistics[i]
                df = ddf_result.df[i]
                p_val = p_values[i]
                sig = _get_signif_code(p_val)
                lines.append(
                    f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {df:8.2f}  "
                    f"{t_val:8.3f}  {_format_pvalue(p_val):>10} {sig}"
                )
            lines.append("---")
            lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        else:
            lines.append("             Estimate  Std. Error  t value")
            for i, name in enumerate(self.matrices.fixed_names):
                t_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
                lines.append(f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {t_val:7.3f}")

        lines.append("")
        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")
            lines.append("  optimizer did not converge; try allFit() to compare optimizers")
        if self.isSingular():
            lines.append("  boundary (singular) fit: some random-effect variances are near zero")
            lines.append("  consider simplifying the random-effects structure")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LmerResult(formula={self.formula}, deviance={self.deviance:.4f})"


from mixedlm.models.lmer_fit import LmerMod as LmerMod
from mixedlm.models.lmer_fit import lmer as lmer
