from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda, _count_theta
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_random_matrix
from mixedlm.models.lmer_types import (
    LogLik,
    ModelTerms,
    PredictResult,
    RanefResult,
    RePCA,
    RePCAGroup,
    VarCorrGroup,
)
from mixedlm.models.result_mixin import MerResultMixin
from mixedlm.models.shared_utils import symmetric_inverse
from mixedlm.utils import _format_pvalue, _get_signif_code


@dataclass
class _WeightedProjection:
    """Reusable weighted mixed-model projection factors."""

    sqrt_weights: NDArray[np.float64]
    weighted_X: NDArray[np.float64]
    weighted_Z: sparse.csc_matrix
    lambda_matrix: sparse.csc_matrix | None
    L_V: NDArray[np.float64] | None
    RZX: NDArray[np.float64]
    XtVinvX: NDArray[np.float64]


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


@dataclass
class LmerResult(MerResultMixin):
    _IS_LMM: ClassVar[bool] = True
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

    @property
    def fe_params(self) -> NDArray[np.floating]:
        """Compatibility alias for fixed-effect parameters."""
        return self.beta

    @property
    def re_params(self) -> NDArray[np.floating]:
        """Compatibility alias for random-effect covariance parameters."""
        return self.theta

    @property
    def resid(self) -> NDArray[np.floating]:
        """Compatibility alias for residuals."""
        return self.residuals()

    @property
    def fittedvalues(self) -> NDArray[np.floating]:
        """Compatibility alias for fitted values."""
        return self.fitted()

    def fixef(self) -> dict[str, float]:
        return self._fixef_dict(self.beta)

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        return self._ranef_with_optional_condvar(self.u, condVar)

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
                L_V=None,
                RZX=np.zeros((0, self.matrices.n_fixed), dtype=np.float64),
                XtVinvX=np.asarray(information, dtype=np.float64),
            )

        lambda_matrix = _build_lambda(self.theta, self.matrices.random_structures)
        ZtWZ = weighted_Z.T @ weighted_Z
        V_factor = lambda_matrix.T @ ZtWZ @ lambda_matrix + sparse.eye(q, format="csc")
        L_V = linalg.cholesky(V_factor.toarray(), lower=True)

        ZtWX = weighted_Z.T @ weighted_X
        RZX = linalg.solve_triangular(L_V, lambda_matrix.T @ ZtWX, lower=True)
        information = weighted_X.T @ weighted_X - RZX.T @ RZX
        information = (information + information.T) / 2.0

        return _WeightedProjection(
            sqrt_weights=sqrt_weights,
            weighted_X=weighted_X,
            weighted_Z=weighted_Z,
            lambda_matrix=lambda_matrix,
            L_V=L_V,
            RZX=np.asarray(RZX, dtype=np.float64),
            XtVinvX=np.asarray(information, dtype=np.float64),
        )

    def _compute_condVar(
        self, include_cov: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        from mixedlm.utils.variance import _conditional_variance_blocks

        q = self.matrices.n_random
        if q == 0:
            return {}

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        sqrt_weights = np.sqrt(self.matrices.weights)
        weighted_z = self.matrices.Z.multiply(sqrt_weights[:, np.newaxis])
        ztwz = weighted_z.T @ weighted_z
        precision = Lambda.T @ ztwz @ Lambda + sparse.eye(q, format="csc")

        return _conditional_variance_blocks(
            precision,
            Lambda,
            self.matrices.random_structures,
            scale=self.sigma**2,
            include_cov=include_cov,
        )

    def get_sigma(self) -> float:
        return self.sigma

    def weights(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the model weights.

        Returns the prior weights used in model fitting.
        If no weights were specified, returns an array of ones.
        The observation-level residual variance is ``sigma**2 / weight``.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the weights array.
            If False, return the original array (faster but should not be modified).

        Returns
        -------
        NDArray
            Array of weights with length equal to number of observations.
        """
        return self._weights_array(copy=copy)

    def offset(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the model offset.

        Returns the offset used in model fitting.
        If no offset was specified, returns an array of zeros.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy of the offset array.
            If False, return the original array (faster but should not be modified).

        Returns
        -------
        NDArray
            Array of offsets with length equal to number of observations.
        """
        return self._offset_array(copy=copy)

    def model_matrix(
        self, type: str = "fixed"
    ) -> NDArray[np.floating] | sparse.csc_matrix | tuple[NDArray[np.floating], sparse.csc_matrix]:
        """Get the model design matrix.

        Parameters
        ----------
        type : str, default "fixed"
            Which design matrix to return:
            - "fixed" or "X": Fixed effects design matrix
            - "random" or "Z": Random effects design matrix (sparse)
            - "both": Tuple of (X, Z)

        Returns
        -------
        NDArray or sparse.csc_matrix or tuple
            The requested design matrix. X is dense, Z is sparse.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> X = result.model_matrix("fixed")
        >>> Z = result.model_matrix("random")
        >>> X, Z = result.model_matrix("both")
        """
        return self._model_matrix(type)

    def terms(self) -> ModelTerms:
        """Get information about the model terms.

        Returns a ModelTerms object containing information about the
        response variable, fixed effect terms, random effect terms,
        and grouping factors.

        Returns
        -------
        ModelTerms
            Object containing:
            - response: Name of the response variable
            - fixed_terms: List of fixed effect term names
            - random_terms: Dict mapping grouping factors to their term names
            - fixed_variables: Set of variables in fixed effects
            - random_variables: Set of variables in random effects
            - grouping_factors: Set of grouping factor names
            - has_intercept: Whether the model has an intercept

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> t = result.terms()
        >>> print(t.response)  # 'y'
        >>> print(t.fixed_terms)  # ['(Intercept)', 'x']
        >>> print(t.random_terms)  # {'group': ['(Intercept)', 'x']}
        """
        return self._build_model_terms(self.formula)

    def model_frame(self) -> Any:
        """Get the model frame.

        Returns the data frame containing only the variables used
        in the model formula, after any NA handling.

        Returns
        -------
        DataFrame
            Data frame with the response variable, fixed effect variables,
            and grouping factors. The fitted input's backend is preserved.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> mf = result.model_frame()
        >>> print(mf.columns.tolist())  # ['y', 'x', 'group']
        """
        return self._model_frame()

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
    ) -> NDArray[np.floating] | PredictResult:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        newdata : DataFrame, optional
            New data for prediction. If None, returns fitted values.
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

        Returns
        -------
        NDArray or PredictResult
            Predictions. Returns PredictResult if se_fit=True or interval!="none".
        """
        valid_intervals = ("none", "confidence", "prediction")
        if interval not in valid_intervals:
            raise ValueError(
                f"Unknown interval type: {interval}. Use 'none', 'confidence', or 'prediction'."
            )
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be a finite number strictly between 0 and 1")

        include_re = re_form != "NA" and re_form != "~0"
        pred_matrices: ModelMatrices | None = None

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
            X = self._prediction_fixed_matrix(newdata)
            pred = X @ self.beta + self._prediction_offset(newdata, offset)

            if include_re:
                pred = self._add_random_effects_to_pred(pred, newdata, allow_new_levels)

            if include_re and (se_fit or interval != "none"):
                Z, random_structures = build_random_matrix(
                    self.formula,
                    newdata,
                    contrasts=self.matrices.contrasts,
                    category_levels=self.matrices.category_levels,
                )
                n_pred = X.shape[0]
                pred_matrices = ModelMatrices(
                    y=np.empty(n_pred, dtype=np.float64),
                    X=X,
                    Z=Z,
                    fixed_names=self.matrices.fixed_names,
                    random_structures=random_structures,
                    n_obs=n_pred,
                    n_fixed=X.shape[1],
                    n_random=Z.shape[1],
                    weights=np.ones(n_pred, dtype=np.float64),
                    offset=np.zeros(n_pred, dtype=np.float64),
                    category_levels=self.matrices.category_levels,
                    contrasts=self.matrices.contrasts,
                )

        if not se_fit and interval == "none":
            return pred

        var_fit = self._compute_prediction_variance(
            X,
            pred_matrices,
            include_re=include_re,
            allow_new_levels=allow_new_levels,
        )
        se = np.sqrt(var_fit)

        if interval == "none":
            return PredictResult(fit=pred, se_fit=se, interval="none", level=level)

        z_crit = stats.norm.ppf(1 - (1 - level) / 2)

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
                residual_var = self.sigma**2
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

    def _add_random_effects_to_pred(
        self,
        pred: NDArray[np.floating],
        newdata: pd.DataFrame,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Add random effects contribution to predictions."""
        pred += self._random_effect_prediction_contrib(newdata, allow_new_levels, self.u)
        return pred

    def _compute_prediction_variance(
        self,
        X: NDArray[np.floating],
        pred_matrices: ModelMatrices | None,
        *,
        include_re: bool,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Compute pointwise mixed-model mean-prediction variance."""
        q = self.matrices.n_random
        if not include_re or q == 0:
            vcov_beta = self.vcov()
            return np.maximum(np.sum((X @ vcov_beta) * X, axis=1), 0.0)

        Z_pred: sparse.csr_matrix
        prior_var: NDArray[np.floating]
        if pred_matrices is None:
            Z_pred = self.matrices.Z.tocsr()
            prior_var = np.zeros(X.shape[0], dtype=np.float64)
        else:
            Z_pred, prior_var = self._align_prediction_random_matrix(
                pred_matrices, allow_new_levels
            )

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        Zt = self.matrices.Zt
        V = Lambda.T @ (Zt @ self.matrices.Z) @ Lambda
        V = V + sparse.eye(q, format="csc")
        V_dense = V.toarray() if sparse.issparse(V) else np.asarray(V)
        L_V = linalg.cholesky(V_dense, lower=True)

        Lambdat_ZtX = Lambda.T @ (Zt @ self.matrices.X)
        beta_adjustment = linalg.cho_solve((L_V, True), Lambdat_ZtX)
        fixed_information = self.matrices.X.T @ self.matrices.X
        fixed_information -= Lambdat_ZtX.T @ beta_adjustment
        fixed_information = (fixed_information + fixed_information.T) / 2.0
        try:
            L_fixed = linalg.cholesky(fixed_information, lower=True)
            vcov_beta = self.sigma**2 * linalg.cho_solve(
                (L_fixed, True), np.eye(fixed_information.shape[0])
            )
        except linalg.LinAlgError:
            vcov_beta = self.sigma**2 * linalg.solve(
                fixed_information, np.eye(fixed_information.shape[0])
            )

        transformed_Z = (Z_pred @ Lambda).tocsr()
        adjusted_X = X - np.asarray(transformed_Z @ beta_adjustment)
        var_fixed = np.sum((adjusted_X @ vcov_beta) * adjusted_X, axis=1)
        var_random = self._conditional_random_prediction_variance(transformed_Z, L_V)

        return np.maximum(var_fixed + var_random + prior_var, 0.0)

    def _align_prediction_random_matrix(
        self,
        pred_matrices: ModelMatrices,
        allow_new_levels: bool,
    ) -> tuple[sparse.csr_matrix, NDArray[np.floating]]:
        """Align new-data random-effect columns to the fitted coefficient order."""
        fitted_structures = self.matrices.random_structures
        new_structures = pred_matrices.random_structures
        if len(new_structures) != len(fitted_structures):
            raise ValueError("New data produced an incompatible random-effects structure")

        from mixedlm.utils.variance import getL

        level_factors = cast(
            list[NDArray[np.floating]],
            getL(self.theta, fitted_structures, sigma=self.sigma, as_blocks=True),
        )

        known_rows: list[NDArray[np.integer]] = []
        known_cols: list[NDArray[np.integer]] = []
        known_values: list[NDArray[np.floating]] = []
        prior_var = np.zeros(pred_matrices.n_obs, dtype=np.float64)
        fitted_offset = 0
        new_offset = 0

        for fitted, new, level_factor in zip(
            fitted_structures, new_structures, level_factors, strict=True
        ):
            if fitted.grouping_factor != new.grouping_factor:
                raise ValueError("New data produced an incompatible random-effects structure")

            term_indices = {name: index for index, name in enumerate(fitted.term_names)}
            if set(new.term_names) != set(fitted.term_names):
                raise ValueError(
                    f"New data produced incompatible random-effect columns for "
                    f"'{fitted.grouping_factor}'"
                )
            mapped_terms = np.asarray(
                [term_indices[name] for name in new.term_names], dtype=np.int64
            )

            fitted_string_levels = {
                str(fitted_level): index for fitted_level, index in fitted.level_map.items()
            }
            new_levels: list[Any | None] = [None] * new.n_levels
            for stored_level, index in new.level_map.items():
                new_levels[index] = stored_level

            mapped_levels = np.full(new.n_levels, -1, dtype=np.int64)
            unknown_levels: list[Any] = []
            for index, candidate_level in enumerate(new_levels):
                if candidate_level in fitted.level_map:
                    mapped_levels[index] = fitted.level_map[candidate_level]
                elif str(candidate_level) in fitted_string_levels:
                    mapped_levels[index] = fitted_string_levels[str(candidate_level)]
                else:
                    unknown_levels.append(candidate_level)

            if unknown_levels and not allow_new_levels:
                raise ValueError(
                    f"New level '{unknown_levels[0]}' in grouping factor "
                    f"'{fitted.grouping_factor}'. Set allow_new_levels=True to predict "
                    "with random effects = 0."
                )

            new_width = new.n_levels * new.n_terms
            block = pred_matrices.Z[:, new_offset : new_offset + new_width].tocoo()
            entry_levels = block.col // new.n_terms
            entry_terms = block.col % new.n_terms
            target_levels = mapped_levels[entry_levels]
            target_terms = mapped_terms[entry_terms]
            known = target_levels >= 0

            if np.any(known):
                known_rows.append(block.row[known])
                known_cols.append(
                    fitted_offset + target_levels[known] * fitted.n_terms + target_terms[known]
                )
                known_values.append(block.data[known])

            unknown = ~known
            if np.any(unknown):
                unknown_design = sparse.coo_matrix(
                    (block.data[unknown], (block.row[unknown], target_terms[unknown])),
                    shape=(pred_matrices.n_obs, fitted.n_terms),
                ).toarray()
                prior_cov = level_factor @ level_factor.T
                prior_var += np.sum((unknown_design @ prior_cov) * unknown_design, axis=1)

            fitted_offset += fitted.n_levels * fitted.n_terms
            new_offset += new_width

        if known_values:
            rows = np.concatenate(known_rows)
            cols = np.concatenate(known_cols)
            values = np.concatenate(known_values)
        else:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float64)

        aligned = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(pred_matrices.n_obs, self.matrices.n_random),
            dtype=np.float64,
        )
        return aligned, prior_var

    def _conditional_random_prediction_variance(
        self,
        transformed_Z: sparse.csr_matrix,
        L_V: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate diagonal quadratic forms without materializing an n-by-q matrix."""
        n_rows, q = transformed_Z.shape
        result = np.empty(n_rows, dtype=np.float64)
        chunk_size = max(1, 1_000_000 // max(q, 1))

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = transformed_Z[start:stop].toarray()
            solved = linalg.solve_triangular(L_V, chunk.T, lower=True)
            result[start:stop] = self.sigma**2 * np.sum(solved**2, axis=0)

        return result

    def vcov(self) -> NDArray[np.floating]:
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
            assert projection.L_V is not None
            B = (projection.weighted_Z @ projection.lambda_matrix).toarray()
            B_Vinv = linalg.cho_solve((projection.L_V, True), B.T).T
            h_random = np.einsum("ij,ij->i", B_Vinv, B)
            V_inv_BtX = linalg.cho_solve((projection.L_V, True), B.T @ projection.weighted_X)
            projected_X = projection.weighted_X - B @ V_inv_BtX

        h_fixed = np.einsum("ij,ij->i", projected_X @ information_inv, projected_X)
        return np.clip(h_fixed + h_random, 0, 1 - 1e-10)

    def hatvalues(self) -> NDArray[np.floating]:
        """Return leverage values (the diagonal of the mixed-model hat matrix).

        For mixed models, the hat matrix projects the response onto fitted values.
        The leverage h_i measures how much observation i influences its own
        fitted value.

        Returns
        -------
        NDArray
            Leverage values for each observation, between 0 and 1.
            Values close to 1 indicate high-leverage observations.

        Notes
        -----
        For linear mixed models, the hat matrix incorporates both fixed and
        random effects. High leverage observations can have undue influence
        on model estimates.
        """
        return self._hat_values.copy()

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        Cook's distance measures the influence of each observation on the
        fitted values. Large values indicate observations that have substantial
        influence on the model fit.

        Returns
        -------
        NDArray
            Cook's distance for each observation.

        Notes
        -----
        Cook's distance is computed as:
            D_i = (e_i^2 / (p * sigma^2)) * (h_i / (1 - h_i)^2)

        where e_i is the residual, h_i is the leverage, p is the number of
        fixed effect parameters, and sigma^2 is the residual variance.

        A common rule of thumb is that observations with D_i > 4/n or D_i > 1
        may be influential and warrant further investigation.
        """
        h = self.hatvalues()
        resid = self.residuals(type="response", na_expand=False)
        weighted_resid = np.sqrt(self.matrices.weights) * resid
        p = self.matrices.n_fixed

        h = np.clip(h, 0, 1 - 1e-10)

        cooks_d = (weighted_resid**2 / (p * self.sigma**2)) * (h / (1 - h) ** 2)

        return cooks_d

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns a dictionary containing various influence measures for
        identifying influential observations.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values (hatvalues)
            - 'cooks_d': Cook's distance
            - 'std_resid': Standardized residuals
            - 'student_resid': Studentized residuals

        Notes
        -----
        Influential observations are those that have a large effect on
        the model estimates. Use these diagnostics to identify:
        - High leverage points (large hat values)
        - Outliers (large standardized residuals)
        - Influential points (large Cook's distance)
        """
        h = self.hatvalues()
        resid = self.residuals(type="response", na_expand=False)
        weighted_resid = np.sqrt(self.matrices.weights) * resid

        h_safe = np.clip(h, 0, 1 - 1e-10)
        std_resid = weighted_resid / (self.sigma * np.sqrt(1 - h_safe))

        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        df = n - p

        mse = np.sum(weighted_resid**2) / df
        loo_var = ((df * mse) - (weighted_resid**2 / (1 - h_safe))) / (df - 1)
        loo_var = np.maximum(loo_var, 1e-10)
        student_resid = weighted_resid / np.sqrt(loo_var * (1 - h_safe))

        return {
            "hat": h,
            "cooks_d": self.cooks_distance(),
            "std_resid": std_resid,
            "student_resid": student_resid,
        }

    def VarCorr(self) -> VarCorr:
        groups: dict[str, VarCorrGroup] = {}

        for struct, cov in self._iter_random_cov_blocks(scale=self.sigma**2):
            if struct.correlated or struct.cov_type in ("cs", "ar1"):
                stddevs = np.sqrt(np.diag(cov))
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = cov / np.outer(stddevs, stddevs)
                    corr = np.where(np.isfinite(corr), corr, 0.0)
                    np.fill_diagonal(corr, 1.0)
            else:
                corr = None

            diag_cov = np.diag(cov)
            variance = {term: diag_cov[i] for i, term in enumerate(struct.term_names)}
            stddev = {term: np.sqrt(diag_cov[i]) for i, term in enumerate(struct.term_names)}

            groups[struct.grouping_factor] = VarCorrGroup(
                name=struct.grouping_factor,
                term_names=list(struct.term_names),
                variance=variance,
                stddev=stddev,
                cov=cov,
                corr=corr,
            )

        return VarCorr(groups=groups, residual=self.sigma**2)

    def rePCA(self) -> RePCA:
        """Perform PCA on the random effects covariance matrix.

        This function computes principal component analysis on the covariance
        matrix of each random effect grouping factor. It's useful for diagnosing
        overparameterization in the random effects structure.

        Returns
        -------
        RePCA
            Object containing PCA results for each random effect group,
            including standard deviations, proportion of variance, and
            cumulative proportion for each principal component.

        Notes
        -----
        If any principal component has very small standard deviation (< 1e-4),
        this suggests the random effects structure may be overparameterized
        (singular or near-singular). Use the `is_singular()` method on the
        result to check for this condition.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> pca = result.rePCA()
        >>> print(pca)
        >>> pca.is_singular()  # Check if any components are near-zero
        """
        groups: dict[str, RePCAGroup] = {}

        for struct, cov in self._iter_random_cov_blocks(scale=self.sigma**2):
            q = struct.n_terms

            eigenvalues = linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)

            sdev = np.sqrt(eigenvalues)
            total_var = np.sum(eigenvalues)

            proportion = eigenvalues / total_var if total_var > 0 else np.zeros(q)

            cumulative = np.cumsum(proportion)

            groups[struct.grouping_factor] = RePCAGroup(
                name=struct.grouping_factor,
                n_terms=q,
                sdev=sdev,
                proportion=proportion,
                cumulative=cumulative,
            )

        return RePCA(groups=groups)

    def dotplot(
        self,
        group: str | None = None,
        term: str | None = None,
        condVar: bool = True,
        order: bool = True,
        figsize: tuple[float, float] | None = None,
    ):
        """Create a caterpillar plot of random effects.

        Dotplots (also called caterpillar plots) show the estimated random
        effects with confidence intervals, ordered by magnitude. They are
        useful for visualizing the distribution of random effects across
        groups and identifying outlier groups.

        Parameters
        ----------
        group : str, optional
            Name of grouping factor to plot. If None, uses the first
            random effect grouping factor.
        term : str, optional
            Name of random effect term to plot. If None, plots all terms
            for the selected group in separate panels.
        condVar : bool, default True
            Whether to show 95% confidence intervals based on the
            conditional variance of the random effects.
        order : bool, default True
            Whether to order groups by random effect magnitude.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with the dotplot(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> fig = result.dotplot()  # Plot all random effects
        >>> fig = result.dotplot(term="(Intercept)")  # Plot only intercepts
        """
        from mixedlm.diagnostics.plots import _check_matplotlib, plot_ranef

        _check_matplotlib()
        import matplotlib.pyplot as plt

        if group is None:
            if not self.matrices.random_structures:
                raise ValueError("No random effects in model")
            group = self.matrices.random_structures[0].grouping_factor

        struct = None
        for s in self.matrices.random_structures:
            if s.grouping_factor == group:
                struct = s
                break

        if struct is None:
            raise ValueError(f"Grouping factor '{group}' not found")

        if term is not None:
            if figsize is None:
                figsize = (8, max(6, struct.n_levels * 0.3))
            fig, ax = plt.subplots(figsize=figsize)
            plot_ranef(self, group=group, term=term, ax=ax, condVar=condVar, order=order)
            fig.tight_layout()
            return fig

        n_terms = struct.n_terms
        if n_terms == 1:
            if figsize is None:
                figsize = (8, max(6, struct.n_levels * 0.3))
            fig, ax = plt.subplots(figsize=figsize)
            plot_ranef(
                self, group=group, term=struct.term_names[0], ax=ax, condVar=condVar, order=order
            )
            fig.tight_layout()
            return fig

        ncols = min(2, n_terms)
        nrows = (n_terms + ncols - 1) // ncols

        if figsize is None:
            figsize = (6 * ncols, max(6, struct.n_levels * 0.25) * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_terms > 1 else [axes]

        for i, term_name in enumerate(struct.term_names):
            if i < len(axes):
                plot_ranef(
                    self, group=group, term=term_name, ax=axes[i], condVar=condVar, order=order
                )

        for i in range(n_terms, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        return fig

    def qqmath(
        self,
        group: str | None = None,
        term: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Create QQ plots of random effects against normal distribution.

        QQ (quantile-quantile) plots compare the distribution of random
        effects to a theoretical normal distribution. Points falling along
        a diagonal line indicate normality. Deviations suggest the random
        effects may not be normally distributed.

        Parameters
        ----------
        group : str, optional
            Name of grouping factor to plot. If None, uses the first
            random effect grouping factor.
        term : str, optional
            Name of random effect term to plot. If None, plots all terms
            for the selected group in separate panels.
        figsize : tuple, optional
            Figure size (width, height) in inches.

        Returns
        -------
        Figure
            Matplotlib figure with the QQ plot(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (x | group)", data)
        >>> fig = result.qqmath()  # QQ plots for all random effects
        >>> fig = result.qqmath(term="(Intercept)")  # Only intercepts
        """
        from mixedlm.diagnostics.plots import _check_matplotlib

        _check_matplotlib()
        import matplotlib.pyplot as plt
        from scipy import stats

        if group is None:
            if not self.matrices.random_structures:
                raise ValueError("No random effects in model")
            group = self.matrices.random_structures[0].grouping_factor

        struct = None
        for s in self.matrices.random_structures:
            if s.grouping_factor == group:
                struct = s
                break

        if struct is None:
            raise ValueError(f"Grouping factor '{group}' not found")

        ranefs = self.ranef()
        group_ranefs = ranefs[group]

        def plot_qq(ax, values, title):
            values = np.asarray(values)
            values_sorted = np.sort(values)
            n = len(values_sorted)

            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

            ax.scatter(theoretical, values_sorted, alpha=0.7, edgecolors="black", linewidths=0.5)

            slope, intercept = np.polyfit(theoretical, values_sorted, 1)
            line_x = np.array([theoretical.min(), theoretical.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, "r--", linewidth=1.5, label="Reference line")

            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title(title)
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        if term is not None:
            if term not in group_ranefs:
                raise ValueError(f"Term '{term}' not found in group '{group}'")
            if figsize is None:
                figsize = (6, 5)
            fig, ax = plt.subplots(figsize=figsize)
            plot_qq(ax, group_ranefs[term], f"QQ Plot: {group} / {term}")
            fig.tight_layout()
            return fig

        n_terms = len(group_ranefs)
        if n_terms == 1:
            if figsize is None:
                figsize = (6, 5)
            fig, ax = plt.subplots(figsize=figsize)
            term_name = list(group_ranefs.keys())[0]
            plot_qq(ax, group_ranefs[term_name], f"QQ Plot: {group} / {term_name}")
            fig.tight_layout()
            return fig

        ncols = min(2, n_terms)
        nrows = (n_terms + ncols - 1) // ncols

        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_terms > 1 else [axes]

        for i, (term_name, values) in enumerate(group_ranefs.items()):
            if i < len(axes):
                plot_qq(axes[i], values, f"QQ Plot: {group} / {term_name}")

        for i in range(n_terms, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        return fig

    def plot(
        self,
        which: list[int] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Create diagnostic plots for the fitted model.

        Creates a panel of residual diagnostic plots similar to R's plot() method
        for lmer objects. This is useful for assessing model assumptions such as
        homoscedasticity, normality of residuals, and detecting outliers.

        Parameters
        ----------
        which : list of int, optional
            Which plots to include. Default is [1, 2, 3, 4].
            1 = Residuals vs Fitted values
            2 = Normal Q-Q plot of residuals
            3 = Scale-Location plot (sqrt of standardized residuals vs fitted)
            4 = Residuals by Group (boxplot, only if random effects exist)
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is calculated based
            on number of plots.

        Returns
        -------
        Figure
            Matplotlib figure containing the diagnostic plots.

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> result = lmer("y ~ x + (1 | group)", data)
        >>> fig = result.plot()  # All 4 diagnostic plots
        >>> fig = result.plot(which=[1, 2])  # Only residuals vs fitted and Q-Q

        See Also
        --------
        qqmath : QQ plots of random effects (normality assessment)
        residuals : Get residuals from the fitted model
        fitted : Get fitted values from the model
        """
        from mixedlm.diagnostics.plots import plot_diagnostics

        return plot_diagnostics(self, which=which, figsize=figsize)

    def isSingular(self, tol: float = 1e-4) -> bool:
        return self._is_singular_covariance(tol)

    def getME(self, name: str):
        """Extract model components by name.

        This method provides access to internal model components, similar to
        R's getME() function in lme4. It's useful for advanced users who need
        access to specific matrices or parameters.

        Parameters
        ----------
        name : str
            Name of the component to extract. Valid names are:
            - "X" : Fixed effects design matrix (n x p)
            - "Z" : Random effects design matrix (n x q)
            - "Zt" : Transpose of Z (q x n)
            - "y" : Response vector
            - "beta" : Fixed effects coefficients
            - "theta" : Variance component parameters (relative covariance factors)
            - "Lambda" : Relative covariance factor (sparse, q x q)
            - "Lambdat" : Transpose of Lambda
            - "u" : Spherical random effects
            - "b" : Conditional modes of random effects (same as u for LMM)
            - "sigma" : Residual standard deviation
            - "n" or "n_obs" : Number of observations
            - "p" or "n_fixed" : Number of fixed effects
            - "q" or "n_random" : Number of random effects
            - "lower" : Lower bounds for theta
            - "weights" : Prior weights
            - "offset" : Offset term
            - "REML" : Whether REML estimation was used
            - "deviance" : Deviance (or REML criterion)
            - "flist" : List of grouping factors
            - "cnms" : Component names for random effects
            - "Gp" : Group pointers (cumulative number of random effects per group)

        Returns
        -------
        The requested component. Type depends on the component name.

        Raises
        ------
        ValueError
            If an unknown component name is requested.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> X = result.getME("X")  # Fixed effects design matrix
        >>> theta = result.getME("theta")  # Variance parameters
        >>> Lambda = result.getME("Lambda")  # Relative covariance factor
        """
        if name == "X":
            return self.matrices.X
        elif name == "Z":
            return self.matrices.Z
        elif name == "Zt":
            return self.matrices.Zt
        elif name == "y":
            return self.matrices.y
        elif name == "beta":
            return self.beta.copy()
        elif name == "theta":
            return self.theta.copy()
        elif name == "Lambda":
            return _build_lambda(self.theta, self.matrices.random_structures)
        elif name == "Lambdat":
            Lambda = _build_lambda(self.theta, self.matrices.random_structures)
            return Lambda.T
        elif name == "u" or name == "b":
            return self.u.copy()
        elif name == "sigma":
            return self.sigma
        elif name in ("n", "n_obs"):
            return self.matrices.n_obs
        elif name in ("p", "n_fixed"):
            return self.matrices.n_fixed
        elif name in ("q", "n_random"):
            return self.matrices.n_random
        elif name == "lower":
            return self._theta_lower_bounds()
        elif name == "weights":
            return self.matrices.weights.copy()
        elif name == "offset":
            return self.matrices.offset.copy()
        elif name == "REML":
            return self.REML
        elif name == "deviance":
            return self.deviance
        elif name == "fixef_names":
            return list(self.matrices.fixed_names)
        elif name == "flist":
            return [s.grouping_factor for s in self.matrices.random_structures]
        elif name == "cnms":
            return {s.grouping_factor: s.term_names for s in self.matrices.random_structures}
        elif name == "Gp":
            gp = [0]
            for s in self.matrices.random_structures:
                gp.append(gp[-1] + s.n_levels * s.n_terms)
            return np.array(gp)
        elif name == "RX":
            return self._compute_RX()
        elif name == "RZX":
            return self._compute_RZX()
        elif name == "Lind":
            return self._build_Lind()
        elif name == "devcomp":
            return self._get_devcomp()
        else:
            valid_names = [
                "X",
                "Z",
                "Zt",
                "y",
                "beta",
                "theta",
                "Lambda",
                "Lambdat",
                "u",
                "b",
                "sigma",
                "n",
                "n_obs",
                "p",
                "n_fixed",
                "q",
                "n_random",
                "lower",
                "weights",
                "offset",
                "REML",
                "deviance",
                "fixef_names",
                "flist",
                "cnms",
                "Gp",
                "RX",
                "RZX",
                "Lind",
                "devcomp",
            ]
            raise ValueError(f"Unknown component name: '{name}'. Valid names are: {valid_names}")

    def _compute_RZX(self) -> NDArray[np.floating]:
        """Compute RZX, the cross-term in the mixed model equations."""
        return self._weighted_projection.RZX.copy()

    def _compute_RX(self) -> NDArray[np.floating]:
        """Compute the upper Cholesky factor of the fixed-effect information."""
        return linalg.cholesky(self._weighted_projection.XtVinvX, lower=False)

    def _build_Lind(self) -> NDArray[np.int64]:
        """Build Lind, the index mapping from theta to Lambda entries."""
        indices = []
        theta_idx = 0

        for struct in self.matrices.random_structures:
            n_terms = struct.n_terms

            if struct.correlated:
                n_theta = n_terms * (n_terms + 1) // 2
                template_indices = []
                idx = 0
                for i in range(n_terms):
                    for _j in range(i + 1):
                        template_indices.append(theta_idx + idx)
                        idx += 1
            else:
                n_theta = n_terms
                template_indices = list(range(theta_idx, theta_idx + n_terms))

            for _ in range(struct.n_levels):
                indices.extend(template_indices)

            theta_idx += n_theta

        return np.array(indices, dtype=np.int64)

    def _get_devcomp(self) -> dict[str, Any]:
        """Get deviance components and model dimensions."""
        from mixedlm.estimation.reml import profiled_deviance_components

        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        q = self.matrices.n_random
        n_theta = len(self.theta)

        try:
            dc = profiled_deviance_components(self.theta, self.matrices, self.REML)
            cmp = {
                "ldL2": dc.ldL2,
                "ldRX2": dc.ldRX2,
                "wrss": dc.wrss,
                "ussq": dc.ussq,
                "pwrss": dc.pwrss,
                "drsum": 0.0,
                "REML": float(self.deviance) if self.REML else np.nan,
                "dev": float(self.deviance) if not self.REML else np.nan,
                "sigmaML": np.sqrt(dc.sigma2) if not self.REML else np.nan,
                "sigmaREML": np.sqrt(dc.sigma2) if self.REML else np.nan,
            }
        except (ImportError, RuntimeError, ValueError, np.linalg.LinAlgError):
            cmp = {
                "ldL2": 0.0,
                "ldRX2": 0.0,
                "wrss": 0.0,
                "ussq": float(np.sum(self.u**2)) if self.u is not None else 0.0,
                "pwrss": 0.0,
                "drsum": 0.0,
                "REML": float(self.deviance) if self.REML else np.nan,
                "dev": float(self.deviance) if not self.REML else np.nan,
                "sigmaML": self.sigma,
                "sigmaREML": self.sigma,
            }

        dims = {
            "n": n,
            "p": p,
            "q": q,
            "nmp": n - p,
            "nth": n_theta,
            "REML": 1 if self.REML else 0,
            "useSc": 1,
            "nAGQ": 1,
            "q0": q,
            "q1": 0,
            "qrx": p,
            "ngrps": len(self.matrices.random_structures),
        }

        return {"cmp": cmp, "dims": dims}

    def get_deviance_components(self):
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
            Additional arguments passed to lmer().

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

        return lmer(new_formula, data, REML=REML, weights=weights, offset=offset, **kwargs)

    def _update_formula(self, new_formula: str) -> str:
        """Process formula update syntax with '.' placeholders."""
        original = str(self.formula)

        if "." not in new_formula:
            return new_formula

        lhs, rhs = original.split("~", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        if "~" in new_formula:
            new_lhs, new_rhs = new_formula.split("~", 1)
            new_lhs = new_lhs.strip()
            new_rhs = new_rhs.strip()

            if new_lhs == ".":
                new_lhs = lhs

            if new_rhs.startswith(". +"):
                new_rhs = rhs + " +" + new_rhs[3:]
            elif new_rhs.startswith(". -"):
                terms_to_remove = new_rhs[3:].strip().split("+")
                terms_to_remove = [t.strip() for t in terms_to_remove]
                rhs_terms = [t.strip() for t in rhs.split("+")]
                rhs_terms = [t for t in rhs_terms if t not in terms_to_remove]
                new_rhs = " + ".join(rhs_terms)
            elif new_rhs == ".":
                new_rhs = rhs

            return f"{new_lhs} ~ {new_rhs}"
        else:
            return new_formula

    def logLik(self) -> LogLik:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        n_theta = _count_theta(self.matrices.random_structures)
        df = p + n_theta + 1

        value = -0.5 * self.deviance

        return LogLik(value=value, df=df, nobs=n, REML=self.REML)

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

    def AIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + 2 * ll.df

    def BIC(self) -> float:
        ll = self.logLik()
        return -2 * ll.value + ll.df * np.log(ll.nobs)

    def extractAIC(self) -> tuple[float, float]:
        """Extract AIC with effective degrees of freedom.

        Returns the effective degrees of freedom and AIC value,
        matching the interface of R's extractAIC function.

        Returns
        -------
        tuple of (float, float)
            (edf, AIC) where edf is the effective degrees of freedom.

        Examples
        --------
        >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
        >>> edf, aic = result.extractAIC()
        """
        ll = self.logLik()
        edf = float(ll.df)
        aic = float(-2 * ll.value + 2 * ll.df)
        return (edf, aic)

    def get_formula(
        self,
        random_only: bool = False,
        fixed_only: bool = False,
    ) -> Formula | str:
        """Get the model formula.

        When called without arguments, returns the Formula object.
        When random_only or fixed_only is specified, returns a string.

        Parameters
        ----------
        random_only : bool, default False
            If True, return only the random effects part as a string.
        fixed_only : bool, default False
            If True, return only the fixed effects part as a string.

        Returns
        -------
        Formula or str
            The Formula object (default), or a string if random_only
            or fixed_only is specified.

        Examples
        --------
        >>> result = lmer("Reaction ~ Days + (Days|Subject)", sleepstudy)
        >>> f = result.get_formula()
        >>> f.response
        'Reaction'
        >>> result.get_formula(fixed_only=True)
        'Reaction ~ Days'
        >>> result.get_formula(random_only=True)
        '(Days | Subject)'
        """
        from mixedlm.formula.parser import (
            getFixedFormulaStr,
            getRandomFormulaStr,
        )

        if random_only and fixed_only:
            raise ValueError("Cannot specify both random_only and fixed_only")

        if random_only:
            return getRandomFormulaStr(str(self.formula))
        elif fixed_only:
            return getFixedFormulaStr(str(self.formula))
        else:
            return self.formula

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
        from mixedlm.estimation.reml import LMMOptimizer

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

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "Wald",
        n_boot: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        from mixedlm.inference.bootstrap import bootstrap_lmer
        from mixedlm.inference.profile import profile_lmer

        if parm is None:
            parm = self.matrices.fixed_names
        elif isinstance(parm, str):
            parm = [parm]

        if method == "Wald":
            vcov = self.vcov()
            alpha = 1 - level
            z_crit = stats.norm.ppf(1 - alpha / 2)

            result: dict[str, tuple[float, float]] = {}
            for p in parm:
                if p not in self.matrices.fixed_names:
                    continue
                idx = self.matrices.fixed_names.index(p)
                se = np.sqrt(vcov[idx, idx])
                lower = self.beta[idx] - z_crit * se
                upper = self.beta[idx] + z_crit * se
                result[p] = (float(lower), float(upper))
            return result

        elif method == "profile":
            profiles = profile_lmer(self, which=parm, level=level)
            return {p: (profiles[p].ci_lower, profiles[p].ci_upper) for p in parm if p in profiles}

        elif method == "boot":
            boot_result = bootstrap_lmer(self, n_boot=n_boot, seed=seed)
            return boot_result.ci(level=level)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'Wald', 'profile', or 'boot'.")

    def simulate(
        self,
        nsim: int = 1,
        seed: int | None = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        if nsim < 1:
            raise ValueError("nsim must be at least 1")

        if seed is not None:
            np.random.seed(seed)

        n = self.matrices.n_obs
        q = self.matrices.n_random

        if nsim == 1:
            return self._simulate_once(use_re, re_form)

        include_re = use_re and q > 0 and re_form not in ("~0", "NA")

        if not include_re:
            fixed_part = self.matrices.X @ self.beta + self.matrices.offset
            result = np.random.randn(nsim, n).T
            result *= self.sigma
            result += fixed_part[:, None]
            return result

        try:
            from mixedlm._rust import simulate_re_batch

            return self._simulate_batch_rust(nsim, seed, simulate_re_batch)
        except ImportError:
            pass

        result = np.zeros((n, nsim), dtype=np.float64)
        for i in range(nsim):
            result[:, i] = self._simulate_once(use_re, re_form)

        return result

    def _simulate_batch_rust(
        self,
        nsim: int,
        seed: int | None,
        simulate_re_batch: Any,
    ) -> NDArray[np.floating]:
        n = self.matrices.n_obs

        fixed_part = self.matrices.X @ self.beta + self.matrices.offset

        n_levels = [s.n_levels for s in self.matrices.random_structures]
        n_terms = [s.n_terms for s in self.matrices.random_structures]
        correlated = [s.correlated for s in self.matrices.random_structures]

        u_batch = simulate_re_batch(
            self.theta,
            self.sigma,
            n_levels,
            n_terms,
            correlated,
            nsim,
            seed,
        )

        Z = self.matrices.Z

        result = np.asarray(Z @ u_batch.T, dtype=np.float64)
        result += fixed_part[:, None]
        noise = np.random.randn(nsim, n).T
        noise *= self.sigma
        result += noise
        return result

    def _simulate_once(
        self,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        n = self.matrices.n_obs
        q = self.matrices.n_random

        fixed_part = self.matrices.X @ self.beta + self.matrices.offset

        if re_form == "~0" or re_form == "NA" or not use_re or q == 0:
            random_part = np.zeros(n)
        else:
            u_new = np.zeros(q, dtype=np.float64)
            u_idx = 0
            theta_start = 0

            for struct in self.matrices.random_structures:
                n_levels = struct.n_levels
                n_terms = struct.n_terms

                n_theta = n_terms * (n_terms + 1) // 2 if struct.correlated else n_terms
                theta_block = self.theta[theta_start : theta_start + n_theta]

                if struct.correlated:
                    L = np.zeros((n_terms, n_terms))
                    idx = 0
                    for i in range(n_terms):
                        for j in range(i + 1):
                            L[i, j] = theta_block[idx]
                            idx += 1
                    cov = L @ L.T * self.sigma**2
                else:
                    cov = np.diag(theta_block**2) * self.sigma**2

                for g in range(n_levels):
                    b_g = np.random.multivariate_normal(np.zeros(n_terms), cov)
                    for j in range(n_terms):
                        u_new[u_idx + g * n_terms + j] = b_g[j]

                u_idx += n_levels * n_terms
                theta_start += n_theta

            random_part = self.matrices.Z @ u_new

        noise = np.random.randn(n) * self.sigma

        return fixed_part + random_part + noise

    def _refit_from_matrices(
        self,
        matrices: ModelMatrices,
        reml: bool,
        **kwargs,
    ) -> LmerResult:
        optimizer = LMMOptimizer(
            matrices,
            REML=reml,
            verbose=0,
        )

        start = kwargs.pop("start", self.theta)
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
        )

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
        >>> p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

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

    def refit(
        self,
        newresp: NDArray[np.floating] | None = None,
        **kwargs,
    ) -> LmerResult:
        """Refit the model with a new response vector.

        This method refits the model using the same formula and design matrices
        but with a different response vector. This is useful for simulation
        studies, bootstrap, and permutation tests.

        Parameters
        ----------
        newresp : array-like, optional
            New response values. Must have the same length as the original
            response. If None, refits with the original response.
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).

        Returns
        -------
        LmerResult
            New fitted model result with the updated response.

        Examples
        --------
        >>> result = lmer("y ~ x + (1|group)", data)
        >>> # Refit with simulated response
        >>> y_sim = result.simulate()
        >>> result_sim = result.refit(newresp=y_sim)

        See Also
        --------
        simulate : Simulate response from the fitted model.
        refitML : Refit using ML instead of REML.
        """
        y_new = self._coerce_new_response(newresp)
        matrices = self._clone_matrices_with_response_base(y_new)
        return self._refit_from_matrices(matrices, reml=self.REML, **kwargs)

    def npar(self) -> int:
        """Get the number of parameters in the model.

        Returns the total number of estimated parameters:
        - Fixed effects (beta)
        - Variance-covariance parameters (theta)
        - Residual standard deviation (sigma)

        Returns
        -------
        int
            Total number of parameters.
        """
        return self._npar_count(include_sigma=True)

    def drop1(self, data: pd.DataFrame, test: str = "Chisq"):
        from mixedlm.inference.drop1 import drop1_lmer

        return drop1_lmer(self, data, test=test)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
    ):
        from mixedlm.inference.allfit import allfit_lmer

        return allfit_lmer(self, data, optimizers=optimizers, verbose=verbose)

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
            from mixedlm.inference.ddf import kenward_roger_df, pvalues_with_ddf, satterthwaite_df

            if ddf_method == "Satterthwaite":
                ddf_result = satterthwaite_df(self)
            elif ddf_method == "Kenward-Roger":
                ddf_result = kenward_roger_df(self)
            else:
                raise ValueError(
                    f"Unknown ddf_method: {ddf_method}. "
                    "Use 'Satterthwaite', 'Kenward-Roger', or None."
                )

            pval_dict = pvalues_with_ddf(self, method=ddf_method)

            lines.append(
                f"{'':12} {'Estimate':>10}  {'Std.Error':>10}  {'df':>8}  "
                f"{'t value':>8}  {'Pr(>|t|)':>10}"
            )
            for i, name in enumerate(self.matrices.fixed_names):
                t_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
                df = ddf_result.df[i]
                _, _, p_val = pval_dict[name]
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

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"LmerResult(formula={self.formula}, deviance={self.deviance:.4f})"


from mixedlm.models.lmer_fit import LmerMod as LmerMod
from mixedlm.models.lmer_fit import lmer as lmer
