from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.nlmm import (
    NLMMOptimizer,
    _as_prior_weights,
    _build_psi_matrix,
    _grouped_observation_indices,
)
from mixedlm.models.lmer_types import LogLik
from mixedlm.models.result_mixin import _ResultBase
from mixedlm.nlme.models import NonlinearModel
from mixedlm.utils.random import RandomSeed, RandomStream
from mixedlm.utils.validation import _validate_confidence_level

_COV_REGULARIZATION = 1e-8
_VCOV_EPS = 1e-5
_JACOBIAN_EPS = 1e-6
_HAT_FALLBACK_EPS = 1e-10
_HAT_CLIP_MAX = 1 - 1e-10
_DEFAULT_MAXITER = 500
_DEFAULT_N_BOOT = 1000
_DEFAULT_SINGULAR_TOL = 1e-4


@dataclass
class _NlmerSimulation:
    """Preparation for a single simulation or bootstrap call with fixed model data."""

    result: NlmerResult
    group_rows: list[NDArray[np.intp]] | None
    offset: NDArray[np.floating]
    residual_scale: NDArray[np.floating]
    factor: NDArray[np.floating] | None
    fixed_mean: NDArray[np.floating] | None

    @classmethod
    def prepare(
        cls,
        result: NlmerResult,
        include_re: bool,
        *,
        cache_fixed_mean: bool = False,
        cache_group_rows: bool = True,
    ) -> _NlmerSimulation:
        n_groups = len(result.group_levels)
        n_random = len(result.random_params)
        rows = (
            _grouped_observation_indices(result.groups, n_groups=n_groups)
            if cache_group_rows
            else None
        )
        offset = result.offset(copy=False)
        residual_scale = result.sigma / np.sqrt(result.weights(copy=False))
        factor = None
        fixed_mean = None
        if include_re and n_random > 0:
            covariance = _build_psi_matrix(result.theta, n_random) * result.sigma**2
            covariance = covariance + _COV_REGULARIZATION * np.eye(n_random)
            # Preserve RandomState.multivariate_normal's transform and draw order.
            _, singular_values, vectors = np.linalg.svd(covariance)
            factor = np.sqrt(singular_values)[:, None] * vectors
        elif cache_fixed_mean:
            fixed_mean = (
                result._conditional_mean(
                    random_effects=np.zeros((n_groups, n_random)), group_rows=rows
                )
                + offset
            )
        return cls(result, rows, offset, residual_scale, factor, fixed_mean)

    def draw(self, rng: RandomStream) -> NDArray[np.floating]:
        mean = self.fixed_mean
        if mean is None:
            shape = (len(self.result.group_levels), len(self.result.random_params))
            effects = (
                rng.standard_normal(shape) @ self.factor
                if self.factor is not None
                else np.zeros(shape)
            )
            mean = (
                self.result._conditional_mean(random_effects=effects, group_rows=self.group_rows)
                + self.offset
            )
        return mean + rng.standard_normal(len(self.residual_scale)) * self.residual_scale


@dataclass
class NlmerVarCorr:
    groups: dict[str, dict[str, float]]
    residual: float

    def __str__(self) -> str:
        lines = ["Random effects:"]
        lines.append(" Groups      Name         Variance  Std.Dev.")
        for group, terms in self.groups.items():
            for i, (name, var) in enumerate(terms.items()):
                grp_name = group if i == 0 else ""
                lines.append(f" {grp_name:11} {name:12} {var:9.4f}  {np.sqrt(var):.4f}")
        lines.append(
            f" {'Residual':11} {' ':12} {self.residual:9.4f}  {np.sqrt(self.residual):.4f}"
        )
        return "\n".join(lines)


@dataclass
class NlmerResult(_ResultBase):
    _IS_NLMM: ClassVar[bool] = True
    model: NonlinearModel
    group_var: str
    phi: NDArray[np.floating]
    theta: NDArray[np.floating]
    sigma: float
    b: NDArray[np.floating]
    random_params: list[int]
    deviance: float
    converged: bool
    n_iter: int
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    groups: NDArray[np.integer]
    group_levels: list[str]
    _weights: NDArray[np.floating] | None = field(default=None, repr=False)
    _offset: NDArray[np.floating] | None = field(default=None, repr=False)
    _data: pd.DataFrame | None = field(default=None, repr=False)
    _x_var: str = field(default="x", repr=False)
    _y_var: str = field(default="y", repr=False)
    pnls_converged: bool = True
    pnls_maxiter: int = 50
    pnls_tol: float = 1e-6

    def fixef(self) -> dict[str, float]:
        from mixedlm.utils.names import _check_unique_coefficient_names

        _check_unique_coefficient_names(
            self.model.param_names,
            alternative="Use tidy() or phi to inspect coefficients by position.",
        )
        return dict(zip(self.model.param_names, self.phi, strict=False))

    def ranef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        random_param_names = [self.model.param_names[i] for i in self.random_params]

        term_ranefs: dict[str, NDArray[np.floating]] = {}
        for j, name in enumerate(random_param_names):
            term_ranefs[name] = self.b[:, j]

        return {self.group_var: term_ranefs}

    def coef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        n_groups = self.b.shape[0]

        group_coef: dict[str, NDArray[np.floating]] = {}
        for j, p_idx in enumerate(self.random_params):
            name = self.model.param_names[p_idx]
            group_coef[name] = self.b[:, j] + self.phi[p_idx]

        for i, name in enumerate(self.model.param_names):
            if i not in self.random_params:
                group_coef[name] = np.full(n_groups, self.phi[i])

        return {self.group_var: group_coef}

    def _conditional_mean(
        self,
        phi: NDArray[np.floating] | None = None,
        random_effects: NDArray[np.floating] | None = None,
        *,
        group_rows: Sequence[NDArray] | None = None,
    ) -> NDArray[np.floating]:
        """Evaluate the nonlinear mean without the observation offset."""
        base_params = self.phi if phi is None else phi
        effects = self.b if random_effects is None else random_effects
        pred = np.zeros(len(self.y), dtype=np.float64)
        n_groups = len(self.group_levels)
        # Direct masks avoid sorting overhead when there are very few groups.
        if group_rows is None and n_groups >= 8:
            group_rows = _grouped_observation_indices(self.groups, n_groups=len(self.group_levels))

        for group_idx in range(n_groups):
            rows = self.groups == group_idx if group_rows is None else group_rows[group_idx]
            params = base_params.copy()
            for effect_idx, param_idx in enumerate(self.random_params):
                params[param_idx] += effects[group_idx, effect_idx]
            pred[rows] = self.model.predict(params, self.x[rows])

        return pred

    def fitted(self) -> NDArray[np.floating]:
        return self._conditional_mean() + self.offset(copy=False)

    def residuals(self, type: str = "response") -> NDArray[np.floating]:
        fitted = self.fitted()
        if type == "response":
            return self.y - fitted
        elif type == "pearson":
            return np.sqrt(self.weights(copy=False)) * (self.y - fitted) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        x_var: str | None = None,
        group_var: str | None = None,
        *,
        offset: ArrayLike | str | None = None,
    ) -> NDArray[np.floating]:
        """Predict responses for new observations.

        Parameters
        ----------
        newdata : DataFrame, optional
            New observations. If omitted, return fitted values for the
            original data.
        x_var : str, optional
            Predictor column. Defaults to the column used when fitting.
        group_var : str, optional
            Grouping column used to add fitted random effects for known
            levels. Unknown levels receive population-level predictions.
        offset : array-like, scalar, or str, optional
            Known offset added to new-data response predictions. A string selects
            a column from ``newdata``. Scalars apply to every row; arrays must
            supply one finite real value per row. Omitted offsets default to zero
            for new data. Without ``newdata``, fitted offsets are already included
            and an explicit offset is not accepted.

        Returns
        -------
        NDArray
            Predicted responses in the same row order as ``newdata``.
        """
        if newdata is None:
            if offset is not None:
                raise ValueError("Prediction offset can only be supplied with newdata.")
            return self.fitted()

        from mixedlm.models.shared_utils import resolve_prediction_vector

        prediction_offset = (
            None
            if offset is None
            else resolve_prediction_vector(newdata, offset, name="offset", default=0.0)
        )
        if x_var is None:
            x_var = self._x_var
        x_new = newdata[x_var].to_numpy(dtype=np.float64)
        n_new = len(x_new)

        if group_var is None or group_var not in newdata.columns:
            pred = self.model.predict(self.phi, x_new)
        else:
            groups_new = newdata[group_var].astype(str).tolist()
            group_lookup = {group: index for index, group in enumerate(self.group_levels)}
            rows_by_group: dict[int | None, list[int]] = {}
            for row, group in enumerate(groups_new):
                rows_by_group.setdefault(group_lookup.get(group), []).append(row)

            pred = np.empty(n_new, dtype=np.float64)
            for group_index, rows in rows_by_group.items():
                row_indices = np.asarray(rows, dtype=np.intp)
                params = self.phi
                if group_index is not None:
                    params = self.phi.copy()
                    params[self.random_params] += self.b[group_index]
                pred[row_indices] = self.model.predict(params, x_new[row_indices])

        return pred if prediction_offset is None else pred + prediction_offset

    def VarCorr(self) -> NlmerVarCorr:
        n_random = len(self.random_params)
        Psi = _build_psi_matrix(self.theta, n_random)

        random_param_names = [self.model.param_names[i] for i in self.random_params]

        term_vars: dict[str, float] = {}
        for i, name in enumerate(random_param_names):
            term_vars[name] = Psi[i, i] * self.sigma**2

        groups = {self.group_var: term_vars}

        return NlmerVarCorr(groups=groups, residual=self.sigma**2)

    def logLik(self) -> LogLik:
        return LogLik(
            value=-0.5 * self.deviance,
            df=self.npar(),
            nobs=self.nobs(),
            REML=False,
        )

    def as_function(
        self,
        type: str = "predict",
    ) -> object:
        """Return the model's prediction function.

        Parameters
        ----------
        type : str, default "predict"
            Type of function to return. For NlmerResult, only "predict"
            is supported.

        Returns
        -------
        callable
            A function that takes x values and returns predictions.
        """
        if type == "predict":
            model = self.model
            phi = self.phi

            def predict_fn(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return model.predict(phi, x)

            return predict_fn
        else:
            raise ValueError(f"Unknown type: {type}. Use 'predict'.")

    def npar(self) -> int:
        """Get the number of parameters in the model.

        Returns the total number of estimated parameters:
        - Fixed effects (phi)
        - Variance-covariance parameters (theta)
        - Residual standard deviation (sigma)

        Returns
        -------
        int
            Total number of parameters.
        """
        n_fixed = len(self.phi)
        n_theta = len(self.theta)
        n_sigma = 1
        return n_fixed + n_theta + n_sigma

    def df_residual(self) -> int:
        """Get the residual degrees of freedom.

        Returns n - p where n is the number of observations
        and p is the number of fixed effect parameters.

        Returns
        -------
        int
            Residual degrees of freedom.
        """
        n = len(self.y)
        p = len(self.phi)
        return n - p

    def nobs(self) -> int:
        """Get the number of observations."""
        return len(self.y)

    def ngrps(self) -> dict[str, int]:
        """Get the number of levels for each grouping factor."""
        return {self.group_var: len(self.group_levels)}

    def weights(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the model weights.

        Returns the strictly positive prior weights used in model fitting.
        Conditional residual variance is ``sigma**2 / weights``. If no
        weights were specified, returns an array of ones.
        """
        if self._weights is not None:
            return self._weights.copy() if copy else self._weights
        return np.ones(len(self.y), dtype=np.float64)

    def offset(self, copy: bool = True) -> NDArray[np.floating]:
        """Get the model offset.

        Returns the offset used in model fitting.
        If no offset was specified, returns an array of zeros.
        """
        if self._offset is not None:
            return self._offset.copy() if copy else self._offset
        return np.zeros(len(self.y), dtype=np.float64)

    def model_frame(self) -> pd.DataFrame:
        """Get the model frame (data used for fitting)."""
        import pandas as pd

        if self._data is not None:
            return self._data.copy()
        return pd.DataFrame({self._x_var: self.x, self._y_var: self.y, self.group_var: self.groups})

    def simulate(
        self,
        nsim: int = 1,
        seed: int | np.random.RandomState | np.random.Generator | None = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        """Simulate responses from the fitted model.

        Parameters
        ----------
        nsim : int, default 1
            Nonnegative integer number of simulations.
        seed : int, RandomState, or Generator, optional
            Local random seed or stream. Integer seeds preserve the legacy
            draw sequence. A supplied stream advances across calls. None
            creates an independent stream without changing NumPy's global
            random state.
        use_re : bool, default True
            If True, simulate new random effects. If False, use fixed effects only.
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" to exclude random effects.

        Returns
        -------
        NDArray
            Simulated responses. Shape (n,) if nsim=1, else (n, nsim).
        """
        if isinstance(nsim, bool | np.bool_) or not isinstance(nsim, Integral):
            raise TypeError("nsim must be a nonnegative integer")
        if nsim < 0:
            raise ValueError("nsim must be a nonnegative integer")
        n = len(self.y)
        if nsim == 0:
            return np.empty((n, 0), dtype=np.float64)
        rng = (
            seed
            if isinstance(seed, np.random.RandomState | np.random.Generator)
            else np.random.RandomState(seed)
        )
        include_re = use_re and re_form not in ("~0", "NA")
        prepared = _NlmerSimulation.prepare(
            self, include_re, cache_fixed_mean=True, cache_group_rows=nsim > 1
        )
        simulations = np.empty((n, nsim), dtype=np.float64)
        for i in range(nsim):
            simulations[:, i] = prepared.draw(rng)

        return simulations[:, 0] if nsim == 1 else simulations

    def refit(
        self,
        newresp: NDArray[np.floating] | None = None,
        **kwargs,
    ) -> NlmerResult:
        """Refit the model with a new response vector.

        Parameters
        ----------
        newresp : array-like, optional
            New response vector. Must have the same length as the original.
            If None, refits with the original response.
        **kwargs
            Additional optimizer arguments. ``pnls_maxiter`` and ``pnls_tol``
            default to the fitted model's inner controls.

        Returns
        -------
        NlmerResult
            New fitted model result.
        """
        if newresp is None:
            newresp = self.y
        else:
            newresp = np.asarray(newresp, dtype=np.float64)
            if len(newresp) != len(self.y):
                raise ValueError(f"newresp has length {len(newresp)}, expected {len(self.y)}")

        adjusted_response = newresp - self.offset(copy=False)
        optimizer = NLMMOptimizer(
            adjusted_response,
            self.x,
            self.groups,
            self.model,
            self.random_params,
            verbose=0,
            weights=self._weights,
            pnls_maxiter=kwargs.pop("pnls_maxiter", self.pnls_maxiter),
            pnls_tol=kwargs.pop("pnls_tol", self.pnls_tol),
        )

        start_phi = kwargs.pop("start", self.phi)
        method = kwargs.pop("method", "L-BFGS-B")
        maxiter = kwargs.pop("maxiter", _DEFAULT_MAXITER)

        opt_result = optimizer.optimize(
            start_theta=self.theta,
            start_phi=start_phi,
            start_b=self.b,
            start_sigma=self.sigma,
            method=method,
            maxiter=maxiter,
        )

        refit_data = None
        if self._data is not None:
            refit_data = self._data.copy()
            refit_data[self._y_var] = newresp

        return NlmerResult(
            model=self.model,
            group_var=self.group_var,
            phi=opt_result.phi,
            theta=opt_result.theta,
            sigma=opt_result.sigma,
            b=opt_result.b,
            random_params=self.random_params,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            pnls_converged=opt_result.pnls_converged,
            pnls_maxiter=optimizer.pnls_maxiter,
            pnls_tol=optimizer.pnls_tol,
            x=self.x,
            y=newresp,
            groups=self.groups,
            group_levels=self.group_levels,
            _weights=self._weights,
            _offset=self._offset,
            _data=refit_data,
            _x_var=self._x_var,
            _y_var=self._y_var,
        )

    def update(
        self,
        data: pd.DataFrame | None = None,
        start: dict[str, float] | None = None,
        **kwargs,
    ) -> NlmerResult:
        """Update and refit the model with new data or parameters.

        Parameters
        ----------
        data : DataFrame, optional
            New data. If None, uses the original data.
        start : dict, optional
            Starting values for parameters.
        **kwargs
            Additional arguments passed to nlmer().

        Returns
        -------
        NlmerResult
            New fitted model result.
        """
        if data is None:
            if self._data is not None:
                data = self._data
            else:
                import pandas as pd

                data = pd.DataFrame(
                    {
                        self._x_var: self.x,
                        self._y_var: self.y,
                        self.group_var: [self.group_levels[g] for g in self.groups],
                    }
                )

        random_param_names = [self.model.param_names[i] for i in self.random_params]

        kwargs.setdefault("pnls_maxiter", self.pnls_maxiter)
        kwargs.setdefault("pnls_tol", self.pnls_tol)
        return nlmer(
            model=self.model,
            data=data,
            x_var=self._x_var,
            y_var=self._y_var,
            group_var=self.group_var,
            random_params=random_param_names,
            start=start,
            weights=self._weights,
            offset=self._offset,
            **kwargs,
        )

    def _invert_with_fallback(self, matrix: NDArray[np.floating]) -> NDArray[np.floating]:
        """Invert a potentially ill-conditioned matrix with finite-value guards."""
        n = matrix.shape[0]
        eye = np.eye(n, dtype=np.float64)
        sym_matrix = 0.5 * (matrix + matrix.T)

        candidates = (sym_matrix, sym_matrix + _COV_REGULARIZATION * eye)
        for candidate in candidates:
            try:
                L = linalg.cholesky(candidate, lower=True)
                inv = linalg.cho_solve((L, True), eye)
                if np.all(np.isfinite(inv)):
                    return 0.5 * (inv + inv.T)
            except linalg.LinAlgError:
                pass

            try:
                inv = linalg.solve(candidate, eye, assume_a="sym")
                if np.all(np.isfinite(inv)):
                    return 0.5 * (inv + inv.T)
            except (linalg.LinAlgError, ValueError):
                pass

        inv = linalg.pinv(sym_matrix + _COV_REGULARIZATION * eye)
        if not np.all(np.isfinite(inv)):
            inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * (inv + inv.T)

    def vcov(self) -> NDArray[np.floating]:
        """Compute the variance-covariance matrix of fixed effects.

        Uses numerical approximation based on the Hessian of the log-likelihood.

        Returns
        -------
        NDArray
            Variance-covariance matrix of shape (n_params, n_params).
        """
        n_params = len(self.phi)
        eps = _VCOV_EPS
        adjusted_response = self.y - self.offset(copy=False)

        def neg_log_lik(phi_vec):
            pred = self._conditional_mean(phi=phi_vec)
            resid = adjusted_response - pred
            return 0.5 * np.dot(self.weights(copy=False), resid**2) / self.sigma**2

        hessian = np.zeros((n_params, n_params), dtype=np.float64)

        for i in range(n_params):
            for j in range(i, n_params):
                phi_pp = self.phi.copy()
                phi_pm = self.phi.copy()
                phi_mp = self.phi.copy()
                phi_mm = self.phi.copy()

                phi_pp[i] += eps
                phi_pp[j] += eps
                phi_pm[i] += eps
                phi_pm[j] -= eps
                phi_mp[i] -= eps
                phi_mp[j] += eps
                phi_mm[i] -= eps
                phi_mm[j] -= eps

                d2f = (
                    neg_log_lik(phi_pp)
                    - neg_log_lik(phi_pm)
                    - neg_log_lik(phi_mp)
                    + neg_log_lik(phi_mm)
                ) / (4 * eps * eps)

                hessian[i, j] = d2f
                hessian[j, i] = d2f

        vcov = self._invert_with_fallback(hessian)
        vcov = np.nan_to_num(vcov, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(vcov, np.maximum(np.diag(vcov), 0.0))
        return vcov

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "boot",
        n_boot: int = _DEFAULT_N_BOOT,
        seed: RandomSeed = None,
        *,
        n_jobs: int = 1,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for fixed effects.

        Parameters
        ----------
        parm : str or list of str, optional
            Parameter names. If None, computes for all parameters.
        level : float, default 0.95
            Confidence level.
        method : str, default "boot"
            Method for computing confidence intervals. Options:
            - "boot": Bootstrap (recommended for NLMMs)
            - "Wald": Wald intervals based on vcov (less accurate)
        n_boot : int, default 1000
            Positive integer number of bootstrap samples (if method="boot").
        seed : int, RandomState, or Generator, optional
            Local random seed or reusable stream.
        n_jobs : int, default 1
            Positive bootstrap refit worker count, or -1 for available CPUs.
            Used only with method="boot".

        Returns
        -------
        dict
            Dictionary mapping parameter names to (lower, upper) tuples.

        Notes
        -----
        Bootstrap intervals exclude failed samples. If every sample fails,
        the bounds are NaN. Use ``bootstrap_nlmer()`` to inspect sample arrays
        and the failure count before interpreting the intervals.
        """
        from scipy import stats

        level = _validate_confidence_level(level)
        if parm is None:
            parm = self.model.param_names
        elif isinstance(parm, str):
            parm = [parm]

        from mixedlm.utils.names import _check_unique_coefficient_names

        _check_unique_coefficient_names(
            self.model.param_names,
            None if method == "boot" else parm,
            alternative="Use tidy(conf_int=True) for intervals in coefficient order.",
        )

        if method == "Wald":
            vcov = self.vcov()
            alpha = 1 - level
            z_crit = stats.norm.isf(alpha / 2)

            result: dict[str, tuple[float, float]] = {}
            for p in parm:
                if p not in self.model.param_names:
                    continue
                idx = self.model.param_names.index(p)
                se = np.sqrt(vcov[idx, idx])
                lower = self.phi[idx] - z_crit * se
                upper = self.phi[idx] + z_crit * se
                result[p] = (float(lower), float(upper))
            return result

        elif method == "boot":
            from mixedlm.inference.bootstrap import _validate_ci_options, bootstrap_nlmer

            _validate_ci_options(level, "percentile")
            boot = bootstrap_nlmer(self, n_boot=n_boot, seed=seed, n_jobs=n_jobs)
            intervals = boot.ci(level=level)
            return {p: intervals[p] for p in parm if p in intervals}

        else:
            raise ValueError(f"Unknown method: {method}. Use 'Wald' or 'boot'.")

    def hatvalues(self) -> NDArray[np.floating]:
        """Compute leverage values (diagonal of the hat matrix).

        For nonlinear models, this is approximated using the Jacobian
        of the fitted values with respect to the response.

        Returns
        -------
        NDArray
            Leverage values for each observation.
        """
        n = len(self.y)
        n_params = len(self.phi)

        J = np.zeros((n, n_params), dtype=np.float64)
        eps = _JACOBIAN_EPS

        fitted_base = self._conditional_mean()

        for j in range(n_params):
            phi_plus = self.phi.copy()
            phi_plus[j] += eps

            pred_plus = self._conditional_mean(phi=phi_plus)

            J[:, j] = (pred_plus - fitted_base) / eps

        sqrt_weights = np.sqrt(self.weights(copy=False))
        weighted_J = sqrt_weights[:, None] * J
        JtWJ = weighted_J.T @ weighted_J
        JtWJ_inv = self._invert_with_fallback(JtWJ)
        J_JtWJ_inv = weighted_J @ JtWJ_inv
        h = np.sum(J_JtWJ_inv * weighted_J, axis=1)

        if not np.all(np.isfinite(h)):
            h = np.sum(weighted_J**2, axis=1) / (np.sum(weighted_J**2) + _HAT_FALLBACK_EPS)

        h = np.nan_to_num(h, nan=0.0, posinf=_HAT_CLIP_MAX, neginf=0.0)
        return np.clip(h, 0, _HAT_CLIP_MAX)

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        Uses the linear and generalized models' formula with the weighted
        response residuals, the Jacobian leverages from ``hatvalues()`` and
        the number of fixed-effect parameters.

        Returns
        -------
        NDArray
            Cook's distance for each observation.
        """
        from mixedlm.diagnostics.influence import _cooks_distance

        resid = self.residuals(type="pearson") * self.sigma
        return _cooks_distance(resid, self.hatvalues(), len(self.phi), self.sigma)

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values
            - 'cooks_d': Cook's distance
            - 'std_resid': Standardized residuals
        """
        h = self.hatvalues()
        resid = self.residuals(type="pearson") * self.sigma
        h_safe = np.clip(h, 0, _HAT_CLIP_MAX)
        std_resid = resid / (self.sigma * np.sqrt(1 - h_safe))

        return {
            "hat": h,
            "cooks_d": self.cooks_distance(),
            "std_resid": std_resid,
        }

    def getME(self, name: str):
        """Extract model components by name.

        Parameters
        ----------
        name : str
            Name of the component to extract. Valid names:
            - "phi": Fixed effects parameters
            - "theta": Variance component parameters
            - "sigma": Residual standard deviation
            - "b": Random effects matrix
            - "y": Response vector
            - "x": Predictor vector
            - "groups": Group indices
            - "n" or "n_obs": Number of observations
            - "n_groups": Number of groups
            - "deviance": Model deviance
            - "weights": Prior weights
            - "offset": Offset term

        Returns
        -------
        The requested component.
        """
        if name == "phi":
            return self.phi.copy()
        elif name == "theta":
            return self.theta.copy()
        elif name == "sigma":
            return self.sigma
        elif name == "b":
            return self.b.copy()
        elif name == "y":
            return self.y.copy()
        elif name == "x":
            return self.x.copy()
        elif name == "groups":
            return self.groups.copy()
        elif name in ("n", "n_obs"):
            return len(self.y)
        elif name == "n_groups":
            return len(self.group_levels)
        elif name == "deviance":
            return self.deviance
        elif name == "weights":
            return self.weights()
        elif name == "offset":
            return self.offset()
        elif name == "group_levels":
            return list(self.group_levels)
        elif name == "random_params":
            return list(self.random_params)
        else:
            valid_names = [
                "phi",
                "theta",
                "sigma",
                "b",
                "y",
                "x",
                "groups",
                "n",
                "n_obs",
                "n_groups",
                "deviance",
                "weights",
                "offset",
                "group_levels",
                "random_params",
            ]
            raise ValueError(f"Unknown component: '{name}'. Valid names: {valid_names}")

    def isSingular(self, tol: float = _DEFAULT_SINGULAR_TOL) -> bool:
        """Check if the model has a singular (boundary) fit.

        Parameters
        ----------
        tol : float, default 1e-4
            Tolerance on the standard deviations of the relative random-effect
            covariance; correlations near zero are not singular.

        Returns
        -------
        bool
            True if the random-effect covariance has an eigenvalue below tol**2.
        """
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite, non-negative number")
        psi = _build_psi_matrix(self.theta, len(self.random_params))
        return bool(np.min(np.linalg.eigvalsh(psi), initial=np.inf) < tol**2)

    def summary(self) -> str:
        lines = []
        lines.append("Nonlinear mixed model fit by maximum likelihood")
        lines.append(f" Model: {self.model.name}")
        lines.append("")

        lines.append("     AIC      BIC   logLik deviance")
        lines.append(
            f"{self.AIC():8.1f} {self.BIC():8.1f} {self.logLik():8.1f} {self.deviance:8.1f}"
        )
        lines.append("")

        lines.append(str(self.VarCorr()))
        lines.append(f"Number of obs: {len(self.y)}")
        lines.append(f"  groups:  {self.group_var}, {len(self.group_levels)}")
        lines.append("")

        lines.append("Fixed effects:")
        lines.append("             Estimate")
        for name, val in zip(self.model.param_names, self.phi, strict=False):
            lines.append(f"{name:12} {val:10.4f}")

        lines.append("")
        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")

        if not self.pnls_converged:
            lines.append("inner PNLS convergence: no")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"NlmerResult(model={self.model.name}, deviance={self.deviance:.4f})"


# Retain the original method identity so custom simulation overrides remain active.
_DEFAULT_NLMER_SIMULATE = NlmerResult.simulate


class NlmerMod:
    def __init__(
        self,
        model: NonlinearModel,
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        group_var: str,
        random_params: list[str] | list[int] | None = None,
        start: dict[str, float] | None = None,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.x_var = x_var
        self.y_var = y_var
        self.group_var = group_var
        self.verbose = verbose

        self.x = data[x_var].to_numpy(dtype=np.float64)
        self.y = data[y_var].to_numpy(dtype=np.float64)
        self.weights: NDArray[np.floating] | None = None
        self.offset: NDArray[np.floating] | None = None

        if weights is not None:
            self.weights = _as_prior_weights(weights, len(self.y)).copy()
        else:
            self.weights = None

        if offset is not None:
            self.offset = np.asarray(offset, dtype=np.float64)
            if len(self.offset) != len(self.y):
                raise ValueError(f"offset has length {len(self.offset)}, expected {len(self.y)}")
            self._adjusted_y = self.y - self.offset
        else:
            self.offset = None
            self._adjusted_y = self.y

        group_col = data[group_var].astype(str)
        self.group_levels = sorted(group_col.unique().tolist())
        level_map = {lv: i for i, lv in enumerate(self.group_levels)}
        self.groups = np.array([level_map[g] for g in group_col], dtype=np.int64)

        if random_params is None:
            self.random_params = list(range(model.n_params))
        elif isinstance(random_params[0], str):
            names = cast(list[str], random_params)
            self.random_params = [model.param_names.index(p) for p in names]
        else:
            self.random_params = list(cast(list[int], random_params))

        self.start_phi: NDArray[np.floating]
        if start is not None:
            self.start_phi = np.array(
                [start.get(name, 1.0) for name in model.param_names],
                dtype=np.float64,
            )
        else:
            self.start_phi = model.get_start(self.x, self._adjusted_y)

    def fit(
        self,
        method: str = "L-BFGS-B",
        maxiter: int = _DEFAULT_MAXITER,
        *,
        pnls_maxiter: int | None = None,
        pnls_tol: float = 1e-6,
    ) -> NlmerResult:
        optimizer = NLMMOptimizer(
            self._adjusted_y,
            self.x,
            self.groups,
            self.model,
            self.random_params,
            verbose=self.verbose,
            weights=self.weights,
            pnls_maxiter=pnls_maxiter,
            pnls_tol=pnls_tol,
        )

        opt_result = optimizer.optimize(
            start_phi=self.start_phi,
            method=method,
            maxiter=maxiter,
        )

        if not opt_result.pnls_converged:
            warnings.warn(
                "The inner PNLS solver did not converge; increase pnls_maxiter "
                "or review starting values and pnls_tol before using the fit.",
                UserWarning,
                stacklevel=2,
            )

        return NlmerResult(
            model=self.model,
            group_var=self.group_var,
            phi=opt_result.phi,
            theta=opt_result.theta,
            sigma=opt_result.sigma,
            b=opt_result.b,
            random_params=self.random_params,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            pnls_converged=opt_result.pnls_converged,
            pnls_maxiter=optimizer.pnls_maxiter,
            pnls_tol=optimizer.pnls_tol,
            x=self.x,
            y=self.y,
            groups=self.groups,
            group_levels=self.group_levels,
            _weights=self.weights,
            _offset=self.offset,
            _data=self.data,
            _x_var=self.x_var,
            _y_var=self.y_var,
        )


def nlmer(
    model: NonlinearModel,
    data: pd.DataFrame,
    x_var: str,
    y_var: str,
    group_var: str,
    random_params: list[str] | list[int] | None = None,
    start: dict[str, float] | None = None,
    verbose: int = 0,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    **kwargs,
) -> NlmerResult:
    """Fit a nonlinear mixed-effects model.

    Parameters
    ----------
    model : NonlinearModel
        The nonlinear model specification (e.g., SSasymp, SSlogis).
    data : DataFrame
        Data containing the variables for fitting.
    x_var : str
        Name of the predictor variable column.
    y_var : str
        Name of the response variable column.
    group_var : str
        Name of the grouping factor column.
    random_params : list, optional
        Which parameters have random effects. If None, all parameters
        have random effects. Can be parameter names or indices.
    start : dict, optional
        Starting values for parameters. If None, uses automatic
        initialization from the model.
    verbose : int, default 0
        Verbosity level for optimization output.
    weights : array-like, optional
        Strictly positive prior weights for observations. Conditional
        residual variance is inversely proportional to the weights.
    offset : array-like, optional
        Known offset added to the fitted nonlinear mean.
    **kwargs
        Additional optimizer arguments: ``method`` and ``maxiter`` for outer
        covariance optimization, and ``pnls_maxiter`` (default 50) and
        ``pnls_tol`` (default 1e-6) for the inner parameter updates. The inner
        tolerance bounds the largest absolute proposed fixed or random parameter update.

    Returns
    -------
    NlmerResult
        Fitted model result.

    Examples
    --------
    >>> from mixedlm.nlme.models import SSasymp
    >>> model = SSasymp()
    >>> result = nlmer(model, data, x_var="time", y_var="conc", group_var="subject")
    """
    mod = NlmerMod(
        model=model,
        data=data,
        x_var=x_var,
        y_var=y_var,
        group_var=group_var,
        random_params=random_params,
        start=start,
        verbose=verbose,
        weights=weights,
        offset=offset,
    )
    return mod.fit(**kwargs)
