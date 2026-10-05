from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mixedlm.inference.allfit import AllFitResult
    from mixedlm.inference.bootstrap import BootstrapResult
    from mixedlm.inference.drop1 import Drop1Result
    from mixedlm.inference.profile_types import ProfileResult
    from mixedlm.utils.random import RandomSeed

from mixedlm.estimation.laplace import GLMMOptimizer, _build_lambda
from mixedlm.families.base import Family
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, _restore_binomial_factor
from mixedlm.models.control import GlmerControl
from mixedlm.models.lmer_types import LogLik, PredictResult, VarCorrGroup
from mixedlm.models.result_mixin import MerResultMixin
from mixedlm.models.shared_utils import (
    _RandomEffectFactor,
    dense_quadratic_form_diagonal,
    symmetric_inverse,
)
from mixedlm.utils import _format_pvalue, _get_signif_code
from mixedlm.utils.simulation import simulate_glmm_response
from mixedlm.utils.validation import _validate_confidence_level


@dataclass
class GlmerVarCorr:
    groups: dict[str, VarCorrGroup]

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
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_groups = len(self.groups)
        return f"GlmerVarCorr({n_groups} groups)"

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {name: group.variance for name, group in self.groups.items()}

    def get_cov(self, group: str) -> NDArray[np.floating]:
        return self.groups[group].cov

    def get_corr(self, group: str) -> NDArray[np.floating] | None:
        return self.groups[group].corr


@dataclass(frozen=True)
class _GLMMRandomSystem:
    """Final working weights and sparse random-effect information."""

    weights: NDArray[np.floating]
    Lambda: sparse.csc_matrix
    weighted_Z: sparse.csc_matrix
    precision: sparse.csc_matrix


@dataclass(frozen=True)
class _GLMMProjection:
    weights: NDArray[np.floating]
    Lambda: sparse.csc_matrix
    random_factor: _RandomEffectFactor | None
    spherical_cross: NDArray[np.float64]
    random_fixed_map: NDArray[np.float64]
    fixed_information: NDArray[np.floating]
    weighted_X: NDArray[np.float64]
    weighted_Z: sparse.csc_matrix
    information_inv: NDArray[np.float64]

    @property
    def random_cholesky(self) -> NDArray[np.float64]:
        if self.random_factor is None:
            return np.empty((0, 0), dtype=np.float64)
        return self.random_factor.cholesky

    @cached_property
    def RZX(self) -> NDArray[np.float64]:
        if self.random_factor is None:
            return self.spherical_cross.copy()
        return linalg.solve_triangular(self.random_cholesky, self.spherical_cross, lower=True)


@dataclass
class GlmerResult(MerResultMixin):
    """Fitted GLMM; joint_fit identifies the joint theta/beta likelihood objective."""

    _IS_GLMM: ClassVar[bool] = True
    formula: Formula
    matrices: ModelMatrices
    family: Family
    theta: NDArray[np.floating]
    beta: NDArray[np.floating]
    u: NDArray[np.floating]
    deviance: float
    converged: bool
    n_iter: int
    nAGQ: int
    pirls_converged: bool = True
    pirls_maxiter: int | None = None
    pirls_tol: float = 1e-6
    joint_fit: bool = False
    message: str = ""
    optimizer: str = ""

    def _refit_control(self) -> GlmerControl:
        """Carry the fitted inner settings into formula-based refitting paths."""
        return GlmerControl(tolPwrss=self.pirls_tol, pirls_maxiter=self.pirls_maxiter)

    def _compute_condVar(
        self, include_cov: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]]:
        from mixedlm.utils.variance import _conditional_variance_blocks

        q = self.matrices.n_random
        if q == 0:
            return {}

        system = self._working_random_system

        return _conditional_variance_blocks(
            system.precision,
            system.Lambda,
            self.matrices.random_structures,
            include_cov=include_cov,
        )

    @property
    def sigma(self) -> float:
        return 1.0

    def get_family(self) -> Family:
        """Get the GLM family.

        Returns the family object used for the generalized
        linear mixed model, including the link function.

        Returns
        -------
        Family
            The GLM family (e.g., Binomial, Poisson, Gaussian).

        Examples
        --------
        >>> result = glmer("y ~ x + (1|g)", data, family=Binomial())
        >>> result.get_family()
        Binomial(link=logit)
        >>> result.get_family().link.name
        'logit'
        """
        return self.family

    @cached_property
    def _linear_predictor(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    @cached_property
    def _working_random_system(self) -> _GLMMRandomSystem:
        """Build sparse random-effect information without a dense factorization."""
        mu = self.family.link.inverse(self._linear_predictor)
        mu = self.family.clamp_mu(mu)
        weights = np.clip(
            self.family.weights(mu) * self.matrices.weights,
            1e-10,
            1e10,
        )
        weighted_Z = self.matrices.Z.multiply(np.sqrt(weights)[:, None]).tocsc()
        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        precision = Lambda.T @ (weighted_Z.T @ weighted_Z) @ Lambda
        precision = (precision + sparse.eye(self.matrices.n_random, format="csc")).tocsc()
        return _GLMMRandomSystem(weights, Lambda, weighted_Z, precision)

    @cached_property
    def _working_projection(self) -> _GLMMProjection:
        """Final PIRLS projection in spherical random-effect coordinates."""
        X = self.matrices.X
        q = self.matrices.n_random
        system = self._working_random_system
        weights = system.weights
        sqrt_weights = np.sqrt(weights)
        WX = sqrt_weights[:, None] * X
        WZ = system.weighted_Z
        XtWX = WX.T @ WX
        Lambda = system.Lambda

        if q == 0:
            return _GLMMProjection(
                weights=weights,
                Lambda=Lambda,
                random_factor=None,
                spherical_cross=np.empty((0, X.shape[1]), dtype=np.float64),
                random_fixed_map=np.empty((0, X.shape[1]), dtype=np.float64),
                fixed_information=np.asarray(XtWX),
                weighted_X=np.asarray(WX),
                weighted_Z=WZ,
                information_inv=symmetric_inverse(XtWX),
            )

        random_factor = _RandomEffectFactor(system.precision, jitter=1e-6)

        XtWZ = WX.T @ WZ
        XtWZ_dense = XtWZ.toarray() if sparse.issparse(XtWZ) else np.asarray(XtWZ)
        spherical_cross = np.asarray(Lambda.T @ XtWZ_dense.T)
        random_fixed_map, correction = random_factor.solve_with_crossproduct(spherical_cross)
        fixed_information = np.asarray(XtWX - correction)
        fixed_information = 0.5 * (fixed_information + fixed_information.T)

        return _GLMMProjection(
            weights=weights,
            Lambda=Lambda,
            random_factor=random_factor,
            spherical_cross=spherical_cross,
            random_fixed_map=random_fixed_map,
            fixed_information=fixed_information,
            weighted_X=np.asarray(WX),
            weighted_Z=WZ,
            information_inv=symmetric_inverse(fixed_information),
        )

    def linear_predictor(self, na_expand: bool = True) -> NDArray[np.floating]:
        values = self._linear_predictor
        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(values)
        return values

    def fitted(self, type: str = "response", na_expand: bool = True) -> NDArray[np.floating]:
        """Get fitted values.

        Parameters
        ----------
        type : str, default "response"
            Type of fitted values: "response" (mean) or "link" (linear predictor).
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Fitted values.
        """
        eta = self._linear_predictor
        values = eta if type == "link" else self.family.link.inverse(eta)

        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(values)
        return values

    def residuals(self, type: str = "deviance", na_expand: bool = True) -> NDArray[np.floating]:
        """Get residuals.

        Parameters
        ----------
        type : str, default "deviance"
            Type of residuals: "response", "pearson", or "deviance".
        na_expand : bool, default True
            If True and na_action="exclude", expand to original length with NA.

        Returns
        -------
        NDArray
            Residuals.
        """
        mu = self.fitted(type="response", na_expand=False)

        if type == "response":
            resid = self.matrices.y - mu
        elif type == "pearson":
            var = self.family.variance(mu)
            resid = np.sqrt(self.matrices.weights) * (self.matrices.y - mu) / np.sqrt(var)
        elif type == "deviance":
            dev_resids = self.family.deviance_resids(self.matrices.y, mu, self.matrices.weights)
            signs = np.sign(self.matrices.y - mu)
            resid = signs * np.sqrt(np.abs(dev_resids))
        else:
            raise ValueError(f"Unknown residual type: {type}")

        if na_expand and self._should_expand_na():
            assert self.matrices.na_info is not None
            return self.matrices.na_info.expand_to_original(resid)
        return resid

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        type: str = "response",
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
        newdata : pandas or Polars DataFrame or LazyFrame, optional
            New data for prediction. If None, returns fitted values. Lazy queries
            are projected to prediction columns and collected once per call.
        type : str, default "response"
            Type of prediction: "response" (mean) or "link" (linear predictor).
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" for fixed effects only.
        allow_new_levels : bool, default False
            Allow new levels in grouping factors (predicts with RE=0).
        se_fit : bool, default False
            If True, return standard errors of predictions.
        interval : str, default "none"
            Type of interval: "none" or "confidence".
            Note: prediction intervals not available for GLMMs.
        level : float, default 0.95
            Confidence level for intervals.
        offset : array-like, scalar, or str, optional
            Offset for new-data predictions on the link scale. A string selects
            a column from ``newdata``. Scalars are broadcast to every row.

        Returns
        -------
        NDArray or PredictResult
            Predictions. Returns PredictResult if se_fit=True or interval!="none".

        Notes
        -----
        Standard errors use the final PIRLS working approximation with fitted
        covariance parameters held fixed. Conditional predictions include the
        joint fixed/random-effect covariance. Allowed new grouping levels add
        their prior random-effect variance on the link scale. Response-scale
        standard errors use the delta method, and confidence limits transform
        the link-scale interval.
        """
        level = _validate_confidence_level(level)
        if not isinstance(type, str) or type not in ("response", "link"):
            raise ValueError("type must be 'response' or 'link'")
        if not isinstance(interval, str) or interval not in ("none", "confidence", "prediction"):
            raise ValueError(f"Unknown interval type: {interval}. Use 'none' or 'confidence'.")
        if interval == "prediction":
            raise ValueError(
                "Prediction intervals not available for GLMMs. Use interval='confidence'."
            )
        include_re = re_form != "NA" and re_form != "~0"
        if newdata is not None:
            newdata = self._prepare_prediction_data(
                newdata,
                include_re=include_re,
                extra_columns=(offset,) if isinstance(offset, str) else (),
            )
        random_design: tuple[sparse.csr_matrix, NDArray[np.floating]] | None = None

        if newdata is None:
            if offset is not None:
                raise ValueError("Prediction offset can only be supplied with newdata.")
            if include_re:
                eta = self._linear_predictor.copy()
            else:
                eta = self.matrices.X @ self.beta + self.matrices.offset
            X = self.matrices.X
        else:
            prediction_offset = self._prediction_offset(newdata, offset)
            X = self._prediction_fixed_matrix(newdata)
            eta = X @ self.beta + prediction_offset

            if include_re:
                if se_fit or interval != "none":
                    random_design = self._prediction_random_matrix(newdata, allow_new_levels)
                    eta += random_design[0] @ self.u
                else:
                    eta += self._random_effect_prediction_contrib(newdata, allow_new_levels, self.u)

        if not se_fit and interval == "none":
            if type == "link":
                return eta
            else:
                return self.family.link.inverse(eta)

        var_eta = self._compute_prediction_variance(X, random_design, include_re=include_re)
        se_eta = np.sqrt(var_eta)

        lower = upper = None
        if interval == "confidence":
            from scipy import stats

            z_crit = stats.norm.isf((1 - level) / 2)
            lower = eta - z_crit * se_eta
            upper = eta + z_crit * se_eta
        if type == "response":
            mu = self.family.link.inverse(eta)
            deriv = self.family.link.deriv(mu)
            se_eta = se_eta / np.abs(deriv)
            if lower is not None and upper is not None:
                lower, upper = self.family.link.inverse_interval(lower, upper)
            eta = mu
        return PredictResult(
            fit=eta, se_fit=se_eta, lower=lower, upper=upper, interval=interval, level=level
        )

    def _compute_prediction_variance(
        self,
        X: NDArray[np.floating],
        random_design: tuple[sparse.csr_matrix, NDArray[np.floating]] | None,
        *,
        include_re: bool,
    ) -> NDArray[np.floating]:
        """Approximate link-scale mean variance from the joint working precision."""
        vcov_beta = self.vcov()
        if not include_re or self.matrices.n_random == 0:
            return np.maximum(dense_quadratic_form_diagonal(X, vcov_beta), 0.0)

        Z_pred: sparse.csr_matrix
        prior_var: NDArray[np.floating]
        if random_design is None:
            Z_pred = self.matrices.Z.tocsr()
            prior_var = np.zeros(X.shape[0], dtype=np.float64)
        else:
            Z_pred, prior_var = random_design

        projection = self._working_projection
        assert projection.random_factor is not None
        transformed_Z = (Z_pred @ projection.Lambda).tocsr()
        adjusted_X = X - np.asarray(transformed_Z @ projection.random_fixed_map)
        var_fixed = dense_quadratic_form_diagonal(adjusted_X, vcov_beta)
        var_random = projection.random_factor.quadratic_diagonal(transformed_Z)
        return np.maximum(var_fixed + var_random + prior_var, 0.0)

    def vcov(self) -> NDArray[np.floating]:
        if self.matrices.n_fixed == 0:
            return np.empty((0, 0), dtype=np.float64)
        return self._working_projection.information_inv.copy()

    @cached_property
    def _hat_values(self) -> NDArray[np.float64]:
        projection = self._working_projection
        if self.matrices.n_random == 0:
            diagonal = np.einsum(
                "ij,ij->i",
                projection.weighted_X @ projection.information_inv,
                projection.weighted_X,
            )
            return np.clip(diagonal, 0, 1 - 1e-10)

        weighted_z_lambda = projection.weighted_Z @ projection.Lambda
        assert projection.random_factor is not None
        adjusted_x = projection.weighted_X - weighted_z_lambda @ projection.random_fixed_map
        diagonal = np.einsum(
            "ij,ij->i",
            adjusted_x @ projection.information_inv,
            adjusted_x,
        )
        diagonal += projection.random_factor.quadratic_diagonal(weighted_z_lambda)
        return np.clip(diagonal, 0, 1 - 1e-10)

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'hat': Leverage values (hatvalues)
            - 'cooks_d': Cook's distance
            - 'pearson_resid': Pearson residuals
            - 'deviance_resid': Deviance residuals

        Notes
        -----
        Values cover the fitted observations, without NA expansion. For GLMMs,
        deviance residuals are often preferred over Pearson residuals for
        identifying outliers. ``mixedlm.diagnostics.influence`` returns the
        same quantities with DFBETAS and DFFITS.
        """
        from mixedlm.diagnostics.influence import influence

        diagnostics = influence(self)
        return {
            "hat": diagnostics.hat_values,
            "cooks_d": diagnostics.cooks_distance,
            "pearson_resid": diagnostics.residuals,
            "deviance_resid": self.residuals(type="deviance", na_expand=False),
        }

    def VarCorr(self) -> GlmerVarCorr:
        return GlmerVarCorr(groups=self._varcorr_groups(scale=1.0))

    def _getme_components(self) -> dict[str, Callable[[], Any]]:
        return {
            **super()._getme_components(),
            "family": lambda: self.family,
            "nAGQ": lambda: self.nAGQ,
        }

    def _compute_RZX(self) -> NDArray[np.floating]:
        """Compute RZX, the cross-term in the mixed model equations."""
        return self._working_projection.RZX.copy()

    def _compute_RX(self) -> NDArray[np.floating]:
        """Compute the upper Cholesky factor of final fixed-effect information."""
        return linalg.cholesky(self._working_projection.fixed_information, lower=False)

    def _devcomp_cmp(self) -> dict[str, float]:
        projection = self._working_projection
        y = self.matrices.y
        mu = self.fitted(na_expand=False)
        u = self._spherical_u()
        wrss = float(np.sum(self.residuals(type="pearson", na_expand=False) ** 2))
        ussq = float(np.dot(u, u))
        return {
            "ldL2": 0.0 if projection.random_factor is None else projection.random_factor.logdet,
            "ldRX2": float(np.linalg.slogdet(projection.fixed_information)[1]),
            "wrss": wrss,
            "ussq": ussq,
            "pwrss": wrss + ussq,
            "drsum": float(np.sum(self.family.deviance_resids(y, mu, self.matrices.weights))),
            "REML": np.nan,
            "dev": float(self.deviance),
            "sigmaML": np.nan,
            "sigmaREML": np.nan,
        }

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        family: Family | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
        nAGQ: int | None = None,
        **kwargs,
    ) -> GlmerResult:
        """Update and re-fit the model with modified arguments.

        This method allows updating the model formula, data, or other arguments
        and refitting.

        Parameters
        ----------
        formula : str, optional
            New formula. If None, uses the original formula.
            Use "." to refer to the original formula components.
        data : DataFrame, optional
            New data. If None, uses the original data (must be stored).
        family : Family, optional
            New GLM family. If None, uses the original family.
        weights : array-like, optional
            New prior weights. If None and the data length is unchanged, uses
            the original prior weights. Grouped binomial trial counts are
            multiplied once using the updated response denominator.
        offset : array-like, optional
            New offset. If None, uses the original offset.
        nAGQ : int, optional
            Number of quadrature points. If None, uses original.
        **kwargs
            Additional arguments passed to glmer().

        Returns
        -------
        GlmerResult
            New fitted model result.

        Raises
        ------
        ValueError
            If data is needed but not available.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
        >>> # Change family
        >>> result2 = result.update(family=Poisson())
        >>> # Add a term
        >>> result3 = result.update(". ~ . + z")
        """

        if data is None:
            if self.matrices.frame is not None:
                data = self.matrices.frame
            else:
                raise ValueError(
                    "No data available. Either provide data or ensure model_frame was stored."
                )

        new_formula = str(self.formula) if formula is None else self._update_formula(formula)

        if family is None:
            family = self.family

        from mixedlm.families.binomial import Binomial
        from mixedlm.formula.parser import parse_formula

        if (
            isinstance(family, Binomial)
            and parse_formula(new_formula).response == self.formula.response
        ):
            data = _restore_binomial_factor(
                data, self.formula.response, self.matrices.response_levels
            )

        data_size_changed = len(data) != self.matrices.n_obs

        if weights is None and not data_size_changed:
            weights = (
                self.matrices.weights
                if self.matrices.trials is None
                else self.matrices.weights / self.matrices.trials
            )
        if offset is None and not data_size_changed:
            offset = self.matrices.offset
        if nAGQ is None:
            nAGQ = self.nAGQ

        kwargs.setdefault("control", self._refit_control())

        return glmer(
            new_formula, data, family=family, weights=weights, offset=offset, nAGQ=nAGQ, **kwargs
        )

    @cached_property
    def _saturated_log_likelihood(self) -> float:
        return self.family.log_likelihood(
            self.matrices.y,
            self.matrices.y,
            self.matrices.weights,
            trials=self.matrices.trials,
        )

    def logLik(self) -> LogLik:
        """Report the marginal log likelihood including response-density constants.

        The stored ``deviance`` remains the fitting objective relative to the
        saturated conditional density. Custom families must implement
        ``Family.log_likelihood()``; quasi likelihoods have no normalized density.
        """
        value = -0.5 * self.deviance + self._saturated_log_likelihood
        return LogLik(value=value, df=self.npar(), nobs=self.matrices.n_obs, REML=False)

    def get_deviance(self) -> float:
        """Get minus twice the normalized marginal log likelihood.

        Uses the fitted Laplace or adaptive quadrature approximation, including
        the response-density constants. The ``deviance`` attribute and
        ``as_function("deviance")`` retain the optimizer's unnormalized criterion.

        Returns
        -------
        float
            The absolute marginal deviance, ``-2 * logLik().value``.

        See Also
        --------
        logLik : Get the log-likelihood.
        """
        return -2 * self.logLik().value

    def REMLcrit(self) -> float:
        """Get the ML deviance (GLMMs do not use REML).

        For generalized linear mixed models, REML estimation is not used.
        This method returns the ML deviance for API compatibility with
        LmerResult.

        Returns
        -------
        float
            The ML deviance.

        Notes
        -----
        Unlike linear mixed models, GLMMs are always fit using maximum
        likelihood (via Laplace approximation or adaptive Gauss-Hermite
        quadrature). This method exists for API consistency with LmerResult.

        See Also
        --------
        get_deviance : Get the deviance value.
        logLik : Get the log-likelihood.
        isREML : Check if the model was fit with REML (always False for GLMMs).
        """
        return self.get_deviance()

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
            - "predict": returns a prediction function (linear predictor)

        Returns
        -------
        callable
            The requested function. For a joint fit, deviance accepts either
            [theta, beta] or theta alone with fitted beta held fixed. For a
            fast PIRLS fit, it accepts theta and recomputes beta by PIRLS.
        """
        from mixedlm.estimation.laplace import GLMMOptimizer

        if type == "deviance":
            optimizer = GLMMOptimizer(
                self.matrices,
                self.family,
                verbose=0,
                nAGQ=self.nAGQ,
                pirls_maxiter=self.pirls_maxiter,
                pirls_tol=self.pirls_tol,
            )
            if not self.joint_fit:
                return optimizer.objective
            joint = optimizer.joint_objective()
            fitted_beta = self.beta.copy()

            def deviance(parameters: NDArray[np.floating]) -> float:
                parameters = np.asarray(parameters)
                if parameters.shape == self.theta.shape:
                    parameters = np.r_[parameters, fitted_beta]
                return joint(parameters)

            return deviance
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
        from mixedlm.inference.profile import profile_glmer

        if n_jobs != 1:
            raise ValueError("Parallel profiling is currently supported only for LmerResult")
        return profile_glmer(self, which=which, n_points=n_points, level=level)

    def _bootstrap(self, n_boot: int, seed: RandomSeed) -> BootstrapResult:
        from mixedlm.inference.bootstrap import bootstrap_glmer

        return bootstrap_glmer(self, n_boot=n_boot, seed=seed)

    def _simulate_from_eta(self, eta: NDArray[np.floating], rng: Any) -> NDArray[np.floating]:
        return self._simulate_response(self.family.link.inverse(eta), rng)

    def _simulate_response(
        self,
        mu: NDArray[np.floating],
        rng: Any | None = None,
    ) -> NDArray[np.floating]:
        return simulate_glmm_response(
            self.family,
            mu,
            self.matrices.weights,
            trials=self.matrices.trials,
            rng=rng,
        )

    def _refit_from_matrices(self, matrices: ModelMatrices, **kwargs) -> GlmerResult:
        optimizer = GLMMOptimizer(
            matrices,
            self.family,
            verbose=0,
            nAGQ=self.nAGQ,
            pirls_maxiter=kwargs.pop("pirls_maxiter", self.pirls_maxiter),
            pirls_tol=kwargs.pop("pirls_tol", self.pirls_tol),
            nAGQ0initStep=kwargs.pop("nAGQ0initStep", True),
        )

        start = kwargs.pop("start", self.theta)
        kwargs.setdefault("method", "L-BFGS-B")
        opt_result = optimizer.optimize(start=start, **kwargs)

        return GlmerResult(
            formula=self.formula,
            matrices=matrices,
            family=self.family,
            theta=opt_result.theta,
            beta=opt_result.beta,
            u=opt_result.u,
            deviance=opt_result.deviance,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
            nAGQ=self.nAGQ,
            pirls_converged=opt_result.pirls_converged,
            pirls_maxiter=optimizer.pirls_maxiter,
            pirls_tol=optimizer.pirls_tol,
            joint_fit=opt_result.joint_fit,
            message=opt_result.message,
            optimizer=kwargs["method"],
        )

    def refitML(self) -> GlmerResult:
        """Return self since GLMMs are always fit with ML.

        GLMMs do not use REML estimation, so this method simply returns
        the current result unchanged. It exists for API consistency with
        LmerResult.

        Returns
        -------
        GlmerResult
            Returns self (GLMMs are already fit with ML).
        """
        return self

    def drop1(self, data: pd.DataFrame, test: str = "Chisq", n_jobs: int = 1) -> Drop1Result:
        from mixedlm.inference.drop1 import drop1_glmer

        return drop1_glmer(self, data, test=test, n_jobs=n_jobs)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> AllFitResult:
        from mixedlm.inference.allfit import allfit_glmer

        return allfit_glmer(self, data, optimizers=optimizers, n_jobs=n_jobs, verbose=verbose)

    def summary(self) -> str:
        lines = []
        approximation = "adaptive Gauss-Hermite quadrature" if self.nAGQ > 1 else "Laplace"
        lines.append(f"Generalized linear mixed model fit by maximum likelihood ({approximation})")
        lines.append(
            f" Family: {self.family.__class__.__name__} ({self.family.link.__class__.__name__})"
        )
        lines.append(f"Formula: {self.formula}")
        lines.append("")

        lines.append("     AIC      BIC   logLik  -2logL")
        try:
            ll = self.logLik()
        except (NotImplementedError, ValueError) as exc:
            lines.append(f"{'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8}")
            lines.append(str(exc))
        else:
            absolute_deviance = -2 * ll.value
            lines.append(
                f"{absolute_deviance + 2 * ll.df:8.1f} "
                f"{absolute_deviance + ll.df * np.log(ll.nobs):8.1f} "
                f"{ll.value:8.1f} {absolute_deviance:8.1f}"
            )
        lines.append("")

        lines.append(str(self.VarCorr()))
        lines.append(f"Number of obs: {self.matrices.n_obs}")
        for struct in self.matrices.random_structures:
            lines.append(f"  groups:  {struct.grouping_factor}, {struct.n_levels}")
        lines.append("")

        lines.append("Fixed effects:")
        vcov = self.vcov()
        se = np.sqrt(np.diag(vcov))

        from scipy import stats

        lines.append("             Estimate  Std. Error  z value  Pr(>|z|)")
        for i, name in enumerate(self.matrices.fixed_names):
            z_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
            p_val = 2 * stats.norm.sf(np.abs(z_val))
            sig = _get_signif_code(p_val)
            lines.append(
                f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {z_val:7.3f}  "
                f"{_format_pvalue(p_val):>10} {sig}"
            )

        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")
            if not self.pirls_converged:
                lines.append(
                    "  inner PIRLS solver did not converge; "
                    "inspect the response and model specification"
                )
            else:
                lines.append("  optimizer did not converge; try allFit() to compare optimizers")
        if self.isSingular():
            lines.append("  boundary (singular) fit: some random-effect variances are near zero")
            lines.append("  consider simplifying the random-effects structure")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GlmerResult(formula={self.formula}, "
            f"family={self.family.__class__.__name__}, deviance={self.deviance:.4f})"
        )


from mixedlm.models.glmer_fit import GlmerMod as GlmerMod
from mixedlm.models.glmer_fit import glmer as glmer
from mixedlm.models.glmer_fit import glmer_nb as glmer_nb
