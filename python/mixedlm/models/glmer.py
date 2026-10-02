from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.laplace import GLMMOptimizer, _build_lambda, _count_theta
from mixedlm.families.base import Family
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, _restore_binomial_factor
from mixedlm.models.control import GlmerControl
from mixedlm.models.lmer_types import (
    LogLik,
    PredictResult,
    RanefResult,
    RePCA,
    VarCorrGroup,
)
from mixedlm.models.lmer_types import RePCAGroup as RePCAGroup
from mixedlm.models.result_mixin import MerResultMixin
from mixedlm.models.shared_utils import (
    _RandomEffectFactor,
    dense_quadratic_form_diagonal,
    symmetric_inverse,
)
from mixedlm.utils import _format_pvalue, _get_signif_code
from mixedlm.utils.random import RandomSeed, native_seed, random_stream, validate_simulation_count
from mixedlm.utils.simulation import (
    simulate_glmm_response,
    simulate_random_effects,
    simulation_parameters,
)
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

    def _refit_control(self) -> GlmerControl:
        """Carry the fitted inner settings into formula-based refitting paths."""
        return GlmerControl(tolPwrss=self.pirls_tol, pirls_maxiter=self.pirls_maxiter)

    def fixef(self) -> dict[str, float]:
        return self._fixef_dict(self.beta)

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        return self._ranef_with_optional_condvar(self.u, condVar)

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

    def get_sigma(self) -> float:
        return 1.0

    @property
    def sigma(self) -> float:
        return 1.0

    def weights(self) -> NDArray[np.floating]:
        """Get the model weights.

        Returns the prior weights used in model fitting.
        If no weights were specified, returns an array of ones.

        Returns
        -------
        NDArray
            Array of weights with length equal to number of observations.
        """
        return self._weights_array(copy=True)

    def offset(self) -> NDArray[np.floating]:
        """Get the model offset.

        Returns the offset used in model fitting.
        If no offset was specified, returns an array of zeros.

        Returns
        -------
        NDArray
            Array of offsets with length equal to number of observations.
        """
        return self._offset_array(copy=True)

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
        """
        return self._model_matrix(type)

    def terms(self):
        """Get information about the model terms.

        Returns a ModelTerms object containing information about the
        response variable, fixed effect terms, random effect terms,
        and grouping factors.

        Returns
        -------
        ModelTerms
            Object containing term information.
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
        >>> result = glmer("y ~ x + (1 | group)", data, family=Binomial())
        >>> mf = result.model_frame()
        >>> print(mf.columns.tolist())  # ['y', 'x', 'group']
        """
        return self._model_frame()

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
                eta = self._add_random_effects_to_eta(eta, newdata, allow_new_levels)

        if not se_fit and interval == "none":
            if type == "link":
                return eta
            else:
                return self.family.link.inverse(eta)

        vcov_beta = self.vcov()
        var_eta = dense_quadratic_form_diagonal(X, vcov_beta)
        se_eta = np.sqrt(np.maximum(var_eta, 0.0))

        lower = upper = None
        if interval == "confidence":
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

    def _add_random_effects_to_eta(
        self,
        eta: NDArray[np.floating],
        newdata: pd.DataFrame,
        allow_new_levels: bool,
    ) -> NDArray[np.floating]:
        """Add random effects contribution to linear predictor."""
        eta += self._random_effect_prediction_contrib(newdata, allow_new_levels, self.u)
        return eta

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

    def hatvalues(self) -> NDArray[np.floating]:
        """Compute leverage values (diagonal of the hat matrix).

        For GLMMs, the hat matrix is computed using the working weights
        from the iteratively reweighted least squares algorithm.

        Returns
        -------
        NDArray
            Leverage values for each observation, between 0 and 1.
            Values close to 1 indicate high-leverage observations.

        Notes
        -----
        For generalized linear mixed models, the hat matrix incorporates
        both fixed and random effects, weighted by the variance function.
        """
        return self._hat_values.copy()

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        For GLMMs, Cook's distance measures the influence of each observation
        on the fitted values, using Pearson residuals.

        Returns
        -------
        NDArray
            Cook's distance for each observation.

        Notes
        -----
        For GLMMs, Cook's distance is computed using Pearson residuals
        and the working weights from the IRLS algorithm.
        Models without fixed effects return NaN because this normalization
        divides by the number of fixed-effect parameters.
        """
        p = self.matrices.n_fixed
        if p == 0:
            return np.full(self.matrices.n_obs, np.nan)
        h = self.hatvalues()
        resid = self.residuals(type="pearson")

        h = np.clip(h, 0, 1 - 1e-10)

        cooks_d = (resid**2 / p) * (h / (1 - h) ** 2)

        return cooks_d

    def influence(self) -> dict[str, NDArray[np.floating]]:
        """Compute influence diagnostics for the model.

        Returns a dictionary containing various influence measures for
        identifying influential observations in GLMMs.

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
        Influential observations are those that have a large effect on
        the model estimates. For GLMMs, deviance residuals are often
        preferred over Pearson residuals for identifying outliers.
        """
        return {
            "hat": self.hatvalues(),
            "cooks_d": self.cooks_distance(),
            "pearson_resid": self.residuals(type="pearson"),
            "deviance_resid": self.residuals(type="deviance"),
        }

    def VarCorr(self) -> GlmerVarCorr:
        return GlmerVarCorr(groups=self._varcorr_groups(scale=1.0))

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
        """
        return self._random_effect_pca(scale=1.0)

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
        >>> result = glmer("y ~ x + (x | group)", data, family=Binomial())
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
        for glmer objects. This is useful for assessing model assumptions and
        detecting patterns in the residuals.

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
        >>> result = glmer("y ~ x + (1 | group)", data, family=Binomial())
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
        R's getME() function in lme4.

        Parameters
        ----------
        name : str
            Name of the component to extract. Valid names are:
            - "X" : Fixed effects design matrix (n x p)
            - "Z" : Random effects design matrix (n x q)
            - "Zt" : Transpose of Z (q x n)
            - "y" : Response vector
            - "beta" : Fixed effects coefficients
            - "theta" : Variance component parameters
            - "Lambda" : Relative covariance factor (sparse, q x q)
            - "Lambdat" : Transpose of Lambda
            - "u" : Spherical random effects
            - "b" : Conditional modes of random effects
            - "n" or "n_obs" : Number of observations
            - "p" or "n_fixed" : Number of fixed effects
            - "q" or "n_random" : Number of random effects
            - "lower" : Lower bounds for theta
            - "weights" : Prior weights
            - "offset" : Offset term
            - "deviance" : Deviance
            - "flist" : List of grouping factors
            - "cnms" : Component names for random effects
            - "Gp" : Group pointers
            - "family" : GLM family

        Returns
        -------
        The requested component.

        Raises
        ------
        ValueError
            If an unknown component name is requested.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=Binomial())
        >>> X = result.getME("X")
        >>> family = result.getME("family")
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
        elif name == "deviance":
            return self.deviance
        elif name == "flist":
            return [s.grouping_factor for s in self.matrices.random_structures]
        elif name == "cnms":
            return {s.grouping_factor: s.term_names for s in self.matrices.random_structures}
        elif name == "Gp":
            gp = [0]
            for s in self.matrices.random_structures:
                gp.append(gp[-1] + s.n_levels * s.n_terms)
            return np.array(gp)
        elif name == "family":
            return self.family
        elif name == "nAGQ":
            return self.nAGQ
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
                "n",
                "n_obs",
                "p",
                "n_fixed",
                "q",
                "n_random",
                "lower",
                "weights",
                "offset",
                "deviance",
                "flist",
                "cnms",
                "Gp",
                "family",
                "nAGQ",
                "RX",
                "RZX",
                "Lind",
                "devcomp",
            ]
            raise ValueError(f"Unknown component name: '{name}'. Valid names are: {valid_names}")

    def _compute_RZX(self) -> NDArray[np.floating]:
        """Compute RZX, the cross-term in the mixed model equations."""
        return self._working_projection.RZX.copy()

    def _compute_RX(self) -> NDArray[np.floating]:
        """Compute the upper Cholesky factor of final fixed-effect information."""
        return linalg.cholesky(self._working_projection.fixed_information, lower=False)

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
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        q = self.matrices.n_random
        n_theta = len(self.theta)

        cmp = {
            "ldL2": 0.0,
            "ldRX2": 0.0,
            "pwrss": 0.0,
            "drsum": 0.0,
            "dev": float(self.deviance),
            "ussq": float(np.sum(self.u**2)) if self.u is not None else 0.0,
        }

        dims = {
            "n": n,
            "p": p,
            "q": q,
            "nmp": n - p,
            "nth": n_theta,
            "REML": 0,
            "useSc": 0,
            "nAGQ": self.nAGQ,
            "q0": q,
            "q1": 0,
            "qrx": p,
            "ngrps": len(self.matrices.random_structures),
        }

        return {"cmp": cmp, "dims": dims}

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
        n = self.matrices.n_obs
        n_theta = _count_theta(self.matrices.random_structures)
        df = self.matrices.n_fixed + n_theta
        value = -0.5 * self.deviance + self._saturated_log_likelihood

        return LogLik(value=value, df=df, nobs=n, REML=False)

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

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "Wald",
        n_boot: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        from scipy import stats

        from mixedlm.inference.bootstrap import bootstrap_glmer
        from mixedlm.inference.profile import profile_glmer

        level = _validate_confidence_level(level)
        if parm is None:
            parm = self.matrices.fixed_names
        elif isinstance(parm, str):
            parm = [parm]

        from mixedlm.utils.names import _check_unique_coefficient_names

        _check_unique_coefficient_names(
            self.matrices.fixed_names,
            None if method == "boot" else parm,
            alternative="Use tidy(conf_int=True) for intervals in coefficient order.",
        )

        if method == "Wald":
            vcov = self.vcov()
            alpha = 1 - level
            z_crit = stats.norm.isf(alpha / 2)

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
            # Endpoints use root finding; confidence intervals need no interior plot grid.
            profiles = profile_glmer(self, which=parm, level=level, n_points=3)
            return {p: (profiles[p].ci_lower, profiles[p].ci_upper) for p in parm if p in profiles}

        elif method == "boot":
            boot_result = bootstrap_glmer(self, n_boot=n_boot, seed=seed)
            return boot_result.ci(level=level)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'Wald', 'profile', or 'boot'.")

    def simulate(
        self,
        nsim: int = 1,
        seed: RandomSeed = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        """Simulate responses using an isolated or caller-provided random stream.

        ``seed`` accepts an integer, ``RandomState``, ``Generator``, or ``None``.
        Integer seeds preserve the existing draw sequence for the selected backend.
        Reusing a stream continues it across calls without changing NumPy's global state.
        """
        validate_simulation_count(nsim)
        rng = random_stream(seed)

        n = self.matrices.n_obs
        q = self.matrices.n_random

        if nsim == 1:
            return self._simulate_once(use_re, re_form, rng)

        include_re = use_re and q > 0 and re_form not in ("~0", "NA")

        if not include_re:
            fixed_eta = self.matrices.X @ self.beta + self.matrices.offset
            eta = np.broadcast_to(fixed_eta[:, None], (n, nsim))
            return self._simulate_response(self.family.link.inverse(eta), rng)

        try:
            from mixedlm._rust import simulate_re_batch

            return self._simulate_batch_rust(nsim, native_seed(seed, rng), simulate_re_batch, rng)
        except ImportError:
            pass

        result = np.zeros((n, nsim), dtype=np.float64)
        for i in range(nsim):
            result[:, i] = self._simulate_once(use_re, re_form, rng)

        return result

    def _simulate_batch_rust(
        self,
        nsim: int,
        seed: int | None,
        simulate_re_batch: Any,
        rng: Any | None = None,
    ) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        fixed_eta = self.matrices.X @ self.beta + self.matrices.offset
        structures = self.matrices.random_structures
        theta, correlated = simulation_parameters(self.theta, structures)
        u_batch = simulate_re_batch(
            theta,
            1.0,
            [structure.n_levels for structure in structures],
            [structure.n_terms for structure in structures],
            correlated,
            nsim,
            seed,
        )
        eta = np.asarray(self.matrices.Z @ u_batch.T, dtype=np.float64)
        eta += fixed_eta[:, None]
        return self._simulate_response(self.family.link.inverse(eta), rng)

    def _simulate_once(
        self,
        use_re: bool = True,
        re_form: str | None = None,
        rng: Any | None = None,
    ) -> NDArray[np.floating]:
        rng = np.random if rng is None else rng
        q = self.matrices.n_random

        eta = self.matrices.X @ self.beta + self.matrices.offset

        if re_form != "~0" and re_form != "NA" and use_re and q > 0:
            u_new = simulate_random_effects(self.theta, self.matrices.random_structures, rng=rng)
            eta += self.matrices.Z @ u_new

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

    def refit(
        self,
        newresp: ArrayLike | None = None,
        **kwargs,
    ) -> GlmerResult:
        """Refit the model with a new response vector.

        This method refits the model using the same formula and design matrices
        but with a different response vector. This is useful for simulation
        studies, bootstrap, and permutation tests.

        Parameters
        ----------
        newresp : array-like, optional
            New response values. Must have the same length as the original
            response. For grouped binomial models, provide success counts;
            the original trial counts are reused. If None, refits with the
            original response. Two-level factor models accept the fitted labels
            or encoded numeric 0/1 responses. Numeric responses keep their
            encoded meaning regardless of the fitted factor order.
        **kwargs
            Additional arguments passed to the optimizer (start, method, maxiter).
            Inner controls pirls_maxiter and pirls_tol default to the original
            fit's settings and can be overridden independently.

        Returns
        -------
        GlmerResult
            New fitted model result with the updated response.

        Examples
        --------
        >>> result = glmer("y ~ x + (1|group)", data, family=families.Binomial())
        >>> # Refit with simulated response
        >>> y_sim = result.simulate()
        >>> result_sim = result.refit(newresp=y_sim)

        See Also
        --------
        simulate : Simulate response from the fitted model.
        """
        y_new = self._coerce_new_response(newresp)
        matrices = self._clone_matrices_with_response_base(y_new)
        return self._refit_from_matrices(matrices, **kwargs)

    def npar(self) -> int:
        """Get the number of parameters in the model.

        Returns the total number of estimated parameters:
        - Fixed effects (beta)
        - Variance-covariance parameters (theta)

        Returns
        -------
        int
            Total number of parameters.
        """
        return self._npar_count(include_sigma=False)

    def drop1(self, data: pd.DataFrame, test: str = "Chisq"):
        from mixedlm.inference.drop1 import drop1_glmer

        return drop1_glmer(self, data, test=test)

    def allFit(
        self,
        data: pd.DataFrame,
        optimizers: list[str] | None = None,
        verbose: bool = False,
    ):
        from mixedlm.inference.allfit import allfit_glmer

        return allfit_glmer(self, data, optimizers=optimizers, verbose=verbose)

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

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"GlmerResult(formula={self.formula}, "
            f"family={self.family.__class__.__name__}, deviance={self.deviance:.4f})"
        )


from mixedlm.models.glmer_fit import GlmerMod as GlmerMod
from mixedlm.models.glmer_fit import glmer as glmer
from mixedlm.models.glmer_fit import glmer_nb as glmer_nb
