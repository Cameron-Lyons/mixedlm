from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.nlmm import NLMMOptimizer, _build_psi_matrix
from mixedlm.nlme.models import NonlinearModel

_COV_REGULARIZATION = 1e-8
_VCOV_EPS = 1e-5
_JACOBIAN_EPS = 1e-6
_HAT_FALLBACK_EPS = 1e-10
_HAT_CLIP_MAX = 1 - 1e-10
_DEFAULT_MAXITER = 500
_DEFAULT_N_BOOT = 1000
_DEFAULT_SINGULAR_TOL = 1e-4


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
class NlmerResult:
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

    def fixef(self) -> dict[str, float]:
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

    def fitted(self) -> NDArray[np.floating]:
        n = len(self.y)
        n_groups = len(np.unique(self.groups))
        pred = np.zeros(n, dtype=np.float64)

        for g in range(n_groups):
            mask = self.groups == g
            x_g = self.x[mask]

            params_g = self.phi.copy()
            for j, p_idx in enumerate(self.random_params):
                params_g[p_idx] += self.b[g, j]

            pred[mask] = self.model.predict(params_g, x_g)

        return pred

    def residuals(self, type: str = "response") -> NDArray[np.floating]:
        fitted = self.fitted()
        if type == "response":
            return self.y - fitted
        elif type == "pearson":
            return (self.y - fitted) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        x_var: str = "x",
        group_var: str | None = None,
    ) -> NDArray[np.floating]:
        if newdata is None:
            return self.fitted()

        x_new = newdata[x_var].to_numpy(dtype=np.float64)
        n_new = len(x_new)

        if group_var is None or group_var not in newdata.columns:
            pred = self.model.predict(self.phi, x_new)
        else:
            groups_new = newdata[group_var].astype(str).tolist()
            pred = np.zeros(n_new, dtype=np.float64)

            for i, g in enumerate(groups_new):
                if g in self.group_levels:
                    g_idx = self.group_levels.index(g)
                    params = self.phi.copy()
                    for j, p_idx in enumerate(self.random_params):
                        params[p_idx] += self.b[g_idx, j]
                    pred[i] = self.model.predict(params, np.array([x_new[i]]))[0]
                else:
                    pred[i] = self.model.predict(self.phi, np.array([x_new[i]]))[0]

        return pred

    def VarCorr(self) -> NlmerVarCorr:
        n_random = len(self.random_params)
        Psi = _build_psi_matrix(self.theta, n_random)

        random_param_names = [self.model.param_names[i] for i in self.random_params]

        term_vars: dict[str, float] = {}
        for i, name in enumerate(random_param_names):
            term_vars[name] = Psi[i, i] * self.sigma**2

        groups = {self.group_var: term_vars}

        return NlmerVarCorr(groups=groups, residual=self.sigma**2)

    def logLik(self) -> float:
        return -0.5 * self.deviance

    def AIC(self) -> float:
        n_params = len(self.phi) + len(self.theta) + 1
        return -2 * self.logLik() + 2 * n_params

    def BIC(self) -> float:
        n_params = len(self.phi) + len(self.theta) + 1
        n = len(self.y)
        return -2 * self.logLik() + n_params * np.log(n)

    def extractAIC(self) -> tuple[float, float]:
        """Extract AIC with effective degrees of freedom.

        Returns the effective degrees of freedom and AIC value,
        matching the interface of R's extractAIC function.

        Returns
        -------
        tuple of (float, float)
            (edf, AIC) where edf is the effective degrees of freedom.
        """
        n_params = len(self.phi) + len(self.theta) + 1
        edf = float(n_params)
        aic = float(-2 * self.logLik() + 2 * n_params)
        return (edf, aic)

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

    def isGLMM(self) -> bool:
        """Check if this is a generalized linear mixed model.

        Always returns False for NlmerResult.
        """
        return False

    def isLMM(self) -> bool:
        """Check if this is a linear mixed model.

        Always returns False for NlmerResult.
        """
        return False

    def isNLMM(self) -> bool:
        """Check if this is a nonlinear mixed model.

        Always returns True for NlmerResult.
        """
        return True

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

        Returns the prior weights used in model fitting.
        If no weights were specified, returns an array of ones.
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
        seed: int | None = None,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        """Simulate responses from the fitted model.

        Parameters
        ----------
        nsim : int, default 1
            Number of simulations.
        seed : int, optional
            Random seed for reproducibility.
        use_re : bool, default True
            If True, simulate new random effects. If False, use fixed effects only.
        re_form : str, optional
            Formula for random effects. Use "NA" or "~0" to exclude random effects.

        Returns
        -------
        NDArray
            Simulated responses. Shape (n,) if nsim=1, else (n, nsim).
        """
        if seed is not None:
            np.random.seed(seed)

        n = len(self.y)

        if nsim == 1:
            return self._simulate_once(use_re, re_form)

        result = np.zeros((n, nsim), dtype=np.float64)
        for i in range(nsim):
            result[:, i] = self._simulate_once(use_re, re_form)

        return result

    def _simulate_once(
        self,
        use_re: bool = True,
        re_form: str | None = None,
    ) -> NDArray[np.floating]:
        """Simulate a single response vector."""
        n = len(self.y)
        n_groups = len(self.group_levels)
        n_random = len(self.random_params)

        include_re = use_re and re_form not in ("~0", "NA")

        if include_re and n_random > 0:
            Psi = _build_psi_matrix(self.theta, n_random)
            cov = Psi * self.sigma**2
            cov = cov + _COV_REGULARIZATION * np.eye(n_random)
            b_new = np.random.multivariate_normal(np.zeros(n_random), cov, size=n_groups)
        else:
            b_new = np.zeros((n_groups, n_random))

        pred = np.zeros(n, dtype=np.float64)
        for g in range(n_groups):
            mask = self.groups == g
            x_g = self.x[mask]

            params_g = self.phi.copy()
            for j, p_idx in enumerate(self.random_params):
                params_g[p_idx] += b_new[g, j]

            pred[mask] = self.model.predict(params_g, x_g)

        noise = np.random.randn(n) * self.sigma
        return pred + noise

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
            Additional arguments passed to the optimizer.

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

        optimizer = NLMMOptimizer(
            newresp,
            self.x,
            self.groups,
            self.model,
            self.random_params,
            verbose=0,
            weights=self._weights,
        )

        start_phi = kwargs.pop("start", self.phi)
        method = kwargs.pop("method", "L-BFGS-B")
        maxiter = kwargs.pop("maxiter", _DEFAULT_MAXITER)

        opt_result = optimizer.optimize(
            start_phi=start_phi,
            method=method,
            maxiter=maxiter,
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
            x=self.x,
            y=newresp,
            groups=self.groups,
            group_levels=self.group_levels,
            _weights=self._weights,
            _offset=self._offset,
            _data=self._data,
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

        def neg_log_lik(phi_vec):
            pred = np.zeros(len(self.y), dtype=np.float64)
            n_groups = len(self.group_levels)
            for g in range(n_groups):
                mask = self.groups == g
                x_g = self.x[mask]
                params_g = phi_vec.copy()
                for j, p_idx in enumerate(self.random_params):
                    params_g[p_idx] += self.b[g, j]
                pred[mask] = self.model.predict(params_g, x_g)

            resid = self.y - pred
            return 0.5 * np.sum(resid**2) / self.sigma**2

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

        try:
            L = linalg.cholesky(hessian, lower=True)
            vcov = linalg.cho_solve((L, True), np.eye(n_params))
        except linalg.LinAlgError:
            try:
                vcov = linalg.solve(hessian, np.eye(n_params))
            except linalg.LinAlgError:
                vcov = linalg.pinv(hessian)

        return vcov

    def confint(
        self,
        parm: str | list[str] | None = None,
        level: float = 0.95,
        method: str = "boot",
        n_boot: int = _DEFAULT_N_BOOT,
        seed: int | None = None,
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
            Number of bootstrap samples (if method="boot").
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            Dictionary mapping parameter names to (lower, upper) tuples.
        """
        from scipy import stats

        if parm is None:
            parm = self.model.param_names
        elif isinstance(parm, str):
            parm = [parm]

        if method == "Wald":
            vcov = self.vcov()
            alpha = 1 - level
            z_crit = stats.norm.ppf(1 - alpha / 2)

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
            if seed is not None:
                np.random.seed(seed)

            n_params = len(self.phi)
            boot_samples = np.zeros((n_boot, n_params), dtype=np.float64)

            for i in range(n_boot):
                y_sim = self.simulate(nsim=1, use_re=True)
                try:
                    fit_i = self.refit(y_sim)
                    boot_samples[i, :] = fit_i.phi
                except Exception:
                    boot_samples[i, :] = self.phi

            alpha = 1 - level
            lower_pct = alpha / 2 * 100
            upper_pct = (1 - alpha / 2) * 100

            result = {}
            for p in parm:
                if p not in self.model.param_names:
                    continue
                idx = self.model.param_names.index(p)
                lower = float(np.percentile(boot_samples[:, idx], lower_pct))
                upper = float(np.percentile(boot_samples[:, idx], upper_pct))
                result[p] = (lower, upper)
            return result

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

        fitted_base = self.fitted()

        for j in range(n_params):
            phi_plus = self.phi.copy()
            phi_plus[j] += eps

            pred_plus = np.zeros(n, dtype=np.float64)
            n_groups = len(self.group_levels)
            for g in range(n_groups):
                mask = self.groups == g
                x_g = self.x[mask]
                params_g = phi_plus.copy()
                for k, p_idx in enumerate(self.random_params):
                    params_g[p_idx] += self.b[g, k]
                pred_plus[mask] = self.model.predict(params_g, x_g)

            J[:, j] = (pred_plus - fitted_base) / eps

        try:
            JtJ = J.T @ J
            try:
                L = linalg.cholesky(JtJ, lower=True)
                JtJ_inv = linalg.cho_solve((L, True), np.eye(n_params))
            except linalg.LinAlgError:
                JtJ_inv = linalg.solve(JtJ, np.eye(n_params))

            J_JtJ_inv = J @ JtJ_inv
            h = np.sum(J_JtJ_inv * J, axis=1)
        except linalg.LinAlgError:
            h = np.sum(J**2, axis=1) / (np.sum(J**2) + _HAT_FALLBACK_EPS)

        return np.clip(h, 0, _HAT_CLIP_MAX)

    def cooks_distance(self) -> NDArray[np.floating]:
        """Compute Cook's distance for each observation.

        Returns
        -------
        NDArray
            Cook's distance for each observation.
        """
        h = self.hatvalues()
        resid = self.residuals(type="response")
        p = len(self.phi)

        h = np.clip(h, 0, _HAT_CLIP_MAX)
        cooks_d = (resid**2 / (p * self.sigma**2)) * (h / (1 - h) ** 2)

        return cooks_d

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
        resid = self.residuals(type="response")
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
            Tolerance for detecting near-zero variance components.

        Returns
        -------
        bool
            True if any variance component is near zero.
        """
        return bool(np.any(np.abs(self.theta) < tol))

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
        for name, val in self.fixef().items():
            lines.append(f"{name:12} {val:10.4f}")

        lines.append("")
        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"NlmerResult(model={self.model.name}, deviance={self.deviance:.4f})"


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
            self.weights = np.asarray(weights, dtype=np.float64)
            if len(self.weights) != len(self.y):
                raise ValueError(f"weights has length {len(self.weights)}, expected {len(self.y)}")
        else:
            self.weights = None

        if offset is not None:
            self.offset = np.asarray(offset, dtype=np.float64)
            if len(self.offset) != len(self.y):
                raise ValueError(f"offset has length {len(self.offset)}, expected {len(self.y)}")
            self.y = self.y - self.offset
        else:
            self.offset = None

        group_col = data[group_var].astype(str)
        self.group_levels = sorted(group_col.unique().tolist())
        level_map = {lv: i for i, lv in enumerate(self.group_levels)}
        self.groups = np.array([level_map[g] for g in group_col], dtype=np.int64)

        if random_params is None:
            self.random_params = list(range(model.n_params))
        elif isinstance(random_params[0], str):
            self.random_params = [model.param_names.index(p) for p in random_params]  # type: ignore
        else:
            self.random_params = list(random_params)  # type: ignore

        self.start_phi: NDArray[np.floating]
        if start is not None:
            self.start_phi = np.array(
                [start.get(name, 1.0) for name in model.param_names],
                dtype=np.float64,
            )
        else:
            self.start_phi = model.get_start(self.x, self.y)

    def fit(
        self,
        method: str = "L-BFGS-B",
        maxiter: int = _DEFAULT_MAXITER,
    ) -> NlmerResult:
        optimizer = NLMMOptimizer(
            self.y,
            self.x,
            self.groups,
            self.model,
            self.random_params,
            verbose=self.verbose,
            weights=self.weights,
        )

        opt_result = optimizer.optimize(
            start_phi=self.start_phi,
            method=method,
            maxiter=maxiter,
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
        Prior weights for observations.
    offset : array-like, optional
        Offset term subtracted from the response.
    **kwargs
        Additional arguments passed to the optimizer (method, maxiter).

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
