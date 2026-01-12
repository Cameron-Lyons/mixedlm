from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda, _count_theta
from mixedlm.formula.parser import parse_formula, update_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices


@dataclass
class RanefResult:
    values: dict[str, dict[str, NDArray[np.floating]]]
    condVar: dict[str, dict[str, NDArray[np.floating]]] | None = None

    def __getitem__(self, key: str) -> dict[str, NDArray[np.floating]]:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def keys(self):
        return self.values.keys()

    def items(self):
        return self.values.items()


@dataclass
class VarCorr:
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
class LmerResult:
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

    def fixef(self) -> dict[str, float]:
        return dict(zip(self.matrices.fixed_names, self.beta, strict=False))

    def ranef(
        self, condVar: bool = False
    ) -> dict[str, dict[str, NDArray[np.floating]]] | RanefResult:
        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            u_block = self.u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            term_ranefs: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_ranefs[term_name] = u_block[:, j]

            result[struct.grouping_factor] = term_ranefs

        if not condVar:
            return result

        cond_var = self._compute_condVar()
        return RanefResult(values=result, condVar=cond_var)

    def _compute_condVar(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        q = self.matrices.n_random
        if q == 0:
            return {}

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        Zt = self.matrices.Zt
        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V = LambdatZtZLambda + I_q

        V_dense = V.toarray()
        V_inv = linalg.inv(V_dense)

        Lambda_dense = Lambda.toarray() if sparse.issparse(Lambda) else Lambda
        cond_cov = self.sigma**2 * Lambda_dense @ V_inv @ Lambda_dense.T

        cond_var_result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            var_block = np.zeros((n_levels, n_terms), dtype=np.float64)
            for i in range(n_levels):
                for j in range(n_terms):
                    idx = u_idx + i * n_terms + j
                    var_block[i, j] = cond_cov[idx, idx]

            u_idx += n_u

            term_vars: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_vars[term_name] = var_block[:, j]

            cond_var_result[struct.grouping_factor] = term_vars

        return cond_var_result

    def coef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        ranefs = self.ranef()
        fixefs = self.fixef()
        result: dict[str, dict[str, NDArray[np.floating]]] = {}

        for group, terms in ranefs.items():
            group_coef: dict[str, NDArray[np.floating]] = {}
            for term_name, ranef_vals in terms.items():
                if term_name in fixefs:
                    group_coef[term_name] = ranef_vals + fixefs[term_name]
                else:
                    group_coef[term_name] = ranef_vals
            result[group] = group_coef

        return result

    def nobs(self) -> int:
        return self.matrices.n_obs

    def ngrps(self) -> dict[str, int]:
        return {
            struct.grouping_factor: struct.n_levels for struct in self.matrices.random_structures
        }

    def get_sigma(self) -> float:
        return self.sigma

    def df_residual(self) -> int:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
        return n - p

    def getME(
        self, name: str
    ) -> NDArray[np.floating] | sparse.csc_matrix | float | int | list[str] | dict[str, int]:
        components = {
            "X": self.matrices.X,
            "Z": self.matrices.Z,
            "Zt": self.matrices.Zt,
            "y": self.matrices.y,
            "beta": self.beta,
            "theta": self.theta,
            "u": self.u,
            "b": self.u,
            "sigma": self.sigma,
            "deviance": self.deviance,
            "REML": self.REML,
            "n_obs": self.matrices.n_obs,
            "n_fixed": self.matrices.n_fixed,
            "n_random": self.matrices.n_random,
            "fixef_names": self.matrices.fixed_names,
            "weights": self.matrices.weights,
            "offset": self.matrices.offset,
        }

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)
        components["Lambda"] = Lambda
        components["Lambdat"] = Lambda.T

        if name not in components:
            valid = ", ".join(sorted(components.keys()))
            raise ValueError(f"Unknown component '{name}'. Valid components: {valid}")

        return components[name]

    def fitted(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part + self.matrices.offset

    def residuals(self, type: str = "response") -> NDArray[np.floating]:
        if type == "response":
            return self.matrices.y - self.fitted()
        elif type == "pearson":
            return (self.matrices.y - self.fitted()) / self.sigma
        else:
            raise ValueError(f"Unknown residual type: {type}")

    def predict(
        self,
        newdata: pd.DataFrame | None = None,
        re_form: str | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        if newdata is None:
            return self.fitted()

        new_matrices = build_model_matrices(self.formula, newdata)
        fixed_part = new_matrices.X @ self.beta

        if offset is None:
            offset = np.zeros(len(newdata), dtype=np.float64)

        if re_form == "NA" or re_form == "~0":
            return fixed_part + offset

        return fixed_part + offset

    def vcov(self) -> NDArray[np.floating]:
        q = self.matrices.n_random

        if q == 0:
            XtX = self.matrices.X.T @ self.matrices.X
            return self.sigma**2 * linalg.inv(XtX)

        Lambda = _build_lambda(self.theta, self.matrices.random_structures)

        Zt = self.matrices.Zt
        ZtZ = Zt @ Zt.T
        LambdatZtZLambda = Lambda.T @ ZtZ @ Lambda

        I_q = sparse.eye(q, format="csc")
        V_factor = LambdatZtZLambda + I_q
        V_factor_dense = V_factor.toarray()
        L_V = linalg.cholesky(V_factor_dense, lower=True)

        ZtX = Zt @ self.matrices.X
        Lambdat_ZtX = Lambda.T @ ZtX
        RZX = linalg.solve_triangular(L_V, Lambdat_ZtX, lower=True)

        XtX = self.matrices.X.T @ self.matrices.X
        RZX_tRZX = RZX.T @ RZX
        XtVinvX = XtX - RZX_tRZX

        return self.sigma**2 * linalg.inv(XtVinvX)

    def VarCorr(self) -> VarCorr:
        groups: dict[str, dict[str, float]] = {}
        theta_idx = 0

        for struct in self.matrices.random_structures:
            q = struct.n_terms

            if struct.correlated:
                n_theta = q * (q + 1) // 2
                theta_block = self.theta[theta_idx : theta_idx + n_theta]
                theta_idx += n_theta

                L_block = np.zeros((q, q), dtype=np.float64)
                idx = 0
                for i in range(q):
                    for j in range(i + 1):
                        L_block[i, j] = theta_block[idx]
                        idx += 1

                cov_scaled = L_block @ L_block.T
                cov = cov_scaled * self.sigma**2
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q
                cov = np.diag(theta_block**2) * self.sigma**2

            term_vars: dict[str, float] = {}
            for i, term_name in enumerate(struct.term_names):
                term_vars[term_name] = cov[i, i]

            groups[struct.grouping_factor] = term_vars

        return VarCorr(groups=groups, residual=self.sigma**2)

    def isSingular(self, tol: float = 1e-4) -> bool:
        theta_idx = 0

        for struct in self.matrices.random_structures:
            q = struct.n_terms

            if struct.correlated:
                n_theta = q * (q + 1) // 2
                theta_block = self.theta[theta_idx : theta_idx + n_theta]
                theta_idx += n_theta

                diag_idx = 0
                for i in range(q):
                    if abs(theta_block[diag_idx]) < tol:
                        return True
                    diag_idx += i + 2
            else:
                theta_block = self.theta[theta_idx : theta_idx + q]
                theta_idx += q

                if np.any(np.abs(theta_block) < tol):
                    return True

        return False

    def logLik(self) -> float:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed

        if self.REML:
            return -0.5 * (self.deviance + (n - p) * np.log(2 * np.pi))
        else:
            return -0.5 * (self.deviance + n * np.log(2 * np.pi))

    def AIC(self) -> float:
        n_theta = _count_theta(self.matrices.random_structures)
        n_params = self.matrices.n_fixed + n_theta + 1
        return -2 * self.logLik() + 2 * n_params

    def BIC(self) -> float:
        n_theta = _count_theta(self.matrices.random_structures)
        n_params = self.matrices.n_fixed + n_theta + 1
        n = self.matrices.n_obs
        return -2 * self.logLik() + n_params * np.log(n)

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
        if seed is not None:
            np.random.seed(seed)

        n = self.matrices.n_obs

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
        n = self.matrices.n_obs
        q = self.matrices.n_random

        fixed_part = self.matrices.X @ self.beta

        if re_form == "~0" or re_form == "NA" or not use_re or q == 0:
            random_part = np.zeros(n)
        else:
            u_new = np.zeros(q, dtype=np.float64)
            u_idx = 0

            for struct in self.matrices.random_structures:
                n_levels = struct.n_levels
                n_terms = struct.n_terms

                theta_start = sum(
                    s.n_terms * (s.n_terms + 1) // 2 if s.correlated else s.n_terms
                    for s in self.matrices.random_structures[
                        : self.matrices.random_structures.index(struct)
                    ]
                )
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

            random_part = self.matrices.Z @ u_new

        noise = np.random.randn(n) * self.sigma

        return fixed_part + random_part + noise

    def refit(self, newresp: NDArray[np.floating]) -> LmerResult:
        if len(newresp) != self.matrices.n_obs:
            raise ValueError(f"newresp has length {len(newresp)}, expected {self.matrices.n_obs}")

        new_matrices = ModelMatrices(
            y=newresp,
            X=self.matrices.X,
            Z=self.matrices.Z,
            fixed_names=self.matrices.fixed_names,
            random_structures=self.matrices.random_structures,
            n_obs=self.matrices.n_obs,
            n_fixed=self.matrices.n_fixed,
            n_random=self.matrices.n_random,
            weights=self.matrices.weights,
            offset=self.matrices.offset,
        )

        optimizer = LMMOptimizer(new_matrices, REML=self.REML, verbose=0)
        opt_result = optimizer.optimize(start=self.theta)

        return LmerResult(
            formula=self.formula,
            matrices=new_matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=self.REML,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
        )

    def update(
        self,
        formula: str | None = None,
        data: pd.DataFrame | None = None,
        REML: bool | None = None,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> LmerResult:
        if data is None:
            raise ValueError(
                "data must be provided to update(). "
                "The original data is not stored in the result object."
            )

        if formula is not None:
            new_formula = update_formula(self.formula, formula)
        else:
            new_formula = self.formula

        new_REML = REML if REML is not None else self.REML

        model = LmerMod(
            new_formula,
            data,
            REML=new_REML,
            weights=weights,
            offset=offset,
        )
        return model.fit()

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

    def summary(self) -> str:
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

        if self.REML:
            self.matrices.n_obs - self.matrices.n_fixed
        else:
            self.matrices.n_obs - self.matrices.n_fixed

        lines.append("             Estimate  Std. Error  t value")
        for i, name in enumerate(self.matrices.fixed_names):
            t_val = self.beta[i] / se[i] if se[i] > 0 else np.nan
            lines.append(f"{name:12} {self.beta[i]:10.4f}  {se[i]:10.4f}  {t_val:7.3f}")

        lines.append("")
        if self.converged:
            lines.append(f"convergence: yes ({self.n_iter} iterations)")
        else:
            lines.append(f"convergence: no ({self.n_iter} iterations)")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"LmerResult(formula={self.formula}, deviance={self.deviance:.4f})"


class LmerMod:
    def __init__(
        self,
        formula: str | Formula,
        data: pd.DataFrame,
        REML: bool = True,
        verbose: int = 0,
        weights: NDArray[np.floating] | None = None,
        offset: NDArray[np.floating] | None = None,
    ) -> None:
        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.REML = REML
        self.verbose = verbose

        self.matrices = build_model_matrices(
            self.formula, self.data, weights=weights, offset=offset
        )

    def fit(
        self,
        start: NDArray[np.floating] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
    ) -> LmerResult:
        optimizer = LMMOptimizer(
            self.matrices,
            REML=self.REML,
            verbose=self.verbose,
        )

        opt_result = optimizer.optimize(
            start=start,
            method=method,
            maxiter=maxiter,
        )

        return LmerResult(
            formula=self.formula,
            matrices=self.matrices,
            theta=opt_result.theta,
            beta=opt_result.beta,
            sigma=opt_result.sigma,
            u=opt_result.u,
            deviance=opt_result.deviance,
            REML=self.REML,
            converged=opt_result.converged,
            n_iter=opt_result.n_iter,
        )


def lmer(
    formula: str,
    data: pd.DataFrame,
    REML: bool = True,
    verbose: int = 0,
    weights: NDArray[np.floating] | None = None,
    offset: NDArray[np.floating] | None = None,
    **kwargs,
) -> LmerResult:
    model = LmerMod(formula, data, REML=REML, verbose=verbose, weights=weights, offset=offset)
    return model.fit(**kwargs)
