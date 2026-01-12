from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse, stats

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.reml import LMMOptimizer, _build_lambda, _count_theta
from mixedlm.formula.parser import parse_formula
from mixedlm.formula.terms import Formula
from mixedlm.matrices.design import ModelMatrices, build_model_matrices


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
        lines.append(f" {'Residual':11} {' ':12} {self.residual:9.4f}  {np.sqrt(self.residual):.4f}")
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
        return dict(zip(self.matrices.fixed_names, self.beta))

    def ranef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        result: dict[str, dict[str, NDArray[np.floating]]] = {}
        u_idx = 0

        for struct in self.matrices.random_structures:
            n_levels = struct.n_levels
            n_terms = struct.n_terms
            n_u = n_levels * n_terms

            u_block = self.u[u_idx : u_idx + n_u].reshape(n_levels, n_terms)
            u_idx += n_u

            levels = sorted(struct.level_map.keys(), key=lambda x: struct.level_map[x])

            term_ranefs: dict[str, NDArray[np.floating]] = {}
            for j, term_name in enumerate(struct.term_names):
                term_ranefs[term_name] = u_block[:, j]

            result[struct.grouping_factor] = term_ranefs

        return result

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

    def fitted(self) -> NDArray[np.floating]:
        fixed_part = self.matrices.X @ self.beta
        random_part = self.matrices.Z @ self.u
        return fixed_part + random_part

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
    ) -> NDArray[np.floating]:
        if newdata is None:
            return self.fitted()

        new_matrices = build_model_matrices(self.formula, newdata)
        fixed_part = new_matrices.X @ self.beta

        if re_form == "NA" or re_form == "~0":
            return fixed_part

        return fixed_part

    def vcov(self) -> NDArray[np.floating]:
        n = self.matrices.n_obs
        p = self.matrices.n_fixed
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
            df_resid = self.matrices.n_obs - self.matrices.n_fixed
        else:
            df_resid = self.matrices.n_obs - self.matrices.n_fixed

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
    ) -> None:
        if isinstance(formula, str):
            self.formula = parse_formula(formula)
        else:
            self.formula = formula

        self.data = data
        self.REML = REML
        self.verbose = verbose

        self.matrices = build_model_matrices(self.formula, self.data)

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
    **kwargs,
) -> LmerResult:
    model = LmerMod(formula, data, REML=REML, verbose=verbose)
    return model.fit(**kwargs)
