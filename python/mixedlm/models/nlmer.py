from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

from mixedlm.estimation.nlmm import NLMMOptimizer, _build_psi_matrix
from mixedlm.nlme.models import NonlinearModel


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

    def fixef(self) -> dict[str, float]:
        return dict(zip(self.model.param_names, self.phi, strict=False))

    def ranef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        self.b.shape[0]
        random_param_names = [self.model.param_names[i] for i in self.random_params]

        term_ranefs: dict[str, NDArray[np.floating]] = {}
        for j, name in enumerate(random_param_names):
            term_ranefs[name] = self.b[:, j]

        return {self.group_var: term_ranefs}

    def coef(self) -> dict[str, dict[str, NDArray[np.floating]]]:
        n_groups = self.b.shape[0]
        [self.model.param_names[i] for i in self.random_params]

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
        len(self.y)
        return -0.5 * self.deviance

    def AIC(self) -> float:
        n_params = len(self.phi) + len(self.theta) + 1
        return -2 * self.logLik() + 2 * n_params

    def BIC(self) -> float:
        n_params = len(self.phi) + len(self.theta) + 1
        n = len(self.y)
        return -2 * self.logLik() + n_params * np.log(n)

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
    ) -> None:
        self.model = model
        self.data = data
        self.x_var = x_var
        self.y_var = y_var
        self.group_var = group_var
        self.verbose = verbose

        self.x = data[x_var].to_numpy(dtype=np.float64)
        self.y = data[y_var].to_numpy(dtype=np.float64)

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
        maxiter: int = 500,
    ) -> NlmerResult:
        optimizer = NLMMOptimizer(
            self.y,
            self.x,
            self.groups,
            self.model,
            self.random_params,
            verbose=self.verbose,
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
    **kwargs,
) -> NlmerResult:
    mod = NlmerMod(
        model=model,
        data=data,
        x_var=x_var,
        y_var=y_var,
        group_var=group_var,
        random_params=random_params,
        start=start,
        verbose=verbose,
    )
    return mod.fit(**kwargs)
