"""Likelihood profiles of random-effect standard deviations, correlations and sigma.

The parameters follow lme4 (version 2) names, order and scale. Each covariance
block lists its standard deviations in term order and then its correlations in
column order of the lower triangle; blocks are numbered as lme4 orders them.
They are named ``.sig01``, ``.sig02``, ... or, as with lme4's
``signames = FALSE``, ``sd_Days|Subject`` and ``cor_Days.(Intercept)|Subject``.
``.sigma`` (also ``sigma``) is a linear model's residual standard deviation.

Each profile fixes one parameter and re-optimizes the others and the fixed
effects: on the ML deviance for linear models, as lme4 does for REML fits too,
and on the integrated likelihood for generalized models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from mixedlm.estimation.reml import _build_lambda_blocks, _count_theta
from mixedlm.inference.profile_types import ProfileResult
from mixedlm.utils.validation import _validate_confidence_level

if TYPE_CHECKING:
    from mixedlm.matrices.design import RandomEffectStructure
    from mixedlm.models.glmer import GlmerResult
    from mixedlm.models.lmer import LmerResult


@dataclass(frozen=True)
class VarianceParameter:
    """A standard deviation, correlation or residual scale, named as in lme4."""

    name: str
    kind: str
    group: str
    terms: tuple[str, ...]
    lower: float
    upper: float

    @property
    def label(self) -> str:
        """lme4's descriptive name, as printed by ``confint(signames = FALSE)``."""
        if self.kind == "sigma":
            return "sigma"
        if self.kind == "sd":
            return f"sd_{self.terms[0]}|{self.group}"
        return f"cor_{self.terms[0]}.{self.terms[1]}|{self.group}"

    @property
    def description(self) -> str:
        if self.kind == "sigma":
            return "Residual standard deviation"
        if self.kind == "sd":
            return f"SD of {self.terms[0]} | {self.group}"
        return f"Correlation of {self.terms[0]} and {self.terms[1]} | {self.group}"


def _block_positions(struct: RandomEffectStructure) -> list[tuple[int, int]]:
    """Positions of one block's SDs and then its correlations, in lme4 order."""
    q = struct.n_terms
    if struct.cov_type in ("cs", "ar1"):
        return [(0, 0), (1, 0)] if q > 1 else [(0, 0)]
    sds = [(i, i) for i in range(q)]
    if not struct.correlated:
        return sds
    return sds + [(i, j) for j in range(q) for i in range(j + 1, q)]


def block_order(structures: list[RandomEffectStructure]) -> list[int]:
    """Order covariance blocks as lme4 numbers them.

    lme4's ``mkReTrms(reorder.terms = TRUE)`` keeps blocks whose level counts
    already decrease and otherwise uses ``rev(order(levels))``: decreasing
    levels, with ties in reverse formula order.
    """
    levels = [struct.n_levels for struct in structures]
    if all(later <= earlier for earlier, later in zip(levels, levels[1:], strict=False)):
        return list(range(len(structures)))
    return sorted(range(len(structures)), key=lambda index: (levels[index], index))[::-1]


def _parameter_blocks(
    structures: list[RandomEffectStructure],
) -> list[tuple[int, list[tuple[int, int]]]]:
    """lme4's covariance blocks in ``.sig`` order, as structure indices and positions.

    lme4 expands an uncorrelated ``(1 + x || g)`` term into one block per term
    before ordering blocks, so each term's SDs are numbered as their own block.
    """
    blocks: list[tuple[int, list[tuple[int, int]]]] = []
    for index, struct in enumerate(structures):
        positions = _block_positions(struct)
        if struct.cov_type == "us" and not struct.correlated:
            # Split by formula term; a multi-column term such as poly(x, 2) stays
            # together in column order. Fits saved without term sizes split by column.
            sizes = getattr(struct, "term_sizes", None) or (1,) * struct.n_terms
            bounds = np.cumsum([0, *sizes])
            blocks.extend(
                (index, positions[start:stop])
                for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
            )
        else:
            blocks.append((index, positions))
    order = block_order([structures[index] for index, _ in blocks])
    return [blocks[position] for position in order]


def variance_parameters(
    structures: list[RandomEffectStructure], *, has_sigma: bool = True
) -> list[VarianceParameter]:
    """Describe each SD, correlation and the residual SD, in lme4's ``.sig`` order."""
    parameters: list[VarianceParameter] = []
    for block, block_positions in _parameter_blocks(structures):
        struct = structures[block]
        q = struct.n_terms
        for i, j in block_positions:
            terms: tuple[str, ...]
            if i == j:
                kind, terms, lower, upper = "sd", (struct.term_names[i],), 0.0, np.inf
            else:
                kind, terms = "cor", (struct.term_names[i], struct.term_names[j])
                # Compound symmetry stays positive definite only above -1/(q-1).
                lower = -1.0 / (q - 1) if struct.cov_type == "cs" else -1.0
                upper = 1.0
            name = f".sig{len(parameters) + 1:02d}"
            parameters.append(
                VarianceParameter(name, kind, struct.grouping_factor, terms, lower, upper)
            )
    if has_sigma:
        parameters.append(VarianceParameter(".sigma", "sigma", "Residual", (), 0.0, np.inf))
    return parameters


def variance_parameter_positions(
    structures: list[RandomEffectStructure],
) -> dict[tuple[int, int, int], str]:
    """Map (block, row, column) positions of SDs and correlations to lme4 names.

    ``block`` indexes ``structures``. Compound-symmetry blocks share one SD and
    one correlation across their terms; AR(1) blocks share one SD, and their
    correlation parameter is that of adjacent terms.
    """
    positions: dict[tuple[int, int, int], str] = {}
    names = iter(p.name for p in variance_parameters(structures, has_sigma=False))
    for block, block_positions in _parameter_blocks(structures):
        struct = structures[block]
        q = struct.n_terms
        for i, j in block_positions:
            name = next(names)
            if struct.cov_type in ("cs", "ar1") and i == j:
                positions.update({(block, term, term): name for term in range(q)})
            elif struct.cov_type == "cs":
                positions.update({(block, r, c): name for r in range(q) for c in range(r)})
            elif struct.cov_type == "ar1":
                positions.update({(block, r, r - 1): name for r in range(1, q)})
            else:
                positions[(block, i, j)] = name
    return positions


def _theta_blocks(
    theta: NDArray[np.floating], structures: list[RandomEffectStructure]
) -> list[NDArray[np.floating]]:
    bounds = np.cumsum([0, *(_count_theta([struct]) for struct in structures)])
    return [theta[start:stop] for start, stop in zip(bounds[:-1], bounds[1:], strict=True)]


def _block_sdcor(
    struct: RandomEffectStructure, theta: NDArray[np.floating], sigma: float
) -> dict[tuple[int, int], float]:
    """One structure's SDs and correlations by (row, column) position."""
    positions = _block_positions(struct)
    if struct.cov_type in ("cs", "ar1"):
        values = [sigma * abs(float(theta[0])), *map(float, theta[1:])]
        return dict(zip(positions, values, strict=True))
    (factor,) = _build_lambda_blocks(np.asarray(theta), [struct])
    covariance = sigma**2 * factor @ factor.T
    sd = np.sqrt(np.diag(covariance))
    result = {}
    for i, j in positions:
        scale = sd[i] * sd[j]
        if i == j:
            result[(i, j)] = float(sd[i])
        else:
            result[(i, j)] = float(covariance[i, j] / scale) if scale > 0 else 0.0
    return result


def sdcor_from_theta(
    theta: NDArray[np.floating], sigma: float, structures: list[RandomEffectStructure]
) -> NDArray[np.float64]:
    """Map relative covariance factors to SDs and correlations in ``.sig`` order."""
    thetas = _theta_blocks(np.asarray(theta, dtype=np.float64), structures)
    by_structure = [
        _block_sdcor(struct, block, sigma) for struct, block in zip(structures, thetas, strict=True)
    ]
    values = [
        by_structure[block][position]
        for block, positions in _parameter_blocks(structures)
        for position in positions
    ]
    return np.asarray(values, dtype=np.float64)


# Scaled factor entries below this are treated as sitting on a zero bound.
_NEAR_ZERO = 1e-3
# Starting value, in scaled units, for entries moved off a zero bound.
_ESCAPE = 0.05
_ESCAPE_ATTEMPTS = 2


@dataclass(frozen=True)
class _Constraint:
    """How a profiled value restricts the nuisance variables.

    ``fixed`` holds variable values. ``row`` lists the factor entries of one term
    that ``row_map`` rebuilds from ``row_size`` free parameters, so that a term's
    SD or one of its correlations takes the profiled value. ``sigma`` gives a
    linear model's residual scale as a function of the variables; by default it
    is profiled out.
    """

    fixed: dict[int, float]
    row: tuple[int, ...] = ()
    row_size: int = 0
    row_map: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None = None
    row_start: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    sigma: Callable[[NDArray[np.float64]], float] | None = None


def _unit(vector: NDArray[np.float64], fallback: int) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-300:
        return vector / norm
    unit = np.zeros_like(vector)
    unit[fallback] = 1.0
    return unit


class _VarianceProfile:
    """Profile one SD, correlation or sigma over valid relative covariance factors.

    Nuisance variables are the relative covariance factors theta, followed for
    generalized models by the fixed effects; every point is a valid covariance.
    The likelihood depends on each factor only through Lambda Lambda', so a
    term's factor row can be rebuilt freely: with a given norm for a profiled
    SD, or at a given angle to another row for a profiled correlation. A linear
    model's SD instead determines the residual scale.
    """

    has_sigma: bool
    structures: list[RandomEffectStructure]
    parameters: list[VarianceParameter]
    x_hat: NDArray[np.float64]
    x_scale: NDArray[np.float64]
    x_bounds: list[tuple[float | None, float | None]]
    n_theta: int
    sd_step: float

    def deviance(self, x: NDArray[np.floating], sigma: float | None = None) -> float:
        raise NotImplementedError

    def estimates(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def _setup_slots(self) -> None:
        offsets = np.cumsum([0, *(_count_theta([struct]) for struct in self.structures)])
        self.slots: list[tuple[int, int, int] | None] = []
        for block, positions in _parameter_blocks(self.structures):
            self.slots.extend((block, i, j) for i, j in positions)
        if self.has_sigma:
            self.slots.append(None)
        self.offsets = offsets

    def _relative(self, x: NDArray[np.floating], index: int) -> float:
        """The parameter's value at residual scale one: an SD or a correlation.

        Reads only the factor rows involved, as each objective evaluation needs it.
        """
        slot = self.slots[index]
        assert slot is not None
        block, i, j = slot
        struct = self.structures[block]
        if struct.cov_type in ("cs", "ar1"):
            offset = int(self.offsets[block])
            return abs(float(x[offset])) if i == j else float(x[offset + 1])
        row_i = x[self._row(block, i)]
        if i == j:
            return float(np.linalg.norm(row_i))
        row_j = x[self._row(block, j)]
        scale = float(np.linalg.norm(row_i) * np.linalg.norm(row_j))
        return float(row_i[: len(row_j)] @ row_j) / scale if scale > 0 else 0.0

    def _scale_for(self, x: NDArray[np.floating], index: int, value: float) -> float:
        """The residual scale at which the SD of parameter ``index`` equals ``value``."""
        relative = self._relative(x, index)
        return value / relative if relative > 0 else np.inf

    def _row(self, block: int, term: int) -> list[int]:
        """Theta entries whose norm is the term's SD in units of the residual scale."""
        struct, offset = self.structures[block], int(self.offsets[block])
        if struct.cov_type in ("cs", "ar1"):
            return [offset]
        if not struct.correlated:
            return [offset + term]
        first = offset + term * (term + 1) // 2
        return list(range(first, first + term + 1))

    def _constraint(self, index: int, value: float) -> _Constraint:
        slot = self.slots[index]
        if slot is None:
            return _Constraint({}, sigma=lambda x: value)
        block, i, j = slot
        struct = self.structures[block]
        if i == j:
            row = self._row(block, i)
            if value == 0:
                return _Constraint(dict.fromkeys(row, 0.0))
            if self.has_sigma:
                return _Constraint({}, sigma=lambda x: self._scale_for(x, index, value))
            if len(row) == 1:
                return _Constraint({row[0]: value})
            # Rescale the term's factor row to the profiled SD.
            return _Constraint(
                {},
                row=tuple(row),
                row_size=len(row),
                row_map=lambda x, z: value * _unit(z, len(z) - 1),
                row_start=lambda x: x[row].copy(),
            )
        if struct.cov_type in ("cs", "ar1"):
            return _Constraint({int(self.offsets[block]) + 1: value})
        return self._correlation_constraint(block, i, j, value)

    def _correlation_constraint(self, block: int, i: int, j: int, value: float) -> _Constraint:
        """Rebuild row i at the angle to row j whose cosine is the correlation."""
        row_i, row_j = self._row(block, i), self._row(block, j)
        sine = float(np.sqrt(max(0.0, 1.0 - value**2)))

        def direction(x: NDArray[np.float64]) -> NDArray[np.float64]:
            padded = np.zeros(len(row_i))
            padded[: len(row_j)] = x[row_j]
            return _unit(padded, 0)

        def row_map(x: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
            radius, z = abs(float(params[0])), params[1:]
            u = direction(x)
            # The orthogonal part of z; i > j, so the last coordinate is free of row j.
            v = _unit(z - (z @ u) * u, len(z) - 1)
            return radius * (value * u + sine * v)

        def row_start(x: NDArray[np.float64]) -> NDArray[np.float64]:
            current = x[row_i]
            return np.r_[max(float(np.linalg.norm(current)), 1e-3), current]

        return _Constraint(
            {}, row=tuple(row_i), row_size=len(row_i) + 1, row_map=row_map, row_start=row_start
        )

    def start(self, index: int) -> None:
        self.index = index
        self.mle = float(self.estimates()[index])
        self.minimum = self.deviance(self.x_hat)
        if not np.isfinite(self.minimum):
            raise RuntimeError("Variance-component profiling requires a finite deviance")
        self.cache: dict[float, tuple[float, NDArray[np.float64]]] = {
            self.mle: (self.minimum, self.x_hat)
        }

    def profile_deviance(self, value: float) -> float:
        if value not in self.cache:
            nearest = min(self.cache, key=lambda point: abs(point - value))
            fitted = self._fit(self._constraint(self.index, value), self.cache[nearest][1])
            tolerance = max(1e-6, 1e-9 * abs(self.minimum))
            if fitted[0] < self.minimum - tolerance:
                raise RuntimeError(
                    "Variance-component profile found a lower deviance than the optimum; "
                    "the likelihood optimization is unreliable for this model"
                )
            self.cache[value] = fitted
        return self.cache[value][0]

    def _fit(
        self, constraint: _Constraint, start: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        template = start.copy()
        for position, fixed in constraint.fixed.items():
            template[position] = fixed
        excluded = set(constraint.fixed) | set(constraint.row)
        free = np.array([i for i in range(len(start)) if i not in excluded], dtype=np.intp)
        n_free = len(free)
        row = np.asarray(constraint.row, dtype=np.intp)
        penalty = self.minimum + 1e6

        def unpack(point: NDArray[np.floating]) -> NDArray[np.float64]:
            x = template.copy()
            x[free] = point[:n_free] * self.x_scale[free]
            if constraint.row_map is not None:
                x[row] = constraint.row_map(x, np.asarray(point[n_free:], dtype=np.float64))
            return x

        def objective(point: NDArray[np.floating]) -> float:
            x = unpack(point)
            sigma = None if constraint.sigma is None else constraint.sigma(x)
            deviance = self.deviance(x, sigma)
            return deviance if np.isfinite(deviance) else penalty

        bounds: list[tuple[float | None, float | None]] = [
            (
                None if self.x_bounds[i][0] is None else self.x_bounds[i][0] / self.x_scale[i],
                None if self.x_bounds[i][1] is None else self.x_bounds[i][1] / self.x_scale[i],
            )
            for i in free
        ]
        point = np.clip(
            start[free] / self.x_scale[free],
            [-np.inf if lo is None else lo for lo, _ in bounds],
            [np.inf if hi is None else hi for _, hi in bounds],
        )
        # The deviance is even in some factor entries, so its gradient vanishes
        # at a zero variance and a solver started there, or one that lands on
        # the bound, need not move. Start off such bounds, and after each solve
        # restart from any entry left on a zero bound, keeping the better fit.
        zero_bound = np.array([self.x_bounds[i][0] == 0 for i in free], dtype=bool)
        point[zero_bound & (point < _NEAR_ZERO)] = _ESCAPE
        if constraint.row_start is not None:
            point = np.r_[point, constraint.row_start(start)]
            bounds += [(None, None)] * constraint.row_size
            zero_bound = np.r_[zero_bound, np.zeros(constraint.row_size, dtype=bool)]
        if not len(point):
            return objective(point), template

        def solve(initial: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            best: tuple[float, NDArray[np.float64]] | None = None
            for method, options in (
                ("L-BFGS-B", {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7}),
                ("COBYQA", {"maxiter": 5000, "final_tr_radius": 1e-9}),
            ):
                fitted = optimize.minimize(
                    objective, initial, method=method, bounds=bounds, options=options
                )
                if best is None or fitted.fun < best[0]:
                    best = float(fitted.fun), np.asarray(fitted.x, dtype=np.float64)
                if fitted.success and fitted.fun < penalty:
                    break
                initial = fitted.x
            assert best is not None
            return best

        value, solution = solve(point)
        for _ in range(_ESCAPE_ATTEMPTS):
            stuck = zero_bound & (solution < _NEAR_ZERO)
            if not stuck.any():
                break
            restart = solution.copy()
            restart[stuck] = _ESCAPE * 6
            candidate = solve(restart)
            if candidate[0] >= value - 1e-10:
                break
            value, solution = candidate
        if value >= penalty:
            raise RuntimeError("Variance-component profile optimization failed")
        return value, unpack(solution)

    def endpoint(self, direction: int, cutoff: float) -> float:
        parameter = self.parameters[self.index]
        bound = parameter.upper if direction > 0 else parameter.lower

        def difference(value: float) -> float:
            return self.profile_deviance(value) - self.minimum - cutoff

        def root(a: float, b: float) -> float:
            a, b = sorted((a, b))
            return float(optimize.brentq(difference, a, b, xtol=1e-8 * max(1.0, abs(b))))

        previous = self.mle
        if parameter.kind == "sigma":
            # Multiplicative steps never reach the open lower bound of zero.
            ratio = 1.1
            for _ in range(60):
                outer = self.mle * ratio**direction
                if difference(outer) >= 0:
                    return root(previous, outer)
                previous, ratio = outer, ratio**2
        else:
            step = (
                0.25 * (1.0 - abs(self.mle)) + 0.01
                if parameter.kind == "cor"
                else max(0.25 * abs(self.mle), 0.05 * self.sd_step)
            )
            for _ in range(60):
                outer = self.mle + direction * step
                at_bound = outer >= bound if direction > 0 else outer <= bound
                if at_bound:
                    outer = bound
                if outer != previous and difference(outer) >= 0:
                    return root(previous, outer)
                if at_bound:
                    # The likelihood-ratio limit lies beyond the parameter space.
                    return float(bound)
                previous, step = outer, step * 2
        raise RuntimeError(f"Could not bracket the profile interval for {parameter.name}")


class _LMMVarianceProfile(_VarianceProfile):
    """Variables: theta. The ML deviance at an explicit residual scale sigma is

    D = D_hat + n log(sigma^2 / sigma_hat^2) + pwrss / sigma^2 - n,

    from the profiled deviance D_hat and residual scale sigma_hat at theta.
    """

    def __init__(self, result: LmerResult) -> None:
        from mixedlm.estimation.reml import _build_theta_bounds
        from mixedlm.inference.lmm_profile import _LMMProfileLikelihood

        likelihood = _LMMProfileLikelihood(result.matrices)
        optimum = likelihood.fit(result.theta)
        self.has_sigma = True
        self.structures = result.matrices.random_structures
        self.parameters = variance_parameters(self.structures)
        self.n = result.matrices.n_obs
        self.n_theta = len(result.theta)
        self.optimizer = likelihood._optimizer((), np.empty(0))
        self.x_hat = np.asarray(optimum.theta, dtype=np.float64)
        self.sigma_hat = float(optimum.evaluation.sigma)
        self.sd_step = self.sigma_hat
        self.x_bounds = _build_theta_bounds(self.structures, self.n_theta)
        self.x_scale = np.maximum(np.abs(self.x_hat), 0.1)
        self._setup_slots()

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("Variance profiles hold native state; create one in each worker")

    def estimates(self) -> NDArray[np.float64]:
        sdcor = sdcor_from_theta(self.x_hat, self.sigma_hat, self.structures)
        return np.r_[sdcor, self.sigma_hat]

    def deviance(self, x: NDArray[np.floating], sigma: float | None = None) -> float:
        evaluation = self.optimizer._evaluate_core(np.ascontiguousarray(x, dtype=np.float64))
        if evaluation is None or not np.isfinite(evaluation.deviance):
            return np.inf
        if sigma is None:
            return float(evaluation.deviance)
        if not np.isfinite(sigma) or sigma <= 0:
            return np.inf
        # Logarithms and divisions in this order avoid overflow for extreme scales.
        return float(
            evaluation.deviance
            + 2.0 * self.n * (np.log(sigma) - np.log(float(evaluation.sigma)))
            + (evaluation.pwrss / sigma) / sigma
            - self.n
        )


class _GLMMVarianceProfile(_VarianceProfile):
    """Variables: theta, then the fixed effects, on the integrated likelihood."""

    def __init__(self, result: GlmerResult) -> None:
        from mixedlm.inference.glmm_profile import _GLMMProfileLikelihood

        if not result.converged or not result.pirls_converged:
            raise ValueError("GLMM likelihood profiling requires a converged fitted model")
        standard_errors = np.sqrt(np.diag(result.vcov()))
        likelihood = _GLMMProfileLikelihood(result, standard_errors)
        _, optimum = likelihood.fit(likelihood.start)
        self.has_sigma = False
        self.objective = likelihood.objective
        self.structures = result.matrices.random_structures
        self.parameters = variance_parameters(self.structures, has_sigma=False)
        self.n_theta = likelihood.n_theta
        self.x_hat = np.asarray(optimum * likelihood.scale, dtype=np.float64)
        self.sd_step = 1.0
        self.x_bounds = list(self.objective.bounds)
        self.x_scale = np.asarray(likelihood.scale, dtype=np.float64)
        self._setup_slots()

    def estimates(self) -> NDArray[np.float64]:
        return sdcor_from_theta(self.x_hat[: self.n_theta], 1.0, self.structures)

    def deviance(self, x: NDArray[np.floating], sigma: float | None = None) -> float:
        deviance, _, _, converged = self.objective.evaluate(np.asarray(x, dtype=np.float64))
        return float(deviance) if converged and np.isfinite(deviance) else np.inf


def _profile(
    profile: _VarianceProfile, index: int, name: str, n_points: int, level: float
) -> ProfileResult:
    from mixedlm.inference.lmm_profile import _grid

    profile.start(index)
    cutoff = float(stats.chi2.isf(1 - level, 1))
    lower = profile.endpoint(-1, cutoff)
    upper = profile.endpoint(1, cutoff)
    values = _grid(lower, upper, profile.mle, n_points)
    zeta = np.empty(n_points)
    for i in np.argsort(np.abs(values - profile.mle)):
        value = float(values[i])
        zeta[i] = np.sign(value - profile.mle) * np.sqrt(
            max(0.0, profile.profile_deviance(value) - profile.minimum)
        )
    return ProfileResult(name, values, zeta, profile.mle, lower, upper, level)


def _lmm_profile_task(task: tuple[Any, ...]) -> tuple[str, ProfileResult]:
    result, index, name, n_points, level = task
    return name, _profile(_LMMVarianceProfile(result), index, name, n_points, level)


def variance_profiles(
    result: LmerResult | GlmerResult,
    names: list[str],
    n_points: int,
    level: float,
    n_jobs: int = 1,
) -> dict[str, ProfileResult]:
    """Profile the named standard deviations, correlations and residual scale."""
    from mixedlm.inference.lmm_profile import _run_tasks, _validate_options

    level = _validate_confidence_level(level)
    jobs = _validate_options(n_points, n_jobs)
    if not names:
        return {}
    if not result.converged:
        raise ValueError("Likelihood profiling requires a converged fitted model")
    parameters = variance_parameters(
        result.matrices.random_structures, has_sigma=bool(result._HAS_SIGMA)
    )
    index = {parameter.name: i for i, parameter in enumerate(parameters)}
    if not result._IS_LMM:
        profile = _GLMMVarianceProfile(result)  # type: ignore[arg-type]
        return {name: _profile(profile, index[name], name, n_points, level) for name in names}
    # Each task prepares its own native design, so tasks can run in workers.
    tasks = [(result, index[name], name, n_points, level) for name in names]
    return dict(_run_tasks(_lmm_profile_task, tasks, jobs))
