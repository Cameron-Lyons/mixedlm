"""In-process executors, unfitted results and stub refits for bootstrap tests."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
from mixedlm.families import Binomial, Gamma, Gaussian, InverseGaussian, NegativeBinomial, Poisson
from mixedlm.formula.parser import parse_formula
from mixedlm.inference import bootstrap
from mixedlm.matrices.design import build_model_matrices
from mixedlm.models.glmer import GlmerResult
from mixedlm.models.lmer import LmerResult


class ImmediateExecutor:
    """Run submitted tasks in the caller and record pool usage."""

    instances = []

    def __init__(self, max_workers, initializer=None, initargs=()):
        self.workers = max_workers
        self.submitted = self.consumed = self.peak_pending = 0
        self.closed = False
        self.tasks = []
        self.instances.append(self)
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def submit(self, fn, *args):
        executor = self

        class ConsumedFuture(Future):
            def result(self, *args, **kwargs):
                executor.consumed += 1
                return super().result(*args, **kwargs)

        self.submitted += 1
        self.peak_pending = max(self.peak_pending, self.submitted - self.consumed)
        self.tasks.append(args)
        future = ConsumedFuture()
        future.set_result(fn(*args))
        return future


class PendingExecutor(ImmediateExecutor):
    """Complete only the first submitted task, optionally with an error."""

    failure = False

    def submit(self, fn, *args):
        self.submitted += 1
        future = Future()
        self.tasks.append(future)
        if self.submitted == 1:
            if self.failure:
                future.set_exception(RuntimeError("pool task failed"))
            else:
                future.set_result(fn(*args))
        return future


def make_result(kind="lmm", n_groups=4, n_per_group=10):
    """An unfitted LMM or GLMM result with fixed parameters for simulation tests."""
    x = np.tile(np.linspace(-0.5, 0.5, n_per_group), n_groups)
    data = pd.DataFrame(
        {"y": np.ones(len(x)), "x": x, "group": np.repeat(np.arange(n_groups), n_per_group)}
    )
    formula = parse_formula("y ~ x + (1 | group)")
    matrices = build_model_matrices(formula, data)
    matrices = replace(
        matrices, weights=np.linspace(0.5, 2.0, len(x)), offset=np.linspace(0.1, 0.3, len(x))
    )
    if kind == "grouped_binomial":
        matrices = replace(matrices, trials=np.full(len(x), 9.0))
    common = dict(
        formula=formula,
        matrices=matrices,
        beta=np.array([0.2, 0.3]),
        theta=np.array([0.3]),
        u=np.zeros(n_groups),
        deviance=0.0,
        converged=True,
        n_iter=0,
    )
    if kind == "lmm":
        return LmerResult(**common, sigma=0.7, REML=False)
    families = dict(
        binomial=Binomial,
        grouped_binomial=Binomial,
        poisson=Poisson,
        gaussian=Gaussian,
        gamma=Gamma,
        inverse_gaussian=InverseGaussian,
        negative_binomial=NegativeBinomial,
    )
    return GlmerResult(**common, family=families[kind](), nAGQ=1)


def fake_refit(matrices, response, theta, *args, **kwargs):
    """A deterministic refit whose estimates summarize the simulated response."""
    return SimpleNamespace(
        beta=np.array([np.mean(response), np.std(response)]),
        theta=np.array([np.var(response)]),
        sigma=float(np.std(response)),
        converged=True,
        pirls_converged=True,
    )


def summarize_response(result, response, *, index=0):
    """A deterministic nonlinear refit outcome that summarizes the response."""
    return bootstrap._BootstrapOutcome(
        index,
        np.array([np.mean(response), np.std(response), response[0]]),
        np.full_like(result.theta, np.var(response)),
        float(np.std(response)),
    )
