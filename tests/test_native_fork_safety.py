"""Children forked after native fits run native kernels without the parent's pools."""

import os
import textwrap

import pytest

from tests._subprocess import run_isolated

# The correlated-slope LMM factors a dense 300 x 300 Schur complement on faer's
# global pool, and the nAGQ > 1 GLMM and batched random-effect simulation run on
# rayon's. A forked child inherits both pools without their threads, so before the native
# module switched forked children to sequential kernels both refits blocked.
FORKED_FITS = textwrap.dedent("""
    import multiprocessing
    import os
    import warnings

    import mixedlm as mlm
    import numpy as np
    import pandas as pd

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(1)
    n = 3000
    data = pd.DataFrame(
        {"x": rng.normal(size=n), "g": rng.integers(400, size=n), "h": rng.integers(300, size=n)}
    )
    data["y"] = data.x + rng.normal(size=400)[data.g] + rng.normal(size=300)[data.h]
    data["y"] += rng.normal(size=n)
    cbpp = mlm.load_cbpp()

    def fit():
        fits = [
            mlm.lmer("y ~ x + (x | g) + (1 | h)", data),
            mlm.glmer(
                "incidence / size ~ period + (1 | herd)",
                cbpp,
                family=mlm.families.Binomial(),
                nAGQ=9,
            ),
        ]
        # Batched random-effect draws fill their output on rayon's pool too.
        draws = mlm._rust.simulate_re_batch(
            np.array([1.0, 0.3, 0.8]), 1.0, [500], [2], [True], 4000, seed=3
        )
        summaries = [np.r_[fit.deviance, fit.theta, fit.beta] for fit in fits]
        return np.concatenate([*summaries, draws.mean(axis=0)])

    if __name__ == "__main__":
        expected = fit()
        pid = os.fork()
        if pid == 0:
            os._exit(0 if np.allclose(fit(), expected, rtol=1e-6, atol=0) else 1)
        assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
        with multiprocessing.get_context("fork").Pool(1) as pool:
            np.testing.assert_allclose(pool.apply(fit), expected, rtol=1e-6, atol=0)
""")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_children_forked_after_native_fits_complete_native_fits():
    completed = run_isolated(["-c", FORKED_FITS])
    assert completed.returncode == 0, completed.stderr
