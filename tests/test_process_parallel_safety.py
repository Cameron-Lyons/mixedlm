"""Process-parallel inference in real worker processes, isolated in subprocesses."""

import json
import os
import textwrap
from pathlib import Path

import pytest

from tests._subprocess import run_isolated

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS")

# A correlated-slope fit and an nAGQ > 1 fit start the native thread pools that a
# forked worker inherits without their threads; each workload then compares its
# two-worker result with the serial result in the same process. From Python 3.12,
# forking this multi-threaded process also raises, so a fork path fails even when
# its workers would not block.
WORKLOADS = textwrap.dedent("""
    import sys
    import warnings

    import mixedlm as mlm
    import numpy as np
    from mixedlm.inference.allfit import allfit_lmer
    from mixedlm.inference.bootstrap import bootstrap_glmer
    from mixedlm.inference.drop1 import drop1_lmer

    warnings.simplefilter("ignore")
    warnings.filterwarnings("error", "This process .* use of fork", DeprecationWarning)
    scenario = sys.argv[1]
    data = mlm.load_sleepstudy()
    fitted = mlm.lmer("Reaction ~ Days + (Days | Subject)", data)

    def run(n_jobs):
        if scenario == "bootMer":
            boot = mlm.bootMer(fitted, nsim=4, seed=11, n_jobs=n_jobs)
            return boot.beta_samples, boot.theta_samples, boot.sigma_samples
        if scenario == "drop1_lmer":
            quadratic = mlm.lmer("Reaction ~ Days + I(Days**2) + (Days | Subject)", data)
            table = drop1_lmer(quadratic, data, n_jobs=n_jobs)
            assert table.terms == ["Days", "I(Days**2)"], table.terms
            return table.aic, table.lrt
        if scenario == "allfit":
            optimizers = ["COBYQA", "Nelder-Mead"]
            comparisons = [
                allfit_lmer(fitted, data, optimizers, n_jobs=n_jobs),
                mlm.allFit("Reaction ~ Days + (Days | Subject)", data, optimizers, n_jobs=n_jobs),
            ]
            assert not any(comparison.errors for comparison in comparisons)
            return [
                list(fit.theta) + [fit.deviance]
                for comparison in comparisons
                for fit in comparison.fits.values()
            ]
        glmm = mlm.glmer(
            "incidence / size ~ period + (1 | herd)",
            mlm.load_cbpp(),
            family=mlm.families.Binomial(),
            nAGQ=5,
        )
        boot = bootstrap_glmer(glmm, n_boot=4, seed=13, n_jobs=n_jobs)
        return boot.beta_samples, boot.theta_samples

    serial = run(1)
    parallel = run(2)
    for expected, actual in zip(serial, parallel, strict=True):
        assert np.all(np.isfinite(expected)), expected
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)
""")

# A user script that fits at import time and guards only the parallel call.
# Python 3.14 passes the script path to a forkserver asked to preload "__main__",
# which then runs that native fit in the server; the script emulates this on
# older versions, whose forkserver never receives the path.
SCRIPT_WITH_TOP_LEVEL_FIT = textwrap.dedent("""
    import sys
    from multiprocessing import spawn

    if sys.version_info < (3, 14):
        preparation_data = spawn.get_preparation_data

        def get_preparation_data(name):
            data = preparation_data(name)
            data["main_path"] = data.get("init_main_from_path")
            return data

        spawn.get_preparation_data = get_preparation_data

    import mixedlm as mlm

    fitted = mlm.lmer("Reaction ~ Days + (Days | Subject)", mlm.load_sleepstudy())

    if __name__ == "__main__":
        boot = mlm.bootMer(fitted, nsim=2, seed=1, n_jobs=2)
        assert boot.n_failed == 0, boot.failures
""")

# Run as a user script; workers report the thread count of the BLAS they loaded.
THREAD_PROBE = textwrap.dedent("""
    import ctypes
    import json
    import os

    import numpy as np
    from mixedlm._parallel import process_pool
    from scipy import linalg

    VARIABLES = {variables!r}

    def thread_settings():
        np.ones((256, 256)) @ np.ones((256, 256))
        linalg.cho_factor(np.eye(256))
        with open("/proc/self/maps") as maps:
            paths = sorted({{line.split()[-1] for line in maps if "openblas" in line}})
        threads = []
        for path in paths:
            library = ctypes.CDLL(path)
            for name in (
                "scipy_openblas_get_num_threads64_",
                "scipy_openblas_get_num_threads",
                "openblas_get_num_threads64_",
                "openblas_get_num_threads",
            ):
                if hasattr(library, name):
                    threads.append(getattr(library, name)())
                    break
        return {{"openblas": threads, "env": {{name: os.environ.get(name) for name in VARIABLES}}}}

    if __name__ == "__main__":
        environment = dict(os.environ)
        with process_pool(2) as executor:
            workers = [executor.submit(thread_settings) for _ in range(4)]
            workers = [future.result() for future in workers]
        print(json.dumps({{
            "workers": workers,
            "parent": thread_settings(),
            "restored": dict(os.environ) == environment,
        }}))
""").format(variables=THREAD_VARIABLES)


@pytest.mark.parametrize("scenario", ["bootMer", "drop1_lmer", "allfit", "bootstrap_glmer_nagq"])
def test_parallel_workers_after_native_fit_finish_and_match_serial(scenario):
    completed = run_isolated(["-c", WORKLOADS, scenario])
    assert completed.returncode == 0, completed.stderr


def test_script_fitting_at_import_time_runs_parallel_inference(tmp_path):
    script = tmp_path / "top_level_fit.py"
    script.write_text(SCRIPT_WITH_TOP_LEVEL_FIT)
    completed = run_isolated([str(script)])
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="needs /proc/self/maps")
def test_workers_start_with_single_threaded_blas_unless_the_caller_chose(tmp_path):
    script = tmp_path / "thread_probe.py"
    script.write_text(THREAD_PROBE)
    env = {name: value for name, value in os.environ.items() if name not in THREAD_VARIABLES}
    env["OMP_NUM_THREADS"] = "3"
    completed = run_isolated([str(script)], env=env)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)

    if not report["parent"]["openblas"]:
        pytest.skip("NumPy and SciPy are not linked against OpenBLAS")
    assert report["restored"]
    assert report["parent"]["env"] == {
        "OPENBLAS_NUM_THREADS": None,
        "OMP_NUM_THREADS": "3",
        "RAYON_NUM_THREADS": None,
    }
    for worker in report["workers"]:
        assert worker["env"] == {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "3",
            "RAYON_NUM_THREADS": "1",
        }
        assert worker["openblas"] == [1] * len(report["parent"]["openblas"])
