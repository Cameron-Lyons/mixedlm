"""One worker-count policy for every parallel entry point."""

from types import SimpleNamespace
from unittest.mock import patch

import mixedlm as mlm
import numpy as np
import pytest
from mixedlm import _parallel
from mixedlm.inference.allfit import allfit_lmer
from mixedlm.inference.drop1 import drop1_lmer
from mixedlm.inference.profile import profile_lmer, slice2D

INVALID = [(0, ValueError), (-7, ValueError), (True, TypeError), (1.5, TypeError)]
MESSAGE = r"^n_jobs must be -1 or a positive integer$"


@pytest.mark.parametrize(
    "jobs,cpu,platform,max_tasks,expected",
    [
        (-1, None, "linux", 10, 1),
        (-1, 6, "linux", None, 6),
        (-1, 128, "linux", 3, 3),
        (-1, 128, "win32", 100, 61),
        (100, 8, "win32", None, 61),
        (np.int64(2), 8, "linux", 10, 2),
        (100, 8, "linux", 3, 3),
        (1, 8, "linux", 5, 1),
        (4, 8, "linux", 0, 1),
    ],
)
def test_worker_count_uses_available_cpus_and_never_exceeds_tasks(
    jobs, cpu, platform, max_tasks, expected
):
    with (
        patch.object(_parallel.os, "cpu_count", return_value=cpu),
        patch.object(_parallel, "sys", SimpleNamespace(platform=platform)),
    ):
        workers = _parallel.resolve_n_jobs(jobs, max_tasks=max_tasks)
    assert type(workers) is int
    assert workers == expected


@pytest.mark.parametrize(
    "jobs,error", [*INVALID, (np.bool_(True), TypeError), ("2", TypeError), (None, TypeError)]
)
def test_invalid_worker_counts_are_rejected(jobs, error):
    with pytest.raises(error, match=MESSAGE):
        _parallel.resolve_n_jobs(jobs)


@pytest.fixture(scope="module")
def sleepstudy_fit():
    data = mlm.load_sleepstudy()
    return mlm.lmer("Reaction ~ Days + (1 | Subject)", data), data


ENTRY_POINTS = {
    "bootMer": lambda fit, data, jobs: mlm.bootMer(fit, nsim=2, n_jobs=jobs),
    "drop1_lmer": lambda fit, data, jobs: drop1_lmer(fit, data, n_jobs=jobs),
    "allfit_lmer": lambda fit, data, jobs: allfit_lmer(fit, data, ["COBYQA"], n_jobs=jobs),
    "allFit": lambda fit, data, jobs: mlm.allFit(
        "Reaction ~ Days + (1 | Subject)", data, ["COBYQA"], n_jobs=jobs
    ),
    "slice2D": lambda fit, data, jobs: slice2D(fit, "(Intercept)", "Days", 3, n_jobs=jobs),
    "profile_lmer": lambda fit, data, jobs: profile_lmer(fit, n_jobs=jobs),
}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("jobs,error", INVALID)
def test_public_entry_points_reject_the_same_worker_counts(sleepstudy_fit, entry, jobs, error):
    fit, data = sleepstudy_fit
    with pytest.raises(error, match=MESSAGE):
        ENTRY_POINTS[entry](fit, data, jobs)
