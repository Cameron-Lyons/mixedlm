"""Shared test setup: source-matched native code, headless plots and canonical fits.

The session-scoped fits are shared by many tests, so tests must treat them as
read-only. Fit a private model when a test mutates or monkeypatches a result.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

from tools.native_build import NativeBuildError, check_native_build

# Render figures off-screen; this must happen before matplotlib is imported.
os.environ["MPLBACKEND"] = "Agg"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "installed_wheel: regression tests run against installed wheels in CI"
    )


def pytest_sessionstart(session):
    if session.config.option.collectonly:
        return
    try:
        native = importlib.import_module("mixedlm._rust")
    except ImportError:
        # Preserve the existing behavior of tests that can run without Rust.
        return
    try:
        check_native_build(Path(__file__).resolve().parents[1], native)
    except NativeBuildError as error:
        raise pytest.UsageError(str(error)) from error


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


# Fixtures import lazily so the native-build check runs before mixedlm loads.
@pytest.fixture(scope="session")
def sleepstudy_lmm():
    """REML fit of ``Reaction ~ Days + (1 | Subject)`` to lme4's sleepstudy."""
    from mixedlm import lmer

    from tests._datasets import SLEEPSTUDY

    return lmer("Reaction ~ Days + (1 | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="session")
def sleepstudy_slopes_lmm():
    """REML fit of ``Reaction ~ Days + (Days | Subject)`` to lme4's sleepstudy."""
    from mixedlm import lmer

    from tests._datasets import SLEEPSTUDY

    return lmer("Reaction ~ Days + (Days | Subject)", SLEEPSTUDY)


@pytest.fixture(scope="session")
def cbpp_glmm():
    """Laplace fit of lme4's CBPP binomial example (non-singular, 15 herds)."""
    from mixedlm import families, glmer

    from tests._datasets import CBPP, CBPP_FORMULA

    return glmer(CBPP_FORMULA, CBPP, family=families.Binomial())


@pytest.fixture(scope="session")
def singular_cbpp_glmm():
    """CBPP proportions fitted as single Bernoulli trials: a boundary fit."""
    from mixedlm import families, glmer

    from tests._datasets import CBPP

    data = CBPP.assign(y=CBPP["incidence"] / CBPP["size"])
    with pytest.warns(UserWarning, match="Model is singular"):
        return glmer("y ~ period + (1 | herd)", data, family=families.Binomial())


@pytest.fixture(scope="session")
def grouped_lmm():
    """REML fit of ``y ~ x + (1 | group)`` to ``grouped_data("gaussian")``."""
    from mixedlm import lmer

    from tests._datasets import grouped_data

    return lmer("y ~ x + (1 | group)", grouped_data())


@pytest.fixture(scope="session")
def grouped_glmm():
    """Laplace fit of ``y ~ x + (1 | group)`` to ``grouped_data("binomial")``."""
    from mixedlm import families, glmer

    from tests._datasets import grouped_data

    return glmer("y ~ x + (1 | group)", grouped_data("binomial"), family=families.Binomial())
