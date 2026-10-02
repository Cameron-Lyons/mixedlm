"""Exercise an installed wheel in isolation from the source checkout.

Run with ``python -I tools/check_wheel.py`` after installing the built wheel.
These checks deliberately require the native backend and packaged datasets;
an editable install or Python fallback must not hide a broken distribution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import mixedlm
import mixedlm._rust as native
import numpy as np
from mixedlm.estimation import laplace, reml
from numpy.testing import assert_allclose
from scipy import sparse


def check_datasets() -> None:
    """Verify that every canonical data asset survives wheel/sdist packaging."""
    resources = files("mixedlm.datasets").joinpath("data")
    manifest = json.loads(resources.joinpath("provenance.json").read_text(encoding="utf-8"))
    expected_names = {
        "sleepstudy",
        "cbpp",
        "Dyestuff",
        "Dyestuff2",
        "Penicillin",
        "cake",
        "Pastes",
        "InstEval",
        "Arabidopsis",
        "grouseticks",
        "VerbAgg",
    }
    if set(manifest["datasets"]) != expected_names:
        raise RuntimeError("The installed dataset provenance manifest is incomplete.")
    for name, metadata in manifest["datasets"].items():
        compressed = resources.joinpath(metadata["file"]).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != metadata["compressed_sha256"]:
            raise RuntimeError(f"Installed {name} data does not match its source manifest.")
        if hashlib.sha256(gzip.decompress(compressed)).hexdigest() != metadata["csv_sha256"]:
            raise RuntimeError(f"Installed {name} CSV does not match its source manifest.")
        frame = getattr(mixedlm, f"load_{name.lower()}")()
        if len(frame) != metadata["rows"] or not set(metadata["columns"]).issubset(frame.columns):
            raise RuntimeError(f"Installed {name} loader changed the canonical data shape.")


def check_wheel(*, expect_free_threaded: bool = False) -> None:
    """Verify packaging, native numerics, fits, and shared native thread safety."""
    if not sys.flags.isolated:
        raise RuntimeError("Run this wheel check with python -I to exclude checkout imports.")
    prefix = Path(sys.prefix).resolve()
    for module in (mixedlm, native):
        if module.__file__ is None:
            raise RuntimeError(f"Installed module {module.__name__} has no filesystem location.")
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(prefix):
            raise RuntimeError(f"Expected an installed wheel under {prefix}; imported {location}.")
    if mixedlm.__version__ != version("mixedlm"):
        raise RuntimeError("Installed distribution metadata disagrees with the package version.")
    if not reml._HAS_RUST or not laplace._HAS_RUST:
        raise RuntimeError("The installed native backend could not initialize model fitting.")
    if expect_free_threaded:
        if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
            raise RuntimeError(
                "The free-threaded wheel check requires a free-threaded interpreter."
            )
        if getattr(sys, "_is_gil_enabled", lambda: True)():
            raise RuntimeError("Importing wheel dependencies enabled the GIL.")

    check_datasets()

    matrix = sparse.csc_matrix([[4.0, -1.0, 0.0], [-1.0, 3.0, -0.5], [0.0, -0.5, 2.0]])
    rhs = np.array([[1.0, -2.0], [2.0, 0.5], [3.0, 1.0]])
    symbolic = native.SparseCholeskySymbolic(
        matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64), 3
    )
    numeric = symbolic.factor(matrix.data)
    expected = np.linalg.solve(matrix.toarray(), rhs)
    assert_allclose(numeric.solve(rhs), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(numeric.logdet(), np.linalg.slogdet(matrix.toarray())[1], rtol=1e-12)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for actual in pool.map(numeric.solve, [rhs] * 16):
            assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    data = mixedlm.load_sleepstudy()
    if len(data) != 180 or data["Subject"].nunique() != 18:
        raise RuntimeError("The packaged sleepstudy dataset is missing rows or grouping levels.")
    model = mixedlm.lmer("Reaction ~ Days + (Days | Subject)", data)
    # Balanced observation times make the GLS fixed effects equal to pooled OLS.
    design = np.column_stack((np.ones(len(data)), data["Days"].to_numpy()))
    expected_beta = np.linalg.lstsq(design, data["Reaction"].to_numpy(), rcond=None)[0]
    assert_allclose(model.beta, expected_beta, rtol=0, atol=1e-8)
    assert_allclose(model.fitted() + model.residuals(), data["Reaction"].to_numpy(), atol=1e-10)
    assert_allclose(model.predict(), model.fitted(), rtol=1e-12, atol=1e-12)
    if not model.converged or not np.isfinite(model.deviance) or model.sigma <= 0:
        raise RuntimeError("Installed wheel failed to fit the packaged linear mixed model.")

    counts = mixedlm.load_cbpp()
    if len(counts) != 56 or counts["herd"].nunique() != 15:
        raise RuntimeError("The packaged cbpp dataset is missing rows or grouping levels.")
    generalized = mixedlm.glmer(
        "incidence / size ~ period + (1 | herd)",
        counts,
        family=mixedlm.families.Binomial(),
    )
    means = generalized.fitted()
    assert_allclose(generalized.predict(), means, rtol=1e-12, atol=1e-12)
    assert_allclose(generalized.family.link.inverse(generalized.fitted(type="link")), means)
    if (
        not generalized.converged
        or not np.isfinite(generalized.deviance)
        or not np.all((means > 0) & (means < 1))
    ):
        raise RuntimeError("Installed wheel failed to fit the packaged grouped binomial model.")
    # Published Laplace fit of the canonical official lme4 CBPP data. This
    # also checks the binomial normalizing constants in likelihood reporting.
    assert_allclose(float(generalized.logLik()), -92.0266, rtol=0, atol=0.002)
    assert_allclose(generalized.AIC(), 194.0531, rtol=0, atol=0.004)

    if expect_free_threaded and getattr(sys, "_is_gil_enabled", lambda: True)():
        raise RuntimeError("Native model evaluation enabled the GIL.")
    print(f"Installed mixedlm {mixedlm.__version__} wheel passed native, model, and thread checks.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-free-threaded", action="store_true")
    args = parser.parse_args(argv)
    check_wheel(expect_free_threaded=args.expect_free_threaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
