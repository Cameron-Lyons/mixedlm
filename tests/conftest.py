"""Require source-matched native code before executing repository tests."""

import importlib
from pathlib import Path

import pytest

from tools.native_build import NativeBuildError, check_native_build


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
