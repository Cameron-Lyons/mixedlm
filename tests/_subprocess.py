"""Run Python subprocesses against the mixedlm under test."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import mixedlm
import pytest

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120


def run_isolated(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run Python against this mixedlm, killing every worker if it times out."""
    package_root = str(Path(mixedlm.__file__).resolve().parents[1])
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
    process = subprocess.Popen(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        pytest.fail(f"{args[-1]} did not finish within {TIMEOUT}s")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
