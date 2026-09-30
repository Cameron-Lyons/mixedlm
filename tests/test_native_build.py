"""Prevent apparently passing tests from exercising a different Rust source tree."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.native_build import NativeBuildError, check_native_build, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "sources with spaces"
    contents = {
        "Cargo.toml": b'[package]\nname = "fixture"\nversion = "1.0.0"\n',
        "Cargo.lock": b"version = 4\n",
        "build.rs": b"fn main() {}\n",
        "src/lib.rs": b"mod nested;\n",
        "src/nested/value.rs": b"pub const VALUE: u32 = 1;\n",
    }
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.mark.parametrize("name", ["Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs"])
def test_missing_native_inputs_are_reported(source_tree, name):
    (source_tree / name).unlink()
    with pytest.raises(NativeBuildError, match=f"missing {name}"):
        source_fingerprint(source_tree)


@pytest.mark.parametrize(
    "name", ["Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs", "src/nested/value.rs"]
)
def test_content_changes_are_detected_even_with_preserved_timestamps(source_tree, name):
    before = source_fingerprint(source_tree)
    path = source_tree / name
    original_stat = path.stat()
    path.write_bytes(path.read_bytes() + b"\n// changed\n")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source_fingerprint(source_tree) != before


def test_adding_and_removing_rust_files_changes_the_fingerprint(source_tree):
    before = source_fingerprint(source_tree)
    extra = source_tree / "src/nested/extra.rs"
    extra.write_text("pub const VALUE: u32 = 2;\n")
    assert source_fingerprint(source_tree) != before
    extra.unlink()
    assert source_fingerprint(source_tree) == before


@pytest.mark.parametrize("name", ["README.md", "src/notes.txt", "src/cache.py"])
def test_non_native_files_do_not_invalidate_the_build(source_tree, name):
    before = source_fingerprint(source_tree)
    (source_tree / name).write_text("documentation or Python changes\n")
    assert source_fingerprint(source_tree) == before


def test_fingerprint_ignores_checkout_location_and_timestamps(source_tree, tmp_path):
    relocated = tmp_path / "relocated"
    # Reverse the creation order as well as moving the tree.
    for path in sorted(source_tree.rglob("*"), reverse=True):
        if path.is_file():
            target = relocated / path.relative_to(source_tree)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            os.utime(target, (1_600_000_000, 1_600_000_000))
    assert source_fingerprint(relocated) == source_fingerprint(source_tree)


@pytest.mark.parametrize("failure", ["file", "directory"])
def test_unreadable_sources_are_reported(source_tree, monkeypatch, failure):
    if failure == "file":

        def unreadable(path):
            raise PermissionError("source cannot be read")

        monkeypatch.setattr(Path, "read_bytes", unreadable)
    else:

        def unreadable_walk(path, *, followlinks, onerror):
            onerror(PermissionError("directory cannot be read"))
            return []

        monkeypatch.setattr("tools.native_build.os.walk", unreadable_walk)
    with pytest.raises(NativeBuildError, match="Cannot read Rust sources"):
        source_fingerprint(source_tree)


@pytest.mark.parametrize("stored", [None, "fnv1a64:0000000000000000", "invalid"])
def test_stale_or_older_libraries_have_actionable_errors(source_tree, stored):
    native = ModuleType("fake_native")
    native.__file__ = str(source_tree / "old_library.so")
    if stored is not None:
        native._source_fingerprint = stored
    with pytest.raises(NativeBuildError) as error:
        check_native_build(source_tree, native)
    message = str(error.value)
    assert "does not match" in message
    assert native.__file__ in message
    assert str(source_tree) in message
    assert "cargo clean --release --package mixedlm" in message
    assert "pip install -e" in message


def test_matching_library_is_accepted(source_tree):
    native = ModuleType("fake_native")
    native._source_fingerprint = source_fingerprint(source_tree)
    check_native_build(source_tree, native)


def test_native_and_python_fingerprints_agree():
    native = pytest.importorskip("mixedlm._rust")
    assert native._source_fingerprint == source_fingerprint(ROOT)
    check_native_build(ROOT, native)


def test_missing_library_is_reported_with_a_rebuild_command(source_tree, monkeypatch):
    def unavailable(name):
        raise ImportError("native library is unavailable")

    monkeypatch.setattr("tools.native_build.importlib.import_module", unavailable)
    with pytest.raises(NativeBuildError, match="Cannot load the native backend") as error:
        check_native_build(source_tree)
    assert "cargo clean" in str(error.value)


@pytest.mark.parametrize("valid", [False, True])
def test_command_checks_the_loaded_library(source_tree, valid):
    root = ROOT if valid else source_tree
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/native_build.py"), "--source-root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == (0 if valid else 1), completed.stderr
    if valid:
        assert "matches the Rust sources" in completed.stdout
    else:
        assert "does not match" in completed.stderr
        assert "cargo clean" in completed.stderr
        assert "Traceback" not in completed.stderr


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("collect_only", [False, True])
@pytest.mark.parametrize("native_available", [False, True])
def test_pytest_rejects_stale_code_before_executing_tests(
    tmp_path, changed, collect_only, native_available
):
    for name in ["Cargo.toml", "Cargo.lock", "build.rs"]:
        shutil.copyfile(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    (tmp_path / "tools").mkdir()
    for name in ["__init__.py", "native_build.py"]:
        shutil.copyfile(ROOT / "tools" / name, tmp_path / "tools" / name)
    (tmp_path / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/conftest.py", tmp_path / "tests/conftest.py")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n'
    )
    (tmp_path / "tests/test_probe.py").write_text(
        "from pathlib import Path\ndef test_probe():\n    Path('test-ran').write_text('yes')\n"
    )
    if not native_available:
        (tmp_path / "mixedlm").mkdir()
        (tmp_path / "mixedlm/__init__.py").write_text("")
        (tmp_path / "mixedlm/_rust.py").write_text(
            "raise ImportError('native backend unavailable')\n"
        )
    if changed:
        path = tmp_path / "src/lib.rs"
        stat = path.stat()
        path.write_bytes(path.read_bytes() + b"\n// a different checkout\n")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(["--collect-only"] if collect_only else [])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    rejected = changed and native_available and not collect_only
    assert completed.returncode == (4 if rejected else 0), completed.stdout + completed.stderr
    assert (tmp_path / "test-ran").exists() == (not rejected and not collect_only)
    if rejected:
        assert "does not match" in completed.stderr
        assert "cargo clean" in completed.stderr
