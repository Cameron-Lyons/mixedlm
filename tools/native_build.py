"""Check the loaded native library against a checkout's Rust source bytes."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

_OFFSET = 0xCBF29CE484222325
_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


class NativeBuildError(RuntimeError):
    """The native backend cannot be verified against the requested source tree."""


def _update(checksum: int, content: bytes) -> int:
    for byte in content:
        checksum = ((checksum ^ byte) * _PRIME) & _MASK
    return checksum


def source_fingerprint(source_root: Path) -> str:
    """Hash native inputs in a stable order, independent of timestamps and location."""
    root = source_root.resolve()
    required = ["Cargo.toml", "Cargo.lock", "build.rs", "src/lib.rs"]
    for name in required:
        if not (root / name).is_file():
            raise NativeBuildError(f"Rust source tree {root} is missing {name}.")

    def directory_error(error: OSError) -> None:
        raise NativeBuildError(f"Cannot read Rust sources from {root}: {error}") from error

    paths = [root / name for name in required[:3]]
    for directory, _, files in os.walk(root / "src", followlinks=False, onerror=directory_error):
        paths.extend(
            Path(directory) / name
            for name in files
            if name.endswith(".rs") and (Path(directory) / name).is_file()
        )
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    checksum = _update(_OFFSET, b"mixedlm-native-v1\0")
    try:
        for path in paths:
            checksum = _update(checksum, path.relative_to(root).as_posix().encode("utf-8"))
            checksum = _update(checksum, b"\0")
            checksum = _update(checksum, path.read_bytes())
            checksum = _update(checksum, b"\0")
    except OSError as error:
        raise NativeBuildError(f"Cannot read Rust sources from {root}: {error}") from error
    return f"fnv1a64:{checksum:016x}"


def check_native_build(source_root: Path, native: ModuleType | None = None) -> None:
    """Reject a missing, unversioned, or stale native backend with rebuild instructions."""
    root = source_root.resolve()
    rebuild = (
        f"Rebuild from {root} using the active virtual environment:\n"
        "  cargo clean --release --package mixedlm\n"
        "  python -m pip install -e '.[dev]'"
    )
    if native is None:
        try:
            native = importlib.import_module("mixedlm._rust")
        except ImportError as error:
            raise NativeBuildError(f"Cannot load the native backend: {error}\n{rebuild}") from error
    expected = source_fingerprint(root)
    actual = getattr(native, "_source_fingerprint", None)
    if actual != expected:
        location = getattr(native, "__file__", "unknown location")
        raise NativeBuildError(
            f"Native backend does not match the Rust sources in {root}.\n"
            f"Loaded library: {location}\n"
            f"Expected checksum: {expected}\n"
            f"Loaded checksum: {actual or 'unavailable (rebuild required)'}\n"
            f"{rebuild}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        check_native_build(args.source_root)
    except NativeBuildError as error:
        print(error, file=sys.stderr)
        return 1
    print("Native backend matches the Rust sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
