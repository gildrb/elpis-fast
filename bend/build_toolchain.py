#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Package the unmodified, pinned Bend 2.0.35 release checker and compiler.

The release ELF and resources stay byte-identical to upstream except for
Nix ELF relocation. A small launcher supplies the OS and JavaScriptCore
stack limits required by the project proofs; it never changes the checker.
The matching immutable source archive is authenticated for provenance.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Literal, NoReturn

VERSION = "bend 2.0.35\n"
RELEASE_URL = (
    "https://github.com/bendlang/bend/releases/download/v2.0.35/"
    "bend-2.0.35-linux-x64.tar.gz"
)
RELEASE_SHA256 = "63039d1a119f716767ac5a7d8fe0717cfacf219c6c253c35192148e0dade722f"
RUNTIME_SHA256 = "af967c7c37aced4491e0f4344e3bb9de837c2c89b7519f7e0b8c71935be1840a"
BASE_SHA256 = "c742fae9c49b14f0cc9128429a2c6109364c8a933a142f2c90b9f2e5fd976661"
SOURCE_COMMIT = "79df8d9c40722ee9507a1e253f283b51025f9d6c"
SOURCE_URL = f"https://codeload.github.com/bendlang/bend/tar.gz/{SOURCE_COMMIT}"
SOURCE_SHA256 = "ad7ac21c0145dacaab012ff6fc6c359e57f956cd6940cf5bd94a1b8b12745949"
UNMODIFIED_CHECKER_SHA256 = (
    "7deae3693eb896f33c73867081b99d2c6f3ed3b57e77e55eb5f6260840dd0e63"
)
SOURCE_MAIN_SHA256 = "d1a3e026f5014f8daec3614df8e39cf261e3fbc47769eb0916fd891bc18703c9"
SOURCE_COMP_SHA256 = "32fb66e09f608ce9e4b173384bcfeec453db8c5bc96650e26ad861bef815a8d9"
WRAPPER_SHA256 = "437f2f10c027d4b1a68d86b08d372a0bc32786b2304b658b5d07eefb79de437d"
COMPILER_NAMES = ("bin/bend", "bin/bend-runtime")
LAUNCHER = """#!/bin/sh
# The unmodified release checker needs explicit OS and JavaScriptCore stacks.
set -eu
if [ -n "${BUN_BE_BUN:-}" ]; then
  echo 'BUN_BE_BUN changes release executable semantics' >&2
  exit 1
fi
root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
ulimit -s 1048576
export BEND_NO_TELEMETRY=1 BUN_JSC_maxPerThreadStackUsage=536870912
exec "$root/bin/bend-runtime" "$@"
"""


class Arguments(argparse.Namespace):
    """Typed command arguments; each command reads only its own fields."""

    command: str = ""
    release: Path = Path()
    source: Path = Path()
    output: Path = Path()
    provenance: Path = Path()
    recipe: Path = Path()
    toolchain: Path = Path()
    nix_relocation: bool = False


def digest(path: Path) -> str:
    """Hash a file.

    Returns:
        The file's SHA-256 hex digest.

    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fail(message: str) -> NoReturn:
    """Stop the build.

    Raises:
        ValueError: Always.

    """
    raise ValueError(message)


def check(path: Path, expected: str) -> None:
    """Require the file's SHA-256 to be the expected one."""
    if digest(path) != expected:
        fail(f"SHA256 mismatch: {path}")


def inventory(root: Path) -> dict[str, str]:
    """Hash every regular file below root, refusing symlinks.

    Returns:
        Relative POSIX path -> SHA-256, in sorted path order.

    """
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail(f"Unexpected symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def write_json(path: Path, value: dict[str, object]) -> None:
    """Write value as sorted, indented JSON to a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        _ = stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def release_tree(archive_path: Path, work: Path) -> Path:
    """Authenticate the complete distribution before extracting any member.

    Returns:
        The extracted release root.

    """
    check(archive_path, RELEASE_SHA256)
    with tarfile.open(archive_path) as archive:
        archive.extractall(work, filter="data")
    root = work / "bend"
    check(root / "bin/bend", RUNTIME_SHA256)
    check(root / "bend2/base.bend", BASE_SHA256)
    return root


def source_tree(archive_path: Path, work: Path) -> Path:
    """Authenticate the immutable upstream source before reading any module.

    Returns:
        The extracted source root.

    """
    check(archive_path, SOURCE_SHA256)
    with tarfile.open(archive_path) as archive:
        archive.extractall(work, filter="data")
    root = work / f"bend-{SOURCE_COMMIT}"
    check(root / "bend2/main.ts", SOURCE_MAIN_SHA256)
    check(root / "bend2/comp.ts", SOURCE_COMP_SHA256)
    check(root / "bend2/bend.ts", UNMODIFIED_CHECKER_SHA256)
    return root


def version(runner: Path) -> None:
    """Require the packaged launcher's documented version command to succeed."""
    if os.environ.get("BUN_BE_BUN"):
        fail("Ambient BUN_BE_BUN changes release executable semantics")
    environment = dict(os.environ, BEND_NO_TELEMETRY="1")
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: freshly built bend runner + 'version', no shell
        [str(runner), "version"],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=60,
    )
    if not (result.stdout == VERSION.encode("utf-8") and result.stderr == b""):
        fail("Packaged Bend is not the selected 2.0.35 release checker")


def assemble(release: Path, output: Path) -> None:
    """Wrap the release runtime without modifying its checker or resources."""
    _ = shutil.copytree(release, output)
    _ = (output / "bin/bend").rename(output / "bin/bend-runtime")
    _ = (output / "bin/bend").write_text(LAUNCHER, encoding="utf-8")
    (output / "bin/bend").chmod(0o755)
    (output / "bin/bend-runtime").chmod(0o755)


def record(
    release: Path,
    toolchain: Path,
    provenance: Path,
    *,
    transport: Literal["release", "nix-patchelf"],
    recipe: Path | None = None,
) -> None:
    """Record assembled bytes; consumers admit only observed, pinned hashes."""
    original = inventory(release)
    installed = inventory(toolchain)
    if set(installed) != set(original) | set(COMPILER_NAMES):
        fail("Installed toolchain file set is not the pinned release layout")
    resources = {name: value for name, value in original.items() if name != "bin/bend"}
    if not all(installed[name] == value for name, value in resources.items()):
        fail("Installed Base/effects/guides differ from the selected release")
    check(toolchain / "bin/bend", WRAPPER_SHA256)
    if transport == "release":
        check(toolchain / "bin/bend-runtime", RUNTIME_SHA256)
    runner = toolchain / "bin/bend"
    if (recipe is not None) != (transport == "nix-patchelf"):
        fail("Relocation recipe is only valid for the reproducible Nix build")
    version(runner)
    if inventory(toolchain) != installed:
        fail("Installed toolchain changed while recording identity")
    write_json(
        provenance,
        {
            "schema": 6,
            "claim": (
                "Pinned, unmodified Bend 2.0.35 release checker/compiler and "
                "Base; launcher configures OS and JavaScriptCore stack limits; "
                "matching upstream source is authenticated without patching"
            ),
            "release": {
                "url": RELEASE_URL,
                "sha256": RELEASE_SHA256,
                "version": VERSION.rstrip("\n"),
            },
            "source": {
                "url": SOURCE_URL,
                "sha256": SOURCE_SHA256,
                "commit": SOURCE_COMMIT,
            },
            "checker": {
                "unmodified_bend_ts_sha256": UNMODIFIED_CHECKER_SHA256,
                "main_ts_sha256": SOURCE_MAIN_SHA256,
                "comp_ts_sha256": SOURCE_COMP_SHA256,
                "launcher_sha256": WRAPPER_SHA256,
            },
            "base_bend_sha256": BASE_SHA256,
            "build_helper_sha256": digest(Path(__file__)),
            "release_resources_sha256": resources,
            "runtime": {
                "release_executable_sha256": RUNTIME_SHA256,
                "packaged_executable_sha256": installed["bin/bend-runtime"],
                "transport": transport,
                "relocation_recipe_sha256": digest(recipe)
                if recipe is not None
                else None,
            },
            "execution": {
                "mode": "release",
                "command_prefix": ["bin/bend"],
                "interpreter": "bin/bend-runtime",
                "environment": {
                    "BEND_NO_TELEMETRY": "1",
                    "BUN_JSC_maxPerThreadStackUsage": "536870912",
                },
                "stack_limit_bytes": 1073741824,
            },
            "installed_compiler_sha256": {
                name: installed[name] for name in COMPILER_NAMES
            },
        },
    )


def build(args: Arguments) -> None:
    """Assemble the toolchain; record it unless Nix relocates it first."""
    output = args.output.resolve()
    provenance = args.provenance.resolve()
    if output.exists():
        fail(f"Output already exists: {output}")
    if provenance.exists():
        fail(f"Provenance already exists: {provenance}")
    if provenance.is_relative_to(output):
        fail("Keep provenance outside the compiler tree")
    with tempfile.TemporaryDirectory(prefix="bend-release-") as temporary:
        work = Path(temporary)
        release = release_tree(args.release.resolve(), work / "release")
        _ = source_tree(args.source.resolve(), work / "source")
        assemble(release, output)
        if not args.nix_relocation:
            record(
                release,
                output,
                provenance,
                transport="release",
            )


def main() -> None:
    """Dispatch the build or record-relocated command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    for name in ("release", "source", "output", "provenance"):
        _ = builder.add_argument(name, type=Path)
    _ = builder.add_argument(
        "--nix-relocation",
        action="store_true",
        help="Defer native version/provenance until Nix has relocated the ELF",
    )
    relocated = commands.add_parser(
        "record-relocated", help="Record the frozen Nix build's native output"
    )
    for name in ("release", "source", "toolchain", "provenance", "recipe"):
        _ = relocated.add_argument(name, type=Path)
    args = parser.parse_args(namespace=Arguments())
    if args.command == "build":
        build(args)
    else:
        with tempfile.TemporaryDirectory(prefix="bend-release-") as temporary:
            work = Path(temporary)
            release = release_tree(args.release.resolve(), work / "release")
            _ = source_tree(args.source.resolve(), work / "source")
            record(
                release,
                args.toolchain.resolve(),
                args.provenance.resolve(),
                transport="nix-patchelf",
                recipe=args.recipe.resolve(),
            )


if __name__ == "__main__":
    main()
