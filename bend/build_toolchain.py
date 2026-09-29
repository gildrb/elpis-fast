#!/usr/bin/env python3
"""Package the unmodified, pinned Bend 2.0.34 release checker and compiler.

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
from typing import Literal


VERSION = "bend 2.0.34\n"
RELEASE_URL = "https://github.com/bendlang/bend/releases/download/v2.0.34/bend-2.0.34-linux-x64.tar.gz"
RELEASE_SHA256 = "78106a97af242429dcc057258eb8d10f69cddebcd5e263022185a52d003e09bf"
RUNTIME_SHA256 = "7fafb749dd33df446a0cb5839c5694df237312a8e8048d2e9420480164ff5934"
BASE_SHA256 = "c742fae9c49b14f0cc9128429a2c6109364c8a933a142f2c90b9f2e5fd976661"
SOURCE_COMMIT = "7d8a3eb036042c6549461054d25a10f26d361c5c"
SOURCE_URL = f"https://codeload.github.com/bendlang/bend/tar.gz/{SOURCE_COMMIT}"
SOURCE_SHA256 = "2232ba4b9f66f5d3729e47417994952b484e8b5f44823a682c08bf68c51eaa00"
UNMODIFIED_CHECKER_SHA256 = (
    "de2b39db2fcbd2f9115053e85e34d791693e1297bfeb6874da1013ff7b44b7dd"
)
SOURCE_MAIN_SHA256 = "e139cf21c6c925dc2a3f76185c8e2c3fd4c44aee3c9bc8eb2595bc069764500e"
SOURCE_COMP_SHA256 = "eee1bfd161a948d1fbe2451526975bc44d237b60d6b469d740f9f702720e0bbb"
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
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(path: Path, expected: str) -> None:
    require(digest(path) == expected, f"SHA256 mismatch: {path}")


def inventory(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Unexpected symlink: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        _ = stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def release_tree(archive_path: Path, work: Path) -> Path:
    """Authenticate the complete distribution before extracting any member."""
    check(archive_path, RELEASE_SHA256)
    with tarfile.open(archive_path) as archive:
        archive.extractall(work, filter="data")
    root = work / "bend"
    check(root / "bin/bend", RUNTIME_SHA256)
    check(root / "bend2/base.bend", BASE_SHA256)
    return root


def source_tree(archive_path: Path, work: Path) -> Path:
    """Authenticate the immutable upstream source before reading any module."""
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
    require(
        not os.environ.get("BUN_BE_BUN"),
        "Ambient BUN_BE_BUN changes release executable semantics",
    )
    environment = dict(os.environ, BEND_NO_TELEMETRY="1")
    result = subprocess.run(
        [str(runner), "version"],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=60,
    )
    require(
        result.stdout == VERSION.encode("utf-8") and result.stderr == b"",
        "Packaged Bend is not the selected 2.0.34 release checker",
    )


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
    require(
        set(installed) == set(original) | set(COMPILER_NAMES),
        "Installed toolchain file set is not the pinned release layout",
    )
    resources = {name: value for name, value in original.items() if name != "bin/bend"}
    require(
        all(installed[name] == value for name, value in resources.items()),
        "Installed Base/effects/guides differ from the selected release",
    )
    check(toolchain / "bin/bend", WRAPPER_SHA256)
    if transport == "release":
        check(toolchain / "bin/bend-runtime", RUNTIME_SHA256)
    runner = toolchain / "bin/bend"
    require(
        (recipe is not None) == (transport == "nix-patchelf"),
        "Relocation recipe is only valid for the reproducible Nix build",
    )
    version(runner)
    require(
        inventory(toolchain) == installed,
        "Installed toolchain changed while recording identity",
    )
    write_json(
        provenance,
        {
            "schema": 6,
            "claim": (
                "Pinned, unmodified Bend 2.0.34 release checker/compiler and "
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
    output = args.output.resolve()
    provenance = args.provenance.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    require(not provenance.exists(), f"Provenance already exists: {provenance}")
    require(
        not provenance.is_relative_to(output),
        "Keep provenance outside the compiler tree",
    )
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
