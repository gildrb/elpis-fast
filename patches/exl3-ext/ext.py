#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Rebuild the pinned exllamav3_ext CUDA extension and record it, fail closed.

Run at image build time by Dockerfile.exl3 from a copy of the repository's
patches/ directory (this tool imports the strict patch rules of
patches/exl3/apply.py):

    python -I -B ext.py prepare <rebuilt|patched> <build-root>
    python -I -B ext.py compose <rebuilt|patched> <shared-object>
    python -I -B ext.py record <rebuilt|patched> <engine-manifest>
    python3 -I -B ext.py pin <pristine-engine-root>

prepare checks the installed engine against exl3-ext.json: the package root, the
SHA-256 listing of the whole compiled source tree, and the vendored upstream
setup.py. The installed shared object must equal the hash that the base image
recorded for its own build (extension.base_record, written by
docker/base/manifest.py; nvcc output differs from build to build, so no global
pin exists). For `patched` it then checks the series, every patch and every
pre-image, applies the series in place with the strict hunk rules of
patches/exl3/apply.py, and requires every post-image. Finally it lays out
<build-root> like the upstream checkout (setup.py plus exllamav3/exllamav3_ext)
for the compiler.

compose prints the image's engine manifest: the patches/exl3 manifest with its
per-file closure extended by the extension's patched files, schema 2, plus an
`extension` record naming the variant, source and toolchain pins, the series, the
base build's recorded shared object hash and the SHA-256 of the built shared
object.

record recomposes that manifest from the installed shared object, requires the
installed manifest to be patches/exl3's, rehashes every recorded file and the
acceptance artifact, and replaces <engine-manifest>.

pin runs on the host from the repository. It requires <pristine-engine-root> (the
upstream exllamav3 package directory) to match the pinned source tree, copies it to
a scratch directory, applies the pinned patches/exl3 series (as the candidate image
does before the extension build), then applies the series file in order with the
same strict applier, recording each touched file's first pre-image and final
post-image. It then replays the result through prepare's own apply path on a fresh
copy and only then rewrites series_sha256, patches and files of exl3-ext.json,
keeping every other field.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patches.exl3.apply import (
    MANIFEST_KEYS,
    SERIES_LINE,
    apply_file,
    check_acceptor,
    check_files,
    check_post_images,
    check_pre_images,
    check_series,
    digest,
    fail,
    parse_patch,
    pinned,
    safe_relative,
    text_value,
)

EXT_DIR = Path(__file__).resolve().parent
EXL3_DIR = EXT_DIR.parent / "exl3"
PIN_ARGC = 3
STEP_ARGC = 4
KEYS = {
    "schema",
    "source",
    "toolchain",
    "extension",
    "series_sha256",
    "patches",
    "files",
}
SOURCE_KEYS = {"revision", "root", "tree", "tree_sha256", "setup_py", "setup_py_sha256"}
VARIANTS = ("rebuilt", "patched")
# Commit time of 355c6ee10fbd25b79070316a81ea0708cc18155a (2026-09-17T16:34:03Z).
SOURCE_MTIME = 1789662843


def mapping(value: object, label: str) -> dict[str, object]:
    """Require a JSON object with string keys.

    Returns:
        The validated object.

    """
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail(f"{label} is not an object")
    return {str(key): item for key, item in value.items()}


def load() -> dict[str, object]:
    """Read and shape-check the extension manifest.

    Returns:
        The manifest.

    """
    manifest = mapping(json.loads((EXT_DIR / "exl3-ext.json").read_bytes()), "manifest")
    if set(manifest) != KEYS:
        fail("bad extension manifest keys")
    if manifest["schema"] != 1:
        fail("unsupported extension manifest schema")
    if set(mapping(manifest["source"], "source")) != SOURCE_KEYS:
        fail("bad extension source record")
    return manifest


def tree_digest(root: Path) -> str:
    """Hash a source tree as sorted `<sha256>  <relative path>` lines.

    Returns:
        The lowercase hex SHA-256 of the listing.

    """
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            fail(f"symlink in source tree: {relative}")
        if path.is_file():
            lines.append(f"{digest(path)}  {relative.as_posix()}\n")
    if not lines:
        fail(f"empty source tree {root}")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def base_extension(extension: dict[str, object]) -> tuple[Path, str]:
    """Read the shared object hash that the base image recorded for its build.

    Returns:
        The extension path and its recorded sha256.

    """
    path = text_value(extension["path"], "extension path")
    source = Path(text_value(extension["base_record"], "base record"))
    record = mapping(json.loads(source.read_bytes()), "base record")
    if record.get("schema") != 1:
        fail("unsupported base record schema")
    recorded = mapping(record.get("per_build"), "base per-build record")
    if path not in recorded:
        fail("the base record does not name the extension")
    return Path(path), pinned(recorded[path], "recorded base extension sha256")


def engine_root(source: dict[str, object]) -> Path:
    """Match the installed engine package to the manifest.

    Returns:
        The engine package root.

    """
    root = Path(text_value(source["root"], "source root"))
    spec = importlib.util.find_spec("exllamav3")
    if not (
        spec is not None
        and spec.origin is not None
        and Path(spec.origin).parent == root
    ):
        fail("installed engine root differs from the extension manifest")
    return root


def series(manifest: dict[str, object]) -> list[tuple[str, str]]:
    """Match the series file to the manifest's pinned patch list.

    Returns:
        (patch name, sha256) in application order.

    """
    path = EXT_DIR / "series"
    if digest(path) != pinned(manifest["series_sha256"], "series_sha256"):
        fail("extension series hash mismatch")
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SERIES_LINE.match(line)
        if match is None:
            fail(f"malformed extension series line {line!r}")
        entries.append((match.group(2), match.group(1)))
    if manifest["patches"] != [{"name": n, "sha256": s} for n, s in entries]:
        fail("extension series differs from the manifest")
    return entries


def apply_pinned(
    patch_dir: Path,
    entries: list[tuple[str, str]],
    pins: dict[str, tuple[str | None, str]],
    root: Path,
) -> None:
    """Apply a pinned series in place between pre- and post-image checks."""
    check_pre_images(root, pins)
    touched: set[str] = set()
    for name, sha in entries:
        path = patch_dir / name
        if digest(path) != sha:
            fail(f"patch hash mismatch: {name}")
        for relative, creates, hunks in parse_patch(path.read_text(encoding="utf-8")):
            if relative not in pins:
                fail(f"patch touches unpinned file {relative}")
            if creates != (pins[relative][0] is None and relative not in touched):
                fail(f"file creation disagrees with the manifest: {relative}")
            apply_file(root / relative, creates=creates, hunks=hunks)
            touched.add(relative)
    if touched != set(pins):
        fail("manifest pins files no patch touches")
    check_post_images(root, pins)


def apply_series(manifest: dict[str, object], root: Path) -> None:
    """Apply the pinned extension series in place between image checks."""
    entries = series(manifest)
    if not entries:
        fail("patched variant needs a nonempty series")
    apply_pinned(EXT_DIR, entries, check_files(manifest), root)


def apply_engine(root: Path) -> None:
    """Apply the pinned patches/exl3 series in place, as the candidate image does."""
    engine = mapping(json.loads((EXL3_DIR / "exl3-patches.json").read_bytes()), "m")
    if set(engine) != MANIFEST_KEYS:
        fail("bad patches/exl3 manifest keys")
    if engine["schema"] != 1:
        fail("unsupported patches/exl3 manifest schema")
    apply_pinned(EXL3_DIR, check_series(EXL3_DIR, engine), check_files(engine), root)


def engine_copy(pristine: Path, destination: Path) -> Path:
    """Copy the pristine engine package and apply the patches/exl3 series to it.

    Bytecode caches are not copied.

    Returns:
        The copy's root.

    """
    _ = shutil.copytree(
        pristine,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    apply_engine(destination)
    return destination


def series_patches(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Read each series patch and check it against its series line hash.

    Args:
        entries: (name, sha256) pairs from the series file.

    Returns:
        (name, decoded patch text) pairs in series order.

    """
    patches: list[tuple[str, str]] = []
    for name, sha in entries:
        text = (EXT_DIR / name).read_bytes()
        if hashlib.sha256(text).hexdigest() != sha:
            fail(f"series line hash differs from the patch: {name}")
        patches.append((name, text.decode("utf-8")))
    return patches


def series_entries(series_bytes: bytes) -> list[tuple[str, str]]:
    """Parse the extension series file and require nonempty, unique patch names.

    Args:
        series_bytes: Raw contents of the series file.

    Returns:
        (name, sha256) pairs in series order.

    """
    entries: list[tuple[str, str]] = []
    for line in series_bytes.decode("utf-8").splitlines():
        match = SERIES_LINE.match(line)
        if match is None:
            fail(f"malformed extension series line {line!r}")
        entries.append((match.group(2), match.group(1)))
    if not entries:
        fail("the extension series is empty")
    if len({name for name, _ in entries}) != len(entries):
        fail("duplicate patch in the extension series")
    return entries


def pin(pristine: Path) -> None:
    """Re-pin exl3-ext.json to the series file against a pristine engine package."""
    manifest = load()
    source = mapping(manifest["source"], "source")
    tree = safe_relative(text_value(source["tree"], "source tree"))
    if not (pristine.is_dir() and not pristine.is_symlink()):
        fail(f"{pristine} is not a directory")
    if tree_digest(pristine / tree) != pinned(source["tree_sha256"], "tree_sha256"):
        fail("pristine extension sources differ from the pinned upstream tree")
    series_bytes = (EXT_DIR / "series").read_bytes()
    entries = series_entries(series_bytes)
    patches = series_patches(entries)
    with tempfile.TemporaryDirectory(prefix="exl3-ext-pin-") as scratch:
        work = engine_copy(pristine, Path(scratch) / "pin")
        images: dict[str, str | None] = {}
        for name, text in patches:
            for relative, creates, hunks in parse_patch(text):
                target = work / relative
                if target.is_symlink():
                    fail(f"{name} patches symlink {relative}")
                if creates == target.exists():
                    fail(f"{name}: file creation disagrees with the tree: {relative}")
                if relative not in images:
                    images[relative] = None if creates else digest(target)
                apply_file(target, creates=creates, hunks=hunks)
        repinned: dict[str, object] = {
            **manifest,
            "series_sha256": hashlib.sha256(series_bytes).hexdigest(),
            "patches": [{"name": name, "sha256": sha} for name, sha in entries],
            "files": {
                relative: {"pre": images[relative], "post": digest(work / relative)}
                for relative in sorted(images)
            },
        }
        apply_series(repinned, engine_copy(pristine, Path(scratch) / "check"))
    document = (json.dumps(repinned, indent=2) + "\n").encode()
    out = EXT_DIR / "exl3-ext.json"
    staged = EXT_DIR / "exl3-ext.json.pin"
    descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        _ = handle.write(document)
    _ = staged.replace(out)
    _ = sys.stdout.write(
        f"Pinned {len(entries)} extension patch(es) touching {len(images)} file(s); "
        f"series_sha256 {repinned['series_sha256']}; "
        f"exl3-ext.json sha256 {hashlib.sha256(document).hexdigest()}\n"
    )


def prepare(variant: str, build: Path) -> None:
    """Verify the compiled sources, apply the series if patched, lay out the build."""
    manifest = load()
    source = mapping(manifest["source"], "source")
    root = engine_root(source)
    tree = safe_relative(text_value(source["tree"], "source tree"))
    if tree_digest(root / tree) != pinned(source["tree_sha256"], "tree_sha256"):
        fail("installed extension sources differ from the pinned upstream tree")
    setup_py = EXT_DIR / safe_relative(text_value(source["setup_py"], "setup.py"))
    if digest(setup_py) != pinned(source["setup_py_sha256"], "setup_py_sha256"):
        fail("vendored setup.py differs from its pin")
    extension = mapping(manifest["extension"], "extension")
    so, base_sha256 = base_extension(extension)
    if digest(so) != base_sha256:
        fail("installed extension differs from the base image's recorded build")
    if variant == "patched":
        apply_series(manifest, root)
    if build.exists():
        fail(f"{build} already exists")
    (build / "exllamav3" / tree).mkdir(parents=True)
    _ = (build / "setup.py").write_bytes(setup_py.read_bytes())
    for path in sorted((root / tree).rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        destination = build / "exllamav3" / relative
        if path.is_dir():
            destination.mkdir()
        else:
            _ = destination.write_bytes(path.read_bytes())
    # ptxas writes each source file's mtime into the -lineinfo line table. Use the
    # commit time of the pinned revision (the mtime in its commit tarball, as in the
    # base build) so the build does not depend on when prepare ran.
    for path in [build, *build.rglob("*")]:
        os.utime(path, (SOURCE_MTIME, SOURCE_MTIME), follow_symlinks=False)
    _ = sys.stdout.write(f"Prepared {variant} exllamav3_ext sources at {build}\n")


def compose(variant: str, so_sha256: str) -> bytes:
    """Build the image's engine manifest.

    Returns:
        The canonical JSON bytes.

    """
    manifest = load()
    source = mapping(manifest["source"], "source")
    engine = mapping(json.loads((EXL3_DIR / "exl3-patches.json").read_bytes()), "m")
    if set(engine) != MANIFEST_KEYS:
        fail("bad patches/exl3 manifest keys")
    if engine["schema"] != 1:
        fail("unsupported patches/exl3 manifest schema")
    if mapping(engine["engine"], "engine")["root"] != source["root"]:
        fail("extension and engine manifests name different package roots")
    files = {
        name: mapping(record, name)
        for name, record in mapping(engine["files"], "files").items()
    }
    so, base_sha256 = base_extension(mapping(manifest["extension"], "extension"))
    record: dict[str, object] = {
        "variant": variant,
        "source": source,
        "toolchain": mapping(manifest["toolchain"], "toolchain"),
        "shared_object": {
            "path": str(so),
            "base_sha256": base_sha256,
            "sha256": pinned(so_sha256, "built extension sha256"),
        },
        "series_sha256": None,
        "patches": [],
        "files": {},
    }
    if variant == "patched":
        _ = series(manifest)
        for name, (pre, post) in check_files(manifest).items():
            earlier = files.get(name)
            if not (earlier is None or earlier["post"] == pre):
                fail(f"extension pre-image of {name} is not the engine post-image")
            files[name] = {
                "pre": pre if earlier is None else earlier["pre"],
                "post": post,
            }
        record["series_sha256"] = manifest["series_sha256"]
        record["patches"] = manifest["patches"]
        record["files"] = manifest["files"]
    document = {**engine, "schema": 2, "files": files, "extension": record}
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def record(variant: str, out: Path) -> None:
    """Replace the installed engine manifest after rehashing everything it names."""
    extension = mapping(load()["extension"], "extension")
    so = Path(text_value(extension["path"], "extension path"))
    document = compose(variant, digest(so))
    if out.read_bytes() != (EXL3_DIR / "exl3-patches.json").read_bytes():
        fail("installed engine manifest is not the patches/exl3 manifest")
    parsed = mapping(json.loads(document), "document")
    root = Path(text_value(mapping(parsed["engine"], "engine")["root"], "root"))
    for name, value in mapping(parsed["files"], "files").items():
        if digest(root / safe_relative(name)) != mapping(value, name)["post"]:
            fail(f"installed engine file differs from its post-image: {name}")
    check_acceptor(parsed["acceptor"])
    out.unlink()
    _ = out.write_bytes(document)
    out.chmod(0o444)
    _ = sys.stdout.write(
        f"Recorded {variant} extension {digest(so)}; "
        f"manifest sha256 {hashlib.sha256(document).hexdigest()}\n"
    )


def main(argv: list[str]) -> None:
    """Dispatch one build-time step, or re-pin the manifest on the host."""
    if len(argv) == PIN_ARGC and argv[1] == "pin":
        pin(Path(argv[2]))
        return
    if len(argv) != STEP_ARGC or argv[1] not in {"prepare", "compose", "record"}:
        fail(
            "usage: ext.py <prepare|compose|record> <rebuilt|patched> <path>"
            " | ext.py pin <pristine-engine-root>"
        )
    command, variant, path = argv[1], argv[2], Path(argv[3])
    if variant not in VARIANTS:
        fail(f"unknown extension variant {variant!r}")
    if command == "prepare":
        prepare(variant, path)
    elif command == "compose":
        _ = sys.stdout.buffer.write(compose(variant, digest(path)))
    else:
        record(variant, path)


if __name__ == "__main__":
    main(sys.argv)
