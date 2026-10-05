# Copyright (c) 2026 Gil Rodrigues
"""Apply the pinned EXL3 engine patch series to the installed package, fail closed.

Run once at image build time by the candidate stage of Dockerfile.exl3:

    python -I -B apply.py <patch-dir> <out-manifest>

Every input is checked against <patch-dir>/exl3-patches.json before anything is
written: the installed package version and root, the series file, each patch, and
the pre-image of every touched file (absent for files a patch creates). Patches are
applied strictly: each hunk must match its recorded position and context exactly;
there is no fuzz, offset search, or partial application. Afterwards every touched
file must equal its recorded post-image and every pinned acceptance-artifact file
its recorded hash; the manifest is then copied byte-for-byte to <out-manifest>.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?\Z")
SERIES_LINE = re.compile(r"([0-9a-f]{64})  ([0-9A-Za-z._-]+\.patch)\Z")
MANIFEST_KEYS = {"schema", "engine", "series_sha256", "patches", "files", "acceptor"}
ENGINE_KEYS = {"package", "version", "revision", "root"}
ARGC = 3

type Hunk = tuple[int, int, list[str]]
type FilePatch = tuple[str, bool, list[Hunk]]


def fail(message: str) -> NoReturn:
    """Abort the build.

    Raises:
        ValueError: Always.

    """
    raise ValueError(message)


def digest(path: Path) -> str:
    """Hash one regular file.

    Returns:
        The lowercase hex SHA-256 of the file bytes.

    """
    if not (path.is_file() and not path.is_symlink()):
        fail(f"{path} is not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(value: object, label: str) -> str:
    """Require a lowercase hex SHA-256 string.

    Returns:
        The validated digest.

    """
    if not isinstance(value, str) or HEX64.match(value) is None:
        fail(f"{label} is not a sha256")
    return value


def text_value(value: object, label: str) -> str:
    """Require a nonempty string.

    Returns:
        The validated string.

    """
    if not isinstance(value, str) or not value:
        fail(f"{label} is not a nonempty string")
    return value


def safe_relative(value: str) -> str:
    """Require a relative path that stays below its root.

    Returns:
        The validated relative path.

    """
    parts = value.split("/")
    if not (
        bool(value)
        and not value.startswith("/")
        and ".." not in parts
        and "" not in parts
    ):
        fail(f"unsafe relative path {value!r}")
    return value


def parse_patch(text: str) -> list[FilePatch]:
    """Split a unified diff into files and hunks without interpreting context.

    Returns:
        (relative path, creates file, hunks) per file, each hunk being
        (old start, old count, body lines).

    """
    if not text.endswith("\n"):
        fail("patch does not end with a newline")
    lines = text.splitlines(keepends=True)
    files: list[FilePatch] = []
    index = 0
    while index < len(lines):
        old = lines[index].rstrip("\n")
        if not old.startswith("--- "):
            fail(f"expected a file header at patch line {index + 1}")
        if index + 1 >= len(lines):
            fail("patch ends inside a file header")
        new = lines[index + 1].rstrip("\n")
        if not new.startswith("+++ b/"):
            fail(f"expected +++ b/ at patch line {index + 2}")
        relative = safe_relative(new.removeprefix("+++ b/"))
        creates = old == "--- /dev/null"
        if not (creates or old == f"--- a/{relative}"):
            fail(f"mismatched headers: {relative}")
        index += 2
        hunks: list[Hunk] = []
        while index < len(lines) and lines[index].startswith("@@"):
            hunk, index = parse_hunk(lines, index, relative)
            hunks.append(hunk)
        if not hunks:
            fail(f"file {relative} has no hunks")
        files.append((relative, creates, hunks))
    return files


def parse_hunk(lines: list[str], index: int, relative: str) -> tuple[Hunk, int]:
    """Parse one hunk starting at its header line.

    Args:
        lines: All patch lines, newlines kept.
        index: Position of the hunk header.
        relative: Path of the file being patched, for messages.

    Returns:
        The (old start, old count, body lines) hunk and the next line index.

    """
    match = HUNK.match(lines[index].rstrip("\n"))
    if match is None:
        fail(f"malformed hunk header at patch line {index + 1}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_count = int(match.group(4) or "1")
    index += 1
    body: list[str] = []
    seen_old = seen_new = 0
    while seen_old < old_count or seen_new < new_count:
        if index >= len(lines):
            fail("patch ends inside a hunk")
        line = lines[index]
        tag = line[:1]
        if tag not in {" ", "-", "+"}:
            fail(f"unsupported line at patch line {index + 1}")
        seen_old += tag != "+"
        seen_new += tag != "-"
        body.append(line)
        index += 1
    if not (seen_old == old_count and seen_new == new_count):
        fail(f"hunk line counts disagree in {relative}")
    return (old_start, old_count, body), index


def apply_file(path: Path, *, creates: bool, hunks: list[Hunk]) -> None:
    """Apply hunks at their exact recorded positions."""
    if creates:
        if path.exists():
            fail(f"{path} already exists")
        source: list[str] = []
    else:
        source = path.read_text(encoding="utf-8").splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    for old_start, old_count, body in hunks:
        start = old_start - 1 if old_count else old_start
        if start < cursor:
            fail(f"overlapping or unordered hunks in {path}")
        result.extend(source[cursor:start])
        cursor = start
        for line in body:
            tag, content = line[:1], line[1:]
            if tag == "+":
                result.append(content)
                continue
            if not (cursor < len(source) and source[cursor] == content):
                fail(f"hunk context mismatch in {path} at line {cursor + 1}")
            if tag == " ":
                result.append(content)
            cursor += 1
    result.extend(source[cursor:])
    _ = path.write_text("".join(result), encoding="utf-8")
    path.chmod(0o644)


def check_engine(engine: object) -> Path:
    """Match the installed engine package to the manifest.

    Returns:
        The engine package root.

    """
    if not isinstance(engine, dict) or set(engine) != ENGINE_KEYS:
        fail("bad engine record")
    package = text_value(engine["package"], "engine package")
    _ = text_value(engine["revision"], "engine revision")
    if importlib.metadata.version(package) != text_value(
        engine["version"], "engine version"
    ):
        fail("installed engine version differs from the manifest")
    root = Path(text_value(engine["root"], "engine root"))
    module = importlib.util.find_spec(package)
    if not (
        module is not None
        and module.origin is not None
        and Path(module.origin).parent == root
    ):
        fail("installed engine root differs from the manifest")
    return root


def check_series(patch_dir: Path, manifest: dict[str, object]) -> list[tuple[str, str]]:
    """Match the series file to the manifest's pinned patch list.

    Returns:
        (patch name, sha256) in application order.

    """
    series = patch_dir / "series"
    if digest(series) != pinned(manifest["series_sha256"], "series_sha256"):
        fail("series hash mismatch")
    entries: list[tuple[str, str]] = []
    for line in series.read_text(encoding="utf-8").splitlines():
        match = SERIES_LINE.match(line)
        if match is None:
            fail(f"malformed series line {line!r}")
        entries.append((match.group(2), match.group(1)))
    recorded = manifest["patches"]
    if not (
        isinstance(recorded, list)
        and bool(entries)
        and recorded == [{"name": name, "sha256": sha} for name, sha in entries]
    ):
        fail("series differs from the manifest")
    return entries


def check_files(manifest: dict[str, object]) -> dict[str, tuple[str | None, str]]:
    """Validate the per-file pre/post pins.

    Returns:
        relative path -> (pre sha256 or None for a created file, post sha256).

    """
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        fail("manifest names no files")
    pins: dict[str, tuple[str | None, str]] = {}
    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            fail("bad file record")
        if set(record) != {"pre", "post"}:
            fail(f"bad file record {relative}")
        pre = record["pre"]
        pins[safe_relative(relative)] = (
            None if pre is None else pinned(pre, f"{relative} pre"),
            pinned(record["post"], f"{relative} post"),
        )
    return pins


def check_acceptor(acceptor: object) -> None:
    """Verify every pinned file of the acceptance artifact."""
    if not isinstance(acceptor, dict) or set(acceptor) != {"root", "files"}:
        fail("bad acceptor record")
    root = Path(text_value(acceptor["root"], "acceptor root"))
    files = acceptor["files"]
    if not isinstance(files, dict) or not files:
        fail("acceptor pins no files")
    for relative, sha in files.items():
        if not isinstance(relative, str):
            fail("bad acceptor file name")
        if digest(root / safe_relative(relative)) != pinned(
            sha, f"acceptor {relative}"
        ):
            fail(f"acceptor artifact hash mismatch: {relative}")


def check_pre_images(root: Path, pins: dict[str, tuple[str | None, str]]) -> None:
    """Require every pinned file's pre-image, or its absence for created files.

    Args:
        root: Engine package root.
        pins: Relative path to (pre, post) hashes.

    """
    for relative, (pre, _) in pins.items():
        if pre is None:
            if (root / relative).exists():
                fail(f"{relative} exists before its patch")
        elif digest(root / relative) != pre:
            fail(f"pre-image mismatch: {relative}")


def check_post_images(root: Path, pins: dict[str, tuple[str | None, str]]) -> None:
    """Require every pinned file's post-image.

    Args:
        root: Engine package root.
        pins: Relative path to (pre, post) hashes.

    """
    for relative, (_, post) in pins.items():
        if digest(root / relative) != post:
            fail(f"post-image mismatch: {relative}")


def apply_parsed(
    root: Path,
    parsed: list[list[FilePatch]],
    pins: dict[str, tuple[str | None, str]],
) -> None:
    """Apply parsed patches in order, requiring every pinned file to be touched.

    Args:
        root: Engine package root.
        parsed: Parsed patches in series order.
        pins: Relative path to (pre, post) hashes.

    """
    touched: set[str] = set()
    for patch in parsed:
        for relative, creates, hunks in patch:
            if relative not in pins:
                fail(f"patch touches unpinned file {relative}")
            if creates != (pins[relative][0] is None and relative not in touched):
                fail(f"file creation disagrees with the manifest: {relative}")
            apply_file(root / relative, creates=creates, hunks=hunks)
            touched.add(relative)
    if touched != set(pins):
        fail("manifest pins files no patch touches")


def main(argv: list[str]) -> None:
    """Verify, apply, re-verify, and record the patch series."""
    if len(argv) != ARGC:
        fail("usage: apply.py <patch-dir> <out-manifest>")
    patch_dir = Path(argv[1])
    out = Path(argv[2])
    if out.exists():
        fail(f"{out} already exists")
    manifest_bytes = (patch_dir / "exl3-patches.json").read_bytes()
    manifest: object = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        fail("bad manifest keys")
    if manifest["schema"] != 1:
        fail("unsupported manifest schema")
    root = check_engine(manifest["engine"])
    entries = check_series(patch_dir, manifest)
    pins = check_files(manifest)

    check_pre_images(root, pins)

    parsed: list[list[FilePatch]] = []
    for name, sha in entries:
        path = patch_dir / name
        if digest(path) != sha:
            fail(f"patch hash mismatch: {name}")
        parsed.append(parse_patch(path.read_text(encoding="utf-8")))
    apply_parsed(root, parsed, pins)
    check_post_images(root, pins)
    check_acceptor(manifest["acceptor"])

    _ = out.write_bytes(manifest_bytes)
    out.chmod(0o444)
    _ = sys.stdout.write(
        f"Applied {len(entries)} EXL3 patch(es); manifest sha256 "
        f"{hashlib.sha256(manifest_bytes).hexdigest()}\n"
    )


if __name__ == "__main__":
    main(sys.argv)
