#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Read-only, CPU-only authentication of the approved EXL3 target/draft pair.

The adjacent exl3-manifest.json is the trust root. Mount it and both model
directories read-only. This checks bytes, not runtime quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import NoReturn

type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None
METADATA_LIMIT = 16 * 1024 * 1024
CHUNK_SIZE = 8 * 1024 * 1024
EXL3_TARGET_NAMES = frozenset({
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "quantization_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
})
EXL3_DRAFT_NAMES = frozenset({
    "config.json",
    "quantization_config.json",
    "model.safetensors",
})


def fail(message: str) -> NoReturn:
    """Reject the verified input.

    Raises:
        ValueError: Always, with the message.

    """
    raise ValueError(message)


def regular(path: Path) -> None:
    """Require a regular file without following a symlink."""
    if not stat.S_ISREG(path.lstat().st_mode):
        fail(f"Not a regular file: {path}")


def directory(path: Path) -> None:
    """Require an absolute directory with no symlink components."""
    if not path.is_absolute():
        fail(f"Expected absolute directory: {path}")
    if ".." in path.parts:
        fail(f"Parent traversal forbidden: {path}")
    for component in (*reversed(path.parents), path):
        if not stat.S_ISDIR(component.lstat().st_mode):
            fail(f"Not a real directory (symlinks forbidden): {component}")


def read_bytes(path: Path, limit: int = METADATA_LIMIT) -> bytes:
    """Read bounded metadata from a regular file.

    Returns:
        The file contents.

    """
    regular(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            fail(f"Not regular: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        fail(f"Metadata too large: {path}")
    return data


def digest(path: Path) -> str:
    """Hash a regular file and reject changes during verification.

    Returns:
        The lowercase SHA256 hex digest.

    """
    regular(path)
    value = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            fail(f"Not regular: {path}")
        while True:
            block = stream.read(CHUNK_SIZE)
            if len(block) == 0:
                break
            value.update(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    for snapshot in (after, current):
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            snapshot.st_dev,
            snapshot.st_ino,
            snapshot.st_size,
            snapshot.st_mtime_ns,
            snapshot.st_ctime_ns,
        ):
            fail(f"File changed during verification: {path}")
    return value.hexdigest()


def json_value(value: object) -> JSON:
    """Validate and rebuild a finite JSON value.

    Returns:
        The rebuilt JSON value.

    Raises:
        TypeError: An object key is not a string.
        ValueError: The value is not JSON.

    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            fail("Nonfinite JSON number")
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JSON] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                message = "Non-string JSON key"
                raise TypeError(message)
            result[key] = json_value(item)
        return result
    message = "Unsupported JSON value"
    raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, JSON]:
    """Build a JSON object while rejecting duplicate keys.

    Returns:
        The validated object.

    """
    result: dict[str, JSON] = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate JSON key: {key}")
        result[key] = json_value(value)
    return result


def obj(value: JSON) -> dict[str, JSON]:
    """Require a JSON object.

    Returns:
        The object.

    Raises:
        TypeError: The value is not an object.

    """
    if not isinstance(value, dict):
        message = "Expected JSON object"
        raise TypeError(message)
    return value


def document(data: bytes) -> dict[str, JSON]:
    """Decode a strict JSON object from UTF-8 bytes.

    Returns:
        The decoded object.

    """
    value: object = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    return obj(json_value(value))


def integer(value: object) -> int:
    """Require an integer other than a boolean.

    Returns:
        The integer.

    Raises:
        TypeError: The value is not an integer or is a boolean.

    """
    if not isinstance(value, int) or isinstance(value, bool):
        message = "Expected integer"
        raise TypeError(message)
    return value


def filename(value: object) -> str:
    """Require a safe inventory filename.

    Returns:
        The filename.

    Raises:
        ValueError: The value is not a safe filename.

    """
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value) is None
    ):
        message = "Invalid inventory filename"
        raise ValueError(message)
    return value


def sha256(value: object) -> str:
    """Require a lowercase SHA256 digest.

    Returns:
        The digest.

    Raises:
        ValueError: The value is not a lowercase SHA256 digest.

    """
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        message = "Invalid SHA256"
        raise ValueError(message)
    return value


def verify(target: Path, draft: Path) -> None:
    """Authenticate every EXL3 runtime file against the manifest."""
    preparation = Path(__file__).absolute().parent
    manifest = document(read_bytes(preparation / "exl3-manifest.json"))
    if integer(manifest["schema_version"]) != 1:
        fail("Unsupported EXL3 manifest")
    for role, path, names in (
        ("target", target, EXL3_TARGET_NAMES),
        ("draft", draft, EXL3_DRAFT_NAMES),
    ):
        hashes = {
            filename(name): sha256(value) for name, value in obj(manifest[role]).items()
        }
        if frozenset(hashes) != names:
            fail(f"Unexpected EXL3 {role} inventory")
        directory(path)
        entries = {entry.name: entry for entry in path.iterdir()}
        if not names <= set(entries):
            fail(f"Missing EXL3 {role} files")
        extras = set(entries) - names
        if not extras <= {
            "README.md",
            "LICENSE",
            ".gitattributes",
            "crc32.txt",
            ".cache",
        }:
            fail(f"Unexpected EXL3 {role} files: {sorted(extras)}")
        for name, entry in entries.items():
            if name == ".cache":
                directory(entry)
            else:
                regular(entry)
        for name, checksum in hashes.items():
            if digest(path / name) != checksum:
                fail(f"EXL3 {role} authentication failed: {name}")


def main() -> int:
    """Run the command and report validation failures.

    Returns:
        Zero when authenticated, one on any validation failure.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument(
        "--representation",
        choices=("exl3",),
        required=True,
        help="Authenticate only this target representation; never detect or fall back",
    )
    args = parser.parse_args()
    try:
        verify(args.target, args.draft)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        sys.stderr.write(f"Model verification failed: {error}\n")
        sys.stderr.flush()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
