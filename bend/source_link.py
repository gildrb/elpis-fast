# Copyright (c) 2026 Gil Rodrigues
"""Shared toolchain resolution for the bend/*_diff.py source links, fail closed.

- REPO: the repository root, found from this file.
- bend(): the `bend` executable on PATH. The version must be exactly bend 2.0.35,
  as in bend/exl3_build.py. Run the drivers inside
  `nix develop --offline --no-write-lock-file`.
- locked(cmd): the command, prefixed with the host CPU lock /tmp/cpu-lock.sh if
  that file is present. The lock file must be a regular file owned by the current
  user (or root) that no other user can write. If it is not present, the command
  runs directly.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEND_VERSION = "bend 2.0.35\n"
CPU_LOCK = Path("/tmp/cpu-lock.sh")  # ruff: ignore[hardcoded-temp-file]  documented host CPU lock; owner and permissions are checked before use


class LinkError(RuntimeError):
    """A toolchain requirement of a source link is not met."""


@functools.cache
def bend() -> str:
    """Return the path of the pinned `bend` executable on PATH.

    Returns:
        The resolved `bend` path.

    Raises:
        LinkError: bend is missing or not exactly the pinned version.

    """
    found = shutil.which("bend")
    if found is None:
        msg = (
            "bend is not on PATH "
            "(run inside `nix develop --offline --no-write-lock-file`)"
        )
        raise LinkError(msg)
    version = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: bend found on PATH + 'version' to check the exact pin, no shell
        [found, "version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0 or version.stdout != BEND_VERSION:
        msg = (
            f"{found}: requires exactly {BEND_VERSION.strip()}, "
            f"got {version.stdout.strip()!r}"
        )
        raise LinkError(msg)
    return found


@functools.cache
def _lock_prefix() -> tuple[str, ...]:
    try:
        info = CPU_LOCK.lstat()
    except FileNotFoundError:
        return ()
    if not stat.S_ISREG(info.st_mode):
        msg = f"{CPU_LOCK} is not a regular file"
        raise LinkError(msg)
    if info.st_uid not in {0, os.getuid()}:
        msg = f"{CPU_LOCK} is owned by another user"
        raise LinkError(msg)
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        msg = f"{CPU_LOCK} is writable by other users"
        raise LinkError(msg)
    return (str(CPU_LOCK),)


def locked(cmd: list[str]) -> list[str]:
    """Return cmd, prefixed with the host CPU lock if it is present.

    Returns:
        The argv to run.

    """
    return [*_lock_prefix(), *cmd]
