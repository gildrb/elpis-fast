# Copyright (c) 2026 Gil Rodrigues
"""Shared toolchain resolution and process launch for the bend/*.py links.

Fail closed:
- REPO: the repository root, found from this file.
- bend(): the `bend` executable on PATH. The version must be exactly bend 2.0.35,
  as in bend/exl3_build.py. Run the drivers inside
  `nix develop --offline --no-write-lock-file`.
- run(argv, check=..., ...): the one place the links start a process. argv[0]
  must be an absolute executable or a bare name found on PATH; it runs resolved,
  without a shell. A cpu_heavy run (the default) takes the host CPU lock first:
  if $XDG_RUNTIME_DIR/elpis-gpu.lock is a regular file owned by the current user
  that neither group nor others can write, run() waits while
  $XDG_RUNTIME_DIR/elpis-gpu.pending exists (a GPU timing window is queued),
  holds a shared flock on the lock file for the child's lifetime, and lowers
  this process (and so every later child) to nice 19 once. Without such a lock
  file the child runs directly.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack, overload

if TYPE_CHECKING:
    from collections.abc import Generator

REPO = Path(__file__).resolve().parent.parent
BEND_VERSION = "bend 2.0.35\n"
LOCK_NAME = "elpis-gpu.lock"
PENDING_NAME = "elpis-gpu.pending"
PENDING_POLL = 5.0
NICE = 19


class LinkError(RuntimeError):
    """A toolchain requirement of a source link is not met."""


class RunOptions(TypedDict, total=False):
    """Optional arguments of run(), with subprocess.run's meaning."""

    cwd: Path | str
    capture_output: bool
    timeout: float
    cpu_heavy: bool


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
    version = run([found, "version"], capture_output=True, text=True, check=False)
    if version.returncode != 0 or version.stdout != BEND_VERSION:
        msg = (
            f"{found}: requires exactly {BEND_VERSION.strip()}, "
            f"got {version.stdout.strip()!r}"
        )
        raise LinkError(msg)
    return found


def _resolve(argv: list[str]) -> list[str]:
    """Return argv with argv[0] resolved to an absolute executable.

    Returns:
        The argv to execute.

    Raises:
        LinkError: argv is not a non-empty list, or argv[0] is a relative path
            or not an executable file.

    """
    if not isinstance(argv, list) or not argv:
        msg = f"run: argv must be a non-empty list, got {argv!r}"
        raise LinkError(msg)
    exe = argv[0]
    if not Path(exe).is_absolute() and os.sep in exe:
        msg = f"run: {exe!r} is a relative path"
        raise LinkError(msg)
    found = shutil.which(exe)
    if found is None:
        msg = f"run: {exe!r} is not an executable file or on PATH"
        raise LinkError(msg)
    return [str(Path(found).absolute()), *argv[1:]]


def _open_lock(runtime: Path) -> int | None:
    """Open the CPU lock file in runtime if it is safe to use.

    Returns:
        A read-only descriptor of the lock file, or None when the file is
        missing, not a regular file, owned by another user, or writable by group
        or others.

    """
    try:
        fd = os.open(
            runtime / LOCK_NAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
    except OSError:
        return None
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        os.close(fd)
        return None
    return fd


@contextlib.contextmanager
def _cpu_slot() -> Generator[None]:
    """Hold the host CPU lock (shared) while the caller runs a CPU-heavy child.

    Yields:
        Once the lock is held, or at once when there is no usable lock file.

    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    fd = _open_lock(Path(runtime)) if runtime else None
    if runtime is None or fd is None:
        yield
        return
    try:
        pending = Path(runtime) / PENDING_NAME
        while pending.exists():
            time.sleep(PENDING_POLL)
        fcntl.flock(fd, fcntl.LOCK_SH)
        if os.nice(0) < NICE:
            os.nice(NICE - os.nice(0))
        yield
    finally:
        os.close(fd)


@overload
def run(
    argv: list[str],
    *,
    check: bool,
    text: Literal[True],
    **options: Unpack[RunOptions],
) -> subprocess.CompletedProcess[str]: ...


@overload
def run(
    argv: list[str],
    *,
    check: bool,
    text: Literal[False] = False,
    **options: Unpack[RunOptions],
) -> subprocess.CompletedProcess[bytes]: ...


def run(
    argv: list[str], *, check: bool, text: bool = False, **options: Unpack[RunOptions]
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run argv without a shell, under the host CPU lock when cpu_heavy.

    Args:
        argv: the command; argv[0] is an absolute executable or a bare name on
            PATH.
        check: raise CalledProcessError on a non-zero exit, as subprocess.run.
        text: decode captured output, as subprocess.run(text=True).
        **options: cwd, capture_output and timeout as subprocess.run, and
            cpu_heavy (default True) to take the host CPU lock.

    Returns:
        The completed process.

    """
    resolved = _resolve(argv)
    slot = _cpu_slot() if options.get("cpu_heavy", True) else contextlib.nullcontext()
    with slot:
        if text:
            return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  text mode; argv: list, argv[0] resolved to an absolute executable by _resolve, no shell, caller's check
                resolved,
                check=check,
                text=True,
                cwd=options.get("cwd"),
                capture_output=options.get("capture_output", False),
                timeout=options.get("timeout"),
            )
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  bytes mode; argv: list, argv[0] resolved to an absolute executable by _resolve, no shell, caller's check
            resolved,
            check=check,
            cwd=options.get("cwd"),
            capture_output=options.get("capture_output", False),
            timeout=options.get("timeout"),
        )
