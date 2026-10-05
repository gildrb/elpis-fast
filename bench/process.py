# Copyright (c) 2026 Gil Rodrigues
"""The audited process launchers of ``bench/`` and ``eval/measure.py``.

Every external program those modules run goes through :func:`start` (Popen) or
:func:`run` (``subprocess.run`` with a required ``check``, text mode). Both
accept only an argv list of strings whose first element is an absolute
executable, resolved by :func:`resolve` (``shutil.which`` or an explicit path),
and never use a shell.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

Stream = int | IO[bytes] | None


@dataclass(frozen=True)
class Launch:
    """Child process setup besides argv; the defaults inherit everything."""

    cwd: Path | str | None = None
    """The child's working directory."""
    env: Mapping[str, str] | None = None
    """The child's whole environment, or ``None`` to inherit."""
    stdin: Stream = None
    """The child's standard input."""
    stdout: Stream = None
    """The child's standard output."""
    stderr: Stream = None
    """The child's standard error."""
    start_new_session: bool = False
    """Start the child in a new session."""


def resolve(
    name: str, *, cwd: Path | str | None = None, path: str | None = None
) -> str:
    """Resolve a program the way ``subprocess`` would, to an absolute path.

    Args:
        name: A bare program name (looked up on ``PATH``) or a path.
        cwd: The child's working directory; a relative path is taken from it.
        path: The ``PATH`` to search instead of the current environment's.

    Returns:
        The absolute executable path.

    Raises:
        FileNotFoundError: If a bare name is not an executable on ``PATH``.

    """
    if os.sep in name:
        explicit = Path(name)
        if explicit.is_absolute():
            return name
        base = Path.cwd() if cwd is None else Path(cwd).absolute()
        return str(base / explicit)
    found = shutil.which(name, path=path)
    if found is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    return str(Path(found).absolute())


def _validated(argv: list[str]) -> list[str]:
    """Require a non-empty list of strings led by an absolute executable.

    Returns:
        The same argv.

    Raises:
        TypeError: If argv is not a non-empty list of strings.
        ValueError: If ``argv[0]`` is not an absolute path.

    """
    if not (
        isinstance(argv, list)
        and argv
        and all(isinstance(argument, str) for argument in argv)
    ):
        msg = "argv must be a non-empty list of strings"
        raise TypeError(msg)
    if not Path(argv[0]).is_absolute():
        msg = f"argv[0] must be an absolute executable path: {argv[0]!r}"
        raise ValueError(msg)
    return argv


def start(argv: list[str], launch: Launch | None = None) -> subprocess.Popen[bytes]:
    """Start one validated argv without a shell.

    Args:
        argv: The program and its arguments; ``argv[0]`` is absolute.
        launch: The child setup; ``None`` inherits everything.

    Returns:
        The running process, in binary mode.

    """
    setup = Launch() if launch is None else launch
    return subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  argv from _validated(): list of str led by an absolute executable from resolve(), shell=False
        _validated(argv),
        shell=False,
        cwd=setup.cwd,
        env=setup.env,
        stdin=setup.stdin,
        stdout=setup.stdout,
        stderr=setup.stderr,
        start_new_session=setup.start_new_session,
    )


def run(
    argv: list[str],
    *,
    check: bool,
    timeout: float | None = None,
    capture_output: bool = False,
    launch: Launch | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one validated argv to completion without a shell, in text mode.

    Args:
        argv: The program and its arguments; ``argv[0]`` is absolute.
        check: Raise on a nonzero exit status.
        timeout: Kill the child and raise after this many seconds.
        capture_output: Capture standard output and error as text.
        launch: The child setup; ``None`` inherits everything.

    Returns:
        The finished process.

    """
    setup = Launch() if launch is None else launch
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv from _validated(): list of str led by an absolute executable from resolve(), shell=False
        _validated(argv),
        shell=False,
        check=check,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
        cwd=setup.cwd,
        env=setup.env,
        stdin=setup.stdin,
        stdout=setup.stdout,
        stderr=setup.stderr,
        start_new_session=setup.start_new_session,
    )
