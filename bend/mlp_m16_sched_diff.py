#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Differential check of the ext 8201 fused-MLP schedule header.

Runs the Bend emitter bend/MLP_M16_SCHED_TABLE.bend (Nat model
bend/mlp_m16_sched.bend) and the independent Python reference gen_table.py,
compares their stdout byte for byte and prints the sha256 of the header.
Exit status 0 iff IDENTICAL. This is differential evidence for the one
instance the kernel bakes (G = 164, PF = 8, gate/up 5120 x 17408, down
17408 x 5120); the laws are in bend/mlp_m16_sched_laws.bend.

Usage: python3 -B bend/mlp_m16_sched_diff.py [GEN_TABLE_PY] [--out HEADER]
  GEN_TABLE_PY: the independent Python reference (default: the tracked
  bend/gen/mlp_m16_sched_ref.py).
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

REPO = source_link.REPO
TABLE = "bend/MLP_M16_SCHED_TABLE.bend"
DEFAULT_REFERENCE = REPO / "bend/gen/mlp_m16_sched_ref.py"


def fail(msg: str) -> NoReturn:
    """Stop with a prefixed error message.

    Args:
        msg: The reason.

    Raises:
        SystemExit: Always.

    """
    text = f"mlp_m16_sched_diff: {msg}"
    raise SystemExit(text)


def run(cmd: list[str]) -> bytes:
    """Run cmd in the repository and return its stdout; stop if it fails.

    Args:
        cmd: The argv to run.

    Returns:
        The captured stdout.

    """
    proc = source_link.run(
        cmd, cwd=REPO, capture_output=True, timeout=1800, check=False, cpu_heavy=False
    )
    if proc.returncode != 0:
        fail(
            f"{' '.join(cmd)} exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def main(argv: list[str]) -> int:
    """Compare the Bend emitter's output with the Python reference.

    Args:
        argv: The command-line arguments without the program name.

    Returns:
        The exit status: 0 iff identical.

    """
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            fail("--out needs a path")
        out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) > 1:
        fail("usage: mlp_m16_sched_diff.py [GEN_TABLE_PY] [--out HEADER]")
    reference = Path(argv[0]) if argv else DEFAULT_REFERENCE
    if not reference.is_file():
        fail(f"reference {reference} is absent")
    bend = run([source_link.bend(), TABLE])
    python = run([sys.executable, "-B", str(reference)])
    if not bend:
        fail("the Bend emitter printed nothing")
    if bend != python:
        diff = difflib.unified_diff(
            python.decode().splitlines(keepends=True),
            bend.decode().splitlines(keepends=True),
            "gen_table.py",
            TABLE,
        )
        sys.stdout.write("".join(list(diff)[:60]))
        sys.stdout.write("DIFFERENT\n")
        return 1
    if out is not None:
        out.write_bytes(bend)
    sys.stdout.write(
        f"IDENTICAL {len(bend)} bytes sha256 {hashlib.sha256(bend).hexdigest()}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
