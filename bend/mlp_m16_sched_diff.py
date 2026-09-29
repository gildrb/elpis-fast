#!/usr/bin/env python3
"""
Differential check of the ext 8201 fused-MLP schedule header: runs the Bend emitter
bend/MLP_M16_SCHED_TABLE.bend (Nat model bend/mlp_m16_sched.bend) and the independent Python
reference gen_table.py, compares their stdout byte for byte and prints the sha256 of the header.
Exit status 0 iff IDENTICAL. This is differential evidence for the one instance the kernel bakes
(G = 164, PF = 8, gate/up 5120 x 17408, down 17408 x 5120); the laws are in
bend/mlp_m16_sched_laws.bend.

Usage: python3 -B bend/mlp_m16_sched_diff.py [GEN_TABLE_PY] [--out HEADER]
"""

from __future__ import annotations

import difflib
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEND = "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend"
TABLE = "bend/MLP_M16_SCHED_TABLE.bend"
DEFAULT_REFERENCE = "/tmp/kernel-work/PersistMLP/gen_table.py"


def fail(msg: str) -> None:
    raise SystemExit(f"mlp_m16_sched_diff: {msg}")


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        fail(f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def main(argv: list[str]) -> int:
    out = None
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            fail("--out needs a path")
        out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2 :]
    if len(argv) > 1:
        fail("usage: mlp_m16_sched_diff.py [GEN_TABLE_PY] [--out HEADER]")
    reference = Path(argv[0] if argv else DEFAULT_REFERENCE)
    if not reference.is_file():
        fail(f"reference {reference} is absent")
    bend = run([BEND, TABLE])
    python = run([sys.executable, "-B", str(reference)])
    if not bend:
        fail("the Bend emitter printed nothing")
    if bend != python:
        diff = difflib.unified_diff(
            python.decode().splitlines(True), bend.decode().splitlines(True), "gen_table.py", TABLE
        )
        sys.stdout.write("".join(list(diff)[:60]))
        print("DIFFERENT")
        return 1
    if out is not None:
        out.write_bytes(bend)
    print(f"IDENTICAL {len(bend)} bytes sha256 {hashlib.sha256(bend).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
