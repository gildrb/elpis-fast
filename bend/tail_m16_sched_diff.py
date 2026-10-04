#!/usr/bin/env python3
"""
Differential check of the ext 8202 layer-tail schedule header: runs the Bend emitter
bend/TAIL_M16_SCHED_TABLE.bend (Nat model bend/tail_m16_sched.bend) and compares its stdout byte for
byte with the header exllamav3_ext/quant/exl3_tail_m16_sched.h that the committed extension patch
patches/exl3-ext/8202-layer-tail-on2102.patch adds (the header the kernel is built with). Prints the
sha256 of the patch and of the header. Exit status 0 iff IDENTICAL. This is differential evidence
for the one instance the kernel bakes (G = 164, PF = 8, o_proj 6144 x 5120, 8201's gate/up
5120 x 17408 for ring2); the laws are in bend/tail_m16_sched_laws.bend.

Usage: python3 -B bend/tail_m16_sched_diff.py [--patch PATCH] [--out HEADER]
"""

from __future__ import annotations

import difflib
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/TAIL_M16_SCHED_TABLE.bend"
PATCH = "patches/exl3-ext/8202-layer-tail-on2102.patch"
HEADER = "exllamav3_ext/quant/exl3_tail_m16_sched.h"


def fail(msg: str) -> NoReturn:
    raise SystemExit(f"tail_m16_sched_diff: {msg}")


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=1800, check=False)
    if proc.returncode != 0:
        fail(f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def patch_header(patch: bytes) -> bytes:
    """The new-file body of HEADER in the unified diff, exactly as the patch creates it."""
    lines = patch.split(b"\n")
    target = f"+++ b/{HEADER}".encode()
    starts = [i for i, line in enumerate(lines) if line == target]
    if len(starts) != 1:
        fail(f"expected one '+++ b/{HEADER}' in the patch, found {len(starts)}")
    i = starts[0]
    if i == 0 or lines[i - 1] != b"--- /dev/null":
        fail(f"{HEADER} is not a new file in the patch")
    hunk = re.fullmatch(rb"@@ -0,0 \+1,(\d+) @@", lines[i + 1])
    if hunk is None:
        fail(f"unexpected hunk header for {HEADER}: {lines[i + 1]!r}")
    count = int(hunk.group(1))
    body = lines[i + 2 : i + 2 + count]
    if len(body) != count or any(not line.startswith(b"+") for line in body):
        fail(f"the {HEADER} hunk does not hold {count} added lines")
    after = lines[i + 2 + count] if i + 2 + count < len(lines) else b""
    newline = b"" if after.startswith(b"\\ No newline at end of file") else b"\n"
    return b"\n".join(line[1:] for line in body) + newline


def take_opt(argv: list[str], key: str) -> tuple[list[str], Path | None]:
    if key not in argv:
        return argv, None
    i = argv.index(key)
    if i + 1 >= len(argv):
        fail(f"{key} needs a path")
    return argv[:i] + argv[i + 2 :], Path(argv[i + 1])


def main(argv: list[str]) -> int:
    argv, out = take_opt(argv, "--out")
    argv, patch_arg = take_opt(argv, "--patch")
    if argv:
        fail("usage: tail_m16_sched_diff.py [--patch PATCH] [--out HEADER]")
    patch_path = patch_arg if patch_arg is not None else REPO / PATCH
    if not patch_path.is_file():
        fail(f"patch {patch_path} is absent")
    patch = patch_path.read_bytes()
    expected = patch_header(patch)
    bend = run(source_link.locked([source_link.bend(), TABLE]))
    if not bend:
        fail("the Bend emitter printed nothing")
    print(f"patch {patch_path} sha256 {hashlib.sha256(patch).hexdigest()}")
    if bend != expected:
        diff = difflib.unified_diff(
            expected.decode().splitlines(True), bend.decode().splitlines(True), f"{PATCH}:{HEADER}", TABLE
        )
        _ = sys.stdout.write("".join(list(diff)[:60]))
        print("DIFFERENT")
        return 1
    if out is not None:
        _ = out.write_bytes(bend)
    print(f"IDENTICAL {len(bend)} bytes sha256 {hashlib.sha256(bend).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
