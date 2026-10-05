#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Differential check of ext 8205's slot-weighted 8201 / 8202 schedule tables.

Runs the Bend emitter bend/M16_WSCHED_TABLE.bend (Nat model bend/m16_wsched.bend
over the shared partition bend/gemm_m16_wpart.bend) and compares every printed
table, byte for byte as a list of uint16 values, with
  1. the independent Python reference bend/gen/m16_wsched_ref.py (flat(), over
     the bend/gen/wpart.py formula), and
  2. with --tree TREE: the host builder the extension ships,
     exllamav3_ext/quant/exl3_m16_wsched.h of TREE, compiled into a small driver
     (c++ from PATH; the dev shell gives clang), and the uniform-weight tables'
     8201 / tail sections against the Bend-baked headers exl3_mlp_m16_sched.h /
     exl3_tail_m16_sched.h of TREE (the extension checks the same at runtime and
     fails closed).
Exit status 0 iff every comparison is IDENTICAL.

Usage: python3 -B bend/m16_wsched_diff.py [--reference GEN_SCHED_PY] [--tree TREE]
  GEN_SCHED_PY: the independent Python reference (default: the tracked
    bend/gen/m16_wsched_ref.py).
  TREE: OUT/patched of bend/engine_trees.py.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

REPO = source_link.REPO
TABLE = "bend/M16_WSCHED_TABLE.bend"
DEFAULT_REFERENCE = REPO / "bend/gen/m16_wsched_ref.py"
G, BF, NP, PFLD, TBF, HDR = 164, 11, 34, 5, 4, 8
NTABLES = 4  # tables M16_WSCHED_TABLE.bend prints
UNIFORM = [1, 1, 1, 1, 1, 1]

DRIVER = (
    r"""
#include "exl3_m16_wsched.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv)
{
    int kind = atoi(argv[1]); int w[6];"""
    r""" for (int i = 0; i < 6; ++i) w[i] = atoi(argv[2 + i]);
    std::vector<uint16_t> t; std::string err;
    if (!exl3_wsched::build(kind, w, t, err))"""
    r""" { printf("INVALID %s\n", err.c_str()); return 0; }
    for (size_t i = 0; i < t.size(); ++i) printf(i ? ",%d" : "%d", t[i]);
    printf("\n");
}
"""
)


def fail(msg: str) -> NoReturn:
    """Exit with the FAIL message.

    Raises:
        SystemExit: always.

    """
    text = f"m16_wsched_diff: FAIL: {msg}"
    raise SystemExit(text)


def bend_tables() -> list[tuple[int, list[int], list[int]]]:
    """Run the Bend emitter and parse its tables.

    Returns:
        (kind, weights, values) per printed table.

    """
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        [source_link.bend(), TABLE],
        cwd=REPO,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:500]
        fail(f"{TABLE} exited {proc.returncode}: {err}")
    out = []
    for line in proc.stdout.decode().splitlines():
        if not line.strip():
            continue
        m = re.fullmatch(r"kind (\d) weights ([\d,]+):([\d,]+)", line)
        if not m:
            fail(f"unparsable line: {line[:80]}")
        out.append((
            int(m.group(1)),
            [int(x) for x in m.group(2).split(",")],
            [int(x) for x in m.group(3).split(",")],
        ))
    if len(out) != NTABLES:
        fail(f"expected 4 tables, got {len(out)}")
    return out


def load_reference(path: str) -> ModuleType:
    """Load the Python reference generator from path.

    Returns:
        The loaded module.

    """
    spec = importlib.util.spec_from_file_location("gen_sched", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_driver(tree: Path) -> list[str]:
    """Compile the driver around TREE's exl3_m16_wsched.h.

    Returns:
        The driver argv prefix.

    """
    quant = tree / "exllamav3_ext" / "quant"
    work = Path(tempfile.mkdtemp(prefix="m16_wsched_diff_"))
    (work / "drv.cpp").write_text(DRIVER)
    shutil.copy(quant / "exl3_m16_wsched.h", work / "exl3_m16_wsched.h")
    cxx = shutil.which("c++")
    if cxx is None:
        fail(
            "c++ is not on PATH "
            "(run inside `nix develop --offline --no-write-lock-file`)"
        )
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: c++ from PATH compiling the generated driver in a private workdir, no shell
        [cxx, "-O2", "-std=c++17", "-o", str(work / "drv"), str(work / "drv.cpp")],
        check=True,
    )
    return [str(work / "drv")]


def header_table(text: str, name: str) -> list[int]:
    """Return the values of the C array name in text.

    Returns:
        The array's values.

    """
    body = re.search(
        r"static const unsigned short " + name + r"\[[^\]]*\] = \{(.*?)\};",
        text,
        re.DOTALL,
    )
    if not body:
        fail(f"{name} not found")
    return [int(x) for x in re.findall(r"\d+", body.group(1))]


def check_driver(
    drv: list[str], tree: Path, kind: int, w: list[int], vals: list[int]
) -> None:
    """Compare one table with the tree's builder."""
    out = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: driver just built in a private workdir + integer arguments, no shell
        [*drv, str(kind), *(str(x) for x in w)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    got = [int(x) for x in out.split(",")] if not out.startswith("INVALID") else None
    if got != vals:
        fail(
            f"kind {kind} weights {w}: Bend != the tree's exl3_m16_wsched.h builder "
            f"({out[:60]})"
        )
    sys.stdout.write(
        f"kind {kind} weights {w}: Bend == exl3_m16_wsched.h builder of {tree} "
        "IDENTICAL\n"
    )


def check_headers(tree: Path, tables: list[tuple[int, list[int], list[int]]]) -> None:
    """Compare the uniform tables' sections with the tree's baked headers."""
    quant = tree / "exllamav3_ext" / "quant"
    mh = (quant / "exl3_mlp_m16_sched.h").read_text()
    th = (quant / "exl3_tail_m16_sched.h").read_text()
    for kind, w, vals in tables:
        if w != UNIFORM:
            continue
        nwait = vals[3]
        blk = vals[HDR : HDR + G * BF]
        pair = vals[HDR + G * BF : HDR + G * BF + NP * PFLD]
        wait = vals[HDR + G * BF + NP * PFLD : HDR + G * BF + NP * PFLD + nwait]
        ok = (
            blk == header_table(mh, "exl3_mlp_sched_block")
            and pair == header_table(mh, "exl3_mlp_sched_pair")
            and wait == header_table(mh, "exl3_mlp_sched_wait")
        )
        if kind == 1:
            grp_end = HDR + G * BF + NP * PFLD + nwait + 4 * NP + 2 * 10
            ok = ok and vals[grp_end : grp_end + G * TBF] == header_table(
                th, "exl3_tail_sched_block"
            )
        if not ok:
            fail(
                f"kind {kind} uniform: the 8201 / tail sections != "
                "the tree's baked Bend headers"
            )
        sections = " and tail" if kind else ""
        sys.stdout.write(
            f"kind {kind} uniform: 8201{sections} sections == baked headers of {tree} "
            "IDENTICAL\n"
        )


def main(argv: list[str]) -> int:
    """Run every comparison.

    Returns:
        The exit status (0: all IDENTICAL).

    """
    ref_path = (
        argv[argv.index("--reference") + 1]
        if "--reference" in argv
        else str(DEFAULT_REFERENCE)
    )
    tree = Path(argv[argv.index("--tree") + 1]) if "--tree" in argv else None
    ref = load_reference(ref_path)
    tables = bend_tables()
    drv = build_driver(tree) if tree else None
    for kind, w, vals in tables:
        wt = ((w[0], w[1]), (w[2], w[3]), (w[4], w[5]))
        want = ref.flat(wt, kind)
        if vals != want:
            i = next(
                (k for k in range(min(len(vals), len(want))) if vals[k] != want[k]),
                min(len(vals), len(want)),
            )
            fail(
                f"kind {kind} weights {w}: Bend != gen_sched.py at word {i} "
                f"(len {len(vals)} vs {len(want)})"
            )
        sys.stdout.write(
            f"kind {kind} weights {w}: Bend == gen_sched.py ({len(vals)} words) "
            "IDENTICAL\n"
        )
        if drv and tree:
            check_driver(drv, tree, kind, w, vals)
    if tree:
        check_headers(tree, tables)
    sys.stdout.write("m16_wsched_diff: PASS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
