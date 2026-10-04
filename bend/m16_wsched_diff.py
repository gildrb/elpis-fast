#!/usr/bin/env python3
"""
Differential check of ext 8205's slot-weighted 8201 / 8202 schedule tables.

Runs the Bend emitter bend/M16_WSCHED_TABLE.bend (Nat model bend/m16_wsched.bend over the shared partition
bend/gemm_m16_wpart.bend) and compares every printed table, byte for byte as a list of uint16 values, with
  1. the independent Python reference bend/gen/m16_wsched_ref.py (flat(), over the bend/gen/wpart.py formula), and
  2. with --tree TREE: the host builder the extension ships, exllamav3_ext/quant/exl3_m16_wsched.h of TREE,
     compiled into a small driver (c++ from PATH; the dev shell gives clang), and the
     uniform-weight tables' 8201 / tail sections against the Bend-baked headers exl3_mlp_m16_sched.h /
     exl3_tail_m16_sched.h of TREE (the extension checks the same at runtime and fails closed).
Exit status 0 iff every comparison is IDENTICAL.

Usage: python3 -B bend/m16_wsched_diff.py [--reference GEN_SCHED_PY] [--tree TREE]
  GEN_SCHED_PY: the independent Python reference (default: the tracked bend/gen/m16_wsched_ref.py).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/M16_WSCHED_TABLE.bend"
DEFAULT_REFERENCE = REPO / "bend/gen/m16_wsched_ref.py"
G, BF, NP, PFLD, TBF, HDR = 164, 11, 34, 5, 4, 8

DRIVER = r'''
#include "exl3_m16_wsched.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv)
{
    int kind = atoi(argv[1]); int w[6]; for (int i = 0; i < 6; ++i) w[i] = atoi(argv[2 + i]);
    std::vector<uint16_t> t; std::string err;
    if (!exl3_wsched::build(kind, w, t, err)) { printf("INVALID %s\n", err.c_str()); return 0; }
    for (size_t i = 0; i < t.size(); ++i) printf(i ? ",%d" : "%d", t[i]);
    printf("\n");
}
'''


def fail(msg: str) -> None:
    raise SystemExit(f"m16_wsched_diff: FAIL: {msg}")


def bend_tables() -> list[tuple[int, list[int], list[int]]]:
    proc = subprocess.run([source_link.bend(), TABLE], cwd=REPO, capture_output=True, timeout=3600, check=False)
    if proc.returncode != 0:
        fail(f"{TABLE} exited {proc.returncode}: {proc.stderr.decode(errors='replace')[:500]}")
    out = []
    for line in proc.stdout.decode().splitlines():
        if not line.strip():
            continue
        m = re.fullmatch(r"kind (\d) weights ([\d,]+):([\d,]+)", line)
        if not m:
            fail(f"unparsable line: {line[:80]}")
        out.append((int(m.group(1)), [int(x) for x in m.group(2).split(",")], [int(x) for x in m.group(3).split(",")]))
    if len(out) != 4:
        fail(f"expected 4 tables, got {len(out)}")
    return out


def load_reference(path: str):
    spec = importlib.util.spec_from_file_location("gen_sched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_driver(tree: Path) -> list[str]:
    quant = tree / "exllamav3_ext" / "quant"
    work = Path(tempfile.mkdtemp(prefix="m16_wsched_diff_"))
    (work / "drv.cpp").write_text(DRIVER)
    shutil.copy(quant / "exl3_m16_wsched.h", work / "exl3_m16_wsched.h")
    cxx = shutil.which("c++")
    if cxx is None:
        fail("c++ is not on PATH (run inside `nix develop --offline --no-write-lock-file`)")
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(work / "drv"), str(work / "drv.cpp")], check=True)
    return [str(work / "drv")]


def header_table(text: str, name: str) -> list[int]:
    body = re.search(r"static const unsigned short " + name + r"\[[^\]]*\] = \{(.*?)\};", text, re.S)
    if not body:
        fail(f"{name} not found")
    return [int(x) for x in re.findall(r"\d+", body.group(1))]


def main(argv: list[str]) -> int:
    ref_path = argv[argv.index("--reference") + 1] if "--reference" in argv else str(DEFAULT_REFERENCE)
    tree = Path(argv[argv.index("--tree") + 1]) if "--tree" in argv else None
    ref = load_reference(ref_path)
    tables = bend_tables()
    drv = build_driver(tree) if tree else None
    for kind, w, vals in tables:
        wt = ((w[0], w[1]), (w[2], w[3]), (w[4], w[5]))
        want = ref.flat(wt, kind)
        if vals != want:
            i = next((k for k in range(min(len(vals), len(want))) if vals[k] != want[k]), min(len(vals), len(want)))
            fail(f"kind {kind} weights {w}: Bend != gen_sched.py at word {i} (len {len(vals)} vs {len(want)})")
        print(f"kind {kind} weights {w}: Bend == gen_sched.py ({len(vals)} words) IDENTICAL")
        if drv:
            out = subprocess.run(drv + [str(kind)] + [str(x) for x in w], capture_output=True, text=True,
                                 check=True).stdout.strip()
            got = [int(x) for x in out.split(",")] if not out.startswith("INVALID") else None
            if got != vals:
                fail(f"kind {kind} weights {w}: Bend != the tree's exl3_m16_wsched.h builder ({out[:60]})")
            print(f"kind {kind} weights {w}: Bend == exl3_m16_wsched.h builder of {tree} IDENTICAL")
    if tree:
        quant = tree / "exllamav3_ext" / "quant"
        mh = (quant / "exl3_mlp_m16_sched.h").read_text()
        th = (quant / "exl3_tail_m16_sched.h").read_text()
        for kind, w, vals in tables:
            if w != [1, 1, 1, 1, 1, 1]:
                continue
            nwait = vals[3]
            blk = vals[HDR:HDR + G * BF]
            pair = vals[HDR + G * BF:HDR + G * BF + NP * PFLD]
            wait = vals[HDR + G * BF + NP * PFLD:HDR + G * BF + NP * PFLD + nwait]
            ok = (blk == header_table(mh, "exl3_mlp_sched_block") and pair == header_table(mh, "exl3_mlp_sched_pair")
                  and wait == header_table(mh, "exl3_mlp_sched_wait"))
            if kind == 1:
                grp_end = HDR + G * BF + NP * PFLD + nwait + 4 * NP + 2 * 10
                ok = ok and vals[grp_end:grp_end + G * TBF] == header_table(th, "exl3_tail_sched_block")
            if not ok:
                fail(f"kind {kind} uniform: the 8201 / tail sections != the tree's baked Bend headers")
            print(f"kind {kind} uniform: 8201{' and tail' if kind else ''} sections == baked headers of {tree} IDENTICAL")
    print("m16_wsched_diff: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
