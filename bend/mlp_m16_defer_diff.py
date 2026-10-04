#!/usr/bin/env python3
"""
Differential check of ext 8204's deferred phase-1 publish (bend/mlp_m16_defer.bend) against an engine tree.

1. Both kernels (exllamav3_ext/quant/exl3_mlp_m16_kernel.cuh = 8201, exl3_tail_m16_kernel.cuh = 8202 tail) must
   contain the deferred-publish block verbatim (quoted below, with the kernel's counter pointer name): the
   defer_pub = 0 branch (fence + add of cnt1[g] at every flush, 8201's order), and the defer_pub = 1 branch (no
   signal at a mid-slice flush; at the flush of the slice's last group g == (s1 + n1 - 1) / KT1 one fence, then
   lane i adds 1 to cnt1[s1 / KT1 + i] for i <= g - s1 / KT1).
2. At the served instance (the baked table exllamav3_ext/quant/exl3_mlp_m16_sched.h of the tree, G = 164,
   KT1 = 320), for every block: the lane rule's groups == the model's phase-1 signal groups (the Flush groups
   of the block's main loop over [s1, s1 + n1), i.e. mlp_m16_sched's sigs1, in order), they are at most 32
   (one per lane), and the publish fires at the last flush, once.
3. The baked headers of the tree equal the Bend emitters' output (bend/MLP_M16_SCHED_TABLE.bend,
   bend/TAIL_M16_SCHED_TABLE.bend): 8204 changes no table.
Exit status 0 iff everything holds. Mutations (--mutate NAME) must make it fail.

Usage: python3 -B bend/mlp_m16_defer_diff.py TREE [--mutate first_lane|last_group|no_pub_last]
  TREE: OUT/patched of bend/engine_trees.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
G, KT1, PF = 164, 5120 // 16, 8

BLOCK = """            if (!{P}.defer_pub)
            {{
                __threadfence();
                __syncwarp();
                if (lane == 0) atomicAdd({P}.cnt + EXL3_MLP_CNT1 + g, 1u);
            }}
            else if (g == (s1 + n1 - 1) / KT1)   // the slice's last group
            {{
                __threadfence();
                __syncwarp();
                const int g_first = s1 / KT1;         // lane i publishes group g_first + i (no loop:
                if (lane <= g - g_first)              // a lane-0 loop here de-converges the hot loop)
                    atomicAdd({P}.cnt + EXL3_MLP_CNT1 + g_first + lane, 1u);
            }}
"""


def fail(msg: str) -> None:
    raise SystemExit(f"mlp_m16_defer_diff: FAIL: {msg}")


def table(quant: Path):
    text = (quant / "exl3_mlp_m16_sched.h").read_text()
    body = re.search(r"exl3_mlp_sched_block\[[^\]]*\] = \{(.*?)\};", text, re.S).group(1)
    v = [int(x) for x in re.findall(r"\d+", body)]
    return [(v[b * 11], v[b * 11 + 1]) for b in range(G)]


def flush_groups(s: int, n: int, kt: int) -> list[int]:
    """Groups flushed by the v2 main loop over [s, s + n) (a flush at every group end and at the range end)."""
    out = []
    for it in range(s, s + n):
        if it + 1 == s + n or (it + 1) % kt == 0:
            out.append(it // kt)
    return out


def lane_rule(s1: int, n1: int, mutate: str) -> list[list[int]]:
    """For each flush of the slice, in order: the groups the 32 lanes of a warp publish at that flush."""
    per_flush = []
    last = (s1 + n1 - 1) // KT1
    if mutate == "last_group":
        last = (s1 + n1) // KT1
    for g in flush_groups(s1, n1, KT1):
        pubs = []
        if g == last and mutate != "no_pub_last":
            g_first = s1 // KT1 + (1 if mutate == "first_lane" else 0)
            pubs = [g_first + lane for lane in range(32) if lane <= g - g_first]
        per_flush.append(pubs)
    return per_flush


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit(__doc__)
    tree = Path(argv[0])
    mutate = argv[argv.index("--mutate") + 1] if "--mutate" in argv else ""
    quant = tree / "exllamav3_ext" / "quant"
    for name, p in (("exl3_mlp_m16_kernel.cuh", "p"), ("exl3_tail_m16_kernel.cuh", "q")):
        src = (quant / name).read_text()
        if BLOCK.format(P=p) not in src:
            fail(f"{name}: deferred-publish block not found verbatim")
        print(f"{name}: deferred-publish block present (quoted rule)")
    n_blocks = 0
    for b, (s1, n1) in enumerate(table(quant)):
        model = flush_groups(s1, n1, KT1)            # mlp_m16_sched sigs1: one Sig{C1{g}} per Flush{g}
        pubs = lane_rule(s1, n1, mutate)
        fired = [i for i, x in enumerate(pubs) if x]
        if fired != [len(pubs) - 1]:
            fail(f"block {b}: the publish fires at flushes {fired}, expected only the last ({len(pubs) - 1})")
        if pubs[-1] != model:
            fail(f"block {b}: lanes publish groups {pubs[-1]}, the model signals {model}")
        if len(model) > 32:
            fail(f"block {b}: {len(model)} groups exceed one per lane")
        n_blocks += 1
    print(f"lane rule == model signal groups, fired once at the last flush, <= 32 groups: all {n_blocks} blocks")
    for tab, hdr in (("bend/MLP_M16_SCHED_TABLE.bend", "exl3_mlp_m16_sched.h"),
                     ("bend/TAIL_M16_SCHED_TABLE.bend", "exl3_tail_m16_sched.h")):
        out = subprocess.run(source_link.locked([source_link.bend(), tab]), cwd=REPO, capture_output=True, timeout=3600,
                             check=False)
        if out.returncode != 0:
            fail(f"{tab} exited {out.returncode}")
        if out.stdout != (quant / hdr).read_bytes():
            fail(f"{hdr} of the tree != {tab} output")
        print(f"{hdr} of the tree == {tab} output (IDENTICAL)")
    print("mlp_m16_defer_diff: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
