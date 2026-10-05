#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Check bend/attn_split.bend against the ext 3003 split kernel, on finite instances.

Finite differential check of the verify-attention split/combine partition (ext
patch 3003) against the index expressions of exllamav3_ext/attn_verify.cuh +
attn_verify.cu. See USAGE for the full description.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

USAGE = (
    "\n"
    "Finite differential check of bend/attn_split.bend (verify-attention "
    "split/combine partition, ext\n"
    "patch 3003) against the index expressions of exllamav3_ext/attn_verify.cuh "
    "+ attn_verify.cu.\n"
    "\n"
    "av_split_len / av_live_splits are extracted verbatim from attn_verify.cuh; "
    "the split kernel's\n"
    "partition lines (s_len, n_start, the `n_start >= L` early return, n_end), "
    "its tile loop bounds and\n"
    "token mask (ntiles, n0, tok0 = n0 + warp * 8 + 2 * t, "
    "`tok < n_end && tok <= q_abs`) and the\n"
    "combine's live count are extracted verbatim from attn_verify.cu and compiled "
    "for the CPU (c++).\n"
    "The C program prints the same table as bend/ATTN_SPLIT_TABLE.bend (compared "
    "byte for byte) and\n"
    "separately replays every (warp, lane, e) token of every tile of every running "
    "split for q_len 8:\n"
    "each kv position must enter row q_pos's softmax exactly once iff "
    "p <= depth + q_pos, and the\n"
    'combine\'s s < live must equal "the split kernel ran s".\n'
    "\n"
    "Differential evidence on finite instances, not a proof. `--mutate NAME` "
    "applies a deliberate\n"
    "kernel mutation that the check must reject.\n"
    "\n"
    "Usage: python3 bend/attn_split_diff.py [--mutate NAME] ATTN_VERIFY_SRC_DIR\n"
    "  ATTN_VERIFY_SRC_DIR: the exllamav3_ext directory of an engine tree with "
    "the av_split_len\n"
    "  revision of patch 3003. The tracked 3003 (fixed kv chunks) does not have "
    "it; see\n"
    "  bend/attn_chunk_diff.py.\n"
)
MUTATE_ARGC = 3
PLAIN_ARGC = 2
REPO = source_link.REPO
TABLE = "bend/ATTN_SPLIT_TABLE.bend"
LS = [
    1,
    8,
    63,
    64,
    65,
    115,
    127,
    128,
    129,
    256,
    257,
    1000,
    1288,
    2056,
    4104,
    8198,
    12808,
    18608,
    32736,
]
SS = [1, 2, 3, 7, 20, 40, 82]

MUTATIONS = {
    # per-split length rounded down instead of up
    "floor_per_split": ("int per = (L + S - 1) / S;", "int per = L / S;"),
    # causal mask off by one: the diagonal (own position) is dropped
    "causal_strict": ("tok < n_end && tok <= q_abs", "tok < n_end && tok < q_abs"),
    # combine reads one split fewer than live
    "combine_live_floor": ("return (L + sl - 1) / sl;", "return L / sl;"),
}

KERNEL_LINES = [
    r"const int s_len = av_split_len\(L, S\);",
    r"const int n_start = split \* s_len;",
    r"if \(n_start >= L\) return;[^\n]*",
    r"const int n_end = min\(n_start \+ s_len, L\);",
    r"const int ntiles = \(n_end - n_start \+ AV_T - 1\) / AV_T;",
    r"const int n0 = n_start \+ it \* AV_T;",
    r"const int tok0 = n0 \+ warp \* 8 \+ 2 \* t;",
    r"bool ok = row_ok\[2 \* i \+ hr\] && [^;]*;",
    r"const int q_abs = L - q_len \+ gid;[^\n]*",
    r"const int live = av_live_splits\(L, num_splits\);",
]


def fail(msg: str) -> NoReturn:
    """Exit with the link's failure message.

    Raises:
        SystemExit: Always.

    """
    text = f"attn_split_diff: {msg}"
    raise SystemExit(text)


def grab(text: str, pattern: str) -> str:
    """Match pattern exactly once in text.

    Returns:
        The match.

    """
    m = re.findall(pattern, text)
    if len(m) != 1:
        fail(f"pattern {pattern!r} matched {len(m)} times")
    found = m[0]
    if not isinstance(found, str):
        fail(f"pattern {pattern!r} has groups")
    return found


def c_program(cuh: str, cu: str) -> str:
    """Build the C++ table and replay program from the quoted kernel lines.

    Returns:
        The program source.

    """
    fns = [
        grab(
            cuh,
            r"__host__ __device__ inline int av_split_len\(int L, int S\)"
            r"\n\{\n(?:.*\n)*?\}",
        ),
        grab(
            cuh,
            r"__host__ __device__ inline int av_live_splits\(int L, int S\)"
            r"\n\{\n(?:.*\n)*?\}",
        ),
    ]
    fns = [f.replace("__host__ __device__ ", "") for f in fns]
    k = [grab(cu, p) for p in KERNEL_LINES]
    ret = k[2].replace("return;", "{ runs = 0; }").split("//")[0]
    tile_ok = k[7].replace("row_ok[2 * i + hr] && ", "")
    ls = ", ".join(map(str, LS))
    ss = ", ".join(map(str, SS))
    return f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using std::min;
#define AV_T 64
{fns[0]}
{fns[1]}
struct Part {{ int s_len, n_start, n_end, runs, ntiles; }};
static Part part(int L, int S, int split) {{
    Part P; int runs = 1;
    {k[0]}
    {k[1]}
    {ret}
    {k[3]}
    {k[4]}
    P.s_len = s_len; P.n_start = n_start; P.n_end = n_end; P.runs = runs; \
P.ntiles = runs ? ntiles : 0;
    return P;
}}
static int live_of(int L, int num_splits) {{ {k[9]} return live; }}
int main(int argc, char** argv) {{
    const int Ls[] = {{{ls}}};
    const int Ss[] = {{{ss}}};
    const int q_len = 8;
    long bad = 0;
    for (int L : Ls) for (int S : Ss) {{
        Part p0 = part(L, S, 0);
        printf("L=%d S=%d sl=%d live=%d", L, S, p0.s_len, live_of(L, S));
        for (int i = 0; i < S; ++i) {{ Part P = part(L, S, i); \
printf(" %d:%d:%d", P.n_start, P.n_end, P.runs); }}
        printf("\\n");
        // combine reads s < live exactly for the splits that ran
        for (int i = 0; i < S; ++i) \
if ((i < live_of(L, S)) != (part(L, S, i).runs == 1)) ++bad;
        if (L < q_len) continue;
        // token replay of the split kernel's tile loop: \
every (warp, t, e) of every tile
        for (int gid = 0; gid < q_len; ++gid) {{
            std::vector<int> cnt(L + 64 * 2, 0);
            for (int split = 0; split < S; ++split) {{
                Part P = part(L, S, split);
                if (!P.runs) continue;
                int n_start = P.n_start, n_end = P.n_end;
                for (int it = 0; it < P.ntiles; ++it) {{
                    {k[5]}
                    for (int warp = 0; warp < 8; ++warp) for (int t = 0; t < 4; ++t) {{
                        {k[6]}
                        {k[8].split("//")[0]}
                        for (int e = 0; e < 2; ++e) {{
                            int tok = tok0 + e;
                            {tile_ok}
                            if (ok) ++cnt[tok];
                        }}
                    }}
                }}
            }}
            int depth = L - q_len;
            for (int p = 0; p < (int) cnt.size(); ++p) \
if (cnt[p] != (p <= depth + gid ? 1 : 0)) ++bad;
        }}
    }}
    fprintf(stderr, "coverage/combine violations: %ld\\n", bad);
    return bad ? 3 : 0;
}}
"""


def main(argv: list[str]) -> None:
    """Compile the quoted kernel lines and compare their table with Bend's."""
    mutate = None
    if len(argv) >= MUTATE_ARGC and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0], *argv[3:]]
    if len(argv) != PLAIN_ARGC:
        fail(USAGE)
    src = Path(argv[1])
    cuh = (src / "attn_verify.cuh").read_text()
    cu = (src / "attn_verify.cu").read_text()
    if mutate:
        a, b = MUTATIONS[mutate]
        tgt = "cuh" if a in cuh else "cu"
        if tgt == "cuh":
            cuh = cuh.replace(a, b)
        else:
            if a not in cu:
                fail(f"mutation {mutate} does not apply")
            cu = cu.replace(a, b)
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "diff.cpp"
        c.write_text(c_program(cuh, cu))
        exe = Path(td) / "diff"
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked(["c++", "-O2", "-std=c++17", "-o", str(exe), str(c)]),
            check=True,
        )
        cres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            source_link.locked([str(exe)]), capture_output=True, text=True, check=False
        )
    bres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        source_link.locked([source_link.bend(), TABLE]),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    same = cres.stdout == bres.stdout
    sys.stdout.write(
        (cres.stderr.strip() or f"C table program exit status {cres.returncode}") + "\n"
    )
    sys.stdout.write(
        f"table lines: C {len(cres.stdout.splitlines())}, "
        f"Bend {len(bres.stdout.splitlines())}; byte-identical: {same}\n"
    )
    if not same or cres.returncode != 0:
        fail("MISMATCH" if not same else "C-side coverage violation")
    sys.stdout.write("attn_split_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
