#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Check bend/pattn_sched.bend against the ext 3010 prefill attention kernel.

Finite differential check of the tile schedule against the index expressions of
exllamav3_ext/pattn_kernel.cuh. See USAGE for the full description.
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
    "Finite differential check of bend/pattn_sched.bend (ext 3010 prefill "
    "attention tile schedule) against the\n"
    "index expressions of exllamav3_ext/pattn_kernel.cuh.\n"
    "\n"
    "Quoted verbatim from the patched engine tree (argument = the exllamav3 "
    "package directory), each required\n"
    "exactly once (the tile start n0 twice: issue and loop, which must agree):\n"
    "  the PA_* constants PA_G, PA_BQ, PA_BN, PA_PAGE;\n"
    "  qb, p0, total, q_abs0, n_hi, ntiles, the issue n0 and row0, the loop n0, "
    "the mask guard, q_abs_r0 / r1,\n"
    "  key, qa, the drop condition, hq (the warp's q head).\n"
    "The C++ program replays every thread (gid < 8, t < 4) of every query block "
    "of each chunk through its tile\n"
    "loop and (nt, e) fragment slots, counts per (row, key) the iterations that "
    "keep the key, prints the same\n"
    "table as bend/PATTN_SCHED_TABLE.bend (compared byte for byte), and "
    "independently checks the counts against\n"
    "causal attention (1 iff key <= total - q_len + row position, over every key "
    "the loop touches), the page\n"
    "bound (row0's page offset + PA_BN <= PA_PAGE) and the (kv head, warp) -> "
    "q head bijection.\n"
    "Differential evidence on finite instances, not a proof. `--mutate NAME` "
    "applies a deliberate source\n"
    "mutation that the check must reject.\n"
    "\n"
    "Usage: python3 bend/pattn_sched_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR\n"
    "  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py.\n"
)
N0_SITES = 2  # the tile start n0: issue and loop
MUTATE_ARGC = 3
PLAIN_ARGC = 2
REPO = source_link.REPO
TABLE = "bend/PATTN_SCHED_TABLE.bend"
# (L, q_len) chunks: must match main() of PATTN_SCHED_TABLE.bend
CASES = [
    (1, 1),
    (17, 5),
    (31, 16),
    (32, 17),
    (33, 33),
    (47, 20),
    (64, 64),
    (65, 16),
    (100, 37),
    (130, 3),
    (62, 32),
    (81, 19),
]

MUTATIONS = {
    # the loop stops one query row early
    "n_hi_short": (
        "min(q_abs0 + PA_BQ, total)",
        "min(q_abs0 + PA_BQ - 1, total)",
        1,
    ),
    # the maskless fast path covers one tile column too many
    "mask_guard": (
        "if (n0 + PA_BN - 1 > q_abs0)",
        "if (n0 + PA_BN - 2 > q_abs0)",
        1,
    ),
    # the causal comparison admits the key after the row
    "causal_off_by_one": (
        "if (key > qa || key >= total)",
        "if (key > qa + 1 || key >= total)",
        1,
    ),
    # the upper row half reads the lower half's bound
    "row_half": (
        "const int q_abs_r0 = q_abs0 + gid, q_abs_r1 = q_abs0 + gid + 8;",
        "const int q_abs_r0 = q_abs0 + gid, q_abs_r1 = q_abs0 + gid;",
        1,
    ),
}


def fail(msg: str) -> NoReturn:
    """Exit with the link's failure message.

    Raises:
        SystemExit: Always.

    """
    text = f"pattn_sched_diff: {msg}"
    raise SystemExit(text)


def one(src: str, pat: str, group: int = 0) -> str:
    """Match pat exactly once in src.

    Returns:
        The requested group of the match.

    """
    m = list(re.finditer(pat, src))
    if len(m) != 1:
        fail(f"pattern {pat!r} matched {len(m)} times, expected 1")
    found = m[0].group(group)
    if not isinstance(found, str):
        fail(f"pattern {pat!r} group {group} did not participate")
    return found


HARNESS = (
    r"""
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
using std::min;
@@DEFS@@
static int bad_ref = 0, bad_page = 0, bad_head = 0;
static void chunk(int L, int q_len) {
    const int kv_append_len = q_len;
    const int cs = L - q_len;
    int grid = (q_len + PA_BQ - 1) / PA_BQ;
    for (int k = 0; k < grid; ++k) {"""
    r"""            // blocks in qb order (the table's order)
        const int blockIdx_x = grid - 1 - k;
        const int gridDim_x = grid;
        @@QB@@
        @@P0@@
        const int cache_seqlens_b = cs;
        @@TOTAL@@
        @@QABS0@@
        @@NHI@@
        @@NTILES@@
        std::vector<std::vector<int>> cnt(PA_BQ, std::vector<int>(L + 2 + 64, 0));
        for (int gid = 0; gid < 8; ++gid) for (int t = 0; t < 4; ++t) {
            @@QABSR@@
            for (int it = 0; it < ntiles; ++it) {
                {
                    @@N0_ISSUE@@
                    int bt0 = n0 / PA_PAGE;   // identity block table
                    @@ROW0@@
                    if ((int) (row0 % PA_PAGE) + PA_BN > PA_PAGE) ++bad_page;
                    (void) bt0;
                }
                @@N0_LOOP@@
                bool masked = false;
                @@GUARD@@ masked = true;
                for (int nt = 0; nt < 4; ++nt) for (int e = 0; e < 4; ++e) {
                    @@KEY@@
                    @@QA@@
                    bool drop = false;
                    if (masked) { @@DROP@@ drop = true; }
                    int r = gid + 8 * (e >> 1);
                    if (!drop && key < (int) cnt[r].size()) cnt[r][key]++;
                    else if (!drop) ++bad_ref;
                }
            }
        }
        for (int r = 0; r < PA_BQ; ++r) {
            if (p0 + r >= q_len) continue;
            int a = cs + p0 + r;
            for (int p = 0; p < (int) cnt[r].size(); ++p)
                if (cnt[r][p] != (p <= a ? 1 : 0)) ++bad_ref;
            printf("V %d %d %d %d ", L, q_len, qb, r);
            for (int p = 0; p < L + 2; ++p) printf("%d", cnt[r][p]);
            printf("\n");
        }
    }
}
int main() {
    const int blockIdx_x = 0, gridDim_x = 1;   // unused outside chunk()
    (void) blockIdx_x; (void) gridDim_x;
    const int CASES[][2] = {@@CASES@@};
    // Emit in the Bend table's order: chunks, then pages, heads, spans
    for (auto& c : CASES) chunk(c[0], c[1]);
    for (int it = 0; it < 20; ++it) {
        @@N0_LOOP@@
        printf("P %d %d\n", it, n0 % PA_PAGE);
    }
    int seen[64] = {0};
    for (int kvh = 0; kvh < 4; ++kvh) for (int warp = 0; warp < PA_G; ++warp) {
        @@HQ@@
        printf("G %d %d %d\n", kvh, warp, hq);
        if (hq < 0 || hq >= 4 * PA_G || seen[hq]++) ++bad_head;
    }
    for (int k = 0, d = 0; k < 40; ++k, d += 7) printf("S %d %d\n", d, d / 32);
    fprintf(stderr, "reference mismatches %d, page overruns %d, """
    r"""head map errors %d\n", bad_ref, bad_page, bad_head);
    return (bad_ref || bad_page || bad_head) ? 1 : 0;
}
"""
)


def c_program(cu: str) -> str:
    """Fill the C++ harness with the quoted kernel lines.

    Returns:
        The program source.

    """
    q: dict[str, str] = {}
    defs = [
        one(cu, rf"#define {name} \d+")
        for name in ("PA_G", "PA_BQ", "PA_BN", "PA_PAGE")
    ]
    q["DEFS"] = "\n".join(defs)
    q["QB"] = (
        one(cu, r"const int qb = gridDim\.x - 1 - blockIdx\.x;[^\n]*")
        .replace("gridDim.x", "gridDim_x")
        .replace("blockIdx.x", "blockIdx_x")
    )
    q["P0"] = one(cu, r"const int p0 = qb \* PA_BQ;")
    q["TOTAL"] = one(
        cu, r"const int total = cache_seqlens\[b\] \+ kv_append_len;"
    ).replace("cache_seqlens[b]", "cache_seqlens_b")
    q["QABS0"] = one(cu, r"const int q_abs0 = [^;\n]*;")
    q["NHI"] = one(cu, r"const int n_hi = [^;\n]*;")
    q["NTILES"] = one(cu, r"const int ntiles = [^;\n]*;")
    n0s = re.findall(r"const int n0 = [^;\n]*;", cu)
    if len(n0s) != N0_SITES or n0s[0] != n0s[1]:
        fail(f"expected two identical n0 lines (issue, loop), found {n0s}")
    q["N0_ISSUE"] = q["N0_LOOP"] = n0s[0]
    q["ROW0"] = one(cu, r"const size_t row0 = [^;\n]*;").replace(
        "bt[n0 / PA_PAGE]", "bt0"
    )
    q["GUARD"] = one(cu, r"if \(n0 \+ PA_BN - \d+ > q_abs0\)")
    q["QABSR"] = one(cu, r"const int q_abs_r0 = [^;\n]*;")
    q["KEY"] = one(cu, r"int key = [^;\n]*;")
    q["QA"] = one(cu, r"int qa = [^;\n]*;")
    q["DROP"] = one(cu, r"if \(key > qa[^\n]*\)(?= sc)")
    q["HQ"] = one(cu, r"const int hq = [^;\n]*;")
    q["CASES"] = ", ".join(f"{{{a}, {b}}}" for a, b in CASES)
    src = HARNESS
    for k, v in q.items():
        src = src.replace(f"@@{k}@@", v)
    left = re.findall(r"@@\w+@@", src)
    if left:
        fail(f"unfilled harness slots {left}")
    return src


def main(argv: list[str]) -> None:
    """Compile the quoted schedule and compare its table with Bend's."""
    mutate = None
    if len(argv) >= MUTATE_ARGC and argv[1] == "--mutate":
        mutate, argv = argv[2], [argv[0], *argv[3:]]
    if len(argv) != PLAIN_ARGC:
        fail(USAGE)
    cu = (Path(argv[1]) / "exllamav3_ext/pattn_kernel.cuh").read_text()
    if mutate:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate}; known: {', '.join(MUTATIONS)}")
        a, b, n = MUTATIONS[mutate]
        if cu.count(a) != n:
            fail(
                f"mutation {mutate} does not apply "
                f"({cu.count(a)} occurrences, expected {n})"
            )
        cu = cu.replace(a, b)
    # The schedule is a function of the launch arguments: the kernel never
    # writes them
    hits = re.findall(
        r"\b(q_len|kv_append_len|num_pages_per_seq)\s*(?:[+\-*/]?=)(?!=)", cu
    )
    if hits:
        fail(f"pattn_kernel.cuh writes a schedule argument: {hits}")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "diff.cpp"
        c.write_text(c_program(cu))
        exe = Path(td) / "diff"
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked([
                "c++",
                "-O2",
                "-std=c++17",
                "-w",
                "-o",
                str(exe),
                str(c),
            ]),
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
    if cres.returncode != 0 or not same:
        sys.stdout.write("pattn_sched_diff: MISMATCH\n")
        sys.exit(1)
    sys.stdout.write("pattn_sched_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
