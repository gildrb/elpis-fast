#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential check of bend/gemm_m16_group.bend (ext patch 2102).

The patch's own index expressions are compiled for the CPU and compared with the
Bend table; USAGE holds the full description printed on a usage error.
"""

from __future__ import annotations

import hashlib
import random
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

USAGE = (
    "\n"
    "Finite differential check of bend/gemm_m16_group.bend (grouped "
    "m16 GEMM, ext patch 2102) against\n"
    "the patch's own index expressions compiled for the CPU.\n"
    "\n"
    "The two new sources of the patch "
    "(exllamav3_ext/quant/exl3_gemm_m16g_kernel.cuh = K,\n"
    "exl3_gemm_m16g.cu = H) are rebuilt from the patch file; the 2001 "
    "header the kernel includes\n"
    "(exl3_gemm_m16_kernel.cuh: EXL3_M16_* constants, exl3_m16_owner) from\n"
    "patches/exl3-ext/2001-gemm-m16-swapab.patch. Every quoted line is "
    "located by its text and must sit\n"
    "at the K:/H: line number that bend/gemm_m16_group.bend cites. The "
    "C++ program assembles the Args\n"
    "struct, exl3_m16g_sel/gbase/mat, the host gbase fill, owner, "
    "max_contrib, shape_ok, launch_fits\n"
    "and the kernel's partition, main-loop, flush, finisher and "
    "prologue lines verbatim into an\n"
    "emulated launch (every block, warp and lane of the grid) and "
    "prints the table of\n"
    "bend/GEMM_M16_GROUP_TABLE.bend, compared byte for byte. For "
    "launchable configurations (shape_ok and\n"
    "launch_fits) the C side also checks, independently of Bend: every "
    "(matrix, local group, k-tile)\n"
    "consumed exactly once; every flushed float stored once, inside "
    "the slot region; every finisher\n"
    "read hits a stored float; every output element of every matrix "
    "written exactly once; every\n"
    "input-transform element written exactly once.\n"
    "\n"
    "Configurations: the served bundles (GDN qkv+z 10240+6144, "
    "attention q+k+v 12288+1024+1024, k 5120,\n"
    "m 1..16, G in {82, 164}) plus seeded random bundles. Differential "
    "evidence on finite instances,\n"
    "not a proof. `--mutate NAME` applies a deliberate source mutation "
    "that the check must reject.\n"
    "\n"
    "Usage: python3 bend/gemm_m16_group_diff.py [--mutate NAME] "
    "PATCH_2102 [PATCH_2001]\n"
    "  PATCH_2102: "
    "patches/exl3-ext/2102-proj-m16-grouped-v2-on3003-5101.patch "
    "(sha256-pinned).\n"
)
REPO = source_link.REPO
TABLE = "bend/GEMM_M16_GROUP_TABLE.bend"
# The shipped 2102: 2102-proj-m16-grouped-v2-on3003-5101.patch (hunk positions
# rebased onto 3003 + 5101; its new kernel/host files are byte-identical to 2102 v2
# 50f75e69...)
PATCH_SHA = "c12f897d48c07ab26fbdda03f1a44e762324ede735b8bb461b2cc8f8a8a66ff3"
OWNER_CONSTANTS = 5
MAX_ARGS = 3
MUTATE_ARGS = 3
MAX_MISMATCH_REPORTS = 3

MUTATIONS = {
    # host gbase prefix sum off by one (every matrix after the first starts one
    # group late)
    "gbase_prefix": (
        "args.gbase[i] = groups;\n        groups += n / EXL3_M16_GW;",
        "args.gbase[i] = groups + (i > 0);\n        groups += n / EXL3_M16_GW;",
    ),
    # matrix lookup with a strict comparison: first group of each matrix goes to the
    # previous one
    "mat_strict": ("if (g >= p.gbase[i]) mat = i;", "if (g > p.gbase[i]) mat = i;"),
    # finisher column ignores the matrix's group base
    "col0_global": (
        "const int col0 = (g - gb) * GW + ch * 128;",
        "const int col0 = g * GW + ch * 128;",
    ),
    # prologue rows per matrix halved
    "per_mat": (
        "const int per_mat = size_m * size_k / 128;",
        "const int per_mat = size_m * size_k / 256;",
    ),
    # weight row stride of half the owning matrix's width
    "row_bytes": (
        "row_bytes = (size_t) (exl3_m16g_gbase(p, mat + 1) - gb) * (GW / 16) * 128;",
        "row_bytes = (size_t) (exl3_m16g_gbase(p, mat + 1) - gb) * (GW / 16) * 64;",
    ),
}

# (file, line) -> exact source text (stripped) quoted by bend/gemm_m16_group.bend
K_LINES = {
    25: "#define EXL3_M16G_MAX_MATS 4",
    68: "if (g >= p.gbase[i]) mat = i;",
    95: "const int KT = size_k >> 4;",
    96: "const int groups = exl3_m16g_gbase(p, EXL3_M16G_MAX_MATS);",
    97: "const int total = groups * KT;",
    100: "const int s_b = (int) (((int64_t) b * total) / G);",
    101: "const int e_b = (int) (((int64_t) (b + 1) * total) / G);",
    102: "const int n_iter = e_b - s_b;",
    109: "size_t row_bytes = 0;",
    114: "row_bytes = (size_t) (exl3_m16g_gbase(p, mat + 1) - gb) * (GW / 16) * 128;",
    115: (
        "return (const uint8_t*) (exl3_m16g_sel(p.B, mat) + ((size_t) (g - gb)"
        " * (GW / 16) + warp * EXL3_M16_TILES_W) * 64) + lane * 16;"
    ),
    118: "int iss_g = s_b / KT;",
    119: "int iss_kt = s_b - iss_g * KT;",
    121: "iss_ptr += (size_t) iss_kt * row_bytes;",
    126: "cp_async_stream(ring_lane + slot * W_STAGE, iss_ptr);",
    132: "if (iss_j < n_iter) iss_ptr = w_src(iss_g);",
    134: "else iss_ptr += row_bytes;",
    135: "};",
    150: "int it = s_b + j;",
    151: "int g = it / KT;",
    152: "int kt = it - g * KT;",
    173: "const int per_mat = size_m * size_k / 128;",
    174: "const int total_warps = p.num_mats * per_mat;",
    175: "const int warps_grid = G * EXL3_M16_WARPS;",
    176: "for (int w = warp + EXL3_M16_WARPS * b; w < total_warps; w += warps_grid)",
    178: "const int mat = w / per_mat;",
    179: "const int r = w - mat * per_mat;",
    182: "A + r * 128,",
    183: "exl3_m16g_sel(p.xh, mat) + r * 128,",
    184: "exl3_m16g_sel(p.suh, mat) + (r * 128) % size_k,",
    239: "int cidx = b - exl3_m16_owner(g * KT, total, G);",
    240: (
        "float* slot = p.ws + ((size_t) g * p.max_contrib + cidx)"
        " * EXL3_M16_SLOT_FLOATS + warp * 64 + (lane >> 2);"
    ),
    248: "int row = mt * 8 + 2 * (lane & 3) + (c & 1);",
    249: (
        "if (row < size_m) __stcg(slot + row * GW + t * 16 + (c >> 1) * 8,"
        " acc[t][mt][c]);"
    ),
    259: "for (int g_cur = s_b / KT; j < n_iter; ++g_cur)",
    261: "const int seg_end = min(e_b, (g_cur + 1) * KT) - s_b;",
    264: "for (; j < seg_end; ++j)",
    317: "flush(g_cur);",
    326: "constexpr int CH = GW / 128;",
    327: "const int jobs = groups * size_m * CH;",
    328: "for (int w = warp + EXL3_M16_WARPS * b; w < jobs; w += EXL3_M16_WARPS * G)",
    330: "int g = w / (size_m * CH);",
    331: "int rem = w - g * size_m * CH;",
    332: "int r = rem / CH;",
    333: "int ch = rem - r * CH;",
    334: (
        "int nc = exl3_m16_owner((g + 1) * KT - 1, total, G)"
        " - exl3_m16_owner(g * KT, total, G) + 1;"
    ),
    335: (
        "float* s0 = p.ws + (size_t) g * p.max_contrib * EXL3_M16_SLOT_FLOATS"
        " + r * GW + ch * 128;"
    ),
    336: "float4 v = __ldcg(((const float4*) s0) + lane);",
    337: "for (int c = 1; c < nc; ++c)",
    339: (
        "float4 u = __ldcg(((const float4*) (s0 + (size_t) c"
        " * EXL3_M16_SLOT_FLOATS)) + lane);"
    ),
    345: "const int mat = exl3_m16g_mat(p, g);",
    346: "const int gb = exl3_m16g_gbase(p, mat);",
    347: "const int n_mat = (exl3_m16g_gbase(p, mat + 1) - gb) * GW;",
    348: "const int col0 = (g - gb) * GW + ch * 128;",
    355: "((float*) C) + (size_t) r * n_mat + col0,",
}
H_LINES = {
    19: "#define EXL3_M16G_SLOT_BYTES (8ull * 1024 * 1024)",
    20: "#define EXL3_M16G_XH_BYTES (1ull * 1024 * 1024)",
    76: "return (int) ((((int64_t) x + 1) * G + total - 1) / total) - 1;",
    86: "int nc = owner((g + 1) * KT - 1, total, G) - owner(g * KT, total, G) + 1;",
    205: "int groups = 0;",
    224: "args.gbase[i] = groups;",
    225: "groups += n / EXL3_M16_GW;",
    227: "for (int i = num_mats; i <= EXL3_M16G_MAX_MATS; ++i) args.gbase[i] = groups;",
    239: "const int KT = size_k / 16;",
    243: (
        "for (int i = 0; i < num_mats; ++i) args.xh[i] = xh"
        " + (size_t) i * size_m * size_k;"
    ),
    248: "args.max_contrib = max_contrib(groups, KT, G);",
}


def unlimited_vm() -> None:
    """Lift a soft RLIMIT_AS to the hard limit (harness shells set 8 GB).

    The Bend runtime reserves its heap up front.
    """
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def fail(msg: str) -> NoReturn:
    """Exit with a prefixed error message.

    Args:
        msg: Error text.

    Raises:
        SystemExit: Always.

    """
    text = f"gemm_m16_group_diff: {msg}"
    raise SystemExit(text)


def new_files(patch: str) -> dict[str, str]:
    """Return the post-image of every file the patch creates (--- /dev/null).

    Args:
        patch: Unified diff text.

    Returns:
        Mapping from created path to its contents.

    """
    out: dict[str, str] = {}
    lines = patch.split("\n")
    i = 0
    while i < len(lines):
        if lines[i] == "--- /dev/null" and lines[i + 1].startswith("+++ b/"):
            path = lines[i + 1][6:]
            m = re.match(r"@@ -0,0 \+1,(\d+) @@", lines[i + 2])
            if not m:
                fail(f"unexpected hunk header for {path}")
            n = int(m.group(1))
            body = lines[i + 3 : i + 3 + n]
            if any(not line.startswith("+") for line in body):
                fail(f"new file {path} hunk is not all additions")
            out[path] = "\n".join(line[1:] for line in body) + "\n"
            i += 3 + n
        else:
            i += 1
    return out


def check_lines(name: str, text: str, want: dict[int, str]) -> list[str]:
    """Fail unless each cited line number holds the expected stripped text.

    Args:
        name: Label used in error messages.
        text: Source text.
        want: Line number to expected stripped text.

    Returns:
        The source split into lines.

    """
    src = text.split("\n")
    for ln, t in want.items():
        got = src[ln - 1].strip() if ln - 1 < len(src) else "<eof>"
        if got != t:
            fail(f"{name}:{ln} is {got!r}, expected {t!r}")
    return src


def block(src: list[str], first: int, last: int) -> str:
    """Join the inclusive 1-based line range of a source.

    Args:
        src: Source lines.
        first: First line number.
        last: Last line number.

    Returns:
        The joined lines.

    """
    return "\n".join(src[first - 1 : last])


def c_program(k_src: list[str], h_src: list[str], v1: str) -> str:
    """Assemble the C++ emulation program from the quoted patch sources.

    Args:
        k_src: Kernel source lines.
        h_src: Host source lines.
        v1: The 2001 header text.

    Returns:
        The C++ program source.

    """

    def k(n: int) -> str:
        return k_src[n - 1].strip()

    def h(n: int) -> str:
        return h_src[n - 1].strip()

    defs2001 = "\n".join(
        re.findall(
            r"^#define EXL3_M16_(?:THREADS|WARPS|TILES_W|GW|SLOT_FLOATS) .*$",
            v1,
            re.MULTILINE,
        )
    )
    owner2001 = re.findall(
        r"__device__ __forceinline__ int exl3_m16_owner\(int x, int total, int G\)"
        r"\n\{\n.*\n\}",
        v1,
    )
    if (
        len(owner2001) != 1
        or len(re.findall(r"^#define EXL3_M16_", defs2001, re.MULTILINE))
        != OWNER_CONSTANTS
    ):
        fail("2001 header: owner / constants not found")

    def strip(s: str) -> str:
        return s.replace("__device__ __forceinline__ ", "")

    args_struct = block(k_src, 27, 40)  # struct Exl3M16GArgs
    # exl3_m16g_sel / exl3_m16g_gbase / exl3_m16g_mat
    sel_fns = strip(block(k_src, 42, 70))
    host_fns = block(
        h_src, 73, 113
    )  # owner, max_contrib, grid_size (dropped), shape_ok, launch_fits
    host_fns = re.sub(
        r"int grid_size\(int size_m, bool c_fp32, int device\)\n\{\n(?:.*\n)*?\}\n",
        "",
        host_fns + "\n",
    )
    if (
        "grid_size" in host_fns
        or "int owner(" not in host_fns
        or "bool launch_fits(" not in host_fns
    ):
        fail("host helper block not as expected")
    return f"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
typedef uint16_t half;
struct float4 {{ float x, y, z, w; }};
static inline int min(int a, int b) {{ return a < b ? a : b; }}
{defs2001}
{strip(owner2001[0])}
{k(25)}
{args_struct}
{sel_fns}
{block(h_src, 19, 21)}
{host_fns}
static long bad = 0;
#define CHECK(c) do {{ if (!(c)) {{ if (++bad <= 20) fprintf(stderr, "CHECK failed \
line %d\\n", __LINE__); }} }} while (0)
static float* g_ws; static std::vector<int> g_wr; static size_t g_nf; static bool \
g_live;
static void __stcg(float* ptr, float) {{
    long off = ptr - g_ws;
    if (!g_live) return;
    CHECK(off >= 0 && (size_t) off < g_nf); if (off < 0 || (size_t) off >= g_nf) return;
    CHECK(g_wr[off] == 0); g_wr[off]++;
    CHECK(((size_t) off + 1) * sizeof(float) <= EXL3_M16G_SLOT_BYTES);
}}
static float4 __ldcg(const float4* p) {{
    long off = (const float*) p - g_ws;
    if (g_live) {{ CHECK(off >= 0 && (size_t) off + 4 <= g_nf);
        if (off >= 0 && (size_t) off + 4 <= g_nf) for (int e = 0; e < 4; ++e) \
CHECK(g_wr[off + e] == 1); }}
    return float4{{0, 0, 0, 0}};
}}
static std::string wstat(const std::vector<int>& v) {{
    if (v.empty()) return "first=- count=0 last=-";
    return "first=" + std::to_string(v[0]) + " count=" + std::to_string(v.size()) + " \
last=" + std::to_string(v.back());
}}
int main(int argc, char** argv) {{
    if (argc < 5) return 2;
    const int size_m = atoi(argv[1]), size_k = atoi(argv[2]), G = atoi(argv[3]);
    std::vector<int> ns; for (int a = 4; a < argc; ++a) ns.push_back(atoi(argv[a]));
    const int num_mats = (int) ns.size();
    // ---- host (H:203-248)
    Exl3M16GArgs args = {{}};
    {h(205)}
    for (int i = 0; i < num_mats; ++i) {{ const int n = ns[i];
        {h(224)}
        {h(225)}
    }}
    {h(227)}
    {h(239)}
    const bool fits = launch_fits(groups, KT, G);
    const bool shape = shape_ok(size_m, size_k, ns, 4, false, true);
    g_live = fits && shape;
    std::vector<uint16_t> xhbuf((size_t) num_mats * size_m * size_k + 1);
    half* xh = xhbuf.data();
    {h(243)}
    args.size_m = size_m; args.size_k = size_k; args.num_mats = num_mats;
    {h(248)}
    const Exl3M16GArgs p = args;
    // ---- device prelude (K:95-102 without b)
    const int KT_d = [&] {{ {k(95)} return KT; }}();
    const int groups_d = [&] {{ {k(96)} return groups; }}();
    const int total = groups_d * KT_d;
    {{ const int KT = KT_d, groups = groups_d; int t2 = [&] {{ {k(97)} return total; \
}}(); CHECK(t2 == total); }}
    constexpr int GW = EXL3_M16_GW;
    const int MT = size_m <= 8 ? 1 : 2;
    g_nf = (size_t) groups_d * p.max_contrib * EXL3_M16_SLOT_FLOATS;
    std::vector<float> wsbuf(g_nf + 16); g_ws = wsbuf.data(); args.ws = g_ws; \
g_wr.assign(g_nf, 0);
    Exl3M16GArgs pw = args; pw.ws = g_ws;
    const int jobs = [&] {{ const int groups = groups_d; {k(326)} {k(327)} return \
jobs; }}();
    const int tw = [&] {{ const Exl3M16GArgs& p = pw; {k(173)} {k(174)} return \
total_warps; }}();
    printf("S ns=");
    for (int i = 0; i < num_mats; ++i) printf(i ? ",%d" : "%d", ns[i]);
    printf(" k=%d G=%d m=%d groups=%d KT=%d total=%d mc=%d fits=%d shape=%d \
gbase=%d,%d,%d,%d,%d jobs=%d tw=%d\\n",
        size_k, G, size_m, groups_d, KT_d, total, p.max_contrib, fits ? 1 : 0, shape \
? 1 : 0,
        p.gbase[0], p.gbase[1], p.gbase[2], p.gbase[3], p.gbase[4], jobs, tw);
    // ---- M rows
    for (int g = 0; g < groups_d; ++g) {{
        const Exl3M16GArgs& p = pw; const int KT = KT_d;
        const int ch = 3, r = size_m - 1, lane = 31, warp = 7;
        {k(345)}
        {k(346)}
        {k(347)}
        {k(348)}
        const int lo = exl3_m16_owner(g * KT, total, G);
        {k(334)}
        const int hi = lo + nc - 1;
        long fl, ff;
        {{ const int b = hi; {k(239)} {k(240)}
           const int t = 3, mt = MT - 1, c = 3; (void) mt; const int row = r;
           fl = (slot + row * GW + t * 16 + (c >> 1) * 8) - g_ws; }}
        {{ {k(335)} const int c = nc - 1;
           ff = ((const float*) (((const float4*) (s0 + (size_t) c * \
EXL3_M16_SLOT_FLOATS)) + lane) - g_ws) + 3; }}
        const long base = (long) g * p.max_contrib;
        printf("M %d mat=%d gb=%d local=%d nmat=%d lo=%d hi=%d nc=%d base=%ld fl=%ld \
ff=%ld\\n",
            g, mat, gb, (col0 - ch * 128) / GW, n_mat, lo, hi, nc, base, fl, ff);
    }}
    // ---- B rows: main loop of every block (K:100-102, K:259-317), flush K:237-252
    std::vector<int> cover((size_t) total + 1, 0);
    for (int b = 0; b < G; ++b) {{
        const Exl3M16GArgs& p = pw; const int KT = KT_d;
        {k(100)}
        {k(101)}
        {k(102)}
        std::string segs; bool have = false, ok = true; int rg = 0, rf = 0, rl = 0; \
const char* sep = "";
        auto flush = [&] (int g) {{
            if (have && ok && rg == g) segs += std::string(sep) + std::to_string(g) + \
":" + std::to_string(rf) + "-" + std::to_string(rl);
            else if (have) segs += std::string(sep) + std::to_string(g) + ":?";
            else segs += std::string(sep) + std::to_string(g) + ":-";
            sep = ","; have = false; ok = true;
            for (int tid = 0; tid < EXL3_M16_THREADS; ++tid) {{
                const int warp = tid >> 5, lane = tid & 31;
                {k(239)}
                {k(240)}
                float acc[4][2][4] = {{}};
                for (int t = 0; t < EXL3_M16_TILES_W; ++t)
                    for (int mt = 0; mt < MT; ++mt)
                        for (int c = 0; c < 4; ++c) {{
                            {k(248)}
                            {k(249)}
                        }}
            }}
        }};
        int j = 0;
        {k(259)}
        {{
            {k(261)}
            {k(264)}
            {{
                {k(150)}
                {k(151)}
                {k(152)}
                if (g_live) {{
                    CHECK(g == g_cur && it >= 0 && it < total && kt >= 0 && kt < KT);
                    if (it >= 0 && it < total) cover[it]++;
                    const int mat = exl3_m16g_mat(p, g), gb = exl3_m16g_gbase(p, mat);
                    CHECK(mat < num_mats && gb <= g && (g - gb) < ns[mat] / GW);
                }}
                if (!have) {{ have = true; rg = g_cur; rf = it; rl = it; }}
                else {{ ok = ok && g_cur == rg && it == rl + 1; rl = it; }}
            }}
            {k(317)}
        }}
        if (have) segs += std::string(sep) + "~" + std::to_string(rf) + "-" + \
std::to_string(rl);
        printf("B %d s=%d e=%d segs=%s\\n", b, s_b, e_b, segs.c_str());
    }}
    if (g_live) for (int it = 0; it < total; ++it) CHECK(cover[it] == 1);
    if (g_live) for (size_t f = 0; f < g_nf; ++f) {{ (void) f; }}
    // ---- W + J rows: finisher (K:325-367)
    std::vector<std::vector<int>> outs(num_mats);
    for (int i = 0; i < num_mats; ++i) outs[i].assign((size_t) size_m * ns[i], 0);
    std::vector<int> jvis(jobs + 1, 0);
    for (int b = 0; b < G; ++b) for (int warp = 0; warp < 8; ++warp) {{
        const Exl3M16GArgs& p = pw; const int KT = KT_d; const int groups = groups_d;
        {k(326)}
        std::vector<int> seen;
        {k(328)}
        {{
            seen.push_back(w); jvis[w < jobs ? w : jobs]++;
            for (int lane = 0; lane < 32; ++lane) {{
                {k(330)}
                {k(331)}
                {k(332)}
                {k(333)}
                {k(334)}
                {k(335)}
                {k(336)}
                (void) v;
                {k(337)}
                {{
                    {k(339)}
                    (void) u;
                }}
                {k(345)}
                {k(346)}
                {k(347)}
                {k(348)}
                if (lane == 0 && g_live) {{
                    CHECK(mat < num_mats && r < size_m && n_mat == ns[mat]);
                    const size_t o = (size_t) r * n_mat + col0;
                    CHECK(o + 128 <= outs[mat].size());
                    if (o + 128 <= outs[mat].size()) for (int e = 0; e < 128; ++e) \
outs[mat][o + e]++;
                }}
            }}
        }}
        printf("W %d %d %s\\n", b, warp, wstat(seen).c_str());
    }}
    if (g_live) {{
        for (int x = 0; x < jobs; ++x) CHECK(jvis[x] == 1);
        CHECK(jvis[jobs] == 0);
        for (int i = 0; i < num_mats; ++i) for (int v : outs[i]) CHECK(v == 1);
    }}
    for (int w = 0; w < jobs; ++w) {{
        const Exl3M16GArgs& p = pw; const int KT = KT_d; (void) KT;
        {k(326)}
        {k(330)}
        {k(331)}
        {k(332)}
        {k(333)}
        {k(345)}
        {k(346)}
        {k(347)}
        {k(348)}
        const size_t out = (size_t) r * n_mat + col0;
        printf("J %d g=%d r=%d ch=%d mat=%d col0=%d out=%zu\\n", w, g, r, ch, mat, \
col0, out);
    }}
    // ---- Q + P rows: prologue (K:169-189), xh[i] = xh + i * size_m * size_k (H:243)
    std::vector<int> xcov((size_t) num_mats * size_m * size_k, 0);
    for (int b = 0; b < G; ++b) for (int warp = 0; warp < 8; ++warp) {{
        const Exl3M16GArgs& p = pw;
        {k(173)}
        {k(174)}
        {k(175)}
        std::vector<int> seen;
        {k(176)}
        {{
            seen.push_back(w);
            {k(178)}
            {k(179)}
            const half* A = nullptr; (void) A;
            const size_t xo = (exl3_m16g_sel(p.xh, mat) + r * 128) - xh;
            if (g_live) {{ CHECK(xo + 128 <= xcov.size()); if (xo + 128 <= \
xcov.size()) for (int e = 0; e < 128; ++e) xcov[xo + e]++; }}
        }}
        printf("Q %d %d %s\\n", b, warp, wstat(seen).c_str());
    }}
    if (g_live) for (int v : xcov) CHECK(v == 1);
    {{
        const Exl3M16GArgs& p = pw;
        {k(173)}
        {k(174)}
        for (int w = 0; w < total_warps; ++w) {{
            {k(178)}
            {k(179)}
            const size_t xo = (exl3_m16g_sel(p.xh, mat) + r * 128) - xh;
            const size_t ao = (size_t) (r * 128);
            const size_t so = (size_t) ((r * 128) % size_k);
            printf("P %d mat=%d r=%d xh=%zu a=%zu suh=%zu\\n", w, mat, r, xo, ao, so);
        }}
    }}
    // ---- I rows: weight issue stream (K:107-135) of (warp, lane) in {{(0,0), \
(7,31)}}; B[i] are
    // disjoint regions of one buffer (KT rows of n_i / 16 * 128 bytes each, 1 MB gaps)
    std::vector<size_t> boff(num_mats + 1, 0);
    for (int i = 0; i < num_mats; ++i) boff[i + 1] = boff[i] + (size_t) KT_d * (ns[i] \
/ 16) * 128 + (1u << 20);
    std::vector<uint8_t> bbuf(boff[num_mats] + 16);
    for (int i = 0; i < num_mats; ++i) pw.B[i] = (const uint16_t*) (bbuf.data() + \
boff[i]);
    auto locate = [&] (const uint8_t* q, int& mi, long& off) {{
        long a = q - bbuf.data(); mi = 0;
        for (int i = 1; i < num_mats; ++i) if (a >= (long) boff[i]) mi = i;
        off = a - (long) boff[mi];
    }};
    uint8_t ringbuf[64];
    for (int b = 0; b < G; ++b) for (int sel = 0; sel < 2; ++sel) {{
        const Exl3M16GArgs& p = pw; const int KT = KT_d;
        const int warp = sel ? 7 : 0, lane = sel ? 31 : 0;
        const int W_STAGE = 0;
        {k(100)}
        {k(101)}
        {k(102)}
        std::vector<const uint8_t*> streamed;
        auto cp_async_stream = [&] (uint8_t*, const uint8_t* src) {{ \
streamed.push_back(src); }};
{block(k_src, 109, 121)}
        uint8_t* ring_lane = ringbuf;
{block(k_src, 124, 135)}
        while (iss_j < n_iter) issue_w(0);
        unsigned long long cs = 0;
        for (size_t j = 0; j < streamed.size(); ++j) {{
            int mi; long off; locate(streamed[j], mi, off);
            cs += ((unsigned long long) mi * 1000003ull + (unsigned long long) off) * \
(unsigned long long) (j + 1);
            if (g_live) {{
                const int it = s_b + (int) j, g = it / KT, kt = it - g * KT;
                const int mat = exl3_m16g_mat(p, g), gb = exl3_m16g_gbase(p, mat);
                const long rbytes = (long) (ns[mat] / 16) * 128;
                CHECK(mi == mat && off >= 0 && off + 16 <= (long) KT * rbytes);
                CHECK(off == ((long) (g - gb) * 32 + warp * 4) * 128 + lane * 16 + \
(long) kt * rbytes);
            }}
        }}
        int mi; long off; locate(iss_ptr, mi, off);
        printf("I %d %d %d cs=%llu end=%d,%d,%d,%ld,%zu\\n", b, warp, lane, cs, \
iss_g, iss_kt, mi, off, row_bytes);
    }}
    fprintf(stderr, "checks %s: violations %ld\\n", g_live ? "live" : "skipped (not \
launchable)", bad);
    return bad ? 3 : 0;
}}
"""


def configs(seed: int = 2102) -> list[tuple[int, int, int, list[int]]]:
    """Return the served bundles plus seeded random and edge configurations.

    Args:
        seed: Random seed.

    Returns:
        (m, k, G, ns) tuples.

    """
    out = [
        (m, 5120, g_count, ns)
        for ns in ([10240, 6144], [12288, 1024, 1024])
        for g_count in (82, 164)
        for m in range(1, 17)
    ]
    rng = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  seeded RNG generates reproducible test configurations
    for _ in range(40):
        nm = rng.randint(1, 4)
        ns = [512 * rng.randint(1, 12) for _ in range(nm)]
        k = 128 * rng.randint(1, 24)
        g_count = rng.choice([1, 2, 3, 7, 82, 164, rng.randint(1, 200)])
        out.append((rng.randint(1, 16), k, g_count, ns))
    out += [
        (1, 128, 1, [512]),
        (16, 128, 3, [512, 512, 512, 512]),
        (5, 256, 300, [512, 1024]),
    ]
    return out


def load_sources(argv: list[str], mutate: str | None) -> tuple[str, str, str]:
    """Read the patches, verify quoted lines and apply the mutation.

    Args:
        argv: Positional arguments (program, PATCH_2102[, PATCH_2001]).
        mutate: Mutation name or None.

    Returns:
        Kernel text, host text and the 2001 header text.

    """
    p2102 = Path(argv[1]).read_bytes()
    sha = hashlib.sha256(p2102).hexdigest()
    p2001 = Path(
        argv[2]
        if len(argv) == MAX_ARGS
        else REPO / "patches/exl3-ext/2001-gemm-m16-swapab.patch"
    ).read_text(encoding="utf-8")
    files = new_files(p2102.decode())
    k_txt = files["exllamav3_ext/quant/exl3_gemm_m16g_kernel.cuh"]
    h_txt = files["exllamav3_ext/quant/exl3_gemm_m16g.cu"]
    v1 = new_files(p2001)["exllamav3_ext/quant/exl3_gemm_m16_kernel.cuh"]
    check_lines("K", k_txt, K_LINES)
    check_lines("H", h_txt, H_LINES)
    pin = "pinned" if sha == PATCH_SHA else "NOT the pinned c12f897d..."
    sys.stdout.write(
        f"patch sha256 {sha} ({pin}); "
        f"quoted lines verified: K {len(K_LINES)}, H {len(H_LINES)}\n"
    )
    if mutate:
        a, b = MUTATIONS[mutate]
        if (k_txt.count(a) + h_txt.count(a)) != 1:
            fail(f"mutation {mutate} does not apply once")
        k_txt, h_txt = k_txt.replace(a, b), h_txt.replace(a, b)
    return k_txt, h_txt, v1


def build(td: Path, program: str) -> tuple[Path, Path]:
    """Compile the C++ program and the Bend table.

    Args:
        td: Scratch directory.
        program: C++ source.

    Returns:
        Paths of the C executable and the Bend table executable.

    """
    c = td / "diff.cpp"
    c.write_text(program)
    exe = td / "diff"
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
    tab = td / "table"
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        source_link.locked([source_link.bend(), TABLE, "-o", str(tab)]),
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return exe, tab


def report_mismatch(cfg: str, cout: str, bout: str) -> None:
    """Print the first differing row of the C and Bend tables.

    Args:
        cfg: Configuration label.
        cout: C table output.
        bout: Bend table output.

    """
    cl, bl = cout.splitlines(), bout.splitlines()
    i = next(
        (i for i in range(min(len(cl), len(bl))) if cl[i] != bl[i]),
        min(len(cl), len(bl)),
    )
    c_row = cl[i] if i < len(cl) else "<eof>"
    b_row = bl[i] if i < len(bl) else "<eof>"
    sys.stdout.write(f"MISMATCH {cfg} at row {i}:\n  C:    {c_row}\n  Bend: {b_row}\n")


def run_configs(
    exe: Path, tab: Path, cfgs: list[tuple[int, int, int, list[int]]]
) -> tuple[int, int, int, int]:
    """Run both tables on every configuration and compare.

    Args:
        exe: C executable.
        tab: Bend table executable.
        cfgs: Configurations.

    Returns:
        Rows, C-side violation configs, mismatching configs, live-check configs.

    """
    rows = bad = mism = live = 0
    for m, k, g_count, ns in cfgs:
        args = [str(m), str(k), str(g_count), *[str(n) for n in ns]]
        cfg = f"m={m} k={k} G={g_count} ns={ns}"
        cres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(exe), *args], capture_output=True, text=True, check=False
        )
        bres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(tab), *args],
            capture_output=True,
            text=True,
            check=True,
            preexec_fn=unlimited_vm,
        )
        bout = "\n".join(line for line in bres.stdout.split("\n") if line) + "\n"
        rows += len(cres.stdout.splitlines())
        live += "checks live" in cres.stderr
        if cres.returncode != 0:
            bad += 1
            sys.stdout.write(f"C-side violation {cfg}: {cres.stderr.strip()}\n")
        if cres.stdout != bout:
            mism += 1
            if mism <= MAX_MISMATCH_REPORTS:
                report_mismatch(cfg, cres.stdout, bout)
    return rows, bad, mism, live


def main(argv: list[str]) -> None:
    """Run the differential check.

    Args:
        argv: Command line.

    """
    mutate = None
    if len(argv) >= MUTATE_ARGS and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0], *argv[3:]]
    if len(argv) not in {2, 3}:
        fail(USAGE)
    k_txt, h_txt, v1 = load_sources(argv, mutate)
    program = c_program(k_txt.split("\n"), h_txt.split("\n"), v1)
    cfgs = configs()
    with tempfile.TemporaryDirectory() as td:
        exe, tab = build(Path(td), program)
        rows, bad, mism, live = run_configs(exe, tab, cfgs)
    sys.stdout.write(
        f"configs {len(cfgs)} (served 64 + random/edge {len(cfgs) - 64}), "
        f"rows {rows}, byte-identical configs {len(cfgs) - mism}, "
        f"C-side checks live in {live} launchable configs, "
        f"violations in {bad} configs\n"
    )
    if mism or bad:
        fail("MISMATCH" if mism else "C-side coverage violation")
    sys.stdout.write("gemm_m16_group_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
