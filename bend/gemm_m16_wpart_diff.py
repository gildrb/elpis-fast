#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential check of bend/gemm_m16_wpart.bend against the engine C.

See USAGE for the checked blocks, the C-side checks and the command line.
"""

from __future__ import annotations

import hashlib
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

HERE = Path(__file__).resolve().parent
REPO = source_link.REPO
PATCH = REPO / "patches/exl3-ext/2105-proj-m16g-weighted-on9003b.patch"
TABLE = "GEMM_M16_WPART_TABLE.bend"
IMPL = "gemm_m16_wpart.bend"
# the Bend sources compile from the dev dir (repo bend/*.bend + links to ours) when
# present
BEND_CWD = HERE.parent / "dev" if (HERE.parent / "dev" / TABLE).exists() else HERE
# str.split(sep, 1) parts when sep is present; argv length of a flag with its value
SPLIT_PARTS = 2
FLAG_WITH_VALUE = 2
# mismatching table rows shown before giving up
MAX_MISMATCHES = 3
USAGE = (
    "Finite differential check of bend/gemm_m16_wpart.bend (slot-weighted "
    "split-K partition, ext patch\n"
    "2105) against the patched engine tree's own C, compiled for the CPU.\n"
    "\n"
    "From TREE (exllamav3_ext/quant/): exl3_gemm_m16g.cu = H (struct Weights, "
    "wpre, wstart, owner,\n"
    "weights_valid, max_contrib, launch_fits, struct WeightRow, g_weights, "
    "launch_weights,\n"
    "EXL3_M16G_SLOT_BYTES), exl3_gemm_m16g_kernel.cuh = K (exl3_m16g_wpre / "
    "_start / _owner, the finisher's\n"
    "nc statement, EXL3_M16G_MAX_MATS), exl3_gemm_m16_kernel.cuh = V2 "
    "(EXL3_M16_* constants, exl3_m16_owner,\n"
    "the s_b statement). Every block is located by its signature and brace "
    "matching, quoted verbatim\n"
    "(CUDA qualifiers and min/max are mapped by macros / using-declarations, "
    "not by editing the text), and\n"
    "every quoted non-blank line of the 2105 blocks must be a `+` line of the "
    "2105 patch. weights_enabled()\n"
    "is stubbed to true (its body reads the environment). The tree's g_weights "
    "rows and draft g_routes rows\n"
    "must equal the configuration list of bend/GEMM_M16_WPART_TABLE.bend.\n"
    "\n"
    "The C++ program prints the Bend table (weights from launch_weights: the "
    "g_weights row / (1, 1) for the\n"
    "routed bundles, the (1, 1) override for the forced rows; starts, owners "
    "and nc from the kernel's\n"
    "functions; valid, mc and fits from the host's), compared byte for byte, "
    "and runs independent C-side\n"
    "checks: host == kernel; owner == brute max{b < G : start(b) <= x}; every "
    "iteration in exactly one\n"
    "slice; max_contrib == brute contributor count; seeded random weighted "
    "configurations (weights_valid\n"
    "== domain && brute non-empty, owner, exactly once); (w, w) == 2001 v2 "
    "(s_b, exl3_m16_owner); a\n"
    "configuration with an empty slice is rejected; owner_le equivalence "
    "(exhaustive on a bounded domain).\n"
    "Differential evidence on finite instances, not a proof.\n"
    "\n"
    "Usage: python3 bend/gemm_m16_wpart_diff.py [--mutate NAME | "
    "--all-mutations] TREE\n"
    "  TREE: OUT/patched of bend/engine_trees.py.\n"
)

H_PATH = "exllamav3_ext/quant/exl3_gemm_m16g.cu"
K_PATH = "exllamav3_ext/quant/exl3_gemm_m16g_kernel.cuh"
V2_PATH = "exllamav3_ext/quant/exl3_gemm_m16_kernel.cuh"

G_SERVED, NSM_SERVED = 164, 82
# (name, k, widths, w0, w1, override): the table's configuration list. override =
# launch_weights' ow0 / ow1 (0 = the g_weights row, else (1, 1))
CONFIGS = [
    ("gdn", 5120, [10240, 6144], 100, 91, 0),
    ("gdn", 5120, [10240, 6144], 1, 1, 1),
    ("attn", 5120, [12288, 1024, 1024], 100, 91, 0),
    ("attn", 5120, [12288, 1024, 1024], 1, 1, 1),
    ("draft_qkv", 5120, [4096, 1024, 1024], 1, 1, 0),
    ("draft_kv4", 5120, [1024, 1024, 1024, 1024], 1, 1, 0),
    ("draft_kv2", 5120, [1024, 1024], 1, 1, 0),
    ("draft_fc", 25600, [5120], 1, 1, 0),
]
BEND_NAMES = {
    "Gdn": "gdn",
    "Attn": "attn",
    "DraftQkv": "draft_qkv",
    "DraftKv4": "draft_kv4",
    "DraftKv2": "draft_kv2",
    "DraftFc": "draft_fc",
}

# (file key, name, signature prefix of the first stripped line, kind): "brace" runs
# to the matching closing brace, "stmt" to the first line ending in ';', "line" is
# one line. new = 2105 block (every non-blank line must be a `+` line of the patch);
# mod = 2105-modified block (every line a `+` or context line, at least one `+`);
# old = pre-2105 source (not checked against the patch)
BLOCKS = [
    ("V2", "EXL3_M16_WARPS", "#define EXL3_M16_WARPS ", "line", "old"),
    ("V2", "EXL3_M16_TILES_W", "#define EXL3_M16_TILES_W ", "line", "old"),
    ("V2", "EXL3_M16_GW", "#define EXL3_M16_GW ", "line", "old"),
    ("V2", "EXL3_M16_SLOT_FLOATS", "#define EXL3_M16_SLOT_FLOATS ", "line", "old"),
    (
        "V2",
        "exl3_m16_owner",
        "__device__ __forceinline__ int exl3_m16_owner(",
        "brace",
        "old",
    ),
    ("V2", "s_b", "const int s_b = (int) (((int64_t) b * total) / G);", "stmt", "old"),
    ("K", "EXL3_M16G_MAX_MATS", "#define EXL3_M16G_MAX_MATS ", "line", "old"),
    (
        "K",
        "exl3_m16g_wpre",
        "__device__ __forceinline__ int64_t exl3_m16g_wpre(",
        "brace",
        "new",
    ),
    (
        "K",
        "exl3_m16g_start",
        "__device__ __forceinline__ int exl3_m16g_start(",
        "brace",
        "new",
    ),
    (
        "K",
        "exl3_m16g_owner",
        "__device__ __forceinline__ int exl3_m16g_owner(",
        "brace",
        "new",
    ),
    ("K", "nc", "int nc = exl3_m16g_owner((g + 1) * KT - 1,", "stmt", "new"),
    ("H", "EXL3_M16G_SLOT_BYTES", "#define EXL3_M16G_SLOT_BYTES ", "line", "old"),
    ("H", "Weights", "struct Weights", "brace", "new"),
    ("H", "wpre", "int64_t wpre(int b, const Weights& w)", "brace", "new"),
    (
        "H",
        "wstart",
        "int wstart(int b, int total, int G, const Weights& w)",
        "brace",
        "new",
    ),
    (
        "H",
        "owner",
        "int owner(int x, int total, int G, const Weights& w)",
        "brace",
        "new",
    ),
    (
        "H",
        "weights_valid",
        "bool weights_valid(int total, int G, const Weights& w)",
        "brace",
        "new",
    ),
    (
        "H",
        "max_contrib",
        "int max_contrib(int groups, int KT, int G, const Weights& w)",
        "brace",
        "mod",
    ),
    (
        "H",
        "launch_fits",
        "bool launch_fits(int groups, int KT, int G, const Weights& w)",
        "brace",
        "mod",
    ),
    ("H", "WeightRow", "struct WeightRow", "brace", "new"),
    ("H", "g_weights", "const WeightRow g_weights[] =", "brace", "new"),
    ("H", "launch_weights", "Weights launch_weights(", "brace", "new"),
]
K_FUNCS = ("exl3_m16g_wpre", "exl3_m16g_start", "exl3_m16g_owner")
H_ORDER = (
    "Weights",
    "wpre",
    "wstart",
    "owner",
    "weights_valid",
    "max_contrib",
    "launch_fits",
    "WeightRow",
    "g_weights",
)

# name -> (quoted block, text, replacement); each must occur exactly once in its block
MUTATIONS = {
    # owner's segment test inclusive: at P == w0 * nsm both branches return nsm
    # (equivalent mutant)
    "owner_le": (
        "exl3_m16g_owner",
        "if (P < (int64_t) w0 * nsm)",
        "if (P <= (int64_t) w0 * nsm)",
    ),
    # owner's segment threshold from the second weight
    "owner_w1": (
        "exl3_m16g_owner",
        "if (P < (int64_t) w0 * nsm)",
        "if (P < (int64_t) w1 * nsm)",
    ),
    # weight prefix counts nsm + 1 first-segment blocks
    "wpre_nsm1": ("exl3_m16g_wpre", "min(b, nsm)", "min(b, nsm + 1)"),
    # slice start rounded up
    "start_ceil": (
        "exl3_m16g_start",
        (
            "(((int64_t) total * exl3_m16g_wpre(b, nsm, w0, w1))"
            " / exl3_m16g_wpre(G, nsm, w0, w1))"
        ),
        (
            "(((int64_t) total * exl3_m16g_wpre(b, nsm, w0, w1)"
            " + exl3_m16g_wpre(G, nsm, w0, w1) - 1)"
            " / exl3_m16g_wpre(G, nsm, w0, w1))"
        ),
    ),
    # one partial slot too many per group
    "mc_off": (
        "max_contrib",
        "owner(g * KT, total, G, w) + 1;",
        "owner(g * KT, total, G, w) + 2;",
    ),
    # GDN row weight w1 changed
    "gdn_w1": (
        "g_weights",
        "{ 10240, 6144 },       100, 91 }",
        "{ 10240, 6144 },       100, 190 }",
    ),
}
EQUIVALENT = {"owner_le"}


type WeightRows = list[tuple[int, list[int], int, int]]
type RouteRows = list[tuple[int, list[int]]]
type BendConfig = tuple[str, int, list[int], int, int]


def say(text: str) -> None:
    """Write one line to stdout."""
    sys.stdout.write(f"{text}\n")


def unlimited_vm() -> None:
    """Lift a soft RLIMIT_AS.

    The Bend runtime reserves its heap up front (harness shells set 8 GB).
    """
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def fail(msg: str) -> NoReturn:
    """Exit with a FAIL message.

    Args:
        msg: Failure description.

    Raises:
        SystemExit: Always, carrying the message.

    """
    text = f"gemm_m16_wpart_diff: FAIL {msg}"
    raise SystemExit(text)


def brace_end(src: list[str], i: int) -> int | None:
    """Find the 0-based line closing the brace block opened from line i.

    Args:
        src: Source lines.
        i: 0-based first line of the block.

    Returns:
        The 0-based closing line, or None when the braces are unbalanced.

    """
    depth, opened = 0, False
    for j in range(i, len(src)):
        code = src[j].split("//")[0]
        for ch in code:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return j
        if depth < 0:
            break
    return None


def extract(src: list[str], name: str, sig: str, kind: str) -> tuple[int, int]:
    """Locate the block whose first stripped line starts with sig.

    Args:
        src: Source lines.
        name: Block name for messages.
        sig: Signature prefix.
        kind: "line", "stmt" or "brace".

    Returns:
        The 1-based inclusive line range.

    """
    hits = [i for i, line in enumerate(src) if line.strip().startswith(sig)]
    if len(hits) != 1:
        fail(f"{name}: signature {sig!r} found {len(hits)} times")
    i = hits[0]
    if kind == "line":
        return i + 1, i + 1
    if kind == "stmt":
        j = i
        while not src[j].split("//")[0].rstrip().endswith(";"):
            j += 1
        return i + 1, j + 1
    j = brace_end(src, i)
    if j is None:
        fail(f"{name}: unbalanced braces from line {i + 1}")
    return i + 1, j + 1


def patch_lines(patch: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Collect, per file, the `+` lines and context lines (without their prefix).

    Args:
        patch: Unified diff text.

    Returns:
        The `+` line sets and the context line sets, keyed by file.

    """
    plus: dict[str, set[str]] = {}
    ctx: dict[str, set[str]] = {}
    cur = None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            cur = line[6:]
            plus.setdefault(cur, set())
            ctx.setdefault(cur, set())
        elif line.startswith("--- "):
            continue
        elif cur and line.startswith("+"):
            plus[cur].add(line[1:])
        elif cur and line.startswith(" "):
            ctx[cur].add(line[1:])
    return plus, ctx


def ints(s: str) -> list[int]:
    """Parse a comma-separated integer list.

    Args:
        s: The list text.

    Returns:
        The integers.

    """
    return [int(n) for n in s.replace(" ", "").split(",") if n]


def tree_rows(h: str) -> tuple[WeightRows, RouteRows, RouteRows]:
    """Parse the g_weights rows and the g_routes rows.

    Args:
        h: The host source text.

    Returns:
        g_weights rows (k, ns, w0, w1); g_routes rows (k, ns) with draft flag true;
        g_routes rows with draft flag false.

    """
    wbody = h.split("const WeightRow g_weights[] =", 1)
    rbody = h.split("const RouteRow g_routes[] =", 1)
    if len(wbody) != SPLIT_PARTS or len(rbody) != SPLIT_PARTS:
        fail("g_weights / g_routes not found in H")
    wt = wbody[1].split("};", 1)[0]
    rt = rbody[1].split("};", 1)[0]
    weights = [
        (int(k), ints(ns), int(a), int(b))
        for k, ns, a, b in re.findall(
            r"\{\s*(\d+),\s*\{\s*([\d,\s]+)\},\s*(\d+),\s*(\d+)\s*\}", wt
        )
    ]
    n_rows = len(re.findall(r"^\s*\{\s*\d", wt, re.MULTILINE))
    if len(weights) != n_rows:
        fail(f"parsed {len(weights)} of {n_rows} g_weights rows")
    row_re = (
        r"\{\s*M16G_\w+,\s*(\d+),\s*\{\s*([\d,\s]+)\},\s*\d+,\s*\d+,"
        r"\s*M16G_\w+,\s*(true|false)\s*\}"
    )
    routes = re.findall(row_re, rt)
    n_routes = len(re.findall(r"^\s*\{\s*M16G_", rt, re.MULTILINE))
    if len(routes) != n_routes:
        fail(f"parsed {len(routes)} of {n_routes} g_routes rows")
    draft = [(int(k), ints(ns)) for k, ns, f in routes if f == "true"]
    main = [(int(k), ints(ns)) for k, ns, f in routes if f == "false"]
    return weights, draft, main


def bend_configs(table: str, impl: str) -> list[BendConfig]:
    """Parse the table(...) calls of GEMM_M16_WPART_TABLE.bend's main.

    Args:
        table: The table source text.
        impl: The Impl source text, resolving the served weights.

    Returns:
        The configurations (name, k, widths, w0, w1).

    """
    served = {}
    for f in ("w0_served", "w1_served"):
        m = re.search(rf"def {f}\(\) -> Nat:\n\s+(\d+)n\b", impl)
        if not m:
            fail(f"Impl.{f} not a literal")
        served[f"Impl.{f}()"] = int(m.group(1))
    if not re.search(r"def nsm_served\(\) -> Nat:\n\s+82n\n", impl) or not re.search(
        r"def grid_served\(\) -> Nat:\n\s+Nat\.mul\(82n, 2n\)\n", impl
    ):
        fail("Impl.nsm_served / grid_served are not 82 / 82 * 2")

    def wv(s: str) -> int:
        return served[s] if s in served else int(s.rstrip("n"))

    calls = re.findall(
        r"^\s+table\((\w+)\{\}, (\d+)n, \[([\dn, ]+)\], ([\w.()]+), ([\w.()]+)\)$",
        table,
        re.MULTILINE,
    )
    return [
        (
            BEND_NAMES[n],
            int(k),
            [int(x.strip().rstrip("n")) for x in ns.split(",")],
            wv(a),
            wv(b),
        )
        for n, k, ns, a, b in calls
    ]


def c_program(q: dict[str, str], q_orig: dict[str, str]) -> str:
    """Build the C++ table printer and checker.

    Args:
        q: Quoted blocks (possibly mutated).
        q_orig: Unmutated quoted blocks.

    Returns:
        The C++ source.

    """
    k_funcs = "\n\n".join(q[n] for n in K_FUNCS)
    k_orig = "\n\n".join(q_orig[n] for n in K_FUNCS)
    a, b = MUTATIONS["owner_le"][1:]
    k_le = k_orig.replace(a, b)
    h_blocks = "\n\n".join(q[n] for n in H_ORDER)
    cfgs = ",\n".join(
        f'    {{ "{n}", {k}, {{ {", ".join(map(str, ns))} }}, {ov}, {w0}, {w1} }}'
        for n, k, ns, w0, w1, ov in CONFIGS
    )
    return f"""// generated by gemm_m16_wpart_diff.py: quoted tree C + table printer + \
independent checks
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <string>

// CUDA qualifiers and device min / max for the quoted kernel functions
#define __device__
#define __forceinline__ inline
using std::min;
using std::max;

{q["EXL3_M16_WARPS"]}
{q["EXL3_M16_TILES_W"]}
{q["EXL3_M16_GW"]}
{q["EXL3_M16_SLOT_FLOATS"]}
{q["EXL3_M16G_SLOT_BYTES"]}
{q["EXL3_M16G_MAX_MATS"]}

namespace v2 {{
{q["exl3_m16_owner"]}

int s_b_of(int b, int total, int G)
{{
{q["s_b"]}
    return s_b;
}}
}}

struct PArgs {{ int nsm; int w0; int w1; }};

namespace kq {{
{k_funcs}

// the finisher's nc statement; p stands for the kernel's Exl3M16GArgs (nsm, w0, w1)
int nc_of(int g, int KT, int total, int G, const PArgs& p)
{{
{q["nc"]}
    return nc;
}}
}}

namespace hq {{
{h_blocks}

// stub: the tree's weights_enabled() reads EXL3_M16G_2105 from the environment
bool weights_enabled() {{ return true; }}

{q["launch_weights"]}
}}

// owner_le equivalence evidence: the unmutated kernel functions and the owner_le \
variant
namespace eq_orig {{
{k_orig}
}}
namespace eq_le {{
{k_le}
}}

struct Cfg {{ const char* name; int k; std::vector<int> ns; int ov; int w0; int w1; }};
static const Cfg CFGS[] = {{
{cfgs}
}};

static long long n_fail = 0;
static void failf(const char* what, const std::string& detail)
{{
    if (n_fail++ < 20) fprintf(stderr, "FAIL %s: %s\\n", what, detail.c_str());
}}
static std::string fmt(const char* f, long long a = 0, long long b = 0, \
long long c = 0, long long d = 0,
                       long long e = 0, long long g = 0, long long h = 0)
{{
    char buf[256];
    snprintf(buf, sizeof buf, f, a, b, c, d, e, g, h);
    return buf;
}}

// independent weight prefix and slice start: sum of per-block weights
static int64_t brute_W(int b, int nsm, int w0, int w1)
{{
    int64_t s = 0;
    for (int i = 0; i < b; ++i) s += i < nsm ? w0 : w1;
    return s;
}}
static std::vector<int> brute_starts(int total, int G, int nsm, int w0, int w1)
{{
    std::vector<int> s(G + 1);
    const int64_t WG = brute_W(G, nsm, w0, w1);
    int64_t W = 0;
    for (int b = 0; b <= G; ++b)
    {{
        s[b] = (int) ((int64_t) total * W / WG);
        W += b < nsm ? w0 : w1;
    }}
    return s;
}}
static std::vector<int> kernel_starts(int total, int G, int nsm, int w0, int w1)
{{
    std::vector<int> s(G + 1);
    for (int b = 0; b <= G; ++b) s[b] = kq::exl3_m16g_start(b, total, G, nsm, w0, w1);
    return s;
}}

// owner == max{{b < G : start(b) <= x}} (sweep over the kernel's starts) and every \
x < total in
// exactly one [start(b), start(b + 1)), b < G. Returns violations.
static long long owner_once(int total, int G, int nsm, int w0, int w1, \
const std::vector<int>& s,
                            long long& n_owner, const char* tag)
{{
    long long bad = 0;
    int bm = 0;
    for (int x = 0; x < total; ++x)
    {{
        while (bm + 1 < G && s[bm + 1] <= x) ++bm;
        const int o = kq::exl3_m16g_owner(x, total, G, nsm, w0, w1);
        ++n_owner;
        if (o != bm || s[bm] > x)
        {{
            ++bad;
            failf(tag, fmt("owner x=%lld total=%lld G=%lld nsm=%lld w=%lld,%lld \
got %lld", x, total, G, nsm,
                           w0, w1, o) + " brute " + std::to_string(bm));
        }}
    }}
    std::vector<int> cnt(total, 0);
    long long outside = 0;
    for (int b = 0; b < G; ++b)
        for (int x = s[b]; x < s[b + 1]; ++x)
        {{
            if (x < 0 || x >= total) ++outside;
            else ++cnt[x];
        }}
    if (s[0] != 0 || s[G] != total || outside)
    {{
        ++bad;
        failf(tag, fmt("exactly-once ends total=%lld G=%lld start0=%lld \
startG=%lld outside=%lld", total, G,
                       s[0], s[G], outside));
    }}
    for (int x = 0; x < total; ++x)
        if (cnt[x] != 1)
        {{
            ++bad;
            failf(tag, fmt("exactly-once x=%lld covered %lld times \
(total=%lld G=%lld)", x, cnt[x], total, G));
        }}
    return bad;
}}

int main()
{{
    const int G = {G_SERVED}, nsm = {NSM_SERVED};
    long long hk_starts = 0, hk_owners = 0, hk_bad = 0, t_owner = 0, t_bad = 0, \
mc_groups = 0, mc_bad = 0;
    long long eq_evals = 0, eq_hits = 0, eq_diff = 0;
    std::vector<std::string> summary;

    // ---- the table
    for (const Cfg& c : CFGS)
    {{
        int groups = 0;
        for (int n : c.ns) groups += n / EXL3_M16_GW;
        const int KT = c.k / 16;
        const int total = groups * KT;
        const hq::Weights w = hq::launch_weights(c.k, c.ns, total, G, nsm, c.ov, c.ov);
        const bool valid = hq::weights_valid(total, G, w);
        const int mc = hq::max_contrib(groups, KT, G, w);
        const uint64_t ws = (uint64_t) groups * mc * EXL3_M16_SLOT_FLOATS * \
sizeof(float);
        const bool fits = hq::launch_fits(groups, KT, G, w);
        std::string ns;
        for (size_t i = 0; i < c.ns.size(); ++i) ns += (i ? "," : "") + \
std::to_string(c.ns[i]);
        printf("C %s k=%d ns=%s G=%d nsm=%d w=%d,%d groups=%d KT=%d total=%d \
WG=%lld valid=%d mc=%d ws=%llu fits=%d\\n",
               c.name, c.k, ns.c_str(), G, w.nsm, w.w0, w.w1, groups, KT, total, \
(long long) hq::wpre(G, w),
               (int) valid, mc, (unsigned long long) ws, (int) fits);
        printf("S %s %d,%d", c.name, w.w0, w.w1);
        for (int b = 0; b <= G; ++b) printf(" %d", kq::exl3_m16g_start(b, total, \
G, w.nsm, w.w0, w.w1));
        printf("\\n");
        const PArgs p = {{ w.nsm, w.w0, w.w1 }};
        for (int g = 0; g < groups; ++g)
            printf("g %s %d,%d %d lo=%d hi=%d nc=%d\\n", c.name, w.w0, w.w1, g,
                   kq::exl3_m16g_owner(g * KT, total, G, w.nsm, w.w0, w.w1),
                   kq::exl3_m16g_owner((g + 1) * KT - 1, total, G, w.nsm, w.w0, w.w1),
                   kq::nc_of(g, KT, total, G, p));

        // host wstart / owner == kernel start / owner
        for (int b = 0; b <= G; ++b, ++hk_starts)
            if (hq::wstart(b, total, G, w) != kq::exl3_m16g_start(b, total, G, \
w.nsm, w.w0, w.w1))
            {{ ++hk_bad; failf("host==kernel", fmt("start b=%lld", b) + " (" + \
c.name + ")"); }}
        for (int x = 0; x < total; ++x, ++hk_owners)
            if (hq::owner(x, total, G, w) != kq::exl3_m16g_owner(x, total, G, \
w.nsm, w.w0, w.w1))
            {{ ++hk_bad; failf("host==kernel", fmt("owner x=%lld", x) + " (" + \
c.name + ")"); }}
        // owner == brute, exactly once
        const std::vector<int> s = kernel_starts(total, G, w.nsm, w.w0, w.w1);
        t_bad += owner_once(total, G, w.nsm, w.w0, w.w1, s, t_owner, \
"table owner/once");
        // max_contrib == brute contributors, every contributor's slot inside the \
group's mc
        if (valid)
        {{
            const std::vector<int> bs = brute_starts(total, G, w.nsm, w.w0, w.w1);
            int bmc = 0;
            for (int g = 0; g < groups; ++g, ++mc_groups)
            {{
                int cnt = 0;
                const int lo = kq::exl3_m16g_owner(g * KT, total, G, w.nsm, w.w0, w.w1);
                for (int b = 0; b < G; ++b)
                    if (bs[b] < bs[b + 1] && bs[b] < (g + 1) * KT && bs[b + 1] > g * KT)
                    {{
                        ++cnt;
                        if (b - lo < 0 || b - lo >= mc)
                        {{ ++mc_bad; failf("slot", fmt("g=%lld b=%lld cidx=%lld \
mc=%lld", g, b, b - lo, mc)); }}
                    }}
                if (cnt != kq::nc_of(g, KT, total, G, p))
                {{ ++mc_bad; failf("nc", fmt("g=%lld brute %lld kernel nc %lld", \
g, cnt, kq::nc_of(g, KT, total, G, p))); }}
                bmc = std::max(bmc, cnt);
            }}
            if (bmc != mc) {{ ++mc_bad; failf("max_contrib", std::string(c.name) + \
fmt(" brute %lld host %lld", bmc, mc)); }}
        }}
        summary.push_back(std::string(c.name) + fmt(" w=%lld,%lld valid=%lld \
max_contrib=%lld slots=%lld B", w.w0, w.w1,
                          valid, mc, (long long) ws) + fmt(" of \
EXL3_M16G_SLOT_BYTES %lld (%lld.%lld%%) fits=%lld",
                          (long long) EXL3_M16G_SLOT_BYTES, (long long) (ws * 100 \
/ EXL3_M16G_SLOT_BYTES),
                          (long long) (ws * 1000 / EXL3_M16G_SLOT_BYTES % 10), fits));
    }}
    for (const std::string& l : summary) fprintf(stderr, "config %s\\n", l.c_str());
    fprintf(stderr, "check host==kernel (table configs): %lld starts, %lld owners, \
%lld failures\\n", hk_starts, hk_owners, hk_bad);
    fprintf(stderr, "check owner==brute max + exactly-once (table configs): %lld \
owners, %lld failures\\n", t_owner, t_bad);
    fprintf(stderr, "check max_contrib==brute (valid table configs): %lld groups, \
%lld failures\\n", mc_groups, mc_bad);

    // ---- random weighted configurations
    std::mt19937 rng(2105);
    auto U = [&](int lo, int hi) {{ return \
std::uniform_int_distribution<int>(lo, hi)(rng); }};
    long long r_cfg = 0, r_valid = 0, r_vbad = 0, r_owner = 0, r_obad = 0, \
r_hk = 0, r_sbad = 0;
    for (int it = 0; it < 20000; ++it)
    {{
        const int n = U(1, 100), g = U(n, 2 * n), total = U(g, 6000), \
w0 = U(1, 200), w1 = U(1, 200);
        const hq::Weights w = {{ w0, w1, n }};
        const std::vector<int> bs = brute_starts(total, g, n, w0, w1);
        const std::vector<int> s = kernel_starts(total, g, n, w0, w1);
        bool nonempty = true;
        for (int b = 0; b < g; ++b) nonempty = nonempty && bs[b + 1] > bs[b];
        const bool want = n >= 1 && n <= g && g <= 2 * n && nonempty;
        ++r_cfg;
        r_valid += want;
        if (hq::weights_valid(total, g, w) != want)
        {{ ++r_vbad; failf("random weights_valid", fmt("total=%lld G=%lld \
nsm=%lld w=%lld,%lld want %lld", total, g, n, w0, w1, want)); }}
        for (int b = 0; b <= g; ++b)
            if (s[b] != bs[b] || hq::wstart(b, total, g, w) != s[b])
            {{ ++r_sbad; failf("random start", fmt("b=%lld total=%lld G=%lld \
nsm=%lld w=%lld,%lld", b, total, g, n, w0, w1)); }}
        r_obad += owner_once(total, g, n, w0, w1, s, r_owner, "random owner/once");
        for (int x = 0; x < total; ++x)
            if (hq::owner(x, total, g, w) != kq::exl3_m16g_owner(x, total, g, n, \
w0, w1))
            {{ ++r_hk; failf("random host==kernel owner", fmt("x=%lld total=%lld \
G=%lld", x, total, g)); }}
    }}
    fprintf(stderr, "check random weighted (seed 2105; nsm 1..100, G in \
[nsm, 2 nsm], total in [G, 6000], w0, w1 in 1..200): "
                    "%lld configs (%lld valid, %lld with an empty slice); \
weights_valid mismatches %lld; start != brute / host %lld; "
                    "%lld owners, owner/exactly-once failures %lld; host owner \
!= kernel %lld\\n",
            r_cfg, r_valid, r_cfg - r_valid, r_vbad, r_sbad, r_owner, r_obad, r_hk);

    // domain guard: G outside [nsm, 2 nsm], a zero weight, nsm 0
    long long d_cfg = 0, d_bad = 0;
    for (int it = 0; it < 2000; ++it)
    {{
        const int n = U(2, 100);
        const int g = it % 2 ? U(1, n - 1) : U(2 * n + 1, 3 * n);
        const int total = U(g, 6000);
        const hq::Weights w = {{ U(1, 200), U(1, 200), n }};
        ++d_cfg;
        if (hq::weights_valid(total, g, w)) {{ ++d_bad; failf("domain", \
fmt("G=%lld nsm=%lld accepted", g, n)); }}
    }}
    const hq::Weights zw[] = {{ {{ 0, 5, 82 }}, {{ 5, 0, 82 }}, {{ 1, 1, 0 }} }};
    for (const hq::Weights& w : zw)
    {{
        ++d_cfg;
        if (hq::weights_valid(10240, 164, w)) {{ ++d_bad; failf("domain", \
fmt("w=%lld,%lld nsm=%lld accepted", w.w0, w.w1, w.nsm)); }}
    }}
    fprintf(stderr, "check domain guard (G outside [nsm, 2 nsm], zero weight, \
nsm 0): %lld configs rejected-expected, %lld failures\\n", d_cfg, d_bad);

    // ---- (w, w) == 2001 v2
    long long v_cfg = 0, v_starts = 0, v_owners = 0, v_bad = 0;
    for (int it = 0; it < 5000; ++it)
    {{
        const int g = U(1, 200), total = U(g, 6000), n = U(1, 200), ww = U(1, 200);
        ++v_cfg;
        for (int b = 0; b <= g; ++b, ++v_starts)
            if (kq::exl3_m16g_start(b, total, g, n, ww, ww) != v2::s_b_of(b, total, g))
            {{ ++v_bad; failf("v2 start", fmt("b=%lld total=%lld G=%lld nsm=%lld \
w=%lld", b, total, g, n, ww)); }}
        for (int x = 0; x < total; ++x, ++v_owners)
            if (kq::exl3_m16g_owner(x, total, g, n, ww, ww) != \
v2::exl3_m16_owner(x, total, g))
            {{ ++v_bad; failf("v2 owner", fmt("x=%lld total=%lld G=%lld nsm=%lld \
w=%lld", x, total, g, n, ww)); }}
    }}
    fprintf(stderr, "check (w, w) == v2 (G 1..200, total in [G, 6000], nsm 1..200, \
w 1..200): %lld configs, %lld starts, %lld owners, %lld failures\\n",
            v_cfg, v_starts, v_owners, v_bad);

    // ---- empty slice rejected: total 200, G 164, nsm 82, (100, 1)
    long long e_bad = 0;
    {{
        const std::vector<int> bs = brute_starts(200, 164, 82, 100, 1);
        int empty = 0;
        for (int b = 0; b < 164; ++b) empty += bs[b + 1] == bs[b];
        const bool v = hq::weights_valid(200, 164, hq::Weights{{ 100, 1, 82 }});
        if (v || empty == 0) {{ ++e_bad; failf("empty slice", fmt("weights_valid \
%lld, empty slices %lld", v, empty)); }}
        fprintf(stderr, "check empty slice rejected (total 200, G 164, nsm 82, \
w 100,1): %d empty slices, weights_valid=%d, %lld failures\\n",
                empty, (int) v, e_bad);
    }}

    // ---- owner_le equivalence: exhaustive on nsm 1..16, G in [nsm, 2 nsm], \
w0, w1 1..16,
    // total in [G, 3 G + 8], every x < total; P == w0 * nsm is the only point the \
branches swap
    for (int n = 1; n <= 16; ++n)
        for (int g = n; g <= 2 * n; ++g)
            for (int w0 = 1; w0 <= 16; ++w0)
                for (int w1 = 1; w1 <= 16; ++w1)
                {{
                    const int64_t WG = brute_W(g, n, w0, w1);
                    for (int total = g; total <= 3 * g + 8; ++total)
                        for (int x = 0; x < total; ++x)
                        {{
                            ++eq_evals;
                            const int64_t P = (((int64_t) x + 1) * WG + total - 1) \
/ total - 1;
                            eq_hits += P == (int64_t) w0 * n;
                            if (eq_orig::exl3_m16g_owner(x, total, g, n, w0, w1) \
!= eq_le::exl3_m16g_owner(x, total, g, n, w0, w1))
                            {{ ++eq_diff; failf("owner_le", fmt("x=%lld \
total=%lld G=%lld nsm=%lld w=%lld,%lld", x, total, g, n, w0, w1)); }}
                        }}
                }}
    fprintf(stderr, "owner_le equivalence: %lld owner evaluations compared \
(exhaustive nsm 1..16, G in [nsm, 2 nsm], w0, w1 1..16, "
                    "total in [G, 3G + 8], every x < total), %lld at the \
boundary P == w0 * nsm, %lld differ\\n",
            eq_evals, eq_hits, eq_diff);

    const long long bad = hk_bad + t_bad + mc_bad + r_vbad + r_sbad + r_obad + \
r_hk + d_bad + v_bad + e_bad + eq_diff;
    fprintf(stderr, bad ? "C-side checks: %lld FAILURES\\n" : "C-side checks: \
all passed (%lld failures)\\n", bad);
    return bad ? 1 : 0;
}}
"""


def mutation_verdict(name: str, out: str, returncode: int) -> tuple[str, bool]:
    """Classify one mutation run.

    Args:
        name: Mutation name.
        out: Combined stdout and stderr of the run.
        returncode: Exit code of the run.

    Returns:
        The verdict text and whether the outcome was expected.

    """
    if returncode == 0:
        ev = next(
            (ln for ln in out.splitlines() if ln.startswith("owner_le equivalence")),
            "",
        )
        if name in EQUIVALENT:
            return (
                "SURVIVED(equivalent): table IDENTICAL and every C-side check "
                f"passed; evidence: {ev}"
            ), True
        return "SURVIVED (unexpected)", False
    ls = out.splitlines()
    fin = next((ln for ln in ls if ln.startswith("gemm_m16_wpart_diff: FAIL")), "")
    cside = next((ln for ln in ls if ln.startswith("C-side checks")), "")
    fails = [ln for ln in ls if ln.startswith("FAIL ")][:2]
    mm = next((i for i, ln in enumerate(ls) if ln.startswith("MISMATCH row")), None)
    table = (
        " / ".join(x.strip() for x in ls[mm : mm + 3])
        if mm is not None
        else "table IDENTICAL"
    )
    verdict = "REJECTED: " + " | ".join([fin, cside, *fails, table])
    if name in EQUIVALENT:
        return verdict + " (claimed equivalent!)", False
    return verdict, True


def run_all_mutations(tree: str) -> int:
    """Run every mutation in a child process and report each verdict.

    Args:
        tree: The patched engine tree.

    Returns:
        0 when every outcome is as expected, else 1.

    """
    rc = 0
    for name in MUTATIONS:
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: sys.executable re-running this script with a fixed mutation name, no shell
            [sys.executable, __file__, "--mutate", name, tree],
            capture_output=True,
            text=True,
            check=False,
        )
        verdict, expected = mutation_verdict(name, r.stdout + r.stderr, r.returncode)
        if not expected:
            rc = 1
        say(f"{name}: {verdict}")
        sys.stdout.flush()
    say(
        "gemm_m16_wpart_diff --all-mutations: "
        + ("as expected" if rc == 0 else "UNEXPECTED OUTCOME")
    )
    return rc


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line; --all-mutations runs and exits here.

    Args:
        argv: The full argv.

    Returns:
        The mutation name (or None) and the tree path.

    """
    args = argv[1:]
    mutate = None
    if args[:1] == ["--all-mutations"]:
        if len(args) != FLAG_WITH_VALUE:
            fail(USAGE)
        sys.exit(run_all_mutations(args[1]))
    if args[:1] == ["--mutate"]:
        if len(args) < FLAG_WITH_VALUE or args[1] not in MUTATIONS:
            fail(f"--mutate NAME, NAME in {sorted(MUTATIONS)}")
        mutate = args[1]
        args = args[2:]
    if len(args) != 1:
        fail(USAGE)
    return mutate, Path(args[0])


def block_tag(
    where: str,
    nonblank: list[str],
    origin: str,
    plus: set[str],
    ctx: set[str],
) -> tuple[str, int]:
    """Check a quoted block against the patch.

    Args:
        where: "FILE NAME lines A-B" prefix for failure messages.
        nonblank: Non-blank lines of the block.
        origin: "new", "mod" or "old".
        plus: The file's `+` lines.
        ctx: The file's context lines.

    Returns:
        The report suffix and the count of `+` lines.

    """
    if origin == "new":
        miss = [ln for ln in nonblank if ln not in plus]
        if miss:
            fail(f"{where}: not `+` lines of the patch: {miss}")
        return (
            f", {len(nonblank)}/{len(nonblank)} non-blank lines occur as `+` lines "
            "of the patch"
        ), len(nonblank)
    if origin == "mod":
        miss = [ln for ln in nonblank if ln not in plus | ctx]
        np_ = sum(ln in plus for ln in nonblank)
        if miss or np_ == 0:
            fail(f"{where}: outside the patch hunks {miss} or no `+` line")
        return (
            f", {np_}/{len(nonblank)} non-blank lines occur as `+` lines, "
            "the rest as patch context"
        ), np_
    return "", 0


def quote_blocks(tree: Path) -> tuple[str, dict[str, str]]:
    """Quote every block from the tree and verify it against the patch.

    Args:
        tree: The patched engine tree.

    Returns:
        The host source text and the quoted blocks by name.

    """
    src = {
        "H": (tree / H_PATH).read_text(),
        "K": (tree / K_PATH).read_text(),
        "V2": (tree / V2_PATH).read_text(),
    }
    fpath = {"H": H_PATH, "K": K_PATH, "V2": V2_PATH}
    lines = {f: t.split("\n") for f, t in src.items()}

    patch = PATCH.read_bytes()
    say(f"patch {PATCH.name} sha256 {hashlib.sha256(patch).hexdigest()}")
    plus, ctx = patch_lines(patch.decode())

    q: dict[str, str] = {}
    n_plus = 0
    for f, name, sig, kind, origin in BLOCKS:
        a, b = extract(lines[f], name, sig, kind)
        body = lines[f][a - 1 : b]
        q[name] = "\n".join(body)
        nonblank = [ln for ln in body if ln.strip()]
        # old blocks are not checked; their files need not occur in the patch
        checked = origin != "old"
        tag, n = block_tag(
            f"{f} {name} lines {a}-{b}",
            nonblank,
            origin,
            plus[fpath[f]] if checked else set(),
            ctx[fpath[f]] if checked else set(),
        )
        n_plus += n
        say(f"quoted {f}:{a}-{b} {name}{tag}")
    say(f"quoted 2105 lines verified as patch `+` lines: {n_plus}")
    return src["H"], q


def check_rows(h: str) -> None:
    """Check the tree's rows and the Bend table against CONFIGS.

    Args:
        h: The host source text.

    """
    weights, draft, routed = tree_rows(h)
    want_w = [
        (k, ns, w0, w1)
        for n, k, ns, w0, w1, ov in CONFIGS
        if ov == 0 and not n.startswith("draft")
    ]
    want_d = [(k, ns) for n, k, ns, w0, w1, ov in CONFIGS if n.startswith("draft")]
    if weights != want_w:
        fail(f"tree g_weights rows {weights} != {want_w}")
    if draft != want_d:
        fail(f"tree draft g_routes rows {draft} != {want_d}")
    if routed != [(k, ns) for k, ns, _, _ in want_w]:
        fail(f"tree non-draft g_routes rows {routed} != the g_weights bundles")
    bcfg = bend_configs((HERE / TABLE).read_text(), (HERE / IMPL).read_text())
    want_b = [(n, k, ns, w0, w1) for n, k, ns, w0, w1, ov in CONFIGS]
    if bcfg != want_b:
        fail(f"{TABLE} configuration list {bcfg} != {want_b}")
    say(
        f"tree rows match the table configs: g_weights {len(weights)}, "
        f"draft g_routes {len(draft)}, "
        f"routed g_routes {len(routed)}; {TABLE} main lists the {len(bcfg)} configs"
    )


def run_programs(
    q: dict[str, str], q_orig: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    """Compile and run the C++ program and the Bend table.

    Args:
        q: Quoted blocks (possibly mutated).
        q_orig: Unmutated quoted blocks.

    Returns:
        The C++ run and the Bend table run.

    """
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "wpart.cpp"
        c.write_text(c_program(q, q_orig))
        exe = Path(td) / "wpart"
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked([
                "c++",
                "-O2",
                "-std=c++17",
                "-Wall",
                "-Wno-unused-function",
                "-o",
                str(exe),
                str(c),
            ]),
            check=True,
        )
        tab = Path(td) / "table"
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
            source_link.locked([source_link.bend(), TABLE, "-o", str(tab)]),
            cwd=BEND_CWD,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            fail(f"bend compile: {r.stdout}{r.stderr}")
        cres = subprocess.run([str(exe)], capture_output=True, text=True, check=False)  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
        bres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            [str(tab)],
            capture_output=True,
            text=True,
            preexec_fn=unlimited_vm,
            check=False,
        )
    return cres, bres


def compare_tables(
    cres: subprocess.CompletedProcess[str], bres: subprocess.CompletedProcess[str]
) -> None:
    """Compare the C++ and Bend tables and fail on any difference.

    Args:
        cres: The C++ run.
        bres: The Bend table run.

    """
    if bres.returncode != 0:
        fail(f"Bend table exited {bres.returncode}: {bres.stderr}")
    sys.stdout.write(cres.stderr)
    cl, bl = cres.stdout.split("\n"), bres.stdout.split("\n")
    kinds = {k: sum(ln.startswith(k + " ") for ln in cl) for k in "CSg"}
    say(
        f"table rows: C {len(cl) - 1} (C {kinds['C']}, S {kinds['S']}, "
        f"g {kinds['g']}), Bend {len(bl) - 1}"
    )
    identical = cres.stdout == bres.stdout
    if identical:
        say("IDENTICAL")
    else:
        shown = 0
        for i in range(max(len(cl), len(bl))):
            a = cl[i] if i < len(cl) else "<eof>"
            b = bl[i] if i < len(bl) else "<eof>"
            if a != b:
                say(f"MISMATCH row {i}:\n  C:    {a}\n  Bend: {b}")
                shown += 1
                if shown == MAX_MISMATCHES:
                    break
    if cres.returncode != 0 or not identical:
        fail(
            " + ".join(
                (["table MISMATCH"] if not identical else [])
                + (["C-side check failures"] if cres.returncode != 0 else [])
            )
        )


def main(argv: list[str]) -> None:
    """Run the differential check.

    Args:
        argv: The full argv.

    """
    mutate, tree = parse_args(argv)
    h, q = quote_blocks(tree)
    check_rows(h)

    q_orig = dict(q)
    if mutate:
        blk, a, b = MUTATIONS[mutate]
        if q[blk].count(a) != 1 or sum(v.count(a) for v in q.values()) != 1:
            fail(f"mutation {mutate} does not apply exactly once")
        q[blk] = q[blk].replace(a, b)
        say(f"mutation {mutate}: {blk}: {a!r} -> {b!r}")

    cres, bres = run_programs(q, q_orig)
    compare_tables(cres, bres)
    say("gemm_m16_wpart_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
