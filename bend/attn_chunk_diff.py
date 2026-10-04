#!/usr/bin/env python3
"""
Finite differential check of bend/attn_chunk.bend + bend/attn_chunk_bounds.bend (ext patch 3003
verify attention: fixed absolute kv chunks) against the shipped expressions.

Quoted verbatim from the patched engine tree (argument = the exllamav3 package directory):
  exllamav3_ext/attn_verify.cuh   av_live_chunks
  exllamav3_ext/attn_verify.cu    the split kernel's chunk loop (n_chunks, trap, c = blockIdx.x,
                                  early return, n_start / n_end / ntiles and their loop updates,
                                  c += gridDim.x, break), tile start n0, tok0, the token mask,
                                  q_abs, pbase and the partial_o / partial_ml store addresses; the
                                  combine's live, trap, base, both partial_ml reads, idx, the
                                  m = -inf skip, the partial_o read and the out store address
  exllamav3_ext/attn_verify_gr.cu the chunk / block-table TORCH_CHECK
  exllamav3_ext/libtorch/attention.cpp  the partial_o / partial_ml / o / q size TORCH_CHECKs
  modules/attention_fn/bc_attn.py av_splits, av_max_chunks, pn_o, pn_ml (evaluated by Python)
The C++ lines are compiled for the CPU. The C program prints the same table as
bend/ATTN_CHUNK_TABLE.bend (compared byte for byte, host lines appended by this script) and replays
every (warp, t, e) token of every tile of every chunk of every CTA:
  - coverage: each kv position enters row q_pos's softmax exactly once iff p <= depth + q_pos;
  - invariance: for sampled absolute positions a and every round shape (q_len 1..8, row j < q_len,
    depth a - j), the row's per-(chunk, tile) visible token sets and its merged chunk list are
    identical;
  - agreement: every partial cell the combine reads for (row, chunk) is the one the split stored;
  - capacity: n_chunks <= av_max_chunks for every live length the block table can span.

Differential evidence on finite instances, not a proof. `--mutate NAME` applies a deliberate
source mutation that the check must reject.

Usage: python3 bend/attn_chunk_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py --through 3003-attn-verify-cuda.patch
  (any prefix with 3003 and without 3006).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/ATTN_CHUNK_TABLE.bend"

# Section A (CTA partition) / T (chunk tiles) / B (row merge lists): (L, chunk, S) cases. Must
# match ATTN_CHUNK_TABLE.bend.
LS = [1, 8, 63, 64, 65, 511, 512, 513, 1000, 1032, 2056, 4104]
CHUNKS = [64, 96, 512, 1024]
SS = [1, 3, 20, 82]
BIG = [(18608, 512), (18608, 1024), (32776, 512), (32776, 1024)]
# Section C (index) configs: (bsz, n_kv, mc, n_q, q_len)
IDX = [(1, 4, 528, 24, 8), (2, 4, 17, 24, 3), (1, 8, 3, 32, 1), (3, 2, 5, 12, 8)]
# Section H (host) configs: (sms, bsz, kvh, max_pages, chunk, splits, nsub, rows, hd_pad)
HOST = [(82, 1, 4, 1056, 512, 39, 3, 16, 256), (82, 1, 4, 1057, 512, 39, 3, 16, 256),
        (82, 1, 4, 1, 1024, 39, 3, 16, 256), (82, 2, 4, 33, 192, 10, 1, 16, 256),
        (46, 1, 8, 5, 64, 2, 4, 16, 256), (82, 1, 4, 1056, 1024, 39, 3, 16, 256)]
# Invariance sample of absolute positions (chunk boundaries of 64/96/512/1024 +- 1)
INV = [0, 1, 7, 63, 64, 95, 96, 511, 512, 513, 1023, 1024, 1025, 1535, 1536, 2047, 2048, 4100,
       18599, 32767]

MUTATIONS = {
    # causal mask off by one: the diagonal (own position) is dropped
    "causal_strict": ("tok < n_end && tok <= q_abs", "tok < n_end && tok < q_abs", "cu"),
    # live chunk count rounded down: the tail chunk is never computed nor combined
    "live_floor": ("return (L + chunk - 1) / chunk;", "return L / chunk;", "cuh"),
    # round-relative chunk (the pre-invariance design): chunk sized from the live length
    "round_relative": ("const int n_chunks = av_live_chunks(L, chunk);",
                       ("chunk = ((L + (int) gridDim.x * 64 - 1) / ((int) gridDim.x * 64)) * 64; "
                        "const int n_chunks = av_live_chunks(L, chunk);"), "cu"),
    # combine without the m = -inf skip
    "combine_no_skip": ("if (ml.x == NEG_INF) continue;", "", "cu"),
    # split partials laid out by the round's live chunk count instead of max_chunks
    "pbase_live": ("* max_chunks + c) * AV_ROWS;", "* n_chunks + c) * AV_ROWS;", "cu"),
    # host sizing floor instead of ceiling
    "host_floor": ("-(-self.max_pages * PAGE_SIZE // _AV_CHUNK)", "self.max_pages * PAGE_SIZE // _AV_CHUNK", "py"),
    # gr check against one page fewer than the block table spans
    "gr_short": ("(int64_t) num_pages * AV_PAGE", "(int64_t) (num_pages - 1) * AV_PAGE", "gr"),
}


def fail(msg: str):
    raise SystemExit(f"attn_chunk_diff: {msg}")


def grab(text: str, pattern: str, count: int = 1) -> list[str]:
    m = re.findall(pattern, text)
    if len(m) != count:
        fail(f"pattern {pattern!r} matched {len(m)} times, expected {count}")
    return m


def one(text: str, pattern: str) -> str:
    return grab(text, pattern)[0]


def c_program(cuh: str, cu: str, gr: str, cpp: str) -> str:
    live_fn = one(cuh, r"__host__ __device__ inline int av_live_chunks\(int L, int chunk\)\n\{\n(?:.*\n)*?\}")
    live_fn = live_fn.replace("__host__ __device__ ", "")
    # both kernels take L = cache_seqlens[b] + q_len (the live length the laws' L stands for)
    _ = grab(cu, r"const int L = cache_seqlens\[b\] \+ q_len;", 2)
    nch = one(cu, r"const int n_chunks = [^\n;]*;")
    trap = one(cu, r"if \(n_chunks > max_chunks\) __trap\(\);")
    c0 = one(cu, r"int c = blockIdx\.x;")
    ret = one(cu, r"if \(c >= n_chunks\) return;")
    ns = one(cu, r"int n_start = c \* chunk;")
    ne = one(cu, r"int n_end = min\(n_start \+ chunk, L\);")
    nt = one(cu, r"int ntiles = \(n_end - n_start \+ AV_T - 1\) / AV_T;")
    inc = one(cu, r"c \+= gridDim\.x;")
    brk = one(cu, r"if \(c >= n_chunks\) break;")
    ns2 = one(cu, r"\n\s+n_start = c \* chunk;").strip()
    ne2 = one(cu, r"\n\s+n_end = min\(n_start \+ chunk, L\);").strip()
    nt2 = one(cu, r"\n\s+ntiles = \(n_end - n_start \+ AV_T - 1\) / AV_T;").strip()
    n0 = one(cu, r"const int n0 = n_start \+ it \* AV_T;")
    tok0 = one(cu, r"const int tok0 = n0 \+ warp \* 8 \+ 2 \* t;")
    ok = one(cu, r"bool ok = row_ok\[2 \* i \+ hr\] && [^;]*;").replace("row_ok[2 * i + hr] && ", "")
    qabs = one(cu, r"const int q_abs = L - q_len \+ gid;")
    pbase = one(cu, r"size_t pbase = [^;]*;")
    st_o = grab(cu, r"partial_o \+ \(pbase \+ [^)]*\) \* AV_HD \+ col", 2)
    st_ml = one(cu, r"partial_ml \+ \(pbase \+ 8 \* s \+ gid\) \* 2")
    clive = one(cu, r"const int live = av_live_chunks\(L, chunk\);")
    ctrap = one(cu, r"if \(live > max_chunks\) __trap\(\);")
    cbase = one(cu, r"const size_t base = [^;]*;")
    cmax = one(cu, r"mmax = fmaxf\(mmax, partial_ml\[[^\]]*\]\);")
    cidx = one(cu, r"size_t idx = base \+ \(size_t\) s \* AV_ROWS;")
    cml = one(cu, r"\*reinterpret_cast<const float2\*>\(partial_ml \+ idx \* 2\)")
    cskip = re.findall(r"if \(ml\.x == NEG_INF\) continue;", cu)
    co = one(cu, r"partial_o\[idx \* AV_HD \+ d\]")
    cout = one(cu, r"out\[\(\(size_t\) \(b \* q_len \+ q_pos\) \* n_q_heads \+ kvh \* group \+ hl\) \* AV_HD \+ d\]")
    gr_chk = one(gr, r"\(int64_t\) max_chunks \* chunk >= \(int64_t\) [^,]*")
    chk_o = one(cpp, r"s\.partial_o\.numel\(\) >= ([^&]*?) &&").strip()
    chk_ml = one(cpp, r"s\.partial_ml\.numel\(\) >= ([^,]*?),").strip()
    chk_out = one(cpp, r"s\.o\.numel\(\) >= ([^&]*?) &&").strip()
    chk_q = one(cpp, r"s\.q\.numel\(\) >= ([^,]*?),").strip()
    if chk_out != chk_q:
        fail("q and o size checks differ")
    # address expressions -> element offsets
    st_o_e = [e.replace("partial_o + ", "") for e in st_o]
    st_ml_e = st_ml.replace("partial_ml + ", "")
    cmax_e = cmax[len("mmax = fmaxf(mmax, partial_ml["):-len("]);")]
    cml_e = cml.replace("*reinterpret_cast<const float2*>(partial_ml + ", "")[:-1]
    co_e = co.replace("partial_o[", "")[:-1]
    cout_e = cout.replace("out[", "")[:-1]
    skip = cskip[0] if cskip else ""
    return f"""
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
using std::min;
#define AV_T 64
#define AV_HD 256
#define AV_ROWS 48
#define AV_PAGE 256
#define NEG_INF (-INFINITY)
struct D3 {{ int x, y, z; }};
static D3 blockIdx, gridDim;
static int trapped = 0;
static void __trap() {{ trapped = 1; }}
struct F2 {{ float x, y; }};
{live_fn}

// Split kernel chunk loop of CTA (x, kvh, b), quoted; visit(c, n_start, n_end, ntiles) per chunk
template <class V> static void split_cta(int L, int chunk, int max_chunks, int S, int x, V visit) {{
    gridDim.x = S; blockIdx.x = x;
    {nch}
    {trap}
    if (trapped) return;
    {c0}
    {ret}
    {ns}
    {ne}
    {nt}
    for (;;) {{
        visit(c, n_start, n_end, ntiles);
        {inc}
        {brk}
        {ns2}
        {ne2}
        {nt2}
    }}
}}
// Tokens of tile it of a chunk that pass row gid's mask, quoted (issue n0 == loop n0 checked)
static std::vector<int> tile_tokens(int L, int q_len, int gid, int n_start, int n_end, int it) {{
    std::vector<int> r;
    {n0}
    for (int warp = 0; warp < 8; ++warp) for (int t = 0; t < 4; ++t) {{
        {tok0}
        {qabs}
        for (int e = 0; e < 2; ++e) {{
            int tok = tok0 + e;
            {ok}
            if (ok) r.push_back(tok);
        }}
    }}
    return r;
}}
static int live_of(int L, int chunk, int max_chunks) {{ {clive} {ctrap} return live; }}
static size_t split_po(int b, int n_kv_heads, int kvh, int max_chunks, int n_chunks, int c, int i, int gid, int col, int hi) {{
    (void) n_chunks;
    {pbase}
    return hi ? (size_t) ({st_o_e[1]}) : (size_t) ({st_o_e[0]});
}}
static size_t split_pml(int b, int n_kv_heads, int kvh, int max_chunks, int n_chunks, int c, int s, int gid) {{
    (void) n_chunks;
    {pbase}
    return (size_t) ({st_ml_e});
}}
static size_t comb_base(int b, int n_kv_heads, int kvh, int max_chunks, int r) {{ {cbase} return base; }}
static size_t comb_max(size_t base, int s) {{ return (size_t) ({cmax_e}); }}
static size_t comb_ml(size_t base, int s) {{ {cidx} return (size_t) ({cml_e}); }}
static size_t comb_o(size_t base, int s, int d) {{ {cidx} return (size_t) ({co_e}); }}
static size_t out_at(int b, int q_len, int q_pos, int n_q_heads, int kvh, int group, int hl, int d) {{ return (size_t) ({cout_e}); }}
static int64_t chk_o(int64_t bsz, int64_t num_kv_heads, int64_t av_max_chunks) {{ return {chk_o}; }}
static int64_t chk_ml(int64_t bsz, int64_t num_kv_heads, int64_t av_max_chunks) {{ return {chk_ml}; }}
static int64_t chk_out(int64_t bsz, int64_t q_len, int64_t num_q_heads) {{ return {chk_out}; }}
static bool gr_ok(int64_t max_chunks, int64_t chunk, int64_t num_pages) {{ return {gr_chk}; }}
// Combine's merged chunk list for a row, given each chunk's m (-inf = no visible token), quoted skip
static std::vector<int> merged(int live, const std::vector<float>& m) {{
    std::vector<int> r;
    for (int s = 0; s < live; ++s) {{
        F2 ml = {{ m[s], 0.f }};
        {skip}
        r.push_back(s);
    }}
    return r;
}}
struct RowPlan {{ std::vector<std::vector<int>> toks; std::vector<int> merged; }};
static RowPlan row_plan(int L, int q_len, int gid, int chunk, int S, int max_chunks) {{
    RowPlan P; int live = live_of(L, chunk, max_chunks); if (trapped) return P;
    std::vector<float> m(live > 0 ? live : 0, NEG_INF);
    std::vector<std::vector<std::vector<int>>> per(live > 0 ? live : 0);
    for (int x = 0; x < S; ++x)
        split_cta(L, chunk, max_chunks, S, x, [&](int c, int n_start, int n_end, int ntiles) {{
            for (int it = 0; it < ntiles; ++it) {{
                auto tk = tile_tokens(L, q_len, gid, n_start, n_end, it);
                if (c < live) {{ if (!tk.empty()) m[c] = 0.f; per[c].push_back(tk); }}
            }}
        }});
    for (auto& ch : per) for (auto& tl : ch) {{ if (!tl.empty()) P.toks.push_back(tl); }}
    P.merged = merged(live, m);
    return P;
}}
static long bad = 0;
static void nums(const std::vector<int>& v) {{ for (size_t i = 0; i < v.size(); ++i) printf(i ? ",%d" : "%d", v[i]); }}
int main() {{
    const int Ls[] = {{{", ".join(map(str, LS))}}};
    const int Cs[] = {{{", ".join(map(str, CHUNKS))}}};
    const int Ss[] = {{{", ".join(map(str, SS))}}};
    const int big[][2] = {{{", ".join("{%d, %d}" % b for b in BIG)}}};
    std::vector<std::pair<int,int>> lc;
    for (int L : Ls) for (int ch : Cs) lc.push_back({{L, ch}});
    for (auto& b : big) lc.push_back({{b[0], b[1]}});
    const int MC = 1 << 20;
    for (auto [L, ch] : lc) {{
        int live = live_of(L, ch, MC);
        // A: stores per chunk s < live + 2 over the grid, per S
        for (int S : Ss) {{
            std::vector<int> w(live + 2, 0);
            for (int x = 0; x < S; ++x)
                split_cta(L, ch, MC, S, x, [&](int c, int, int, int) {{ if (c < live + 2) ++w[c]; }});
            printf("A L=%d chunk=%d S=%d live=%d w=", L, ch, S, live);
            for (int v : w) printf("%d", v);
            printf("\\n");
        }}
        // T: chunk geometry
        printf("T L=%d chunk=%d", L, ch);
        split_cta(L, ch, MC, 1, 0, [&](int c, int n_start, int n_end, int ntiles) {{ printf(" %d:%d:%d", n_start, n_end, ntiles); }});
        printf("\\n");
        // B: merged lists of the rows of a q_len 8 round, coverage replay
        if (L < 8) continue;
        for (int gid = 0; gid < 8; ++gid) {{
            RowPlan P = row_plan(L, 8, gid, ch, 20, MC);
            printf("B L=%d chunk=%d q=%d m=", L, ch, gid); nums(P.merged); printf("\\n");
            std::vector<int> cnt(L + 256, 0);
            for (auto& tl : P.toks) for (int t : tl) ++cnt[t];
            int a = L - 8 + gid;
            for (int p = 0; p < (int) cnt.size(); ++p) if (cnt[p] != (p <= a ? 1 : 0)) ++bad;
        }}
    }}
    // Invariance: every round shape of each sampled absolute position gives the same row plan
    const int inv[] = {{{", ".join(map(str, INV))}}};
    long inv_bad = 0, inv_n = 0;
    for (int ch : Cs) for (int a : inv) for (int S : {{1, 20, 82}}) {{
        RowPlan ref = row_plan(a + 1, 1, 0, ch, S, MC);
        for (int q_len = 1; q_len <= 8; ++q_len) for (int j = 0; j < q_len; ++j) {{
            if (a - j < 0) continue;
            RowPlan P = row_plan(a - j + q_len, q_len, j, ch, S, MC);
            ++inv_n;
            if (P.toks != ref.toks || P.merged != ref.merged) ++inv_bad;
        }}
    }}
    // C: index table + split/combine cell agreement
    const int idx[][5] = {{{", ".join("{%d, %d, %d, %d, %d}" % c for c in IDX)}}};
    long agree_bad = 0;
    for (auto& cf : idx) {{
        int bsz = cf[0], nkv = cf[1], mc = cf[2], nq = cf[3], ql = cf[4], group = nq / nkv;
        printf("C bsz=%d nkv=%d mc=%d nq=%d ql=%d o=%lld ml=%lld out=%lld", bsz, nkv, mc, nq, ql,
               (long long) chk_o(bsz, nkv, mc), (long long) chk_ml(bsz, nkv, mc), (long long) chk_out(bsz, ql, nq));
        int bs[] = {{0, bsz - 1}}, ks[] = {{0, nkv - 1}}, cs[] = {{0, mc - 1}};
        for (int b : bs) for (int kvh : ks) for (int c : cs) {{
            // corner rows (0, 47) and columns (0, 255): the stores' (i, gid, hi) / col decomposition
            size_t po0 = split_po(b, nkv, kvh, mc, mc, c, 0, 0, 0, 0), po1 = split_po(b, nkv, kvh, mc, mc, c, 2, 7, 255, 1);
            size_t ml0 = split_pml(b, nkv, kvh, mc, mc, c, 0, 0), ml1 = split_pml(b, nkv, kvh, mc, mc, c, 5, 7) + 1;
            printf(" %zu:%zu:%zu:%zu", po0, po1, ml0, ml1);
        }}
        int ds[] = {{0, 255}};
        for (int b : bs) for (int d : ds) printf(" %zu", out_at(b, ql, ql - 1, nq, nkv - 1, group, group - 1, d));
        printf("\\n");
        // agreement over every (b, kvh, c, row, col) with n_chunks = mc (no trap)
        for (int b = 0; b < bsz; ++b) for (int kvh = 0; kvh < nkv; ++kvh) for (int c = 0; c < mc; ++c)
            for (int i = 0; i < 3; ++i) for (int hi = 0; hi < 2; ++hi) for (int gid = 0; gid < 8; ++gid) {{
                int r = 16 * i + 8 * hi + gid;
                size_t base = comb_base(b, nkv, kvh, mc, r);
                for (int col = 0; col < 256; col += 37)
                    if (split_po(b, nkv, kvh, mc, (c + 1), c, i, gid, col, hi) != comb_o(base, c, col)) ++agree_bad;
                int s6 = r / 8, g8 = r % 8;
                size_t sml = split_pml(b, nkv, kvh, mc, (c + 1), c, s6, g8);
                if (sml != comb_ml(base, c) || sml != comb_max(base, c)) ++agree_bad;
                if (split_po(b, nkv, kvh, mc, c + 1, c, i, gid, 255, hi) >= (size_t) chk_o(bsz, nkv, mc)) ++agree_bad;
                if (sml + 1 >= (size_t) chk_ml(bsz, nkv, mc)) ++agree_bad;
            }}
    }}
    // Capacity: every L the block table spans (L <= num_pages * 256) with the gr check passed
    long cap_bad = 0;
    for (int ch : Cs) for (int np : {{1, 2, 3, 5, 33, 1056, 1057}}) for (int mc : {{1, 2, 3, 17, 528, 529, 1057}}) {{
        if (!gr_ok(mc, ch, np)) continue;
        for (int L = 1; L <= np * 256; L += (np > 40 ? 97 : 1)) {{ trapped = 0; live_of(L, ch, mc); if (trapped) ++cap_bad; }}
        trapped = 0; live_of(np * 256, ch, mc); if (trapped) ++cap_bad;
    }}
    trapped = 0;
    fprintf(stderr, "coverage violations: %ld; invariance: %ld of %ld round shapes differ; split/combine cell disagreements + out-of-bounds: %ld; capacity traps: %ld\\n",
            bad, inv_bad, inv_n, agree_bad, cap_bad);
    return (bad || inv_bad || agree_bad || cap_bad) ? 3 : 0;
}}
"""


def host_lines(py: str) -> list[str]:
    splits_e = one(py, r"av_splits = (max\(1, _get_sm_count\(dev\) // \(bsz \* kvh\)\))")
    mc_e = one(py, r"av_max_chunks = (-\(-self\.max_pages \* PAGE_SIZE // _AV_CHUNK\)|[^\n]*_AV_CHUNK[^\n]*)")
    po_e = one(py, r"pn_o = (max\(pn_o, bsz \* kvh \* av_max_chunks \* 48 \* 256\))")
    pml_e = one(py, r"pn_ml = (max\(pn_ml, bsz \* kvh \* av_max_chunks \* 48 \* 2\))")
    out = []
    for sms, bsz, kvh, max_pages, chunk, splits, nsub, rows, hd_pad in HOST:
        env = {"max": max, "bsz": bsz, "kvh": kvh, "_get_sm_count": lambda dev: sms, "dev": 0,
               "PAGE_SIZE": 256, "_AV_CHUNK": chunk, "self": type("S", (), {"max_pages": max_pages})()}
        av_splits = eval(splits_e, env)
        env["av_max_chunks"] = av_max_chunks = eval(mc_e, env)
        # 3004 dense sizes (bc_attn.py: dense_blocks * block_rows * hd_pad / * 2)
        dense = bsz * kvh * splits * nsub
        env["pn_o"], env["pn_ml"] = dense * rows * hd_pad, dense * rows * 2
        out.append(f"H sms={sms} bsz={bsz} kvh={kvh} pages={max_pages} chunk={chunk} S={av_splits} "
                   f"mc={av_max_chunks} pn_o={eval(po_e, env)} pn_ml={eval(pml_e, env)}")
    return out


def main(argv: list[str]) -> None:
    mutate = None
    if len(argv) >= 3 and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0]] + argv[3:]
    if len(argv) != 2:
        fail(__doc__)
    root = Path(argv[1])
    src = {
        "cuh": (root / "exllamav3_ext/attn_verify.cuh").read_text(),
        "cu": (root / "exllamav3_ext/attn_verify.cu").read_text(),
        "gr": (root / "exllamav3_ext/attn_verify_gr.cu").read_text(),
        "cpp": (root / "exllamav3_ext/libtorch/attention.cpp").read_text(),
        "py": (root / "modules/attention_fn/bc_attn.py").read_text(),
    }
    if mutate:
        a, b, where = MUTATIONS[mutate]
        if src[where].count(a) != 1:
            fail(f"mutation {mutate} does not apply")
        src[where] = src[where].replace(a, b)
    # The chunk partition is a function of the kernel arguments chunk / max_chunks and the live
    # length only: the kernels never write those arguments (a reassignment on any line, quoted or
    # not, fails here; the quoted-line extraction alone would not see it)
    writes = re.compile(r"(?<![\w.>])(?:chunk|max_chunks)\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)"
                        r"|(?:\+\+|--)\s*(?:chunk|max_chunks)\b|\b(?:chunk|max_chunks)\s*(?:\+\+|--)")
    hits = [m.group(0) for m in writes.finditer(src["cu"])]
    if hits:
        fail(f"attn_verify.cu writes a chunk-partition kernel argument: {hits}")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "diff.cpp"
        c.write_text(c_program(src["cuh"], src["cu"], src["gr"], src["cpp"]))
        exe = Path(td) / "diff"
        subprocess.run(source_link.locked(["c++", "-O2", "-std=c++17", "-w", "-o", str(exe), str(c)]), check=True)
        cres = subprocess.run(source_link.locked([str(exe)]), capture_output=True, text=True)
    ctext = cres.stdout + "".join(line + "\n" for line in host_lines(src["py"]))
    bres = subprocess.run(source_link.locked([source_link.bend(), TABLE]), cwd=REPO, capture_output=True, text=True,
                          check=True)
    same = ctext == bres.stdout
    print(cres.stderr.strip() or f"C table program exit status {cres.returncode}")
    print(f"table lines: C+host {len(ctext.splitlines())}, Bend {len(bres.stdout.splitlines())}; byte-identical: {same}")
    if not same:
        for i, (x, y) in enumerate(zip(ctext.splitlines(), bres.stdout.splitlines())):
            if x != y:
                print(f"first difference at line {i + 1}:\n  C:    {x[:300]}\n  Bend: {y[:300]}")
                break
    if not same or cres.returncode != 0:
        fail("MISMATCH" if not same else "C-side violation")
    print("attn_chunk_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
