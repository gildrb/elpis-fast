#!/usr/bin/env python3
"""
Finite differential check of bend/attn_stride.bend + bend/attn_stride_bounds.bend (ext patch 3006
verify attention: strided absolute 64-token tiles, one partial slot per split CTA) against the
shipped expressions.

Quoted verbatim from the patched engine tree (argument = the exllamav3 package directory):
  exllamav3_ext/attn_verify.cuh   the AV_* constants, av_live_tiles, av_cta_tiles, av_live_slots
  exllamav3_ext/attn_verify.cu    split: L, the block-table trap, n_tiles, cta, S, the early return,
                                  ntiles, bt, the issue lambda (n0, row0, n_valid), the prologue and
                                  in-loop issue guards, the tile loop header and its n0 (identical to
                                  the issue n0), tok0, tok, row_ok, the mask, q_abs, pbase, col, the
                                  partial_o / partial_ml stores and the ml store guard;
                                  combine: group, q_pos, hl, the row guard, L, live, the live trap,
                                  base, the (m, l) staging, mmax, m_use, the weight, the weighted
                                  loop header, its partial_o read, the skip, the out store address
  exllamav3_ext/attn_verify_gr.cu the splits check, the two partial-size checks, the grids and the
                                  split / combine argument lists
  exllamav3_ext/libtorch/attention.cpp  configure_slot's partial / o / q size checks, the
                                  av_splits plumbing
  modules/attention_fn/bc_attn.py av_splits, the 3004 dense sizes, pn_o, pn_ml (evaluated by Python
                                  with a fake SM count)
The C++ lines are compiled for the CPU. The C program prints the same table as
bend/ATTN_STRIDE_TABLE.bend (compared byte for byte, host lines appended by this script) and replays
every (warp, t, e) token of every tile of every CTA x < S (issue schedule, mask, stores, combine):
  - coverage: in every replayed round each kv position enters row q_pos's softmax exactly once iff
    p <= L - q_len + q_pos (every row of the q_len 8 rounds, the invariance rounds);
  - slots: every (m, l) / partial_o cell the combine reads was stored exactly once by this launch,
    by split CTA s for the same (b, kvh, row, column); the merged slot list of a row at absolute
    position a is range(min(S, a / 64 + 1));
  - bounds: every store / read index is below the size configure_slot and attn_verify_gr check,
    the index grids are dense (every cell stored once), every issued tile of a non-trapping CTA
    reads a block-table page < num_pages inside one page and starts in the live range; the trap
    fires iff L > num_pages * 256; the combine never traps for a splits value attn_verify_gr
    accepts; the host allocation passes both checks;
  - invariance: a row at absolute position a has the same visible (slot, tile, tokens) plan and
    merge list in every round shape (q_len 1..8, row j < q_len, depth a - j).

Differential evidence on finite instances, not a proof. `--mutate NAME` applies a deliberate
source mutation that the check must reject.

Usage: python3 bend/attn_stride_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEND = "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend"
TABLE = "bend/ATTN_STRIDE_TABLE.bend"

# Sections A (slot stores) / T (CTA tiles) / B (row merge lists): (L, S) cases. Must match
# ATTN_STRIDE_TABLE.bend.
LS = [1, 8, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1000, 1279, 1280, 1281, 1343, 2560, 4104]
SS = [1, 3, 20, 82]
BIG = [(18608, 20), (32776, 20), (262152, 20)]
# Section C (index) configs: (bsz, n_kv, S, n_q, q_len)
IDX = [(1, 4, 20, 24, 8), (2, 4, 3, 24, 3), (1, 8, 1, 32, 1), (2, 2, 82, 12, 8), (3, 1, 5, 6, 5)]
# Section G (splits check / combine trap): S x L
GS = [0, 1, 82, 255, 256, 257, 512, 513]
GL = [1, 16384, 16385, 16448, 16449, 38400]
# Section H (host) configs: (sms, bsz, kvh, splits, nsub, rows, hd_pad)
HOST = [(82, 1, 4, 39, 3, 16, 256), (82, 1, 4, 10, 1, 16, 256), (46, 1, 8, 2, 4, 16, 256),
        (82, 2, 1, 4, 1, 16, 256), (4, 1, 8, 1, 1, 16, 256), (82, 4, 4, 20, 2, 16, 128)]
# Invariance sample of absolute positions: every a < 260, tile boundaries +- 1 up to 1600, the far
# positions of the big cases
INV = sorted(set(range(260)) | {64 * k + d for k in range(1, 26) for d in (-1, 0, 1)} |
             {1279, 1280, 1281, 1343, 2559, 2560, 4103, 18607, 32775, 262151})

# name: (original, replacement, file, occurrences)
MUTATIONS = {
    # round-relative stride: S shrinks to the live tile count of the round
    "round_relative": ("const int S = gridDim.x;", "const int S = min((int) gridDim.x, n_tiles);", "cu", 1),
    # tile start without the stride scaling
    "n0_wrong": ("(cta + it * S) * AV_T", "cta * AV_T + it * S", "cu", 2),
    # CTA tile count rounded down: the CTA's last tile is dropped
    "ntiles_floor": ("return (n_tiles - x + S - 1) / S;", "return (n_tiles - x) / S;", "cuh", 1),
    # combine without the m = -inf (w = -1) skip
    "combine_no_skip": ("if (w < 0.f) continue;", "", "cu", 1),
    # combine reads every grid slot, live or not
    "combine_live_S": ("const int live = av_live_slots(av_live_tiles(L), splits);", "const int live = splits;", "cu", 1),
    # causal mask admits the next position
    "mask_off_by_one": ("tok <= q_abs;", "tok <= q_abs + 1;", "cu", 1),
    # split slots laid out by kv heads instead of the grid width
    "pbase_nkv": ("* S + cta) * AV_ROWS;", "* n_kv_heads + cta) * AV_ROWS;", "cu", 1),
    # block-table trap one page short: rounds the table spans trap
    "trap_short": ("if (L > num_pages_per_seq * AV_PAGE)", "if (L > (num_pages_per_seq - 1) * AV_PAGE)", "cu", 1),
    # host sizes the partials with 47 rows per slot
    "host_pn_o_47": ("bsz * kvh * av_splits * 48 * 256", "bsz * kvh * av_splits * 47 * 256", "py", 1),
    # host launches half the split CTAs the model fixes (SMs // kv heads)
    "host_splits_small": ("max(1, _get_sm_count(dev) // kvh)", "max(1, _get_sm_count(dev) // (kvh * 2))", "py", 1),
    # live tile count rounded down: the partial tail tile is never processed
    "live_tiles_floor": ("return (L + AV_T - 1) / AV_T;", "return L / AV_T;", "cuh", 1),
    # attn_verify_gr accepts more slots than the combine has staging threads
    "gr_splits_wide": ("splits > 0 && splits <= AV_HD", "splits > 0 && splits <= 2 * AV_HD", "gr", 1),
    # combine without its live > AV_HD trap
    "combine_no_trap": ("if (live > AV_HD) __trap();", "", "cu", 1),
}


def fail(msg: str):
    raise SystemExit(f"attn_stride_diff: {msg}")


def grab(text: str, pattern: str, count: int = 1) -> list:
    m = re.findall(pattern, text)
    if len(m) != count:
        fail(f"pattern {pattern!r} matched {len(m)} times, expected {count}")
    return m


def one(text: str, pattern: str):
    return grab(text, pattern)[0]


def opt(text: str, pattern: str) -> str:
    """A guard that may be absent (its absence must then fail the semantic checks), never doubled."""
    m = re.findall(pattern, text)
    if len(m) > 1:
        fail(f"pattern {pattern!r} matched {len(m)} times, expected at most 1")
    return m[0] if m else ""


HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
using std::min;
using std::max;
@@DEFINES@@
#define CUDART_INF_F INFINITY
@@NEG_INF@@
struct D3 { int x, y, z; };
static D3 blockIdx, gridDim;
static int trapped = 0;
static void __trap() { trapped = 1; }
static float ex2(float x) { return exp2f(x); }
@@LIVE_TILES@@
@@CTA_TILES@@
@@LIVE_SLOTS@@

static long bad_cov = 0, bad_slot = 0, bad_bound = 0, bad_page = 0, bad_merge = 0, bad_inv = 0,
            bad_trap = 0, bad_host = 0, bad_sched = 0;
#define BAD(c, ...) do { if (!c++) { fprintf(stderr, "first " #c ": "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Simulated buffer: per element the launch that stored it last, its store count in that launch,
// the owner (row-major id of the (b, kvh, slot, row, column) the storing thread holds) and, for
// (m, l), whether the row saw no token (m = -inf)
struct Mem {
    std::vector<uint32_t> gen; std::vector<uint16_t> n; std::vector<int64_t> own; std::vector<uint8_t> empty;
    uint32_t cur = 1;
    void init(size_t sz) { gen.assign(sz, 0); n.assign(sz, 0); own.assign(sz, -1); empty.assign(sz, 1); cur = 1; }
    void next() { ++cur; }
    size_t size() const { return gen.size(); }
    bool store(size_t i, int64_t o, bool e) {
        if (i >= gen.size()) return false;
        if (gen[i] != cur) { gen[i] = cur; n[i] = 0; }
        ++n[i]; own[i] = o; empty[i] = e; return true;
    }
    int count(size_t i) const { return i < gen.size() && gen[i] == cur ? n[i] : 0; }
};
struct Cfg { int bsz, n_kv_heads, n_q_heads, S; };
static int64_t cell(const Cfg& C, int b, int kvh, int x, int row, int col, int w) {
    return ((((int64_t) b * C.n_kv_heads + kvh) * C.S + x) * AV_ROWS + row) * w + col;
}

// ---- split kernel, CTA (x, kvh, b) of the grid (C.S, C.n_kv_heads, C.bsz): quoted control flow,
// issue schedule, mask (rows gid in [g_lo, g_hi), head slot 0) and epilogue stores
struct CtaRun {
    bool trap = false, ret = false;
    std::vector<int> n0s, iss_it, iss_n0, iss_pg;
    std::vector<std::vector<std::vector<int>>> vis;   // [gid - g_lo][tile] visible tokens
};
static CtaRun split_cta(int seqlen, int q_len, int num_pages_per_seq, int b, int kvh, const Cfg& C, int x,
                        int g_lo, int g_hi, Mem* po, Mem* pml) {
    CtaRun R;
    const int n_kv_heads = C.n_kv_heads, n_q_heads = C.n_q_heads;
    std::vector<int> seqv(b + 1, seqlen);
    const int* cache_seqlens = seqv.data();
    std::vector<int> table((size_t) (b + 1) * num_pages_per_seq + 1);
    for (size_t k = 0; k < table.size(); ++k) table[k] = (int) k;
    const int* block_table = table.data();
    gridDim = {C.S, C.n_kv_heads, C.bsz};
    blockIdx = {x, kvh, b};
    @@GROUP@@
    (void) group;
    trapped = 0;
    @@L@@
    @@TRAP@@
    if (trapped) { R.trap = true; return R; }
    @@N_TILES@@
    @@CTA@@
    @@S@@
    R.ret = true;
    @@RET@@
    R.ret = false;
    @@NT@@
    @@BT@@
    auto issue = [&] (int it)
    {
        @@N0_ISSUE@@
        int pg = (int) (@@PAGE@@);
        R.iss_it.push_back(it); R.iss_n0.push_back(n0); R.iss_pg.push_back(pg);
        if (pg < 0 || pg >= num_pages_per_seq) { BAD(bad_page, "L=%d pages=%d x=%d it=%d n0=%d page=%d", L, num_pages_per_seq, x, it, n0, pg); return; }
        @@ROW0@@
        int n_valid = (int) (@@NVALID@@);
        if (row0 != (b * num_pages_per_seq + pg) * AV_PAGE + n0 % AV_PAGE || n0 % AV_PAGE + AV_T > AV_PAGE || n_valid <= 0)
            BAD(bad_page, "L=%d x=%d it=%d n0=%d row0=%d n_valid=%d", L, x, it, n0, row0, n_valid);
    };
    @@PRO_FOR@@
    {
        @@PRO_IF@@
    }
    std::vector<char> empty(8, 1);
    R.vis.assign(max(g_hi - g_lo, 0), {});
    @@LOOP_FOR@@
    {
        @@LOOP_IF@@
        @@N0_LOOP@@
        R.n0s.push_back(n0);
        for (int gid = g_lo; gid < g_hi; ++gid) {
            @@QABS@@
            bool row_ok[6];
            @@ROW_OK@@
            std::vector<int> v;
            for (int warp = 0; warp < AV_NW; ++warp) for (int t = 0; t < 4; ++t) {
                @@TOK0@@
                const int i = 0, hr = 0;
                for (int e = 0; e < 2; ++e) {
                    @@TOK@@
                    @@OK@@
                    if (ok) v.push_back(tok);
                }
            }
            if (!v.empty()) empty[gid] = 0;
            R.vis[gid - g_lo].push_back(v);
        }
    }
    @@PBASE@@
    for (int warp = 0; warp < AV_NW; ++warp) for (int lane = 0; lane < 32; ++lane) {
        const int gid = lane >> 2, t = lane & 3;
        if (po)
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) {
                @@COL@@
                size_t a0 = (size_t) (@@ST_O0@@), a1 = (size_t) (@@ST_O1@@);
                for (int k = 0; k < 2; ++k) {
                    if (!po->store(a0 + k, cell(C, b, kvh, x, 16 * i + gid, col + k, AV_HD), false))
                        BAD(bad_bound, "partial_o store %zu >= %zu", a0 + k, po->size());
                    if (!po->store(a1 + k, cell(C, b, kvh, x, 16 * i + 8 + gid, col + k, AV_HD), false))
                        BAD(bad_bound, "partial_o store %zu >= %zu", a1 + k, po->size());
                }
            }
        if (pml && (@@ML_COND@@))
            for (int s = 0; s < 6; ++s) {
                size_t a = (size_t) (@@ST_ML@@);
                bool e = (gid >= g_lo && gid < g_hi) ? empty[gid] : true;
                for (int k = 0; k < 2; ++k)
                    if (!pml->store(a + k, cell(C, b, kvh, x, 8 * s + gid, k, 2), e))
                        BAD(bad_bound, "partial_ml store %zu >= %zu", a + k, pml->size());
            }
    }
    return R;
}

// ---- combine CTA (r, kvh, b): quoted live / trap / staging / weights / merge loop. reads = false:
// stop after the trap (live and trap outcome only)
struct Comb { bool ret = false, trap = false; int live = -1; std::vector<int> merged; };
static Comb combine(int seqlen, int q_len, int b, int kvh, int r, const Cfg& C, const Mem* po, const Mem* pml, bool reads) {
    Comb K;
    const int n_kv_heads = C.n_kv_heads, n_q_heads = C.n_q_heads, splits = C.S;
    std::vector<int> seqv(b + 1, seqlen);
    const int* cache_seqlens = seqv.data();
    @@GROUP@@
    @@QPOS@@
    @@HL@@
    K.ret = true;
    @@CRET@@
    K.ret = false;
    @@L@@
    @@LIVE@@
    K.live = live;
    trapped = 0;
    @@CTRAP@@
    if (trapped) { K.trap = true; return K; }
    if (!reads) return K;
    @@BASE@@
    std::vector<float> sw(AV_HD, NAN), sl(AV_HD, NAN);
    for (int d = 0; d < AV_HD; ++d) {
        if (@@STAGE_IF@@) {
            size_t a = (size_t) (@@STAGE@@);
            bool ok = true;
            for (int k = 0; k < 2; ++k) {
                if (a + k >= pml->size()) { BAD(bad_bound, "partial_ml read %zu >= %zu", a + k, pml->size()); ok = false; continue; }
                if (pml->count(a + k) != 1 || pml->own[a + k] != cell(C, b, kvh, d, r, k, 2)) {
                    BAD(bad_slot, "L=%d S=%d r=%d slot %d: (m, l) cell %zu stored %d times this launch, owner %lld", seqlen + q_len, C.S, r, d, a + k, pml->count(a + k), (long long) pml->own[a + k]);
                    ok = false;
                }
            }
            sw[d] = ok ? (pml->empty[a] ? NEG_INF : 0.f) : NAN;
            sl[d] = 1.f;
        }
    }
    @@MMAX0@@
    @@MMAX@@
    @@MUSE@@
    for (int d = 0; d < AV_HD; ++d) {
        @@WLINE@@
    }
    @@WFOR@@
    {
        if (po)
            for (int d = 0; d < AV_HD; ++d) {
                size_t a = (size_t) (@@OREAD@@);
                if (a >= po->size()) { BAD(bad_bound, "partial_o read %zu >= %zu", a, po->size()); continue; }
                if (po->count(a) != 1 || po->own[a] != cell(C, b, kvh, s, r, d, AV_HD))
                    BAD(bad_slot, "L=%d S=%d r=%d slot %d d=%d: partial_o cell %zu stored %d times, owner %lld", seqlen + q_len, C.S, r, s, d, a, po->count(a), (long long) po->own[a]);
            }
        @@W@@
        @@SKIP@@
        K.merged.push_back(s);
    }
    return K;
}
static size_t comb_ml_at(int b, int kvh, int r, int d, const Cfg& C) {
    const int n_kv_heads = C.n_kv_heads, splits = C.S;
    @@BASE@@
    return (size_t) (@@STAGE@@);
}
static size_t comb_o_at(int b, int kvh, int r, int s, int d, const Cfg& C) {
    const int n_kv_heads = C.n_kv_heads, splits = C.S;
    @@BASE@@
    return (size_t) (@@OREAD@@);
}
static size_t split_po_at(int b, int kvh, int cta, int i, int gid, int hi, int col, const Cfg& C) {
    const int n_kv_heads = C.n_kv_heads, S = C.S;
    @@PBASE@@
    return hi ? (size_t) (@@ST_O1@@) : (size_t) (@@ST_O0@@);
}
static size_t split_pml_at(int b, int kvh, int cta, int s, int gid, const Cfg& C) {
    const int n_kv_heads = C.n_kv_heads, S = C.S;
    @@PBASE@@
    return (size_t) (@@ST_ML@@);
}
static size_t out_at(int b, int q_len, int q_pos, int n_q_heads, int kvh, int group, int hl, int d) { return (size_t) (@@OUT@@); }
// configure_slot (T) and attn_verify_gr (G) size checks, and the G splits check
static int64_t chk_o(int64_t bsz, int64_t num_kv_heads, int64_t av_splits) { return @@CHK_O@@; }
static int64_t chk_ml(int64_t bsz, int64_t num_kv_heads, int64_t av_splits) { return @@CHK_ML@@; }
static int64_t chk_out(int64_t bsz, int64_t q_len, int64_t num_q_heads) { return @@CHK_OUT@@; }
static int64_t chk_q(int64_t bsz, int64_t q_len, int64_t num_q_heads) { return @@CHK_Q@@; }
static int64_t gr_o(int64_t bsz, int64_t n_kv_heads, int64_t splits) { return @@GR_O@@; }
static int64_t gr_ml(int64_t bsz, int64_t n_kv_heads, int64_t splits) { return @@GR_ML@@; }
static bool gr_ok(int splits) { return @@GR_OK@@; }
static bool slot_ok(int av_splits) { return @@SLOT_OK@@; }

static void nums(const std::vector<int>& v) { for (size_t i = 0; i < v.size(); ++i) printf(i ? ",%d" : "%d", v[i]); }
static std::vector<int> range_upto(int n) { std::vector<int> r; for (int i = 0; i < n; ++i) r.push_back(i); return r; }

// One launch (split grid of (b, kvh) = (1, 1) in bsz 2 x n_kv 2, then the combine of row j) for a
// round of q_len rows at depth seqlen; returns the row's visible (slot, tile start, tokens) plan and
// merge list, checking coverage, the merge list and the slot reads
typedef std::vector<std::tuple<int, int, std::vector<int>>> Plan;
static void row_round(int seqlen, int q_len, int j, int S, Mem& pml, Plan& plan, std::vector<int>& merged) {
    Cfg C{2, 2, 12, S};
    int L = seqlen + q_len, a = seqlen + j, np = (L + AV_PAGE - 1) / AV_PAGE;
    pml.next();
    plan.clear();
    std::vector<int> cnt((size_t) (L + 2 * AV_T), 0);
    for (int x = 0; x < S; ++x) {
        CtaRun R = split_cta(seqlen, q_len, np, 1, 1, C, x, j, j + 1, nullptr, &pml);
        if (R.trap) BAD(bad_trap, "L=%d pages=%d trapped", L, np);
        for (size_t t = 0; t < R.n0s.size(); ++t) {
            const std::vector<int>& v = R.vis[0][t];
            for (int p : v) { if (p >= 0 && p < (int) cnt.size()) ++cnt[p]; else BAD(bad_cov, "token %d out of range", p); }
            if (!v.empty()) plan.emplace_back(x, R.n0s[t], v);
        }
    }
    for (int p = 0; p < (int) cnt.size(); ++p)
        if (cnt[p] != (p <= a ? 1 : 0)) BAD(bad_cov, "L=%d S=%d q_len=%d row %d (a=%d): position %d entered %d times", L, S, q_len, j, a, p, cnt[p]);
    Comb K = combine(seqlen, q_len, 1, 1, j, C, nullptr, &pml, true);
    if (K.ret || K.trap) BAD(bad_merge, "L=%d S=%d row %d: combine returned / trapped", L, S, j);
    merged = K.merged;
    if (merged != range_upto(min(S, a / AV_T + 1))) BAD(bad_merge, "L=%d S=%d q_len=%d row %d: merged list wrong", L, S, q_len, j);
}

int main() {
    const int Ls[] = {@@LS@@};
    const int Ss[] = {@@SS@@};
    const int big[][2] = {@@BIG@@};
    std::vector<std::pair<int, int>> cases;
    for (int L : Ls) for (int S : Ss) cases.push_back({L, S});
    for (auto& bg : big) cases.push_back({bg[0], bg[1]});
    for (auto [L, S] : cases) {
        Cfg C{2, 2, 12, S};
        int ql = min(L, 8), np = (L + AV_PAGE - 1) / AV_PAGE;
        bool track_o = L <= 4104;
        Mem pml, po;
        pml.init((size_t) chk_ml(2, 2, S));
        if (track_o) po.init((size_t) chk_o(2, 2, S));
        std::vector<CtaRun> runs;
        for (int x = 0; x < S; ++x) {
            runs.push_back(split_cta(L - ql, ql, np, 1, 1, C, x, 0, ql, track_o ? &po : nullptr, &pml));
            CtaRun& R = runs.back();
            if (R.trap) BAD(bad_trap, "L=%d pages=%d trapped", L, np);
            // issue schedule: tiles 0 .. ntiles - 1 issued once each, in order, at the loop's n0
            bool ok = R.iss_it.size() == R.n0s.size();
            for (size_t k = 0; ok && k < R.iss_it.size(); ++k) ok = R.iss_it[k] == (int) k && R.iss_n0[k] == R.n0s[k];
            if (!ok) BAD(bad_sched, "L=%d S=%d x=%d: issue schedule differs from the tile loop", L, S, x);
        }
        // A
        Comb K0 = combine(L - ql, ql, 1, 1, 0, C, nullptr, &pml, false);
        int live = K0.live;
        printf("A L=%d S=%d tiles=%d live=%d w=", L, S, av_live_tiles(L), live);
        for (int s = 0; s < live + 2; ++s) {
            size_t a = comb_ml_at(1, 1, 0, s, C);
            printf("%d", a < pml.size() ? pml.count(a) : 0);
        }
        printf("\n");
        // T
        int ts[2];
        for (int k = 0; k < 2; ++k) { CtaRun R = split_cta(L - ql, ql, np - k, 1, 1, C, 0, 0, 0, nullptr, nullptr); ts[k] = !R.trap; }
        printf("T L=%d S=%d runs=%d%d", L, S, ts[0], ts[1]);
        for (int x = 0; x < S; ++x) {
            const CtaRun& R = runs[x];
            int k = (int) R.n0s.size();
            if (!k) { printf(" 0:-"); continue; }
            int pg = -1;
            for (size_t q = 0; q < R.iss_it.size(); ++q) if (R.iss_it[q] == k - 1) pg = R.iss_pg[q];
            printf(" %d:%d:%d:%d", k, R.n0s[0], R.n0s[k - 1], pg);
        }
        printf("\n");
        // B + coverage of every row of the round
        for (int gid = 0; gid < ql; ++gid) {
            int a = L - ql + gid;
            std::vector<int> cnt((size_t) (L + 2 * AV_T), 0);
            // a CTA that took the kernel's early return (cta >= n_tiles) visits no token
            for (auto& R : runs) { if (R.ret) continue; for (auto& v : R.vis[gid]) for (int p : v) { if (p >= 0 && p < (int) cnt.size()) ++cnt[p]; else BAD(bad_cov, "token %d", p); } }
            for (int p = 0; p < (int) cnt.size(); ++p)
                if (cnt[p] != (p <= a ? 1 : 0)) BAD(bad_cov, "L=%d S=%d row %d: position %d entered %d times", L, S, gid, p, cnt[p]);
            Comb K = combine(L - ql, ql, 1, 1, gid, C, track_o ? &po : nullptr, &pml, true);
            if (K.ret || K.trap) BAD(bad_merge, "L=%d S=%d row %d: combine returned / trapped", L, S, gid);
            if (K.merged != range_upto(min(S, a / AV_T + 1))) BAD(bad_merge, "L=%d S=%d row %d: merged list wrong", L, S, gid);
            if (L >= 8) { printf("B L=%d S=%d q=%d m=", L, S, gid); nums(K.merged); printf("\n"); }
        }
    }
    // Invariance: every round shape of a sampled absolute position gives the same row plan
    const int inv[] = {@@INV@@};
    long inv_n = 0;
    for (int S : Ss) {
        Mem pml; pml.init((size_t) chk_ml(2, 2, S));
        for (int a : inv) {
            Plan ref, P; std::vector<int> mref, m;
            row_round(a, 1, 0, S, pml, ref, mref);
            for (int q_len = 1; q_len <= 8; ++q_len) for (int j = 0; j < q_len; ++j) {
                if (a - j < 0) continue;
                row_round(a - j, q_len, j, S, pml, P, m);
                ++inv_n;
                if (P != ref || m != mref) BAD(bad_inv, "a=%d S=%d q_len=%d j=%d: row plan differs", a, S, q_len, j);
            }
        }
    }
    // Block-table trap: fires iff L > num_pages * 256 (every CTA, before any tile)
    for (int np : {1, 2, 3, 5}) for (int S : Ss) for (int L = 1; L <= np * AV_PAGE + 130; ++L) {
        Cfg C{1, 1, 6, S};
        for (int x = 0; x < S; ++x) {
            CtaRun R = split_cta(L - 1, 1, np, 0, 0, C, x, 0, 0, nullptr, nullptr);
            if (R.trap != (L > np * AV_PAGE)) BAD(bad_trap, "L=%d pages=%d x=%d: trap %d", L, np, x, (int) R.trap);
        }
    }
    // C: index table; dense store grids; every combine read is the cell split CTA s stored
    const int idx[][5] = {@@IDX@@};
    for (auto& cf : idx) {
        int bsz = cf[0], nkv = cf[1], S = cf[2], nq = cf[3], ql = cf[4], group = nq / nkv;
        Cfg C{bsz, nkv, nq, S};
        if (gr_o(bsz, nkv, S) != chk_o(bsz, nkv, S) || gr_ml(bsz, nkv, S) != chk_ml(bsz, nkv, S) || chk_q(bsz, ql, nq) != chk_out(bsz, ql, nq))
            BAD(bad_bound, "configure_slot and attn_verify_gr size checks differ");
        printf("C bsz=%d nkv=%d S=%d nq=%d ql=%d o=%lld ml=%lld out=%lld", bsz, nkv, S, nq, ql,
               (long long) chk_o(bsz, nkv, S), (long long) chk_ml(bsz, nkv, S), (long long) chk_out(bsz, ql, nq));
        int bs[] = {0, bsz - 1}, ks[] = {0, nkv - 1}, xs[] = {0, S - 1};
        for (int b : bs) for (int kvh : ks) for (int x : xs) {
            printf(" %zu:%zu:%zu:%zu/%zu:%zu:%zu:%zu",
                   split_po_at(b, kvh, x, 0, 0, 0, 0, C), split_po_at(b, kvh, x, 2, 7, 1, 254, C) + 1,
                   split_pml_at(b, kvh, x, 0, 0, C), split_pml_at(b, kvh, x, 5, 7, C) + 1,
                   comb_o_at(b, kvh, 0, x, 0, C), comb_o_at(b, kvh, 47, x, 255, C),
                   comb_ml_at(b, kvh, 0, x, C), comb_ml_at(b, kvh, 47, x, C) + 1);
        }
        int ds[] = {0, 255};
        for (int b : bs) for (int d : ds) printf(" %zu", out_at(b, ql, ql - 1, nq, nkv - 1, group, group - 1, d));
        printf("\n");
        for (int L : {2 * S * AV_T, 1, AV_T + 1}) {
            Mem po, pml;
            po.init((size_t) chk_o(bsz, nkv, S)); pml.init((size_t) chk_ml(bsz, nkv, S));
            int lq = min(ql, L), np = (L + AV_PAGE - 1) / AV_PAGE, nt = 0;
            for (int b = 0; b < bsz; ++b) for (int kvh = 0; kvh < nkv; ++kvh) for (int x = 0; x < S; ++x) {
                CtaRun R = split_cta(L - lq, lq, np, b, kvh, C, x, 0, 0, &po, &pml);
                if (R.trap) BAD(bad_trap, "C: L=%d trapped", L);
                nt += !R.ret;
            }
            // the grid of all-live CTAs stores every cell of both checked sizes exactly once
            if (nt == bsz * nkv * S) {
                for (size_t a = 0; a < po.size(); ++a) if (po.count(a) != 1) BAD(bad_bound, "partial_o cell %zu stored %d times", a, po.count(a));
                for (size_t a = 0; a < pml.size(); ++a) if (pml.count(a) != 1) BAD(bad_bound, "partial_ml cell %zu stored %d times", a, pml.count(a));
            }
            std::vector<uint8_t> oc((size_t) chk_out(bsz, lq, nq), 0);
            for (int b = 0; b < bsz; ++b) for (int kvh = 0; kvh < nkv; ++kvh) for (int r = 0; r < AV_ROWS; ++r) {
                Comb K = combine(L - lq, lq, b, kvh, r, C, &po, &pml, true);
                if (K.trap) BAD(bad_trap, "C: combine trapped");
                if (K.ret) continue;
                int q_pos = r % AV_QPOS, hl = r / AV_QPOS;
                for (int d = 0; d < AV_HD; ++d) {
                    size_t o = out_at(b, lq, q_pos, nq, kvh, group, hl, d);
                    if (o >= oc.size()) BAD(bad_bound, "out %zu >= %zu", o, oc.size()); else ++oc[o];
                }
            }
            for (size_t o = 0; o < oc.size(); ++o) if (oc[o] != 1) BAD(bad_bound, "out cell %zu stored %d times", o, oc[o]);
        }
    }
    // G: splits check and combine trap
    const int gs[] = {@@GS@@}, gl[] = {@@GL@@};
    for (int S : gs) for (int L : gl) {
        Comb K = combine(L - 1, 1, 0, 0, 0, Cfg{1, 1, 6, S}, nullptr, nullptr, false);
        printf("G S=%d L=%d gr=%d live=%d runs=%d\n", S, L, (int) gr_ok(S), K.live, (int) !K.trap);
    }
    // the combine never traps for a splits value attn_verify_gr accepts
    for (int S = 0; S <= 1024; ++S) {
        if (!gr_ok(S)) continue;
        for (int nt : {1, S - 1, S, S + 1, 4 * S + 3, 5000})
            if (nt >= 1) {
                Comb K = combine(nt * AV_T - 1, 1, 0, 0, 0, Cfg{1, 1, 6, S}, nullptr, nullptr, false);
                if (K.trap) BAD(bad_trap, "combine traps for accepted splits %d (live %d)", S, K.live);
            }
    }
    // Host: the allocation for the av_splits it launches passes configure_slot and attn_verify_gr
    const long long host[][5] = {@@HOSTV@@};
    for (auto& h : host) {
        long long bsz = h[0], kvh = h[1], sp = h[2], pn_o = h[3], pn_ml = h[4];
        if (!slot_ok((int) sp) || !gr_ok((int) sp) || pn_o < chk_o(bsz, kvh, sp) || pn_ml < chk_ml(bsz, kvh, sp) ||
            pn_o < gr_o(bsz, kvh, sp) || pn_ml < gr_ml(bsz, kvh, sp))
            BAD(bad_host, "bsz=%lld kvh=%lld av_splits=%lld pn_o=%lld pn_ml=%lld fails a check", bsz, kvh, sp, pn_o, pn_ml);
    }
    fprintf(stderr, "coverage: %ld; slot reads: %ld; bounds: %ld; pages: %ld; issue schedule: %ld; merge lists: %ld; "
            "invariance: %ld of %ld round shapes differ; traps: %ld; host: %ld\n",
            bad_cov, bad_slot, bad_bound, bad_page, bad_sched, bad_merge, bad_inv, inv_n, bad_trap, bad_host);
    return (bad_cov || bad_slot || bad_bound || bad_page || bad_sched || bad_merge || bad_inv || bad_trap || bad_host) ? 3 : 0;
}
"""


def c_program(cuh: str, cu: str, gr: str, cpp: str, hostv: list) -> str:
    q = {}
    defs = re.findall(r"#define AV_(\w+) (\d+)", cuh)
    names = [n for n, _ in defs]
    if sorted(names) != sorted(["HD", "ROWS", "QPOS", "NW", "T", "STAGES", "PAGE"]):
        fail(f"unexpected AV_* constants {names}")
    q["DEFINES"] = "\n".join(f"#define AV_{n} {v}" for n, v in defs)
    q["NEG_INF"] = one(cu, r"#define NEG_INF \(-CUDART_INF_F\)")
    for key, fn in (("LIVE_TILES", r"av_live_tiles\(int L\)"), ("CTA_TILES", r"av_cta_tiles\(int n_tiles, int x, int S\)"),
                    ("LIVE_SLOTS", r"av_live_slots\(int n_tiles, int S\)")):
        q[key] = one(cuh, r"__host__ __device__ inline int " + fn + r"\n\{\n(?:.*\n)*?\}").replace("__host__ __device__ ", "static ")
    # split
    q["L"] = grab(cu, r"const int L = cache_seqlens\[b\] \+ q_len;", 2)[0]
    q["GROUP"] = grab(cu, r"const int group = n_q_heads / n_kv_heads;", 2)[0]
    q["TRAP"] = one(cu, r"if \(L > [^\n]*\) __trap\(\);")
    q["N_TILES"] = one(cu, r"const int n_tiles = [^;\n]*;")
    q["CTA"] = one(cu, r"const int cta = [^;\n]*;")
    q["S"] = one(cu, r"const int S = [^;\n]*;")
    q["RET"] = one(cu, r"if \(cta >= n_tiles\) return;")
    q["NT"] = one(cu, r"const int ntiles = [^;\n]*;")
    q["BT"] = one(cu, r"const int\* bt = [^;\n]*;")
    n0_issue = one(cu, r"\n\s+(int n0 = ([^;\n]*);)")
    n0_loop = one(cu, r"(const int n0 = ([^;\n]*);)")
    if n0_issue[1] != n0_loop[1]:
        fail(f"issue n0 {n0_issue[1]!r} and tile-loop n0 {n0_loop[1]!r} differ")
    q["N0_ISSUE"], q["N0_LOOP"] = n0_issue[0], n0_loop[0]
    row0 = one(cu, r"int row0 = [^;\n]*;")
    q["ROW0"] = row0
    q["PAGE"] = one(row0, r"bt\[([^\]]*)\]")
    q["NVALID"] = one(cu, r"row0, ([^,\n]*), kvh, n_kv_heads, tid\)")
    q["PRO_FOR"] = one(cu, r"for \(int s = 0; s < AV_STAGES - 1; \+\+s\)")
    q["PRO_IF"] = one(cu, r"if \([^\n]*\) issue\(s\);")
    q["LOOP_FOR"] = one(cu, r"for \(int it = 0; it < ntiles; \+\+it\)")
    q["LOOP_IF"] = one(cu, r"if \([^\n]*\) issue\(it \+ AV_STAGES - 1\);")
    q["QABS"] = one(cu, r"const int q_abs = [^;\n]*;")
    q["ROW_OK"] = one(cu, r"for \(int s = 0; s < 6; \+\+s\) row_ok\[s\] = [^;\n]*;")
    q["TOK0"] = one(cu, r"const int tok0 = [^;\n]*;")
    q["TOK"] = one(cu, r"\n\s+(int tok = tok0[^;\n]*;)")
    q["OK"] = one(cu, r"bool ok = [^;\n]*;")
    q["PBASE"] = one(cu, r"size_t pbase = [^;\n]*;")
    q["COL"] = one(cu, r"(int col = [^;\n]*;)\n\s+\*reinterpret_cast<float2\*>\(partial_o")
    st_o = grab(cu, r"partial_o \+ (\(pbase \+ [^)]*\) \* AV_HD \+ col)\) = ", 2)
    q["ST_O0"], q["ST_O1"] = st_o
    ml = one(cu, r"if \(([^\n]*)\)\n\s+#pragma unroll\n\s+for \(int s = 0; s < 6; \+\+s\)\n\s+"
                 r"\*reinterpret_cast<float2\*>\(partial_ml \+ ([^;\n]*)\) = make_float2\(m\[s\], l\[s\]\);")
    q["ML_COND"], q["ST_ML"] = ml
    # combine
    q["QPOS"] = one(cu, r"const int q_pos = [^;\n]*;")
    q["HL"] = one(cu, r"const int hl = [^;\n]*;")
    q["CRET"] = one(cu, r"if \(q_pos >= q_len \|\| hl >= group\) return;")
    q["LIVE"] = one(cu, r"const int live = [^;\n]*;")
    q["CTRAP"] = opt(cu, r"if \(live > [^\n]*\) __trap\(\);")
    q["BASE"] = one(cu, r"const size_t base = [^;\n]*;")
    stage = one(cu, r"if \(([^\n]*)\)\n\s+\{\n\s+float2 ml = \*reinterpret_cast<const float2\*>\(partial_ml \+ ([^;\n]*)\);\n"
                    r"\s+sw\[d\] = ml\.x;\n\s+sl\[d\] = ml\.y;")
    q["STAGE_IF"], q["STAGE"] = stage
    q["MMAX0"] = one(cu, r"float mmax = NEG_INF;")
    q["MMAX"] = one(cu, r"for \(int s = 0; s < live; \+\+s\) mmax = fmaxf\(mmax, sw\[s\]\);")
    q["MUSE"] = one(cu, r"float m_use = mmax[^;\n]*;")
    q["WLINE"] = one(cu, r"if \([^\n]*\) sw\[d\] = [^;\n]*;")
    q["WFOR"] = one(cu, r"\n\s+(for \(int s = 0; s < live; \+\+s\))\n\s+\{\n\s+float v = partial_o")
    q["OREAD"] = one(cu, r"float v = partial_o\[([^\]\n]*)\];")
    q["W"] = one(cu, r"float w = sw\[s\];")
    q["SKIP"] = opt(cu, r"if \(w < 0\.f\) continue;")
    q["OUT"] = one(cu, r"\n\s+out\[([^\]\n]*)\] = ")
    # attn_verify_gr: checks, grids and argument lists (the combine's splits is the split grid's x)
    q["GR_OK"] = one(gr, r"TORCH_CHECK\((splits > 0[^,\n]*), ")
    q["GR_O"] = one(gr, r"\n\s+TORCH_CHECK\(partial_o\.numel\(\) >= ([^&\n]*?) &&")
    q["GR_ML"] = one(gr, r"\n\s+partial_ml\.numel\(\) >= ([^,\n]*?),")
    one(gr, r"dim3\(splits, n_kv_heads, bsz\), dim3\(AV_NW \* 32\), split_args")
    one(gr, r"dim3\(AV_ROWS, n_kv_heads, bsz\), dim3\(AV_HD\)")
    one(gr, r"&cpo_ptr, &cpml_ptr, &h32_ptr, &sl_ptr, &out_ptr, &splits, &q_len, &n_q_heads, &n_kv_heads")
    one(gr, r"&num_pages, &q_len, &n_q_heads, &n_kv_heads, &scale_log2")
    one(gr, r"int num_pages = \(int\) block_table\.size\(1\);")
    # configure_slot
    q["SLOT_OK"] = one(cpp, r"TORCH_CHECK\((av_splits > 0) &&\n\s+s\.partial_o\.numel\(\)")
    q["CHK_O"] = one(cpp, r"s\.partial_o\.numel\(\) >= ([^&\n]*?) &&").strip()
    q["CHK_ML"] = one(cpp, r"s\.partial_ml\.numel\(\) >= ([^,\n]*?),").strip()
    q["CHK_OUT"] = one(cpp, r"s\.o\.numel\(\) >= ([^&\n]*?) &&").strip()
    q["CHK_Q"] = one(cpp, r"s\.q\.numel\(\) >= ([^,\n]*?),").strip()
    one(cpp, r"s\.av_splits = av_splits;")
    one(cpp, r"s\.partial_o, s\.partial_ml, s\.o, s\.av_splits,")
    q["LS"] = ", ".join(map(str, LS))
    q["SS"] = ", ".join(map(str, SS))
    q["BIG"] = ", ".join("{%d, %d}" % b for b in BIG)
    q["INV"] = ", ".join(map(str, INV))
    q["IDX"] = ", ".join("{%d, %d, %d, %d, %d}" % c for c in IDX)
    q["GS"] = ", ".join(map(str, GS))
    q["GL"] = ", ".join(map(str, GL))
    q["HOSTV"] = ", ".join("{%d, %d, %d, %d, %d}" % h for h in hostv)
    # The quoted kernel early returns (verified verbatim above) leave a void __global__ body; the
    # harness bodies return their run record, marked by the preceding `.ret = true`. Only the
    # statement's `return;` is rewritten, never its condition (clang rejects a bare return here).
    for key, record in (("RET", "R"), ("CRET", "K")):
        require_one = q[key].count("return;") == 1
        if not require_one:
            fail(f"quoted {key} must end in exactly one bare return")
        q[key] = q[key].replace("return;", f"return {record};")
    src = HARNESS
    for k, v in q.items():
        src = src.replace(f"@@{k}@@", v)
    left = re.findall(r"@@\w+@@", src)
    if left:
        fail(f"unfilled harness slots {left}")
    return src


def host_eval(py: str) -> tuple[list[str], list]:
    dpb = one(py, r"def dense_partial_blocks\(bsz: int, n_kv_heads: int, splits: int, nsub: int\) -> int:\n(?:.*\n)*?    return ([^\n]*)\n")
    blocks_e = one(py, r"dense_blocks = (dense_partial_blocks\(bsz, kvh, splits_p, nsub_p\))")
    po0_e = one(py, r"\n\s+pn_o = (dense_blocks[^\n]*)")
    pml0_e = one(py, r"\n\s+pn_ml = (dense_blocks[^\n]*)")
    splits_e = one(py, r"av_splits = (max\([^\n]*)")
    po_e = one(py, r"pn_o = (max\(pn_o, [^\n]*)")
    pml_e = one(py, r"pn_ml = (max\(pn_ml, [^\n]*)")
    one(py, r"\n\s+av_splits,\n\s+float\(self\.sm_scale\)")
    out, vals = [], []
    for sms, bsz, kvh, splits, nsub, rows, hd_pad in HOST:
        env = {"max": max, "bsz": bsz, "kvh": kvh, "_get_sm_count": lambda dev: sms, "dev": 0,
               "splits_p": splits, "nsub_p": nsub, "block_rows": rows, "hd_pad": hd_pad,
               "dense_partial_blocks": lambda bsz, n_kv_heads, splits, nsub: eval(
                   dpb, {"bsz": bsz, "n_kv_heads": n_kv_heads, "splits": splits, "nsub": nsub})}
        env["dense_blocks"] = eval(blocks_e, env)
        env["pn_o"], env["pn_ml"] = eval(po0_e, env), eval(pml0_e, env)
        env["av_splits"] = av_splits = eval(splits_e, env)
        pn_o, pn_ml = eval(po_e, env), eval(pml_e, env)
        out.append(f"H sms={sms} bsz={bsz} kvh={kvh} S={av_splits} pn_o={pn_o} pn_ml={pn_ml}")
        vals.append((bsz, kvh, av_splits, pn_o, pn_ml))
    return out, vals


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
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate}; known: {', '.join(MUTATIONS)}")
        a, b, where, n = MUTATIONS[mutate]
        if src[where].count(a) != n:
            fail(f"mutation {mutate} does not apply ({src[where].count(a)} occurrences, expected {n})")
        src[where] = src[where].replace(a, b)
    # The partition is a function of the kernel arguments q_len / num_pages_per_seq / splits and the
    # grid only: the kernels never write those arguments (a reassignment on any line, quoted or
    # not, fails here; the quoted-line extraction alone would not see it)
    writes = re.compile(r"(?<![\w.>])(?:q_len|num_pages_per_seq|splits|n_kv_heads)\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)"
                        r"|(?:\+\+|--)\s*(?:q_len|num_pages_per_seq|splits|n_kv_heads)\b"
                        r"|\b(?:q_len|num_pages_per_seq|splits|n_kv_heads)\s*(?:\+\+|--)")
    hits = [m.group(0) for m in writes.finditer(src["cu"])]
    if hits:
        fail(f"attn_verify.cu writes a partition kernel argument: {hits}")
    hlines, hvals = host_eval(src["py"])
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "diff.cpp"
        c.write_text(c_program(src["cuh"], src["cu"], src["gr"], src["cpp"], hvals))
        exe = Path(td) / "diff"
        subprocess.run(["/tmp/cpu-lock.sh", "c++", "-O2", "-std=c++17", "-w", "-o", str(exe), str(c)], check=True)
        cres = subprocess.run(["/tmp/cpu-lock.sh", str(exe)], capture_output=True, text=True)
    ctext = cres.stdout + "".join(line + "\n" for line in hlines)
    bres = subprocess.run(["/tmp/cpu-lock.sh", BEND, TABLE], cwd=REPO, capture_output=True, text=True, check=True)
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
    print("attn_stride_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
