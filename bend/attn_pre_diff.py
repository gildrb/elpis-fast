#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Check bend/attn_pre.bend differentially against the patched engine CUDA source."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

HERE = Path(__file__).resolve().parent
# Argument counts: `prog --mutate NAME ...` and `prog ENGINE_PACKAGE_DIR`.
MUTATE_ARGC = 3
PATH_ARGC = 2
# The fused kernel's load / store / l4 lane guards.
LANE_GUARDS = 3

USAGE = (
    "\n"
    "Finite differential check of bend/attn_pre.bend (ext 3007 fused pre-attention "
    "kernel, addressing) and\n"
    "its unfused reference bend/attn_pre_spec.bend against the CUDA source of the "
    "patched engine tree.\n"
    "\n"
    "Quoted verbatim (argument = the patched exllamav3 package directory, e.g. a "
    "tree built from the pinned\n"
    "series) and compiled for the CPU (c++):\n"
    "  exllamav3_ext/attn_small.cu     attn_pre_gr: warps / thr / parallel_heads / "
    "blocks / threads;\n"
    "                                  attn_pre_kernel: batch, token_pos, row, the "
    "head loop header, is_q,\n"
    "                                  kv_head, the four head pointer lines, the "
    "half2 lane guard + load /\n"
    "                                  store lines, the gate copy (gs, gd, loop), "
    "the RoPE offset and pair\n"
    "                                  reads / stores (NEOX, GPT-J), token_idx / "
    "page_idx / token_phys,\n"
    "                                  chunks, warp_id / warps, the chunk loop, "
    "is_v, cc, g0, active, base,\n"
    "                                  and both quant_block_x4 calls\n"
    "  exllamav3_ext/activation.cu     deinterleave_qg_kernel: d, h, src and the q "
    "/ g source indices\n"
    "  exllamav3_ext/rope.cu           rope_kernel: the four head pointer lines, "
    "offset, NEOX / GPT-J pairs\n"
    "  exllamav3_ext/cache/q_cache_kernels.cuh  quant_cache_paged_kernel "
    "(batch_idx .. both quant calls,\n"
    "                                  the g0 return) and quant_block_x4's lane "
    "loads and store guards\n"
    "  exllamav3_ext/cache/q_cache.cu  quant_cache_paged_gr: groups_per_token, "
    "chunks_per_token,\n"
    "                                  tb_per_token, tb_usage, blocks, threads\n"
    "  exllamav3_ext/util.h            CEIL_DIVIDE, MIN\n"
    "The C++ program replays both launch grids (head loops, lanes, chunk loops, "
    "unfused warps) with the\n"
    "quoted lines, prints the same table as bend/ATTN_PRE_TABLE.bend (compared "
    "byte for byte), and checks\n"
    "on every instance: each q half written once and each q-half of qg read once, "
    "each g half written once\n"
    "and each gate half of qg read once, each k half read / written once, each v "
    "half read once, every\n"
    "cache word / scale of the round written exactly once and by the same (base, "
    "source span) as the\n"
    "unfused quant_cache_paged_kernel, and the RoPE pairs of both kernels equal "
    "and partitioning the\n"
    "rotated span. Modelling assumption (both sides): the K quantization reads the "
    "stored k head (sh_head\n"
    "after store_head); the harness passes that head's pointer for `sh_head`.\n"
    "\n"
    "Differential evidence on finite instances, not a proof. `--mutate NAME` "
    "applies a deliberate\n"
    "mutation of attn_small.cu that the check must reject.\n"
    "\n"
    "Usage: python3 attn_pre_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR\n"
    "  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py.\n"
)

TABLE = "ATTN_PRE_TABLE.bend"

# Must match ATTN_PRE_TABLE.bend main
HS = [(28, 1), (28, 2), (28, 3), (28, 4), (28, 8), (5, 2), (4, 8), (1, 1)]
# nq, nk, hd, bsz, S, P, bits, bps, sl0, sl1
INSTS = [
    (24, 4, 256, 1, 8, 4, 3, 150, 1000, 0),
    (6, 2, 128, 2, 3, 3, 4, 4, 254, 511),
    (3, 1, 256, 1, 1, 8, 3, 2, 0, 0),
    (8, 8, 256, 1, 2, 2, 3, 3, 300, 0),
]
RS = [(64, 1, 1), (64, 1, 0), (8, 3, 1), (8, 3, 0), (256, 1, 1)]

MUTATIONS = {
    # q heads read the interleaved row at the q-only stride
    "qg_stride": (
        "g_head_in_ptr = qg + ((size_t) row * num_heads_q + head_idx) * 2 * head_dim;",
        "g_head_in_ptr = qg + ((size_t) row * num_heads_q + head_idx) * head_dim;",
    ),
    # the gate copy reads the q half
    "gate_half": (
        "const uint4* gs = (const uint4*) (g_head_in_ptr + head_dim);",
        "const uint4* gs = (const uint4*) (g_head_in_ptr);",
    ),
    # head loop strides by one block's rows only (heads visited by several blocks)
    "head_step": ("head_idx += gridDim.z * blockDim.y)", "head_idx += blockDim.y)"),
    # quant group offset at half the group width
    "g0_div": (
        "int g0 = (kv_head * head_dim + cc * 128) / 32;",
        "int g0 = (kv_head * head_dim + cc * 128) / 16;",
    ),
    # page slot of the absolute position taken modulo half a page
    "page_slot": (
        "* CQ_PAGE_SIZE + (token_idx % CQ_PAGE_SIZE);",
        "* CQ_PAGE_SIZE + (token_idx % 128);",
    ),
    # V span indexed by the loop counter instead of the chunk
    "v_span": ("* head_dim + cc * 128,", "* head_dim + c * 128,"),
    # NEOX partner off by one
    "neox_pair": (
        "float v2 = __half2float(sh_head[offset + t + partial_head_dim / 2]);",
        "float v2 = __half2float(sh_head[offset + t + partial_head_dim / 2 + 1]);",
    ),
    # NEOX partner off by one in both the read and the store (consistent, wrong pairing)
    "neox_pair_both": (
        "offset + t + partial_head_dim / 2]",
        "offset + t + partial_head_dim / 2 + 1]",
        2,
    ),
    # launch grid one block short
    "grid_short": (
        "dim3 blocks(seq_len, bsz, CEIL_DIVIDE(heads, parallel_heads));",
        "dim3 blocks(seq_len, bsz, heads / parallel_heads);",
    ),
}


def fail(msg: str) -> NoReturn:
    """Exit with an attn_pre_diff error message.

    Args:
        msg: The message to report.

    Raises:
        SystemExit: Always, carrying the prefixed message.

    """
    text = f"attn_pre_diff: {msg}"
    raise SystemExit(text)


def one(text: str, pattern: str, group: int = 0) -> str:
    """Return the unique regex match of pattern in text.

    Args:
        text: The text to search.
        pattern: The regex; with several groups, `group` selects one.
        group: The group index used when the pattern has several groups.

    Returns:
        The matched string.

    """
    m = re.findall(pattern, text)
    if len(m) != 1:
        fail(f"pattern matched {len(m)} times, expected 1: {pattern!r}")
    first = m[0]
    x = first[group] if isinstance(first, tuple) else first
    if not isinstance(x, str):
        fail(f"pattern matched a non-string: {pattern!r}")
    return x


def line(text: str, needle: str) -> str:
    """Return the unique source line containing needle, comment-free and stripped.

    Args:
        text: The source text.
        needle: The substring identifying the line.

    Returns:
        The line without its `//` comment and surrounding whitespace.

    """
    hits = [
        re.sub(r"\s*//.*$", "", ln).strip() for ln in text.splitlines() if needle in ln
    ]
    if len(hits) != 1:
        fail(f"line not unique ({len(hits)}): {needle!r}")
    return hits[0]


def body(text: str, marker: str) -> str:
    """Return the text from the unique marker through its body's closing brace.

    Args:
        text: The source text.
        marker: The unique text preceding the body.

    Returns:
        The marker through the brace closing the body that follows it.

    """
    start = text.find(marker)
    if start < 0 or text.find(marker, start + 1) >= 0:
        fail(f"marker not unique: {marker!r}")
    i = text.find("{", start)
    depth = 0
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1


def inside(stmt: str, pat: str) -> str:
    """Return the unique match of pat inside a quoted statement.

    Args:
        stmt: The quoted statement.
        pat: The regex to match.

    Returns:
        The matched string.

    """
    return one(stmt, pat)


def _engine_bodies(root: Path, mutate: str | None) -> dict[str, str]:
    """Read the engine sources, apply the mutation, and cut out the quoted bodies.

    Args:
        root: The patched exllamav3 package directory.
        mutate: The mutation of attn_small.cu to apply, if any.

    Returns:
        The util.h and q_cache_kernels.cuh texts and the quoted function bodies.

    """
    ext = root / "exllamav3_ext"
    small = (ext / "attn_small.cu").read_text()
    if mutate:
        a, b, *n = MUTATIONS[mutate]
        if small.count(a) != (n[0] if n else 1):
            fail(f"mutation {mutate} does not apply ({small.count(a)} occurrences)")
        small = small.replace(a, b)
    act = (ext / "activation.cu").read_text()
    rope = (ext / "rope.cu").read_text()
    qk = (ext / "cache/q_cache_kernels.cuh").read_text()
    qc = (ext / "cache/q_cache.cu").read_text()
    util = (ext / "util.h").read_text()
    return {
        "util": util,
        "qk": qk,
        "kern": body(small, "void attn_pre_kernel\n"),
        "host": body(small, "void attn_pre_gr\n"),
        "deint": body(act, "void deinterleave_qg_kernel\n"),
        "ropek": body(rope, "void rope_kernel\n"),
        "qpk": body(qk, "void quant_cache_paged_kernel\n"),
        "qb4": body(qk, "__device__ __forceinline__ void quant_block_x4\n"),
        "qgr": body(qc, "void quant_cache_paged_gr\n"),
    }


def _quote_lines(
    q: dict[str, str], text: str, needles: tuple[tuple[str, str], ...]
) -> None:
    """Quote the unique line of text containing each needle under its key.

    Args:
        q: The quotes, updated in place.
        text: The source text.
        needles: (key, needle) pairs in quoting order.

    """
    for k, n in needles:
        q[k] = line(text, n)


def quotes(root: Path, mutate: str | None) -> dict[str, str]:
    """Quote every engine source fragment the C++ program is built from.

    Args:
        root: The patched exllamav3 package directory.
        mutate: The mutation of attn_small.cu to apply, if any.

    Returns:
        The quoted fragments by template key.

    """
    src = _engine_bodies(root, mutate)
    q: dict[str, str] = {}
    q["CEIL"] = line(src["util"], "#define CEIL_DIVIDE(")
    q["MINM"] = line(src["util"], "#define MIN(")
    q["PAGE"] = line(src["qk"], "#define CQ_PAGE_SIZE")
    q["MAXW"] = line(src["qk"], "#define MAX_WARPS")
    _quote_fused(q, src["host"], src["kern"])
    _quote_unfused(q, src["deint"], src["ropek"], src["qpk"])
    _quote_quant_block(q, src["qb4"], src["qgr"])
    _check_neox_stores(q)
    return q


def _quote_fused(q: dict[str, str], host: str, kern: str) -> None:
    """Quote the fused launcher and kernel.

    Args:
        q: The quotes, updated in place.
        host: The attn_pre_gr body.
        kern: The attn_pre_kernel body.

    """
    # fused launcher
    _quote_lines(
        q,
        host,
        (
            ("F_WARPS", "int warps = CEIL_DIVIDE(head_dim / 2, 32);"),
            ("F_THR", "int thr = warps * 32;"),
            ("F_PAR", "int parallel_heads = MIN("),
            ("F_BLOCKS", "dim3 blocks("),
            ("F_THREADS", "dim3 threads("),
        ),
    )
    # fused kernel
    _quote_lines(
        q,
        kern,
        (
            ("F_BATCH", "int batch = blockIdx.y;"),
            ("F_TOK", "int token_pos = blockIdx.x;"),
            ("F_ROW", "const int row = "),
            ("F_ISQ", "const bool is_q = "),
            ("F_KVH", "const int kv_head = "),
            ("F_QIN", "g_head_in_ptr = qg + "),
            ("F_QOUT", "g_head_out_ptr = q_out + "),
            ("F_KOUT", "g_head_out_ptr = k + "),
            ("F_KIN", "g_head_in_ptr = g_head_out_ptr;"),
            ("F_LOAD", "((half2*) sh_head)[t] = ((half2*)g_head_in_ptr)[t];"),
            ("F_STORE", "((half2*) g_head_out_ptr)[t] = ((half2*) sh_head)[t];"),
            ("F_GS", "const uint4* gs = "),
            ("F_GD", "uint4* gd = "),
            ("F_GLOOP", "for (int i = t; i < head_dim / 8"),
            ("F_GCOPY", "gd[i] = gs[i];"),
            ("F_OFFSET", "int offset = "),
            ("F_NV1", "float v1 = __half2float(sh_head[offset + t]);"),
            ("F_NV2", "float v2 = __half2float(sh_head[offset + t + partial_head_dim"),
            ("F_NS1", "sh_head[offset + t] = __float2half_rn(r1);"),
            ("F_NS2", "= __float2half_rn(r2);"),
            ("F_GPTJ", "half2 *tptr = (half2*)(sh_head + offset"),
            ("F_TI", "int token_idx = "),
            ("F_PI", "int page_idx = "),
            ("F_PHYS", "int token_phys = "),
            ("F_WID", "int warp_id = threadIdx.x >> 5;"),
            ("F_WS", "int warps = blockDim.x >> 5;"),
            ("F_CH", "int chunks = head_dim / 128;"),
            ("F_CLOOP", "for (int c = warp_id;"),
            ("F_ISV", "bool is_v = "),
            ("F_CC", "int cc = "),
            ("F_G0", "int g0 = "),
            ("F_ACT", "int active = "),
            ("F_BASE", "int base = "),
            ("F_KQ", "quant_block_x4<BITS>(sh_head + "),
        ),
    )
    hdr = one(
        kern,
        r"(for \(int head_idx = [^;]*;\s*head_idx < [^;]*;\s*head_idx \+= [^)]*\))",
    )
    q["F_HLOOP"] = " ".join(hdr.split())
    q["F_VQ"] = " ".join(one(kern, r"(quant_block_x4<BITS>\(v \+ [^;]*;)").split())
    # lane guards: the three `if (t < head_dim / 2)` of load / store / l4 are
    # identical text
    if kern.count("if (t < head_dim / 2)") != LANE_GUARDS:
        fail("fused kernel: expected three `if (t < head_dim / 2)` guards")
    q["F_LGUARD"] = "t < head_dim / 2"
    q["F_PGUARD"] = one(
        kern, r"\n\s+if \((t < partial_head_dim / 2)\)\n\s+\{\n\s+float sin"
    )


def _quote_unfused(q: dict[str, str], deint: str, ropek: str, qpk: str) -> None:
    """Quote the unfused deinterleave, RoPE, and paged quantization kernels.

    Args:
        q: The quotes, updated in place.
        deint: The deinterleave_qg_kernel body.
        ropek: The rope_kernel body.
        qpk: The quant_cache_paged_kernel body.

    """
    # Unfused deinterleave kernel.
    _quote_lines(
        q,
        deint,
        (
            ("D_I", "size_t i = blockIdx.x"),
            ("D_RET", "if (i >= n8) return;"),
            ("D_D", "size_t d = "),
            ("D_H", "size_t h = "),
            ("D_SRC", "size_t src = "),
            ("D_Q", "q[i] = qg["),
            ("D_G", "g[i] = qg["),
        ),
    )
    q["D_QIDX"] = inside(q["D_Q"], r"q\[i\] = qg\[(.*)\];")
    q["D_GIDX"] = inside(q["D_G"], r"g\[i\] = qg\[(.*)\];")
    # Unfused RoPE kernel.
    _quote_lines(
        q,
        ropek,
        (
            ("R_QIN", "g_head_in_ptr = q + "),
            ("R_QOUT", "g_head_out_ptr = out_q + "),
            ("R_KIN", "g_head_in_ptr = k + "),
            ("R_KOUT", "g_head_out_ptr = out_k + "),
            ("R_OFFSET", "int offset = "),
            ("R_NV1", "float v1 = __half2float(sh_head[offset + t]);"),
            ("R_NV2", "float v2 = __half2float(sh_head[offset + t + partial_head_dim"),
            ("R_GPTJ", "half2 *tptr = (half2*)(sh_head + offset"),
        ),
    )
    # Unfused quant kernel.
    _quote_lines(
        q,
        qpk,
        (
            ("U_B", "int batch_idx = blockIdx.z;"),
            ("U_TI", "int token_idx = "),
            ("U_PI", "int page_idx = "),
            ("U_POS", "int token_pos = "),
            ("U_IN", "int in_pos = "),
            ("U_W", "int warp = threadIdx.x >> 5;"),
            ("U_G0", "int g0 = "),
            ("U_RET", "if (g0 >= groups_per_token) return;"),
            ("U_ACT", "int active = "),
            ("U_BASE", "int base = "),
            ("U_INB", "int in_base = "),
            ("U_KQ", "quant_block_x4<k_bits>("),
            ("U_VQ", "quant_block_x4<v_bits>("),
        ),
    )


def _quote_quant_block(q: dict[str, str], qb4: str, qgr: str) -> None:
    """Quote quant_block_x4's lane addressing and the paged quantization launcher.

    Args:
        q: The quotes, updated in place.
        qb4: The quant_block_x4 body.
        qgr: The quant_cache_paged_gr body.

    """
    q["B_X01"] = inside(line(qb4, "half2 x01 = ((const half2*) in)["), r"\[(.*)\];")
    q["B_X23"] = inside(line(qb4, "half2 x23 = ((const half2*) in)["), r"\[(.*)\];")
    q["B_WG"] = one(qb4, r"\n\s+if \(([^\n]*)\)\n\s+out\[lane\] = sh_pack\[lane\];")
    q["B_SG"] = one(qb4, r"\n\s+if \(([^\n]*)\)\n\s+out_scales\[lane\] = ")
    _quote_lines(
        q,
        qgr,
        (
            ("G_GPT", "int groups_per_token = dim / 32;"),
            ("G_CPT", "int chunks_per_token = "),
            ("G_TB", "int tb_per_token = "),
            ("G_TU", "int tb_usage = "),
            ("G_BL", "dim3 blocks(tb_per_token"),
            ("G_TH", "dim3 threads(32 * tb_usage);"),
        ),
    )


def _check_neox_stores(q: dict[str, str]) -> None:
    """Check that the quoted fused NEOX stores use their reads' indices.

    Args:
        q: The quotes.

    """
    # consistency of quoted pairs: the stores use the reads' indices
    if inside(q["F_NV2"], r"sh_head\[(.*)\]\);") != inside(
        q["F_NS2"], r"sh_head\[(.*)\] = "
    ):
        fail("fused NEOX store index differs from its read index")
    if inside(q["F_NV1"], r"sh_head\[(.*)\]\);") != inside(
        q["F_NS1"], r"sh_head\[(.*)\] = "
    ):
        fail("fused NEOX low store index differs from its read index")


C_TEMPLATE = (
    r"""
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
using std::min; using std::max;
@@CEIL@@
@@MINM@@
@@PAGE@@
@@MAXW@@
typedef uint16_t half;
struct half2 { half x, y; };
struct uint4 { uint32_t a, b, c, d; };
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, """
    r"""unsigned c = 1) : x(a), y(b), z(c) {} };
static dim3 blockIdx, blockDim, gridDim, threadIdx;
static int bad = 0;
#define BAD(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); bad """
    r"""= 1; } while (0)

static std::string summ(const std::vector<long long>& xs) {
    uint32_t h = 2166136261u;
    for (long long x : xs) h = (h ^ (uint32_t) x) * 16777619u;
    return std::to_string(h) + ":" + std::to_string(xs.size());
}
static std::string rng(long long a, long long n) { return std::to_string(a) """
    r"""+ "-" + std::to_string(a + n - 1); }

// Buffers: element index = pointer - base (every buffer is its own array)
static half *QG, *QO, *GO, *KB, *VB, *KS, *VS;
static uint32_t *KC, *VC;
static int *SEQ, *BT;

// quant_block_x4 lane addressing, quoted: loads in[X01], in[X23] (half2), """
    r"""word guard, scale guard
struct Q4 { std::vector<long long> src; long long w0 = -1, nw = 0, s0 = -1, """
    r"""ns = 0; bool contig = true; };
static Q4 q4rec;
template <int num_bits>
static void quant_block_x4(const half* in, uint32_t* out, half* out_scales, """
    r"""uint32_t* sh_pack, int active_groups, float compand_a)
{
    (void) sh_pack; (void) compand_a;
    q4rec = Q4();
    for (int lane = 0; lane < 32; ++lane) {
        long long e01 = (long long) (@@B_X01@@) * 2, e23 = (long long) (@@B_X23@@) * 2;
        q4rec.src.push_back(e01); q4rec.src.push_back(e01 + 1); """
    r"""q4rec.src.push_back(e23); q4rec.src.push_back(e23 + 1);
    }
    long long ib = (long long) (in - (const half*) 0);
    for (auto& x : q4rec.src) x += ib;
    for (int lane = 0; lane < 32; ++lane)
        if (@@B_WG@@) {
            long long w = (long long) (&out[lane] - (uint32_t*) 0);
            if (q4rec.w0 < 0) q4rec.w0 = w; else if (w != q4rec.w0 + """
    r"""q4rec.nw) q4rec.contig = false;
            q4rec.nw++;
        }
    for (int lane = 0; lane < 32; ++lane)
        if (@@B_SG@@) {
            long long s = (long long) (&out_scales[lane] - (half*) 0);
            if (q4rec.s0 < 0) q4rec.s0 = s; else if (s != q4rec.s0 + """
    r"""q4rec.ns) q4rec.contig = false;
            q4rec.ns++;
        }
}
// null-based pointers: element index of p in buffer B is p - B with B == 0
#define ZB(T) ((T*) 0)

static int bt_fn(int i) { return i * 5 + 3; }

static std::string h_line(int J, int P) {
    int heads = J, heads_per_block = P;
    int parallel_heads_ = 0;
    { @@F_PAR@@ parallel_heads_ = parallel_heads; }
    int parallel_heads = parallel_heads_, seq_len = 1, bsz = 1;
    @@F_BLOCKS@@
    std::string s = "H J=" + std::to_string(J) + " P=" + std::to_string(P) """
    r"""+ " par=" + std::to_string(parallel_heads) +
                    " Z=" + std::to_string(blocks.z);
    gridDim = blocks; blockDim = dim3(1, parallel_heads, 1);
    int num_heads_q = J, num_heads_k = 0;
    for (unsigned z = 0; z < blocks.z; ++z)
        for (unsigned y = 0; y < (unsigned) parallel_heads; ++y) {
            blockIdx = dim3(0, 0, z); threadIdx = dim3(0, y, 0);
            std::string l;
            @@F_HLOOP@@
            { l += (l.empty() ? "" : ",") + std::to_string(head_idx); }
            s += " " + std::to_string(z) + ":" + std::to_string(y) + "=" + """
    r"""(l.empty() ? "-" : l);
        }
    return s;
}

struct Counts { std::vector<int> qo, qgr, go, kb, kr, vr, kw, vw, ks, vs; };

template <int BITS>
static void inst(int nq, int nk, int hd, int bsz_, int S, int P, int bits, """
    r"""int bps, int sl0, int sl1)
{
    printf("I nq=%d nk=%d hd=%d bsz=%d S=%d P=%d bits=%d bps=%d sl=%d,%d\n", """
    r"""nq, nk, hd, bsz_, S, P, bits, bps, sl0, sl1);
    int R = bsz_ * S;
    int head_dim = hd, seq_len = S, bsz = bsz_, heads = nq + nk, heads_per_block = P;
    int num_heads_q = nq, num_heads_k = nk, blocks_per_seq = bps, """
    r"""groups_per_token = nk * hd / 32;
    int seqlens[2] = {sl0, sl1};
    int pages = 0;
    for (int b = 0; b < bsz; ++b) pages = std::max(pages, (seqlens[b] + S) / """
    r"""CQ_PAGE_SIZE + 1);
    int bt_n = bps * bsz, max_page = 0;
    std::vector<int> btv(bt_n);
    for (int i = 0; i < bt_n; ++i) { btv[i] = bt_fn(i); max_page = """
    r"""std::max(max_page, btv[i]); }
    long long cache_rows = (long long) (max_page + 1) * CQ_PAGE_SIZE;
    Counts cnt;
    cnt.qo.assign((size_t) R * nq * hd, 0); cnt.qgr.assign((size_t) R * nq * """
    r"""hd * 2, 0); cnt.go.assign((size_t) R * nq * hd, 0);
    cnt.kb.assign((size_t) R * nk * hd, 0); cnt.kr.assign((size_t) R * nk * """
    r"""hd, 0); cnt.vr.assign((size_t) R * nk * hd, 0);
    cnt.kw.assign((size_t) cache_rows * groups_per_token * BITS, 0); """
    r"""cnt.vw.assign(cnt.kw.size(), 0);
    cnt.ks.assign((size_t) cache_rows * groups_per_token, 0); """
    r"""cnt.vs.assign(cnt.ks.size(), 0);
    // fused launch geometry
    int warps_ = 0, thr_ = 0, par_ = 0;
    { @@F_WARPS@@ @@F_THR@@ @@F_PAR@@ warps_ = warps; thr_ = thr; par_ = """
    r"""parallel_heads; }
    int thr = thr_, parallel_heads = par_;
    @@F_BLOCKS@@
    @@F_THREADS@@
    gridDim = blocks; blockDim = threads;
    const half* qg = ZB(const half); half* q_out = ZB(half); half* g_out = """
    r"""ZB(half); half* k = ZB(half); const half* v = ZB(const half);
    uint32_t* k_cache = ZB(uint32_t); uint32_t* v_cache = ZB(uint32_t); """
    r"""half* k_scales = ZB(half); half* v_scales = ZB(half);
    const int* cache_seqlens = seqlens; const int* block_table = btv.data();
    uint32_t sp[32]; float compand_a = 0.0f;
    // unfused quant geometry
    int dim = nk * hd;
    int tb_per_token_ = 0, tb_usage_ = 0;
    { @@G_GPT@@ @@G_CPT@@ @@G_TB@@ @@G_TU@@ tb_per_token_ = tb_per_token; """
    r"""tb_usage_ = tb_usage; (void) groups_per_token; }
    struct Chunk { long long w0, nw, s0, ns; std::vector<long long> src; };
    std::vector<Chunk> fusedK, fusedV;
    for (int b = 0; b < bsz; ++b)
    for (int tokx = 0; tokx < S; ++tokx) {
        for (unsigned z = 0; z < blocks.z; ++z)
        for (unsigned y = 0; y < threads.y; ++y) {
            blockIdx = dim3(tokx, b, z);
            threadIdx = dim3(0, y, 0);
            @@F_BATCH@@
            @@F_TOK@@
            @@F_ROW@@
            @@F_HLOOP@@
            {
                @@F_ISQ@@
                @@F_KVH@@
                const half* g_head_in_ptr; half* g_head_out_ptr;
                if (is_q) { @@F_QIN@@ @@F_QOUT@@ }
                else { @@F_KOUT@@ @@F_KIN@@ }
                std::vector<long long> in, out, gin, gout, dq, dg, rr;
                for (int t = 0; t < (int) blockDim.x; ++t)
                    if (@@F_LGUARD@@)
                        for (int j = 0; j < 2; ++j) {
                            in.push_back((long long) ((const half*) """
    r"""&((half2*) g_head_in_ptr)[t] - ZB(const half)) + j);
                            out.push_back((long long) ((half*) &((half2*) """
    r"""g_head_out_ptr)[t] - ZB(half)) + j);
                        }
                // rope_kernel's head pointers for the same (batch, token, head)
                {
                    const half* q = ZB(const half); half* out_q = ZB(half); """
    r"""int q_head_stride = head_dim, k_head_stride = head_dim;
                    const half* g_head_in_ptr; half* g_head_out_ptr;
                    if (head_idx < num_heads_q) { @@R_QIN@@ @@R_QOUT@@ }
                    else { const half* k = ZB(const half); half* out_k = """
    r"""ZB(half); @@R_KIN@@ @@R_KOUT@@ }
                    for (int t = 0; t < head_dim / 2; ++t)
                        for (int j = 0; j < 2; ++j) {
                            long long ri = (long long) (&((const half2*) """
    r"""g_head_in_ptr)[t].x - ZB(const half)) + j;
                            long long ro = (long long) (&((half2*) """
    r"""g_head_out_ptr)[t].x - ZB(half)) + j;
                            if (ri != ro) BAD("rope in/out pointers differ");
                            rr.push_back(ro);
                        }
                }
                if (is_q) {
                    char line[4096];
                    for (int t = 0; t < (int) blockDim.x; ++t) {
                        @@F_GS@@
                        @@F_GD@@
                        @@F_GLOOP@@
                        for (int j = 0; j < 8; ++j) {
                            gin.push_back((long long) ((const half*) &gs[i] """
    r"""- ZB(const half)) + j);
                            gout.push_back((long long) ((half*) &gd[i] - ZB(half)) + j);
                        }
                    }
                    // deinterleave_qg_kernel: source of each q / g half written
                    size_t hd8 = head_dim / 8;
                    auto dsrc = [&](long long e, bool isg) -> long long {
                        blockDim = dim3(256); blockIdx = dim3((unsigned) ((e """
    r"""/ 8) / 256)); threadIdx = dim3((unsigned) ((e / 8) % 256));
                        size_t n8 = (size_t) R * num_heads_q * head_dim / 8;
                        @@D_I@@
                        if (i >= n8) { BAD("deinterleave thread out of """
    r"""range"); return -1; }
                        @@D_D@@
                        @@D_H@@
                        @@D_SRC@@
                        size_t vi = isg ? (size_t) (@@D_GIDX@@) : (size_t) (@@D_QIDX@@);
                        return (long long) vi * 8 + e % 8;
                    };
                    for (long long e : out) dq.push_back(dsrc(e, false));
                    for (long long e : gout) dg.push_back(dsrc(e, true));
                    blockDim = threads; blockIdx = dim3(tokx, b, z); """
    r"""threadIdx = dim3(0, y, 0);
                    for (long long e : out) cnt.qo[e]++;
                    for (long long e : in) cnt.qgr[e]++;
                    for (long long e : gin) cnt.qgr[e]++;
                    for (long long e : gout) cnt.go[e]++;
                    snprintf(line, sizeof line, "Q b=%d tok=%d h=%d qin=%s """
    r"""qout=%s gin=%s gout=%s dq=%s dg=%s rq=%s", b, tokx, head_idx,
                             summ(in).c_str(), summ(out).c_str(), """
    r"""summ(gin).c_str(), summ(gout).c_str(), summ(dq).c_str(),
                             summ(dg).c_str(), summ(rr).c_str());
                    puts(line);
                } else {
                    for (long long e : out) cnt.kb[e]++;
                    std::string s = "K b=" + std::to_string(b) + " tok=" + """
    r"""std::to_string(tokx) + " kvh=" + std::to_string(kv_head) +
                                    " k=" + summ(out) + " rk=" + summ(rr);
                    half* sh_head = g_head_out_ptr;   // modelling """
    r"""assumption: the stored k head
                    @@F_TI@@
                    @@F_PI@@
                    @@F_PHYS@@
                    for (int tx = 0; tx < (int) blockDim.x; tx += 32) {
                        threadIdx = dim3(tx, y, 0);
                        @@F_WID@@
                        @@F_WS@@
                        @@F_CH@@
                        s += " w" + std::to_string(warp_id) + ":";
                        @@F_CLOOP@@
                        {
                            @@F_ISV@@
                            @@F_CC@@
                            @@F_G0@@
                            @@F_ACT@@
                            @@F_BASE@@
                            if (!is_v) {
                                @@F_KQ@@
                                for (long long e : q4rec.src) cnt.kr[e]++;
                                for (long long w = 0; w < q4rec.nw; ++w) """
    r"""cnt.kw[q4rec.w0 + w]++;
                                for (long long w = 0; w < q4rec.ns; ++w) """
    r"""cnt.ks[q4rec.s0 + w]++;
                                fusedK.push_back({q4rec.w0, q4rec.nw, """
    r"""q4rec.s0, q4rec.ns, q4rec.src});
                            } else {
                                @@F_VQ@@
                                for (long long e : q4rec.src) cnt.vr[e]++;
                                for (long long w = 0; w < q4rec.nw; ++w) """
    r"""cnt.vw[q4rec.w0 + w]++;
                                for (long long w = 0; w < q4rec.ns; ++w) """
    r"""cnt.vs[q4rec.s0 + w]++;
                                fusedV.push_back({q4rec.w0, q4rec.nw, """
    r"""q4rec.s0, q4rec.ns, q4rec.src});
                            }
                            if (!q4rec.contig) BAD("fused words / scales not """
    r"""contiguous");
                            s += " " + std::to_string(c) + "/" + """
    r"""std::to_string(cc) + "/" + (is_v ? "1" : "0") + "/" + std::to_string(g0) +
                                 "/" + std::to_string(base) + "/src=" + """
    r"""summ(q4rec.src) + "/words=" + rng(q4rec.w0, q4rec.nw) +
                                 "/scales=" + rng(q4rec.s0, q4rec.ns);
                        }
                    }
                    threadIdx = dim3(0, y, 0);
                    puts(s.c_str());
                }
            }
        }
        // unfused quant_cache_paged_kernel of this (b, token): grid """
    r"""(tb_per_token, seq_len, bsz) x 32 tb_usage
        {
            int tb_per_token = tb_per_token_, tb_usage = tb_usage_;
            @@G_BL@@
            @@G_TH@@
            dim3 sgrid = gridDim, sblock = blockDim;
            gridDim = blocks; blockDim = threads;
            const half* k_in = ZB(const half); const half* v_in = ZB(const half);
            uint32_t* k_out = ZB(uint32_t); uint32_t* v_out = ZB(uint32_t); """
    r"""half* k_out_scales = ZB(half); half* v_out_scales = ZB(half);
            constexpr int k_bits = BITS, v_bits = BITS; int in_contiguous = 1;
            uint32_t sh_pack[8][32];
            std::string s = "U b=" + std::to_string(b) + " tok=" + """
    r"""std::to_string(tokx) + " tb=" + std::to_string(blocks.x) +
                            " usage=" + std::to_string(threads.x / 32);
            for (unsigned bx = 0; bx < blocks.x; ++bx)
            for (unsigned tx = 0; tx < threads.x; tx += 32) {
                blockIdx = dim3(bx, tokx, b); threadIdx = dim3(tx);
                [&]() {
                    @@U_B@@
                    @@U_TI@@
                    @@U_PI@@
                    @@U_POS@@
                    @@U_IN@@
                    @@U_W@@
                    @@U_G0@@
                    s += " " + std::to_string(bx) + ":" + std::to_string(warp);
                    if (g0 >= groups_per_token) { s += "/-"; }
                    @@U_RET@@
                    @@U_ACT@@
                    @@U_BASE@@
                    @@U_INB@@
                    @@U_KQ@@
                    Q4 kq = q4rec;
                    @@U_VQ@@
                    Q4 vq = q4rec;
                    if (kq.src != vq.src || kq.w0 != vq.w0 || kq.s0 != """
    r"""vq.s0) BAD("unfused K and V addressing differ");
                    // exactly the fused chunk with the same destination (K and V)
                    for (auto* F : {&fusedK, &fusedV}) {
                        int hits = 0;
                        for (auto& f : *F)
                            if (f.w0 == kq.w0) { hits++; if (f.nw != kq.nw """
    r"""|| f.s0 != kq.s0 || f.ns != kq.ns || f.src != kq.src) BAD("fused chunk """
    r"""differs from unfused at word %lld", kq.w0); }
                        if (hits != 1) BAD("unfused chunk at word %lld """
    r"""matched %d fused chunks", kq.w0, hits);
                    }
                    s += "/" + std::to_string(g0) + "/" + """
    r"""std::to_string(base) + "/src=" + summ(kq.src) + "/words=" + rng(kq.w0, """
    r"""kq.nw) +
                         "/scales=" + rng(kq.s0, kq.ns);
                }();
            }
            gridDim = sgrid; blockDim = sblock;
            puts(s.c_str());
            fusedK.clear(); fusedV.clear();
        }
    }
    // exactly-once checks over the whole round
    auto once = [&](const std::vector<int>& v, const char* what) {
        for (size_t i = 0; i < v.size(); ++i) if (v[i] != 1) { BAD("%s: """
    r"""element %zu touched %d times", what, i, v[i]); return; }
    };
    once(cnt.qo, "q written"); once(cnt.qgr, "qg read"); once(cnt.go, "g """
    r"""written"); once(cnt.kb, "k written");
    once(cnt.kr, "k quantized"); once(cnt.vr, "v quantized");
    // cache: exactly the round's rows, each word / scale once
    for (int b = 0; b < bsz; ++b)
        for (int tokx = 0; tokx < S; ++tokx) {
            int ti = tokx + seqlens[b];
            long long pos = (long long) btv[bps * b + ti / CQ_PAGE_SIZE] * """
    r"""CQ_PAGE_SIZE + ti % CQ_PAGE_SIZE;
            for (int g = 0; g < groups_per_token; ++g) {
                if (cnt.ks[pos * groups_per_token + g] != 1 || cnt.vs[pos * """
    r"""groups_per_token + g] != 1) BAD("scale of row %lld group %d not written """
    r"""once", pos, g);
                cnt.ks[pos * groups_per_token + g] = cnt.vs[pos * """
    r"""groups_per_token + g] = 0;
                for (int w = 0; w < BITS; ++w) {
                    long long a = (pos * groups_per_token + g) * BITS + w;
                    if (cnt.kw[a] != 1 || cnt.vw[a] != 1) BAD("word %lld not """
    r"""written once", a);
                    cnt.kw[a] = cnt.vw[a] = 0;
                }
            }
        }
    for (auto* v : {&cnt.kw, &cnt.vw, &cnt.ks, &cnt.vs})
        for (int x : *v) if (x) { BAD("cache written outside the round's """
    r"""rows"); break; }
}

static std::string pair_s(long long a, long long b) { return """
    r"""std::to_string(a) + "/" + std::to_string(b); }

static void r_line(int partial_head_dim, int rotate_dims, int neox)
{
    std::string s = "R pd=" + std::to_string(partial_head_dim) + " rd=" + """
    r"""std::to_string(rotate_dims) + " neox=" + std::to_string(neox);
    std::vector<int> seen((size_t) partial_head_dim * rotate_dims, 0);
    half* sh_head = ZB(half);
    for (int rdim = 0; rdim < rotate_dims; ++rdim) {
        s += " |";
        for (int t = 0; t < 1024; ++t) {
            if (!(@@F_PGUARD@@)) continue;
            long long fa, fb, ua, ub;
            {
                @@F_OFFSET@@
                if (neox) { fa = (long long) (&(@@F_NV1E@@) - ZB(half)); fb """
    r"""= (long long) (&(@@F_NV2E@@) - ZB(half)); }
                else { @@F_GPTJ@@ fa = (long long) (&tptr->x - ZB(half)); fb = fa + 1; }
            }
            {
                int rotate_offset = 0;
                @@R_OFFSET@@
                if (neox) { ua = (long long) (&(@@R_NV1E@@) - ZB(half)); ub """
    r"""= (long long) (&(@@R_NV2E@@) - ZB(half)); }
                else { @@R_GPTJ@@ ua = (long long) (&tptr->x - ZB(half)); ub = ua + 1; }
            }
            s += " " + pair_s(fa, fb) + "=" + pair_s(ua, ub);
            for (long long e : {fa, fb}) {
                if (e < 0 || e >= (long long) seen.size()) BAD("rope pair """
    r"""element %lld outside the rotated span", e);
                else seen[e]++;
            }
        }
    }
    for (size_t i = 0; i < seen.size(); ++i) if (seen[i] != 1) { BAD("rope """
    r"""element %zu in %d pairs", i, seen[i]); break; }
    puts(s.c_str());
}

int main()
{
@@CALLS@@
    if (bad) { fprintf(stderr, "C-side violation\n"); return 1; }
    fprintf(stderr, "C checks: every q / g / k half written once, every qg / """
    r"""k / v half read once, cache rows of the round written once and as the """
    r"""unfused kernel, RoPE pairs equal and partitioning\n");
    return 0;
}
"""
)


def c_program(q: dict[str, str]) -> str:
    """Fill the C++ template with the quotes and the table calls.

    Args:
        q: The quoted engine fragments.

    Returns:
        The complete C++ program text.

    """
    src = C_TEMPLATE
    q = dict(q)
    q["F_NV1E"] = one(q["F_NV1"], r"__half2float\((sh_head\[.*\])\);")
    q["F_NV2E"] = one(q["F_NV2"], r"__half2float\((sh_head\[.*\])\);")
    q["R_NV1E"] = one(q["R_NV1"], r"__half2float\((sh_head\[.*\])\);")
    q["R_NV2E"] = one(q["R_NV2"], r"__half2float\((sh_head\[.*\])\);")
    calls = [f"    puts(h_line({J}, {P}).c_str());" for J, P in HS]
    calls += [f"    inst<{i[6]}>({', '.join(map(str, i))});" for i in INSTS]
    calls += [f"    r_line({pd}, {rd}, {nx});" for pd, rd, nx in RS]
    q["CALLS"] = "\n".join(calls)
    for k, v in q.items():
        src = src.replace(f"@@{k}@@", v)
    left = re.findall(r"@@\w+@@", src)
    if left:
        fail(f"unfilled quotes: {left}")
    return src


def main(argv: list[str]) -> None:
    """Run the differential check.

    Args:
        argv: The command line: [--mutate NAME] ENGINE_PACKAGE_DIR.

    """
    mutate = None
    if len(argv) >= MUTATE_ARGC and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0], *argv[3:]]
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate}; known: {', '.join(MUTATIONS)}")
    if len(argv) != PATH_ARGC:
        fail(USAGE)
    root = Path(argv[1])
    q = quotes(root, mutate)
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "diff.cpp"
        c.write_text(c_program(q))
        exe = Path(td) / "diff"
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked([
                "c++",
                "-O0",
                "-std=c++17",
                "-w",
                "-fno-strict-aliasing",
                "-o",
                str(exe),
                str(c),
            ]),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            sys.stdout.write(f"{r.stderr[-3000:]}\n")
            fail("C++ harness does not compile")
        cres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            source_link.locked([str(exe)]),
            capture_output=True,
            text=True,
            check=False,
        )
    bres = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
        source_link.locked([source_link.bend(), TABLE]),
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    )
    same = cres.stdout == bres.stdout
    status = (
        cres.stderr.strip()[-2000:] or f"C table program exit status {cres.returncode}"
    )
    sys.stdout.write(f"{status}\n")
    sys.stdout.write(
        f"table lines: C {len(cres.stdout.splitlines())}, "
        f"Bend {len(bres.stdout.splitlines())}; byte-identical: {same}\n"
    )
    if not same:
        for i, (x, y) in enumerate(
            zip(cres.stdout.splitlines(), bres.stdout.splitlines(), strict=False)
        ):
            if x != y:
                sys.stdout.write(
                    f"first difference at line {i + 1}:\n"
                    f"  C:    {x[:400]}\n  Bend: {y[:400]}\n"
                )
                break
    if not same or cres.returncode != 0:
        fail("MISMATCH" if not same else "C-side violation")
    sys.stdout.write("attn_pre_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
