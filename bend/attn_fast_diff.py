#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite source link of bend/attn_fast.bend to the patched engine text.

The model is ext 3032's int8 QK^T operand layer (EXL3_AV_FAST); the link adds host
C evidence for what the Bend model does not transcribe. Text and test evidence,
not a proof.

1. Every kernel line the model transcribes (QS8, the s32.s8.s8.s32 IMMA asm string,
   qk8_slot, k3_s8, i2f_exact, the q-scale base, the per-(row, group) absmax and
   quantisation, the q8 / qsc8 stores, the QK8 operand loads, the IMMA and the four
   group FMA lines) occurs exactly once in exllamav3_ext/attn_verify.cu, in the
   order the model assumes; the q clamp is pattn8i's (|q8| <= 127).
2. The kernels and the switch: attn_verify.cuh defines AVF_QK8 1 and AVF_PV16 2 and
   declares the six fast 3-bit kernels; attn_verify.cu instantiates them as
   split_body<3, 3, TREE, F> for F = 1, 2, 3; attn_verify_gr.cu maps mode m of the
   kernel table to suffix _f<m> (mode 0 the exact kernels) and parses EXL3_AV_FAST
   once in a static lambda.
3. Host C (the dev shell's c++, -O2 -ffp-contract=off, no fast-math; run via
   source_link.run), compiled from the quoted kernel text (plane_base, plane_one,
   qk8_slot, k3_s8, i2f_exact, the EXL3_AV_FAST lambda):
   - k3_s8: for every lane t < 4, half h < 2, byte i < 4 and every 3-bit code at
     that byte's dim 16 h + t + 4 i (64 random backgrounds each, plus 2^18 random
     word triples), byte i is the s8 value 2 raw - 7 of the exact kernel's own
     decoding (plane_one<3, 2>, plane_one<3, 1>) of that dim;
   - qk8_slot is a bijection of 0..31 and places dim 16 h + t + 4 i at k-slot
     16 h + 4 t + i, the slot k3_s8's lane t byte i occupies;
   - the q bytes r * QS8 + 32 g + qk8_slot(d) (r < 48, g < 8, d < 32) and the scales
     g * 48 + r are distinct, inside 48 * QS8 and 8 * 48, and fit the fp16 q region;
   - an end-to-end QK8 group: the kernel's q8 store, the ldmatrix x4 rows of the QK8
     load (PTX ldmatrix layout), k3_s8 and the m16n8k32 fragment layout (PTX) give
     C[row][token] == sum_d q8[row][d] (2 code[token][d] - 7) for random q8 in
     [-127, 127] and codes (2048 groups x 3 m-tiles x 8 tokens x 48 rows);
   - i2f_exact(x) == x for |x| <= 32 * 127 * 7 = 28448;
   - EXL3_AV_FAST: unset and "" give 1, "0" .. "3" give 0 .. 3, anything else
     ("4", "-1", "01", "3 ", "x", "10") is an error.
`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/attn_fast_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_link

KERNEL = "exllamav3_ext/attn_verify.cu"
HEADER = "exllamav3_ext/attn_verify.cuh"
LAUNCH = "exllamav3_ext/attn_verify_gr.cu"

# (model def, kernel line), in kernel order
LINES = [
    ("Impl.qaddr (QS8 = 272)", "constexpr int QS8 = AV_HD + 16;"),
    (
        "PImpl.mma (asm)",
        (
            '"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 '
            '{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"'
        ),
    ),
    ("Impl.qk8_slot", "return (d & 16) | ((d & 3) << 2) | ((d >> 2) & 3);"),
    ("Impl.kbyte (x2s)", "uint32_t x2s = (w2h >> (2 * t)) << 2;"),
    ("Impl.kbyte (y)", "uint32_t y = __byte_perm(w1 >> (16 * h + t), 0u, 0x4140);"),
    (
        "Impl.kbyte (u)",
        "uint32_t u = (x2s & 0x0C0C0C0Cu) | (((y << 1) | (y << 5)) & 0x02020202u);",
    ),
    ("Impl.kbyte", "return (u + 0x79797979u) ^ 0x80808080u;"),
    (
        "PImpl.i2f_exact",
        "return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f);",
    ),
    ("Impl.qsidx (base)", "float* qsc8 = (float*) (smem + AV_ROWS * QS8);"),
    (
        "Impl.partner (s = 1)",
        "amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));",
    ),
    (
        "Impl.partner (s = 2)",
        "amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));",
    ),
    ("q quantisation", "const float inv = amax > 0.f ? __fdiv_rn(127.f, amax) : 0.f;"),
    ("Impl.qaddr (row)", "const int r = 16 * mt + 8 * hr + gid;"),
    ("Impl.qaddr", "unsigned char* qrow = qs8 + r * QS8 + 32 * g;"),
    (
        "PImpl.clamp",
        (
            "int q8 = max(-127, min(127, __float2int_rn(__fmul_rn(c[nt][2 * hr + e], "
            "inv))));"
        ),
    ),
    (
        "Impl.qaddr (store)",
        "qrow[qk8_slot(8 * nt + 2 * t + e)] = (unsigned char) (q8 & 255);",
    ),
    (
        "Impl.qsidx (store)",
        (
            "if (t == 0) qsc8[g * AV_ROWS + r] = "
            "__fmul_rn(__fdiv_rn(amax, 127.f), 0.125f);"
        ),
    ),
    ("Impl.kbyte (b0)", "uint32_t b0 = k3_s8(gw[0], gw[2], 0, t);"),
    ("Impl.kbyte (b1)", "uint32_t b1 = k3_s8(gw[1], gw[2], 1, t);"),
    (
        "Impl.qaddr (load)",
        "ldsm_x4(a, qs8 + (16 * i + (lane & 15)) * QS8 + 32 * g + 16 * (lane >> 4));",
    ),
    ("PImpl.mma (group)", "imma16832(ci, a, b0, b1);"),
    ("Impl.qsidx (load)", "float qa = qsc8[g * AV_ROWS + 16 * i + gid];"),
    ("Impl.qsidx (load)", "float qb = qsc8[g * AV_ROWS + 16 * i + 8 + gid];"),
    (
        "group FMA",
        "sc[i][0] = fmaf(i2f_exact(ci[0]), __fmul_rn(qa, s0), sc[i][0]);",
    ),
    (
        "group FMA",
        "sc[i][1] = fmaf(i2f_exact(ci[1]), __fmul_rn(qa, s1), sc[i][1]);",
    ),
    (
        "group FMA",
        "sc[i][2] = fmaf(i2f_exact(ci[2]), __fmul_rn(qb, s0), sc[i][2]);",
    ),
    (
        "group FMA",
        "sc[i][3] = fmaf(i2f_exact(ci[3]), __fmul_rn(qb, s1), sc[i][3]);",
    ),
]

# attn_verify.cuh / .cu / attn_verify_gr.cu text of the kernels and the switch
HEADER_LINES = [
    "#define AVF_QK8 1",
    "#define AVF_PV16 2",
    *(
        f'extern "C" __global__ void attn_verify_split_k3v3{tree}_f{m}('
        for tree in ("", "_tree")
        for m in (1, 2, 3)
    ),
]
KERNEL_LINES = [
    "constexpr bool QK8 = (FAST & AVF_QK8) != 0;",
    "constexpr bool PV16 = (FAST & AVF_PV16) != 0;",
    "split_body<3, 3, false, F>(",
    "split_body<3, 3, true, F>(",
    "AV_FAST_SPLIT(1)",
    "AV_FAST_SPLIT(2)",
    "AV_FAST_SPLIT(3)",
]
LAUNCH_LINES = [
    (
        "{(void*) attn_verify_split_k3v3, (void*) attn_verify_split_k3v3_f1,\n"
        "         (void*) attn_verify_split_k3v3_f2, "
        "(void*) attn_verify_split_k3v3_f3},"
    ),
    (
        "{(void*) attn_verify_split_k3v3_tree, (void*) attn_verify_split_k3v3_tree_f1,"
        "\n         (void*) attn_verify_split_k3v3_tree_f2, "
        "(void*) attn_verify_split_k3v3_tree_f3},"
    ),
    "return k3[tree ? 1 : 0][attn_verify_fast_mode()];",
    "static const int mode = [] {",
]

MUTATIONS = {
    "slot": ("((d & 3) << 2) | ((d >> 2) & 3)", "((d & 3) << 2) | ((d >> 3) & 3)"),
    "kbias": ("(u + 0x79797979u)", "(u + 0x78787878u)"),
    "kshift": ("(w2h >> (2 * t)) << 2", "(w2h >> (2 * t)) << 1"),
    "magic": ("x + 0x4B400000", "x + 0x4B000000"),
    "env": ("env[0] <= '3'", "env[0] <= '4'"),
}
MUTATE_ARGC = 3

C_SRC = r"""
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
static_assert(FLT_EVAL_METHOD == 0, "fp32 arithmetic must be evaluated in fp32");
#define AV_HD 256
#define AV_ROWS 48
#define __forceinline__
#define __device__
// PTX prmt.b32, default mode
static uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    uint64_t v = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) {
        uint32_t sel = (s >> (4 * n)) & 7;
        r |= (uint32_t) ((v >> (8 * sel)) & 255) << (8 * n);
    }
    return r;
}
static float __int_as_float(int x) { float f; std::memcpy(&f, &x, 4); return f; }
static float __fsub_rn(float a, float b) { return a - b; }
struct TorchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
#define TORCH_CHECK(c, ...) do { if (!(c)) throw TorchError("check"); } while (0)
@@QUOTED@@
static const char* g_env = nullptr;
static int parse_mode() {
    struct Env {
        static const char* getenv(const char*) { return g_env; }
    };
    return [] {
@@LAMBDA@@
    }();
}
static int code_at(const uint32_t* gw, int j) {
    uint32_t r = 0;
    plane_one<3, 8>(gw, j, r);
    plane_one<3, 4>(gw, j, r);
    plane_one<3, 2>(gw, j, r);
    plane_one<3, 1>(gw, j, r);
    return (int) r;
}
static int s8(uint32_t w, int i) { return (int) (int8_t) (uint8_t) (w >> (8 * i)); }
// the CQ plane layout plane_one<3, 2> / <3, 1> decode
static void set_code(uint32_t* gw, int j, int c) {
    uint32_t sh = 2 * (j % 16);
    gw[j / 16] = (gw[j / 16] & ~(3u << sh)) | ((uint32_t) (c >> 1) << sh);
    gw[2] = (gw[2] & ~(1u << j)) | ((uint32_t) (c & 1) << j);
}
int main() {
    int bad = 0;
    std::mt19937 rng(3032);
    long kn = 0, kbad = 0;
    auto check_k = [&](const uint32_t* gw) {
        for (int h = 0; h < 2; ++h)
            for (int t = 0; t < 4; ++t) {
                uint32_t b = k3_s8(gw[h], gw[2], h, t);
                for (int i = 0; i < 4; ++i, ++kn)
                    if (s8(b, i) != 2 * code_at(gw, 16 * h + t + 4 * i) - 7) ++kbad;
            }
    };
    for (int j = 0; j < 32; ++j)
        for (int c = 0; c < 8; ++c)
            for (int bg = 0; bg < 64; ++bg) {
                uint32_t gw[3] = {(uint32_t) rng(), (uint32_t) rng(), (uint32_t) rng()};
                set_code(gw, j, c);
                if (code_at(gw, j) != c) ++kbad;
                check_k(gw);
            }
    for (int n = 0; n < (1 << 18); ++n) {
        uint32_t gw[3] = {(uint32_t) rng(), (uint32_t) rng(), (uint32_t) rng()};
        check_k(gw);
    }
    std::printf("k3_s8: %ld bytes vs the exact kernel's plane decoding, %ld wrong\n",
                kn, kbad);
    if (kbad) bad = 1;
    std::set<int> slots;
    long pbad = 0;
    for (int d = 0; d < 32; ++d) {
        slots.insert(qk8_slot(d));
        if (qk8_slot(d) < 0 || qk8_slot(d) > 31) ++pbad;
    }
    for (int h = 0; h < 2; ++h)
        for (int t = 0; t < 4; ++t)
            for (int i = 0; i < 4; ++i)
                if (qk8_slot(16 * h + t + 4 * i) != 16 * h + 4 * t + i) ++pbad;
    if (slots.size() != 32) ++pbad;
    std::printf("qk8_slot: %zu distinct slots of 0..31, pairing errors %ld\n",
                slots.size(), pbad);
    if (pbad) bad = 1;
    std::set<long> addr, sidx;
    long abad = 0;
    const long q8_bytes = (long) AV_ROWS * QS8;
    const long region = (long) AV_ROWS * (AV_HD + 8) * 2;
    for (int r = 0; r < AV_ROWS; ++r)
        for (int g = 0; g < 8; ++g) {
            for (int d = 0; d < 32; ++d) {
                long a = (long) r * QS8 + 32 * g + qk8_slot(d);
                if (a < 0 || a >= q8_bytes || !addr.insert(a).second) ++abad;
            }
            long s = (long) g * AV_ROWS + r;
            if (s >= 8L * AV_ROWS || !sidx.insert(s).second) ++abad;
        }
    if (q8_bytes + 8L * AV_ROWS * 4 > region) ++abad;
    std::printf("q operands: %zu q8 bytes, %zu scales distinct, %ld of %ld region "
                "bytes, errors %ld\n", addr.size(), sidx.size(),
                q8_bytes + 8L * AV_ROWS * 4, region, abad);
    if (abad) bad = 1;
    // End to end: q8 store -> ldmatrix x4 (QK8 rows) -> m16n8k32 fragments -> IMMA
    static uint8_t qsm[AV_ROWS * QS8];
    std::uniform_int_distribution<int> qd(-127, 127), cd(0, 7);
    long gbad = 0, gn = 0;
    for (int rep = 0; rep < 2048; ++rep) {
        const int g = rep & 7;
        int q8v[AV_ROWS][32], code[8][32];
        uint32_t kw[8][3];
        for (int r = 0; r < AV_ROWS; ++r)
            for (int d = 0; d < 32; ++d) {
                q8v[r][d] = qd(rng);
                uint8_t* qrow = qsm + r * QS8 + 32 * g;
                qrow[qk8_slot(d)] = (uint8_t) (q8v[r][d] & 255);
            }
        for (int n = 0; n < 8; ++n) {
            for (int w = 0; w < 3; ++w) kw[n][w] = (uint32_t) rng();
            for (int d = 0; d < 32; ++d) {
                code[n][d] = cd(rng);
                set_code(kw[n], d, code[n][d]);
            }
        }
        for (int i = 0; i < 3; ++i) {
            uint32_t a[32][4], b0[32], b1[32];
            for (int lane = 0; lane < 32; ++lane) {
                // ldmatrix x4: the rows of matrix k come from lanes 8 k .. 8 k + 7
                for (int k = 0; k < 4; ++k) {
                    int src = 8 * k + (lane >> 2);
                    const uint8_t* row =
                        qsm + (16 * i + (src & 15)) * QS8 + 32 * g + 16 * (src >> 4);
                    std::memcpy(&a[lane][k], row + 4 * (lane & 3), 4);
                }
                int gid = lane >> 2, t = lane & 3;
                b0[lane] = k3_s8(kw[gid][0], kw[gid][2], 0, t);
                b1[lane] = k3_s8(kw[gid][1], kw[gid][2], 1, t);
            }
            for (int row = 0; row < 16; ++row)
                for (int col = 0; col < 8; ++col, ++gn) {
                    long acc = 0, ref = 0;
                    for (int kk = 0; kk < 32; ++kk) {   // PTX m16n8k32 s8 fragments
                        int ta = 4 * (row % 8) + (kk % 16) / 4;
                        int ra = row / 8 + 2 * (kk / 16);
                        int tb = 4 * col + (kk % 16) / 4;
                        long av = s8(a[ta][ra], kk % 4);
                        long bv = s8(kk < 16 ? b0[tb] : b1[tb], kk % 4);
                        acc += av * bv;
                    }
                    for (int d = 0; d < 32; ++d)
                        ref += (long) q8v[16 * i + row][d] * (2 * code[col][d] - 7);
                    if (acc != ref) ++gbad;
                }
        }
    }
    std::printf("QK8 group end to end: %ld (row, token) dots, %ld differ from "
                "sum_d q8 (2 c - 7)\n", gn, gbad);
    if (gbad) bad = 1;
    long ibad = 0;
    for (int x = -32 * 127 * 7; x <= 32 * 127 * 7; ++x)
        if (i2f_exact(x) != (float) x) ++ibad;
    std::printf("i2f_exact exact on |x| <= 28448: %ld wrong\n", ibad);
    if (ibad) bad = 1;
    struct Case { const char* v; int want; } cases[] = {
        {nullptr, 1}, {"", 1}, {"0", 0}, {"1", 1}, {"2", 2}, {"3", 3},
        {"4", -1}, {"-1", -1}, {"01", -1}, {"3 ", -1}, {"x", -1}, {"10", -1}};
    long ebad = 0;
    for (const Case& c : cases) {
        g_env = c.v;
        int got;
        try { got = parse_mode(); } catch (const TorchError&) { got = -1; }
        if (got != c.want) {
            ++ebad;
            std::printf("EXL3_AV_FAST=%s: got %d want %d\n", c.v ? c.v : "(unset)",
                        got, c.want);
        }
    }
    std::printf("EXL3_AV_FAST parse: 12 cases, %ld wrong\n", ebad);
    if (ebad) bad = 1;
    return bad;
}
"""


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"attn_fast_diff: FAIL: {msg}"
    raise SystemExit(text)


def load_src(argv: list[str]) -> dict[str, str]:
    """Parse the command line, read the sources and apply the mutation.

    Args:
        argv: The command line.

    Returns:
        The (possibly mutated) kernel, header and launcher sources.

    """
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: attn_fast_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    src: dict[str, str] = {
        KERNEL: (root / KERNEL).read_text(),
        HEADER: (root / HEADER).read_text(),
        LAUNCH: (root / LAUNCH).read_text(),
    }
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}")
        old, new = MUTATIONS[mutate]
        hits = [name for name, text in src.items() if text.count(old) == 1]
        if len(hits) != 1:
            fail(f"mutation anchor {old!r}")
        src[hits[0]] = src[hits[0]].replace(old, new)
        sys.stdout.write(f"attn_fast_diff: applied mutation {mutate}\n")
    return src


def check_lines(src: dict[str, str], errors: list[str]) -> None:
    """Record every transcribed line not found once, or out of order."""
    pos: list[int] = []
    cu = src[KERNEL]
    for what, line in LINES:
        n = cu.count(line)
        if n != 1:
            errors.append(f"{KERNEL}: {line!r} ({what}) occurs {n} times (want 1)")
        else:
            pos.append(cu.index(line))
    if len(pos) == len(LINES) and pos != sorted(pos):
        errors.append(f"{KERNEL}: transcribed lines out of order: {pos}")
    groups = ((HEADER, HEADER_LINES), (KERNEL, KERNEL_LINES), (LAUNCH, LAUNCH_LINES))
    for name, lines in groups:
        errors.extend(
            f"{name}: {line!r} occurs {src[name].count(line)} times (want 1)"
            for line in lines
            if src[name].count(line) != 1
        )
    sys.stdout.write(
        f"attn_fast_diff: {len(LINES)} transcribed kernel lines and "
        f"{len(HEADER_LINES) + len(KERNEL_LINES) + len(LAUNCH_LINES)} kernel / switch "
        f"lines checked, {len(errors)} errors\n"
    )


def quote_fn(cu: str, sig: str) -> str:
    """Quote one device function (from its signature line to its closing brace).

    Args:
        cu: attn_verify.cu.
        sig: The regex of the signature line.

    Returns:
        The function text.

    """
    hits = re.findall(r"(?m)^" + sig + r"[^\n]*\n\{\n(?:.*\n)*?\}\n", cu)
    if len(hits) != 1:
        fail(f"cannot quote {sig!r} ({len(hits)} matches)")
    return str(hits[0])


def quoted(src: dict[str, str]) -> tuple[str, str, int]:
    """Quote the device functions, the EXL3_AV_FAST lambda body and QS8.

    Args:
        src: The sources.

    Returns:
        The device functions, the lambda body and the QS8 value.

    """
    cu = src[KERNEL]
    qs8 = re.search(r"constexpr int QS8 = AV_HD \+ (\d+);", cu)
    if not qs8:
        fail("cannot extract QS8")
    dev = r"__device__ __forceinline__ "
    tpl = r"template <int BITS, int W>"
    parts = [
        quote_fn(cu, tpl + r" __device__ constexpr int plane_base\(\)"),
        quote_fn(cu, tpl + r"\n" + dev + r"void plane_one"),
        quote_fn(cu, dev + r"int qk8_slot\(int d\)"),
        quote_fn(cu, dev + r"uint32_t k3_s8\("),
        quote_fn(cu, dev + r"float i2f_exact\(int x\)"),
    ]
    body = re.search(
        r"static const int mode = \[\] \{\n((?:.*\n)*?)    \}\(\);", src[LAUNCH]
    )
    if not body:
        fail("cannot quote the EXL3_AV_FAST lambda")
    lam = str(body[1]).replace("std::getenv(", "Env::getenv(")
    return "\n".join(parts), lam, 256 + int(qs8[1])


def run_host_c(src: dict[str, str], errors: list[str]) -> None:
    """Build and run the host C evidence from the quoted kernel text."""
    funcs, lam, qs8 = quoted(src)
    code = C_SRC.replace("@@QUOTED@@", funcs).replace("@@LAMBDA@@", lam)
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "attn_fast.cpp"
        c.write_text(code)
        exe = Path(td) / "attn_fast"
        r = source_link.run(
            [
                "c++",
                "-O2",
                "-std=c++17",
                "-ffp-contract=off",
                "-fno-fast-math",
                f"-DQS8={qs8}",
                "-o",
                str(exe),
                str(c),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            fail("C harness does not compile:\n" + r.stderr[-3000:])
        cres = source_link.run([str(exe)], capture_output=True, text=True, check=False)
    sys.stdout.write(cres.stdout.rstrip() + "\n")
    if cres.returncode != 0:
        errors.append(f"host C evidence failed (exit {cres.returncode})")


def main(argv: list[str]) -> None:
    """Run the source link."""
    src = load_src(argv)
    errors: list[str] = []
    check_lines(src, errors)
    run_host_c(src, errors)
    if errors:
        fail("; ".join(errors))
    sys.stdout.write("attn_fast_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
