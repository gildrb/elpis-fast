#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite source link of the fused SiLU-gate epilogue to the patched engine text.

Links bend/act_fuse.bend (ext 5112 fused SiLU-gate epilogue) and
bend/act_fuse_spec.bend (stock hgemm_recon wide store + silu_mul) to the patched
engine text.

Argument: the patched exllamav3 package directory. Checks, failing closed:
  1. gemm_wide_silu_kernel (hgemm_act.cu) is gemm_wide_kernel
     (hgemm_f16acc_wide.cuh) token for token from the tile constants through the
     K loop and the epilogue's (row, col) loop header (comments, blank lines and
     indentation dropped), so the accumulator pair of every (row, col) is the
     same fp32 value and the store map is the same;
  2. the epilogue expressions act_fuse.bend transcribes occur verbatim once in
     hgemm_act.cu;
  3. silu_h2 (hgemm_act.cu) is the stock _silu(half2) (activation_kernels.cuh)
     with the name changed;
  4. the stock expressions act_fuse_spec.bend transcribes occur verbatim in
     activation_kernels.cuh / activation.cu (act_mul_kernel_h<ACT_SILU>,
     silu_mul launch) and hgemm_f16acc.cu (worthwhile, tuned_device, MIN_ROWS,
     BM / BN);
  5. the fused launch uses the served Wide config and passes ldc = c.stride(0);
     the fallback is hgemm_recon + silu_mul;
  6. the Python caller reconstructs W with the same call as the stock
     fused-reconstruct path.
Text evidence, not a proof. `--mutate NAME` applies a deliberate source mutation
that must be rejected.

Usage: python3 bend/act_fuse_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NoReturn

EPI = [
    "int col = bn + wn * WN + j * 8 + t * 2;",
    "int row = bm + wm * WM + i * 16 + g + h * 8;",
    "if (row >= M) continue;",
    "float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];",
    "half2 y2 = __floats2half2_rn(v0, v1);",
    (
        "half2 x2 = silu_h2(*reinterpret_cast<const half2*>"
        "(G + (long long) row * ldc + col));"
    ),
    "const bool limit = act_limit != 0.0f;",
    (
        "y2 = __hmax2(y2, __float2half2_rn(-act_limit));\n"
        "                    y2 = __hmin2(y2, __float2half2_rn(act_limit));\n"
        "                    x2 = __hmin2(x2, __float2half2_rn(act_limit));"
    ),
    "*reinterpret_cast<half2*>(C + (long long) row * ldc + col) = __hmul2(x2, y2);",
    "using WideAct = f16acc_wide::Cfg<128, 128, 32, 2, 2, 3, 8, 2>;",
    "dim3 grid(N / WideAct::BN, (M + WideAct::BM - 1) / WideAct::BM, 1);",
    "M, N, K, (int) c.stride(0), act_limit);",
    (
        "if (K != 5120 || N != 17408 || b.size(0) != K || c.size(0) != M"
        " || c.size(1) != N || M < 1024) return false;"
    ),
    "if (M > (int64_t) 65535 * WideAct::BM) return false;",
    (
        'if (!env_on("EXL3_MLP_ACT_FUSE") || !env_on("EXL3_HGEMM_F16ACC_WIDE"))'
        " return false;"
    ),
    "if (props->major != 8 || props->minor != 6) return false;",
    "return hgemm_f16acc_status(a.device().index()) == 1;",
    "hgemm_recon(a, b, c);\n        silu_mul(gate, c, c, act_limit);",
]
STOCK_ACT = [
    "size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);",
    "if (idx >= numel / 2) return;",
    "half2 x2 = ((const half2*) x)[idx];",
    "half2 y2 = ((const half2*) y)[idx];",
    "x2 = _silu(x2);",
    (
        "y2 = __hmax2(y2, __float2half2_rn(-act_limit));\n"
        "        y2 = __hmin2(y2, __float2half2_rn(act_limit));\n"
        "        x2 = __hmin2(x2, __float2half2_rn(act_limit));"
    ),
    "((half2*) z)[idx] = __hmul2(x2, y2);",
]
STOCK_ACT_CU = [
    "#define NUM_THREADS 256",
    "size_t blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS);",
]
STOCK_F16 = [
    "constexpr int BM = 128, BN = 128, BK = 64, PAD = 8;",
    "constexpr int MIN_ROWS = 384;",
    "return at::cuda::getDeviceProperties(device)->major == 12;",
    "if (M < MIN_ROWS) return false;",
    "int64_t blocks = ((M + BM - 1) / BM) * (N / BN) * batch;",
    "return blocks >= props->multiProcessorCount;",
    "using Wide = f16acc_wide::Cfg<128, 128, 32, 2, 2, 3,  8, 2>;",
    "if (!f16acc::covered(a, b, c) || !f16acc::worthwhile(a, b)) return false;",
    "if (wide_route(a, b, c)) launch_wide<OUT_F32>(a, b, c, stream);",
]
PY_STOCK = (
    "ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K,"
    " self.mcg, self.mul1, 0)"
)
PY_STOCK_COUNT = 2
MUTATE_ARGC = 3

MUTATIONS = {
    # clamp y in the other order (min before max)
    "clamp_order": (
        (
            "y2 = __hmax2(y2, __float2half2_rn(-act_limit));\n"
            "                    y2 = __hmin2(y2, __float2half2_rn(act_limit));"
        ),
        (
            "y2 = __hmin2(y2, __float2half2_rn(act_limit));\n"
            "                    y2 = __hmax2(y2, __float2half2_rn(-act_limit));"
        ),
    ),
    # silu_h2 adds in the other operand order
    "silu_add": (
        (
            "half2 sum = __hadd2(one, e);\n    half2 r = h2rcp(sum);\n"
            "    half2 result = __hmul2(x, r);\n    return result;\n}\n\n"
            "template <class CF>"
        ),
        (
            "half2 sum = __hadd2(e, one);\n    half2 r = h2rcp(sum);\n"
            "    half2 result = __hmul2(x, r);\n    return result;\n}\n\n"
            "template <class CF>"
        ),
    ),
    # K loop: the two MMA k-halves swapped
    "mma_order": (
        (
            "mma_h(h, af[0][i], bf[0][j]);\n"
            "                    mma_h(h, af[1][i], bf[1][j]);"
        ),
        (
            "mma_h(h, af[1][i], bf[1][j]);\n"
            "                    mma_h(h, af[0][i], bf[0][j]);"
        ),
    ),
}

KERNEL_START = "constexpr int BM = CF::BM, BN = CF::BN"
KERNEL_END = "const int g = lane >> 2, t = lane & 3;"
HEADER_END = "float v0 = acc[i][j][h * 2]"


def out(text: str) -> None:
    """Write one line to stdout."""
    sys.stdout.write(f"{text}\n")


def fail(msg: str) -> NoReturn:
    """Exit with a FAIL message.

    Args:
        msg: Failure description.

    Raises:
        SystemExit: Always.

    """
    text = f"act_fuse_diff: FAIL: {msg}"
    raise SystemExit(text)


def norm(text: str) -> list[str]:
    """Drop comments, blank lines and indentation; collapse whitespace.

    Args:
        text: Source text.

    Returns:
        The normalized non-empty lines.

    """
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"//.*", "", raw).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def seg(text: str, start: str, end: str, what: str) -> str:
    """Return the text from a unique start anchor up to the next end anchor.

    Args:
        text: Source text.
        start: Start anchor, must occur exactly once.
        end: End anchor, searched after the start.
        what: Label for failure messages.

    Returns:
        The segment text.

    """
    i = text.find(start)
    if i < 0 or text.find(start, i + 1) >= 0:
        fail(f"{what}: start anchor {start!r} must occur exactly once")
    j = text.find(end, i)
    if j < 0:
        fail(f"{what}: end anchor {end!r} not found")
    return text[i:j]


def once(text: str, expr: str, where: str) -> None:
    """Fail unless expr occurs exactly once in text."""
    n = text.count(expr)
    if n != 1:
        fail(f"{where}: {expr!r} occurs {n} times (want 1)")


def compare(a: list[str], b: list[str], what: str) -> None:
    """Fail on the first differing line, else report equality."""
    if a != b:
        for k, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                fail(f"{what}: line {k}: {x!r} != {y!r}")
        fail(f"{what}: lengths differ ({len(a)} vs {len(b)})")
    out(f"act_fuse_diff: {what}: {len(a)} normalized lines equal")


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line.

    Args:
        argv: Full argv.

    Returns:
        The mutation name (or None) and the engine package root.

    """
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: act_fuse_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    return mutate, Path(args[0])


def apply_mutation(act: str, mutate: str) -> str:
    """Apply a named mutation to the hgemm_act.cu text.

    Args:
        act: hgemm_act.cu text.
        mutate: Mutation name.

    Returns:
        The mutated text.

    """
    if mutate not in MUTATIONS:
        fail(f"unknown mutation {mutate!r}")
    old, new = MUTATIONS[mutate]
    once(act, old, "mutation anchor")
    act = act.replace(old, new)
    out(f"act_fuse_diff: applied mutation {mutate}")
    return act


def check_kernel(act: str, wide: str) -> None:
    """Check 1: K loop and store map."""
    k_act = norm(seg(act, KERNEL_START, KERNEL_END, "5112 kernel"))
    k_wide = norm(seg(wide, KERNEL_START, KERNEL_END, "3011 kernel"))
    compare(
        k_wide, k_act, "K loop (3011 gemm_wide_kernel vs 5112 gemm_wide_silu_kernel)"
    )
    h_act = norm(seg(act, KERNEL_END, HEADER_END, "5112 epilogue header"))
    h_act = [x for x in h_act if x != "const bool limit = act_limit != 0.0f;"]
    h_wide = norm(seg(wide, KERNEL_END, HEADER_END, "3011 epilogue header"))
    compare(h_wide, h_act, "epilogue (row, col) loops")
    once(
        wide,
        "*reinterpret_cast<half2*>(c) = __floats2half2_rn(v0, v1);",
        "hgemm_f16acc_wide.cuh",
    )


def check_epilogue(act: str) -> None:
    """Check 2: transcribed epilogue / host expressions."""
    for e in EPI:
        once(act, e, "hgemm_act.cu")
    out(f"act_fuse_diff: {len(EPI)} epilogue / host expressions found once each")


def check_silu(act: str, ak: str) -> None:
    """Check 3: silu_h2 == stock _silu(half2)."""
    s_act = norm(
        seg(
            act,
            "__device__ __forceinline__ half2 silu_h2(half2 x)",
            "template <class CF>",
            "silu_h2",
        )
    )
    s_stock = norm(
        seg(
            ak,
            "__device__ __forceinline__ half2 _silu(half2 x)",
            "__device__ __forceinline__ float _silu(float x)",
            "_silu(half2)",
        )
    )
    compare(
        [x.replace("_silu(", "silu_h2(") for x in s_stock],
        s_act,
        "silu_h2 vs stock _silu(half2)",
    )


def check_stock(ak: str, acu: str, f16: str) -> None:
    """Check 4: stock expressions."""
    for e in STOCK_ACT:
        once(
            seg(
                ak, "void act_mul_kernel_h", "void act_mul_kernel_f", "act_mul_kernel_h"
            ),
            e,
            "act_mul_kernel_h",
        )
    for e in STOCK_ACT_CU:
        if e not in acu:
            fail(f"activation.cu: {e!r} missing")
    silu_gr = seg(acu, "void silu_mul_gr", "void silu_mul\n", "silu_mul_gr")
    once(
        silu_gr,
        "act_mul_kernel_h<ACT_SILU><<<blocks, NUM_THREADS, 0, stream>>>",
        "silu_mul_gr",
    )
    for e in STOCK_F16:
        once(f16, e, "hgemm_f16acc.cu")
    total = len(STOCK_ACT) + len(STOCK_ACT_CU) + len(STOCK_F16)
    out(f"act_fuse_diff: {total} stock expressions found")


def check_python(exl3: str) -> None:
    """Check 6: same W reconstruct call in both fused reconstruct paths."""
    if exl3.count(PY_STOCK) != PY_STOCK_COUNT:
        fail(
            f"exl3.py: {PY_STOCK!r} occurs {exl3.count(PY_STOCK)} times"
            " (want 2: reconstruct_hgemm, reconstruct_hgemm_silu)"
        )
    out(
        "act_fuse_diff: W reconstruct call identical in reconstruct_hgemm"
        " and reconstruct_hgemm_silu"
    )


def main(argv: list[str]) -> None:
    """Run all checks against the engine package named in argv."""
    mutate, root = parse_args(argv)
    act = (root / "exllamav3_ext/hgemm_act.cu").read_text()
    wide = (root / "exllamav3_ext/hgemm_f16acc_wide.cuh").read_text()
    f16 = (root / "exllamav3_ext/hgemm_f16acc.cu").read_text()
    ak = (root / "exllamav3_ext/activation_kernels.cuh").read_text()
    acu = (root / "exllamav3_ext/activation.cu").read_text()
    exl3 = (root / "modules/quant/exl3.py").read_text()
    if mutate is not None:
        act = apply_mutation(act, mutate)
    check_kernel(act, wide)
    check_epilogue(act)
    check_silu(act, ak)
    check_stock(ak, acu, f16)
    check_python(exl3)
    out("act_fuse_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
