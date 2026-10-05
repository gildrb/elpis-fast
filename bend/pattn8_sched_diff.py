#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Link bend/pattn8_sched.bend (ext 3020 8-warp prefill attention) to the engine text.

Finite source link to the patched engine text.

Argument: the patched exllamav3 package directory (the one holding
exllamav3_ext/pattn8_kernel.cuh, exllamav3_ext/pattn_kernel.cuh and
exllamav3_ext/pattn.cu). Checks, failing closed:
  1. every index expression the Bend model transcribes occurs verbatim exactly
     once in pattn8_kernel.cuh (or in pattn.cu's pattn_prefill for the host
     grid), and the 3010 constants the model uses are the ones in
     pattn_kernel.cuh;
  2. the per-warp arithmetic is 3010's text: after dropping comments, blank
     lines and indentation, and the two structural differences the Bend laws
     cover (3020's `if (act) { ... }` wrappers, and `uint32_t pa[2][4];` /
     `const int hq = ...;` declared at a different place), these segments are
     token-for-token equal in both kernels: QK^T + mask + online softmax + P
     packing (one tile), P V (one tile), the epilogue (l reduction, 1 / l,
     store), the q scaling of the prologue, the accumulator / m / l
     initialisation and the per-thread row bounds;
  3. the tile issue lambda is 3010's with PA_THREADS -> P8_THREADS;
  4. the host launches pattn8_kernel<true> exactly for mode 1 (the 3010
     dispatch);
  5. ext 3021c (pattn8i_kernel.cuh, the int8-QK kernel that serves 3-bit K
     caches): its schedule is 3020's, so the schedule laws of
     bend/pattn8_sched.bend cover it too: every block / warp / tile / mask /
     row-bound / store-row expression of the model occurs exactly once in
     pattn8i_kernel.cuh, its q prologue maps smem row `row` to 3020's
     (position, head) with `row` warp-strided over all P8_NW * PA_BQ rows, and
     pattn_prefill_int8 launches it on 3020's grid with P8_THREADS. Its
     arithmetic is NOT 3010's (bend/pattn8i_int.bend covers the int8 core).
Text evidence, not a proof: it ties the Bend model to one source revision.
`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/pattn8_sched_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NoReturn

MUTATE_ARGC = 3
USAGE = "Usage: python3 bend/pattn8_sched_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR"
MODEL_EXPRS_K8 = [
    "#define P8_NW 8",
    "#define P8_NB 4",
    "#define P8_NH 2",
    "#define P8_PAIRS (PA_G / P8_NH)",
    "#define P8_THREADS (P8_NW * 32)",
    "const int kvh = blockIdx.x / P8_PAIRS, pair = blockIdx.x % P8_PAIRS;",
    "const int qc = gridDim.y - 1 - blockIdx.y;",
    "const int wb = warp % P8_NB, wh = warp / P8_NB;",
    "const int c0 = qc * P8_NB * PA_BQ;",
    "const int n_hi_cta = min(total - q_len + c0 + P8_NB * PA_BQ, total);",
    "const int ntiles_cta = (n_hi_cta + PA_BN - 1) / PA_BN;",
    "const int p0 = c0 + wb * PA_BQ;",
    "const int hq = kvh * PA_G + pair * P8_NH + wh;",
    "const int q_abs0 = total - q_len + p0;",
    "const int n_hi = min(q_abs0 + PA_BQ, total);",
    "const int ntiles = p0 < q_len ? (n_hi + PA_BN - 1) / PA_BN : 0;",
    "for (int it = 0; it < ntiles_cta; ++it)",
    "const bool act = it < ntiles;",
    "if (n0 + PA_BN - 1 > q_abs0)",
    "for (int c = tid; c < P8_NW * PA_BQ * 32; c += P8_THREADS)",
    "int row = c >> 5, ch = c & 31, w = row / PA_BQ, pos = row - w * PA_BQ;",
    (
        "int p = c0 + (w % P8_NB) * PA_BQ + pos, "
        "h = kvh * PA_G + pair * P8_NH + w / P8_NB;"
    ),
    "const int qrow = warp * PA_BQ;",
    "if (ntiles == 0) return;",
    "int pos = p0 + gid + 8 * r;",
    "for (int i = 0; i < (PA_BN * 32 + P8_THREADS - 1) / P8_THREADS; ++i)",
    "int c = tid + i * P8_THREADS;",
]
MODEL_EXPRS_K6 = [
    "#define PA_G 6",
    "#define PA_BQ 16",
    "#define PA_BN 32",
    "#define PA_PAGE 256",
    "#define PA_NW 6",
]
MODEL_EXPRS_HOST = [
    "const int nqc = (q_len + P8_NB * PA_BQ - 1) / (P8_NB * PA_BQ);",
    "dim3 grid(n_kv_heads * P8_PAIRS, nqc, bsz);",
]
# ext 3021c pattn8i_kernel.cuh: the model's schedule expressions it shares
# with 3020 (each once)
SCHED_EXPRS_K8I = [
    "const int kvh = blockIdx.x / P8_PAIRS, pair = blockIdx.x % P8_PAIRS;",
    "const int qc = gridDim.y - 1 - blockIdx.y;",
    "const int wb = warp % P8_NB, wh = warp / P8_NB;",
    "const int c0 = qc * P8_NB * PA_BQ;",
    "const int n_hi_cta = min(total - q_len + c0 + P8_NB * PA_BQ, total);",
    "const int ntiles_cta = (n_hi_cta + PA_BN - 1) / PA_BN;",
    "const int p0 = c0 + wb * PA_BQ;",
    "const int hq = kvh * PA_G + pair * P8_NH + wh;",
    "const int q_abs0 = total - q_len + p0;",
    "const int n_hi = min(q_abs0 + PA_BQ, total);",
    "const int ntiles = p0 < q_len ? (n_hi + PA_BN - 1) / PA_BN : 0;",
    "for (int it = 0; it < ntiles_cta; ++it)",
    "const bool act = it < ntiles;",
    "if (n0 + PA_BN - 1 > q_abs0)",
    "int key = n0 + 8 * nt + 2 * t + (e & 1);",
    "if (key > qa || key >= total) sc[nt][e] = -INFINITY;",
    "for (int row = warp; row < P8_NW * PA_BQ; row += P8_NW)",
    "int w = row / PA_BQ, pos = row - w * PA_BQ;",
    (
        "int p = c0 + (w % P8_NB) * PA_BQ + pos, "
        "h = kvh * PA_G + pair * P8_NH + w / P8_NB;"
    ),
    "const int qrow = warp * PA_BQ;",
    "if (ntiles == 0) return;",
    "int pos = p0 + gid + 8 * r;",
]

MUTATIONS = {
    # act admits one iteration past the warp's own tiles
    "act_le": ("const bool act = it < ntiles;", "const bool act = it <= ntiles;"),
    # PV MMA k-halves in the opposite order (different fp32 accumulation order)
    "pv_order": (
        (
            "mma_f32(o, pa[0], bv[0][2 * hn], bv[0][2 * hn + 1]);\n"
            "                        "
            "mma_f32(o, pa[1], bv[1][2 * hn], bv[1][2 * hn + 1]);"
        ),
        (
            "mma_f32(o, pa[1], bv[1][2 * hn], bv[1][2 * hn + 1]);\n"
            "                        "
            "mma_f32(o, pa[0], bv[0][2 * hn], bv[0][2 * hn + 1]);"
        ),
    ),
    # softmax partial sum over the two columns in the other order
    "sum_order": (
        "sc[nt][2 * r + e] = p;\n                        sum += p;",
        "sc[nt][2 * r + e] = p;\n                        sum = p + sum;",
    ),
    # q head pair stride 3 instead of 2
    "head_map": (
        "const int hq = kvh * PA_G + pair * P8_NH + wh;",
        "const int hq = kvh * PA_G + pair * 3 + wh;",
    ),
}
# mutations of the ext 3021c kernel pattn8i_kernel.cuh (section 5)
MUTATIONS_K8I = {
    # prologue rows strided by 4 warps: rows of warps 4..7 never staged
    "i8_row_stride": (
        "for (int row = warp; row < P8_NW * PA_BQ; row += P8_NW)",
        "for (int row = warp; row < P8_NW * PA_BQ; row += P8_NB)",
    ),
    # causal mask admits the diagonal's next key
    "i8_mask": (
        "if (key > qa || key >= total) sc[nt][e] = -INFINITY;",
        "if (key > qa + 1 || key >= total) sc[nt][e] = -INFINITY;",
    ),
}


def fail(msg: str) -> NoReturn:
    """Exit with the link's failure message.

    Raises:
        SystemExit: Always.

    """
    text = f"pattn8_sched_diff: FAIL: {msg}"
    raise SystemExit(text)


def out(text: str) -> None:
    """Write one line to stdout."""
    sys.stdout.write(f"{text}\n")


def norm(text: str) -> list[str]:
    """Drop comments, blank lines and indentation; collapse whitespace.

    Returns:
        The normalized non-empty lines.

    """
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", re.sub(r"//.*", "", raw).strip())
        if line:
            lines.append(line)
    return lines


def seg(text: str, start: str, end: str, what: str) -> str:
    """Cut text from the unique start anchor up to the next end anchor.

    Returns:
        The segment.

    """
    i = text.find(start)
    if i < 0 or text.find(start, i + 1) >= 0:
        fail(f"{what}: start anchor {start!r} must occur exactly once")
    j = text.find(end, i)
    if j < 0:
        fail(f"{what}: end anchor {end!r} not found after start")
    return text[i:j]


def once(text: str, expr: str, where: str) -> None:
    """Fail unless expr occurs exactly once in text."""
    n = text.count(expr)
    if n != 1:
        fail(f"{where}: {expr!r} occurs {n} times (want 1)")


def drop(lines: list[str], unwanted: list[str]) -> list[str]:
    """Remove the unwanted lines.

    Returns:
        The remaining lines.

    """
    return [line for line in lines if line not in unwanted]


def unwrap_act(lines: list[str], what: str) -> list[str]:
    """Strip 3020's `if (act) { ... }` wrapper.

    3020 wraps the segment in `if (act) { ... }`: exactly one leading
    `if (act)`, `{` and one trailing `}`.

    Returns:
        The wrapped lines.

    """
    if lines[:2] != ["if (act)", "{"] or lines[-1] != "}":
        fail(f"{what}: 3020 segment is not wrapped in if (act) {{ ... }}")
    return lines[2:-1]


def compare(a: list[str], b: list[str], what: str) -> None:
    """Fail at the first differing line of a and b."""
    if a != b:
        for k, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                fail(f"{what}: line {k}: 3010 {x!r} != 3020 {y!r}")
        fail(f"{what}: lengths differ ({len(a)} vs {len(b)})")
    out(f"pattn8_sched_diff: {what}: {len(a)} normalized lines equal")


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line.

    Returns:
        The mutation name (or None) and the engine package directory.

    """
    mutate = None
    args = argv[1:]
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate = args[1]
        args = args[2:]
    if len(args) != 1:
        fail(USAGE)
    return mutate, Path(args[0])


def apply_mutation(mutate: str, k8: str, k8i: str) -> tuple[str, str]:
    """Apply a named mutation to the 3020 or 3021c kernel text.

    Returns:
        The (possibly mutated) pattn8_kernel.cuh and pattn8i_kernel.cuh texts.

    """
    if mutate in MUTATIONS_K8I:
        old, new = MUTATIONS_K8I[mutate]
        once(k8i, old, "mutation anchor")
        k8i = k8i.replace(old, new, 1)
    elif mutate not in MUTATIONS:
        fail(f"unknown mutation {mutate!r}")
    else:
        old, new = MUTATIONS[mutate]
        if mutate in {"head_map", "act_le"}:
            once(k8, old, "mutation anchor")
        elif k8.count(old) < 1:
            fail(f"mutation anchor {old!r} missing")
        k8 = k8.replace(old, new, 1)
    out(f"pattn8_sched_diff: applied mutation {mutate}")
    return k8, k8i


def check_exprs(k6: str, k8: str, host: str) -> None:
    """Check 1: the transcribed expressions occur once each."""
    for e in MODEL_EXPRS_K8:
        once(k8, e, "pattn8_kernel.cuh")
    for e in MODEL_EXPRS_K6:
        once(k6, e, "pattn_kernel.cuh")
    for e in MODEL_EXPRS_HOST:
        once(host, e, "pattn.cu")
    total = len(MODEL_EXPRS_K8) + len(MODEL_EXPRS_K6) + len(MODEL_EXPRS_HOST)
    out(f"pattn8_sched_diff: {total} transcribed expressions found once each")


def check_tile(k6: str, k8: str) -> None:
    """Check 2, per tile: QK^T + softmax + P pack, and P V."""
    qk6 = drop(
        norm(seg(k6, "// S = Q K^T (log2 domain)", "cp_wait0();", "3010 QK")),
        ["uint32_t pa[2][4];"],
    )
    qk8 = unwrap_act(
        [
            "if (act)",
            "{",
            *norm(seg(k8, "// S = Q K^T (log2 domain)", "cp_wait0();", "3020 QK")),
        ],
        "QK",
    )
    compare(qk6, qk8, "QK^T + mask + online softmax + P pack")
    wait = 'asm volatile("cp.async.wait_group 1;\\n" ::);   // K(it + 1)'
    pv6 = norm(seg(k6, "// O += P V", wait, "3010 PV"))
    pv8 = unwrap_act(
        ["if (act)", "{", *norm(seg(k8, "// O += P V", wait, "3020 PV"))],
        "PV",
    )
    compare(pv6, pv8, "P V")


def check_frame(k6: str, k8: str) -> None:
    """Check 2, outside the tile: epilogue, q scaling, init; and check 3."""
    ep6 = drop(
        norm(
            seg(
                k6,
                "// out = acc / l, fp16",
                "} // namespace pattn_detail",
                "3010 epilogue",
            )
        ),
        ["const int hq = kvh * PA_G + warp;"],
    )
    ep8 = norm(
        seg(
            k8, "// out = acc / l, fp16", "} // namespace pattn_detail", "3020 epilogue"
        )
    )
    compare(ep6, ep8, "epilogue")
    h2 = "half2* h2 = reinterpret_cast<half2*>(&u);"
    sc6 = norm(seg(k6, h2, "float acc[32][4];", "3010 q scale"))
    # 3020 renames the local packed-word array w[4] to w4[4] (w is its warp
    # index there): alpha-rename it back
    sc8 = norm(re.sub(r"\bw4\b", "w", seg(k8, h2, "float acc[32][4];", "3020 q scale")))
    compare(sc6, sc8, "q scaling")
    in6 = norm(seg(k6, "float acc[32][4];", "#pragma unroll 1", "3010 init"))
    in8 = drop(
        norm(seg(k8, "float acc[32][4];", "#pragma unroll 1", "3020 init")),
        ["uint32_t pa[2][4];"],
    )
    compare(in6, in8, "accumulator / m / l init, qrow, row bounds")

    # 3. tile issue
    is6 = norm(
        seg(
            k6, "auto issue = [&]", "if (ntiles > 0) issue(k16, ks, 0);", "3010 issue"
        ).replace("PA_THREADS", "P8_THREADS")
    )
    is8 = norm(
        seg(
            k8,
            "auto issue = [&]",
            "if (ntiles_cta > 0) issue(k16, ks, 0);",
            "3020 issue",
        )
    )
    compare(is6, is8, "tile issue (PA_THREADS -> P8_THREADS)")


def check_host(host: str) -> None:
    """Check 4: host dispatch mode 1 -> <true>, else <false>, as 3010."""
    d8 = norm(
        seg(
            host,
            "if (pattn8_enabled())",
            "dim3 grid((q_len + PA_BQ - 1) / PA_BQ, n_kv_heads, bsz);",
            "3020 host",
        )
    )
    launch = "pattn_detail::pattn8_kernel<true><<<grid, P8_THREADS, P8_SMEM, stream>>>("
    if (
        "if (mode == 1)" not in d8
        or not any(x.startswith(launch) for x in d8)
        or d8.index("if (mode == 1)") + 1
        != next(
            k
            for k, x in enumerate(d8)
            if x.startswith("pattn_detail::pattn8_kernel<true>")
        )
    ):
        fail("host: pattn8_kernel<true> is not the mode == 1 branch")
    out("pattn8_sched_diff: host dispatch mode 1 -> pattn8_kernel<true>, else <false>")


def check_k8i(k8i: str, host_i8: str) -> None:
    """Check 5: the ext 3021c int8-QK kernel runs 3020's schedule."""
    for e in SCHED_EXPRS_K8I:
        once(k8i, e, "pattn8i_kernel.cuh")
    for e in [
        "#define P8_THREADS (P8_NW * 32)",
        "#define P8_NW 8",
        "#define P8_NB 4",
        "#define P8_NH 2",
    ]:
        if k8i.count(e) != 0:
            fail(f"pattn8i_kernel.cuh redefines {e!r}")
    if k8i.count('#include "pattn8_kernel.cuh"') != 1:
        fail("pattn8i_kernel.cuh does not take 3020's constants from pattn8_kernel.cuh")
    for e in MODEL_EXPRS_HOST:
        once(host_i8, e, "pattn.cu pattn_prefill_int8")
    l8i = [x for x in norm(host_i8) if "<<<" in x]
    launch = (
        "pattn_detail::pattn8i_kernel<<<grid, P8_THREADS, P8I_SMEM, stream>>>"
        "(P8I_ARGS);"
    )
    if l8i != [launch]:
        fail(
            f"pattn_prefill_int8: launches are {l8i}, "
            "want one pattn8i_kernel on 3020's grid with P8_THREADS"
        )
    out(
        f"pattn8_sched_diff: 3021c pattn8i_kernel: {len(SCHED_EXPRS_K8I)} "
        "schedule expressions found once each, 3020 grid and threads"
    )


def main(argv: list[str]) -> None:
    """Run checks 1-5 against the engine package named in argv."""
    mutate, root = parse_args(argv)
    k6 = (root / "exllamav3_ext/pattn_kernel.cuh").read_text()
    k8 = (root / "exllamav3_ext/pattn8_kernel.cuh").read_text()
    host_all = (root / "exllamav3_ext/pattn.cu").read_text()
    k8i = (root / "exllamav3_ext/pattn8i_kernel.cuh").read_text()
    # pattn_prefill only: ext 3021c adds pattn_stage_int8k / pattn_prefill_int8
    # after it (checked in section 5)
    host = seg(
        host_all,
        "void pattn_prefill\n(",
        "\nvoid pattn_stage_int8k\n(",
        "pattn.cu pattn_prefill",
    )
    host_i8 = host_all[host_all.index("\nvoid pattn_prefill_int8\n(") :]
    if mutate is not None:
        k8, k8i = apply_mutation(mutate, k8, k8i)

    check_exprs(k6, k8, host)
    check_tile(k6, k8)
    check_frame(k6, k8)
    check_host(host)
    check_k8i(k8i, host_i8)
    out("pattn8_sched_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
