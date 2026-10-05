#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Source pin of bend/m16_discard.bend.

Usage: m16_discard_diff.py <engine tree with 2113 applied> [--mutate NAME].

The Bend model transcribes ext 2113's discard loop, the finishes' chunk and load
addresses, the contributors' slot stores, the workspace layout and the
EXL3_SPLITK_DISCARD parser by hand, and orders each finish's loads, warp barrier
and discard. This check fails unless the engine source still contains exactly
those expressions (whitespace-normalized), each discard call directly follows its
finish's output Hadamard (the last load of the slot lines), and every finish
carries exactly one discard call (m16g 1, 8201 MLP 2, layer tail 3), so an edit to
the kernels that the model no longer describes is caught. It is a conformance
check of the transcription, not a proof of the CUDA code: the model's laws are
theorems about the model only.

--mutate NAME applies one built-in source mutation in memory first; the check
must then FAIL (self-test).
"""

import re
import sys
from pathlib import Path

Q = "exllamav3_ext/quant/"
PINS = {
    Q + "exl3_gemm_m16_kernel.cuh": [
        "#define EXL3_M16_WARPS 8",
        "#define EXL3_M16_TILES_W 4",
        "#define EXL3_M16_GW (EXL3_M16_WARPS * EXL3_M16_TILES_W * 16)",
        "#define EXL3_M16_SLOT_FLOATS (16 * EXL3_M16_GW)",
        (
            "__device__ __forceinline__ void exl3_m16_discard_slots("
            "const float* s0, int nc, int lane) { __syncwarp(); "
            "for (int i = lane; i < 4 * nc; i += 32) { "
            "const float* a = s0 + (size_t) (i >> 2) * EXL3_M16_SLOT_FLOATS"
            " + (i & 3) * 32; "
            'asm volatile ("discard.global.L2 [%0], 128;\\n" :: "l"(a)'
            ' : "memory"); } }'
        ),
    ],
    Q + "hadamard_inner.cuh": [
        "int t = threadIdx.x & 31; // Load float4 v = ((float4*) input_ptr)[t];",
    ],
    Q + "exl3_gemm_m16g_kernel.cuh": [
        (
            "float* slot = p.ws + ((size_t) g * p.max_contrib + cidx)"
            " * EXL3_M16_SLOT_FLOATS + warp * 64 + (lane >> 2);"
        ),
        "int row = mt * 8 + 2 * (lane & 3) + (c & 1);",
        (
            "if (row < size_m) __stcg(slot + row * GW + t * 16 + (c >> 1) * 8,"
            " acc[t][mt][c]);"
        ),
        "grid.sync(); // Finish:",
        (
            "float* s0 = p.ws + (size_t) g * p.max_contrib * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128; "
            "float4 v = __ldcg(((const float4*) s0) + lane);"
            " for (int c = 1; c < nc; ++c) { "
            "float4 u = __ldcg(((const float4*) (s0 + (size_t) c"
            " * EXL3_M16_SLOT_FLOATS)) + lane);"
        ),
        "((float4*) s0)[lane] = v; __syncwarp();",
        (
            "had_fh_r_128_inner<false, true> ( s0, ((half*) C) + (size_t) r"
            " * n_mat + col0, svh, 0.088388347648f // 1/sqrt(128) ); "
            "if (p.discard) exl3_m16_discard_slots(s0, nc, lane); }"
        ),
    ],
    Q + "exl3_mlp_m16_kernel.cuh": [
        (
            "float* ws = PH == 1 ? p.ws + (size_t) g * p.mc1 * EXL3_M16_SLOT_FLOATS"
            " : p.ws2 + (size_t) g * p.mc2 * EXL3_M16_SLOT_FLOATS; "
            "float* slot = ws + (size_t) cidx * EXL3_M16_SLOT_FLOATS"
            " + warp * 64 + (lane >> 2);"
        ),
        (
            "int row = mt * 8 + 2 * (lane & 3) + (c & 1); if (row < size_m)"
            " __stcg(slot + row * GW + t * 16 + (c >> 1) * 8, acc[t][mt][c]);"
        ),
        (
            "float* s0 = p.ws + (size_t) g * p.mc1 * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128; "
            "float4 acc4 = __ldcg(((const float4*) s0) + lane);"
            " for (int c = 1; c < nc; ++c) { "
            "float4 w4 = __ldcg(((const float4*) (s0 + (size_t) c"
            " * EXL3_M16_SLOT_FLOATS)) + lane);"
        ),
        (
            "((float4*) s0)[lane] = acc4; __syncwarp();"
            " had_fh_r_128_inner<false, true> ( s0, (mat ? Cu : Cg)"
            " + (size_t) r * N1 + col0, "
            "(mat ? p.svh_u : p.svh_g) + col0, 0.088388347648f // 1/sqrt(128) );"
            " if (p.discard) exl3_m16_discard_slots(s0, nc, lane); }"
        ),
        "exl3_mlp_grid_sync(p.cnt + EXL3_MLP_GBAR); stamp(6);",
        (
            "float* s0 = p.ws2 + (size_t) g * p.mc2 * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128;"
        ),
        (
            "had_fh_r_128_inner<false, true>(s0, ((half*) D) + (size_t) r * N2"
            " + col0, svh, 0.088388347648f); "
            "if (p.discard) exl3_m16_discard_slots(s0, nc, lane); }"
        ),
    ],
    Q + "exl3_tail_m16_kernel.cuh": [
        (
            "float* ws = PH == 0 ? p.ws0 + (size_t) g * p.mc0"
            " * EXL3_M16_SLOT_FLOATS : PH == 1 ? q.ws + (size_t) g * q.mc1"
            " * EXL3_M16_SLOT_FLOATS "
            ": q.ws2 + (size_t) g * q.mc2 * EXL3_M16_SLOT_FLOATS; "
            "float* slot = ws + (size_t) cidx * EXL3_M16_SLOT_FLOATS"
            " + warp * 64 + (lane >> 2);"
        ),
        "exl3_mlp_grid_sync(q.cnt + EXL3_MLP_GBAR); stamp(3);",
        (
            "float* sl0 = p.ws0 + (size_t) g * p.mc0 * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128; "
            "float4 acc4 = __ldcg(((const float4*) sl0) + lane);"
        ),
        (
            "had_ff_r_128_inner<false, true>(sl0, yo, p.svh_o + col0,"
            " 0.088388347648f); if (q.discard)"
            " exl3_m16_discard_slots(sl0, nc, lane);"
        ),
        (
            "float* sl0 = q.ws + (size_t) g * q.mc1 * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128;"
        ),
        (
            "((float4*) sl0)[lane] = acc4; __syncwarp();"
            " had_fh_r_128_inner<false, true> ( sl0, (mat ? p.u : p.g)"
            " + (size_t) r * N1 + col0, "
            "(mat ? q.svh_u : q.svh_g) + col0, 0.088388347648f // 1/sqrt(128) );"
            " if (q.discard) exl3_m16_discard_slots(sl0, nc, lane); }"
        ),
        "exl3_mlp_grid_sync(q.cnt + EXL3_MLP_GBAR); stamp(10);",
        (
            "float* sl0 = q.ws2 + (size_t) g * q.mc2 * EXL3_M16_SLOT_FLOATS"
            " + r * GW + ch * 128;"
        ),
        (
            "had_ff_r_128_inner<false, true>(sl0, D + (size_t) r * N2 + col0,"
            " q.svh_d + col0, 0.088388347648f); "
            "if (q.discard) exl3_m16_discard_slots(sl0, nc, lane); }"
        ),
    ],
    Q + "exl3_gemm_m16.cu": [
        (
            'const char* env = std::getenv("EXL3_SPLITK_DISCARD");'
            " if (!env || !env[0]) return 1; "
            "TORCH_CHECK((env[0] == '0' || env[0] == '1') && !env[1],"
            ' "EXL3_SPLITK_DISCARD must be 0 or 1, got \'", env, "\'"); '
            "return env[0] - '0'; }(); if (set >= 0) mode = set > 0 ? 1 : 0;"
            " return mode;"
        ),
    ],
    Q + "exl3_gemm_m16g.cu": [
        "args.discard = exl3_m16_discard();",
        "args.ws = ws;",
    ],
    Q + "exl3_mlp_m16.cu": [
        (
            "args.ws = dev.ws; args.ws2 = dev.ws + (size_t) 2"
            " * EXL3_MLP_SCHED_PAIRS * sd.mc1 * EXL3_M16_SLOT_FLOATS;"
        ),
        "args.discard = exl3_m16_discard();",
    ],
    Q + "exl3_tail_m16.cu": [
        (
            "const size_t ws1_floats = (size_t) 2 * EXL3_MLP_SCHED_PAIRS"
            " * sd.mc1 * EXL3_M16_SLOT_FLOATS;"
        ),
        (
            "const size_t ws2_floats = (size_t) (EXL3_MLP_SCHED_N2 / EXL3_M16_GW)"
            " * sd.mc2 * EXL3_M16_SLOT_FLOATS;"
        ),
        "args.ws0 = dev.ws + ws1_floats + ws2_floats;",
        "m.ws = dev.ws; m.ws2 = dev.ws + ws1_floats;",
        "m.discard = exl3_m16_discard();",
    ],
}
# finish calls per kernel file (each finish: one discard after its last load)
CALLS = {
    Q + "exl3_gemm_m16g_kernel.cuh": (
        "if (p.discard) exl3_m16_discard_slots(s0, nc, lane);",
        1,
    ),
    Q + "exl3_mlp_m16_kernel.cuh": (
        "if (p.discard) exl3_m16_discard_slots(s0, nc, lane);",
        2,
    ),
    Q + "exl3_tail_m16_kernel.cuh": (
        "if (q.discard) exl3_m16_discard_slots(sl0, nc, lane);",
        3,
    ),
}
MUTATIONS = {
    "stride16": (Q + "exl3_gemm_m16_kernel.cuh", "i += 32)", "i += 16)"),
    "no_bar": (
        Q + "exl3_gemm_m16_kernel.cuh",
        "{\n    __syncwarp();\n    for (int i = lane;",
        "{\n    for (int i = lane;",
    ),
    "line64": (Q + "exl3_gemm_m16_kernel.cuh", "(i & 3) * 32;", "(i & 3) * 64;"),
    "default0": (
        Q + "exl3_gemm_m16.cu",
        "if (!env || !env[0]) return 1;",
        "if (!env || !env[0]) return 0;",
    ),
    "early_discard": (
        Q + "exl3_gemm_m16g_kernel.cuh",
        "            ((float4*) s0)[lane] = v;\n            __syncwarp();\n",
        (
            "            ((float4*) s0)[lane] = v;\n            __syncwarp();\n"
            "            if (p.discard) exl3_m16_discard_slots(s0, nc, lane);\n"
        ),
    ),
    "tail_drop_call": (
        Q + "exl3_tail_m16_kernel.cuh",
        (
            "had_ff_r_128_inner<false, true>(sl0, yo, p.svh_o + col0, 0.088388347648f);"
            "\n            if (q.discard) exl3_m16_discard_slots(sl0, nc, lane);\n"
        ),
        "had_ff_r_128_inner<false, true>(sl0, yo, p.svh_o + col0, 0.088388347648f);\n",
    ),
    "ws2_offset": (
        Q + "exl3_tail_m16.cu",
        "m.ws2 = dev.ws + ws1_floats;",
        "m.ws2 = dev.ws + ws2_floats;",
    ),
    "nc_plus1": (Q + "exl3_gemm_m16_kernel.cuh", "i < 4 * nc;", "i < 4 * (nc + 1);"),
}


def norm(s: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends.

    Args:
        s: Source text.

    Returns:
        The whitespace-normalized text.

    """
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    """Check the engine tree for the transcribed expressions and call counts.

    Raises:
        SystemExit: If a mutation anchor is not unique or the source diverges.

    """
    root = Path(sys.argv[1])
    mutate = (
        sys.argv[sys.argv.index("--mutate") + 1] if "--mutate" in sys.argv else None
    )
    src = {f: (root / f).read_text() for f in set(PINS) | set(CALLS)}
    if mutate is not None:
        f, old, new = MUTATIONS[mutate]
        if src[f].count(old) != 1:
            msg = f"mutation {mutate}: anchor not found once in {f}"
            raise SystemExit(msg)
        src[f] = src[f].replace(old, new)
        sys.stdout.write(f"mutation {mutate} applied to {f}\n")
    missing = 0
    for f, pins in PINS.items():
        text = norm(src[f])
        for p in pins:
            ok = norm(p) in text
            missing += not ok
            sys.stdout.write(
                ("ok      " if ok else "MISSING ") + f + ": " + p[:110] + "\n"
            )
    for f, (call, want) in CALLS.items():
        n = norm(src[f]).count(norm(call))
        ok = n == want
        missing += not ok
        sys.stdout.write(
            ("ok      " if ok else "COUNT   ")
            + f"{f}: {n} discard call(s), want {want}\n"
        )
    if missing:
        msg = (
            f"FAIL: {missing} transcribed expression(s) / call counts"
            " do not match the source"
        )
        raise SystemExit(msg)
    total = sum(len(p) for p in PINS.values())
    sys.stdout.write(
        f"all {total} transcribed expressions present; discard call counts match\n"
    )


if __name__ == "__main__":
    main()
