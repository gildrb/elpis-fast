#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Pin the source of bend/gdn_replay_gather.bend to an engine tree with 5108 applied.

Usage: gdn_replay_gather_diff.py <engine tree with 5108 applied>.

The Bend model transcribes the 5108 loader's index and widening expressions by
hand. This check fails unless the kernel source still contains exactly those
expressions (whitespace-normalized), so an edit to the kernel that the model no
longer describes is caught. It is a conformance check of the transcription, not
a proof of the CUDA code: the model's laws are theorems about the model only.
"""

import re
import sys
from pathlib import Path

PINS = [
    "#define REPLAY_PARTS 20",
    "#define REPLAY_GI ((GR_MAX_S * REPLAY_PARTS + 127) / 128)",
    "#define GR_MAX_S 16",
    "const int k_ch = (j.num_k_heads + kh) * HD;",
    "const int v_ch = 2 * j.num_k_heads * HD + h * HD + blockIdx.x * 32;",
    "const int e = threadIdx.x + i * 128;",
    "if (e < S * REPLAY_PARTS)",
    "const int s = e / REPLAY_PARTS;",
    "const int p = e - s * REPLAY_PARTS;",
    "const int ch = p < 16 ? k_ch + p * 8 : v_ch + (p - 16) * 8;",
    "r[i] = *(const uint4*) (src + (size_t) s * F + ch);",
    (
        "float* dst = p < 16 ? m.k + s * HD + p * 8 : "
        "m.v + s * HD + blockIdx.x * 32 + (p - 16) * 8;"
    ),
    "lo.x = __uint_as_float(w[0] << 16); lo.y = __uint_as_float(w[0] & 0xffff0000u);",
    "hi.x = __uint_as_float(w[2] << 16); hi.y = __uint_as_float(w[2] & 0xffff0000u);",
    "gdn_rule_l2norm_warp(m.k + task * 128, lane);",
    (
        "int ch = part == 0 ? kh * 128 + c : "
        "(part == 1 ? (Nk + kh) * 128 + c : 2 * Nk * 128 + h * 128 + c);"
    ),
    (
        "TORCH_CHECK((env[0] == '0' || env[0] == '1') && env[1] == '\\0', "
        '"EXL3_GDN_REPLAY_GATHER must be 0 or 1");'
    ),
    "if (!env) return 1;",
]


def norm(s: str) -> str:
    """Collapse whitespace runs to one space.

    Returns:
        The normalized string.

    """
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    """Report every pin and fail if one is missing.

    Raises:
        SystemExit: A transcribed expression is not in the source.

    """
    src = norm((Path(sys.argv[1]) / "exllamav3_ext" / "gdn.cu").read_text())
    missing = [p for p in PINS if norm(p) not in src]
    for p in PINS:
        sys.stdout.write(("ok      " if p not in missing else "MISSING ") + p + "\n")
    if missing:
        msg = f"FAIL: {len(missing)} transcribed expression(s) not in the source"
        raise SystemExit(msg)
    sys.stdout.write(f"all {len(PINS)} transcribed expressions present\n")


if __name__ == "__main__":
    main()
