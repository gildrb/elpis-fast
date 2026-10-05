#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Link bend/gdn_conv_qkv.bend and bend/gdn_conv_qkv_spec.bend to the engine text.

Finite source link of the ext 5111 fused GDN prefill conv and its spec (stock
transpose / cast / split path) to the patched engine text.

Argument: the patched exllamav3 package directory. Checks, failing closed:
  1. _conv1d_prefill_qkv_output_kernel and _conv1d_prefill_qkv_state_kernel are
     the stock _causal_conv1d_update_slotted_output_kernel / _state_kernel line
     for line (comments, blank lines, indentation dropped) except exactly these
     rewrites, which the Bend laws cover: pid_b = 0 (bsz 1: the pid_b line
     dropped, slots + pid_b -> slots), the x address
     (pid_b * dim + offs_d) * seq_len + x_t -> x_t * dim + offs_d, the loaded x
     cast .to(tl.bfloat16) (stock: the bf16 copy made by .to(torch.bfloat16)
     before the kernel), and the output store;
  2. the fused store / view expressions the Bend model transcribes occur
     verbatim once;
  3. the fused launch uses the stock long-chunk tiling (BLOCK_D 32, BLOCK_S 256,
     BLOCK_K next_power_of_2(K), 4 warps, grid (1, ceil(dim / 32),
     ceil(seq_len / 256)); state grid (1, ceil(dim / 32))) and seq_len > 256
     (stock branch);
  4. the stock path still reads qkv.transpose(1, 2).to(torch.bfloat16)
     .contiguous() and torch.split(.., [k_dim, k_dim, v_dim], -1) when the
     switch is off.
Text evidence, not a proof. `--mutate NAME` applies a deliberate source mutation
that must be rejected.

Usage: python3 bend/gdn_conv_qkv_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NoReturn

MUTATE_ARGC = 3
X_ADDR_REWRITE = 2

REWRITES = [
    ("pid_b = tl.program_id(0)", None),
    ("slot = tl.load(slots + pid_b)", "slot = tl.load(slots)"),
    (
        "x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],",
        "x + x_t[None, :].to(tl.int64) * dim + offs_d[:, None],",
    ),
]
MODEL = [
    "VD: tl.constexpr = dim - 2 * KD",
    "d64 < KD, s64 * KD + d64,",
    (
        "tl.where(d64 < 2 * KD, seq_len * KD + s64 * KD + (d64 - KD), "
        "2 * seq_len * KD + s64 * VD + (d64 - 2 * KD)),"
    ),
    "tl.store(out + dst, acc, mask = mask_d[:, None] & mask_s[None, :])",
    "q = buf[: seq_len * k_dim].view(1, seq_len, k_dim)",
    "k = buf[seq_len * k_dim: 2 * seq_len * k_dim].view(1, seq_len, k_dim)",
    "v = buf[2 * seq_len * k_dim:].view(1, seq_len, v_dim)",
    "v_dim = dim - 2 * k_dim",
    "buf = torch.empty((seq_len * dim,), dtype = torch.bfloat16, device = qkv.device)",
    "block_d, block_s = 32, 256",
    (
        "_conv1d_prefill_qkv_output_kernel[(1, triton.cdiv(dim, block_d), "
        "triton.cdiv(seq_len, block_s))]("
    ),
    (
        "BLOCK_D = block_d, BLOCK_S = block_s, "
        "BLOCK_K = triton.next_power_of_2(conv_kernel_size), num_warps = 4,"
    ),
    "_conv1d_prefill_qkv_state_kernel[(1, triton.cdiv(dim, block_d))](",
    (
        "BLOCK_D = block_d, BLOCK_STATE = triton.next_power_of_2(state_size), "
        "num_warps = 4,"
    ),
    (
        "if (bsz != 1 or seq_len <= 256 or qkv.dtype != torch.float32 "
        "or not qkv.is_contiguous() or not"
    ),
]
STOCK_HOST = [
    "block_d = 32",
    "block_s = 256",
    "output_grid = (bsz, triton.cdiv(dim, block_d), triton.cdiv(seq_len, block_s))",
    "state_grid = (bsz, triton.cdiv(dim, block_d))",
    "block_k = triton.next_power_of_2(conv_kernel_size)",
    "block_state = triton.next_power_of_2(state_size)",
    "if seq_len <= 256:",
    "MAX_CUDA_SEQLEN = 32",
    "transpose_output = True,",
]
PY = [
    (
        "modules/gated_delta_net.py",
        (
            "mixed_qkv = None if fuse_qkv else "
            "qkv.transpose(1, 2).to(torch.bfloat16).contiguous()"
        ),
    ),
    (
        "modules/gated_delta_net.py",
        (
            "bsz == 1 and seqlen > 256 and seqlen >= self.num_v_heads "
            "and not save_history and"
        ),
    ),
    (
        "modules/gated_delta_net_fn/gated_delta_rule.py",
        "q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)",
    ),
    ("modules/gated_delta_net_fn/gated_delta_rule.py", "q, k, v = qkv_split"),
]
MUTATIONS = {
    "no_cast": (
        ").to(tl.bfloat16)\n            vals = tl.where",
        ")\n            vals = tl.where",
    ),
    "tap_order": (
        (
            "w = tl.load(weight + offs_d * conv_kernel_size + k, "
            "mask = mask_d, other = 0.0)\n"
            "            acc += vals * w[:, None]\n\n"
            "    if has_bias:\n"
            "        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)\n"
            "        acc += b[:, None]\n\n"
            "    acc = acc * tl.sigmoid(acc)\n"
            "    s64"
        ),
        (
            "w = tl.load(weight + offs_d * conv_kernel_size + k, "
            "mask = mask_d, other = 0.0)\n"
            "            acc = vals * w[:, None] + acc\n\n"
            "    if has_bias:\n"
            "        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)\n"
            "        acc += b[:, None]\n\n"
            "    acc = acc * tl.sigmoid(acc)\n"
            "    s64"
        ),
    ),
    "x_addr": (
        (
            "x + x_t[None, :].to(tl.int64) * dim + offs_d[:, None],\n"
            "                mask = mask_d[:, None] & mask_s[None, :]"
        ),
        (
            "x + offs_d[:, None] * seq_len + x_t[None, :],\n"
            "                mask = mask_d[:, None] & mask_s[None, :]"
        ),
    ),
}


def fail(msg: str) -> NoReturn:
    """Exit with the link's failure message.

    Raises:
        SystemExit: Always.

    """
    text = f"gdn_conv_qkv_diff: FAIL: {msg}"
    raise SystemExit(text)


def norm(text: str) -> list[str]:
    """Drop comments, blank lines and indentation; collapse whitespace.

    Returns:
        The normalized non-empty lines.

    """
    out: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", re.sub(r"#.*", "", raw).strip())
        if line:
            out.append(line)
    return out


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
        fail(f"{what}: end anchor {end!r} not found")
    return text[i:j]


def rewrite_stock(lines: list[str], what: str) -> list[str]:
    """Apply REWRITES (and the x cast) to the stock kernel lines.

    Returns:
        The rewritten lines.

    """
    out: list[str] = []
    used: set[int] = set()
    cast_next = False
    for x in lines:
        hit = False
        for k, (old, new) in enumerate(REWRITES):
            if x == old:
                used.add(k)
                hit = True
                if new is not None:
                    out.append(new)
                if k == X_ADDR_REWRITE:
                    cast_next = True
        if hit:
            continue
        if cast_next and x == ")":
            out.append(").to(tl.bfloat16)")
            cast_next = False
            continue
        out.append(x)
    if used != {0, 1, 2}:
        fail(f"{what}: stock rewrite anchors used {sorted(used)} (want all three)")
    return out


def compare(a: list[str], b: list[str], what: str) -> None:
    """Fail at the first differing line of a and b."""
    if a != b:
        for k, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                fail(f"{what}: line {k}: stock {x!r} != fused {y!r}")
        fail(f"{what}: lengths differ ({len(a)} vs {len(b)})")
    sys.stdout.write(f"gdn_conv_qkv_diff: {what}: {len(a)} normalized lines equal\n")


def mutated(name: str, conv: str) -> str:
    """Apply mutation `name` at its first anchor inside the fused kernels.

    Returns:
        The mutated conv1d.py source.

    """
    if name not in MUTATIONS:
        fail(f"unknown mutation {name!r}")
    old, new = MUTATIONS[name]
    if conv.count(old) < 1:
        fail(f"mutation anchor missing: {old!r}")
    fused_at = conv.find("def _conv1d_prefill_qkv_output_kernel(")
    k = conv.find(old, fused_at)
    conv = conv[:k] + new + conv[k + len(old) :]
    sys.stdout.write(f"gdn_conv_qkv_diff: applied mutation {name}\n")
    return conv


def main(argv: list[str]) -> None:
    """Check the fused kernels and host code against the engine tree."""
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: gdn_conv_qkv_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    conv = (root / "modules/gated_delta_net_fn/conv1d.py").read_text()
    if mutate is not None:
        conv = mutated(mutate, conv)

    so = norm(
        seg(
            conv,
            "def _causal_conv1d_update_slotted_output_kernel(",
            "acc = acc * tl.sigmoid(acc)",
            "stock output kernel",
        )
    )
    fo = norm(
        seg(
            conv,
            "def _conv1d_prefill_qkv_output_kernel(",
            "acc = acc * tl.sigmoid(acc)",
            "fused output kernel",
        )
    )
    so = so[so.index("):") + 1 :]
    fo = fo[fo.index("):") + 1 :]
    compare(rewrite_stock(so, "output kernel"), fo, "output kernel body up to the SiLU")
    once_in(
        conv,
        "acc = acc * tl.sigmoid(acc)",
        3,
        "the SiLU line (stock fused / output kernels, 5111 output kernel)",
    )
    ss = norm(
        seg(
            conv,
            "def _causal_conv1d_update_slotted_state_kernel(",
            "tl.store(",
            "stock state kernel",
        )
    )
    fs = norm(
        seg(
            conv,
            "def _conv1d_prefill_qkv_state_kernel(",
            "tl.store(",
            "fused state kernel",
        )
    )
    ss = ss[ss.index("):") + 1 :]
    fs = fs[fs.index("):") + 1 :]
    compare(rewrite_stock(ss, "state kernel"), fs, "state kernel body up to the store")
    st_s = norm(
        seg(
            conv,
            "def _causal_conv1d_update_slotted_state_kernel(",
            "def causal_conv1d_update_slotted_triton(",
            "stock state",
        )
    )
    st_f = norm(
        seg(
            conv,
            "def _conv1d_prefill_qkv_state_kernel(",
            "def conv1d_prefill_qkv(",
            "fused state",
        )
    )
    compare(
        st_s[st_s.index("tl.store(") :],
        st_f[st_f.index("tl.store(") :],
        "state kernel store",
    )

    for e in MODEL:
        once_in(conv, e, 1, "fused conv1d.py")
    for e in STOCK_HOST:
        if e not in conv:
            fail(f"stock host line missing: {e!r}")
    for f, e in PY:
        if e not in (root / f).read_text():
            fail(f"{f}: {e!r} missing")
    sys.stdout.write(
        f"gdn_conv_qkv_diff: {len(MODEL) + len(STOCK_HOST) + len(PY)} "
        "transcribed / host expressions found\n"
    )
    sys.stdout.write("gdn_conv_qkv_diff: OK\n")


def once_in(text: str, e: str, n: int, where: str) -> None:
    """Fail unless e occurs exactly n times in text."""
    c = text.count(e)
    if c != n:
        fail(f"{where}: {e!r} occurs {c} times (want {n})")


if __name__ == "__main__":
    main(sys.argv)
