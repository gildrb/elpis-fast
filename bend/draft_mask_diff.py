#!/usr/bin/env python3
"""Byte-for-byte differential of the DFlash2 draft mask model against the engine's own code.

Bend side: bend/DRAFT_MASK_TABLE.bend (bend/draft_mask.bend, proven against the z-lab reference
by bend/draft_mask_proof.bend) compiled with the pinned toolchain and run.
Engine side, from the patched engine tree (exl3 0001-0003 + 0005, then the exl3-ext series),
every file hash-checked against its pin:
  - architecture/dflash.py: dflash2_kernel_window (0005), executed as written (ast-extracted);
  - architecture/dflash2.py: the draft forward's causal flag (`params["causal"] = False`);
  - modules/attention_fn/triton_paged.py: _normalize_window and gqa_geometry executed as written;
    the GQA split kernel's window start / split / tile lines (_gqa_live_split, _gqa_pass) and
    _gqa_step's mask lines quoted verbatim (each asserted present) and evaluated per scalar key
    (tl.* -> Python builtins); the 3005 dispatch lines that keep every row of a non-causal launch
    (the draft) in pass A with _gqa_live_split's (lo, s_len) are asserted present.
For every table row the set of keys the kernel reaches its softmax with must equal the Bend bits,
under the served split geometry (bsz 1, 8 kv heads, 2 h-blocks, grid_y 10, 82 SMs) and two
others. `--mutate NAME` perturbs one engine line and must FAIL.

Usage: python3 -I -B draft_mask_diff.py --engine <tree> [--table <table.txt>] [--mutate NAME]
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
BEND = "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend"
LOCK = ["flock", "-s", "/tmp/elpis-gpu.lock", "nice", "-n", "19"]
PINS = {  # post-images: 0005 scratch manifest (dflash.py), repo exl3-ext.json (the others)
    "architecture/dflash.py": "fa07b1b8263b3f725f01f7ba5f27438702ac3ed5993e8839d1dafe44b568b8f4",
    "architecture/dflash2.py": "16a2982724b070b7328a652e4127fda0da3e3836228234bc864f6190b8657fa2",
    "modules/attention_fn/triton_paged.py": "12896d430c94639ec4157a967f1635cc4a0dcc2198294d6349a9a2749ed9623e",
}
KERNEL_START = [
    "    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len",
    "    lo = total_k_len * 0",
    "    if WINDOW_LEFT >= 0:",
    "        lo = (tl.maximum(0, total_k_len - q_len - WINDOW_LEFT) // BLOCK_N) * BLOCK_N",
    "    span = total_k_len - lo",
    "    live = tl.maximum(1, tl.minimum(SPLITS, tl.cdiv(span, MIN_SPLIT)))",
    "    return lo, tl.cdiv(tl.cdiv(span, live), BLOCK_N) * BLOCK_N",
    "    n_start = lo + split * s_len",
    "    n_end = tl.minimum(n_start + s_len, total_k_len)",
]
KERNEL_TILE = [
    "    for n0 in range(n_start, n_end, BLOCK_N):",
    "        offs_n = n0 + tl.arange(0, BLOCK_N)",
]
# 3005 dispatch: rows start in pass A; only a causal full-context launch (not the draft) reassigns
# s_a or runs pass B; pass A takes (lo, s_a) from _gqa_live_split, which _gqa_pass reads as (lo, s_len)
KERNEL_DISPATCH = [
    "    b0 = vr0 & False",
    "    lo, s_a = _gqa_live_split(total_k_len, q_len, WINDOW_LEFT, SPLITS, MIN_SPLIT, BLOCK_N)",
    "    if CAUSAL:",
    "        if WINDOW_LEFT < 0:",
    "              batch, kv_head, bh, split, lo, s_a, total_k_len, num_pages_per_seq, store_acc,",
    "              q0, vr0 & ~b0, qa0, ~b0, q1, vr1 & ~b1, qa1, ~b1,",
    "              batch, kv_head, bh, split, lo, s_len, total_k_len, num_pages_per_seq, store_acc,",
]
LIVE_SPLIT_RETURN = "return lo, "
KERNEL_QABS = "    q_abs = total_k_len - q_len + row_q"
KERNEL_MASK = [
    "    valid = valid_row[:, None] & (offs_n[None, :] < n_end)",
    "    if CAUSAL:",
    "        valid = valid & (offs_n[None, :] <= q_abs[:, None])",
    "    if WINDOW_LEFT >= 0:",
    "        valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)",
    "    if WINDOW_RIGHT >= 0:",
    "        valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)",
]
DRAFT_CAUSAL = '        params["causal"] = False'
MUTATIONS = {
    # the pre-0005 block causality: right window 0 for a non-causal checkpoint
    "right_zero": ("architecture/dflash.py", "return (window - 1, 0 if causal else window - 1)",
                   "return (window - 1, 0 if causal else 0)"),
    # g7n's window convention (keys q - W .. q, one extra)
    "left_off_by_one": ("architecture/dflash.py", "return (window - 1, 0 if causal else window - 1)",
                        "return (window, 0 if causal else window - 1)"),
    # window start rounded up instead of down
    "start_round_up": ("modules/attention_fn/triton_paged.py",
                       "        lo = (tl.maximum(0, total_k_len - q_len - WINDOW_LEFT) // BLOCK_N) * BLOCK_N",
                       "        lo = tl.cdiv(tl.maximum(0, total_k_len - q_len - WINDOW_LEFT), BLOCK_N) * BLOCK_N"),
}
GEOMETRIES = [  # (bsz, n_kv_heads, h_blocks, grid_y, sm_count): served draft, then a 3- and 1-split
    (1, 8, 2, 10, 82),
    (1, 8, 2, 10, 12),
    (1, 8, 2, 10, 1),
]


def fail(msg: str) -> NoReturn:
    raise SystemExit(f"FAIL: {msg}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def extract(src: str, names: list[str], env: dict) -> dict:
    tree = ast.parse(src)
    ns = dict(env)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.get_source_segment(src, node)
            exec(compile(code, node.name, "exec"), ns)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            exec(compile(ast.get_source_segment(src, node), node.targets[0].id, "exec"), ns)
            found.add(node.targets[0].id)
    if found != set(names):
        fail(f"missing definitions {set(names) - found}")
    return ns


def need(lines: set[str], quoted: list[str]):
    for q in quoted:
        if q not in lines:
            fail(f"kernel line not found verbatim: {q!r}")


def scalar(line: str) -> str:
    s = line.strip()
    for a, b in (("tl.load(cache_seqlens + batch)", "S"), ("tl.maximum", "max"), ("tl.minimum", "min"),
                 ("tl.cdiv", "cdiv"), ("offs_n[None, :]", "n"), ("q_abs[:, None]", "q_abs"),
                 ("valid_row[:, None]", "True"), ("n0 + tl.arange(0, BLOCK_N)", "n0")):
        s = s.replace(a, b)
    return s


def as_assignment(s: str) -> str:
    # _gqa_live_split returns (lo, s_len): its return line is the kernel's s_len assignment
    if s.startswith(LIVE_SPLIT_RETURN):
        return "s_len = " + s[len(LIVE_SPLIT_RETURN):]
    return s


def build_kernel(tp_src: str, swap: tuple[str, str] | None = None):
    lines = set(tp_src.splitlines())
    need(lines, KERNEL_START + KERNEL_TILE + [KERNEL_QABS] + KERNEL_MASK + KERNEL_DISPATCH)
    # A kernel mutation perturbs the quoted line after the verbatim check, so it reaches the evaluation
    scalar_of = (lambda q: scalar(swap[1]) if swap and q == swap[0] else scalar(q))
    body = ["def keys(S, L, r, BLOCK_N, SPLITS, MIN_SPLIT, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT):",
            "    q_len = kv_append_len = L",
            "    row_q = r",
            "    out = set()",
            "    for split in range(SPLITS):"]
    for q in KERNEL_START:
        # the one indented line is the body of the preceding `if WINDOW_LEFT >= 0:`
        body.append(("            " if q.startswith("        ") else "        ") + as_assignment(scalar_of(q)))
    body.append("        " + scalar(KERNEL_TILE[0]))
    body.append("            base = " + scalar(KERNEL_TILE[1]).split("=", 1)[1].strip())
    body.append("            for n in range(base, base + BLOCK_N):")
    body.append("                " + scalar(KERNEL_QABS))
    for q in KERNEL_MASK:
        pad = "                    " if q.startswith("        ") else "                "
        body.append(pad + scalar(q))
    body.append("                if valid:")
    body.append("                    out.add(n)")
    body.append("    return out")
    ns = {"cdiv": lambda a, b: -(-a // b)}
    exec("\n".join(body), ns)
    return ns["keys"], "\n".join(body)


def bend_table(path: str | None) -> str:
    if path:
        return Path(path).read_text()
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "table"
        r = subprocess.run(LOCK + [BEND, str(HERE / "DRAFT_MASK_TABLE.bend"), "-o", str(exe)],
                           capture_output=True, text=True, cwd=HERE)
        if r.returncode:
            fail("Bend table compile failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        # The Bend runtime reserves its heap up front: lift an address-space cap (ulimit -v)
        def unlimited():
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        return subprocess.run([str(exe), "--gpu", "off"], capture_output=True, text=True, check=True,
                              preexec_fn=unlimited).stdout


def main(argv: list[str]) -> None:
    args = argv[1:]
    engine = None
    table = None
    mutate = None
    while args:
        k = args.pop(0)
        if k == "--engine":
            engine = Path(args.pop(0))
        elif k == "--table":
            table = args.pop(0)
        elif k == "--mutate":
            mutate = args.pop(0)
        else:
            fail(f"unknown argument {k}")
    if engine is None:
        fail("--engine <patched exllamav3 tree> is required")
    src = {}
    for rel, pin in PINS.items():
        b = (engine / rel).read_bytes()
        if sha(b) != pin:
            fail(f"{rel} differs from its pinned post-image")
        src[rel] = b.decode()
    swap = None
    if mutate:
        rel, old, new = MUTATIONS[mutate]
        if src[rel].count(old) != 1:
            fail(f"mutation anchor not unique: {mutate}")
        if rel == "modules/attention_fn/triton_paged.py":
            swap = (old, new)
        else:
            src[rel] = src[rel].replace(old, new)
    if DRAFT_CAUSAL not in src["architecture/dflash2.py"].splitlines():
        fail("dflash2.py no longer runs the draft forward with causal=False")

    os.environ.pop("EXL3_DFLASH2_WINDOW", None)
    fa = extract(src["architecture/dflash.py"],
                 ["DFLASH2_WINDOW_ENV", "DFLASH2_WINDOW_MAX", "dflash2_kernel_window"],
                 {"os": os, "re": re})
    tp = extract(src["modules/attention_fn/triton_paged.py"],
                 ["_normalize_window", "GQA_MAX_NSUB", "gqa_geometry"],
                 {"_gqa_enable": True, "_gqa_ctas_per_sm": 2, "_gqa_min_split": 0})
    keys, _ = build_kernel(src["modules/attention_fn/triton_paged.py"], swap)
    causal_of = {"none": None, "true": True, "false": False}

    text = bend_table(table)
    rows = 0
    bad = []
    pat = re.compile(r"cfg=(\w+) W=(\d+) S=(\d+) L=(\d+) r=(\d+) BN=(\d+) ([01]+)\Z")
    for line in text.splitlines():
        m = pat.match(line)
        if m is None:
            fail(f"malformed table line {line!r}")
        cfg, W, S, L, r, BN, bits = m.group(1), *map(int, m.groups()[1:6]), m.group(7)
        win = tp["_normalize_window"](fa["dflash2_kernel_window"](W, causal_of[cfg], ["sliding_attention"] * 5))
        for geo in GEOMETRIES:
            use, splits, min_split = tp["gqa_geometry"](*geo, BN)
            if not use:
                fail(f"geometry {geo} does not select the GQA kernel")
            got = keys(S, L, r, BN, splits, min_split, False, win[0], win[1])
            mine = "".join("1" if n in got else "0" for n in range(S + L + 2))
            if mine != bits:
                bad.append((line.split(" ")[:6], geo, mine, bits))
        rows += 1
    expect_rows = 3 * 2 * 9 * 21 * sum(range(1, 9))
    if rows != expect_rows:
        fail(f"table has {rows} rows, expected {expect_rows}")
    print(f"engine {engine}; table sha256 {sha(text.encode())}; {rows} rows x {len(GEOMETRIES)} split geometries")
    if bad:
        for b in bad[:5]:
            print("  mismatch", b)
        print(f"FAIL: {len(bad)} mismatching rows" + (f" (mutation {mutate})" if mutate else ""))
        sys.exit(1)
    print("PASS: engine mask == Bend model bit for bit" + (f" (mutation {mutate} NOT detected)" if mutate else ""))


if __name__ == "__main__":
    main(sys.argv)
