#!/usr/bin/env python3
"""
Finite differential for the row-invariant verify-attention partition (ext patch 3005).

Extracts, verbatim by regex, the index expressions of the pinned
modules/attention_fn/triton_paged.py (the _gqa_row_split and _gqa_live_split bodies, the q_abs
line of _gqa_rows, the n_start / n_end lines of _gqa_pass, the kernel's pass-selection block
(lo / s_a / s_b / b0), its pass-B launch guard, the pass row masks and the store_acc line),
evaluates them in Python with a tiny `tl` shim (Python // is Triton integer division on these
non-negative values), prints the table of bend/ATTN_ROWINV_TABLE.bend and compares it byte for
byte with the Bend output. Independently checks, per (c, r), that the row's non-empty chunks tile
[0, c + r] exactly once inside the SPLITS launched splits, that exactly one launched pass computes
the row, that store_acc holds on every split with a non-empty chunk, and that every absolute
position p has the same chunk list in every (c, r) with c + r = p.

usage: attn_rowinv_diff.py SHIPPED_TRITON_PAGED_PY [--bend-output FILE] [--mutate NAME]
  SHIPPED_TRITON_PAGED_PY: modules/attention_fn/triton_paged.py of OUT/patched of
                      bend/engine_trees.py --through 3005-attn-row-invariant-split.patch
  --bend-output FILE  compare with a saved `bend bend/ATTN_ROWINV_TABLE.bend` stdout instead of
                      running bend
  --mutate round_len  row split sized from the round length (_gqa_row_split(qa0, ->
                      _gqa_row_split(total_k_len - 1,)
  --mutate old_3002   keep 3002's round-relative s_a from _gqa_live_split (drop the causal
                      branch's s_a / s_b / b0 assignment)
Exit 0 only when the tables match and no check is violated.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
TABLE = "bend/ATTN_ROWINV_TABLE.bend"
# pristine engine + committed ext series (through 3005)
TRITON_PAGED_SHA256 = "12896d430c94639ec4157a967f1635cc4a0dcc2198294d6349a9a2749ed9623e"

# Served geometry (ATTN_ROWINV_TABLE.bend): gqa_geometry MIN_SPLIT 0 -> 4 * BLOCK_N
SPLITS, BLOCK_N, MIN_SPLIT, Q_LEN = 39, 32, 128, 8
ROUND_STARTS = range(0, 8193)
CAUSAL, WINDOW_LEFT = True, -1


def fail(msg: str):
    raise SystemExit(f"attn_rowinv_diff: {msg}")


def check(cond: bool, msg: str):
    if not cond:
        fail(msg)


def once(pattern: str, src: str, what: str, flags=re.M) -> re.Match:
    ms = list(re.finditer(pattern, src, flags))
    check(len(ms) == 1, f"{what}: pattern matched {len(ms)} times (expected 1)")
    return ms[0]


# ---------------------------------------------------------------------------------------------
# tl shim: the integer / boolean subset used by the extracted expressions

class _Constexpr:
    def __getitem__(self, _):
        return self


class TL:
    constexpr = _Constexpr()

    @staticmethod
    def maximum(a, b):
        return max(a, b)

    @staticmethod
    def minimum(a, b):
        return min(a, b)

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def static_assert(cond, *_):
        check(bool(cond), "tl.static_assert failed")


class TB:
    """a Triton boolean row mask (one row): & and ~ as on int1 tensors"""

    def __init__(self, v: bool):
        self.v = bool(v)

    def __and__(self, o):
        return TB(self.v and bool(o))

    __rand__ = __and__

    def __invert__(self):
        return TB(not self.v)

    def __bool__(self):
        return self.v


# ---------------------------------------------------------------------------------------------
# Extraction

MUTATIONS = {
    "round_len": ("_gqa_row_split(qa0,", "_gqa_row_split(total_k_len - 1,"),
}

OLD_3002_BLOCK = (r"^            s_a = _gqa_row_split\(total_k_len - q_len, MIN_SPLIT, SPLITS, BLOCK_N\)\n"
                  r"            s_b = _gqa_row_split\(total_k_len - 1, MIN_SPLIT, SPLITS, BLOCK_N\)\n"
                  r"            b0 = vr0 & \(_gqa_row_split\(qa0, MIN_SPLIT, SPLITS, BLOCK_N\) != s_a\)\n")


def mutate(src: str, name: str | None) -> str:
    if name is None:
        return src
    if name == "old_3002":
        m = once(OLD_3002_BLOCK, src, "old_3002 causal s_a/s_b/b0 lines")
        return src[:m.start()] + src[m.end():]
    check(name in MUTATIONS, f"unknown mutation {name}")
    old, new = MUTATIONS[name]
    check(src.count(old) == 1, f"mutation {name}: {old!r} occurs {src.count(old)} times")
    return src.replace(old, new)


def func(src: str, name: str) -> str:
    """the source of the top-level @triton.jit function `name` (through the next blank-line pair)"""
    return once(rf"^@triton\.jit\n(def {name}\(.*?)\n\n\n", src, f"def {name}", re.M | re.S).group(1)


def extract(src: str) -> dict:
    ns = {"tl": TL, "triton": None}
    for fn in ("_gqa_row_split", "_gqa_live_split"):
        exec(compile(func(src, fn), f"triton_paged.py:{fn}", "exec"), ns)
    rows, gpass = func(src, "_gqa_rows"), func(src, "_gqa_pass")
    kern = func(src, "_paged_attn_decode_gqa_split_kernel")
    x = {"ns": ns}
    x["q_abs"] = once(r"^    q_abs = (total_k_len - q_len \+ row_q)$", rows, "_gqa_rows q_abs").group(1)
    x["n_start"] = once(r"^    n_start = (lo \+ split \* s_len)$", gpass, "_gqa_pass n_start").group(1)
    x["n_end"] = once(r"^    n_end = (tl\.minimum\(n_start \+ s_len, total_k_len\))$", gpass, "_gqa_pass n_end").group(1)
    once(r"^    for n0 in range\(n_start, n_end, BLOCK_N\):$", gpass, "_gqa_pass tile loop")
    # the kernel's pass selection: from `b0 = vr0 & False` through the causal branch's b0 line
    # (under old_3002: through the causal branch header and its static_assert)
    blk = once(r"^    b0 = vr0 & False\n.*?^(?:            b0 = vr0 & \(.*?\)\n|            if NSUB > 1:\n)",
               kern, "kernel pass selection block", re.M | re.S)
    text = blk.group(0)
    if text.endswith("            if NSUB > 1:\n"):
        text = text[:-len("            if NSUB > 1:\n")] + "            pass\n"
    x["select"] = compile(textwrap.dedent(text), "triton_paged.py:kernel pass selection", "exec")
    x["store_acc"] = once(r"^    store_acc = (lo \+ split \* s_a < total_k_len)$", kern, "store_acc").group(1)
    guard = once(r"^            if (q_len > 1):\n                if (s_b != s_a):\n                    _gqa_pass\(",
                 kern, "pass B launch guard")
    x["guard"] = (guard.group(1), guard.group(2))
    x["mask_a"] = once(r"q0, (vr0 & ~b0), qa0, ~b0,", kern, "pass A row mask").group(1)
    x["mask_b"] = once(r"q0, (vr0 & b0), qa0, b0,", kern, "pass B row mask").group(1)
    return x


# ---------------------------------------------------------------------------------------------
# Evaluation

def row(x: dict, c: int, r: int):
    ns = x["ns"]
    env = dict(ns, total_k_len=c + Q_LEN, q_len=Q_LEN, row_q=r, CAUSAL=CAUSAL,
               WINDOW_LEFT=WINDOW_LEFT, SPLITS=SPLITS, MIN_SPLIT=MIN_SPLIT, BLOCK_N=BLOCK_N)
    q_abs = eval(x["q_abs"], env)
    env.update(vr0=TB(True), qa0=q_abs)
    exec(x["select"], env)
    s_a, s_b, b0, lo = env["s_a"], env["s_b"], env["b0"], env["lo"]
    runs_b = bool(eval(x["guard"][0], env)) and bool(eval(x["guard"][1], env))
    in_a = bool(eval(x["mask_a"], env))
    in_b = bool(eval(x["mask_b"], env)) and runs_b
    s_len = s_a if in_a else s_b
    chunks, stored = [], []
    for split in range(SPLITS):
        e = dict(env, lo=lo, split=split, s_len=s_len)
        n_start = eval(x["n_start"], e)
        e["n_start"] = n_start
        n_end = eval(x["n_end"], e)
        # tile loop covers [n_start, n_end); causal mask keeps keys <= q_abs
        hi = min(n_end, q_abs + 1)
        if n_start < hi:
            chunks.append((split, n_start, hi))
            stored.append(bool(eval(x["store_acc"], dict(env, lo=lo, split=split, s_a=s_a))))
    return dict(q_abs=q_abs, s_a=s_a, s_b=s_b, b=bool(b0), run=runs_b, s=s_len, in_a=in_a, in_b=in_b,
                chunks=chunks, stored=stored)


def line(c: int, r: int, t: dict) -> str:
    return (f"c={c} r={r} sA={t['s_a']} sB={t['s_b']} b={int(t['b'])} run={int(t['run'])} s={t['s']}"
            + "".join(f" {k}={lo}:{hi}" for k, lo, hi in t["chunks"]) + "\n")


def table(x: dict):
    out, viol, by_pos = [], [], {}
    for c in ROUND_STARTS:
        for r in range(Q_LEN):
            t = row(x, c, r)
            out.append(line(c, r, t))
            p = t["q_abs"]
            if p != c + r:
                viol.append(f"c={c} r={r}: q_abs {p} != c + r")
            # tiling of [0, p] exactly once (interval arithmetic)
            at = 0
            for k, lo, hi in t["chunks"]:
                if k >= SPLITS:
                    viol.append(f"c={c} r={r}: chunk index {k} >= SPLITS")
                if lo != at:
                    viol.append(f"c={c} r={r}: chunk {k} starts at {lo}, expected {at} (gap/overlap)")
                at = max(at, hi)
            if at != p + 1:
                viol.append(f"c={c} r={r}: chunks end at {at}, expected {p + 1}")
            if int(t["in_a"]) + int(t["in_b"]) != 1:
                viol.append(f"c={c} r={r}: computed by {int(t['in_a']) + int(t['in_b'])} launched passes")
            for (k, _, _), ok in zip(t["chunks"], t["stored"]):
                if not ok:
                    viol.append(f"c={c} r={r}: store_acc False on live split {k}")
            cl = tuple(t["chunks"])
            prev = by_pos.setdefault(p, (cl, c, r))
            if prev[0] != cl:
                viol.append(f"p={p}: chunks {list(cl)} at (c={c}, r={r}) != {list(prev[0])} at (c={prev[1]}, r={prev[2]})")
    return "".join(out), viol


def bend_table(saved: str | None) -> str:
    if saved is not None:
        return Path(saved).read_text()
    # Compiled with the pinned toolchain and run natively: the interpreter needs hours for the
    # 65,544 rows, the compiled program about a minute
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "table"
        bend = source_link.bend()
        p = subprocess.run(source_link.locked([bend, TABLE, "-o", str(exe)]), cwd=REPO, capture_output=True, text=True)
        check(p.returncode == 0, f"{bend} {TABLE} -o exited {p.returncode}: {p.stderr.strip()}")

        def unlimited() -> None:
            # The Bend runtime reserves its heap up front: lift an address-space cap (ulimit -v)
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

        p = subprocess.run(source_link.locked([str(exe), "--gpu", "off"]), capture_output=True, text=True, timeout=3600,
                           preexec_fn=unlimited)
    check(p.returncode == 0, f"compiled {TABLE} exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout


def main(argv: list) -> int:
    args = argv[1:]
    saved = mutation = None
    pos = []
    while args:
        a = args.pop(0)
        if a == "--bend-output":
            check(bool(args), "--bend-output needs FILE")
            saved = args.pop(0)
        elif a == "--mutate":
            check(bool(args), "--mutate needs NAME")
            mutation = args.pop(0)
        else:
            pos.append(a)
    check(len(pos) == 1, "usage: attn_rowinv_diff.py SHIPPED_TRITON_PAGED_PY [--bend-output FILE] [--mutate NAME]")
    path = Path(pos[0])
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    check(got == TRITON_PAGED_SHA256, f"{path}: sha256 {got} != {TRITON_PAGED_SHA256}")
    src = mutate(path.read_text(), mutation)
    x = extract(src)
    ours, viol = table(x)
    theirs = bend_table(saved)
    tag = f" [mutation {mutation}]" if mutation else ""
    status = 0
    if ours != theirs:
        diff = list(difflib.unified_diff(theirs.splitlines(keepends=True), ours.splitlines(keepends=True),
                                         "ATTN_ROWINV_TABLE.bend", "triton_paged.py", n=0))
        body = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---"))]
        print(f"attn_rowinv_diff{tag}: TABLE MISMATCH: {sum(d.startswith('-') for d in body)} Bend lines vs "
              f"{sum(d.startswith('+') for d in body)} Python lines differ; first:")
        sys.stdout.writelines(diff[:12])
        status = 1
    if viol:
        print(f"attn_rowinv_diff{tag}: {len(viol)} VIOLATIONS; first:")
        for v in viol[:8]:
            print("  " + (v if len(v) <= 240 else v[:240] + " ..."))
        status = 1
    if status == 0:
        print(f"attn_rowinv_diff{tag}: {len(ours.splitlines())} rows match byte for byte "
              f"({len(ours)} bytes); 0 violations")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
