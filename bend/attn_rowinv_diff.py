#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential for the row-invariant verify-attention split (patch 3005).

Extracts, verbatim by regex, the index expressions of the pinned
modules/attention_fn/triton_paged.py (the _gqa_row_split and _gqa_live_split bodies, the
q_abs line of _gqa_rows, the n_start / n_end lines of _gqa_pass, the kernel's
pass-selection block (lo / s_a / s_b / b0), its pass-B launch guard, the pass row masks
and the store_acc line), evaluates them with bend/pysubset.py and a tiny `tl` shim
(Python // is Triton integer division on these non-negative values), prints the table
of bend/ATTN_ROWINV_TABLE.bend and compares it byte for byte with the Bend output.
Independently checks, per (c, r), that the row's non-empty chunks tile [0, c + r]
exactly once inside the SPLITS launched splits, that exactly one launched pass computes
the row, that store_acc holds on every split with a non-empty chunk, and that every
absolute position p has the same chunk list in every (c, r) with c + r = p.

usage: attn_rowinv_diff.py SHIPPED_TRITON_PAGED_PY [--bend-output FILE] [--mutate NAME]
  SHIPPED_TRITON_PAGED_PY: modules/attention_fn/triton_paged.py of OUT/patched of
                      bend/engine_trees.py --through 3005-attn-row-invariant-split.patch
  --bend-output FILE  compare with a saved `bend bend/ATTN_ROWINV_TABLE.bend` stdout
                      instead of running bend
  --mutate round_len  row split sized from the round length (_gqa_row_split(qa0, ->
                      _gqa_row_split(total_k_len - 1,)
  --mutate old_3002   keep 3002's round-relative s_a from _gqa_live_split (drop the
                      causal branch's s_a / s_b / b0 assignment)
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

REPO = source_link.REPO
TABLE = "bend/ATTN_ROWINV_TABLE.bend"
# pristine engine + committed ext series (through 3005)
TRITON_PAGED_SHA256 = "12896d430c94639ec4157a967f1635cc4a0dcc2198294d6349a9a2749ed9623e"

# Served geometry (ATTN_ROWINV_TABLE.bend): gqa_geometry MIN_SPLIT 0 -> 4 * BLOCK_N
SPLITS, BLOCK_N, MIN_SPLIT, Q_LEN = 39, 32, 128, 8
ROUND_STARTS = range(8193)
CAUSAL, WINDOW_LEFT = True, -1
# Violations longer than this are shortened in the report
VIOLATION_WIDTH = 240
TABLE_TIMEOUT = 3600

type Expr = Callable[[Mapping[str, object]], object]


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"attn_rowinv_diff: {msg}"
    raise SystemExit(text)


def once(
    pattern: str, src: str, what: str, flags: re.RegexFlag = re.MULTILINE
) -> re.Match[str]:
    """Return the only match of `pattern` in `src`.

    Args:
        pattern: The regular expression.
        src: The text.
        what: The name of the match in the failure message.
        flags: The regular expression flags.

    Returns:
        The match.

    """
    ms = list(re.finditer(pattern, src, flags))
    if len(ms) != 1:
        fail(f"{what}: pattern matched {len(ms)} times (expected 1)")
    return ms[0]


# ---------------------------------------------------------------------------------
# tl shim: the integer / boolean subset used by the extracted expressions


class _Constexpr:
    def __getitem__(self, _: object) -> _Constexpr:
        return self


class TL:
    """The `tl` names that the extracted functions use, on Python ints."""

    constexpr = _Constexpr()

    @staticmethod
    def maximum(a: int, b: int) -> int:
        """Return tl.maximum on scalars.

        Returns:
            The larger of a and b.

        """
        return max(a, b)

    @staticmethod
    def minimum(a: int, b: int) -> int:
        """Return tl.minimum on scalars.

        Returns:
            The smaller of a and b.

        """
        return min(a, b)

    @staticmethod
    def cdiv(a: int, b: int) -> int:
        """Return tl.cdiv on scalars.

        Returns:
            a / b rounded up.

        """
        return (a + b - 1) // b

    @staticmethod
    def static_assert(cond: object, *_: object) -> None:
        """Fail unless `cond` holds, as tl.static_assert at compile time."""
        if not cond:
            fail("tl.static_assert failed")


class TB:
    """A Triton boolean row mask (one row): & and ~ as on int1 tensors."""

    def __init__(self, *, v: object) -> None:
        """Hold the truth of `v`."""
        self.v = bool(v)

    def __and__(self, o: object) -> TB:
        """Return the mask and `o`.

        Returns:
            The conjunction.

        """
        return TB(v=self.v and bool(o))

    __rand__ = __and__

    def __invert__(self) -> TB:
        """Return the negated mask.

        Returns:
            The negation.

        """
        return TB(v=not self.v)

    def __bool__(self) -> bool:
        """Return the row's bit.

        Returns:
            The bit.

        """
        return self.v


# ---------------------------------------------------------------------------------
# Extraction

MUTATIONS = {
    "round_len": ("_gqa_row_split(qa0,", "_gqa_row_split(total_k_len - 1,"),
}

OLD_3002_BLOCK = (
    r"^            s_a = _gqa_row_split\(total_k_len - q_len, MIN_SPLIT, SPLITS, "
    r"BLOCK_N\)\n"
    r"            s_b = _gqa_row_split\(total_k_len - 1, MIN_SPLIT, SPLITS, BLOCK_N\)\n"
    r"            b0 = vr0 & \(_gqa_row_split\(qa0, MIN_SPLIT, SPLITS, BLOCK_N\) "
    r"!= s_a\)\n"
)
NSUB_HEADER = "            if NSUB > 1:\n"


def mutate(src: str, name: str | None) -> str:
    """Apply the mutation `name` (or none) to the source.

    Returns:
        The mutated source.

    """
    if name is None:
        return src
    if name == "old_3002":
        m = once(OLD_3002_BLOCK, src, "old_3002 causal s_a/s_b/b0 lines")
        return src[: m.start()] + src[m.end() :]
    if name not in MUTATIONS:
        fail(f"unknown mutation {name}")
    old, new = MUTATIONS[name]
    if src.count(old) != 1:
        fail(f"mutation {name}: {old!r} occurs {src.count(old)} times")
    return src.replace(old, new)


def group(m: re.Match[str], index: int) -> str:
    """Return group `index` of `m`.

    Returns:
        The group's text.

    """
    value = m.group(index)
    if not isinstance(value, str):
        fail(f"regex group {index} did not match")
    return value


def func(src: str, name: str) -> str:
    """Return the top-level @triton.jit function `name` (through the blank-line pair).

    Returns:
        The function's source, from its `def` line.

    """
    m = once(
        rf"^@triton\.jit\n(def {name}\(.*?)\n\n\n",
        src,
        f"def {name}",
        re.MULTILINE | re.DOTALL,
    )
    return group(m, 1)


@dataclass(frozen=True)
class Extracted:
    """The compiled source fragments of one triton_paged.py."""

    ns: dict[str, object]
    q_abs: Expr
    select: Callable[[dict[str, object]], None]
    store_acc: Expr
    guard: tuple[Expr, Expr]
    mask_a: Expr
    mask_b: Expr
    n_start: Expr
    n_end: Expr


def extract(src: str) -> Extracted:
    """Extract and compile the quoted fragments of the kernel source.

    Returns:
        The compiled fragments.

    """
    ns: dict[str, object] = {"tl": TL, "triton": None}
    for fn in ("_gqa_row_split", "_gqa_live_split"):
        pysubset.exec_block(func(src, fn), ns)
    rows, gpass = func(src, "_gqa_rows"), func(src, "_gqa_pass")
    kern = func(src, "_paged_attn_decode_gqa_split_kernel")
    q_abs = group(
        once(r"^    q_abs = (total_k_len - q_len \+ row_q)$", rows, "_gqa_rows q_abs"),
        1,
    )
    n_start = group(
        once(r"^    n_start = (lo \+ split \* s_len)$", gpass, "_gqa_pass n_start"),
        1,
    )
    n_end = group(
        once(
            r"^    n_end = (tl\.minimum\(n_start \+ s_len, total_k_len\))$",
            gpass,
            "_gqa_pass n_end",
        ),
        1,
    )
    once(
        r"^    for n0 in range\(n_start, n_end, BLOCK_N\):$",
        gpass,
        "_gqa_pass tile loop",
    )
    # the kernel's pass selection: from `b0 = vr0 & False` through the causal branch's
    # b0 line (under old_3002: through the causal branch header and its static_assert)
    blk = once(
        r"^    b0 = vr0 & False\n.*?^(?:            b0 = vr0 & \(.*?\)\n"
        r"|            if NSUB > 1:\n)",
        kern,
        "kernel pass selection block",
        re.MULTILINE | re.DOTALL,
    )
    text = group(blk, 0)
    if text.endswith(NSUB_HEADER):
        text = text[: -len(NSUB_HEADER)] + "            pass\n"
    store_acc = group(
        once(
            r"^    store_acc = (lo \+ split \* s_a < total_k_len)$", kern, "store_acc"
        ),
        1,
    )
    guard = once(
        r"^            if (q_len > 1):\n                if (s_b != s_a):\n"
        r"                    _gqa_pass\(",
        kern,
        "pass B launch guard",
    )
    mask_a = group(once(r"q0, (vr0 & ~b0), qa0, ~b0,", kern, "pass A row mask"), 1)
    mask_b = group(once(r"q0, (vr0 & b0), qa0, b0,", kern, "pass B row mask"), 1)
    return Extracted(
        ns=ns,
        q_abs=pysubset.compile_expr(q_abs),
        select=pysubset.compile_block(textwrap.dedent(text)),
        store_acc=pysubset.compile_expr(store_acc),
        guard=(
            pysubset.compile_expr(group(guard, 1)),
            pysubset.compile_expr(group(guard, 2)),
        ),
        mask_a=pysubset.compile_expr(mask_a),
        mask_b=pysubset.compile_expr(mask_b),
        n_start=pysubset.compile_expr(n_start),
        n_end=pysubset.compile_expr(n_end),
    )


# ---------------------------------------------------------------------------------
# Evaluation


@dataclass(frozen=True)
class Row:
    """One query row of the table: its splits, passes and chunks."""

    q_abs: int
    s_a: int
    s_b: int
    b: bool
    run: bool
    s: int
    in_a: bool
    in_b: bool
    chunks: list[tuple[int, int, int]]
    stored: list[bool]


def integer(value: object, what: str) -> int:
    """Return `value` as an int (a kernel index value).

    Returns:
        The value.

    """
    if not isinstance(value, int):
        fail(f"{what} is not an integer: {value!r}")
    return value


def split_chunks(
    x: Extracted, env: dict[str, object], q_abs: int, s_len: int
) -> tuple[list[tuple[int, int, int]], list[bool]]:
    """Evaluate the splits of one row after its pass selection ran in `env`.

    Returns:
        The non-empty (split, start, end) chunks and store_acc of each.

    """
    lo, s_a = env["lo"], env["s_a"]
    chunks: list[tuple[int, int, int]] = []
    stored: list[bool] = []
    for split in range(SPLITS):
        e = dict(env, lo=lo, split=split, s_len=s_len)
        n_start = integer(x.n_start(e), "n_start")
        e["n_start"] = n_start
        n_end = integer(x.n_end(e), "n_end")
        # tile loop covers [n_start, n_end); causal mask keeps keys <= q_abs
        hi = min(n_end, q_abs + 1)
        if n_start < hi:
            chunks.append((split, n_start, hi))
            acc = x.store_acc(dict(env, lo=lo, split=split, s_a=s_a))
            stored.append(bool(acc))
    return chunks, stored


def row(x: Extracted, c: int, r: int) -> Row:
    """Evaluate the extracted kernel lines for query row `r` of round start `c`.

    Returns:
        The row.

    """
    env = dict(
        x.ns,
        total_k_len=c + Q_LEN,
        q_len=Q_LEN,
        row_q=r,
        CAUSAL=CAUSAL,
        WINDOW_LEFT=WINDOW_LEFT,
        SPLITS=SPLITS,
        MIN_SPLIT=MIN_SPLIT,
        BLOCK_N=BLOCK_N,
    )
    q_abs = integer(x.q_abs(env), "q_abs")
    env.update(vr0=TB(v=True), qa0=q_abs)
    x.select(env)
    s_a, s_b = integer(env["s_a"], "s_a"), integer(env["s_b"], "s_b")
    b0 = env["b0"]
    runs_b = bool(x.guard[0](env)) and bool(x.guard[1](env))
    in_a = bool(x.mask_a(env))
    in_b = bool(x.mask_b(env)) and runs_b
    s_len = s_a if in_a else s_b
    chunks, stored = split_chunks(x, env, q_abs, s_len)
    return Row(
        q_abs=q_abs,
        s_a=s_a,
        s_b=s_b,
        b=bool(b0),
        run=runs_b,
        s=s_len,
        in_a=in_a,
        in_b=in_b,
        chunks=chunks,
        stored=stored,
    )


def line(c: int, r: int, t: Row) -> str:
    """Return the table line of row `t`.

    Returns:
        The line, with its newline.

    """
    return (
        f"c={c} r={r} sA={t.s_a} sB={t.s_b} b={int(t.b)} run={int(t.run)} s={t.s}"
        + "".join(f" {k}={lo}:{hi}" for k, lo, hi in t.chunks)
        + "\n"
    )


def row_violations(c: int, r: int, t: Row) -> list[str]:
    """Check one row: q_abs, the tiling of [0, p], one pass, store_acc.

    Returns:
        The violations.

    """
    viol: list[str] = []
    p = t.q_abs
    if p != c + r:
        viol.append(f"c={c} r={r}: q_abs {p} != c + r")
    # tiling of [0, p] exactly once (interval arithmetic)
    at = 0
    for k, lo, hi in t.chunks:
        if k >= SPLITS:
            viol.append(f"c={c} r={r}: chunk index {k} >= SPLITS")
        if lo != at:
            viol.append(
                f"c={c} r={r}: chunk {k} starts at {lo}, expected {at} (gap/overlap)"
            )
        at = max(at, hi)
    if at != p + 1:
        viol.append(f"c={c} r={r}: chunks end at {at}, expected {p + 1}")
    if int(t.in_a) + int(t.in_b) != 1:
        viol.append(
            f"c={c} r={r}: computed by {int(t.in_a) + int(t.in_b)} launched passes"
        )
    for (k, _, _), ok in zip(t.chunks, t.stored, strict=False):
        if not ok:
            viol.append(f"c={c} r={r}: store_acc False on live split {k}")
    return viol


def table(x: Extracted) -> tuple[str, list[str]]:
    """Evaluate every row and check them.

    Returns:
        The table text and the violations.

    """
    out: list[str] = []
    viol: list[str] = []
    by_pos: dict[int, tuple[tuple[tuple[int, int, int], ...], int, int]] = {}
    for c in ROUND_STARTS:
        for r in range(Q_LEN):
            t = row(x, c, r)
            out.append(line(c, r, t))
            viol += row_violations(c, r, t)
            p = t.q_abs
            cl = tuple(t.chunks)
            prev = by_pos.setdefault(p, (cl, c, r))
            if prev[0] != cl:
                viol.append(
                    f"p={p}: chunks {list(cl)} at (c={c}, r={r}) != "
                    f"{list(prev[0])} at (c={prev[1]}, r={prev[2]})"
                )
    return "".join(out), viol


def bend_table(saved: str | None) -> str:
    """Return the Bend table: a saved stdout, or the compiled table's output.

    Returns:
        The table text.

    """
    if saved is not None:
        return Path(saved).read_text(encoding="utf-8")
    # Compiled with the pinned toolchain and run natively: the interpreter needs
    # hours for the 65,544 rows, the compiled program about a minute
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "table"
        bend = source_link.bend()
        p = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
            source_link.locked([bend, TABLE, "-o", str(exe)]),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            fail(f"{bend} {TABLE} -o exited {p.returncode}: {p.stderr.strip()}")
        # The Bend runtime reserves its heap up front: lift an address-space cap
        # (ulimit -v); the table process inherits it
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        p = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
            source_link.locked([str(exe), "--gpu", "off"]),
            capture_output=True,
            text=True,
            timeout=TABLE_TIMEOUT,
            check=False,
        )
    if p.returncode != 0:
        fail(f"compiled {TABLE} exited {p.returncode}: {p.stderr.strip()}")
    return p.stdout


def report(tag: str, ours: str, theirs: str, viol: list[str]) -> int:
    """Print the table mismatch and the violations.

    Returns:
        The exit status.

    """
    status = 0
    if ours != theirs:
        diff = list(
            difflib.unified_diff(
                theirs.splitlines(keepends=True),
                ours.splitlines(keepends=True),
                "ATTN_ROWINV_TABLE.bend",
                "triton_paged.py",
                n=0,
            )
        )
        body = [d for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---"))]
        sys.stdout.write(
            f"attn_rowinv_diff{tag}: TABLE MISMATCH: "
            f"{sum(d.startswith('-') for d in body)} Bend lines vs "
            f"{sum(d.startswith('+') for d in body)} Python lines differ; first:\n"
        )
        sys.stdout.writelines(diff[:12])
        status = 1
    if viol:
        sys.stdout.write(f"attn_rowinv_diff{tag}: {len(viol)} VIOLATIONS; first:\n")
        for v in viol[:8]:
            shown = v if len(v) <= VIOLATION_WIDTH else v[:VIOLATION_WIDTH] + " ..."
            sys.stdout.write(f"  {shown}\n")
        status = 1
    if status == 0:
        sys.stdout.write(
            f"attn_rowinv_diff{tag}: {len(ours.splitlines())} rows match byte for "
            f"byte ({len(ours)} bytes); 0 violations\n"
        )
    return status


def main(argv: list[str]) -> int:
    """Run the differential.

    Args:
        argv: The command line.

    Returns:
        The exit status.

    """
    args = argv[1:]
    saved = mutation = None
    pos: list[str] = []
    while args:
        a = args.pop(0)
        if a == "--bend-output":
            if not args:
                fail("--bend-output needs FILE")
            saved = args.pop(0)
        elif a == "--mutate":
            if not args:
                fail("--mutate needs NAME")
            mutation = args.pop(0)
        else:
            pos.append(a)
    if len(pos) != 1:
        fail(
            "usage: attn_rowinv_diff.py SHIPPED_TRITON_PAGED_PY "
            "[--bend-output FILE] [--mutate NAME]"
        )
    path = Path(pos[0])
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != TRITON_PAGED_SHA256:
        fail(f"{path}: sha256 {got} != {TRITON_PAGED_SHA256}")
    src = mutate(path.read_text(encoding="utf-8"), mutation)
    x = extract(src)
    ours, viol = table(x)
    theirs = bend_table(saved)
    tag = f" [mutation {mutation}]" if mutation else ""
    return report(tag, ours, theirs, viol)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
