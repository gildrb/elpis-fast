#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential check of bend/gdn_ba_ksplit.bend against the shipped kernel.

The model describes ext 5106: the k-split b/a GEMV of
gdn_conv_rule_norm_kernel<..., KS, STAMP>.

Quoted verbatim from the patched engine tree (argument = the exllamav3 package
directory), exllamav3_ext/gdn.cu:
  - the constants GR_KS_ITERS, GR_KS_ROWS, GR_WARPS, GR_KS_DEFAULT and the kernel's
    THREADS; warp_id / lane_id; k2
  - ks_load: the i loop header, the element index `const int j = ...` (also in the
    FMA loop: both must be the same expression), the `if (j < k2)` guard, the ba_w
    row bases (b row h, a row H + h), the x index
  - the FMA loop: the i and s loop headers, `if (s < S)`, the four fmaf lines (their
    order and the x / w components they pair)
  - the reduction: the offset loop header, the two __shfl_down_sync lines,
    `if (lane_id == 0)`, the two ks_part stores (write slots)
  - the thread-t read: `bv = 0.0f; av = 0.0f;`, the w loop header, the two ks_part
    loads (read slots), the two bias lines
  - the launcher's fallback line
    `if (S > GR_KS_ROWS || k / 2 > 512 * GR_KS_ITERS) ks = 0;`
  - the block event order: the positions of the `if constexpr (KS ...)` state /
    ks_load lines, the norm-operand prefetch, the conv, the GEMV branches, the first
    __syncthreads() after them, the conv_sync arrival, the v-window write-back, the
    b/a read and gdn_rule_tokens (whose own l2-norm / __syncthreads / m.dot2 order is
    quoted from its body); no other m.dot2 / ks_part store between the partial
    stores and the token loop
bend/pysubset.py evaluates the quoted index expressions and Python re-runs the
quoted loops symbolically (an fp32 value is the tree of the fmaf / add operations
that produced it; shfl_down lanes past 31 read their own value, as on the GPU). The
result is printed in GDN_BA_KSPLIT_TABLE.bend's format and compared byte for byte
with that table's output (compiled with the pinned bend).

Differential evidence on finite instances, not a proof. `--mutate NAME` applies a
deliberate source mutation that the check must reject.

Usage: python3 bend/gdn_ba_ksplit_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

REPO = source_link.REPO
TABLE = "bend/GDN_BA_KSPLIT_TABLE.bend"
# The usage message: the module docstring of the pre-lint revision, verbatim.
USAGE = (
    "\n"
    "Finite differential check of bend/gdn_ba_ksplit.bend (ext 5106: the k-split b/a "
    "GEMV of\n"
    "gdn_conv_rule_norm_kernel<..., KS, STAMP>) against the shipped kernel source.\n"
    "\n"
    "Quoted verbatim from the patched engine tree (argument = the exllamav3 package "
    "directory),\n"
    "exllamav3_ext/gdn.cu:\n"
    "  - the constants GR_KS_ITERS, GR_KS_ROWS, GR_WARPS, GR_KS_DEFAULT and the "
    "kernel's THREADS;\n"
    "    warp_id / lane_id; k2\n"
    "  - ks_load: the i loop header, the element index `const int j = ...` (also in "
    "the FMA loop: both\n"
    "    must be the same expression), the `if (j < k2)` guard, the ba_w row bases "
    "(b row h, a row H + h),\n"
    "    the x index\n"
    "  - the FMA loop: the i and s loop headers, `if (s < S)`, the four fmaf lines "
    "(their order and the\n"
    "    x / w components they pair)\n"
    "  - the reduction: the offset loop header, the two __shfl_down_sync lines, `if "
    "(lane_id == 0)`, the two\n"
    "    ks_part stores (write slots)\n"
    "  - the thread-t read: `bv = 0.0f; av = 0.0f;`, the w loop header, the two "
    "ks_part loads (read slots),\n"
    "    the two bias lines\n"
    "  - the launcher's fallback line `if (S > GR_KS_ROWS || k / 2 > 512 * "
    "GR_KS_ITERS) ks = 0;`\n"
    "  - the block event order: the positions of the `if constexpr (KS ...)` state / "
    "ks_load lines, the\n"
    "    norm-operand prefetch, the conv, the GEMV branches, the first "
    "__syncthreads() after them, the\n"
    "    conv_sync arrival, the v-window write-back, the b/a read and "
    "gdn_rule_tokens (whose own l2-norm /\n"
    "    __syncthreads / m.dot2 order is quoted from its body); no other m.dot2 / "
    "ks_part store between\n"
    "    the partial stores and the token loop\n"
    "Python evaluates the quoted index expressions and re-runs the quoted loops "
    "symbolically (an fp32\n"
    "value is the tree of the fmaf / add operations that produced it; shfl_down "
    "lanes past 31 read their\n"
    "own value, as on the GPU). The result is printed in GDN_BA_KSPLIT_TABLE.bend's "
    "format and compared\n"
    "byte for byte with that table's output (compiled with the pinned bend).\n"
    "\n"
    "Differential evidence on finite instances, not a proof. `--mutate NAME` applies "
    "a deliberate source\n"
    "mutation that the check must reject.\n"
    "\n"
    "Usage: python3 bend/gdn_ba_ksplit_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR\n"
    "  ENGINE_PACKAGE_DIR: OUT/patched of bend/engine_trees.py.\n"
)
MUTATE_ARGC = 3
PLAIN_ARGC = 2
THREADS_PER_BLOCK = 512
ITERS_PER_THREAD = 5
WARP_LANES = 32
SLOT_CASES = 256
KS_MODES = 3
BAR_PARTIAL_STORES = 2
BEND_TIMEOUT = 1800
# The V lines' launches: S rows and k columns each
FALLBACK_CASES = (
    (1, 5120),
    (8, 5120),
    (9, 5120),
    (16, 5120),
    (8, 5121),
    (8, 5122),
    (1, 2),
    (3, 6144),
)
ALL_KS = frozenset({0, 1, 2})

# Each mutation maps its name to the original text, its replacement and the
# number of occurrences.
MUTATIONS = {
    "stride256": ("lane_id + i * THREADS;", "lane_id + i * 256;", 2),
    "warps_reversed": (
        "for (int w = 0; w < GR_WARPS; ++w)",
        "for (int w = GR_WARPS - 1; w >= 0; --w)",
        1,
    ),
    "no_barrier": (
        (
            "        gdn_wstamp<STAMP>(wstamps, 2, pb[0] + pa[0]);\n"
            "    }\n"
            "    __syncthreads();\n"
        ),
        "        gdn_wstamp<STAMP>(wstamps, 2, pb[0] + pa[0]);\n    }\n",
        1,
    ),
    "fallback16": (
        "if (S > GR_KS_ROWS || k / 2 > 512 * GR_KS_ITERS) ks = 0;",
        "if (S > 16 || k / 2 > 512 * GR_KS_ITERS) ks = 0;",
        1,
    ),
    "slot_collide": (
        "ks_part[warp_id * 2 * GR_KS_ROWS + GR_KS_ROWS + s] = pa[s];",
        "ks_part[warp_id * 2 * GR_KS_ROWS + s] = pa[s];",
        1,
    ),
    "fma_order": (
        (
            "pb[s] = fmaf(xf.x, wbf.x, pb[s]);\n"
            "                        pb[s] = fmaf(xf.y, wbf.y, pb[s]);"
        ),
        (
            "pb[s] = fmaf(xf.y, wbf.y, pb[s]);\n"
            "                        pb[s] = fmaf(xf.x, wbf.x, pb[s]);"
        ),
        1,
    ),
}


def fail(msg: str) -> NoReturn:
    """Stop with a FAIL message.

    Args:
        msg: The failure description.

    Raises:
        SystemExit: Always.

    """
    text = f"gdn_ba_ksplit_diff: {msg}"
    raise SystemExit(text)


def grab_groups(text: str, pattern: str, count: int = 1) -> list[tuple[str, ...]]:
    """Return every match's groups ("" for a group that did not participate).

    Args:
        text: The searched text.
        pattern: The regular expression.
        count: The required number of matches.

    Returns:
        The group tuples, as `re.findall` returns them for two or more groups.

    """
    m = [
        tuple(g if isinstance(g, str) else "" for g in found.groups())
        for found in re.finditer(pattern, text)
    ]
    if len(m) != count:
        fail(f"pattern {pattern!r}: {len(m)} matches, expected {count}")
    return m


def grab(text: str, pattern: str, count: int = 1) -> list[str]:
    """Return the `re.findall` matches of a pattern with at most one group.

    Args:
        text: The searched text.
        pattern: The regular expression.
        count: The required number of matches.

    Returns:
        The matches (group 1 if the pattern has a group, else the whole match).

    """
    rx = re.compile(pattern)
    group = 1 if rx.groups else 0
    m: list[str] = []
    for found in rx.finditer(text):
        g = found.group(group)
        m.append(g if isinstance(g, str) else "")
    if len(m) != count:
        fail(f"pattern {pattern!r}: {len(m)} matches, expected {count}")
    return m


def one(text: str, pattern: str) -> str:
    """Return the unique match of a pattern with at most one group.

    Args:
        text: The searched text.
        pattern: The regular expression.

    Returns:
        The match.

    """
    return grab(text, pattern)[0]


_CEXPR: dict[str, Callable[[Mapping[str, object]], object]] = {}


def cexpr(expr: str, env: Mapping[str, int]) -> int:
    """Evaluate a C integer expression over the non-negative ints of env.

    Only + - * / % and names occur; / is floor division on these operands.

    Args:
        expr: The C expression.
        env: The values of its names.

    Returns:
        The value.

    """
    fn = _CEXPR.get(expr)
    if fn is None:
        if not re.fullmatch(r"[\w\s+*/%()\-]+", expr):
            fail(f"unexpected C expression {expr!r}")
        fn = _CEXPR[expr] = pysubset.compile_expr(
            re.sub(r"(?<![/])/(?![/])", "//", expr)
        )
    value = fn(env)
    if not isinstance(value, int):
        fail(f"C expression {expr!r} is not an integer: {value!r}")
    return value


def kernel_body(src: str) -> str:
    """Return the text of gdn_conv_rule_norm_kernel.

    Args:
        src: gdn.cu.

    Returns:
        The kernel from its name to its closing brace.

    """
    start = src.index("void gdn_conv_rule_norm_kernel\n")
    end = src.index("\n}\n", start)
    return src[start:end]


# ---- symbolic values, rendered as GDN_BA_KSPLIT_TABLE.bend's `ex`
@dataclass(frozen=True, slots=True)
class Zero:
    """The fp32 zero an accumulator starts from."""


@dataclass(frozen=True, slots=True)
class Slot:
    """The unknown contents of a ks_part slot no warp stored."""

    slot: int


@dataclass(frozen=True, slots=True)
class Fma:
    """fmaf(x, w, acc) of x element `x`, weight element `w`, component `comp`."""

    x: int
    w: int
    comp: int
    acc: Value


@dataclass(frozen=True, slots=True)
class Add:
    """The fp32 sum of two values."""

    left: Value
    right: Value


@dataclass(frozen=True, slots=True)
class Bias:
    """A value plus bias element `bias`."""

    value: Value
    bias: int


type Value = Zero | Slot | Fma | Add | Bias


def show(e: Value) -> str:
    """Render a value as the table's `ex`.

    Args:
        e: The value.

    Returns:
        Its text.

    """
    match e:
        case Zero():
            return "0"
        case Fma():
            return f"F({e.x},{e.w},{e.comp},{show(e.acc)})"
        case Add():
            return f"A({show(e.left)},{show(e.right)})"
        case Bias():
            return f"B({show(e.value)},{e.bias})"
        case Slot():
            return f"S{e.slot}"


@dataclass(frozen=True, slots=True)
class Quotes:
    """The quoted kernel constants and expressions."""

    consts: dict[str, int]
    ks_default: int
    warp_e: str
    lane_e: str
    k2_e: str
    j_e: str
    iters: str
    wb_e: str
    wa_e: str
    x_e: str
    fmas: list[tuple[str, ...]]
    off0: int
    wslots: tuple[str, ...]
    wl: tuple[str, ...]
    rslots: tuple[str, ...]
    bias: tuple[str, ...]
    fb: str


def quote_loads(body: str) -> dict[str, str]:
    """Quote the element, row and x indices of ks_load and the FMA loop.

    Args:
        body: The kernel body.

    Returns:
        The expressions by Quotes field.

    """
    # ---- element index (ks_load and the FMA loop: the same expression)
    jx = grab(body, r"const int j = ([^;]+);\n\s+if \(j < k2\)", 2)
    if jx[0] != jx[1]:
        fail(f"ks_load and the FMA loop index elements differently: {jx}")
    return {
        "j_e": jx[0],
        "iters": grab(body, r"for \(int i = 0; i < (GR_KS_ITERS); \+\+i\)", 2)[0],
        "wb_e": one(
            body,
            r"const half2\* wb = \(const half2\*\) "
            r"\(ba_w \+ \(size_t\) ([^;]+) \* k\);",
        ),
        "wa_e": one(
            body,
            r"const half2\* wa = \(const half2\*\) "
            r"\(ba_w \+ \(size_t\) ([^;]+) \* k\);",
        ),
        "x_e": one(
            body, r"if \(s < S\) ks_x\[s\]\[i\] = x2\[\(size_t\) (s \* k2 \+ j)\];"
        ),
    }


def quote_fmas(body: str) -> list[tuple[str, ...]]:
    """Quote the four fmaf lines and check each pairs matching components.

    Args:
        body: The kernel body.

    Returns:
        (accumulator, x component, weight row, weight component) per line.

    """
    grab(body, r"for \(int s = 0; s < (GR_KS_ROWS); \+\+s\)", 5)
    fmas = grab_groups(
        body, r"(p[ab])\[s\] = fmaf\(xf\.([xy]), w([ab])f\.([xy]), p[ab]\[s\]\);", 4
    )
    for acc, xc, wr, wc in fmas:
        if xc != wc or acc[1] != wr:
            fail(f"fmaf pairs x.{xc} with w{wr}.{wc} into {acc}")
    return fmas


def quote(src: str, body: str) -> Quotes:
    """Quote every expression the table replays, in source-check order.

    Args:
        src: gdn.cu.
        body: The kernel body.

    Returns:
        The quotes.

    """
    consts = {
        "GR_KS_ITERS": int(one(src, r"#define GR_KS_ITERS (\d+)\n")),
        "GR_KS_ROWS": int(one(src, r"#define GR_KS_ROWS (\d+)\n")),
        "GR_WARPS": int(one(src, r"#define GR_WARPS (\d+)\n")),
        "THREADS": int(one(body, r"constexpr int THREADS = (\d+);")),
    }
    ks_default = int(one(src, r"#define GR_KS_DEFAULT (\d+)"))
    warp_e = one(body, r"const int warp_id = (t / \d+);")
    lane_e = one(body, r"const int lane_id = (t % \d+);")
    k2_e = grab(body, r"const int k2 = (k / 2);", 3)[0]  # ks_load, KS 0, KS > 0
    loads = quote_loads(body)
    fmas = quote_fmas(body)
    # ---- reduction and slots
    off0 = int(
        one(
            body,
            r"for \(int offset = (\d+); offset > 0; offset >>= 1\)\n"
            r"\s+\{\n\s+pb\[s\] \+= ",
        )
    )
    shfl = grab_groups(
        body,
        r"(p[ab])\[s\] \+= __shfl_down_sync\(0xffffffff, (p[ab])\[s\], offset\);",
        2,
    )
    if any(a != b for a, b in shfl) or [a for a, _ in shfl] != ["pb", "pa"]:
        fail(f"shfl lines {shfl}")
    wslots = grab_groups(
        body,
        r"if \(lane_id == 0\)\n\s+\{\n\s+ks_part\[([^\]]+)\] = pb\[s\];\n"
        r"\s+ks_part\[([^\]]+)\] = pa\[s\];",
        1,
    )[0]
    grab(body, r"bv = 0\.0f;\n\s+av = 0\.0f;", 1)
    wl = grab_groups(
        body,
        r"for \(int w = ([^;]+); w (<|>=) ([^;]+); (\+\+w|--w)\)\n\s+\{\n"
        r"\s+bv \+= ks_part\[",
    )[0]
    rslots = grab_groups(
        body, r"bv \+= ks_part\[([^\]]+)\];\n\s+av \+= ks_part\[([^\]]+)\];", 1
    )[0]
    bias = grab_groups(
        body,
        r"if \(ba_bias\)\n\s+\{\n\s+bv \+= __half2float\(ba_bias\[([^\]]+)\]\);\n"
        r"\s+av \+= __half2float\(ba_bias\[([^\]]+)\]\);",
        1,
    )[0]
    fb = one(src, r"if \((S > [\w ]+ \|\| k / 2 > [\w *]+)\) ks = 0;")
    return Quotes(
        consts=consts,
        ks_default=ks_default,
        warp_e=warp_e,
        lane_e=lane_e,
        k2_e=k2_e,
        j_e=loads["j_e"],
        iters=loads["iters"],
        wb_e=loads["wb_e"],
        wa_e=loads["wa_e"],
        x_e=loads["x_e"],
        fmas=fmas,
        off0=off0,
        wslots=wslots,
        wl=wl,
        rslots=rslots,
        bias=bias,
        fb=fb,
    )


@dataclass(frozen=True, slots=True)
class Event:
    """A block event: its body position, name, the KS modes and blocks it runs in."""

    pos: int
    name: str
    ks: frozenset[int]
    other_only: bool = False


def pos(body: str, pattern: str) -> int:
    """Return the position of the unique match of an event pattern.

    Args:
        body: The kernel body.
        pattern: The regular expression.

    Returns:
        The match's start.

    """
    m = list(re.finditer(pattern, body))
    if len(m) != 1:
        fail(f"event pattern {pattern!r}: {len(m)} matches")
    return m[0].start()


def events(src: str, body: str) -> list[Event]:
    """Return the block events in body order and check the token-loop order.

    Args:
        src: gdn.cu.
        body: The kernel body.

    Returns:
        The events sorted by position.

    """
    gemv0 = pos(
        body, r"if constexpr \(KS == 0\)\n\s+\{\n\s+// One \(row, output\) per warp"
    )
    fma_p = pos(body, r"pb\[s\] = fmaf\(xf\.[xy], wbf\.[xy], pb\[s\]\);\n\s+pb")
    tree_p = pos(body, r"pb\[s\] \+= __shfl_down_sync")
    partw_p = pos(body, r"ks_part\[[^\]]+\] = pb\[s\];")
    read0_p = pos(body, r"bv = m\.ba\[t\];")
    tok_p = pos(body, r"gdn_rule_tokens<RULE_VERIFY, 4>\(")
    bar_p = [
        m.start()
        for m in re.finditer(r"__syncthreads\(\);", body)
        if max(gemv0, partw_p) < m.start() < read0_p
    ]
    ev_all = [
        Event(
            pos(body, r"if constexpr \(KS != 2\) gdn_rule_load_state\("),
            "state",
            frozenset({0, 1}),
        ),
        Event(
            pos(body, r"if constexpr \(KS == 2\) ks_load\(\);"),
            "ks_load",
            frozenset({2}),
        ),
        Event(pos(body, r"gdn_rule_norm_prefetch\(w4, g4"), "norm_ops", ALL_KS),
        Event(pos(body, r"if \(t < 3 \* HD\)"), "conv", ALL_KS),
        Event(
            pos(body, r"if constexpr \(KS == 1\) ks_load\(\);"),
            "ks_load",
            frozenset({1}),
        ),
        Event(
            pos(body, r"if constexpr \(KS == 2\) gdn_rule_load_state\("),
            "state",
            frozenset({2}),
        ),
        Event(gemv0, "gemv", frozenset({0})),
        Event(fma_p, "ks_fma", frozenset({1, 2})),
        Event(tree_p, "ks_tree", frozenset({1, 2})),
        Event(partw_p, "part_w", frozenset({1, 2})),
        *[Event(p, "bar", ALL_KS) for p in bar_p],
        Event(
            pos(
                body,
                r"if \(!owner && t == 0\)\n\s+\{\n\s+__threadfence\(\);\n"
                r"\s+atomicAdd\(conv_sync",
            ),
            "arrive",
            ALL_KS,
            other_only=True,
        ),
        Event(pos(body, r"state_v\[\(size_t\) c \* state_size"), "win_v", ALL_KS),
        Event(read0_p, "part_r", ALL_KS),
    ]
    rt = src[src.index("void gdn_rule_tokens(") :]
    rt = rt[: rt.index("\n}\n")]
    l2, sync1, d2 = (
        rt.find("gdn_rule_l2norm_warp("),
        rt.find("__syncthreads();"),
        rt.find("m.dot2["),
    )
    if not (0 <= l2 < sync1 < d2):
        fail("gdn_rule_tokens: l2 norms, __syncthreads, m.dot2 store not in that order")
    if not (read0_p < tok_p):
        fail("gdn_rule_tokens runs before the b/a read")
    # no other store to the partial area between the partial stores and the token loop
    mid = body[partw_p:tok_p]
    stores = re.findall(r"(?:m\.dot2|ks_part)\[[^\]]+\]\s*=(?!=)", mid)
    if len(stores) != BAR_PARTIAL_STORES:
        fail(
            "stores to m.dot2 / ks_part between the partial stores and the token loop: "
            f"{stores}"
        )
    ev_all.sort(key=lambda e: e.pos)
    return ev_all


def prog(ev_all: list[Event], ks: int, *, own: bool) -> list[str]:
    """Return the event program of a block.

    Args:
        ev_all: The events in body order.
        ks: The KS mode.
        own: Whether the block owns the conv window.

    Returns:
        The event names, then the token loop's own events.

    """
    names = [e.name for e in ev_all if ks in e.ks and not (e.other_only and own)]
    return [*names, "l2", "bar", "dot2_w"]


def lane_tree(vals: list[Value], off0: int) -> Value:
    """Return lane 0's value after the quoted shfl_down reduction.

    Args:
        vals: The 32 lanes' values.
        off0: The first offset.

    Returns:
        Lane 0's sum tree.

    """
    v = list(vals)
    off = off0
    while off > 0:
        v = [
            Add(v[lane], v[lane + off] if lane + off < WARP_LANES else v[lane])
            for lane in range(WARP_LANES)
        ]
        off >>= 1
    return v[0]


@dataclass(frozen=True, slots=True)
class Task:
    """One thread's b/a output: S rows, row trow, b (0) / a (1), k2, h, H, bias."""

    rows: int
    trow: int
    ab: int
    k2: int
    h: int
    heads: int
    has_bias: bool


def lane_accs(
    q: Quotes, t: Task, env: dict[str, int], lane: int
) -> dict[tuple[str, int], Value]:
    """Return one lane's pb / pa accumulators, FMA loop as quoted.

    Args:
        q: The quotes.
        t: The task.
        env: The block's names.
        lane: The thread index t.

    Returns:
        The final pb and pa values by (accumulator, row).

    """
    e = dict(env, t=lane)
    e["warp_id"] = cexpr(q.warp_e, e)
    e["lane_id"] = cexpr(q.lane_e, e)
    rows = q.consts["GR_KS_ROWS"]
    p: dict[tuple[str, int], Value] = {("pb", s): Zero() for s in range(rows)}
    p.update({("pa", s): Zero() for s in range(rows)})
    for i in range(q.consts["GR_KS_ITERS"]):
        j = cexpr(q.j_e, dict(e, i=i))
        if j >= t.k2:
            continue
        for s in range(min(rows, t.rows)):
            x2 = cexpr(q.x_e, dict(e, s=s, j=j, k2=t.k2))
            for acc, comp, wr, _ in q.fmas:
                row = cexpr(q.wb_e if wr == "b" else q.wa_e, e)
                p[acc, s] = Fma(x2, row * t.k2 + j, 0 if comp == "x" else 1, p[acc, s])
    return p


def task(q: Quotes, t: Task) -> Value:
    """Return thread trow's b or a value: FMA loop, reduction, slots and read.

    Args:
        q: The quotes.
        t: The task.

    Returns:
        The value's operation tree.

    """
    env = dict(q.consts, h=t.h, H=t.heads, k=2 * t.k2)
    if cexpr(q.k2_e, env) != t.k2:
        fail("k2")
    # every thread's accumulators, FMA loop as quoted
    part: dict[int, Value] = {}
    for w in range(q.consts["GR_WARPS"]):
        lanes: dict[int, list[Value]] = {0: [], 1: []}
        for lane in range(WARP_LANES):
            p = lane_accs(q, t, env, WARP_LANES * w + lane)
            if t.trow < t.rows:
                lanes[0].append(p["pb", t.trow])
                lanes[1].append(p["pa", t.trow])
        for s in range(q.consts["GR_KS_ROWS"]):
            if s < t.rows and s == t.trow:
                e = dict(env, warp_id=w, s=s)
                for abx in (0, 1):
                    part[cexpr(q.wslots[abx], e)] = lane_tree(lanes[abx], q.off0)
    e = dict(env, t=t.trow)
    start, cmp_, stop, step = q.wl
    w = cexpr(start, e)
    v: Value = Zero()
    while (w < cexpr(stop, e)) if cmp_ == "<" else (w >= cexpr(stop, e)):
        slot = cexpr(q.rslots[t.ab], dict(e, w=w))
        v = Add(v, part.get(slot, Slot(slot)))
        w = w + 1 if step == "++w" else w - 1
    if t.has_bias:
        v = Bias(v, cexpr(q.bias[t.ab], e))
    return v


def index_lines(q: Quotes) -> list[str]:
    """Return the J (element indices) and W (write / read slot) lines.

    Args:
        q: The quotes.

    Returns:
        The lines.

    """
    out: list[str] = []
    for t in range(THREADS_PER_BLOCK):
        e = dict(q.consts, t=t)
        e["warp_id"] = cexpr(q.warp_e, e)
        e["lane_id"] = cexpr(q.lane_e, e)
        js = [cexpr(q.j_e, dict(e, i=i)) for i in range(q.consts[q.iters])]
        if len(js) != ITERS_PER_THREAD:
            fail(f"{len(js)} iterations per thread")
        out.append(f"J {t}:" + ",".join(map(str, js)))
    for slot in range(SLOT_CASES):
        w, ab, s = slot // 16, (slot % 16) // 8, slot % 8
        e = dict(q.consts, warp_id=w, w=w, s=s, t=s)
        wslot = cexpr(q.wslots[ab], e)
        rslot = cexpr(q.rslots[ab], e)
        out.append(f"W {w} {s} {ab}:{wslot},{rslot}")
    return out


def table(q: Quotes, ev_all: list[Event]) -> str:
    """Return the source side of the table.

    Args:
        q: The quotes.
        ev_all: The block events in body order.

    Returns:
        The table text.

    """
    out = index_lines(q)
    for ks in range(KS_MODES):
        for rows, k in FALLBACK_CASES:
            cond = q.fb.replace("||", " or ")
            lhs, rhs = [x.strip() for x in cond.split(" or ")]
            e = dict(q.consts, S=rows, k=k)
            big = cexpr(lhs.split(">")[0], e) > cexpr(lhs.split(">")[1], e) or cexpr(
                rhs.split(">")[0], e
            ) > cexpr(rhs.split(">")[1], e)
            out.append(f"V {ks} {rows} {k}:{0 if big else ks}")
    for ks in range(KS_MODES):
        out.append(f"P {ks} owner: " + " ".join(prog(ev_all, ks, own=True)))
        out.append(f"P {ks} other: " + " ".join(prog(ev_all, ks, own=False)))
    out.append(
        "L " + show(lane_tree([Slot(lane) for lane in range(WARP_LANES)], q.off0))
    )
    out.append(
        "T 2 1 1 520 1 2 1:"
        + show(task(q, Task(rows=2, trow=1, ab=1, k2=520, h=1, heads=2, has_bias=True)))
    )
    out.append(
        "T 1 0 0 3 0 1 0:"
        + show(task(q, Task(rows=1, trow=0, ab=0, k2=3, h=0, heads=1, has_bias=False)))
    )
    return "\n".join(out) + "\n"


def load_source(argv: list[str]) -> str:
    """Read gdn.cu from the command line's package directory, mutated if asked.

    Args:
        argv: The command line, program name first.

    Returns:
        The (mutated) source.

    """
    mutate = None
    if len(argv) >= MUTATE_ARGC and argv[1] == "--mutate":
        mutate = argv[2]
        argv = [argv[0], *argv[3:]]
    if len(argv) != PLAIN_ARGC:
        fail(USAGE)
    src = (Path(argv[1]) / "exllamav3_ext/gdn.cu").read_text()
    if mutate:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate}; known: {', '.join(MUTATIONS)}")
        a, b, n = MUTATIONS[mutate]
        if src.count(a) != n:
            fail(
                f"mutation {mutate} does not apply ({src.count(a)} occurrences, "
                f"expected {n})"
            )
        src = src.replace(a, b)
    return src


def bend_table() -> str:
    """Run the Bend table.

    Returns:
        Its output without IO.print's own final newline.

    """
    bend = source_link.bend()
    bres = source_link.run(
        [bend, TABLE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=BEND_TIMEOUT,
        check=False,
    )
    if bres.returncode != 0:
        fail(f"{bend} {TABLE} exited {bres.returncode}: {bres.stderr.strip()[-500:]}")
    btext = bres.stdout
    if btext.endswith("\n\n"):
        btext = btext[:-1]  # IO.print's own newline after the table's last "\n"
    return btext


def main(argv: list[str]) -> None:
    """Run the source link.

    Args:
        argv: The command line, program name first.

    """
    src = load_source(argv)
    body = kernel_body(src)
    q = quote(src, body)
    ev_all = events(src, body)
    ctext = table(q, ev_all)
    btext = bend_table()
    same = ctext == btext
    sys.stdout.write(
        f"constants {q.consts}, default KS {q.ks_default}, fallback `{q.fb}`\n"
    )
    sys.stdout.write(
        f"table lines: source {len(ctext.splitlines())}, "
        f"Bend {len(btext.splitlines())}; byte-identical: {same}\n"
    )
    if not same:
        pairs = zip(ctext.splitlines(), btext.splitlines(), strict=False)
        for i, (x, y) in enumerate(pairs):
            if x != y:
                sys.stdout.write(
                    f"first difference at line {i + 1}:\n"
                    f"  source: {x[:300]}\n  Bend:   {y[:300]}\n"
                )
                break
        fail("MISMATCH")
    sys.stdout.write("gdn_ba_ksplit_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
