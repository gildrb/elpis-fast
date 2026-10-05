#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Byte-for-byte differential of the DFlash2 draft mask model against the engine.

Bend side: bend/DRAFT_MASK_TABLE.bend (bend/draft_mask.bend, proven against the z-lab
reference by bend/draft_mask_proof.bend) compiled with the pinned toolchain and run.
Engine side, from the patched engine tree (exl3 0001-0003 + 0005, then the exl3-ext
series), every file hash-checked against its pin, evaluated by bend/pysubset.py:
  - architecture/dflash.py: dflash2_kernel_window (0005), executed as written
    (ast-extracted) with EXL3_DFLASH2_WINDOW unset;
  - architecture/dflash2.py: the draft forward's causal flag
    (`params["causal"] = False`);
  - modules/attention_fn/triton_paged.py: _normalize_window and gqa_geometry executed
    as written; the GQA split kernel's window start / split / tile lines
    (_gqa_live_split, _gqa_pass) and _gqa_step's mask lines quoted verbatim (each
    asserted present) and evaluated per scalar key (tl.* -> Python builtins); the 3005
    dispatch lines that keep every row of a non-causal launch (the draft) in pass A
    with _gqa_live_split's (lo, s_len) are asserted present.
For every table row the set of keys the kernel reaches its softmax with must equal the
Bend bits, under the served split geometry (bsz 1, 8 kv heads, 2 h-blocks, grid_y 10,
82 SMs) and two others. `--mutate NAME` perturbs one engine line and must FAIL.

Usage: python3 -I -B draft_mask_diff.py --engine <tree> [--table <table.txt>]
       [--mutate NAME]
  --engine <tree>: OUT/patched of bend/engine_trees.py (the full series).
"""

from __future__ import annotations

import ast
import hashlib
import operator
import re
import resource
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

HERE = Path(__file__).resolve().parent
# post-images of the full patched tree (patches/exl3/series + patches/exl3-ext/series)
PINS = {
    "architecture/dflash.py": (
        "1440f55c4cbf7e1367ce4bc41cb633c673d44e137b966ee5cf2ce320bd062d48"
    ),
    "architecture/dflash2.py": (
        "3eb3fda82aa99652eed42998b0d79a6c2add74d094bf453650b29c74e90dbd69"
    ),
    "modules/attention_fn/triton_paged.py": (
        "792a461592bfdadbd3b0f903b9098951e0ce5964c9e860d354e6d5906b331290"
    ),
}
TRITON_PAGED = "modules/attention_fn/triton_paged.py"
# The window start line (the start_round_up mutation's anchor)
WINDOW_START = (
    "        lo = (tl.maximum(0, total_k_len - q_len - WINDOW_LEFT) // BLOCK_N) * "
    "BLOCK_N"
)
KERNEL_START = [
    "    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len",
    "    lo = total_k_len * 0",
    "    if WINDOW_LEFT >= 0:",
    WINDOW_START,
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
# 3005 dispatch: rows start in pass A; only a causal full-context launch (not the
# draft) reassigns s_a or runs pass B; pass A takes (lo, s_a) from _gqa_live_split,
# which _gqa_pass reads as (lo, s_len)
KERNEL_DISPATCH = [
    "    b0 = vr0 & False",
    (
        "    lo, s_a = _gqa_live_split(total_k_len, q_len, WINDOW_LEFT, SPLITS, "
        "MIN_SPLIT, BLOCK_N)"
    ),
    "    if CAUSAL:",
    "        if WINDOW_LEFT < 0:",
    (
        "              batch, kv_head, bh, split, lo, s_a, total_k_len, "
        "num_pages_per_seq, store_acc,"
    ),
    "              q0, vr0 & ~b0, qa0, ~b0, q1, vr1 & ~b1, qa1, ~b1,",
    (
        "              batch, kv_head, bh, split, lo, s_len, total_k_len, "
        "num_pages_per_seq, store_acc,"
    ),
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
    "right_zero": (
        "architecture/dflash.py",
        "return (window - 1, 0 if causal else window - 1)",
        "return (window - 1, 0 if causal else 0)",
    ),
    # g7n's window convention (keys q - W .. q, one extra)
    "left_off_by_one": (
        "architecture/dflash.py",
        "return (window - 1, 0 if causal else window - 1)",
        "return (window, 0 if causal else window - 1)",
    ),
    # window start rounded up instead of down
    "start_round_up": (
        TRITON_PAGED,
        WINDOW_START,
        (
            "        lo = tl.cdiv(tl.maximum(0, total_k_len - q_len - WINDOW_LEFT), "
            "BLOCK_N) * BLOCK_N"
        ),
    ),
}
# (bsz, n_kv_heads, h_blocks, grid_y, sm_count): served draft, then a 3- and 1-split
GEOMETRIES = [
    (1, 8, 2, 10, 82),
    (1, 8, 2, 10, 12),
    (1, 8, 2, 10, 1),
]
TABLE_LINE = re.compile(r"cfg=(\w+) W=(\d+) S=(\d+) L=(\d+) r=(\d+) BN=(\d+) ([01]+)\Z")
CAUSAL_OF = {"none": None, "true": True, "false": False}
SHOWN_MISMATCHES = 5
WINDOW_PAIR = 2
GEOMETRY_FIELDS = 3
# The extracted dflash.py code reads only os.environ and re.fullmatch; the window
# variable is unset
OS_STUB = SimpleNamespace(environ={})
RE_STUB = SimpleNamespace(fullmatch=re.fullmatch)

type Keys = Callable[..., object]


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"FAIL: {msg}"
    raise SystemExit(text)


def sha(b: bytes) -> str:
    """Return the SHA-256 of `b`.

    Returns:
        The hex digest.

    """
    return hashlib.sha256(b).hexdigest()


def definition_name(node: ast.stmt) -> str | None:
    """Return the name a top-level `def` or single-name assignment defines.

    Returns:
        The name, or None for any other statement.

    """
    if isinstance(node, ast.FunctionDef):
        return node.name
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def extract(src: str, names: list[str], env: dict[str, object]) -> dict[str, object]:
    """Run the named top-level definitions of `src`, as written, in a copy of `env`.

    Returns:
        The namespace with the definitions.

    """
    tree = ast.parse(src)
    ns = dict(env)
    found: set[str] = set()
    for node in tree.body:
        name = definition_name(node)
        if name is None or name not in names:
            continue
        code = ast.get_source_segment(src, node)
        if code is None:
            fail(f"no source segment for {name}")
        pysubset.exec_block(code, ns)
        found.add(name)
    if found != set(names):
        fail(f"missing definitions {set(names) - found}")
    return ns


def need(lines: set[str], quoted: list[str]) -> None:
    """Fail unless every quoted kernel line occurs verbatim."""
    for q in quoted:
        if q not in lines:
            fail(f"kernel line not found verbatim: {q!r}")


def scalar(line: str) -> str:
    """Return the kernel line rewritten for one query row (tl.* -> Python).

    Returns:
        The rewritten line.

    """
    s = line.strip()
    for a, b in (
        ("tl.load(cache_seqlens + batch)", "S"),
        ("tl.maximum", "maximum"),
        ("tl.minimum", "minimum"),
        ("tl.cdiv", "cdiv"),
        ("tl.arange", "arange"),
        ("offs_n[None, :]", "offs_n"),
        ("q_abs[:, None]", "q_abs"),
        ("valid_row[:, None]", "True"),
    ):
        s = s.replace(a, b)
    return s


def as_assignment(s: str) -> str:
    """Return _gqa_live_split's return line as the kernel's s_len assignment.

    _gqa_live_split returns (lo, s_len); every other line is returned unchanged.

    Returns:
        The line.

    """
    if s.startswith(LIVE_SPLIT_RETURN):
        return "s_len = " + s[len(LIVE_SPLIT_RETURN) :]
    return s


def cdiv(a: int, b: int) -> int:
    """Return a / b rounded up.

    Returns:
        The quotient.

    """
    return -(-a // b)


type Lane = Callable[[int, int], int]


class Mask:
    """A boolean Triton vector, as a bitmask over its `n` lanes (bit i: lane i)."""

    __slots__ = ("bits", "n")

    def __init__(self, bits: int, n: int) -> None:
        """Hold the lane bits."""
        self.bits = bits
        self.n = n

    def __and__(self, o: object) -> Mask:
        """Return each lane & o (a mask, or a scalar bool / int).

        Returns:
            The conjunction.

        """
        if isinstance(o, Mask):
            return Mask(self.bits & o.bits, self.n)
        if not isinstance(o, int):
            fail(f"mask operand {o!r}")
        # a set lane is 1: 1 & o keeps the lane iff o is odd
        return Mask(self.bits if o & 1 else 0, self.n)

    __rand__ = __and__

    def lane(self, i: int) -> bool:
        """Return lane i.

        Returns:
            Whether the lane is set.

        """
        return bool(self.bits >> i & 1)


def mask_of(values: Iterable[bool]) -> Mask:
    """Return the mask with lane i set iff values[i].

    Returns:
        The mask.

    """
    bits = n = 0
    for i, x in enumerate(values):
        bits |= int(x) << i
        n = i + 1
    return Mask(bits, n)


def clamp(k: int, n: int) -> int:
    """Return k clamped to 0 .. n.

    Returns:
        The clamped value.

    """
    return max(0, min(k, n))


class Lanes:
    """An integer Triton vector (one value per lane: a key, or a split program).

    Its operators work lane by lane, as Triton's do on blocks. A tl.arange
    vector keeps its consecutive values as a range, so its comparisons with a
    scalar are computed per run of lanes; the result is the same lane by lane.
    """

    __slots__ = ("v",)

    def __init__(self, v: tuple[int, ...] | range) -> None:
        """Hold the lane values."""
        self.v = v

    def map(self, o: object, fn: Lane) -> Lanes:
        """Return fn(lane, o) lane by lane (o a scalar or a vector).

        Returns:
            The results.

        """
        if isinstance(o, Lanes):
            return Lanes(tuple(starmap(fn, zip(self.v, o.v, strict=True))))
        if not isinstance(o, int):
            fail(f"lane operand {o!r}")
        return Lanes(tuple(fn(x, o) for x in self.v))

    def consecutive(self, o: object) -> tuple[range, int] | None:
        """Return (lanes, o) if the lanes are consecutive ints and o is an int.

        Returns:
            The lane range and the scalar, or None.

        """
        v = self.v
        if isinstance(v, range) and v.step == 1 and isinstance(o, int):
            return v, o
        return None

    def __add__(self, o: object) -> Lanes:
        """Return each lane + o.

        Returns:
            The sums.

        """
        run = self.consecutive(o)
        if run is not None:
            v, k = run
            return Lanes(range(v.start + k, v.stop + k))
        return self.map(o, operator.add)

    __radd__ = __add__

    def __mul__(self, o: object) -> Lanes:
        """Return each lane * o.

        Returns:
            The products.

        """
        return self.map(o, operator.mul)

    __rmul__ = __mul__

    def __lt__(self, o: object) -> Mask:
        """Return each lane < o.

        Returns:
            The comparisons.

        """
        run = self.consecutive(o)
        if run is None:
            return mask_of(x < y for x, y in self.pairs(o))
        v, k = run
        # lanes v.start .. k - 1 are below k
        return Mask((1 << clamp(k - v.start, len(v))) - 1, len(v))

    def __le__(self, o: object) -> Mask:
        """Return each lane <= o.

        Returns:
            The comparisons.

        """
        run = self.consecutive(o)
        if run is None:
            return mask_of(x <= y for x, y in self.pairs(o))
        v, k = run
        return Mask((1 << clamp(k + 1 - v.start, len(v))) - 1, len(v))

    def __ge__(self, o: object) -> Mask:
        """Return each lane >= o.

        Returns:
            The comparisons.

        """
        run = self.consecutive(o)
        if run is None:
            return mask_of(x >= y for x, y in self.pairs(o))
        v, k = run
        below = (1 << clamp(k - v.start, len(v))) - 1
        return Mask(((1 << len(v)) - 1) & ~below, len(v))

    def pairs(self, o: object) -> list[tuple[int, int]]:
        """Return (lane, o lane or o) for each lane.

        Returns:
            The operand pairs.

        """
        if isinstance(o, Lanes):
            return list(zip(self.v, o.v, strict=True))
        if not isinstance(o, int):
            fail(f"lane operand {o!r}")
        return [(x, o) for x in self.v]

    def where(self, mask: Mask) -> list[int]:
        """Return the lanes whose mask lane is set.

        Returns:
            The selected values.

        """
        return [x for i, x in enumerate(self.v) if mask.lane(i)]


def arange(start: int, end: int) -> Lanes:
    """Return tl.arange(start, end).

    Returns:
        The lane vector start, start + 1, ..., end - 1.

    """
    return Lanes(range(start, end))


def minimum(a: int | Lanes, b: int | Lanes) -> int | Lanes:
    """Return tl.minimum: min of scalars, lane by lane on vectors.

    Returns:
        The minimum.

    """
    if isinstance(a, Lanes):
        return a.map(b, min)
    if isinstance(b, Lanes):
        return b.map(a, min)
    return min(a, b)


def maximum(a: int | Lanes, b: int | Lanes) -> int | Lanes:
    """Return tl.maximum: max of scalars, lane by lane on vectors.

    Returns:
        The maximum.

    """
    if isinstance(a, Lanes):
        return a.map(b, max)
    if isinstance(b, Lanes):
        return b.map(a, max)
    return max(a, b)


def programs(n_start: Lanes, n_end: Lanes) -> list[tuple[int, int]]:
    """Return the (n_start, n_end) of the split programs that run a tile.

    A program whose range(n_start, n_end, BLOCK_N) is empty runs no tile.

    Returns:
        The bounds of the other programs, in split order.

    """
    return [(s, e) for s, e in zip(n_start.v, n_end.v, strict=True) if s < e]


def split_invariant(lines: list[str]) -> int:
    """Return how many leading _gqa_live_split / _gqa_pass lines ignore `split`.

    They give every split program the same values and are evaluated once per row;
    the rest (n_start, n_end) must be plain statements.

    Returns:
        The length of the split-invariant prefix.

    """
    first = next(
        (i for i, q in enumerate(lines) if re.search(r"\bsplit\b", q)), len(lines)
    )
    if any(q.startswith(" ") for q in lines[first:]):
        fail(f"split-dependent kernel lines are not plain statements: {lines[first:]}")
    return first


def build_kernel(tp_src: str, swap: tuple[str, str] | None = None) -> tuple[Keys, str]:
    """Build the kernel function of one query row from the quoted lines.

    The split grid and each tile's keys are Triton-style vectors (Lanes); the
    split-invariant lines run once per row.

    Returns:
        The function keys(S, L, r, BLOCK_N, SPLITS, MIN_SPLIT, CAUSAL, WINDOW_LEFT,
        WINDOW_RIGHT) and its source.

    """
    lines = set(tp_src.splitlines())
    need(
        lines,
        KERNEL_START + KERNEL_TILE + [KERNEL_QABS] + KERNEL_MASK + KERNEL_DISPATCH,
    )

    # A kernel mutation perturbs the quoted line after the verbatim check, so it
    # reaches the evaluation. The one indented line is the body of the preceding
    # `if WINDOW_LEFT >= 0:`.
    start = [
        ("    " if q.startswith("        ") else "")
        + as_assignment(scalar(swap[1]) if swap and q == swap[0] else scalar(q))
        for q in KERNEL_START
    ]
    hoisted = split_invariant(start)
    body = [
        (
            "def keys(S, L, r, BLOCK_N, SPLITS, MIN_SPLIT, CAUSAL, WINDOW_LEFT, "
            "WINDOW_RIGHT):"
        ),
        "    q_len = kv_append_len = L",
        "    row_q = r",
        "    out = set()",
        *("    " + q for q in start[:hoisted]),
        "    " + scalar(KERNEL_QABS),
        "    split = arange(0, SPLITS)",
        *("    " + q for q in start[hoisted:]),
        "    for n_start, n_end in programs(n_start, n_end):",
        "        " + scalar(KERNEL_TILE[0]),
        "            " + scalar(KERNEL_TILE[1]),
    ]
    for q in KERNEL_MASK:
        pad = "                " if q.startswith("        ") else "            "
        body.append(pad + scalar(q))
    body.extend(("            out.update(offs_n.where(valid))", "    return out"))
    ns: dict[str, object] = {
        "cdiv": cdiv,
        "arange": arange,
        "minimum": minimum,
        "maximum": maximum,
        "programs": programs,
    }
    pysubset.exec_block("\n".join(body), ns)
    keys = ns["keys"]
    if not callable(keys):
        fail("the generated kernel defines no keys()")
    return keys, "\n".join(body)


def bend_table(path: str | None) -> str:
    """Return the Bend table: a saved file, or the compiled table's output.

    Returns:
        The table text.

    """
    if path:
        return Path(path).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "table"
        r = source_link.run(
            [
                source_link.bend(),
                str(HERE / "DRAFT_MASK_TABLE.bend"),
                "-o",
                str(exe),
            ],
            capture_output=True,
            text=True,
            cwd=HERE,
            check=False,
        )
        if r.returncode:
            fail("Bend table compile failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        # The Bend runtime reserves its heap up front: lift an address-space cap
        # (ulimit -v); the table process inherits it
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        return source_link.run(
            [str(exe), "--gpu", "off"],
            capture_output=True,
            text=True,
            check=True,
            cpu_heavy=False,
        ).stdout


@dataclass(frozen=True)
class Args:
    """The command line."""

    engine: Path
    table: str | None
    mutate: str | None


def parse_args(argv: list[str]) -> Args:
    """Parse the command line.

    Returns:
        The arguments.

    """
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
    return Args(engine=engine, table=table, mutate=mutate)


def load_sources(
    engine: Path, mutate: str | None
) -> tuple[dict[str, str], tuple[str, str] | None]:
    """Read the pinned engine files and apply the mutation.

    Returns:
        The sources by path, and the kernel line swap of a kernel mutation.

    """
    src: dict[str, str] = {}
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
        if rel == TRITON_PAGED:
            swap = (old, new)
        else:
            src[rel] = src[rel].replace(old, new)
    if DRAFT_CAUSAL not in src["architecture/dflash2.py"].splitlines():
        fail("dflash2.py no longer runs the draft forward with causal=False")
    return src, swap


def call(ns: dict[str, object], name: str, *args: object) -> object:
    """Call the extracted function `name` of `ns`.

    Returns:
        Its result.

    """
    fn = ns[name]
    if not callable(fn):
        fail(f"{name} is not a function")
    return fn(*args)


def int_tuple(value: object, size: int, what: str) -> tuple[int, ...]:
    """Return `value` as a tuple of `size` ints (bools count as ints).

    Returns:
        The tuple.

    """
    if not (
        isinstance(value, tuple)
        and len(value) == size
        and all(isinstance(v, int) for v in value)
    ):
        fail(f"{what} returned {value!r}")
    return tuple(v for v in value if isinstance(v, int))


@dataclass(frozen=True)
class Engine:
    """The executed engine functions and their results so far.

    The functions are pure over the constant module globals they read, so each
    distinct call is evaluated once and its result reused.
    """

    fa: dict[str, object]
    tp: dict[str, object]
    keys: Keys
    memo: dict[tuple[object, ...], object]

    def once(self, key: tuple[object, ...], compute: Callable[[], object]) -> object:
        """Return the result of the call `key`, computed by `compute` the first time.

        Returns:
            The result.

        """
        if key not in self.memo:
            self.memo[key] = compute()
        return self.memo[key]


@dataclass(frozen=True)
class Query:
    """One table row's key query: S, L, r, BLOCK_N and the kernel window."""

    s: int
    length: int
    r: int
    bn: int
    win: tuple[int, ...]


def kernel_bits(engine: Engine, query: Query, geo: tuple[int, ...]) -> str:
    """Return the keys the kernel reaches under split geometry `geo`, as bits.

    Returns:
        One "0" / "1" per key position 0 .. S + L + 1.

    """
    use, splits, min_split = int_tuple(
        engine.once(
            ("gqa_geometry", *geo, query.bn),
            lambda: call(engine.tp, "gqa_geometry", *geo, query.bn),
        ),
        GEOMETRY_FIELDS,
        "gqa_geometry",
    )
    if not use:
        fail(f"geometry {geo} does not select the GQA kernel")
    args = (query.s, query.length, query.r, query.bn, splits, min_split, *query.win)
    got = engine.once(
        ("keys", *args),
        lambda: engine.keys(
            query.s,
            query.length,
            query.r,
            query.bn,
            splits,
            min_split,
            CAUSAL=False,
            WINDOW_LEFT=query.win[0],
            WINDOW_RIGHT=query.win[1],
        ),
    )
    if not isinstance(got, set):
        fail(f"keys() returned {got!r}")
    return "".join("1" if n in got else "0" for n in range(query.s + query.length + 2))


def row_mismatches(engine: Engine, line: str) -> list[tuple[object, ...]]:
    """Compare one table line with the engine under every geometry.

    Returns:
        The mismatches.

    """
    m = TABLE_LINE.match(line)
    if m is None:
        fail(f"malformed table line {line!r}")
    cfg, bits = m.group(1), m.group(7)
    window, s, length, r, bn = (int(g) for g in m.groups()[1:6])
    kernel_window = engine.once(
        ("dflash2_kernel_window", window, cfg),
        lambda: call(
            engine.fa,
            "dflash2_kernel_window",
            window,
            CAUSAL_OF[cfg],
            ["sliding_attention"] * 5,
        ),
    )
    win = int_tuple(
        engine.once(
            ("_normalize_window", kernel_window),
            lambda: call(engine.tp, "_normalize_window", kernel_window),
        ),
        WINDOW_PAIR,
        "_normalize_window",
    )
    query = Query(s=s, length=length, r=r, bn=bn, win=win)
    bad: list[tuple[object, ...]] = []
    for geo in GEOMETRIES:
        mine = kernel_bits(engine, query, geo)
        if mine != bits:
            bad.append((line.split(" ")[:6], geo, mine, bits))
    return bad


def main(argv: list[str]) -> None:
    """Run the differential."""
    a = parse_args(argv)
    src, swap = load_sources(a.engine, a.mutate)
    fa = extract(
        src["architecture/dflash.py"],
        ["DFLASH2_WINDOW_ENV", "DFLASH2_WINDOW_MAX", "dflash2_kernel_window"],
        {"os": OS_STUB, "re": RE_STUB},
    )
    tp = extract(
        src[TRITON_PAGED],
        ["_normalize_window", "GQA_MAX_NSUB", "gqa_geometry"],
        {"_gqa_enable": True, "_gqa_ctas_per_sm": 2, "_gqa_min_split": 0},
    )
    keys, _ = build_kernel(src[TRITON_PAGED], swap)
    engine = Engine(fa=fa, tp=tp, keys=keys, memo={})

    text = bend_table(a.table)
    rows = 0
    bad: list[tuple[object, ...]] = []
    for line in text.splitlines():
        bad += row_mismatches(engine, line)
        rows += 1
    expect_rows = 3 * 2 * 9 * 21 * sum(range(1, 9))
    if rows != expect_rows:
        fail(f"table has {rows} rows, expected {expect_rows}")
    sys.stdout.write(
        f"engine {a.engine}; table sha256 {sha(text.encode())}; {rows} rows x "
        f"{len(GEOMETRIES)} split geometries\n"
    )
    mutation = f" (mutation {a.mutate})" if a.mutate else ""
    if bad:
        for b in bad[:SHOWN_MISMATCHES]:
            sys.stdout.write(f"  mismatch {b}\n")
        sys.stdout.write(f"FAIL: {len(bad)} mismatching rows{mutation}\n")
        sys.exit(1)
    detected = f" (mutation {a.mutate} NOT detected)" if a.mutate else ""
    sys.stdout.write(f"PASS: engine mask == Bend model bit for bit{detected}\n")


if __name__ == "__main__":
    main(sys.argv)
