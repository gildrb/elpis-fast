#!/usr/bin/env python3
"""Admit and call the Bend-proven EXL3 tree acceptance leaves.

  python3 exl3_bend_tree_accept.py DIRECTORY     # full admission, exit 0 on success

The directory is the acceptor root shared with the chain acceptor. It holds
libexl3_tree_accept.so (unchanged Bend 2.0.34 emitted C of
bend/EXL3_TREE_ACCEPT.bend plus bend/exl3_tree_accept_glue.c), the canonical
table printed by the Bend reference program bend/EXL3_TREE_ACCEPT_SPEC.bend,
this file and tree_identity.json, beside the four chain acceptor files.
admit() verifies every file hash and the identity digest, loads the library,
renders the complete finite differential table through the loaded leaves
(every legal parent array: its 128 derive bytes, the verdict for each of the
128 match masks, and the chain rules along the lowest-child path) and
requires it to equal the pinned reference table byte for byte, then probes
the domain edges: the edge call must succeed and every call just outside the
domain must raise. Any failure raises ValueError; there is no fallback.

Each accept_tree call packs all 31 cells with one precompiled struct layout
and calls the exported glue once with the GIL held; derive packs 8 cells and
reads 128 bytes. The glue alone validates the domain; this loader checks only
the lengths it needs to choose the layout.

Proved in Bend (bend/exl3_tree_accept_proof.bend, exl3_tree_derive_proof.bend):
the leaves equal the list reference for every input (derive on every legal
parent array). Bridged by this finite differential only: the clang-compiled
emitted C, this ctypes ABI and the glue's validation.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import struct
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

SCHEMA = "elpis-exl3-bend-tree-accept/1"
TABLE_NAME = "exl3_tree_accept_table.txt"
LIBRARY_NAME = "libexl3_tree_accept.so"
LOADER_NAME = "exl3_bend_tree_accept.py"
IDENTITY_NAME = "tree_identity.json"
# The acceptor root holds exactly the chain and the tree acceptor files.
ROOT_FILES = frozenset({
    "libexl3_accept.so",
    "exl3_accept_table.txt",
    "exl3_bend_accept.py",
    "identity.json",
    LIBRARY_NAME,
    TABLE_NAME,
    LOADER_NAME,
    IDENTITY_NAME,
})
TABLE_SHA256 = "2bc7c90b66a77c0fcc8f0cf1231f471e0093c7a660185a563ff0290df74e0718"
IDENTITY_KEYS = frozenset({
    "schema",
    "bend_version",
    "toolchain",
    "sources",
    "leaves",
    "artifacts",
    "table_sha256",
    "identity_sha256",
})
ARTIFACTS = frozenset({LIBRARY_NAME, TABLE_NAME, LOADER_NAME})
ROWS = 8
MAX_STOPS = 4
ID_LIMIT = 1 << 32
BUDGET_LIMIT = (1 << 48) - 1
CELLS = 31
DESC_BYTES = 128

AcceptTree = Callable[
    [Sequence[int], Sequence[int], Sequence[int], Sequence[int], int, int],
    tuple[int, bool, int],
]
Derive = Callable[[Sequence[int]], bytes]


def fail(message: str) -> ValueError:
    return ValueError(f"EXL3 Bend tree acceptance: {message}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(record: dict[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def identity_digest(identity: dict[str, object]) -> str:
    return digest(
        canonical({k: v for k, v in identity.items() if k != "identity_sha256"})
    )


def shapes() -> list[list[int]]:
    """Every legal parent array (row 0 = -1, row r in 0..r-1), lexicographic."""
    return [[-1, *tail] for tail in itertools.product(*[range(r) for r in range(1, ROWS)])]


def glyph(verdict: tuple[int, bool, int]) -> str:
    count, eos, last = verdict
    return chr(64 + 2 * count + eos) + chr(48 + last)


def render(accept_tree: AcceptTree, derive: Derive) -> str:
    """Canonical table of bend/EXL3_TREE_ACCEPT_SPEC.bend, computed by the leaves."""
    lines = ["EXL3_TREE_ACCEPT_V1"]
    verify = [100 + r for r in range(ROWS)]
    for parent in shapes():
        cells = []
        for mask in range(128):
            tokens = [0] + [
                100 + parent[c] if mask >> (c - 1) & 1 else 99 for c in range(1, ROWS)
            ]
            cells.append(glyph(accept_tree(verify, tokens, parent, (), 262144, 0)))
        full = [0] + [100 + parent[c] for c in range(1, ROWS)]
        rules = [
            glyph(accept_tree(verify, full, parent, (), budget, checkpoint))
            for budget in range(1, 10)
            for checkpoint in range(ROWS)
        ]
        rules += [
            glyph(accept_tree(verify, full, parent, (100 + r,), 262144, 0))
            for r in range(ROWS)
        ]
        rules += [
            glyph(accept_tree(verify, full, parent, (107, 106, 105, 104)[:n], 262144, 0))
            for n in range(MAX_STOPS + 1)
        ]
        lines.append(
            f"S {''.join(map(str, parent[1:]))} {derive(parent).hex()} "
            f"{''.join(cells)} {''.join(rules)}"
        )
    lines.append("END_EXL3_TREE_ACCEPT")
    return "\n".join(lines) + "\n"


def layout(stops: int) -> struct.Struct:
    """Cells for `stops` stop ids; unused stop slots are zero."""
    return struct.Struct(f"=3q{stops}q{8 * (MAX_STOPS - stops)}x24q")


# One precompiled cell layout per stop count.
LAYOUTS = tuple(layout(stops) for stops in range(MAX_STOPS + 1))
PARENTS = struct.Struct("=8q")
# The glue validates every cell; its error codes, in its checking order.
REJECTIONS = {
    -1: "parents are not a legal 8-row tree (row 0 = -1, row r in 0..r-1)",
    -2: f"stop id count exceeds {MAX_STOPS}",
    -3: f"budget outside 1..{BUDGET_LIMIT}",
    -4: "checkpoint outside 0..7",
    -5: "stop, verify or token id outside 0..2^32-1",
    -6: "leaf failed or returned a verdict outside count 1..8, last 0..7",
}


class TreeAcceptor:
    """One loaded, admitted library. Reuses its buffers: call from one thread only."""

    def __init__(self, library: Path) -> None:
        # PyDLL keeps the GIL across the calls: each leaf is a few hundred
        # nanoseconds of pure C, so releasing the GIL would only add latency.
        handle = ctypes.PyDLL(str(library), mode=ctypes.RTLD_LOCAL)
        accept = handle.elpis_exl3_tree_accept
        accept.restype = ctypes.c_int32
        derive = handle.elpis_exl3_tree_derive
        derive.restype = ctypes.c_int32
        # No argtypes: the arguments are always this object's ctypes arrays,
        # which ctypes passes by address without a converter.
        self._handle = handle
        self._accept = accept
        self._derive = derive
        self._cells = (ctypes.c_int64 * CELLS)()
        self._parents = (ctypes.c_int64 * ROWS)()
        self._desc = (ctypes.c_uint8 * DESC_BYTES)()

    def accept_tree(
        self,
        verify_ids: Sequence[int],
        tokens: Sequence[int],
        parents: Sequence[int],
        stop_ids: Sequence[int],
        budget: int,
        checkpoint: int,
    ) -> tuple[int, bool, int]:
        """Return (count, eos, last): commit verify_ids[path[i]] for i < count,
        where path is the maximal matching path and last = path[count - 1]."""
        stops = len(stop_ids)
        if len(verify_ids) != ROWS or len(tokens) != ROWS or len(parents) != ROWS:
            raise fail(f"verify ids, tokens and parents must each hold {ROWS} rows")
        if stops > MAX_STOPS:
            raise fail(f"{stops} stop ids exceed {MAX_STOPS}")
        # One pack writes every cell. struct rejects non-integers and values
        # outside int64 (ctypes stores would wrap them silently); the glue
        # rejects every other out-of-domain value.
        try:
            LAYOUTS[stops].pack_into(
                self._cells, 0, budget, checkpoint, stops,
                *stop_ids, *verify_ids, *tokens, *parents,
            )
        except (struct.error, TypeError, OverflowError) as error:
            raise fail(f"argument is not a 64-bit integer: {error}") from None
        result = self._accept(self._cells)
        if result < 2:
            raise fail(REJECTIONS.get(result, f"leaf returned code {result}"))
        return (result >> 1) & 15, bool(result & 1), result >> 5

    def derive(self, parents: Sequence[int]) -> bytes:
        """The 128 TreeDesc bytes (DESIGN.md §2.2) of a legal parent array."""
        if len(parents) != ROWS:
            raise fail(f"parents must hold {ROWS} rows")
        try:
            PARENTS.pack_into(self._parents, 0, *parents)
        except (struct.error, TypeError, OverflowError) as error:
            raise fail(f"parent is not a 64-bit integer: {error}") from None
        result = self._derive(self._parents, self._desc)
        if result != 0:
            raise fail(REJECTIONS.get(result, f"leaf returned code {result}"))
        return bytes(self._desc)


def domain(tree: TreeAcceptor) -> None:
    """The loaded calls accept the domain's edges and reject just past them."""
    top = ID_LIMIT - 1
    chain = [-1, 0, 1, 2, 3, 4, 5, 6]
    if tree.accept_tree([top] * 8, [top] * 8, chain, (0, 1, 2, 3), BUDGET_LIMIT, 7) != (
        7,
        False,
        6,
    ):
        raise fail("leaf rejects or misjudges the domain's upper edges")
    rows = [0] * 8
    outside: list[tuple[object, ...]] = [
        (rows[:7], rows, chain, (), 1, 0),
        (rows, rows[:7], chain, (), 1, 0),
        (rows, rows, chain[:7], (), 1, 0),
        (rows, rows, chain, (0,) * 5, 1, 0),
        (rows, rows, chain, (), 0, 0),
        (rows, rows, chain, (), BUDGET_LIMIT + 1, 0),
        (rows, rows, chain, (), 1 << 63, 0),
        (rows, rows, chain, (), 1, -1),
        (rows, rows, chain, (), 1, 8),
        (rows, rows, chain, (ID_LIMIT,), 1, 0),
        (rows, rows, chain, (-1,), 1, 0),
        ([0] * 7 + [ID_LIMIT], rows, chain, (), 1, 0),
        (rows, [ID_LIMIT] + [0] * 7, chain, (), 1, 0),
        (rows, [0] * 7 + [-1], chain, (), 1, 0),
        (rows, [0] * 7 + [1 << 64], chain, (), 1, 0),
        (rows, [0] * 7 + [1.0], chain, (), 1, 0),
        (rows, rows, [0, 0, 1, 2, 3, 4, 5, 6], (), 1, 0),
        (rows, rows, [-1, 0, 1, 2, 3, 4, 5, 7], (), 1, 0),
        (rows, rows, [-1, 0, 2, 2, 3, 4, 5, 6], (), 1, 0),
        (rows, rows, [-1, 0, 1, 2, 3, -1, 5, 6], (), 1, 0),
    ]
    for arguments in outside:
        try:
            tree.accept_tree(*arguments)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise fail(f"leaf accepted out-of-domain call {arguments!r}")
    for parents in ([0, 0, 1, 2, 3, 4, 5, 6], [-1, 1, 1, 2, 3, 4, 5, 6],
                    [-1, 0, 1, 2, 3, 4, 5, 7], [-1, 0, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3, 4, 5, 6.0]):
        try:
            tree.derive(parents)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise fail(f"derive accepted out-of-domain parents {parents!r}")


def read_identity(directory: Path) -> dict[str, object]:
    raw = json.loads((directory / IDENTITY_NAME).read_bytes())
    if not isinstance(raw, dict):
        raise fail("identity is not an object")
    identity: dict[str, object] = {str(key): value for key, value in raw.items()}
    if frozenset(identity) != IDENTITY_KEYS:
        raise fail(f"identity keys {sorted(identity)} differ from the schema")
    if identity["schema"] != SCHEMA:
        raise fail("identity schema differs")
    if identity["table_sha256"] != TABLE_SHA256:
        raise fail("identity table differs from the pinned reference table")
    if identity["identity_sha256"] != identity_digest(identity):
        raise fail("identity digest does not match its contents")
    return identity


def verify_files(directory: Path, identity: dict[str, object]) -> None:
    artifacts = identity["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACTS:
        raise fail("artifact set differs")
    present = {path.name for path in directory.iterdir()}
    if present != ROOT_FILES:
        raise fail(f"directory holds {sorted(present)}, not exactly the admitted files")
    for name, expected in artifacts.items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise fail(f"{name} is not a regular file")
        if digest(path.read_bytes()) != expected:
            raise fail(f"{name} hash differs from identity")
    if digest((directory / TABLE_NAME).read_bytes()) != TABLE_SHA256:
        raise fail("retained table differs from the pinned reference table")
    if digest(Path(__file__).read_bytes()) != artifacts[LOADER_NAME]:
        raise fail("running loader differs from the admitted loader")


def admit(directory: str | Path) -> TreeAcceptor:
    """Verify, load and differentially admit the leaves; raise ValueError on any failure."""
    root = Path(directory).resolve(strict=True)
    identity = read_identity(root)
    verify_files(root, identity)
    tree = TreeAcceptor(root / LIBRARY_NAME)
    rendered = render(tree.accept_tree, tree.derive).encode("ascii")
    if rendered != (root / TABLE_NAME).read_bytes() or digest(rendered) != TABLE_SHA256:
        raise fail("loaded leaves disagree with the Bend reference table")
    domain(tree)
    return tree


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        raise SystemExit("usage: exl3_bend_tree_accept.py DIRECTORY")
    admit(arguments[0])
    print(f"EXL3 Bend tree acceptance admitted: table {TABLE_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
