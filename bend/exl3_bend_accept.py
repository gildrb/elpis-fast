#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Admit and call the Bend-proven EXL3 greedy acceptance leaf.

  python3 exl3_bend_accept.py DIRECTORY     # full admission, exit 0 on success

The directory holds libexl3_accept.so (unchanged Bend 2.0.35 emitted C of
bend/EXL3_ACCEPT.bend plus bend/exl3_accept_glue.c), the canonical table
printed by the Bend reference program bend/EXL3_ACCEPT_SPEC.bend, this file and
identity.json, beside the four tree acceptor files (exl3_bend_tree_accept.py).
admit() verifies every file hash and the identity digest, loads
the library, renders the complete finite differential table through the
loaded leaf and requires it to equal the pinned reference table byte for
byte, then probes the domain edges: the edge call must succeed and every call
just outside the domain must raise. Any failure raises ValueError; there is
no fallback implementation.

Each call packs all 23 cells with one precompiled struct layout and calls the
exported glue once with the GIL held. The glue alone validates the domain;
this loader checks only the lengths it needs to choose the layout.

Proved in Bend (bend/exl3_accept_proof.bend): the leaf equals the list
reference for every input. Bridged by this finite differential only: the
clang-compiled emitted C, this ctypes ABI and the glue's validation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

SCHEMA = "elpis-exl3-bend-accept/1"
TABLE_NAME = "exl3_accept_table.txt"
LIBRARY_NAME = "libexl3_accept.so"
LOADER_NAME = "exl3_bend_accept.py"
IDENTITY_NAME = "identity.json"
TABLE_SHA256 = "7c84bd88fbd6ad01b0872e2516e24f44029bfc3f48c50978d6ca86ad93085d63"
IDENTITY_KEYS = frozenset({
    "schema",
    "bend_version",
    "toolchain",
    "sources",
    "leaf",
    "artifacts",
    "table_sha256",
    "identity_sha256",
})
ARTIFACTS = frozenset({LIBRARY_NAME, TABLE_NAME, LOADER_NAME})
# The acceptor root holds exactly the chain and the tree acceptor files.
ROOT_FILES = ARTIFACTS | {
    IDENTITY_NAME,
    "libexl3_tree_accept.so",
    "exl3_tree_accept_table.txt",
    "exl3_bend_tree_accept.py",
    "tree_identity.json",
}
MAX_PROPOSALS = 7
MAX_STOPS = 4
ID_LIMIT = 1 << 32
BUDGET_LIMIT = (1 << 48) - 1
CELLS = 23
BINARY = (17, 248046)
QUATERNARY = (0, 1, 248046, 4294967295)
# Glue results below this are rejection codes; verdicts are 2 * count + eos.
FIRST_VERDICT = 2
PREFIX = "EXL3 Bend acceptance: "

Accept = Callable[
    [Sequence[int], Sequence[int], Sequence[int], int, int], tuple[int, bool]
]


def reason(message: str) -> str:
    """Prefix a failure message with this loader's name.

    Args:
        message: The failure description.

    Returns:
        The full ValueError text.

    """
    return f"{PREFIX}{message}"


def digest(data: bytes) -> str:
    """Hash bytes.

    Args:
        data: The bytes to hash.

    Returns:
        The lowercase hex SHA-256 of `data`.

    """
    return hashlib.sha256(data).hexdigest()


def canonical(record: dict[str, object]) -> bytes:
    """Serialize a record as canonical JSON.

    Args:
        record: The JSON object.

    Returns:
        Sorted-key, compact UTF-8 JSON bytes.

    """
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def identity_digest(identity: dict[str, object]) -> str:
    """Hash an identity record without its own digest field.

    Args:
        identity: The identity record.

    Returns:
        The SHA-256 of the canonical record minus `identity_sha256`.

    """
    return digest(
        canonical({k: v for k, v in identity.items() if k != "identity_sha256"})
    )


def glyph(count: int, *, eos: bool) -> str:
    """Encode one verdict as a table character.

    Args:
        count: The accepted count.
        eos: Whether the verdict ends the job.

    Returns:
        The single-character cell.

    """
    return chr((96 if eos else 64) + count)


def render(accept: Accept) -> str:
    """Render the canonical table of bend/EXL3_ACCEPT_SPEC.bend through `accept`.

    Args:
        accept: The acceptance call to tabulate.

    Returns:
        The complete table text.

    """
    lines = ["EXL3_GREEDY_ACCEPT_V1"]
    for k in range(1, MAX_PROPOSALS + 1):
        patterns = [
            (
                [BINARY[(x >> (2 * i)) & 1] for i in range(k + 1)],
                [BINARY[(x >> (2 * i + 1)) & 1] for i in range(k)],
            )
            for x in range(1 << (2 * k + 1))
        ]
        for budget in [*range(1, k + 3), 262144]:
            for checkpoint in range(k + 1):
                cells: list[str] = []
                for targets, proposals in patterns:
                    count, eos = accept(
                        targets, proposals, (248046,), budget, checkpoint
                    )
                    cells.append(glyph(count, eos=eos))
                lines.append(f"A {k} {budget} {checkpoint} {''.join(cells)}")
    for mask in range(16):
        stops = tuple(QUATERNARY[j] for j in range(4) if mask >> j & 1)
        cells = []
        for x in range(64):
            targets = [QUATERNARY[x & 3], QUATERNARY[(x >> 4) & 3]]
            proposals = [QUATERNARY[(x >> 2) & 3]]
            count, eos = accept(targets, proposals, stops, 3, 0)
            cells.append(glyph(count, eos=eos))
        lines.append(f"B {mask} {''.join(cells)}")
    lines.append("END_EXL3_GREEDY_ACCEPT")
    return "\n".join(lines) + "\n"


def layout(k: int, stops: int) -> struct.Struct:
    """Build the cell layout for k proposals and `stops` stop ids.

    Unused registers are zero.

    Args:
        k: The proposal count.
        stops: The stop id count.

    Returns:
        The precompiled struct layout.

    """
    return struct.Struct(
        f"=4q{stops}q{8 * (MAX_STOPS - stops)}x{k + 1}q"
        f"{8 * (MAX_PROPOSALS - k)}x{k}q{8 * (MAX_PROPOSALS - k)}x"
    )


# One precompiled cell layout per (k, stop count); index [k][stops].
LAYOUTS = tuple(
    tuple(layout(k, stops) for stops in range(MAX_STOPS + 1)) if k else ()
    for k in range(MAX_PROPOSALS + 1)
)
# The glue validates every cell; its error codes, in its checking order.
REJECTIONS = {
    -1: f"proposal count outside 1..{MAX_PROPOSALS}",
    -2: f"stop id count exceeds {MAX_STOPS}",
    -3: f"budget outside 1..{BUDGET_LIMIT}",
    -4: "checkpoint outside 0..k",
    -5: "stop, verify or proposal id outside 0..2^32-1",
    -6: "leaf failed or returned a verdict outside 1..k+1",
}


class IdentityTypeError(TypeError, ValueError):
    """Identity record of the wrong type; `except ValueError` still catches it."""


class Acceptor:
    """One loaded, admitted leaf. Reuses one buffer: call from one thread only."""

    def __init__(self, library: Path) -> None:
        """Load the leaf library.

        Args:
            library: Path of libexl3_accept.so.

        """
        # PyDLL keeps the GIL across the call: the leaf is a few nanoseconds of
        # pure C, so releasing and reacquiring the GIL would only add latency.
        handle = ctypes.PyDLL(str(library), mode=ctypes.RTLD_LOCAL)
        function = handle.elpis_exl3_accept
        # No argtypes: the only argument is always self._cells, an int64[23]
        # ctypes array, which ctypes passes by address without a converter.
        function.restype = ctypes.c_int32
        self._handle = handle
        self._function = function
        self._cells = (ctypes.c_int64 * CELLS)()

    @property
    def function(self) -> Callable[..., object]:
        """The exported leaf function (read-only)."""
        return self._function

    @property
    def cells(self) -> ctypes.Array[ctypes.c_int64]:
        """The reused cell buffer the leaf reads (read-only)."""
        return self._cells

    def accept(
        self,
        verify_ids: Sequence[int],
        proposals: Sequence[int],
        stop_ids: Sequence[int],
        budget: int,
        checkpoint: int,
    ) -> tuple[int, bool]:
        """Decide one verify round: commit verify_ids[:count]; eos ends the job.

        Args:
            verify_ids: The k + 1 target ids.
            proposals: The k proposed ids.
            stop_ids: Up to MAX_STOPS stop ids.
            budget: The remaining token budget.
            checkpoint: The forced stop position, 0 for none.

        Returns:
            The (count, eos) verdict.

        Raises:
            ValueError: If any argument is outside the leaf's domain.
            TypeError: If the leaf returns a non-integer.

        """
        k = len(proposals)
        stops = len(stop_ids)
        if not 1 <= k <= MAX_PROPOSALS:
            msg = f"proposal count {k} outside 1..{MAX_PROPOSALS}"
            raise ValueError(reason(msg))
        if len(verify_ids) != k + 1:
            msg = f"{len(verify_ids)} verify ids for {k} proposals"
            raise ValueError(reason(msg))
        if stops > MAX_STOPS:
            msg = f"{stops} stop ids exceed {MAX_STOPS}"
            raise ValueError(reason(msg))
        # One pack writes every cell. struct rejects non-integers and values
        # outside int64 (ctypes stores would wrap them silently); the glue
        # rejects every other out-of-domain budget, checkpoint or id.
        try:
            LAYOUTS[k][stops].pack_into(
                self._cells,
                0,
                k,
                budget,
                checkpoint,
                stops,
                *stop_ids,
                *verify_ids,
                *proposals,
            )
        except (struct.error, TypeError, OverflowError) as error:
            msg = f"argument is not a 64-bit integer: {error}"
            raise ValueError(reason(msg)) from None
        # restype c_int32: ctypes always returns a Python int.
        result = self._function(self._cells)
        if not isinstance(result, int):
            msg = f"leaf returned {type(result).__name__}, not int"
            raise TypeError(reason(msg))
        if result < FIRST_VERDICT:
            msg = REJECTIONS.get(result, f"leaf returned code {result}")
            raise ValueError(reason(msg))
        return result >> 1, bool(result & 1)


def read_identity(directory: Path) -> dict[str, object]:
    """Read and check the identity record.

    Args:
        directory: The acceptor root.

    Returns:
        The identity record.

    Raises:
        ValueError: If the record is malformed or differs from the pins.
        IdentityTypeError: If the record is not a JSON object.

    """
    raw = json.loads((directory / IDENTITY_NAME).read_bytes())
    if not isinstance(raw, dict):
        msg = "identity is not an object"
        raise IdentityTypeError(reason(msg))
    identity: dict[str, object] = {str(key): value for key, value in raw.items()}
    if frozenset(identity) != IDENTITY_KEYS:
        msg = f"identity keys {sorted(identity)} differ from the schema"
        raise ValueError(reason(msg))
    if identity["schema"] != SCHEMA:
        msg = "identity schema differs"
        raise ValueError(reason(msg))
    if identity["table_sha256"] != TABLE_SHA256:
        msg = "identity table differs from the pinned reference table"
        raise ValueError(reason(msg))
    if identity["identity_sha256"] != identity_digest(identity):
        msg = "identity digest does not match its contents"
        raise ValueError(reason(msg))
    return identity


def verify_files(directory: Path, identity: dict[str, object]) -> None:
    """Check the directory holds exactly the admitted files with their hashes.

    Args:
        directory: The acceptor root.
        identity: The checked identity record.

    Raises:
        ValueError: If any file is missing, extra, or differs.

    """
    artifacts = identity["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACTS:
        msg = "artifact set differs"
        raise ValueError(reason(msg))
    present = {path.name for path in directory.iterdir()}
    if present != ROOT_FILES:
        msg = f"directory holds {sorted(present)}, not exactly the admitted files"
        raise ValueError(reason(msg))
    for name, expected in artifacts.items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            msg = f"{name} is not a regular file"
            raise ValueError(reason(msg))
        if digest(path.read_bytes()) != expected:
            msg = f"{name} hash differs from identity"
            raise ValueError(reason(msg))
    if digest((directory / TABLE_NAME).read_bytes()) != TABLE_SHA256:
        msg = "retained table differs from the pinned reference table"
        raise ValueError(reason(msg))
    if digest(Path(__file__).read_bytes()) != artifacts[LOADER_NAME]:
        msg = "running loader differs from the admitted loader"
        raise ValueError(reason(msg))


def domain(accept: Callable[..., tuple[int, bool]]) -> None:
    """Check the loaded call accepts the domain's edges and rejects just past them.

    Args:
        accept: The loaded acceptance call.

    Raises:
        ValueError: If the edge call fails or an outside call succeeds.

    """
    top = ID_LIMIT - 1
    if accept([top] * 8, [top] * 7, (0, 1, 2, 3), BUDGET_LIMIT, 7) != (7, False):
        msg = "leaf rejects or misjudges the domain's upper edges"
        raise ValueError(reason(msg))
    ids, window = [0] * 8, [0] * 7
    outside: list[tuple[object, ...]] = [
        ([0], [], (), 1, 0),
        ([0] * 9, [0] * 8, (), 1, 0),
        ([0] * 7, window, (), 1, 0),
        (ids, window, (0,) * 5, 1, 0),
        (ids, window, (), 0, 0),
        (ids, window, (), BUDGET_LIMIT + 1, 0),
        (ids, window, (), 1 << 63, 0),
        (ids, window, (), 1, -1),
        (ids, window, (), 1, 8),
        (ids, window, (ID_LIMIT,), 1, 0),
        (ids, window, (-1,), 1, 0),
        ([0] * 7 + [ID_LIMIT], window, (), 1, 0),
        ([0] * 7 + [1 << 64], window, (), 1, 0),
        (ids, [0] * 6 + [-1], (), 1, 0),
        (ids, [0] * 6 + [1.0], (), 1, 0),
    ]
    for arguments in outside:
        try:
            accept(*arguments)
        except ValueError:
            continue
        msg = f"leaf accepted out-of-domain call {arguments!r}"
        raise ValueError(reason(msg))


def admit(directory: str | Path) -> Acceptor:
    """Verify, load and differentially admit the leaf.

    Args:
        directory: The acceptor root.

    Returns:
        The admitted acceptor.

    Raises:
        ValueError: On any admission failure.

    """
    root = Path(directory).resolve(strict=True)
    identity = read_identity(root)
    verify_files(root, identity)
    acceptor = Acceptor(root / LIBRARY_NAME)
    rendered = render(acceptor.accept).encode("ascii")
    if rendered != (root / TABLE_NAME).read_bytes() or digest(rendered) != TABLE_SHA256:
        msg = "loaded leaf disagrees with the Bend reference table"
        raise ValueError(reason(msg))
    domain(acceptor.accept)
    return acceptor


def main(arguments: list[str]) -> int:
    """Admit the acceptor root named on the command line.

    Args:
        arguments: The command-line arguments after the program name.

    Returns:
        The exit status.

    Raises:
        SystemExit: On a usage error.

    """
    if len(arguments) != 1:
        msg = "usage: exl3_bend_accept.py DIRECTORY"
        raise SystemExit(msg)
    admit(arguments[0])
    sys.stdout.write(f"EXL3 Bend acceptance admitted: table {TABLE_SHA256}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
