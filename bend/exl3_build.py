#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
r"""Build the EXL3 Bend acceptor root on the host (CPU only).

  nix develop --offline --no-write-lock-file -c \
    python3 bend/exl3_build.py --output build/bend-exl3

Uses the flake-pinned bend 2.0.35 and clang 19.1.7 from the dev shell PATH.
The root holds two artifacts: the chain acceptor (bend/EXL3_ACCEPT.bend,
libexl3_accept.so, identity.json) and the tree acceptor
(bend/EXL3_TREE_ACCEPT.bend, libexl3_tree_accept.so, tree_identity.json).
Steps, each fail-closed: the proof gates (bend/exl3_accept_proof.bend and
bend/exl3_tree_accept_gate.bend print exactly bend's pass verdict, VERDICT); C emission
of each production program and its reference checker; both compiled and run,
their tables byte-identical and equal to the loader's pinned reference table;
admission of the emitted scalar leaves (unique signature and output arity,
acyclic call closure, scalar-only tokens, runtime polling/word ABI
established); the UNCHANGED emitted C plus its glue compiled into each
library; the identities; and each loader's complete ctypes differential
admission of the resulting root. The output directory must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
BEND_VERSION = "bend 2.0.35\n"
CLANG_VERSION = "clang version 19.1.7"
PROOFS = ("bend/exl3_accept_proof.bend", "bend/exl3_tree_accept_gate.bend")
# bend 2.0.35 bend2/main.ts cli_verdict: PASS plus the --verdict hint, on
# stdout. It prints PASS only when every def checks and none relies on @unsafe
# or foreign code, imports included.
VERDICT = "ALL PROOFS CHECK\nUse --verdict for mathematical validity.\n"
SOURCES = (
    "bend/exl3_accept.bend",
    "bend/exl3_accept_spec.bend",
    "bend/exl3_accept_laws.bend",
    "bend/exl3_accept_proof.bend",
    "bend/EXL3_ACCEPT.bend",
    "bend/EXL3_ACCEPT_SPEC.bend",
    "bend/exl3_accept_glue.c",
    "bend/exl3_bend_accept.py",
    "bend/exl3_tree_accept.bend",
    "bend/exl3_tree_accept_spec.bend",
    "bend/exl3_tree_accept_laws.bend",
    "bend/exl3_tree_accept_proof.bend",
    "bend/exl3_tree_derive_proof.bend",
    "bend/exl3_tree_path_proof.bend",
    "bend/exl3_tree_accept_gate.bend",
    "bend/exl3_tree_table.bend",
    "bend/EXL3_TREE_ACCEPT.bend",
    "bend/EXL3_TREE_ACCEPT_SPEC.bend",
    "bend/exl3_tree_accept_glue.c",
    "bend/exl3_bend_tree_accept.py",
    "bend/exl3_build.py",
)
BEND_SOURCES = tuple(name for name in SOURCES if name.endswith(".bend"))

PROGRAM_FLAGS = ("-std=c11", "-O2")
# The library exports only elpis_exl3_accept: section GC drops the unreachable
# Bend runtime (IO loop, thread pool, the 1 MiB effect table) that the
# emitted program carries for its own main, and --as-needed its libraries.
LIBRARY_FLAGS = (
    "-std=c11",
    "-O2",
    "-fPIC",
    "-shared",
    "-fvisibility=hidden",
    "-ffunction-sections",
    "-fdata-sections",
    "-Wl,--gc-sections",
    "-Wl,--as-needed",
)
LINK_FLAGS = ("-lpthread", "-lm")
# The nix cc wrapper otherwise records build-shell paths as RUNPATH.
LIBRARY_ENVIRONMENT = {"NIX_DONT_SET_RPATH_x86_64_unknown_linux_gnu": "1"}

RUNTIME_ABI = (
    "#define WL_SPIN     for (;;) { if (err_spun(e.mem, &wpoll)) { return 0; }",
    "#define err_seen(H)    (DEVICE && a32_load(a32_at(H, H_ERROR_CODE)) != 0)",
    "#define err_spun(H, n) ((++*(n) & 4095) == 0 && err_seen(H))",
    "#define U32_BIN(a, o, b) ((u64)((u32)(a) o (u32)(b)))",
    "#define DEVICE  0",
    "#define FAR static __attribute__((noinline))",
    "#define TAB_AT(T, S, I) T[S < I ? S : I]",
    "#define WL_AGAIN(F) continue",
    "#define CONSTV  static const",
    "#define BANGS   0",
    "typedef u64 Term;",
    "#define NAT_IMM   ((1ull << 48) - 1)",
    "typedef struct {\n  DEV u64* mem;\n  DEV u64* alc;\n} Env;",
)
SIGNATURE = re.compile(
    r"^(?:INLINE|FAR) Term (spin_[0-9]+)\(Env e, THR Term\* o"
    r"((?:, (?:u32|Term) r[0-9]+)*)\) \{$",
    re.MULTILINE,
)
TOKEN = re.compile(
    r"\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:ull)?|==|!=|>=|<=|[-+*|&=<>;,()\[\]{}])"
)
# u64 is a scalar type name only: bend 2.0.34 emits a U32 register copied into
# a Term (u64) slot as the explicit widening cast ((u64)x), which bend 2.0.29
# left implicit (same C conversion).
KEYWORDS = frozenset(
    {
        "INLINE",
        "FAR",
        "Term",
        "Env",
        "THR",
        "e",
        "o",
        "u32",
        "u64",
        "wpoll",
        "WL_SPIN",
        "WL_AGAIN",
        "U32_BIN",
        "TAB_AT",
    }
    | {"if", "else", "return", "break"}
)
LOCAL = re.compile(r"_[A-Za-z0-9_]*_(?:0|[1-9][0-9]*)|r(?:0|[1-9][0-9]*)")
SPIN = re.compile(r"spin_(?:0|[1-9][0-9]*)")
# Constant lookup tables the compiler emits for small Nat functions: a read
# TAB_AT(TAB_n, index, bound) clamps the index to bound, which must lie inside
# the table's static constant u64 initializer.
TABLE = re.compile(r"TAB_(?:0|[1-9][0-9]*)")
TABLE_READ = re.compile(r"TAB_AT\((TAB_(?:0|[1-9][0-9]*)), [^,()]+, ([0-9]+)\)")
UNIT = (
    "#define main {original}\n"
    '#include "{program}.c"\n'
    "#undef main\n"
    "{bindings}"
    '#include "{glue}"\n'
)
OUTPUT = re.compile(r"\bo\[([0-9]+)\] =")

# Argument kinds and output count of one admitted leaf.
type LeafAbi = tuple[tuple[str, ...], int]
# Emitted scalar functions: name -> (argument kinds, body).
type FunctionTable = dict[str, tuple[tuple[str, ...], str]]


@dataclass(frozen=True)
class Artifact:
    """One acceptor: its programs, glue, loader, library, table and identity."""

    program: str
    reference: str
    production: str
    checker: str
    glue: str
    loader: str
    library: str
    table: str
    identity: str
    schema: str
    # The admitted leaves: macro -> (argument kinds, output count).
    leaves: dict[str, LeafAbi]


@dataclass(frozen=True)
class Toolchain:
    """The resolved bend and clang executables and bend's environment."""

    bend: Path
    clang: Path
    bend_environment: dict[str, str]


ARTIFACTS = (
    Artifact(
        program="exl3_accept",
        reference="exl3_accept_spec",
        production="bend/EXL3_ACCEPT.bend",
        checker="bend/EXL3_ACCEPT_SPEC.bend",
        glue="bend/exl3_accept_glue.c",
        loader="bend/exl3_bend_accept.py",
        library="libexl3_accept.so",
        table="exl3_accept_table.txt",
        identity="identity.json",
        schema="elpis-exl3-bend-accept/1",
        leaves={"ELPIS_EXL3_LEAF": (("Term",) * 4 + ("u32",) * 19, 2)},
    ),
    Artifact(
        program="exl3_tree_accept",
        reference="exl3_tree_accept_spec",
        production="bend/EXL3_TREE_ACCEPT.bend",
        checker="bend/EXL3_TREE_ACCEPT_SPEC.bend",
        glue="bend/exl3_tree_accept_glue.c",
        loader="bend/exl3_bend_tree_accept.py",
        library="libexl3_tree_accept.so",
        table="exl3_tree_accept_table.txt",
        identity="tree_identity.json",
        schema="elpis-exl3-bend-tree-accept/1",
        leaves={
            "ELPIS_EXL3_TREE_LEAF": (
                ("Term",) * 3 + ("u32",) * 19 + ("Term",) * 7,
                3,
            ),
            "ELPIS_EXL3_TREE_DERIVE": (("Term",) * 7, 128),
        },
    ),
)


class BuildError(SystemExit):
    """Exit the build with an ``exl3_build:`` prefixed message."""

    def __init__(self, message: str) -> None:
        """Prefix ``message`` with the program name.

        Args:
            message: The failure description.

        """
        super().__init__(f"exl3_build: {message}")


def digest(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``data``.

    Args:
        data: The bytes to hash.

    Returns:
        The hexadecimal digest.

    """
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    """Run ``command`` and return its standard output.

    Args:
        command: The argv; its first element is a resolved executable path.
        cwd: The working directory.
        environment: The complete process environment.

    Returns:
        The captured standard output.

    Raises:
        BuildError: The command exited with a nonzero status.

    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned clang 19.1.7 or a just-built table binary, no shell
        command, cwd=cwd, env=environment, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        message = f"{command} exited {result.returncode}: {result.stderr.strip()}"
        raise BuildError(message)
    return result.stdout


def resolve(name: str) -> Path:
    """Return the path of the executable ``name`` on PATH.

    Args:
        name: The executable name.

    Returns:
        The path ``shutil.which`` finds.

    Raises:
        BuildError: The executable is not on PATH.

    """
    found = shutil.which(name)
    if found is None:
        message = f"{name} is not on PATH (run inside the pinned nix dev shell)"
        raise BuildError(message)
    return Path(found)


def compile_environment(clang: Path, extra: dict[str, str]) -> dict[str, str]:
    """Return the compiler environment: clang's directory as PATH plus ``extra``.

    Args:
        clang: The resolved clang executable.
        extra: Additional environment variables.

    Returns:
        The complete compiler environment.

    """
    return {"PATH": str(clang.parent), **extra}


def load_loader(path: str) -> ModuleType:
    """Import the acceptance loader at the repository path ``path``.

    Args:
        path: The loader path relative to the repository root.

    Returns:
        The executed loader module.

    Raises:
        BuildError: The loader cannot be loaded.

    """
    spec = importlib.util.spec_from_file_location(Path(path).stem, REPO / path)
    if spec is None or spec.loader is None:
        message = "cannot load the acceptance loader"
        raise BuildError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(text: str) -> FunctionTable:
    """Return the emitted scalar functions of ``text``.

    Args:
        text: The emitted C program.

    Returns:
        Each function's name mapped to its argument kinds and body.

    Raises:
        BuildError: A function is incomplete or emitted twice.

    """
    found: FunctionTable = {}
    for match in SIGNATURE.finditer(text):
        closing = re.search(r"^}", text[match.end() :], re.MULTILINE)
        if closing is None:
            message = "incomplete emitted scalar function"
            raise BuildError(message)
        arguments = tuple(re.findall(r", (u32|Term) r[0-9]+", match.group(2)))
        if match.group(1) in found:
            message = "duplicate emitted scalar function"
            raise BuildError(message)
        body = text[match.start() : match.end() + closing.end()]
        found[match.group(1)] = (arguments, body)
    return found


def callees(name: str, body: str) -> Iterator[str]:
    """Admit the tokens of ``body`` in order, yielding each scalar callee.

    Args:
        name: The function name; it is not yielded for self-references.
        body: The function's emitted C.

    Yields:
        Each called scalar function other than ``name``, at its position.

    Raises:
        BuildError: The body holds unadmitted text or an unadmitted token.

    """
    position = 0
    while position < len(body):
        match = TOKEN.match(body, position)
        if match is None:
            if body[position:].strip():
                message = (
                    f"unadmitted text in {name}: {body[position : position + 40]!r}"
                )
                raise BuildError(message)
            return
        # Group 1 always participates; the slice is its text, typed str.
        token = body[match.start(1) : match.end(1)]
        position = match.end()
        if not (token[0].isalpha() or token[0] == "_"):
            continue
        if SPIN.fullmatch(token):
            if token != name:
                yield token
            continue
        if TABLE.fullmatch(token):
            continue
        if token not in KEYWORDS and LOCAL.fullmatch(token) is None:
            message = f"unadmitted token {token!r} in {name}"
            raise BuildError(message)


def table_reads(text: str, name: str, body: str) -> dict[str, str]:
    """Return the constant tables that ``body`` reads, each read in bounds.

    Args:
        text: The emitted C program.
        name: The function name.
        body: The function's emitted C.

    Returns:
        Each read table's name mapped to its initializer.

    Raises:
        BuildError: A read is out of bounds or not admitted.

    """
    tables: dict[str, str] = {}
    for read in TABLE_READ.finditer(body):
        table_name, bound = read.group(1), int(read.group(2))
        found = re.findall(
            rf"^CONSTV u64 {table_name}\[\] = "
            r"\{ ((?:[0-9]+ull, )*[0-9]+ull) \};$",
            text,
            re.MULTILINE,
        )
        if len(found) != 1 or len(found[0].split(", ")) <= bound:
            message = f"table read {read.group(0)!r} in {name} is not in bounds"
            raise BuildError(message)
        tables[table_name] = found[0]
    if len(TABLE_READ.findall(body)) != body.count("TAB_AT("):
        message = f"unadmitted table read in {name}"
        raise BuildError(message)
    return tables


@dataclass
class Admission:
    """The acyclic scalar-only call closure admitted so far."""

    text: str
    table: FunctionTable
    admitted: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)

    def visit(self, name: str) -> None:
        """Admit ``name`` and, depth first, its callees.

        Args:
            name: The scalar function to admit.

        Raises:
            BuildError: The call is recursive or to an unknown function.

        """
        if name in self.active or name not in self.table:
            message = f"recursive or unknown scalar callee {name}"
            raise BuildError(message)
        if name in self.admitted:
            return
        self.active.add(name)
        body = self.table[name][1]
        for callee in callees(name, body):
            self.visit(callee)
        self.tables.update(table_reads(self.text, name, body))
        self.active.remove(name)
        self.admitted[name] = digest(body.encode("utf-8"))


def admit_leaf(
    text: str, arguments: tuple[str, ...], outputs: int
) -> dict[str, object]:
    """Admit the unique leaf with this ABI and its acyclic scalar-only call closure.

    Args:
        text: The emitted C program.
        arguments: The leaf's argument kinds.
        outputs: The leaf's output count.

    Returns:
        The leaf's name, its closure's function digests and any read tables.

    Raises:
        BuildError: The runtime ABI or the leaf is not established uniquely.

    """
    for line in RUNTIME_ABI:
        if text.count(line) != 1:
            message = f"emitted runtime ABI not established: {line!r}"
            raise BuildError(message)
    table = functions(text)
    leaves = [
        name
        for name, (kinds, body) in table.items()
        if kinds == arguments
        and {int(k) for k in OUTPUT.findall(body)} == set(range(outputs))
    ]
    if len(leaves) != 1:
        message = f"expected one emitted leaf with the {arguments} ABI, found {leaves}"
        raise BuildError(message)
    admission = Admission(text, table)
    admission.visit(leaves[0])
    leaf: dict[str, object] = {
        "name": leaves[0],
        "functions": dict(sorted(admission.admitted.items())),
    }
    if admission.tables:
        leaf["tables"] = dict(sorted(admission.tables.items()))
    return leaf


def emit_table(artifact: Artifact, work: Path, toolchain: Toolchain) -> bytes:
    """Emit, compile and run the production and reference programs.

    Args:
        artifact: The acceptor.
        work: The scratch directory.
        toolchain: The resolved toolchain.

    Returns:
        The reference program's table.

    Raises:
        BuildError: The two programs' tables differ.

    """
    program = artifact.program
    reference = artifact.reference
    for name, source in (
        (program, artifact.production),
        (reference, artifact.checker),
    ):
        command = [
            str(toolchain.bend),
            str(REPO / source),
            "-o",
            str(work / f"{name}.c"),
        ]
        run(command, REPO, toolchain.bend_environment)
    clang = toolchain.clang
    tables: dict[str, str] = {}
    for name in (program, reference):
        run(
            [str(clang), *PROGRAM_FLAGS, f"{name}.c", *LINK_FLAGS, "-o", name],
            work,
            compile_environment(clang, {}),
        )
        tables[name] = run([str(work / name), "--gpu", "off"], work, {})
    if tables[program] != tables[reference]:
        message = f"{program} table differs from the reference program"
        raise BuildError(message)
    return tables[reference].encode("ascii")


def link_library(
    artifact: Artifact,
    work: Path,
    clang: Path,
    leaves: dict[str, dict[str, object]],
) -> str:
    """Compile the UNCHANGED emitted C plus its glue into the library.

    Args:
        artifact: The acceptor.
        work: The scratch directory holding the emitted C.
        clang: The resolved clang executable.
        leaves: The admitted leaves by macro.

    Returns:
        The translation unit's file name.

    """
    program = artifact.program
    glue = Path(artifact.glue).name
    shutil.copyfile(REPO / artifact.glue, work / glue)
    unit = f"{program}_unit.c"
    (work / unit).write_text(
        UNIT.format(
            original=f"elpis_{program}_original_main",
            program=program,
            bindings="".join(
                f"#define {macro} {leaf['name']}\n" for macro, leaf in leaves.items()
            ),
            glue=glue,
        ),
        encoding="utf-8",
    )
    run(
        [str(clang), *LIBRARY_FLAGS, unit, *LINK_FLAGS, "-o", artifact.library],
        work,
        compile_environment(clang, LIBRARY_ENVIRONMENT),
    )
    return unit


def build(
    artifact: Artifact, work: Path, staging: Path, toolchain: Toolchain
) -> ModuleType:
    """Emit, compile, compare, admit and stage one artifact and its identity.

    Args:
        artifact: The acceptor.
        work: The scratch directory.
        staging: The directory of the root being built.
        toolchain: The resolved toolchain.

    Returns:
        The artifact's loader module.

    Raises:
        BuildError: The loader pins no table or the table differs from it.

    """
    program = artifact.program
    reference = artifact.reference
    loader = load_loader(artifact.loader)
    reference_sha256 = loader.TABLE_SHA256
    if not isinstance(reference_sha256, str):
        message = f"{artifact.loader} does not pin a reference table"
        raise BuildError(message)
    table = emit_table(artifact, work, toolchain)
    if digest(table) != reference_sha256:
        message = f"{reference} table differs from the pinned table"
        raise BuildError(message)
    emitted = (work / f"{program}.c").read_text(encoding="utf-8")
    leaves = {
        macro: admit_leaf(emitted, kinds, outputs)
        for macro, (kinds, outputs) in artifact.leaves.items()
    }
    unit = link_library(artifact, work, toolchain.clang, leaves)
    library = artifact.library
    shutil.copyfile(work / library, staging / library)
    (staging / artifact.table).write_bytes(table)
    loader_name = Path(artifact.loader).name
    shutil.copyfile(REPO / artifact.loader, staging / loader_name)
    bend = toolchain.bend
    clang = toolchain.clang
    identity: dict[str, object] = {
        "schema": artifact.schema,
        "bend_version": BEND_VERSION.strip(),
        "toolchain": {
            "bend": str(bend.resolve()),
            "bend_sha256": digest(bend.resolve().read_bytes()),
            "clang": str(clang.resolve()),
            "clang_sha256": digest(clang.resolve().read_bytes()),
            "clang_version": CLANG_VERSION,
            "program_flags": [*PROGRAM_FLAGS, *LINK_FLAGS],
            "library_flags": [*LIBRARY_FLAGS, *LINK_FLAGS],
            "library_environment": LIBRARY_ENVIRONMENT,
            "emitted_sha256": {
                name: digest((work / name).read_bytes())
                for name in (f"{program}.c", f"{reference}.c", unit)
            },
        },
        "sources": {name: digest((REPO / name).read_bytes()) for name in SOURCES},
        "artifacts": {
            name: digest((staging / name).read_bytes())
            for name in sorted((library, artifact.table, loader_name))
        },
        "table_sha256": reference_sha256,
    }
    if program == "exl3_accept":
        identity["leaf"] = leaves["ELPIS_EXL3_LEAF"]
    else:
        identity["leaves"] = dict(sorted(leaves.items()))
    identity["identity_sha256"] = digest(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    (staging / artifact.identity).write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return loader


def check_toolchain() -> Toolchain:
    """Resolve bend and clang and require their pinned versions.

    Returns:
        The resolved toolchain.

    Raises:
        BuildError: A version differs from its pin.

    """
    bend = resolve("bend")
    clang = resolve("clang")
    bend_environment = {**os.environ, "BEND_NO_TELEMETRY": "1"}
    if run([str(bend), "version"], REPO, bend_environment) != BEND_VERSION:
        message = "requires exactly bend 2.0.35"
        raise BuildError(message)
    clang_version = run([str(clang), "--version"], REPO, compile_environment(clang, {}))
    if not clang_version.startswith(CLANG_VERSION + "\n"):
        message = "requires exactly clang 19.1.7"
        raise BuildError(message)
    return Toolchain(bend, clang, bend_environment)


def check_proofs(toolchain: Toolchain) -> None:
    """Require hole-free Bend sources and exactly the pass verdict of each gate.

    Args:
        toolchain: The resolved toolchain.

    Raises:
        BuildError: A source is unsafe or open, or a gate does not pass.

    """
    for name in BEND_SOURCES:
        text = (REPO / name).read_text(encoding="utf-8")
        if (
            "@unsafe" in text
            or "?TODO" in text
            or re.search(r"def [A-Za-z0-9_.]+\?\(", text)
        ):
            message = f"{name} contains an unsafe definition or an open hole"
            raise BuildError(message)
    for proof in PROOFS:
        command = [str(toolchain.bend), str(REPO / proof)]
        if run(command, REPO, toolchain.bend_environment) != VERDICT:
            message = f"proof gate {proof} did not report exactly {VERDICT!r}"
            raise BuildError(message)


def main(arguments: list[str]) -> int:
    """Build the acceptor root at ``--output`` and print its file digests.

    Args:
        arguments: The command-line arguments.

    Returns:
        The exit status, 0.

    Raises:
        BuildError: The output already exists.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args(arguments).output.resolve()
    if output.exists():
        message = f"{output} already exists"
        raise BuildError(message)
    toolchain = check_toolchain()
    check_proofs(toolchain)

    with tempfile.TemporaryDirectory(prefix="exl3-bend-build-") as scratch:
        work = Path(scratch)
        staging = work / "artifact"
        staging.mkdir()
        loaders = [build(artifact, work, staging, toolchain) for artifact in ARTIFACTS]
        for loader in loaders:
            loader.admit(staging)
        shutil.copytree(staging, output)
    files = {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())}
    sys.stdout.write(
        json.dumps({"output": str(output), "sha256": files}, indent=2, sort_keys=True)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
