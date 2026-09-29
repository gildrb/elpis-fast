#!/usr/bin/env python3
"""Build the EXL3 Bend acceptor root on the host (CPU only).

  nix develop --offline --no-write-lock-file -c \\
    python3 bend/exl3_build.py --output build/bend-exl3

Uses the flake-pinned bend 2.0.34 and clang 19.1.7 from the dev shell PATH.
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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEND_VERSION = "bend 2.0.34\n"
CLANG_VERSION = "clang version 19.1.7"
PROOFS = ("bend/exl3_accept_proof.bend", "bend/exl3_tree_accept_gate.bend")
# bend 2.0.34 bend2/main.ts cli_verdict: PASS plus the --verdict hint, on stdout. It prints
# PASS only when every def checks and none relies on @unsafe or foreign code, imports included.
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
    r"^(?:INLINE|FAR) Term (spin_[0-9]+)\(Env e, THR Term\* o((?:, (?:u32|Term) r[0-9]+)*)\) \{$",
    re.MULTILINE,
)
TOKEN = re.compile(
    r"\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:ull)?|==|!=|>=|<=|[-+*|&=<>;,()\[\]{}])"
)
# u64 is a scalar type name only: bend 2.0.34 emits a U32 register copied into a Term (u64) slot
# as the explicit widening cast ((u64)x), which bend 2.0.29 left implicit (same C conversion).
KEYWORDS = frozenset(
    {"INLINE", "FAR", "Term", "Env", "THR", "e", "o", "u32", "u64", "wpoll", "WL_SPIN", "WL_AGAIN", "U32_BIN", "TAB_AT"}
    | {"if", "else", "return", "break"}
)
LOCAL = re.compile(r"_[A-Za-z0-9_]*_(?:0|[1-9][0-9]*)|r(?:0|[1-9][0-9]*)")
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
# One artifact per acceptor: its programs, glue, loader, library, table and
# identity, and the admitted leaves (argument kinds, output count).
ARTIFACTS = (
    {
        "program": "exl3_accept",
        "reference": "exl3_accept_spec",
        "production": "bend/EXL3_ACCEPT.bend",
        "checker": "bend/EXL3_ACCEPT_SPEC.bend",
        "glue": "bend/exl3_accept_glue.c",
        "loader": "bend/exl3_bend_accept.py",
        "library": "libexl3_accept.so",
        "table": "exl3_accept_table.txt",
        "identity": "identity.json",
        "schema": "elpis-exl3-bend-accept/1",
        "leaves": {"ELPIS_EXL3_LEAF": (("Term",) * 4 + ("u32",) * 19, 2)},
    },
    {
        "program": "exl3_tree_accept",
        "reference": "exl3_tree_accept_spec",
        "production": "bend/EXL3_TREE_ACCEPT.bend",
        "checker": "bend/EXL3_TREE_ACCEPT_SPEC.bend",
        "glue": "bend/exl3_tree_accept_glue.c",
        "loader": "bend/exl3_bend_tree_accept.py",
        "library": "libexl3_tree_accept.so",
        "table": "exl3_tree_accept_table.txt",
        "identity": "tree_identity.json",
        "schema": "elpis-exl3-bend-tree-accept/1",
        "leaves": {
            "ELPIS_EXL3_TREE_LEAF": (("Term",) * 3 + ("u32",) * 19 + ("Term",) * 7, 3),
            "ELPIS_EXL3_TREE_DERIVE": (("Term",) * 7, 128),
        },
    },
)
OUTPUT = re.compile(r"\bo\[([0-9]+)\] =")


def fail(message: str) -> SystemExit:
    return SystemExit(f"exl3_build: {message}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=environment, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise fail(f"{command} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def resolve(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise fail(f"{name} is not on PATH (run inside the pinned nix dev shell)")
    return Path(found)


def compile_environment(clang: Path, extra: dict[str, str]) -> dict[str, str]:
    return {"PATH": str(clang.parent), **extra}


def load_loader(path: str) -> object:
    spec = importlib.util.spec_from_file_location(Path(path).stem, REPO / path)
    if spec is None or spec.loader is None:
        raise fail("cannot load the acceptance loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(text: str) -> dict[str, tuple[tuple[str, ...], str]]:
    found: dict[str, tuple[tuple[str, ...], str]] = {}
    for match in SIGNATURE.finditer(text):
        closing = re.search(r"^}", text[match.end() :], re.MULTILINE)
        if closing is None:
            raise fail("incomplete emitted scalar function")
        arguments = tuple(re.findall(r", (u32|Term) r[0-9]+", match.group(2)))
        if match.group(1) in found:
            raise fail("duplicate emitted scalar function")
        body = text[match.start() : match.end() + closing.end()]
        found[match.group(1)] = (arguments, body)
    return found


def admit_leaf(
    text: str, arguments: tuple[str, ...], outputs: int
) -> dict[str, object]:
    """Admit the unique leaf with this ABI and its acyclic scalar-only call closure."""
    for line in RUNTIME_ABI:
        if text.count(line) != 1:
            raise fail(f"emitted runtime ABI not established: {line!r}")
    table = functions(text)
    leaves = [
        name
        for name, (kinds, body) in table.items()
        if kinds == arguments
        and {int(k) for k in OUTPUT.findall(body)} == set(range(outputs))
    ]
    if len(leaves) != 1:
        raise fail(f"expected one emitted leaf with the {arguments} ABI, found {leaves}")
    admitted: dict[str, str] = {}
    tables: dict[str, str] = {}
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active or name not in table:
            raise fail(f"recursive or unknown scalar callee {name}")
        if name in admitted:
            return
        active.add(name)
        body = table[name][1]
        position = 0
        while position < len(body):
            match = TOKEN.match(body, position)
            if match is None:
                if body[position:].strip():
                    raise fail(
                        f"unadmitted text in {name}: {body[position : position + 40]!r}"
                    )
                break
            token = match.group(1)
            position = match.end()
            if not (token[0].isalpha() or token[0] == "_"):
                continue
            if re.fullmatch(r"spin_(?:0|[1-9][0-9]*)", token):
                if token != name:
                    visit(token)
                continue
            if TABLE.fullmatch(token):
                continue
            if token not in KEYWORDS and LOCAL.fullmatch(token) is None:
                raise fail(f"unadmitted token {token!r} in {name}")
        for read in TABLE_READ.finditer(body):
            table_name, bound = read.group(1), int(read.group(2))
            found = re.findall(
                rf"^CONSTV u64 {table_name}\[\] = \{{ ((?:[0-9]+ull, )*[0-9]+ull) \}};$",
                text,
                re.MULTILINE,
            )
            if len(found) != 1 or len(found[0].split(", ")) <= bound:
                raise fail(f"table read {read.group(0)!r} in {name} is not in bounds")
            tables[table_name] = found[0]
        if len(TABLE_READ.findall(body)) != body.count("TAB_AT("):
            raise fail(f"unadmitted table read in {name}")
        active.remove(name)
        admitted[name] = digest(body.encode("utf-8"))

    visit(leaves[0])
    leaf: dict[str, object] = {"name": leaves[0], "functions": dict(sorted(admitted.items()))}
    if tables:
        leaf["tables"] = dict(sorted(tables.items()))
    return leaf


def build(
    artifact: dict[str, object],
    work: Path,
    staging: Path,
    bend: Path,
    clang: Path,
    bend_environment: dict[str, str],
) -> dict[str, object]:
    """Emit, compile, compare, admit and stage one artifact; return its identity."""
    program = str(artifact["program"])
    reference = str(artifact["reference"])
    loader = load_loader(str(artifact["loader"]))
    reference_sha256 = getattr(loader, "TABLE_SHA256")
    if not isinstance(reference_sha256, str):
        raise fail(f"{artifact['loader']} does not pin a reference table")
    for name, source in (
        (program, artifact["production"]),
        (reference, artifact["checker"]),
    ):
        command = [str(bend), str(REPO / str(source)), "-o", str(work / f"{name}.c")]
        run(command, REPO, bend_environment)
    tables: dict[str, str] = {}
    for name in (program, reference):
        run(
            [str(clang), *PROGRAM_FLAGS, f"{name}.c", *LINK_FLAGS, "-o", name],
            work,
            compile_environment(clang, {}),
        )
        tables[name] = run([str(work / name), "--gpu", "off"], work, {})
    if tables[program] != tables[reference]:
        raise fail(f"{program} table differs from the reference program")
    table = tables[reference].encode("ascii")
    if digest(table) != reference_sha256:
        raise fail(f"{reference} table differs from the pinned table")
    emitted = (work / f"{program}.c").read_text(encoding="utf-8")
    leaves_abi = artifact["leaves"]
    assert isinstance(leaves_abi, dict)
    leaves = {
        macro: admit_leaf(emitted, kinds, outputs)
        for macro, (kinds, outputs) in leaves_abi.items()
    }
    glue = Path(str(artifact["glue"])).name
    shutil.copyfile(REPO / str(artifact["glue"]), work / glue)
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
    library = str(artifact["library"])
    run(
        [str(clang), *LIBRARY_FLAGS, unit, *LINK_FLAGS, "-o", library],
        work,
        compile_environment(clang, LIBRARY_ENVIRONMENT),
    )
    shutil.copyfile(work / library, staging / library)
    (staging / str(artifact["table"])).write_bytes(table)
    loader_name = Path(str(artifact["loader"])).name
    shutil.copyfile(REPO / str(artifact["loader"]), staging / loader_name)
    artifacts = {
        name: digest((staging / name).read_bytes())
        for name in sorted((library, str(artifact["table"]), loader_name))
    }
    identity: dict[str, object] = {
        "schema": artifact["schema"],
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
        "artifacts": artifacts,
        "table_sha256": reference_sha256,
    }
    if program == "exl3_accept":
        identity["leaf"] = leaves["ELPIS_EXL3_LEAF"]
    else:
        identity["leaves"] = {macro: leaf for macro, leaf in sorted(leaves.items())}
    identity["identity_sha256"] = digest(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    (staging / str(artifact["identity"])).write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"loader": loader}


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args(arguments).output.resolve()
    if output.exists():
        raise fail(f"{output} already exists")
    bend = resolve("bend")
    clang = resolve("clang")
    bend_environment = {**os.environ, "BEND_NO_TELEMETRY": "1"}
    if run([str(bend), "version"], REPO, bend_environment) != BEND_VERSION:
        raise fail("requires exactly bend 2.0.34")
    clang_version = run([str(clang), "--version"], REPO, compile_environment(clang, {}))
    if not clang_version.startswith(CLANG_VERSION + "\n"):
        raise fail("requires exactly clang 19.1.7")
    for name in BEND_SOURCES:
        text = (REPO / name).read_text(encoding="utf-8")
        if (
            "@unsafe" in text
            or "?TODO" in text
            or re.search(r"def [A-Za-z0-9_.]+\?\(", text)
        ):
            raise fail(f"{name} contains an unsafe definition or an open hole")
    for proof in PROOFS:
        if run([str(bend), str(REPO / proof)], REPO, bend_environment) != VERDICT:
            raise fail(f"proof gate {proof} did not report exactly {VERDICT!r}")

    with tempfile.TemporaryDirectory(prefix="exl3-bend-build-") as scratch:
        work = Path(scratch)
        staging = work / "artifact"
        staging.mkdir()
        built = [
            build(artifact, work, staging, bend, clang, bend_environment)
            for artifact in ARTIFACTS
        ]
        for artifact in built:
            getattr(artifact["loader"], "admit")(staging)
        shutil.copytree(staging, output)
    files = {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir())}
    print(
        json.dumps({"output": str(output), "sha256": files}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
