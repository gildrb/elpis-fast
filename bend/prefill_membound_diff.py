#!/usr/bin/env python3
"""
Finite source link of bend/prefill_membound.bend (ext 5110h: the 5110g merge decision plus the staging-scratch page
bound) to the patched engine text. Run it on an engine tree made through 5110h (bend/engine_trees.py
`--through 5110h-prefill-m4096-membound.patch`; on main b18b03f that is OUT/patched itself). Later patches (9503f /
9503b) change the stage function and the bound; bend/qc_staging_diff.py links those.

Checks, each quoted verbatim:
  - the 5110h constants M4096_MAX_STAGE_PAGES = 1024 and M4096_MAX_PROMPT = 262143 (once each, module level), and
    m4096_stage_pages, whose only statement is `return max(1, 1 << (block_table_pages - 1).bit_length())`;
  - the merge `if (...)` block of prefill() with its conjunct lines in the model's order, the stage conjunct right
    after `last_tok <= M4096_MAX_PROMPT and` (exactly once);
  - the 5110g lines the model still uses (bend/prefill_merge_diff.py's JOB and ORDER lists, minus the old 131072
    constant): prefill end, merge widening, last-page cut, kv_position update, in prefill() order;
  - the domain claim that len(seq.allocated_pages) is fixed per job (ast over the whole package): the attribute is
    assigned only in Sequence.__init__ / Sequence.allocate_pages (generator/pagetable.py) and Job.deallocate_pages,
    one element is replaced in Job.receive_sample, and nothing calls append / extend / insert / pop / remove / clear
    on it or deletes from it;
  - the engine's own m4096_stage_pages (extracted with ast, executed without torch) equals the Bend model's
    stage_pages for n = 0..4096 (the model evaluated by a generated Bend main, built with the pinned bend).
Text and finite-evaluation evidence, not a proof. `--mutate NAME` applies a deliberate source mutation that must make it
FAIL.

Usage: python3 bend/prefill_membound_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

N_MAX = 4096
JOB_PY = "generator/job.py"
CONSTS = ["M4096_MAX_STAGE_PAGES = 1024", "M4096_MAX_PROMPT = 262143"]
STAGE_RET = "return max(1, 1 << (block_table_pages - 1).bit_length())"
MERGE_IF = """            if (
                self.recurrent_state is not None and not self.embeddings and prefill_m4096_enabled() and
                last_tok <= M4096_MAX_PROMPT and
                m4096_stage_pages(len(seq.allocated_pages)) <= M4096_MAX_STAGE_PAGES and
                prefill_start == seq.kv_position and prefill_start % (2 * mc) == 0 and
                prefill_end - prefill_start == mc and prefill_end + mc <= last_tok - 2 * mc and mc % PAGE_SIZE == 0
            ):
"""
# bend/prefill_merge_diff.py JOB / ORDER (5110g), minus "M4096_MAX_PROMPT = 131072"; the conjunct lines are in MERGE_IF
JOB = [
    "prefill_end = seq.kv_position + self.generator.max_chunk_size",
    "prefill_end = (prefill_end // PAGE_SIZE) * PAGE_SIZE",
    "prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)",
    "last_tok = len(seq.sequence_ids) - 1",
    "e2 = prefill_end + mc",
    "prefill_end = e2",
    "seqlen = len(seq.sequence_ids) - 1",
    "last_page_b = seqlen // PAGE_SIZE * PAGE_SIZE",
    "if prefill_start < last_page_b <= prefill_end:",
    "prefill_end = last_page_b",
    "seq.kv_position = prefill_end",
]
ORDER = [
    "prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)",
    "if prefill_end <= prefill_start:",
    "m4096_stage_pages(len(seq.allocated_pages)) <= M4096_MAX_STAGE_PAGES and",
    "e2 = prefill_end + mc",
    "if prefill_start < last_page_b <= prefill_end:",
    "seq.kv_position = prefill_end",
]
# every write of an allocated_pages attribute in the package: (file, enclosing def, target)
ALLOC_WRITES = sorted([
    ("generator/pagetable.py", "__init__", "self.allocated_pages"),
    ("generator/pagetable.py", "allocate_pages", "self.allocated_pages"),
    ("generator/job.py", "receive_sample", "seq.allocated_pages[page_before]"),
    ("generator/job.py", "deallocate_pages", "seq.allocated_pages"),
])
RESIZE = {"append", "extend", "insert", "pop", "remove", "clear"}
MUTATIONS = {
    # the 9503b bound in a 5110h tree: the constant check must reject it
    "stage_1088": (JOB_PY, "M4096_MAX_STAGE_PAGES = 1024", "M4096_MAX_STAGE_PAGES = 1088"),
    # drop the stage conjunct (back to 5110g's decision): the merge block check must reject it
    "no_stage_conjunct": (JOB_PY, "                m4096_stage_pages(len(seq.allocated_pages)) <= M4096_MAX_STAGE_PAGES and\n", ""),
    # round without the -1 (1024 pages -> 2048): the evaluation against the model must reject it
    "bitlen_no_dec": (JOB_PY, "return max(1, 1 << (block_table_pages - 1).bit_length())",
                      "return max(1, 1 << block_table_pages.bit_length())"),
    # grow the block table during prefill: the allocated_pages domain check must reject it
    "grow_pages": (JOB_PY, "            if prefill_end <= prefill_start:\n",
                   "            seq.allocated_pages.append(seq.allocated_pages[-1])\n"
                   "            if prefill_end <= prefill_start:\n"),
}

BEND_MAIN = """import Base
import {repo}/bend/prefill_membound.bend as MB

def tab(k: Nat, +n: Nat) -> List<&2, Nat>:
  match k:
    case 0n:
      []
    case 1n+j:
      MB.stage_pages(n) <> tab(j, 1n+n)

def main() -> List<&2, Nat>:
  tab({count}n, 0n)
"""


def fail(msg: str) -> None:
    raise SystemExit(f"prefill_membound_diff: FAIL: {msg}")


def engine_stage(job: str):
    defs = [n for n in ast.parse(job).body if isinstance(n, ast.FunctionDef) and n.name == "m4096_stage_pages"]
    if len(defs) != 1:
        fail(f"job.py: {len(defs)} top-level m4096_stage_pages defs (want 1)")
    body = [s for s in defs[0].body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or ast.unparse(body[0]) != STAGE_RET:
        fail(f"job.py: m4096_stage_pages body is {[ast.unparse(s) for s in body]} (want [{STAGE_RET!r}])")
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=list(defs), type_ignores=[]), "job.py:m4096_stage_pages", "exec"), ns)  # noqa: S102
    return ns["m4096_stage_pages"]


def alloc_writes(files: dict[str, str]) -> list[tuple[str, str, str]]:
    """Every assignment / resize / delete of an allocated_pages attribute (or one of its elements)."""
    found: list[tuple[str, str, str]] = []

    def is_ap(t: ast.AST) -> bool:
        return isinstance(t, ast.Attribute) and t.attr == "allocated_pages"

    def visit(name: str, node: ast.AST, fn: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node.name
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = list(node.targets) if isinstance(node, ast.Delete) else [node.target]
        flat: list[ast.AST] = []
        while targets:
            t = targets.pop()
            if isinstance(t, (ast.Tuple, ast.List)):
                targets.extend(t.elts)
            else:
                flat.append(t)
        for t in flat:
            if is_ap(t) or (isinstance(t, ast.Subscript) and is_ap(t.value)):
                kind = "del " if isinstance(node, ast.Delete) else ""
                found.append((name, fn, kind + ast.unparse(t)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in RESIZE \
                and is_ap(node.func.value):
            found.append((name, fn, ast.unparse(node)))
        for child in ast.iter_child_nodes(node):
            visit(name, child, fn)

    for name, text in files.items():
        visit(name, ast.parse(text), "<module>")
    return sorted(found)


def bend_values(count: int) -> list[int]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "main.bend"
        src.write_text(BEND_MAIN.format(repo=source_link.REPO, count=count))
        exe = Path(td) / "main"
        subprocess.run(source_link.locked([source_link.bend(), str(src), "-o", str(exe)]), check=True,
                       capture_output=True, text=True)
        res = subprocess.run(source_link.locked([str(exe)]), check=True, capture_output=True, text=True)
    vals = [int(x) for x in re.findall(r"(\d+)n", res.stdout)]
    if len(vals) != count:
        fail(f"bend main printed {len(vals)} values (want {count}): {res.stdout[:200]!r}")
    return vals


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    if len(args) == 3 and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: prefill_membound_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    files = {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))}
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}; known: {', '.join(MUTATIONS)}")
        name, old, new = MUTATIONS[mutate]
        if files[name].count(old) != 1:
            fail(f"mutation anchor {old!r} occurs {files[name].count(old)} times in {name}")
        files[name] = files[name].replace(old, new)
        print(f"prefill_membound_diff: applied mutation {mutate}")
    job = files[JOB_PY]
    for c in CONSTS:
        if len(re.findall(rf"^{re.escape(c)}$", job, re.MULTILINE)) != 1:
            fail(f"job.py: {c!r} must occur once as a module-level line")
    if job.count(MERGE_IF) != 1:
        fail(f"job.py: the 5110h merge condition block (conjuncts in model order) occurs {job.count(MERGE_IF)} times "
             "(want 1)")
    body = job[job.index("    def prefill(self, results: list):"):job.index("    def allocate_pages(self):")]
    for e in JOB:
        if body.count(e) != 1:
            fail(f"job.py prefill(): {e!r} occurs {body.count(e)} times (want 1)")
    pos = [body.index(e) for e in ORDER]
    if pos != sorted(pos):
        fail(f"job.py prefill(): order of {ORDER} is {pos}")
    writes = alloc_writes(files)
    if writes != ALLOC_WRITES:
        fail(f"allocated_pages writes {writes} (want {ALLOC_WRITES})")
    eng = engine_stage(job)
    model = bend_values(N_MAX + 1)
    bad = [(n, eng(n), model[n]) for n in range(N_MAX + 1) if eng(n) != model[n]]
    if bad:
        fail(f"engine m4096_stage_pages != Bend stage_pages at (n, engine, model) {bad[:5]} ({len(bad)} total)")
    print(f"prefill_membound_diff: constants {CONSTS}, m4096_stage_pages body, merge block, {len(JOB)} 5110g lines in "
          f"order, {len(writes)} allocated_pages writes (none resizing); engine m4096_stage_pages == Bend stage_pages "
          f"for n = 0..{N_MAX} (1024 -> {eng(1024)}, 1025 -> {eng(1025)})")
    print("prefill_membound_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
