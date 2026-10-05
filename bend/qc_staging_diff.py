#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite source link of bend/qc_staging.bend and its reference to the patched engine.

The model is ext 9503f qc_staging_pages + 9503b M4096_MAX_STAGE_PAGES = 1088; its
reference is bend/qc_staging_spec.bend. Run it on P2Baseline's pfast3 tree (main +
9503f + 9503b; bend/engine_trees.py through 9503b-prefill-m4096-stage1088.patch, e.g.
OUTp3/patched).

Checks, each quoted verbatim:
  - modules/attention_fn/triton_paged.py: QC_STAGING_ROUND_PAGES = 64 (once, module
    level); qc_staging_pages, whose statements are exactly the model's (pow2, env
    read, the round64 return, the "0" return, the ValueError);
    `pages_alloc = qc_staging_pages(bsz * npps_w)` the only pages_alloc assignment,
    and every torch.empty of the staging branch (int8 K / scales / V of 3021c's
    stage, fp16 K / V of the dequant window) sized by pages_alloc; the old
    power-of-two expression gone;
  - generator/job.py: M4096_MAX_STAGE_PAGES = 1088 and M4096_MAX_PROMPT = 262143
    (once each, module level); m4096_stage_pages = import qc_staging_pages + return
    qc_staging_pages(block_table_pages); the 5110h merge block with the stage
    conjunct in the model's order;
  - exllamav3_ext/pattn.cu: pages = k_out.size(0) and TORCH_CHECK(pages >= (int64_t)
    bsz * pps, ...) in pattn_stage_int8k (law covers: pages_alloc >= bsz * npps_w);
  - evaluation: the engine's own qc_staging_pages (extracted with ast, evaluated by
    bend/pysubset.py with a stub os.environ, no torch) for n = 0..4096 under
    EXL3_QC_STAGING_ROUND64 unset / "1" / "0" / bad values equals the Bend model
    Impl.stage (pages, or the ValueError) for every n, and the reference S.sstage
    for every n >= 1; at n = 0 the only difference is the switch "0" (engine 2
    pages, reference 1: the pinned law zero_pages). The Bend side is a generated
    main built with the pinned bend.
Text and finite-evaluation evidence, not a proof. `--mutate NAME` applies a
deliberate source mutation that must make it FAIL.

Usage: python3 bend/qc_staging_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

N_MAX = 4096
TP = "modules/attention_fn/triton_paged.py"
JOB = "generator/job.py"
CU = "exllamav3_ext/pattn.cu"
# (engine env value, Bend constructor); the bad values all map to QcOther
ENVS = [
    (None, "S.QcUnset{}"),
    ("1", "S.QcOne{}"),
    ("0", "S.QcZero{}"),
    ("2", "S.QcOther{}"),
]
BAD_VALUES = ["2", "", "true", " 1", "01", "on"]
TP_CONST = "QC_STAGING_ROUND_PAGES = 64"
QC_BODY = [
    "pow2 = max(1, 1 << (pages - 1).bit_length())",
    "env = os.environ.get('EXL3_QC_STAGING_ROUND64')",
    (
        "if env is None or env == '1':\n    return min(pow2, -(-pages // "
        "QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES)"
    ),
    "if env == '0':\n    return pow2",
    "raise ValueError('EXL3_QC_STAGING_ROUND64 must be 0 or 1')",
]
ALLOC = "pages_alloc = qc_staging_pages(bsz * npps_w)"
BRANCH_END = "k_cache, v_cache = kd, vd"
N_EMPTY = 5  # k8, k8s, vd (3021c int8 stage) + kd, vd (dequant window)
OLD_ALLOC = "1 << (bsz * npps_w - 1).bit_length()"
JOB_CONSTS = ["M4096_MAX_STAGE_PAGES = 1088", "M4096_MAX_PROMPT = 262143"]
STAGE_BODY = [
    "from ..modules.attention_fn.triton_paged import qc_staging_pages",
    "return qc_staging_pages(block_table_pages)",
]
MERGE_IF = (
    "            if (\n"
    "                self.recurrent_state is not None and not self.embeddings and "
    "prefill_m4096_enabled() and\n"
    "                last_tok <= M4096_MAX_PROMPT and\n"
    "                m4096_stage_pages(len(seq.allocated_pages)) <= "
    "M4096_MAX_STAGE_PAGES and\n"
    "                prefill_start == seq.kv_position and "
    "prefill_start % (2 * mc) == 0 and\n"
    "                prefill_end - prefill_start == mc and "
    "prefill_end + mc <= last_tok - 2 * mc and mc % PAGE_SIZE == 0\n"
    "            ):\n"
)
CU_LINES = [
    "const int64_t pages = k_out.size(0);",
    "const int bsz = (int) block_table.size(0), pps = (int) block_table.size(1);",
    (
        "TORCH_CHECK(pages >= (int64_t) bsz * pps, "
        '"pattn_stage_int8k: scratch too small for block table span");'
    ),
]
MUTATIONS = {
    # 128-page rounding: the constant check must reject it
    "round128": (TP, "QC_STAGING_ROUND_PAGES = 64", "QC_STAGING_ROUND_PAGES = 128"),
    # drop the min with the power of two (short jobs grow to 64 pages): the
    # evaluation must reject it
    "no_min": (
        TP,
        (
            "return min(pow2, -(-pages // QC_STAGING_ROUND_PAGES) * "
            "QC_STAGING_ROUND_PAGES)"
        ),
        "return -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES",
    ),
    # unset no longer selects the 64-page rounding (it raises): the evaluation must
    # reject it
    "unset_raises": (TP, 'if env is None or env == "1":', 'if env == "1":'),
    # the allocation back to the power of two: the pages_alloc check must reject it
    "alloc_pow2": (
        TP,
        ALLOC,
        "pages_alloc = max(1, 1 << (bsz * npps_w - 1).bit_length())",
    ),
    # one staging buffer sized by the span instead of pages_alloc: the torch.empty
    # check must reject it
    "empty_span": (
        TP,
        "kd = torch.empty((pages_alloc,",
        "kd = torch.empty((bsz * npps_w,",
    ),
    # the 5110h bound in a 9503b tree: the constant check must reject it
    "stage_1024": (JOB, "M4096_MAX_STAGE_PAGES = 1088", "M4096_MAX_STAGE_PAGES = 1024"),
    # a strict span check: the pattn.cu check must reject it
    "check_gt": (
        CU,
        "TORCH_CHECK(pages >= (int64_t) bsz * pps,",
        "TORCH_CHECK(pages > (int64_t) bsz * pps,",
    ),
}
MUTATE_ARGC = 3
SHOWN = 5
SHOWN_STDOUT = 200

BEND_MAIN = """import Base
import {repo}/bend/qc_staging.bend as Impl
import {repo}/bend/qc_staging_spec.bend as S

# pages p -> p + 1, the ValueError -> 0
def enc(o: S.QcOut) -> Nat:
  match o:
    case S.QcPages{{p}}:
      1n+p
    case S.QcError{{}}:
      0n

def pick(model: Bool, +env: S.QcEnv, +n: Nat) -> S.QcOut:
  match model:
    case True{{}}:
      Impl.stage(env, n)
    case False{{}}:
      S.sstage(env, n)

def tab(k: Nat, +n: Nat, +env: S.QcEnv, +model: Bool) -> List<&2, Nat>:
  match k:
    case 0n:
      []
    case 1n+j:
      enc(pick(model, env, n)) <> tab(j, 1n+n, env, model)

def main() -> List<&2, Nat>:
{body}
"""

type Engine = Callable[[str | None, int], int]


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"qc_staging_diff: FAIL: {msg}"
    raise SystemExit(text)


def stmts(fn: ast.FunctionDef) -> list[str]:
    """Return the statements of `fn` as source, docstrings dropped.

    Returns:
        One unparsed source string per statement.

    """
    return [
        ast.unparse(s)
        for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]


def top_def(src: str, name: str, where: str) -> ast.FunctionDef:
    """Return the only top-level def `name` of `src`.

    Args:
        src: The module source.
        name: The function name.
        where: The file, for the failure message.

    Returns:
        The function definition.

    """
    defs = [
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(defs) != 1:
        fail(f"{where}: {len(defs)} top-level {name} defs (want 1)")
    return defs[0]


def engine_fn(tp: str) -> Engine:
    """Extract the engine's qc_staging_pages with a stub os.environ.

    Args:
        tp: The triton_paged.py source.

    Returns:
        A function of (env value or None, n): pages + 1, or 0 for the ValueError.

    """
    tree = ast.parse(tp)
    consts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and [ast.unparse(t) for t in n.targets] == ["QC_STAGING_ROUND_PAGES"]
    ]
    if len(consts) != 1:
        fail(
            f"{TP}: {len(consts)} module-level QC_STAGING_ROUND_PAGES assignments "
            "(want 1)"
        )
    fn = top_def(tp, "qc_staging_pages", TP)
    env: dict[str, str] = {}
    ns: dict[str, object] = {"os": types.SimpleNamespace(environ=env)}
    pysubset.exec_block(ast.Module(body=[consts[0], fn], type_ignores=[]), ns)
    stage = ns["qc_staging_pages"]
    if not callable(stage):
        fail(f"{TP}: qc_staging_pages is not callable")

    def call(value: str | None, n: int) -> int:
        env.clear()
        if value is not None:
            env["EXL3_QC_STAGING_ROUND64"] = value
        try:
            pages = stage(n)
        except ValueError:
            return 0
        if not isinstance(pages, int):
            fail(f"{TP}: qc_staging_pages({n}) returned {pages!r}")
        return pages + 1

    return call


def bend_tables(count: int) -> dict[tuple[str, bool], list[int]]:
    """Build and run a Bend main that tabulates Impl.stage and S.sstage.

    Args:
        count: The number of n values (0 .. count - 1) per table.

    Returns:
        The encoded table (pages + 1, or 0 for the error) per (constructor, model).

    """
    order = [(ctor, model) for _, ctor in ENVS for model in (True, False)]
    tabs = [
        f"tab({count}n, 0n, {ctor}, {'True{}' if model else 'False{}'})"
        for ctor, model in order
    ]
    body = tabs[-1]
    for t in reversed(tabs[:-1]):
        body = f"List.append(&2, Nat, {t}, {body})"
    body = "  " + body
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "main.bend"
        src.write_text(BEND_MAIN.format(repo=source_link.REPO, body=body))
        exe = Path(td) / "main"
        source_link.run(
            [source_link.bend(), str(src), "-o", str(exe)],
            check=True,
            capture_output=True,
            text=True,
        )
        res = source_link.run([str(exe)], check=True, capture_output=True, text=True)
    vals = [int(x) for x in re.findall(r"(\d+)n", res.stdout)]
    if len(vals) != count * len(order):
        fail(
            f"bend main printed {len(vals)} values (want {count * len(order)}): "
            f"{res.stdout[:SHOWN_STDOUT]!r}"
        )
    return {key: vals[i * count : (i + 1) * count] for i, key in enumerate(order)}


def load_files(argv: list[str]) -> dict[str, str]:
    """Parse the command line, read the engine files and apply the mutation.

    Args:
        argv: The command line.

    Returns:
        The (possibly mutated) source of each checked file, by path.

    """
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: qc_staging_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    files = {name: (root / name).read_text() for name in (TP, JOB, CU)}
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}; known: {', '.join(MUTATIONS)}")
        name, old, new = MUTATIONS[mutate]
        if files[name].count(old) != 1:
            fail(
                f"mutation anchor {old!r} occurs {files[name].count(old)} times "
                f"in {name}"
            )
        files[name] = files[name].replace(old, new)
        sys.stdout.write(f"qc_staging_diff: applied mutation {mutate}\n")
    return files


def check_constants(tp: str, job: str) -> None:
    """Fail unless each pinned constant occurs once as a module-level line."""
    for src, where, consts in ((tp, TP, [TP_CONST]), (job, JOB, JOB_CONSTS)):
        for c in consts:
            if len(re.findall(rf"^{re.escape(c)}$", src, re.MULTILINE)) != 1:
                fail(f"{where}: {c!r} must occur once as a module-level line")


def check_evaluation(eng: Engine) -> None:
    """Compare the engine with the model (every n) and the reference (n >= 1).

    At n = 0 the only allowed reference difference is the pinned "0" one.
    """
    count = N_MAX + 1
    tables = bend_tables(count)
    for value, ctor in ENVS:
        model, spec = tables[ctor, True], tables[ctor, False]
        values = BAD_VALUES if ctor == "S.QcOther{}" else [value]
        for v in values:
            got = [eng(v, n) for n in range(count)]
            bad = [
                (n, got[n] - 1, model[n] - 1)
                for n in range(count)
                if got[n] != model[n]
            ]
            if bad:
                fail(
                    f"env {v!r}: engine qc_staging_pages != Bend Impl.stage at "
                    f"(n, engine, model) {bad[:SHOWN]} "
                    f"({len(bad)} total; -1 = ValueError)"
                )
            diff = [
                (n, got[n] - 1, spec[n] - 1) for n in range(count) if got[n] != spec[n]
            ]
            want = [(0, 2, 1)] if ctor == "S.QcZero{}" else []
            if diff != want:
                fail(
                    f"env {v!r}: engine vs reference S.sstage differs at "
                    f"(n, engine, spec) {diff[:SHOWN]} (want {want})"
                )


def check_triton_paged(tp: str) -> None:
    """Check qc_staging_pages statements, pages_alloc and the staging buffers."""
    if stmts(top_def(tp, "qc_staging_pages", TP)) != QC_BODY:
        fail(
            f"{TP}: qc_staging_pages statements "
            f"{stmts(top_def(tp, 'qc_staging_pages', TP))} (want {QC_BODY})"
        )
    if tp.count(ALLOC) != 1 or len(re.findall(r"\bpages_alloc\s*=(?!=)", tp)) != 1:
        fail(f"{TP}: {ALLOC!r} must be the only pages_alloc assignment")
    if OLD_ALLOC in tp:
        fail(f"{TP}: the old power-of-two allocation {OLD_ALLOC!r} is still present")
    seg = tp[tp.index(ALLOC) : tp.index(BRANCH_END)]
    dims = re.findall(r"torch\.empty\(\(([^,]+),", seg)
    if len(dims) != N_EMPTY or set(dims) != {"pages_alloc"}:
        fail(
            f"{TP}: staging torch.empty first dims {dims} "
            f"(want {N_EMPTY} x pages_alloc)"
        )


def check_job(job: str) -> None:
    """Check m4096_stage_pages and the merge block of job.py."""
    if stmts(top_def(job, "m4096_stage_pages", JOB)) != STAGE_BODY:
        fail(
            f"{JOB}: m4096_stage_pages statements "
            f"{stmts(top_def(job, 'm4096_stage_pages', JOB))} (want {STAGE_BODY})"
        )
    if job.count(MERGE_IF) != 1:
        fail(
            f"{JOB}: the merge condition block (stage conjunct in model order) "
            f"occurs {job.count(MERGE_IF)} times"
        )


def check_cu(cu: str) -> None:
    """Check the span check of pattn_stage_int8k in pattn.cu."""
    head = "void pattn_stage_int8k\n("
    if cu.count(head) != 1:
        fail(f"{CU}: {head!r} occurs {cu.count(head)} times (want 1)")
    start = cu.index(head)
    fn = cu[start : cu.index("\n}\n", start)]
    pos: list[int] = []
    for line in CU_LINES:
        if cu.count(line) != 1:
            fail(f"{CU}: {line!r} occurs {cu.count(line)} times (want 1)")
        pos.append(fn.find(line))
    if -1 in pos or pos != sorted(pos):
        fail(f"{CU}: span check lines out of order in pattn_stage_int8k: {pos}")


def main(argv: list[str]) -> None:
    """Run the source link."""
    files = load_files(argv)
    tp, job, cu = files[TP], files[JOB], files[CU]
    check_constants(tp, job)
    eng = engine_fn(tp)
    check_evaluation(eng)
    check_triton_paged(tp)
    check_job(job)
    check_cu(cu)
    sys.stdout.write(
        f"qc_staging_diff: constants {[TP_CONST, *JOB_CONSTS]}, qc_staging_pages "
        f"({len(QC_BODY)} statements), pages_alloc + {N_EMPTY} staging buffers, "
        "m4096_stage_pages, merge block, pattn.cu span check; engine == Bend "
        f"model for n = 0..{N_MAX} under unset / '1' / '0' / {BAD_VALUES}, == "
        "reference for n >= 1 (n = 0, '0': "
        f"engine {eng('0', 0) - 1}, reference 1); 1025 -> {eng(None, 1025) - 1} "
        f"(switch '0': {eng('0', 1025) - 1})\n"
    )
    sys.stdout.write("qc_staging_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
