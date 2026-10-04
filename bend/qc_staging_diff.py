#!/usr/bin/env python3
"""
Finite source link of bend/qc_staging.bend (ext 9503f qc_staging_pages + 9503b M4096_MAX_STAGE_PAGES = 1088) and its
reference bend/qc_staging_spec.bend to the patched engine. Run it on P2Baseline's pfast3 tree (main + 9503f +
9503b; bend/engine_trees.py through 9503b-prefill-m4096-stage1088.patch, e.g. OUTp3/patched).

Checks, each quoted verbatim:
  - modules/attention_fn/triton_paged.py: QC_STAGING_ROUND_PAGES = 64 (once, module level); qc_staging_pages, whose
    statements are exactly the model's (pow2, env read, the round64 return, the "0" return, the ValueError);
    `pages_alloc = qc_staging_pages(bsz * npps_w)` the only pages_alloc assignment, and every torch.empty of the staging
    branch (int8 K / scales / V of 3021c's stage, fp16 K / V of the dequant window) sized by pages_alloc; the old
    power-of-two expression gone;
  - generator/job.py: M4096_MAX_STAGE_PAGES = 1088 and M4096_MAX_PROMPT = 262143 (once each, module level);
    m4096_stage_pages = import qc_staging_pages + return qc_staging_pages(block_table_pages); the 5110h merge block with
    the stage conjunct in the model's order;
  - exllamav3_ext/pattn.cu: pages = k_out.size(0) and TORCH_CHECK(pages >= (int64_t) bsz * pps, ...) in
    pattn_stage_int8k (law covers: pages_alloc >= bsz * npps_w);
  - evaluation: the engine's own qc_staging_pages (extracted with ast, executed with a stub os.environ, no torch) for
    n = 0..4096 under EXL3_QC_STAGING_ROUND64 unset / "1" / "0" / bad values equals the Bend model Impl.stage (pages, or
    the ValueError) for every n, and the reference S.sstage for every n >= 1; at n = 0 the only difference is the
    switch "0" (engine 2 pages, reference 1: the pinned law zero_pages). The Bend side is a generated main built with
    the pinned bend.
Text and finite-evaluation evidence, not a proof. `--mutate NAME` applies a deliberate source mutation that must make it
FAIL.

Usage: python3 bend/qc_staging_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

N_MAX = 4096
TP = "modules/attention_fn/triton_paged.py"
JOB = "generator/job.py"
CU = "exllamav3_ext/pattn.cu"
# (engine env value, Bend constructor); the bad values all map to QcOther
ENVS = [(None, "S.QcUnset{}"), ("1", "S.QcOne{}"), ("0", "S.QcZero{}"), ("2", "S.QcOther{}")]
BAD_VALUES = ["2", "", "true", " 1", "01", "on"]
TP_CONST = "QC_STAGING_ROUND_PAGES = 64"
QC_BODY = [
    "pow2 = max(1, 1 << (pages - 1).bit_length())",
    "env = os.environ.get('EXL3_QC_STAGING_ROUND64')",
    "if env is None or env == '1':\n    return min(pow2, -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES)",
    "if env == '0':\n    return pow2",
    "raise ValueError('EXL3_QC_STAGING_ROUND64 must be 0 or 1')",
]
ALLOC = "pages_alloc = qc_staging_pages(bsz * npps_w)"
BRANCH_END = "k_cache, v_cache = kd, vd"
N_EMPTY = 5  # k8, k8s, vd (3021c int8 stage) + kd, vd (dequant window)
OLD_ALLOC = "1 << (bsz * npps_w - 1).bit_length()"
JOB_CONSTS = ["M4096_MAX_STAGE_PAGES = 1088", "M4096_MAX_PROMPT = 262143"]
STAGE_BODY = ["from ..modules.attention_fn.triton_paged import qc_staging_pages",
              "return qc_staging_pages(block_table_pages)"]
MERGE_IF = """            if (
                self.recurrent_state is not None and not self.embeddings and prefill_m4096_enabled() and
                last_tok <= M4096_MAX_PROMPT and
                m4096_stage_pages(len(seq.allocated_pages)) <= M4096_MAX_STAGE_PAGES and
                prefill_start == seq.kv_position and prefill_start % (2 * mc) == 0 and
                prefill_end - prefill_start == mc and prefill_end + mc <= last_tok - 2 * mc and mc % PAGE_SIZE == 0
            ):
"""
CU_LINES = [
    "const int64_t pages = k_out.size(0);",
    "const int bsz = (int) block_table.size(0), pps = (int) block_table.size(1);",
    'TORCH_CHECK(pages >= (int64_t) bsz * pps, "pattn_stage_int8k: scratch too small for block table span");',
]
MUTATIONS = {
    # 128-page rounding: the constant check must reject it
    "round128": (TP, "QC_STAGING_ROUND_PAGES = 64", "QC_STAGING_ROUND_PAGES = 128"),
    # drop the min with the power of two (short jobs grow to 64 pages): the evaluation must reject it
    "no_min": (TP, "return min(pow2, -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES)",
               "return -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES"),
    # unset no longer selects the 64-page rounding (it raises): the evaluation must reject it
    "unset_raises": (TP, 'if env is None or env == "1":', 'if env == "1":'),
    # the allocation back to the power of two: the pages_alloc check must reject it
    "alloc_pow2": (TP, ALLOC, "pages_alloc = max(1, 1 << (bsz * npps_w - 1).bit_length())"),
    # one staging buffer sized by the span instead of pages_alloc: the torch.empty check must reject it
    "empty_span": (TP, "kd = torch.empty((pages_alloc,", "kd = torch.empty((bsz * npps_w,"),
    # the 5110h bound in a 9503b tree: the constant check must reject it
    "stage_1024": (JOB, "M4096_MAX_STAGE_PAGES = 1088", "M4096_MAX_STAGE_PAGES = 1024"),
    # a strict span check: the pattn.cu check must reject it
    "check_gt": (CU, "TORCH_CHECK(pages >= (int64_t) bsz * pps,", "TORCH_CHECK(pages > (int64_t) bsz * pps,"),
}

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


def fail(msg: str) -> None:
    raise SystemExit(f"qc_staging_diff: FAIL: {msg}")


def stmts(fn: ast.FunctionDef) -> list[str]:
    return [ast.unparse(s) for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]


def top_def(src: str, name: str, where: str) -> ast.FunctionDef:
    defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(defs) != 1:
        fail(f"{where}: {len(defs)} top-level {name} defs (want 1)")
    return defs[0]


def engine_fn(tp: str):
    tree = ast.parse(tp)
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and
              [ast.unparse(t) for t in n.targets] == ["QC_STAGING_ROUND_PAGES"]]
    if len(consts) != 1:
        fail(f"{TP}: {len(consts)} module-level QC_STAGING_ROUND_PAGES assignments (want 1)")
    fn = top_def(tp, "qc_staging_pages", TP)
    env: dict[str, str] = {}
    ns: dict[str, Any] = {"os": types.SimpleNamespace(environ=env)}
    exec(compile(ast.Module(body=[consts[0], fn], type_ignores=[]), f"{TP}:qc_staging_pages", "exec"), ns)  # noqa: S102

    def call(value: str | None, n: int) -> int:
        env.clear()
        if value is not None:
            env["EXL3_QC_STAGING_ROUND64"] = value
        try:
            return ns["qc_staging_pages"](n) + 1
        except ValueError:
            return 0
    return call


def bend_tables(count: int) -> dict[tuple[str, bool], list[int]]:
    order = [(ctor, model) for _, ctor in ENVS for model in (True, False)]
    tabs = [f"tab({count}n, 0n, {ctor}, {'True{}' if model else 'False{}'})" for ctor, model in order]
    body = tabs[-1]
    for t in reversed(tabs[:-1]):
        body = f"List.append(&2, Nat, {t}, {body})"
    body = "  " + body
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "main.bend"
        src.write_text(BEND_MAIN.format(repo=source_link.REPO, body=body))
        exe = Path(td) / "main"
        subprocess.run(source_link.locked([source_link.bend(), str(src), "-o", str(exe)]), check=True,
                       capture_output=True, text=True)
        res = subprocess.run(source_link.locked([str(exe)]), check=True, capture_output=True, text=True)
    vals = [int(x) for x in re.findall(r"(\d+)n", res.stdout)]
    if len(vals) != count * len(order):
        fail(f"bend main printed {len(vals)} values (want {count * len(order)}): {res.stdout[:200]!r}")
    return {key: vals[i * count:(i + 1) * count] for i, key in enumerate(order)}


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    if len(args) == 3 and args[0] == "--mutate":
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
            fail(f"mutation anchor {old!r} occurs {files[name].count(old)} times in {name}")
        files[name] = files[name].replace(old, new)
        print(f"qc_staging_diff: applied mutation {mutate}")
    tp, job, cu = files[TP], files[JOB], files[CU]
    # constants
    for src, where, consts in ((tp, TP, [TP_CONST]), (job, JOB, JOB_CONSTS)):
        for c in consts:
            if len(re.findall(rf"^{re.escape(c)}$", src, re.MULTILINE)) != 1:
                fail(f"{where}: {c!r} must occur once as a module-level line")
    # evaluation: engine vs model (every n) and vs reference (n >= 1; n = 0 only the pinned "0" difference)
    eng = engine_fn(tp)
    count = N_MAX + 1
    tables = bend_tables(count)
    for value, ctor in ENVS:
        model, spec = tables[(ctor, True)], tables[(ctor, False)]
        values = BAD_VALUES if ctor == "S.QcOther{}" else [value]
        for v in values:
            got = [eng(v, n) for n in range(count)]
            bad = [(n, got[n] - 1, model[n] - 1) for n in range(count) if got[n] != model[n]]
            if bad:
                fail(f"env {v!r}: engine qc_staging_pages != Bend Impl.stage at (n, engine, model) {bad[:5]} "
                     f"({len(bad)} total; -1 = ValueError)")
            diff = [(n, got[n] - 1, spec[n] - 1) for n in range(count) if got[n] != spec[n]]
            want = [(0, 2, 1)] if ctor == "S.QcZero{}" else []
            if diff != want:
                fail(f"env {v!r}: engine vs reference S.sstage differs at (n, engine, spec) {diff[:5]} (want {want})")
    # qc_staging_pages statements, pages_alloc, the staging buffers
    if stmts(top_def(tp, "qc_staging_pages", TP)) != QC_BODY:
        fail(f"{TP}: qc_staging_pages statements {stmts(top_def(tp, 'qc_staging_pages', TP))} (want {QC_BODY})")
    if tp.count(ALLOC) != 1 or len(re.findall(r"\bpages_alloc\s*=(?!=)", tp)) != 1:
        fail(f"{TP}: {ALLOC!r} must be the only pages_alloc assignment")
    if OLD_ALLOC in tp:
        fail(f"{TP}: the old power-of-two allocation {OLD_ALLOC!r} is still present")
    seg = tp[tp.index(ALLOC):tp.index(BRANCH_END)]
    dims = re.findall(r"torch\.empty\(\(([^,]+),", seg)
    if len(dims) != N_EMPTY or set(dims) != {"pages_alloc"}:
        fail(f"{TP}: staging torch.empty first dims {dims} (want {N_EMPTY} x pages_alloc)")
    # job.py: m4096_stage_pages and the merge block
    if stmts(top_def(job, "m4096_stage_pages", JOB)) != STAGE_BODY:
        fail(f"{JOB}: m4096_stage_pages statements {stmts(top_def(job, 'm4096_stage_pages', JOB))} (want {STAGE_BODY})")
    if job.count(MERGE_IF) != 1:
        fail(f"{JOB}: the merge condition block (stage conjunct in model order) occurs {job.count(MERGE_IF)} times")
    # pattn.cu: the span check of pattn_stage_int8k
    head = "void pattn_stage_int8k\n("
    if cu.count(head) != 1:
        fail(f"{CU}: {head!r} occurs {cu.count(head)} times (want 1)")
    start = cu.index(head)
    fn = cu[start:cu.index("\n}\n", start)]
    pos: list[int] = []
    for line in CU_LINES:
        if cu.count(line) != 1:
            fail(f"{CU}: {line!r} occurs {cu.count(line)} times (want 1)")
        pos.append(fn.find(line))
    if -1 in pos or pos != sorted(pos):
        fail(f"{CU}: span check lines out of order in pattn_stage_int8k: {pos}")
    print(f"qc_staging_diff: constants {[TP_CONST] + JOB_CONSTS}, qc_staging_pages ({len(QC_BODY)} statements), "
          f"pages_alloc + {N_EMPTY} staging buffers, m4096_stage_pages, merge block, pattn.cu span check; engine == Bend "
          f"model for n = 0..{N_MAX} under unset / '1' / '0' / {BAD_VALUES}, == reference for n >= 1 (n = 0, '0': "
          f"engine {eng('0', 0) - 1}, reference 1); 1025 -> {eng(None, 1025) - 1} (switch '0': {eng('0', 1025) - 1})")
    print("qc_staging_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
