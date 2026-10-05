#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Link bend/pattn8i_pipe.bend (ext 3022 ping-pong K / V pipeline) to the engine text.

Finite source link to the patched engine text. Argument: the patched exllamav3
package directory (holding exllamav3_ext/pattn8i_kernel.cuh and
exllamav3_ext/pattn.cu). Checks, failing closed:
  1. the CTA loop of pattn8i_kernel is, after dropping comments, blank lines and
     indentation, exactly the loop the model transcribes (P.epochs, P.stage,
     P.other, P.acts): `cp_wait0(); __syncthreads();` first (the epoch
     boundary), then `if (it + 1 < ntiles_cta)` issue_k / issue_v of tile it + 1
     into stage `st ^ 1`, then `qk(st); softmax(it); pv(st);` under act;
  2. the prologue issues tile 0 into stage 0 (P.pro_c0) before the loop, and
     issue_k / issue_v are called nowhere else;
  3. every stage access goes through stage `st` at P8I_ST_OFF + st * P8I_STAGE
     (issue_k, issue_v, qk, pv) and the issued tile is `n0 = it * PA_BN`;
  4. shared memory: P8I_NST == 2 (the model's two stages); evaluated from the
     #defines, the stages are 16-byte aligned, disjoint and inside P8I_SMEM =
     P8I_ST_OFF + P8I_NST * P8I_STAGE <= 99 KiB (the sm86 opt-in limit, also a
     static_assert in the kernel), and the host sets that limit and launches
     with P8I_SMEM;
  5. per-thread arithmetic untouched: the 3022 patch
     (patches/exl3-ext/3022-prefill-pattn8i-pingpong.patch) touches only
     pattn8i_kernel.cuh, removes exactly 3021c's loop body and
     P8I_SMEM define, and adds only comments, the P8I_NST / P8I_SMEM defines and
     the loop of check 1. The qk / softmax / pv lambdas, the issue lambdas, the
     q prologue, the epilogue and the stage pass are 3021c's bytes.
Text evidence, not a proof: it ties the Bend model to one source revision.
`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/pattn8i_pipe_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NoReturn

MUTATE_ARGC = 3
USAGE = "Usage: python3 bend/pattn8i_pipe_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR"
KERNEL = "exllamav3_ext/pattn8i_kernel.cuh"
HOST = "exllamav3_ext/pattn.cu"
PATCH = (
    Path(__file__).resolve().parent.parent
    / "patches/exl3-ext/3022-prefill-pattn8i-pingpong.patch"
)
SMEM_LIMIT = 99 * 1024
ISSUE_SITES = 2  # prologue and loop
N0_SITES = 3  # issue_k, issue_v, softmax
STAGES = 2  # the model's stages (P.stage, P.other)

# check 1: (model def, normalized kernel line), in kernel order
LOOP = [
    ("P.loop", "#pragma unroll 1"),
    ("P.loop / P.epochs", "for (int it = 0; it < ntiles_cta; ++it)"),
    ("", "{"),
    ("P.stage", "const int st = it & 1;"),
    ("epoch boundary", "cp_wait0();"),
    ("epoch boundary", "__syncthreads();"),
    ("Ep.wr = Nat.is_lt(1n+it, nt)", "if (it + 1 < ntiles_cta)"),
    ("", "{"),
    ("Ep.wt = 1n+it, Ep.ws = P.other(st)", "issue_k(it + 1, st ^ 1, tid, P8_THREADS);"),
    ("Ep.wt = 1n+it, Ep.ws = P.other(st)", "issue_v(it + 1, st ^ 1, tid, P8_THREADS);"),
    ("", "}"),
    ("", "cp_commit();"),
    ("P.acts (act)", "const bool act = it < ntiles;"),
    ("Ep.rt = it, Ep.rs = st", "if (act) { qk(st); softmax(it); pv(st); }"),
    ("", "}"),
    ("", "cp_wait0();"),
]
# check 2
PROLOGUE = [
    "if (ntiles_cta > 0) issue_k(0, 0, tid, P8_THREADS);",
    "if (ntiles_cta > 0) issue_v(0, 0, tid, P8_THREADS);",
]
# check 3
STAGE_ACCESS = [
    "unsigned char* base = smem + P8I_ST_OFF + st * P8I_STAGE;",
    "half* vs = (half*) (smem + P8I_ST_OFF + st * P8I_STAGE + P8I_V_REL);",
    "const unsigned char* ks = smem + P8I_ST_OFF + st * P8I_STAGE;",
    "const half* vs = (const half*) (smem + P8I_ST_OFF + st * P8I_STAGE + P8I_V_REL);",
]
# check 4
DEFINES = {
    "P8I_Q_OFF": "0",
    "P8I_QS_OFF": "(P8I_Q_OFF + P8_NW * PA_BQ * PA_HD)",
    "P8I_ST_OFF": "(P8I_QS_OFF + P8_NW * PA_BQ * 4)",
    "P8I_KS_REL": "(PA_BN * PA_HD)",
    "P8I_V_REL": "(P8I_KS_REL + PA_BN * 4)",
    "P8I_STAGE": "(P8I_V_REL + PA_BN * PA_HD * 2)",
    "P8I_NST": "2",
    "P8I_SMEM": "(P8I_ST_OFF + P8I_NST * P8I_STAGE)",
}
BASE_CONSTS = {"PA_HD": 256, "PA_BQ": 16, "PA_BN": 32, "P8_NW": 8}
HOST_LINES = [
    (
        "cuda_check(cudaFuncSetAttribute(pattn_detail::pattn8i_kernel, "
        "cudaFuncAttributeMaxDynamicSharedMemorySize, P8I_SMEM));"
    ),
    "pattn_detail::pattn8i_kernel<<<grid, P8_THREADS, P8I_SMEM, stream>>>(P8I_ARGS);",
]
# check 5: the 3021c lines 3022 removes, in order (comments dropped)
REMOVED = [
    "#define P8I_SMEM (P8I_ST_OFF + P8I_STAGE)",
    'asm volatile("cp.async.wait_group 1;\\n" ::);',
    "__syncthreads();",
    "const bool act = it < ntiles;",
    "if (act) { qk(0); softmax(it); }",
    "cp_wait0();",
    "__syncthreads();",
    "if (it + 1 < ntiles_cta) issue_k(it + 1, 0, tid, P8_THREADS);",
    "cp_commit();",
    "if (act) pv(0);",
    'asm volatile("cp.async.wait_group 1;\\n" ::);',
    "__syncthreads();",
    "if (it + 1 < ntiles_cta) issue_v(it + 1, 0, tid, P8_THREADS);",
]
ADDED = {
    "#define P8I_NST 2",
    "#define P8I_SMEM (P8I_ST_OFF + P8I_NST * P8I_STAGE)",
    *(line for _, line in LOOP),
}

MUTATIONS = {
    # tile it + 1 issued into the stage tile it is read from
    "pp_same_stage": (
        "issue_k(it + 1, st ^ 1, tid, P8_THREADS);",
        "issue_k(it + 1, st, tid, P8_THREADS);",
    ),
    # no wait for the landed tile before the epoch barrier
    "pp_no_wait": (
        "const int st = it & 1;\n        cp_wait0();",
        "const int st = it & 1;",
    ),
    # one stage only
    "pp_one_stage": ("#define P8I_NST 2", "#define P8I_NST 1"),
    # the prologue stages tile 0 into stage 1
    "pp_prologue_stage": (
        "if (ntiles_cta > 0) issue_v(0, 0, tid, P8_THREADS);",
        "if (ntiles_cta > 0) issue_v(0, 1, tid, P8_THREADS);",
    ),
    # a PV read from stage 0 regardless of st
    "pp_pv_stage": ("pv(st); }", "pv(0); }"),
}


def fail(msg: str) -> NoReturn:
    """Exit with the link's failure message.

    Raises:
        SystemExit: always.

    """
    text = f"pattn8i_pipe_diff: FAIL: {msg}"
    raise SystemExit(text)


def out(text: str) -> None:
    """Write one line to stdout."""
    sys.stdout.write(f"{text}\n")


def code(line: str) -> str:
    """Drop a // comment and indentation; collapse whitespace.

    Returns:
        The code of the line ("" for a comment or blank line).

    """
    return " ".join(line.split("//", 1)[0].split())


def norm(text: str) -> list[str]:
    """Normalize every line with code() and drop the empty ones.

    Returns:
        The normalized non-empty lines.

    """
    return [c for c in map(code, text.splitlines()) if c]


def at(text: str, expr: str, where: str) -> int:
    """Fail unless expr occurs exactly once in text.

    Returns:
        The offset of the occurrence.

    """
    n = text.count(expr)
    if n != 1:
        fail(f"{where}: {expr!r} occurs {n} times (want 1)")
    return text.index(expr)


def once(text: str, expr: str, where: str) -> None:
    """Fail unless expr occurs exactly once in text."""
    _ = at(text, expr, where)


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line.

    Returns:
        The mutation name (or None) and the engine package directory.

    """
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail(USAGE)
    return mutate, Path(args[0])


def check_loop(k8i: str) -> int:
    """Check 1: the CTA loop is the model's.

    Returns:
        The offset of the loop in the kernel text.

    """
    start = at(k8i, "for (int it = 0; it < ntiles_cta; ++it)", KERNEL)
    head = k8i.rindex("#pragma unroll 1", 0, start)
    end = at(k8i, "if (ntiles == 0) return;", KERNEL)
    got = norm(k8i[head:end])
    want = [line for _, line in LOOP]
    if got != want:
        for i, (g, w) in enumerate(zip(got, want, strict=False)):
            if g != w:
                fail(f"loop line {i + 1}: {g!r} != {w!r} ({LOOP[i][0] or 'frame'})")
        fail(f"loop has {len(got)} normalized lines, want {len(want)}")
    out(f"pattn8i_pipe_diff: CTA loop = model ({len(want)} normalized lines)")
    return head


def check_issue(k8i: str, loop_at: int) -> None:
    """Check 2 (prologue, issue call sites) and 3 (stage addressing)."""
    for line in PROLOGUE:
        if at(k8i, line, KERNEL) > loop_at:
            fail(f"prologue issue {line!r} is not before the loop")
    for name in ("issue_k(", "issue_v("):
        n = k8i.count(name)
        if n != ISSUE_SITES:
            fail(f"{name} called {n} times (want the prologue and the loop)")
    for line in STAGE_ACCESS:
        once(k8i, line, KERNEL)
    if k8i.count("P8I_ST_OFF + ") != len(STAGE_ACCESS) + 1:
        fail("a stage is addressed outside issue_k / issue_v / qk / pv")
    n = k8i.count("const int n0 = it * PA_BN;")
    if n != N0_SITES:
        fail(f"`const int n0 = it * PA_BN;` occurs {n} times (want {N0_SITES})")
    out(
        "pattn8i_pipe_diff: prologue tile 0 -> stage 0; stages addressed by st "
        "only; tile it at n0 = it * PA_BN"
    )


def evaluate(expr: str, env: dict[str, int]) -> int:
    """Evaluate a +, * integer expression over named constants.

    Returns:
        Its value.

    """

    def ev(node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return ev(node.left) * ev(node.right)
        fail(f"cannot evaluate {expr!r}")

    return ev(ast.parse(expr, mode="eval").body)


def check_smem(k8i: str, host: str) -> None:
    """Check 4: two stages, aligned, disjoint, within the opt-in limit."""
    env = dict(BASE_CONSTS)
    for name, rhs in DEFINES.items():
        found = re.findall(rf"^#define {name} (.*?)\s*(?://.*)?$", k8i, re.MULTILINE)
        if found != [rhs]:
            fail(f"#define {name}: {found} (want [{rhs!r}])")
        env[name] = evaluate(rhs, env)
    once(
        k8i,
        'static_assert(P8I_SMEM <= 99 * 1024, "sm86 opt-in shared memory");',
        KERNEL,
    )
    once(
        k8i,
        "static_assert(P8I_ST_OFF % 16 == 0 && P8I_STAGE % 16 == 0 && "
        'P8I_V_REL % 16 == 0, "16-byte cp.async slots");',
        KERNEL,
    )
    if env["P8I_NST"] != STAGES:
        fail(f"P8I_NST = {env['P8I_NST']}, the model has {STAGES} stages")
    st_off, stage, smem = env["P8I_ST_OFF"], env["P8I_STAGE"], env["P8I_SMEM"]
    spans = [
        (st_off + s * stage, st_off + (s + 1) * stage) for s in range(env["P8I_NST"])
    ]
    if any(a % 16 for a, _ in spans) or env["P8I_V_REL"] % 16:
        fail("stage slots are not 16-byte aligned")
    if (
        spans[0][1] > spans[1][0]
        or spans[-1][1] > smem
        or st_off < env["P8I_QS_OFF"] + 4 * 128
    ):
        fail(f"stages {spans} overlap each other, q, or exceed P8I_SMEM = {smem}")
    if smem > SMEM_LIMIT:
        fail(f"P8I_SMEM = {smem} > {SMEM_LIMIT}")
    for line in HOST_LINES:
        once(host, line, HOST)
    out(
        f"pattn8i_pipe_diff: smem q {env['P8I_QS_OFF']} + scales "
        f"{st_off - env['P8I_QS_OFF']} + {STAGES} x stage {stage} = {smem} "
        f"<= {SMEM_LIMIT}; stages {spans}"
    )


def check_patch() -> None:
    """Check 5: 3022 changes only the schedule lines."""
    text = PATCH.read_text(encoding="utf-8")
    files = re.findall(r"^\+\+\+ b/(\S+)$", text, re.MULTILINE)
    if files != [KERNEL]:
        fail(f"3022 touches {files}, want only {KERNEL}")
    removed: list[str] = []
    added: list[str] = []
    for line in text.splitlines():
        if line.startswith(("--- ", "+++ ", "@@")):
            continue
        if line.startswith("-"):
            removed.append(code(line[1:]))
        elif line.startswith("+"):
            added.append(code(line[1:]))
    if removed != REMOVED:
        fail(f"3022 removes {removed}, want 3021c's loop body and P8I_SMEM define only")
    stray = [x for x in added if x and x not in ADDED]
    if stray:
        fail(f"3022 adds non-schedule lines {stray}")
    out(
        f"pattn8i_pipe_diff: 3022 removes {len(removed)} schedule lines and adds "
        f"{len(added)} lines, schedule and comments only"
    )


def main(argv: list[str]) -> None:
    """Run checks 1-5 against the engine package named in argv."""
    mutate, root = parse_args(argv)
    k8i = (root / KERNEL).read_text()
    host = (root / HOST).read_text()
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r}")
        old, new = MUTATIONS[mutate]
        once(k8i, old, "mutation anchor")
        k8i = k8i.replace(old, new, 1)
        out(f"pattn8i_pipe_diff: applied mutation {mutate}")
    loop_at = check_loop(k8i)
    check_issue(k8i, loop_at)
    check_smem(k8i, host)
    check_patch()
    out("pattn8i_pipe_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
