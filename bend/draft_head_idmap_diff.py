#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential check of bend/draft_head_idmap.bend against the engine source.

bend/draft_head_idmap.bend models the pruned int4 draft head (ext patch 9005c); the
check runs the patched engine tree's own Python and CUDA source on the CPU.

From TREE: modules/arch_specific/dflash2_head_blocks.py = S (constants,
DRAFT_HEAD_BLOCK_ORDER, pruned_mode, kept_blocks, PrunedHeadPolicy),
modules/arch_specific/dflash2_q4_head.py = Q (HEAD_TOP_K, the kept-block check,
kept = set(kb), keep_cols, the id_map statements, the topk slicing statements),
exllamav3_ext/dflash2_head.cu = D (HEAD_TOPK, better(), the epilogue's tv / ti
statements), architecture/dflash2.py = A (begin_job -> reset, the observe_anchor
call), generator/job.py = J (begin_job at each job's prefill). Every block is located
by its signature (exactly once), quoted verbatim with its line range, and every quoted
non-blank line of a 9005c block must be a `+` line of the pinned 9005c patch (sha256
checked). The parsed constants and the 1024-entry block order must equal the Bend
model's literals (draft_head_idmap.bend); if --order-json FILE gives the DraftProj
full order (block_order_code.json) it must be a permutation of the 1940 blocks,
extend the source table and match the source's pinned full-order sha256.

Reference computations, each on the quoted source itself, run by bend/pysubset.py:
  - S is executed as a module (it imports only os; a stub with the environment the
    check sets stands in for it): kept_blocks(248320) for N = 896 and 1024
    (EXL3_DRAFT_HEAD_BLOCKS), and kept_blocks with its module globals replaced by
    fixed small orders;
  - Q's statements are executed with a numpy stand-in for the torch calls they use
    (arange / view / zeros / tensor indexing / ~ / cat / to): the kept-block check
    (raises = rejected), keep_cols, the id map, and the topk slicing (n and im[:n]
    for pruned False / True);
  - D's better() and epilogue statements are compiled into a C++ program that
    computes, for four value seeds, the top-16 of the identity call, of the full call
    with Q's id map and of the prefix call with Q's im[:n] (std::sort by the quoted
    better(); the warp / block / grid merge network is not emulated), and checks on
    the C side that the full call equals the identity call and the prefix call
    equals the identity ranking restricted to the prefix's tokens;
  - PrunedHeadPolicy replays the Bend table's anchor / reset sequences.
The compiled Bend table (DRAFT_HEAD_IDMAP_TABLE.bend) must equal the reference output
byte for byte, and the policy replay must cover a trigger, a reset during a hold, a
completed hold, out-of-range anchors and the static mode. Differential evidence on
finite instances, not a proof; the proof is draft_head_idmap_proof.bend.

Usage: see USAGE (printed on a usage error).
"""

from __future__ import annotations

import hashlib
import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, Protocol, runtime_checkable

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError:
    NUMPY_MISSING = (
        "draft_head_idmap_diff: FAIL numpy is not importable. Run the driver with the "
        "python313.withPackages (ps: [ ps.numpy ]) command in the module docstring."
    )
    raise SystemExit(NUMPY_MISSING) from None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

HERE = Path(__file__).resolve().parent
REPO = source_link.REPO

USAGE = (
    "\n"
    "Finite differential check of bend/draft_head_idmap.bend (pruned int4 draft head, "
    "ext patch 9005c) against the\n"
    "patched engine tree's own Python and CUDA source, executed on the CPU.\n"
    "\n"
    "From TREE: modules/arch_specific/dflash2_head_blocks.py = S (constants, "
    "DRAFT_HEAD_BLOCK_ORDER, pruned_mode,\n"
    "kept_blocks, PrunedHeadPolicy), modules/arch_specific/dflash2_q4_head.py = Q "
    "(HEAD_TOP_K, the kept-block check,\n"
    "kept = set(kb), keep_cols, the id_map statements, the topk slicing statements), "
    "exllamav3_ext/dflash2_head.cu = D\n"
    "(HEAD_TOPK, better(), the epilogue's tv / ti statements), architecture/dflash2.py "
    "= A (begin_job -> reset, the\n"
    "observe_anchor call), generator/job.py = J (begin_job at each job's prefill). "
    "Every block is located by its signature\n"
    "(exactly once), quoted verbatim with its line range, and every quoted non-blank "
    "line of a 9005c block must be a `+`\n"
    "line of the pinned 9005c patch (sha256 checked). The parsed constants and the "
    "1024-entry block order must equal the\n"
    "Bend model's literals (draft_head_idmap.bend); if --order-json FILE gives the "
    "DraftProj full order\n"
    "(block_order_code.json) it must be a permutation of the 1940 blocks, extend the "
    "source table and match the source's\n"
    "pinned full-order sha256.\n"
    "\n"
    "Reference computations, each on the quoted source itself:\n"
    "  - S is executed as a module (it imports only os): kept_blocks(248320) for N = "
    "896 and 1024\n"
    "    (EXL3_DRAFT_HEAD_BLOCKS), and kept_blocks with its module globals replaced by "
    "fixed small orders;\n"
    "  - Q's statements are executed with a numpy stand-in for the torch calls they "
    "use (arange / view / zeros / tensor\n"
    "    indexing / ~ / cat / to): the kept-block check (raises = rejected), "
    "keep_cols, the id map, and the topk slicing\n"
    "    (n and im[:n] for pruned False / True);\n"
    "  - D's better() and epilogue statements are compiled into a C++ program that "
    "computes, for four value seeds, the\n"
    "    top-16 of the identity call, of the full call with Q's id map and of the "
    "prefix call with Q's im[:n]\n"
    "    (std::sort by the quoted better(); the warp / block / grid merge network is "
    "not emulated), and checks on the C\n"
    "    side that the full call equals the identity call and the prefix call equals "
    "the identity ranking restricted to\n"
    "    the prefix's tokens;\n"
    "  - PrunedHeadPolicy replays the Bend table's anchor / reset sequences.\n"
    "The compiled Bend table (DRAFT_HEAD_IDMAP_TABLE.bend) must equal the reference "
    "output byte for byte, and the policy\n"
    "replay must cover a trigger, a reset during a hold, a completed hold, "
    "out-of-range anchors and the static mode.\n"
    "Differential evidence on finite instances, not a proof; the proof is "
    "draft_head_idmap_proof.bend.\n"
    "\n"
    "Usage: python3 bend/draft_head_idmap_diff.py [--order-json FILE] "
    "[--mutate NAME | --all-mutations] TREE\n"
    "  TREE: OUT/patched of bend/engine_trees.py.\n"
    "  FILE: block_order_code.json, written by DraftProj's select_order.py from host "
    "corpora. It is not in the repo and\n"
    "  the repo cannot make it again. Without it, the full-order check does not run "
    "(the output tells this).\n"
    "Needs numpy. The dev shell does not provide numpy. From the repo root, run:\n"
    "  nix develop --offline --no-write-lock-file -c nix shell --impure --expr 'let "
    "p = (builtins.getFlake\n"
    '  "git+file://${toString ./.}").inputs.nixpkgs.legacyPackages.x86_64-linux; in '
    "p.python313.withPackages\n"
    "  (ps: [ ps.numpy ])' -c python3 -I -B bend/draft_head_idmap_diff.py "
    "[--order-json FILE] TREE\n"
)

PATCH_NAME = "9005c-elpis-draft-q4-head-pruned-n896.patch"
PATCH_PATH = REPO / "patches/exl3-ext" / PATCH_NAME
PATCH_SHA256 = "e7a2952ff6485c74d8b443fdeb751b9961f2c834b60b0d2940af6e77b7ce4db9"
TABLE = "DRAFT_HEAD_IDMAP_TABLE.bend"
IMPL = "draft_head_idmap.bend"
PROOF = "draft_head_idmap_proof.bend"

PATHS = {
    "S": "modules/arch_specific/dflash2_head_blocks.py",
    "Q": "modules/arch_specific/dflash2_q4_head.py",
    "D": "exllamav3_ext/dflash2_head.cu",
    "A": "architecture/dflash2.py",
    "J": "generator/job.py",
}

# (file key, name, signature prefix of the first stripped line, kind, origin). kinds:
# "line"; "py" = a Python def / class (the first line and every following line
# indented deeper, or blank); "paren" = a Python statement up to its balanced closing
# parenthesis; "lines:N" = N lines; "brace" = a C block to its matching brace.
# origin: "new" = a 9005c line (each non-blank line a `+` line of the patch), "old" =
# pre-9005c source (not checked).
BLOCKS = [
    ("S", "DRAFT_HEAD_VOCAB", "DRAFT_HEAD_VOCAB = ", "line", "new"),
    ("S", "DRAFT_HEAD_BLOCK", "DRAFT_HEAD_BLOCK = ", "line", "new"),
    ("S", "DRAFT_HEAD_BLOCKS_ALLOWED", "DRAFT_HEAD_BLOCKS_ALLOWED = ", "line", "new"),
    ("S", "DRAFT_HEAD_BLOCKS_DEFAULT", "DRAFT_HEAD_BLOCKS_DEFAULT = ", "line", "new"),
    ("S", "FALLBACK_WINDOW", "FALLBACK_WINDOW = ", "line", "new"),
    ("S", "FALLBACK_TRIGGER", "FALLBACK_TRIGGER = ", "line", "new"),
    ("S", "FALLBACK_HOLD", "FALLBACK_HOLD = ", "line", "new"),
    ("S", "full_order_sha", "the Full-order sha256", "line", "new"),
    ("S", "DRAFT_HEAD_BLOCK_ORDER", "DRAFT_HEAD_BLOCK_ORDER = (", "paren", "new"),
    ("S", "pruned_mode", "def pruned_mode() -> str:", "py", "new"),
    (
        "S",
        "kept_blocks",
        "def kept_blocks(vocab: int) -> list[int] | None:",
        "py",
        "new",
    ),
    ("S", "PrunedHeadPolicy", "class PrunedHeadPolicy:", "py", "new"),
    ("Q", "HEAD_TOP_K", "HEAD_TOP_K = ", "line", "old"),
    (
        "Q",
        "kept_check",
        "if (vocab % 128 or not kb or len(kb) * 128 < HEAD_TOP_K",
        "lines:4",
        "new",
    ),
    ("Q", "kept_set", "kept = set(kb)", "line", "new"),
    (
        "Q",
        "keep_cols",
        "self.keep_cols = vocab if kept is None else 128 * len(kept)",
        "line",
        "new",
    ),
    (
        "Q",
        "id_map",
        "ids = torch.arange(vocab, dtype = torch.int64).view(-1, 128)",
        "lines:4",
        "new",
    ),
    (
        "Q",
        "topk_slice",
        "n = self.keep_cols if pruned else self.vocab",
        "lines:4",
        "new",
    ),
    ("D", "HEAD_TOPK", "#define HEAD_TOPK ", "line", "old"),
    ("D", "better", "__device__ __forceinline__ bool better(", "brace", "old"),
    ("D", "tv", "tv[e] = id < vocab ? val : -INFINITY;", "line", "old"),
    (
        "D",
        "ti",
        "ti[e] = id < vocab ? (id_map ? id_map[id] : id) : INT_MAX;",
        "line",
        "new",
    ),
    ("A", "begin_job", "def begin_job(self):", "py", "new"),
    (
        "A",
        "observe",
        "self.head_policy.observe_anchor(int(input_ids[0, -1]))",
        "line",
        "new",
    ),
    (
        "J",
        "begin_job_call",
        'begin_job = getattr(self.generator.draft_model, "begin_job", None)',
        "lines:3",
        "new",
    ),
]
# the lines the constants are parsed from (exact text, stripped)
CONST_LINES: dict[str, tuple[str, object]] = {
    "DRAFT_HEAD_VOCAB": ("DRAFT_HEAD_VOCAB = 248320", 248320),
    "DRAFT_HEAD_BLOCK": ("DRAFT_HEAD_BLOCK = 128", 128),
    "DRAFT_HEAD_BLOCKS_ALLOWED": (
        "DRAFT_HEAD_BLOCKS_ALLOWED = (896, 1024)",
        (896, 1024),
    ),
    "DRAFT_HEAD_BLOCKS_DEFAULT": ("DRAFT_HEAD_BLOCKS_DEFAULT = 896", 896),
    "FALLBACK_WINDOW": ("FALLBACK_WINDOW = 32", 32),
    "FALLBACK_TRIGGER": ("FALLBACK_TRIGGER = 2", 2),
    "FALLBACK_HOLD": ("FALLBACK_HOLD = 128", 128),
    "HEAD_TOP_K": (
        "HEAD_TOP_K = 16                             # the kernel's fused top-k",
        16,
    ),
    "HEAD_TOPK": ("#define HEAD_TOPK 16", 16),
}
# Bend model defs holding those constants (def name -> source constant)
BEND_CONSTS = {
    "idm_vocab": "DRAFT_HEAD_VOCAB",
    "idm_block": "DRAFT_HEAD_BLOCK",
    "idm_n_default": "DRAFT_HEAD_BLOCKS_DEFAULT",
    "idm_window": "FALLBACK_WINDOW",
    "idm_trigger": "FALLBACK_TRIGGER",
    "idm_hold_len": "FALLBACK_HOLD",
    "idm_topk_k": "HEAD_TOPK",
}
ALLOWED_BLOCKS = (896, 1024)
ORDER_LEN = 1024
FULL_ORDER_LEN = 1940
# value seeds (a, b, m): token t has value (a t + b) % m; must equal the table's S rows
SEEDS = [
    (40503, 12345, 1000003),
    (1, 0, 20000),
    (1, 0, 16777216),
    (16777213, 977, 16777216),
]
# the policy scenario of the static (non-adaptive) mode in the Bend table
STATIC_SCENARIO = 3
MISMATCH_WIDTH = 200
MISMATCHES_SHOWN = 3
SUMMARY_WIDTH = 160
# an option and its value (--order-json FILE, --mutate NAME, --all-mutations TREE)
OPTION_ARGS = 2

# name -> (file key, quoted block, text, replacement): source mutations, each must be
# rejected by this diff
MUTATIONS = {
    # a non-increasing kept-block list: the table prefix unsorted
    "unsorted_kept": (
        "S",
        "kept_blocks",
        "blocks = sorted(DRAFT_HEAD_BLOCK_ORDER[:n])",
        "blocks = list(DRAFT_HEAD_BLOCK_ORDER[:n])",
    ),
    # reset() leaves the hold pending
    "reset_keeps_hold": (
        "S",
        "PrunedHeadPolicy",
        (
            "        self.recent = []\n        self.hold = 0\n"
            "        self.use_pruned = True\n"
        ),
        "        self.recent = []\n        self.use_pruned = True\n",
    ),
    # a 31-round window
    "window_31": (
        "S",
        "PrunedHeadPolicy",
        "del self.recent[:-FALLBACK_WINDOW]",
        "del self.recent[:-(FALLBACK_WINDOW - 1)]",
    ),
    # the pruned head returns one round early
    "hold_ends_early": (
        "S",
        "PrunedHeadPolicy",
        "self.use_pruned = self.hold == 0",
        "self.use_pruned = self.hold <= 1",
    ),
    # anchors past the vocabulary count as evidence
    "big_is_evidence": (
        "S",
        "PrunedHeadPolicy",
        "0 <= b < len(self.kept) and not self.kept[b]",
        "0 <= b and (b >= len(self.kept) or not self.kept[b])",
    ),
    # the other ids first in the id map
    "rest_first": (
        "Q",
        "id_map",
        "torch.cat((ids[mask].flatten(), ids[~mask].flatten()))",
        "torch.cat((ids[~mask].flatten(), ids[mask].flatten()))",
    ),
    # one block too many in the pruned call
    "keep_cols_plus": ("Q", "keep_cols", "128 * len(kept)", "128 * len(kept) + 128"),
    # the epilogue returns column ids instead of token ids
    "ti_column_id": ("D", "ti", "(id_map ? id_map[id] : id)", "(id)"),
    # ties to the higher id
    "better_tie_high": ("D", "better", "ai < bi", "ai > bi"),
}
# name -> (Bend file, text, replacement): model mutations, each must be rejected by
# `bend draft_head_idmap_proof.bend`
LAW_MUTATIONS = {
    # a non-increasing kept-block list: kept_blocks without the sort
    "model_unsorted_kept": (
        IMPL,
        "  idm_sort(List.take(&2, Nat, order, n))",
        "  List.take(&2, Nat, order, n)",
    ),
    # reset() leaving hold != 0
    "model_reset_keeps_hold": (
        IMPL,
        (
            "def idm_reset(st: Spec.IdmPol) -> Spec.IdmPol:\n"
            "  Spec.IdmPol{Nil{}, 0n, True{}}"
        ),
        (
            "def idm_reset(st: Spec.IdmPol) -> Spec.IdmPol:\n"
            "  Spec.IdmPol{Nil{}, 1n, True{}}"
        ),
    ),
}


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"draft_head_idmap_diff: FAIL {msg}"
    raise SystemExit(text)


def say(text: str) -> None:
    """Print one line of the report and flush it."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def unlimited_vm() -> None:
    """Lift a soft RLIMIT_AS: the Bend runtime reserves its heap up front.

    Harness shells set 8 GB; processes started afterwards inherit the lifted limit.
    """
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def indent(line: str) -> int:
    """Return the indentation width of a line.

    Returns:
        The number of leading spaces.

    """
    return len(line) - len(line.lstrip(" "))


def _py_block(src: list[str], i: int) -> int:
    j, base = i + 1, indent(src[i])
    while j < len(src) and (not src[j].strip() or indent(src[j]) > base):
        j += 1
    while not src[j - 1].strip():
        j -= 1
    return j


def _paren_block(src: list[str], i: int, name: str) -> int:
    depth = 0
    for j in range(i, len(src)):
        depth += src[j].count("(") - src[j].count(")")
        if depth == 0:
            return j + 1
    fail(f"{name}: unbalanced parentheses from line {i + 1}")


def _brace_block(src: list[str], i: int, name: str) -> int:
    depth, opened = 0, False
    for j in range(i, len(src)):
        for ch in src[j].split("//")[0]:
            if ch == "{":
                depth, opened = depth + 1, True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return j + 1
    fail(f"{name}: unbalanced braces from line {i + 1}")


def extract(src: list[str], name: str, sig: str, kind: str) -> tuple[int, int]:
    """Locate the block whose first stripped line starts with sig (exactly one).

    Returns:
        The block's 1-based inclusive line range.

    """
    if name == "full_order_sha":
        marker = "Full-order sha256 (json of all 1940 blocks):"
        hits = [i for i, line in enumerate(src) if marker in line]
    else:
        hits = [i for i, line in enumerate(src) if line.strip().startswith(sig)]
    if len(hits) != 1:
        fail(f"{name}: signature {sig!r} found {len(hits)} times")
    i = hits[0]
    if kind == "line":
        return i + 1, i + 1
    if kind.startswith("lines:"):
        return i + 1, i + int(kind.removeprefix("lines:"))
    if kind == "py":
        return i + 1, _py_block(src, i)
    if kind == "paren":
        return i + 1, _paren_block(src, i, name)
    return i + 1, _brace_block(src, i, name)


def patch_plus(patch: str) -> dict[str, set[str]]:
    """Return, per file, the set of `+` lines (without their prefix).

    Returns:
        The lines by file.

    """
    plus: dict[str, set[str]] = {}
    cur = None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            cur = line[6:]
            plus.setdefault(cur, set())
        elif line.startswith("--- "):
            continue
        elif cur and line.startswith("+"):
            plus[cur].add(line[1:])
    return plus


# ---- a numpy stand-in for the torch calls of Q's quoted statements ----

type Array = npt.NDArray[np.generic]
type DType = type[np.generic] | None


class _T:
    """A torch tensor stand-in over a numpy array."""

    def __init__(self, a: object) -> None:
        """Wrap `a` as an array."""
        self.a: Array = np.asarray(a)

    def view(self, *shape: int) -> _T:
        """Return the reshaped tensor.

        Returns:
            The view.

        """
        return _T(self.a.reshape(*shape))

    def flatten(self) -> _T:
        """Return the tensor as one dimension.

        Returns:
            The flat tensor.

        """
        return _T(self.a.reshape(-1))

    def index(self) -> npt.NDArray[np.bool_] | npt.NDArray[np.intp]:
        """Return the tensor as a numpy index: a boolean mask or integer indices.

        Returns:
            The index array.

        """
        if self.a.dtype == np.bool_:
            return self.a.astype(np.bool_, copy=False)
        if not np.issubdtype(self.a.dtype, np.integer):
            fail(f"tensor of dtype {self.a.dtype} used as an index")
        return self.a.astype(np.intp, copy=False)

    def __getitem__(self, k: _T | slice) -> _T:
        """Return the indexed elements (a mask / index tensor or a slice).

        Returns:
            The elements.

        """
        if isinstance(k, slice):
            return _T(self.a[k])
        return _T(self.a[k.index()])

    def __setitem__(self, k: _T, v: bool) -> None:
        """Set the elements an index tensor selects."""
        self.a[k.index()] = v

    def __invert__(self) -> _T:
        """Return the elementwise negation.

        Returns:
            The negated tensor.

        """
        return _T(np.invert(self.a))

    def to(self, device: object = None, dtype: DType = None) -> _T:
        """Return the tensor converted to `dtype` (the device is ignored).

        Returns:
            The converted tensor.

        """
        del device
        return _T(self.a.astype(dtype) if dtype is not None else self.a)


def _torch_arange(n: int, dtype: DType = None) -> _T:
    return _T(np.arange(n, dtype=dtype))


def _torch_zeros(n: int, dtype: DType = None) -> _T:
    return _T(np.zeros(n, dtype=dtype))


def _torch_tensor(x: object) -> _T:
    return _T(np.array(x))


def _torch_cat(ts: tuple[_T, ...]) -> _T:
    return _T(np.concatenate([t.a for t in ts]))


TORCH = SimpleNamespace(
    int64=np.int64,
    int32=np.int32,
    bool=np.bool_,
    arange=_torch_arange,
    zeros=_torch_zeros,
    tensor=_torch_tensor,
    cat=_torch_cat,
)


def run_block(code: str, ns: dict[str, object]) -> None:
    """Run a quoted statement block (dedented) in `ns`."""
    pysubset.exec_block(textwrap.dedent(code), ns)


@dataclass(frozen=True)
class SModule:
    """S executed as a module: its namespace and the environment its os stub reads."""

    ns: dict[str, object]
    environ: dict[str, str]

    def set_environ(self, **kv: str | None) -> None:
        """Set (or unset, value None) the variables of the environment S reads."""
        self.environ.clear()
        self.environ.update({k: v for k, v in kv.items() if v is not None})

    def call(self, name: str, *args: object) -> object:
        """Call the module function `name`.

        Returns:
            Its result.

        """
        fn = self.ns[name]
        if not callable(fn):
            fail(f"S.{name} is not callable")
        return fn(*args)


def load_s(text: str) -> SModule:
    """Execute S as a module; its `import os` binds a stub with an environ mapping.

    Returns:
        The module.

    """
    environ: dict[str, str] = {}
    ns: dict[str, object] = {"__name__": "dflash2_head_blocks"}
    pysubset.exec_block(text, ns, modules={"os": SimpleNamespace(environ=environ)})
    return SModule(ns=ns, environ=environ)


def as_int(value: object, what: str) -> int:
    """Return `value` as an int.

    Returns:
        The value.

    """
    if not isinstance(value, int):
        fail(f"{what} is not an int: {value!r}")
    return value


def as_tensor(value: object, what: str) -> _T:
    """Return `value` as a stand-in tensor.

    Returns:
        The tensor.

    """
    if not isinstance(value, _T):
        fail(f"{what} is not a tensor: {value!r}")
    return value


@dataclass(frozen=True)
class Head:
    """Q's kept check, keep_cols, id map and top-k calls for one kept-block list."""

    ok: bool
    keep_cols: int = 0
    id_map: Array | None = None
    call_full: tuple[int, Array] | None = None
    call_pruned: tuple[int, Array] | None = None


def topk_call(
    q: dict[str, str], ns: dict[str, object], *, pruned: bool
) -> tuple[int, Array]:
    """Run Q's topk slicing statements for one call.

    Returns:
        n and the id map im[:n] of the call.

    """
    ns["pruned"] = pruned
    run_block(q["topk_slice"], ns)
    im = as_tensor(ns["im"], "im").a.astype(np.int64)
    return as_int(ns["n"], "n"), im


def q_head(q: dict[str, str], vocab: int, kb: list[int]) -> Head:
    """Run Q's kept check, kept set, keep_cols and id map for kb.

    Returns:
        The head; ok = the check did not raise.

    """
    head = SimpleNamespace(kept_blocks=list(kb), vocab=vocab, id_map=None)
    ns: dict[str, object] = {
        "torch": TORCH,
        "vocab": vocab,
        "kb": list(kb),
        "dev": "cpu",
        "self": head,
    }
    run_block(q["HEAD_TOP_K"], ns)
    try:
        run_block(q["kept_check"], ns)
    except ValueError:
        return Head(ok=False)
    run_block(q["kept_set"], ns)
    run_block(q["keep_cols"], ns)
    run_block(q["id_map"], ns)
    keep_cols = as_int(head.keep_cols, "self.keep_cols")
    id_map = as_tensor(head.id_map, "self.id_map").a.astype(np.int64)
    calls = []
    for pruned in (False, True):
        head.wq = np.zeros(vocab // 16)
        head.scales = np.zeros(vocab // 16)
        calls.append(topk_call(q, ns, pruned=pruned))
    return Head(
        ok=True,
        keep_cols=keep_cols,
        id_map=id_map,
        call_full=calls[0],
        call_pruned=calls[1],
    )


def q_check(q: dict[str, str], vocab: int, kb: list[int]) -> bool:
    """Run Q's kept check alone.

    Returns:
        Whether it accepts kb.

    """
    ns: dict[str, object] = {"vocab": vocab, "kb": list(kb)}
    run_block(q["HEAD_TOP_K"], ns)
    try:
        run_block(q["kept_check"], ns)
    except ValueError:
        return False
    return True


def int_list(value: object, what: str) -> list[int]:
    """Return `value` as a list of ints.

    Returns:
        The list.

    """
    if not (isinstance(value, list) and all(isinstance(x, int) for x in value)):
        fail(f"{what} is not a list of ints: {value!r}")
    return [x for x in value if isinstance(x, int)]


def s_kept_rows(
    s_text: str, order: list[int], n: int, vocab: int
) -> tuple[bool, list[int] | None]:
    """Run kept_blocks(vocab) with the module's table replaced by order and N = n.

    Returns:
        (False, None) on any exception, else (True, the blocks).

    """
    s = load_s(s_text)
    s.ns["DRAFT_HEAD_BLOCK_ORDER"] = tuple(order)
    s.ns["DRAFT_HEAD_VOCAB"] = vocab
    s.ns["DRAFT_HEAD_BLOCKS_ALLOWED"] = (n,)
    s.set_environ(EXL3_DRAFT_HEAD_PRUNED=None, EXL3_DRAFT_HEAD_BLOCKS=str(n))
    try:
        blocks = s.call("kept_blocks", vocab)
    except (ValueError, IndexError):
        return False, None
    return True, None if blocks is None else int_list(blocks, "kept_blocks")


C_PRELUDE = (
    "// generated by draft_head_idmap_diff.py: quoted dflash2_head.cu better() and "
    "epilogue + reference top-16\n"
    "#include <algorithm>\n"
    "#include <climits>\n"
    "#include <cmath>\n"
    "#include <cstdint>\n"
    "#include <cstdio>\n"
    "#include <cstdlib>\n"
    "#include <vector>\n"
    "#define __device__\n"
    "#define __forceinline__ inline\n"
    "\n"
    "// D: quoted\n"
)
C_SEEDS = "\nstruct Seed { uint64_t a, b, m; };\nstatic const Seed seeds[] = {\n    "
C_CALL = (
    "\n"
    "};\n"
    "\n"
    "static float value_of(const Seed& s, int tok) { return (float) ((s.a * "
    "(uint64_t) tok + s.b) % s.m); }\n"
    "\n"
    "static long long pad_cols = 0;\n"
    "\n"
    "// One dflash2_q4_head_topk call over `vocab` columns (id_map = nullptr: "
    "identity): column id carries the head column\n"
    "// of its token, so its value is that token's value; tv / ti are the kernel's "
    "quoted epilogue statements.\n"
    "static std::vector<std::pair<float, int>> call(const Seed& s, int vocab, "
    "const int* id_map)\n"
    "{\n"
    "    const int n_tiles = (vocab + 15) / 16;\n"
    "    std::vector<std::pair<float, int>> c;\n"
    "    for (int id = 0; id < n_tiles * 16; ++id)\n"
    "    {\n"
    "        const int e = 0;\n"
    "        float tv[4]; int ti[4];\n"
    "        const int tok = id < vocab ? (id_map ? id_map[id] : id) : 0;\n"
    "        const float val = value_of(s, tok);\n"
    "        "
)
C_MAIN = (
    "\n"
    "        pad_cols += id >= vocab;\n"
    "        c.push_back({tv[e], ti[e]});\n"
    "    }\n"
    "    std::sort(c.begin(), c.end(), [](const std::pair<float, int>& x, "
    "const std::pair<float, int>& y)\n"
    "              { return better(x.first, x.second, y.first, y.second); });\n"
    "    return c;\n"
    "}\n"
    "\n"
    "static void print(int N, int si, const char* kind, "
    "const std::vector<std::pair<float, int>>& c)\n"
    "{\n"
    '    printf("T %d %d %s", N, si, kind);\n'
    "    for (int i = 0; i < HEAD_TOPK && i < (int) c.size(); ++i) "
    'printf(" %lld:%d", (long long) c[i].first, c[i].second);\n'
    '    printf("\\n");\n'
    "}\n"
    "\n"
    "static std::vector<int> read_ints(FILE* f)\n"
    "{\n"
    "    long long n = 0;\n"
    '    if (fscanf(f, "%lld", &n) != 1) '
    '{ fprintf(stderr, "bad input\\n"); exit(2); }\n'
    "    std::vector<int> v(n);\n"
    '    for (long long i = 0; i < n; ++i) if (fscanf(f, "%d", &v[i]) != 1) '
    '{ fprintf(stderr, "bad input\\n"); exit(2); }\n'
    "    return v;\n"
    "}\n"
    "\n"
    "int main(int argc, char** argv)\n"
    "{\n"
    '    FILE* f = fopen(argv[1], "r");\n'
    "    long long bad = 0, checks = 0;\n"
    "    for (int run = 0; run < "
)


def c_program(q: dict[str, str], v: int, runs: list[tuple[int, str]]) -> str:
    """Return the C++ reference program around D's quoted lines.

    Returns:
        The program source.

    """
    seeds = ",\n    ".join(f"{{{a}ull, {b}ull, {m}ull}}" for a, b, m in SEEDS)
    return (
        C_PRELUDE
        + f"{q['HEAD_TOPK']}\n{q['better']}\n"
        + C_SEEDS
        + seeds
        + C_CALL
        + f"{q['tv'].strip()}\n        {q['ti'].strip()}"
        + C_MAIN
        + f"{len(runs)}; ++run)\n"
        "    {\n"
        "        int N = 0, n_pruned = 0;\n"
        '        if (fscanf(f, "%d %d", &N, &n_pruned) != 2) return 2;\n'
        "        std::vector<int> full = read_ints(f), pruned = read_ints(f);\n"
        f"        if ((int) full.size() != {v} || (int) pruned.size() != n_pruned) "
        '{ fprintf(stderr, "bad sizes\\n"); return 2; }\n'
        f"        std::vector<char> in_prefix({v}, 0);\n"
        "        for (int t : pruned) in_prefix[t] = 1;\n"
        "        for (int si = 0; si < (int) (sizeof(seeds) / sizeof(seeds[0])); "
        "++si)\n"
        "        {\n"
        f"            auto id = call(seeds[si], {v}, nullptr);\n"
        f"            auto fu = call(seeds[si], {v}, full.data());\n"
        "            auto pr = call(seeds[si], n_pruned, pruned.data());\n"
        '            print(N, si, "id", id);\n'
        '            print(N, si, "full", fu);\n'
        '            print(N, si, "pruned", pr);\n'
        "            // C side: the full call on the permuted copy is the identity "
        "call (whole ranking), and the prefix call\n"
        "            // is the identity ranking restricted to the prefix's tokens\n"
        "            ++checks;\n"
        "            if (fu != id) { ++bad; "
        'fprintf(stderr, "FAIL full != id (N %d seed %d)\\n", N, si); }\n'
        "            std::vector<std::pair<float, int>> r;\n"
        "            for (auto& x : id) if (in_prefix[x.second]) r.push_back(x);\n"
        "            ++checks;\n"
        "            if (pr != r) { ++bad; fprintf(stderr, "
        '"FAIL pruned != restricted identity ranking (N %d seed %d)\\n", N, si); }\n'
        "        }\n"
        "    }\n"
        '    fprintf(stderr, "C-side checks: %lld comparisons (whole rankings), '
        '%lld padding columns, %lld failures\\n", checks, pad_cols, bad);\n'
        "    return bad || pad_cols ? 1 : 0;\n"
        "}\n"
    )


def bend_literals(impl: str) -> tuple[dict[str, int], list[int]]:
    """Return the Bend model's constant defs and its idm_order literal.

    Returns:
        The constants by def name and the order.

    """
    consts: dict[str, int] = {}
    for d in BEND_CONSTS:
        m = re.search(rf"\ndef {d}\(\) -> Nat:\n  (\d+)n\n", impl)
        if not m:
            fail(f"{IMPL}: {d} is not a Nat literal")
        consts[d] = int(m.group(1))
    m = re.search(r"\ndef idm_order\(\) -> List<&2, Nat>:\n  \[([\dn,\s]+)\]\n", impl)
    if not m:
        fail(f"{IMPL}: idm_order is not a list literal")
    return consts, [int(x.strip().rstrip("n")) for x in m.group(1).split(",")]


def parse_order(block: str) -> list[int]:
    """Return the block numbers of the quoted DRAFT_HEAD_BLOCK_ORDER.

    Returns:
        The order.

    """
    body = block.split("(", 1)[1].rsplit(")", 1)[0]
    return [int(x) for x in re.findall(r"\d+", body)]


@runtime_checkable
class Policy(Protocol):
    """The PrunedHeadPolicy fields and methods the replay uses."""

    hold: int
    recent: list[bool]
    use_pruned: bool

    def reset(self) -> None:
        """Start a new job."""
        ...

    def observe_anchor(self, token: int) -> None:
        """Observe one round's anchor."""
        ...


def new_policy(s: SModule, kept: list[int], vocab: int, sc: int) -> Policy:
    """Return a PrunedHeadPolicy for scenario `sc`.

    Returns:
        The policy.

    """
    cls = s.ns["PrunedHeadPolicy"]
    if not callable(cls):
        fail("PrunedHeadPolicy is not a class")
    p = cls(kept, vocab, adaptive=sc != STATIC_SCENARIO)
    if not isinstance(p, Policy):
        fail("PrunedHeadPolicy lacks hold / recent / use_pruned / reset / observe")
    return p


@dataclass(frozen=True)
class Replay:
    """A policy replay: the module, its vocabulary and the scenario coverage."""

    s: SModule
    vocab: int
    cov: dict[str, int]

    def observe(self, p: Policy, t: int, sc: int) -> None:
        """Observe one anchor and count the scenarios it covers."""
        cov = self.cov
        hold0, recent0 = p.hold, list(p.recent)
        cov["big"] += t >= self.vocab
        cov["negative"] += t < 0
        p.observe_anchor(t)
        cov["trigger"] += p.hold == self.s.ns["FALLBACK_HOLD"]
        cov["hold_done"] += hold0 == 1 and p.hold == 0 and p.use_pruned
        cov["static_rounds"] += sc == STATIC_SCENARIO
        cov["window_expired"] += (
            hold0 == 0
            and len(recent0) == self.s.ns["FALLBACK_WINDOW"]
            and recent0[0]
            and p.hold == 0
            and sum(p.recent) == 1
            and p.recent[-1]
        )


def replay_policy(
    s_text: str, kept: list[int], vocab: int, p_lines: list[str]
) -> tuple[list[str], dict[str, int]]:
    """Replay the table's P rows through the tree's PrunedHeadPolicy.

    Returns:
        The reference rows and the scenario coverage.

    """
    s = load_s(s_text)
    out: list[str] = []
    pols: dict[int, Policy] = {}
    replay = Replay(
        s=s,
        vocab=vocab,
        cov={
            "trigger": 0,
            "reset_in_hold": 0,
            "hold_done": 0,
            "big": 0,
            "negative": 0,
            "static_rounds": 0,
            "window_expired": 0,
        },
    )
    for line in p_lines:
        f = line.split(" ")
        sc, i, ev = int(f[1]), int(f[2]), f[3]
        if sc not in pols:
            pols[sc] = new_policy(s, kept, vocab, sc)
        p = pols[sc]
        if ev == "r":
            replay.cov["reset_in_hold"] += p.hold > 0
            p.reset()
        else:
            replay.observe(p, int(ev), sc)
        out.append(
            f"P {sc} {i} {ev} {int(p.use_pruned)} {p.hold} {len(p.recent)} "
            f"{sum(p.recent)}"
        )
    return out, replay.cov


def run_law_mutation(name: str) -> tuple[bool, str]:
    """Apply a model mutation in a scratch copy of the Bend sources.

    Returns:
        Whether the proof gate rejects it, and its summary.

    """
    fname, a, b = LAW_MUTATIONS[name]
    with tempfile.TemporaryDirectory() as td:
        for p in HERE.glob("*.bend"):
            shutil.copy(p, Path(td) / p.name)
        t = (Path(td) / fname).read_text(encoding="utf-8")
        if t.count(a) != 1:
            fail(f"law mutation {name} does not apply exactly once")
        (Path(td) / fname).write_text(t.replace(a, b), encoding="utf-8")
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend + PROOF.bend in the mutation tempdir, no shell
            source_link.locked([source_link.bend(), PROOF]),
            cwd=td,
            capture_output=True,
            text=True,
            check=False,
        )
    out = (r.stdout + r.stderr).strip().splitlines()
    loc = next((x.strip() for x in out if x.startswith("Location")), "")
    where = next((x.strip() for x in out if ">|" in x), "")
    first = out[0] if out else ""
    return r.returncode != 0, f"exit {r.returncode}; {first} {loc} {where}".strip()


def mutation_summary(lines: list[str]) -> str:
    """Return the summary of a rejected source mutation's output.

    Returns:
        Its failure, C-side and first-mismatch lines.

    """
    fin = next((x for x in lines if x.startswith("draft_head_idmap_diff: FAIL")), "")
    cside = next((x for x in lines if x.startswith("C-side checks")), "")
    mm = next((i for i, x in enumerate(lines) if x.startswith("MISMATCH")), None)
    tab = (
        " / ".join(x.strip()[:SUMMARY_WIDTH] for x in lines[mm : mm + 3])
        if mm is not None
        else "table IDENTICAL"
    )
    return " | ".join(x for x in (fin, cside, tab) if x)


def run_all_mutations(tree: str) -> int:
    """Run every source mutation (each must fail) and every law mutation.

    Returns:
        The exit status: 0 if every mutation was rejected.

    """
    rc = 0
    for name in MUTATIONS:
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: sys.executable re-running this script with a fixed mutation name, no shell
            [sys.executable, __file__, "--mutate", name, tree],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            say(f"{name}: SURVIVED (unexpected)")
            rc = 1
            continue
        lines = (r.stdout + r.stderr).splitlines()
        say(f"{name}: REJECTED: " + mutation_summary(lines))
    for name in LAW_MUTATIONS:
        rejected, why = run_law_mutation(name)
        verdict = "REJECTED by the proof gate" if rejected else "SURVIVED (unexpected)"
        say(f"{name}: {verdict}: {why}")
        rc |= not rejected
    outcome = "as expected" if rc == 0 else "UNEXPECTED OUTCOME"
    sys.stdout.write(f"draft_head_idmap_diff --all-mutations: {outcome}\n")
    return rc


@dataclass(frozen=True)
class Cli:
    """The command line of a single run."""

    tree: Path
    order_json: Path | None
    mutate: str | None


def parse_cli(argv: list[str]) -> Cli:
    """Parse the command line (--all-mutations runs and exits).

    Returns:
        The arguments of a single run.

    """
    args = argv[1:]
    mutate = None
    order_json = None
    if args[:1] == ["--order-json"]:
        if len(args) < OPTION_ARGS:
            fail(USAGE)
        order_json, args = Path(args[1]), args[2:]
    if args[:1] == ["--all-mutations"]:
        if len(args) != OPTION_ARGS:
            fail(USAGE)
        sys.exit(run_all_mutations(args[1]))
    if args[:1] == ["--mutate"]:
        if len(args) < OPTION_ARGS or args[1] not in MUTATIONS:
            fail(f"--mutate NAME, NAME in {sorted(MUTATIONS)}")
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail(USAGE)
    return Cli(tree=Path(args[0]), order_json=order_json, mutate=mutate)


def quote_blocks(tree: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Check the 9005c patch and quote every block of BLOCKS from the tree.

    Returns:
        The source texts by file key and the quoted blocks by name.

    """
    patch = PATCH_PATH.read_bytes()
    sha = hashlib.sha256(patch).hexdigest()
    sys.stdout.write(f"patch {PATCH_PATH} sha256 {sha}\n")
    if sha != PATCH_SHA256:
        fail(f"patch sha256 {sha} != pinned {PATCH_SHA256}")
    plus = patch_plus(patch.decode())
    text = {k: (tree / p).read_text(encoding="utf-8") for k, p in PATHS.items()}
    lines = {k: t.split("\n") for k, t in text.items()}
    q: dict[str, str] = {}
    n_plus = 0
    for f, name, sig, kind, origin in BLOCKS:
        a, b = extract(lines[f], name, sig, kind)
        body = lines[f][a - 1 : b]
        q[name] = "\n".join(body)
        tag = ""
        if origin == "new":
            nonblank = [x for x in body if x.strip()]
            miss = [x for x in nonblank if x not in plus.get(PATHS[f], set())]
            if miss:
                fail(
                    f"{f} {name} lines {a}-{b}: not `+` lines of the patch: {miss[:3]}"
                )
            n_plus += len(nonblank)
            tag = (
                f", {len(nonblank)}/{len(nonblank)} non-blank lines occur as `+` "
                "lines of the patch"
            )
        sys.stdout.write(f"quoted {f}:{a}-{b} {name}{tag}\n")
    sys.stdout.write(f"quoted 9005c lines verified as patch `+` lines: {n_plus}\n")
    return text, q


def check_constants(q: dict[str, str]) -> dict[str, object]:
    """Check the constants' exact source lines.

    Returns:
        The constants by name.

    """
    src_const: dict[str, object] = {}
    for name, (line, val) in CONST_LINES.items():
        if q[name].strip() != line:
            fail(f"{name}: source line {q[name].strip()!r} != expected {line!r}")
        src_const[name] = val
        sys.stdout.write(f"parsed {name} = {val!r} from {line!r}\n")
    return src_const


def check_model(q: dict[str, str], src_const: dict[str, object]) -> list[int]:
    """Check the Bend model's literals against the source constants and order.

    Returns:
        The source's block order.

    """
    order = parse_order(q["DRAFT_HEAD_BLOCK_ORDER"])
    impl = (HERE / IMPL).read_text(encoding="utf-8")
    bconst, border = bend_literals(impl)
    for d, name in BEND_CONSTS.items():
        if bconst[d] != src_const[name]:
            fail(f"{IMPL} {d} = {bconst[d]} != source {name} = {src_const[name]}")
    if (
        not re.search(r"\ndef idm_n_alt\(\) -> Nat:\n  1024n\n", impl)
        or src_const["DRAFT_HEAD_BLOCKS_ALLOWED"] != ALLOWED_BLOCKS
    ):
        fail("idm_n_alt / DRAFT_HEAD_BLOCKS_ALLOWED mismatch")
    if border != order or len(order) != ORDER_LEN or len(set(order)) != ORDER_LEN:
        fail(
            f"{IMPL} idm_order ({len(border)} entries) != source "
            f"DRAFT_HEAD_BLOCK_ORDER ({len(order)} entries)"
        )
    sys.stdout.write(
        f"Bend model constants {sorted(bconst.items())} and idm_order "
        f"({len(border)} entries) equal the source\n"
    )
    return order


def check_full_order(
    q: dict[str, str], order: list[int], order_json: Path | None
) -> None:
    """Check the full block order of --order-json against the source table and pin."""
    m = re.search(r"[0-9a-f]{64}", q["full_order_sha"])
    if m is None:
        fail("full_order_sha: no sha256 in the quoted line")
    pinned = m.group(0)
    if order_json is None:
        sys.stdout.write(
            "full order not given (--order-json): permutation check of the full "
            f"order not run (source pin {pinned})\n"
        )
        return
    loaded = json.loads(order_json.read_text(encoding="utf-8"))
    full = loaded["order"] if isinstance(loaded, dict) else None
    fsha = hashlib.sha256(json.dumps(full).encode()).hexdigest()
    blocks = [x for x in full if isinstance(x, int)] if isinstance(full, list) else []
    if (
        not isinstance(full, list)
        or len(blocks) != len(full)
        or sorted(blocks) != list(range(FULL_ORDER_LEN))
        or blocks[:ORDER_LEN] != order
        or fsha != pinned
    ):
        fail(
            f"{order_json}: not a permutation of 1940 blocks extending the source "
            f"table with sha256 {pinned} ({fsha})"
        )
    sys.stdout.write(
        f"full order {order_json}: permutation of the 1940 blocks, prefix = source "
        f"table, sha256 {fsha} = source pin\n"
    )


def apply_mutation(mutate: str | None, q: dict[str, str], text: dict[str, str]) -> str:
    """Apply a source mutation to the quoted block (and to S's text).

    Returns:
        S's text to execute.

    """
    s_text = text["S"]
    if mutate:
        fk, blk, a, b = MUTATIONS[mutate]
        if q[blk].count(a) != 1 or text[fk].count(a) != 1:
            fail(f"mutation {mutate} does not apply exactly once")
        q[blk] = q[blk].replace(a, b)
        if fk == "S":
            s_text = s_text.replace(a, b)
        sys.stdout.write(f"mutation {mutate}: {blk}: {a!r} -> {b!r}\n")
    return s_text


def bend_rows() -> list[str]:
    """Compile and run the Bend table.

    Returns:
        Its output lines.

    """
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "table"
        r = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: pinned bend 2.0.35 + repo .bend table, no shell
            source_link.locked([source_link.bend(), TABLE, "-o", str(exe)]),
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            fail(f"bend compile: {r.stdout}{r.stderr}")
        unlimited_vm()
        bres = subprocess.run([str(exe)], capture_output=True, text=True, check=False)  # ruff: ignore[subprocess-without-shell-equals-true]  argv: binary this script just built in its private temp dir, no shell
    if bres.returncode != 0:
        fail(f"Bend table exited {bres.returncode}: {bres.stderr}")
    return bres.stdout.split("\n")


@dataclass
class Reference:
    """The reference rows and the C program's inputs, built per allowed N."""

    rows: list[str]
    runs: list[tuple[int, str]]
    cfile: list[str]
    kept_by_n: dict[int, list[int]]


def kept_blocks_for(s: SModule, n_blocks: int, vocab: int) -> list[int]:
    """Run S's pruned_mode and kept_blocks with EXL3_DRAFT_HEAD_BLOCKS for N.

    Returns:
        The kept blocks.

    """
    blocks_env = (
        None
        if n_blocks == CONST_LINES["DRAFT_HEAD_BLOCKS_DEFAULT"][1]
        else str(n_blocks)
    )
    s.set_environ(EXL3_DRAFT_HEAD_PRUNED=None, EXL3_DRAFT_HEAD_BLOCKS=blocks_env)
    if s.call("pruned_mode") != "adaptive":
        fail("pruned_mode() default is not adaptive")
    try:
        kb = s.call("kept_blocks", vocab)
    except (ValueError, IndexError) as e:
        fail(f"kept_blocks({vocab}) with N = {n_blocks} raised {e!r}")
    return int_list(kb, "kept_blocks")


def head_rows(ref: Reference, q: dict[str, str], n_blocks: int, kb: list[int]) -> None:
    """Add the B / M / @T rows and the C program input of one N."""
    vocab = as_int(CONST_LINES["DRAFT_HEAD_VOCAB"][1], "DRAFT_HEAD_VOCAB")
    block = as_int(CONST_LINES["DRAFT_HEAD_BLOCK"][1], "DRAFT_HEAD_BLOCK")
    h = q_head(q, vocab, kb)
    kbs = " ".join(map(str, kb))
    if not h.ok:
        ref.rows.append(f"B {n_blocks} ok=1 head_ok=0 keep_cols=? " + kbs)
        sys.stdout.write(f"N {n_blocks}: Q4DraftHead rejects the kept blocks\n")
        return
    if h.id_map is None or h.call_full is None or h.call_pruned is None:
        fail(f"N {n_blocks}: incomplete head")
    ref.rows.append(f"B {n_blocks} ok=1 head_ok=1 keep_cols={h.keep_cols} " + kbs)
    m = h.id_map
    ref.rows += [
        f"M {n_blocks} {r} " + " ".join(map(str, m[r * block : (r + 1) * block]))
        for r in range(vocab // block)
    ]
    (nf, imf), (np_, imp) = h.call_full, h.call_pruned
    if nf != vocab or len(imf) != vocab:
        fail(f"N {n_blocks}: full call n = {nf}, |im| = {len(imf)}")
    ref.rows.append(f"@T {n_blocks}")
    ref.runs.append((n_blocks, "run"))
    ref.cfile.append(
        f"{n_blocks} {np_}\n{len(imf)} "
        + " ".join(map(str, imf))
        + f"\n{len(imp)} "
        + " ".join(map(str, imp))
        + "\n"
    )
    sys.stdout.write(
        f"N {n_blocks}: kept_blocks {len(kb)} blocks, keep_cols {h.keep_cols}, "
        f"id map {len(m)} ids, prefix call n = {np_} (|im[:n]| = {len(imp)})\n"
    )


def reference(q: dict[str, str], s_text: str, order: list[int]) -> Reference:
    """Build the K / O / S / B / M rows and the C program's inputs.

    Returns:
        The reference.

    """
    c = {name: val for name, (_, val) in CONST_LINES.items()}
    rows = [
        (
            f"K vocab={c['DRAFT_HEAD_VOCAB']} block={c['DRAFT_HEAD_BLOCK']} "
            f"n={c['DRAFT_HEAD_BLOCKS_DEFAULT']} n_alt={max(ALLOWED_BLOCKS)} "
            f"window={c['FALLBACK_WINDOW']} trigger={c['FALLBACK_TRIGGER']} "
            f"hold={c['FALLBACK_HOLD']} topk={c['HEAD_TOPK']}"
        ),
        "O " + " ".join(map(str, order)),
    ]
    rows += [f"S {i} {a} {b} {m}" for i, (a, b, m) in enumerate(SEEDS)]
    ref = Reference(rows=rows, runs=[], cfile=[], kept_by_n={})
    vocab = as_int(c["DRAFT_HEAD_VOCAB"], "DRAFT_HEAD_VOCAB")
    s = load_s(s_text)
    for n_blocks in ALLOWED_BLOCKS:
        kb = kept_blocks_for(s, n_blocks, vocab)
        ref.kept_by_n[n_blocks] = kb
        head_rows(ref, q, n_blocks, kb)
    return ref


def run_c(q: dict[str, str], ref: Reference) -> subprocess.CompletedProcess[str]:
    """Compile and run the C++ reference program on the id maps.

    Returns:
        The finished program (stdout: T rows, stderr: C-side checks).

    """
    cxx = shutil.which("c++")
    if cxx is None:
        fail("c++ is not on PATH")
    vocab = as_int(CONST_LINES["DRAFT_HEAD_VOCAB"][1], "DRAFT_HEAD_VOCAB")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "topk.cpp"
        c.write_text(c_program(q, vocab, ref.runs), encoding="utf-8")
        exe = Path(td) / "topk"
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: C++ compiler from the nix shell building the generated harness in a private temp dir, no shell
            source_link.locked([
                cxx,
                "-O2",
                "-std=c++17",
                "-Wall",
                "-o",
                str(exe),
                str(c),
            ]),
            check=True,
        )
        inp = Path(td) / "maps.txt"
        inp.write_text("".join(ref.cfile), encoding="utf-8")
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: topk harness built in a private tempdir + its input file, no shell
            [str(exe), str(inp)], capture_output=True, text=True, check=False
        )


def table_check_rows(q: dict[str, str], s_text: str, bl: list[str]) -> list[str]:
    """Return the reference H (Q's kept check) and R (S's kept_blocks) rows.

    Returns:
        The rows, for the table's fixed lists.

    """
    out: list[str] = []
    for line in bl:
        if line.startswith("H "):
            f = line.split(" ")
            vocab = int(f[2].split("=")[1])
            kb = [int(x) for x in f[5:]]
            ok = int(q_check(q, vocab, kb))
            out.append(
                f"H {f[1]} vocab={vocab} ok={ok} kb" + "".join(f" {x}" for x in kb)
            )
        elif line.startswith("R "):
            f = line.split(" ")
            n, vocab = int(f[2].split("=")[1]), int(f[3].split("=")[1])
            order_c = [int(x) for x in f[f.index("order") + 1 :]]
            ok, blocks = s_kept_rows(s_text, order_c, n, vocab)
            shown = blocks if ok and blocks is not None else sorted(order_c[:n])
            out.append(
                f"R {f[1]} n={n} vocab={vocab} ok={int(ok)} blocks"
                + "".join(f" {x}" for x in shown)
                + " order"
                + "".join(f" {x}" for x in order_c)
            )
    return out


def compare(full_ref: list[str], bl: list[str]) -> bool:
    """Print the row counts and the first mismatches.

    Returns:
        Whether the reference equals the Bend table.

    """
    kinds = {k: sum(x.startswith(k + " ") for x in bl) for k in "KOSBMTPHR"}
    sys.stdout.write(
        "table rows: Bend "
        + ", ".join(f"{k} {v}" for k, v in kinds.items())
        + f"; reference {len(full_ref) - 1}\n"
    )
    identical = full_ref == bl
    if identical:
        sys.stdout.write("IDENTICAL\n")
        return True
    shown = 0
    for i in range(max(len(full_ref), len(bl))):
        a = full_ref[i] if i < len(full_ref) else "<eof>"
        b = bl[i] if i < len(bl) else "<eof>"
        if a != b:
            sys.stdout.write(
                f"MISMATCH row {i}:\n  source: {a[:MISMATCH_WIDTH]}\n"
                f"  Bend:   {b[:MISMATCH_WIDTH]}\n"
            )
            shown += 1
            if shown == MISMATCHES_SHOWN:
                break
    return False


def merge_t_rows(rows: list[str], c_stdout: str) -> list[str]:
    """Replace each "@T N" placeholder row by the C program's T rows of N.

    Returns:
        The rows.

    """
    t_rows = [x for x in c_stdout.split("\n") if x]
    full_ref: list[str] = []
    for row in rows:
        if row.startswith("@T "):
            n_blocks = row.split()[1]
            full_ref += [t for t in t_rows if t.split()[1] == n_blocks]
        else:
            full_ref.append(row)
    return full_ref


def policy_rows(
    s_text: str, ref: Reference, bl: list[str], mutate: str | None
) -> list[str]:
    """Replay the table's P rows (kept = the N = 896 blocks) and check coverage.

    Returns:
        The reference P rows.

    """
    p_lines = [x for x in bl if x.startswith("P ")]
    vocab = as_int(CONST_LINES["DRAFT_HEAD_VOCAB"][1], "DRAFT_HEAD_VOCAB")
    default = as_int(
        CONST_LINES["DRAFT_HEAD_BLOCKS_DEFAULT"][1], "DRAFT_HEAD_BLOCKS_DEFAULT"
    )
    p_ref, cov = replay_policy(s_text, ref.kept_by_n[default], vocab, p_lines)
    sys.stdout.write(
        "policy replay coverage: "
        + ", ".join(f"{k} {v}" for k, v in cov.items())
        + "\n"
    )
    if not mutate and min(cov.values()) == 0:
        fail(f"policy scenarios do not cover every case: {cov}")
    return p_ref


def main(argv: list[str]) -> None:
    """Run the differential."""
    cli = parse_cli(argv)
    text, q = quote_blocks(cli.tree)
    order = check_model(q, check_constants(q))
    check_full_order(q, order, cli.order_json)
    s_text = apply_mutation(cli.mutate, q, text)
    bl = bend_rows()
    ref = reference(q, s_text, order)
    cres = run_c(q, ref)
    sys.stdout.write(cres.stderr)
    full_ref = merge_t_rows(ref.rows, cres.stdout)
    full_ref += policy_rows(s_text, ref, bl, cli.mutate)

    # the two checks on the table's fixed lists
    full_ref += table_check_rows(q, s_text, bl)
    full_ref.append("")

    identical = compare(full_ref, bl)
    if cres.returncode != 0 or not identical:
        fail(
            " + ".join(
                (["table MISMATCH"] if not identical else [])
                + (["C-side check failures"] if cres.returncode != 0 else [])
            )
        )
    sys.stdout.write("draft_head_idmap_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
