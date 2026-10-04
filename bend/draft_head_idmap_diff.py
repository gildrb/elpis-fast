#!/usr/bin/env python3
"""
Finite differential check of bend/draft_head_idmap.bend (pruned int4 draft head, ext patch 9005c) against the
patched engine tree's own Python and CUDA source, executed on the CPU.

From TREE: modules/arch_specific/dflash2_head_blocks.py = S (constants, DRAFT_HEAD_BLOCK_ORDER, pruned_mode,
kept_blocks, PrunedHeadPolicy), modules/arch_specific/dflash2_q4_head.py = Q (HEAD_TOP_K, the kept-block check,
kept = set(kb), keep_cols, the id_map statements, the topk slicing statements), exllamav3_ext/dflash2_head.cu = D
(HEAD_TOPK, better(), the epilogue's tv / ti statements), architecture/dflash2.py = A (begin_job -> reset, the
observe_anchor call), generator/job.py = J (begin_job at each job's prefill). Every block is located by its signature
(exactly once), quoted verbatim with its line range, and every quoted non-blank line of a 9005c block must be a `+`
line of the pinned 9005c patch (sha256 checked). The parsed constants and the 1024-entry block order must equal the
Bend model's literals (draft_head_idmap.bend); if --order-json FILE gives the DraftProj full order
(block_order_code.json) it must be a permutation of the 1940 blocks, extend the source table and match the source's
pinned full-order sha256.

Reference computations, each on the quoted source itself:
  - S is executed as a module (it imports only os): kept_blocks(248320) for N = 896 and 1024
    (EXL3_DRAFT_HEAD_BLOCKS), and kept_blocks with its module globals replaced by fixed small orders;
  - Q's statements are executed with a numpy stand-in for the torch calls they use (arange / view / zeros / tensor
    indexing / ~ / cat / to): the kept-block check (raises = rejected), keep_cols, the id map, and the topk slicing
    (n and im[:n] for pruned False / True);
  - D's better() and epilogue statements are compiled into a C++ program that computes, for four value seeds, the
    top-16 of the identity call, of the full call with Q's id map and of the prefix call with Q's im[:n]
    (std::sort by the quoted better(); the warp / block / grid merge network is not emulated), and checks on the C
    side that the full call equals the identity call and the prefix call equals the identity ranking restricted to
    the prefix's tokens;
  - PrunedHeadPolicy replays the Bend table's anchor / reset sequences.
The compiled Bend table (DRAFT_HEAD_IDMAP_TABLE.bend) must equal the reference output byte for byte, and the policy
replay must cover a trigger, a reset during a hold, a completed hold, out-of-range anchors and the static mode.
Differential evidence on finite instances, not a proof; the proof is draft_head_idmap_proof.bend.

Usage: python3 bend/draft_head_idmap_diff.py [--order-json FILE] [--mutate NAME | --all-mutations] TREE
  TREE: OUT/patched of bend/engine_trees.py.
  FILE: block_order_code.json, written by DraftProj's select_order.py from host corpora. It is not in the repo and
  the repo cannot make it again. Without it, the full-order check does not run (the output tells this).
Needs numpy. The dev shell does not provide numpy. From the repo root, run:
  nix develop --offline --no-write-lock-file -c nix shell --impure --expr 'let p = (builtins.getFlake
  "git+file://${toString ./.}").inputs.nixpkgs.legacyPackages.x86_64-linux; in p.python313.withPackages
  (ps: [ ps.numpy ])' -c python3 -I -B bend/draft_head_idmap_diff.py [--order-json FILE] TREE
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:
    raise SystemExit("draft_head_idmap_diff: FAIL numpy is not importable. Run the driver with the "
                     "python313.withPackages (ps: [ ps.numpy ]) command in the module docstring.") from None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

REPO = source_link.REPO
PATCH_NAME = "9005c-elpis-draft-q4-head-pruned-n896.patch"
PATCH_PATH = REPO / "patches/exl3-ext" / PATCH_NAME
PATCH_SHA256 = "e7a2952ff6485c74d8b443fdeb751b9961f2c834b60b0d2940af6e77b7ce4db9"
TABLE = "DRAFT_HEAD_IDMAP_TABLE.bend"
IMPL = "draft_head_idmap.bend"
PROOF = "draft_head_idmap_proof.bend"

PATHS = {"S": "modules/arch_specific/dflash2_head_blocks.py", "Q": "modules/arch_specific/dflash2_q4_head.py",
         "D": "exllamav3_ext/dflash2_head.cu", "A": "architecture/dflash2.py", "J": "generator/job.py"}

# (file key, name, signature prefix of the first stripped line, kind, origin). kinds: "line"; "py" = a Python def /
# class (the first line and every following line indented deeper, or blank); "paren" = a Python statement up to its
# balanced closing parenthesis; "lines:N" = N lines; "brace" = a C block to its matching brace. origin: "new" = a
# 9005c line (each non-blank line a `+` line of the patch), "old" = pre-9005c source (not checked).
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
    ("S", "kept_blocks", "def kept_blocks(vocab: int) -> list[int] | None:", "py", "new"),
    ("S", "PrunedHeadPolicy", "class PrunedHeadPolicy:", "py", "new"),
    ("Q", "HEAD_TOP_K", "HEAD_TOP_K = ", "line", "old"),
    ("Q", "kept_check", "if (vocab % 128 or not kb or len(kb) * 128 < HEAD_TOP_K", "lines:4", "new"),
    ("Q", "kept_set", "kept = set(kb)", "line", "new"),
    ("Q", "keep_cols", "self.keep_cols = vocab if kept is None else 128 * len(kept)", "line", "new"),
    ("Q", "id_map", "ids = torch.arange(vocab, dtype = torch.int64).view(-1, 128)", "lines:4", "new"),
    ("Q", "topk_slice", "n = self.keep_cols if pruned else self.vocab", "lines:4", "new"),
    ("D", "HEAD_TOPK", "#define HEAD_TOPK ", "line", "old"),
    ("D", "better", "__device__ __forceinline__ bool better(", "brace", "old"),
    ("D", "tv", "tv[e] = id < vocab ? val : -INFINITY;", "line", "old"),
    ("D", "ti", "ti[e] = id < vocab ? (id_map ? id_map[id] : id) : INT_MAX;", "line", "new"),
    ("A", "begin_job", "def begin_job(self):", "py", "new"),
    ("A", "observe", "self.head_policy.observe_anchor(int(input_ids[0, -1]))", "line", "new"),
    ("J", "begin_job_call", 'begin_job = getattr(self.generator.draft_model, "begin_job", None)', "lines:3", "new"),
]
# the lines the constants are parsed from (exact text, stripped)
CONST_LINES = {"DRAFT_HEAD_VOCAB": ("DRAFT_HEAD_VOCAB = 248320", 248320),
               "DRAFT_HEAD_BLOCK": ("DRAFT_HEAD_BLOCK = 128", 128),
               "DRAFT_HEAD_BLOCKS_ALLOWED": ("DRAFT_HEAD_BLOCKS_ALLOWED = (896, 1024)", (896, 1024)),
               "DRAFT_HEAD_BLOCKS_DEFAULT": ("DRAFT_HEAD_BLOCKS_DEFAULT = 896", 896),
               "FALLBACK_WINDOW": ("FALLBACK_WINDOW = 32", 32),
               "FALLBACK_TRIGGER": ("FALLBACK_TRIGGER = 2", 2),
               "FALLBACK_HOLD": ("FALLBACK_HOLD = 128", 128),
               "HEAD_TOP_K": ("HEAD_TOP_K = 16                             # the kernel's fused top-k", 16),
               "HEAD_TOPK": ("#define HEAD_TOPK 16", 16)}
# Bend model defs holding those constants (def name -> source constant)
BEND_CONSTS = {"idm_vocab": "DRAFT_HEAD_VOCAB", "idm_block": "DRAFT_HEAD_BLOCK", "idm_n_default": "DRAFT_HEAD_BLOCKS_DEFAULT",
               "idm_window": "FALLBACK_WINDOW", "idm_trigger": "FALLBACK_TRIGGER", "idm_hold_len": "FALLBACK_HOLD",
               "idm_topk_k": "HEAD_TOPK"}
# value seeds (a, b, m): token t has value (a t + b) % m; must equal the table's S rows
SEEDS = [(40503, 12345, 1000003), (1, 0, 20000), (1, 0, 16777216), (16777213, 977, 16777216)]

# name -> (file key, quoted block, text, replacement): source mutations, each must be rejected by this diff
MUTATIONS = {
    # a non-increasing kept-block list: the table prefix unsorted
    "unsorted_kept": ("S", "kept_blocks", "blocks = sorted(DRAFT_HEAD_BLOCK_ORDER[:n])", "blocks = list(DRAFT_HEAD_BLOCK_ORDER[:n])"),
    # reset() leaves the hold pending
    "reset_keeps_hold": ("S", "PrunedHeadPolicy", "        self.recent = []\n        self.hold = 0\n        self.use_pruned = True\n",
                         "        self.recent = []\n        self.use_pruned = True\n"),
    # a 31-round window
    "window_31": ("S", "PrunedHeadPolicy", "del self.recent[:-FALLBACK_WINDOW]", "del self.recent[:-(FALLBACK_WINDOW - 1)]"),
    # the pruned head returns one round early
    "hold_ends_early": ("S", "PrunedHeadPolicy", "self.use_pruned = self.hold == 0", "self.use_pruned = self.hold <= 1"),
    # anchors past the vocabulary count as evidence
    "big_is_evidence": ("S", "PrunedHeadPolicy", "0 <= b < len(self.kept) and not self.kept[b]",
                        "0 <= b and (b >= len(self.kept) or not self.kept[b])"),
    # the other ids first in the id map
    "rest_first": ("Q", "id_map", "torch.cat((ids[mask].flatten(), ids[~mask].flatten()))",
                   "torch.cat((ids[~mask].flatten(), ids[mask].flatten()))"),
    # one block too many in the pruned call
    "keep_cols_plus": ("Q", "keep_cols", "128 * len(kept)", "128 * len(kept) + 128"),
    # the epilogue returns column ids instead of token ids
    "ti_column_id": ("D", "ti", "(id_map ? id_map[id] : id)", "(id)"),
    # ties to the higher id
    "better_tie_high": ("D", "better", "ai < bi", "ai > bi"),
}
# name -> (Bend file, text, replacement): model mutations, each must be rejected by `bend draft_head_idmap_proof.bend`
LAW_MUTATIONS = {
    # a non-increasing kept-block list: kept_blocks without the sort
    "model_unsorted_kept": (IMPL, "  idm_sort(List.take(&2, Nat, order, n))", "  List.take(&2, Nat, order, n)"),
    # reset() leaving hold != 0
    "model_reset_keeps_hold": (IMPL, "def idm_reset(st: Spec.IdmPol) -> Spec.IdmPol:\n  Spec.IdmPol{Nil{}, 0n, True{}}",
                               "def idm_reset(st: Spec.IdmPol) -> Spec.IdmPol:\n  Spec.IdmPol{Nil{}, 1n, True{}}"),
}


def fail(msg: str):
    raise SystemExit(f"draft_head_idmap_diff: FAIL {msg}")


def unlimited_vm() -> None:
    """The Bend runtime reserves its heap up front; lift a soft RLIMIT_AS (harness shells set 8 GB)."""
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def extract(src: list[str], name: str, sig: str, kind: str) -> tuple[int, int]:
    """1-based inclusive line range of the block whose first stripped line starts with sig (exactly one)."""
    if name == "full_order_sha":
        hits = [i for i, l in enumerate(src) if "Full-order sha256 (json of all 1940 blocks):" in l]
    else:
        hits = [i for i, l in enumerate(src) if l.strip().startswith(sig)]
    if len(hits) != 1:
        fail(f"{name}: signature {sig!r} found {len(hits)} times")
    i = hits[0]
    if kind == "line":
        return i + 1, i + 1
    if kind.startswith("lines:"):
        return i + 1, i + int(kind[6:])
    if kind == "py":
        j, base = i + 1, indent(src[i])
        while j < len(src) and (not src[j].strip() or indent(src[j]) > base):
            j += 1
        while not src[j - 1].strip():
            j -= 1
        return i + 1, j
    if kind == "paren":
        depth = 0
        for j in range(i, len(src)):
            depth += src[j].count("(") - src[j].count(")")
            if depth == 0:
                return i + 1, j + 1
        fail(f"{name}: unbalanced parentheses from line {i + 1}")
    depth, opened = 0, False
    for j in range(i, len(src)):
        for ch in src[j].split("//")[0]:
            if ch == "{":
                depth, opened = depth + 1, True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return i + 1, j + 1
    fail(f"{name}: unbalanced braces from line {i + 1}")


def patch_plus(patch: str) -> dict[str, set[str]]:
    """Per file: the set of `+` lines (without their prefix)."""
    plus: dict[str, set[str]] = {}
    cur = None
    for l in patch.split("\n"):
        if l.startswith("+++ b/"):
            cur = l[6:]
            plus.setdefault(cur, set())
        elif l.startswith("--- "):
            continue
        elif cur and l.startswith("+"):
            plus[cur].add(l[1:])
    return plus


# ---- a numpy stand-in for the torch calls of Q's quoted statements ----

class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def view(self, *shape):
        return _T(self.a.reshape(*shape))

    def flatten(self):
        return _T(self.a.reshape(-1))

    def __getitem__(self, k):
        return _T(self.a[k.a if isinstance(k, _T) else k])

    def __setitem__(self, k, v):
        self.a[k.a if isinstance(k, _T) else k] = v

    def __invert__(self):
        return _T(~self.a)

    def to(self, device=None, dtype=None):
        return _T(self.a.astype(dtype) if dtype is not None else self.a)


class _Torch:
    int64, int32, bool = np.int64, np.int32, np.bool_

    @staticmethod
    def arange(n, dtype=None):
        return _T(np.arange(n, dtype=dtype))

    @staticmethod
    def zeros(n, dtype=None):
        return _T(np.zeros(n, dtype=dtype))

    @staticmethod
    def tensor(x):
        return _T(np.array(x))

    @staticmethod
    def cat(ts):
        return _T(np.concatenate([t.a for t in ts]))


def run_block(code: str, ns: dict) -> None:
    exec(compile(textwrap.dedent(code), "<quoted>", "exec"), ns)


class Env:
    """Temporarily set (or unset, value None) environment variables."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_s(text: str) -> dict:
    ns = {"__name__": "dflash2_head_blocks"}
    exec(compile(text, PATHS["S"], "exec"), ns)
    return ns


def q_head(q: dict, vocab: int, kb: list[int]) -> SimpleNamespace:
    """Q's kept check, kept set, keep_cols and id map for kb; ok = the check did not raise."""
    ns = {"torch": _Torch, "vocab": vocab, "kb": list(kb), "dev": "cpu",
          "self": SimpleNamespace(kept_blocks=list(kb), vocab=vocab, id_map=None)}
    run_block(q["HEAD_TOP_K"], ns)
    try:
        run_block(q["kept_check"], ns)
    except ValueError:
        return SimpleNamespace(ok=False)
    run_block(q["kept_set"], ns)
    run_block(q["keep_cols"], ns)
    run_block(q["id_map"], ns)
    self = ns["self"]
    out = SimpleNamespace(ok=True, keep_cols=self.keep_cols, id_map=self.id_map.a.astype(np.int64))
    for pruned in (False, True):
        self.wq = np.zeros(vocab // 16)
        self.scales = np.zeros(vocab // 16)
        ns["pruned"] = pruned
        run_block(q["topk_slice"], ns)
        setattr(out, "call_pruned" if pruned else "call_full", (ns["n"], ns["im"].a.astype(np.int64)))
    return out


def q_check(q: dict, vocab: int, kb: list[int]) -> bool:
    ns = {"vocab": vocab, "kb": list(kb)}
    run_block(q["HEAD_TOP_K"], ns)
    try:
        run_block(q["kept_check"], ns)
    except ValueError:
        return False
    return True


def s_kept_rows(s_text: str, order: list[int], n: int, vocab: int) -> tuple[bool, list[int] | None]:
    """kept_blocks(vocab) with the module's table replaced by order and N = n; False on any exception."""
    ns = load_s(s_text)
    ns["DRAFT_HEAD_BLOCK_ORDER"] = tuple(order)
    ns["DRAFT_HEAD_VOCAB"] = vocab
    ns["DRAFT_HEAD_BLOCKS_ALLOWED"] = (n,)
    with Env(EXL3_DRAFT_HEAD_PRUNED=None, EXL3_DRAFT_HEAD_BLOCKS=str(n)):
        try:
            return True, ns["kept_blocks"](vocab)
        except (ValueError, IndexError):
            return False, None


def c_program(q: dict, v: int, runs: list[tuple[int, str]]) -> str:
    seeds = ",\n    ".join(f"{{{a}ull, {b}ull, {m}ull}}" for a, b, m in SEEDS)
    return f"""// generated by draft_head_idmap_diff.py: quoted dflash2_head.cu better() and epilogue + reference top-16
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __device__
#define __forceinline__ inline

// D: quoted
{q["HEAD_TOPK"]}
{q["better"]}

struct Seed {{ uint64_t a, b, m; }};
static const Seed seeds[] = {{
    {seeds}
}};

static float value_of(const Seed& s, int tok) {{ return (float) ((s.a * (uint64_t) tok + s.b) % s.m); }}

static long long pad_cols = 0;

// One dflash2_q4_head_topk call over `vocab` columns (id_map = nullptr: identity): column id carries the head column
// of its token, so its value is that token's value; tv / ti are the kernel's quoted epilogue statements.
static std::vector<std::pair<float, int>> call(const Seed& s, int vocab, const int* id_map)
{{
    const int n_tiles = (vocab + 15) / 16;
    std::vector<std::pair<float, int>> c;
    for (int id = 0; id < n_tiles * 16; ++id)
    {{
        const int e = 0;
        float tv[4]; int ti[4];
        const int tok = id < vocab ? (id_map ? id_map[id] : id) : 0;
        const float val = value_of(s, tok);
        {q["tv"].strip()}
        {q["ti"].strip()}
        pad_cols += id >= vocab;
        c.push_back({{tv[e], ti[e]}});
    }}
    std::sort(c.begin(), c.end(), [](const std::pair<float, int>& x, const std::pair<float, int>& y)
              {{ return better(x.first, x.second, y.first, y.second); }});
    return c;
}}

static void print(int N, int si, const char* kind, const std::vector<std::pair<float, int>>& c)
{{
    printf("T %d %d %s", N, si, kind);
    for (int i = 0; i < HEAD_TOPK && i < (int) c.size(); ++i) printf(" %lld:%d", (long long) c[i].first, c[i].second);
    printf("\\n");
}}

static std::vector<int> read_ints(FILE* f)
{{
    long long n = 0;
    if (fscanf(f, "%lld", &n) != 1) {{ fprintf(stderr, "bad input\\n"); exit(2); }}
    std::vector<int> v(n);
    for (long long i = 0; i < n; ++i) if (fscanf(f, "%d", &v[i]) != 1) {{ fprintf(stderr, "bad input\\n"); exit(2); }}
    return v;
}}

int main(int argc, char** argv)
{{
    FILE* f = fopen(argv[1], "r");
    long long bad = 0, checks = 0;
    for (int run = 0; run < {len(runs)}; ++run)
    {{
        int N = 0, n_pruned = 0;
        if (fscanf(f, "%d %d", &N, &n_pruned) != 2) return 2;
        std::vector<int> full = read_ints(f), pruned = read_ints(f);
        if ((int) full.size() != {v} || (int) pruned.size() != n_pruned) {{ fprintf(stderr, "bad sizes\\n"); return 2; }}
        std::vector<char> in_prefix({v}, 0);
        for (int t : pruned) in_prefix[t] = 1;
        for (int si = 0; si < (int) (sizeof(seeds) / sizeof(seeds[0])); ++si)
        {{
            auto id = call(seeds[si], {v}, nullptr);
            auto fu = call(seeds[si], {v}, full.data());
            auto pr = call(seeds[si], n_pruned, pruned.data());
            print(N, si, "id", id);
            print(N, si, "full", fu);
            print(N, si, "pruned", pr);
            // C side: the full call on the permuted copy is the identity call (whole ranking), and the prefix call
            // is the identity ranking restricted to the prefix's tokens
            ++checks;
            if (fu != id) {{ ++bad; fprintf(stderr, "FAIL full != id (N %d seed %d)\\n", N, si); }}
            std::vector<std::pair<float, int>> r;
            for (auto& x : id) if (in_prefix[x.second]) r.push_back(x);
            ++checks;
            if (pr != r) {{ ++bad; fprintf(stderr, "FAIL pruned != restricted identity ranking (N %d seed %d)\\n", N, si); }}
        }}
    }}
    fprintf(stderr, "C-side checks: %lld comparisons (whole rankings), %lld padding columns, %lld failures\\n", checks, pad_cols, bad);
    return bad || pad_cols ? 1 : 0;
}}
"""


def bend_literals(impl: str) -> tuple[dict[str, int], list[int]]:
    consts = {}
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
    body = block.split("(", 1)[1].rsplit(")", 1)[0]
    return [int(x) for x in re.findall(r"\d+", body)]


def replay_policy(s_text: str, kept: list[int], vocab: int, p_lines: list[str]) -> tuple[list[str], dict]:
    """Replay the table's P rows through the tree's PrunedHeadPolicy; returns the reference rows and coverage."""
    ns = load_s(s_text)
    out, pols = [], {}
    cov = {"trigger": 0, "reset_in_hold": 0, "hold_done": 0, "big": 0, "negative": 0, "static_rounds": 0,
           "window_expired": 0}
    for l in p_lines:
        f = l.split(" ")
        sc, i, ev = int(f[1]), int(f[2]), f[3]
        if sc not in pols:
            pols[sc] = ns["PrunedHeadPolicy"](kept, vocab, adaptive=sc != 3)
        p = pols[sc]
        hold0, recent0 = p.hold, list(p.recent)
        if ev == "r":
            cov["reset_in_hold"] += hold0 > 0
            p.reset()
        else:
            t = int(ev)
            cov["big"] += t >= vocab
            cov["negative"] += t < 0
            p.observe_anchor(t)
            cov["trigger"] += p.hold == ns["FALLBACK_HOLD"]
            cov["hold_done"] += hold0 == 1 and p.hold == 0 and p.use_pruned
            cov["static_rounds"] += sc == 3
            cov["window_expired"] += (hold0 == 0 and len(recent0) == ns["FALLBACK_WINDOW"] and recent0[0]
                                      and p.hold == 0 and sum(p.recent) == 1 and p.recent[-1])
        out.append(f"P {sc} {i} {ev} {int(p.use_pruned)} {p.hold} {len(p.recent)} {sum(p.recent)}")
    return out, cov


def run_law_mutation(name: str) -> tuple[bool, str]:
    """Apply a model mutation in a scratch copy of the Bend sources; True if the proof gate rejects it."""
    fname, a, b = LAW_MUTATIONS[name]
    with tempfile.TemporaryDirectory() as td:
        for p in HERE.glob("*.bend"):
            shutil.copy(p, Path(td) / p.name)
        t = (Path(td) / fname).read_text()
        if t.count(a) != 1:
            fail(f"law mutation {name} does not apply exactly once")
        (Path(td) / fname).write_text(t.replace(a, b))
        r = subprocess.run(source_link.locked([source_link.bend(), PROOF]), cwd=td, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    loc = next((l.strip() for l in out if l.startswith("Location")), "")
    where = next((l.strip() for l in out if ">|" in l), "")
    return r.returncode != 0, f"exit {r.returncode}; {out[0] if out else ''} {loc} {where}".strip()


def run_all_mutations(tree: str) -> int:
    rc = 0
    for name in MUTATIONS:
        r = subprocess.run([sys.executable, __file__, "--mutate", name, tree], capture_output=True, text=True)
        ls = (r.stdout + r.stderr).splitlines()
        if r.returncode == 0:
            print(f"{name}: SURVIVED (unexpected)", flush=True)
            rc = 1
            continue
        fin = next((l for l in ls if l.startswith("draft_head_idmap_diff: FAIL")), "")
        cside = next((l for l in ls if l.startswith("C-side checks")), "")
        mm = next((i for i, l in enumerate(ls) if l.startswith("MISMATCH")), None)
        tab = " / ".join(x.strip()[:160] for x in ls[mm:mm + 3]) if mm is not None else "table IDENTICAL"
        print(f"{name}: REJECTED: " + " | ".join(x for x in (fin, cside, tab) if x), flush=True)
    for name in LAW_MUTATIONS:
        rejected, why = run_law_mutation(name)
        print(f"{name}: {'REJECTED by the proof gate' if rejected else 'SURVIVED (unexpected)'}: {why}", flush=True)
        rc |= not rejected
    print("draft_head_idmap_diff --all-mutations: " + ("as expected" if rc == 0 else "UNEXPECTED OUTCOME"))
    return rc


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    order_json = None
    if args[:1] == ["--order-json"]:
        if len(args) < 2:
            fail(__doc__)
        order_json, args = Path(args[1]), args[2:]
    if args[:1] == ["--all-mutations"]:
        if len(args) != 2:
            fail(__doc__)
        sys.exit(run_all_mutations(args[1]))
    if args[:1] == ["--mutate"]:
        if len(args) < 2 or args[1] not in MUTATIONS:
            fail(f"--mutate NAME, NAME in {sorted(MUTATIONS)}")
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail(__doc__)
    tree = Path(args[0])

    patch_path = PATCH_PATH
    patch = patch_path.read_bytes()
    sha = hashlib.sha256(patch).hexdigest()
    print(f"patch {patch_path} sha256 {sha}")
    if sha != PATCH_SHA256:
        fail(f"patch sha256 {sha} != pinned {PATCH_SHA256}")
    plus = patch_plus(patch.decode())

    text = {k: (tree / p).read_text() for k, p in PATHS.items()}
    lines = {k: t.split("\n") for k, t in text.items()}
    q: dict[str, str] = {}
    rng: dict[str, tuple[str, int, int]] = {}
    n_plus = 0
    for f, name, sig, kind, origin in BLOCKS:
        a, b = extract(lines[f], name, sig, kind)
        body = lines[f][a - 1:b]
        q[name] = "\n".join(body)
        rng[name] = (f, a, b)
        tag = ""
        if origin == "new":
            nonblank = [l for l in body if l.strip()]
            miss = [l for l in nonblank if l not in plus.get(PATHS[f], set())]
            if miss:
                fail(f"{f} {name} lines {a}-{b}: not `+` lines of the patch: {miss[:3]}")
            n_plus += len(nonblank)
            tag = f", {len(nonblank)}/{len(nonblank)} non-blank lines occur as `+` lines of the patch"
        print(f"quoted {f}:{a}-{b} {name}{tag}")
    print(f"quoted 9005c lines verified as patch `+` lines: {n_plus}")

    # constants: the exact source lines, the Bend model's literals
    src_const = {}
    for name, (line, val) in CONST_LINES.items():
        if q[name].strip() != line:
            fail(f"{name}: source line {q[name].strip()!r} != expected {line!r}")
        src_const[name] = val
        print(f"parsed {name} = {val!r} from {line!r}")
    order = parse_order(q["DRAFT_HEAD_BLOCK_ORDER"])
    impl = (HERE / IMPL).read_text()
    bconst, border = bend_literals(impl)
    for d, name in BEND_CONSTS.items():
        if bconst[d] != src_const[name]:
            fail(f"{IMPL} {d} = {bconst[d]} != source {name} = {src_const[name]}")
    if not re.search(r"\ndef idm_n_alt\(\) -> Nat:\n  1024n\n", impl) or src_const["DRAFT_HEAD_BLOCKS_ALLOWED"] != (896, 1024):
        fail("idm_n_alt / DRAFT_HEAD_BLOCKS_ALLOWED mismatch")
    if border != order or len(order) != 1024 or len(set(order)) != 1024:
        fail(f"{IMPL} idm_order ({len(border)} entries) != source DRAFT_HEAD_BLOCK_ORDER ({len(order)} entries)")
    print(f"Bend model constants {sorted(bconst.items())} and idm_order ({len(border)} entries) equal the source")
    pinned = re.search(r"[0-9a-f]{64}", q["full_order_sha"]).group(0)
    if order_json is not None:
        full = json.load(open(order_json))["order"]
        fsha = hashlib.sha256(json.dumps(full).encode()).hexdigest()
        if sorted(full) != list(range(1940)) or full[:1024] != order or fsha != pinned:
            fail(f"{order_json}: not a permutation of 1940 blocks extending the source table with sha256 {pinned} ({fsha})")
        print(f"full order {order_json}: permutation of the 1940 blocks, prefix = source table, sha256 {fsha} = source pin")
    else:
        print(f"full order not given (--order-json): permutation check of the full order not run (source pin {pinned})")

    s_text = text["S"]
    if mutate:
        fk, blk, a, b = MUTATIONS[mutate]
        if q[blk].count(a) != 1 or text[fk].count(a) != 1:
            fail(f"mutation {mutate} does not apply exactly once")
        q[blk] = q[blk].replace(a, b)
        if fk == "S":
            s_text = s_text.replace(a, b)
        print(f"mutation {mutate}: {blk}: {a!r} -> {b!r}")

    V, W = src_const["DRAFT_HEAD_VOCAB"], src_const["DRAFT_HEAD_BLOCK"]
    nb = V // W

    # the Bend table
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "table"
        r = subprocess.run(source_link.locked([source_link.bend(), TABLE, "-o", str(exe)]), cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"bend compile: {r.stdout}{r.stderr}")
        bres = subprocess.run([str(exe)], capture_output=True, text=True, preexec_fn=unlimited_vm)
    if bres.returncode != 0:
        fail(f"Bend table exited {bres.returncode}: {bres.stderr}")
    bl = bres.stdout.split("\n")

    # reference rows
    ref: list[str] = []
    ref.append(f"K vocab={V} block={W} n={src_const['DRAFT_HEAD_BLOCKS_DEFAULT']} n_alt={max(src_const['DRAFT_HEAD_BLOCKS_ALLOWED'])} "
               f"window={src_const['FALLBACK_WINDOW']} trigger={src_const['FALLBACK_TRIGGER']} hold={src_const['FALLBACK_HOLD']} "
               f"topk={src_const['HEAD_TOPK']}")
    ref.append("O " + " ".join(map(str, order)))
    ref += [f"S {i} {a} {b} {m}" for i, (a, b, m) in enumerate(SEEDS)]
    s_ns = load_s(s_text)
    runs, cfile = [], []
    kept_by_n = {}
    for N in src_const["DRAFT_HEAD_BLOCKS_ALLOWED"]:
        with Env(EXL3_DRAFT_HEAD_PRUNED=None, EXL3_DRAFT_HEAD_BLOCKS=None if N == src_const["DRAFT_HEAD_BLOCKS_DEFAULT"] else str(N)):
            if s_ns["pruned_mode"]() != "adaptive":
                fail("pruned_mode() default is not adaptive")
            try:
                kb = s_ns["kept_blocks"](V)
            except (ValueError, IndexError) as e:
                fail(f"kept_blocks({V}) with N = {N} raised {e!r}")
        kept_by_n[N] = kb
        h = q_head(q, V, kb)
        if not h.ok:
            ref.append(f"B {N} ok=1 head_ok=0 keep_cols=? " + " ".join(map(str, kb)))
            print(f"N {N}: Q4DraftHead rejects the kept blocks")
            continue
        ref.append(f"B {N} ok=1 head_ok=1 keep_cols={h.keep_cols} " + " ".join(map(str, kb)))
        m = h.id_map
        ref += [f"M {N} {r} " + " ".join(map(str, m[r * W:(r + 1) * W])) for r in range(nb)]
        (nf, imf), (np_, imp) = h.call_full, h.call_pruned
        if nf != V or len(imf) != V:
            fail(f"N {N}: full call n = {nf}, |im| = {len(imf)}")
        ref.append(f"@T {N}")
        runs.append((N, "run"))
        cfile.append(f"{N} {np_}\n{len(imf)} " + " ".join(map(str, imf)) + f"\n{len(imp)} " + " ".join(map(str, imp)) + "\n")
        print(f"N {N}: kept_blocks {len(kb)} blocks, keep_cols {h.keep_cols}, id map {len(m)} ids, "
              f"prefix call n = {np_} (|im[:n]| = {len(imp)})")

    # the kernel's quoted ranking on the CPU
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "topk.cpp"
        c.write_text(c_program(q, V, runs))
        exe = Path(td) / "topk"
        subprocess.run(source_link.locked(["c++", "-O2", "-std=c++17", "-Wall", "-o", str(exe), str(c)]), check=True)
        inp = Path(td) / "maps.txt"
        inp.write_text("".join(cfile))
        cres = subprocess.run([str(exe), str(inp)], capture_output=True, text=True)
    sys.stdout.write(cres.stderr)
    t_rows = [l for l in cres.stdout.split("\n") if l]
    full_ref = []
    for l in ref:
        if l.startswith("@T "):
            N = l.split()[1]
            full_ref += [t for t in t_rows if t.split()[1] == N]
        else:
            full_ref.append(l)

    # policy replay on the table's own event sequences (kept = the N = 896 blocks)
    p_lines = [l for l in bl if l.startswith("P ")]
    p_ref, cov = replay_policy(s_text, kept_by_n[src_const["DRAFT_HEAD_BLOCKS_DEFAULT"]], V, p_lines)
    full_ref += p_ref
    print("policy replay coverage: " + ", ".join(f"{k} {v}" for k, v in cov.items()))
    if not mutate and min(cov.values()) == 0:
        fail(f"policy scenarios do not cover every case: {cov}")

    # the two checks on the table's fixed lists
    for l in bl:
        if l.startswith("H "):
            f = l.split(" ")
            vocab = int(f[2].split("=")[1])
            kb = [int(x) for x in f[5:]]
            full_ref.append(f"H {f[1]} vocab={vocab} ok={int(q_check(q, vocab, kb))} kb" + "".join(f" {x}" for x in kb))
        elif l.startswith("R "):
            f = l.split(" ")
            n, vocab = int(f[2].split("=")[1]), int(f[3].split("=")[1])
            order_c = [int(x) for x in f[f.index("order") + 1:]]
            ok, blocks = s_kept_rows(s_text, order_c, n, vocab)
            shown = blocks if ok else sorted(order_c[:n])
            full_ref.append(f"R {f[1]} n={n} vocab={vocab} ok={int(ok)} blocks" + "".join(f" {x}" for x in shown) +
                            " order" + "".join(f" {x}" for x in order_c))
    full_ref.append("")

    kinds = {k: sum(l.startswith(k + " ") for l in bl) for k in "KOSBMTPHR"}
    print("table rows: Bend " + ", ".join(f"{k} {v}" for k, v in kinds.items()) + f"; reference {len(full_ref) - 1}")
    identical = full_ref == bl
    if identical:
        print("IDENTICAL")
    else:
        shown = 0
        for i in range(max(len(full_ref), len(bl))):
            a = full_ref[i] if i < len(full_ref) else "<eof>"
            b = bl[i] if i < len(bl) else "<eof>"
            if a != b:
                print(f"MISMATCH row {i}:\n  source: {a[:200]}\n  Bend:   {b[:200]}")
                shown += 1
                if shown == 3:
                    break
    if cres.returncode != 0 or not identical:
        fail(" + ".join((["table MISMATCH"] if not identical else []) +
                        (["C-side check failures"] if cres.returncode != 0 else [])))
    print("draft_head_idmap_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
