#!/usr/bin/env python3
"""
Finite differential check of bend/gdn_replay.bend against the pinned ExLlamaV3 engine.

Regenerates the table bend/GDN_REPLAY_TABLE.bend prints, but from the engine's own code: the
host job builders (RewindPlans, GDNState / GDNLayerState, _collect_rewind_jobs,
_dispatch_rewind_jobs, advance_recurrent_states) are extracted with `ast` from the pinned
sources and executed against stub torch/ext objects; the launched jobs are then interpreted by
line-by-line Python ports of the CUDA integer indexing of gdn.cu. The result is compared byte for
byte with the Bend program's output. This is differential evidence on a finite instance, not a
proof of equivalence.

Usage: python3 bend/gdn_replay_diff.py [TREE_ROOT] [RECURRENT_UTIL_PY]
"""

from __future__ import annotations

import __future__ as _future
import ast
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEND = "/nix/store/kqhwjzdm96d14fvzblb4jz9m73cr3i0j-bend-2.0.34/bin/bend"
TABLE = "bend/GDN_REPLAY_TABLE.bend"
DEFAULT_TREE = "/tmp/kernel-work/ReplayProof/tree"
DEFAULT_RECURRENT_UTIL = "/tmp/kernel-work/DraftHead2/base_c/cache/recurrent_util.py"

# Pins: post-images of the patch series, read from the manifests (never hardcoded here)
MANIFEST_FILES = {
    "patches/exl3/exl3-patches.json": ("generator/gdn_rewind.py", "generator/generator.py"),
    "patches/exl3-ext/exl3-ext.json": (
        "modules/gated_delta_net.py", "exllamav3_ext/gdn.cu", "exllamav3_ext/gdn.cuh"
    ),
}
# cache/recurrent_util.py is not patched by any series, so no manifest pins it: this is the
# upstream exllamav3 1.5.0 file (engine revision 355c6ee1 in exl3-patches.json `engine`)
RECURRENT_UTIL_SHA256 = "1d6922211adf2e4e514f3a8ddaf5943259bab276ce921dfb3d730b8d4762ba66"

# Instance (GDN_REPLAY_TABLE.bend:3-8)
K_CONV = 4                     # module.conv_kernel_size
MAX_HISTORY = 7                # state_size = K + H = 11 conv columns (gated_delta_net.py:190)
START_POSITION = 1000
WINDOWS = range(8)
SLOT = 0

# gdn.cu constants
HEAD_DIM = 128                 # gdn.cu:907, 2931
CONV1D_MAX_K = 16              # gdn.cu:1828
CONV_BA_MAX_S = 16             # gdn.cu:2268
RULE_INPLACE, RULE_HISTORY, RULE_VERIFY, RULE_COMMIT = 0, 1, 2, 3   # gdn.cu:883

NUM_K_HEADS = 1
NUM_V_HEADS = 1
CONV_CHANNELS = 1              # module.fdim_qkv of the instance: one conv channel is modeled
# Width of one token row of the rule kernel's mixed_qkv staging (gdn.cu:1022 per-step advance,
# gdn.cu:2932 F); the one-channel conv instance keeps the real rule-input layout
F_QKV = 2 * HEAD_DIM * NUM_K_HEADS + HEAD_DIM * NUM_V_HEADS

BF16, F32 = 2, 4               # element sizes in bytes
# Symbolic, disjoint byte addresses of the device tensors
CONV_BASE = 0x0
RS_BASE = 0x10000
QKV_BASE = 0x20000
G_BASE = 0x40000
BETA_BASE = 0x50000

REFUSALS = (IndexError, AssertionError, RuntimeError)


def fail(msg: str):
    raise SystemExit(f"gdn_replay_diff: {msg}")


def check(cond: bool, msg: str):
    if not cond:
        fail(msg)


# ---------------------------------------------------------------------------------------------
# Pins

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins(tree: Path, recurrent_util: Path):
    for manifest, keys in MANIFEST_FILES.items():
        files = json.loads((REPO / manifest).read_text())["files"]
        for key in keys:
            check(key in files, f"{manifest} has no pin for {key}")
            pin = files[key]["post"]
            check(isinstance(pin, str) and len(pin) == 64, f"{manifest}: bad post pin for {key}")
            path = tree / key
            check(path.is_file(), f"missing pinned source {path}")
            got = sha256(path)
            check(got == pin, f"{path}: sha256 {got} != post pin {pin} ({manifest})")
    got = sha256(recurrent_util)
    check(got == RECURRENT_UTIL_SHA256, f"{recurrent_util}: sha256 {got} != {RECURRENT_UTIL_SHA256}")


# ---------------------------------------------------------------------------------------------
# Decision: generator.py's own expression

NUM_REJECTED_TEXT = "num_rejected = window + 1 - count if not eos and count <= window else 0"


def load_num_rejected(tree: Path):
    """generator.py:1295 (pinned post-image): the round's decision, evaluated verbatim"""
    path = tree / "generator/generator.py"
    lines = path.read_text().splitlines()
    hits = [i for i, line in enumerate(lines, 1) if NUM_REJECTED_TEXT in line]
    check(len(hits) == 1, f"{path}: expected one `{NUM_REJECTED_TEXT}` line, found {hits}")
    stmt = ast.parse(lines[hits[0] - 1].strip()).body
    check(len(stmt) == 1 and isinstance(stmt[0], ast.Assign)
          and [t.id for t in stmt[0].targets if isinstance(t, ast.Name)] == ["num_rejected"],
          f"{path}:{hits[0]}: not a plain num_rejected assignment")
    code = compile(ast.Expression(stmt[0].value), f"{path}:{hits[0]}", "eval")

    def num_rejected(window: int, count: int, eos: bool) -> int:
        return eval(code, {"__builtins__": {}}, {"window": window, "count": count, "eos": eos})

    return num_rejected


# ---------------------------------------------------------------------------------------------
# Host code: extracted from the pinned sources and executed against stubs

def _top(tree_ast: ast.Module, path: Path, name: str):
    nodes = [n for n in tree_ast.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    check(len(nodes) == 1, f"{path}: expected one top-level `{name}`, found {len(nodes)}")
    return nodes[0]


def _class_subset(tree_ast: ast.Module, path: Path, name: str, methods: tuple) -> ast.ClassDef:
    cls = _top(tree_ast, path, name)
    check(isinstance(cls, ast.ClassDef) and not cls.bases and not cls.decorator_list,
          f"{path}: `{name}` is not a plain class")
    body = []
    for m in methods:
        defs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == m]
        check(len(defs) == 1, f"{path}: expected one {name}.{m}, found {len(defs)}")
        body.append(defs[0])
    sub = ast.ClassDef(name=cls.name, bases=[], keywords=[], body=body, decorator_list=[])
    ast.copy_location(sub, cls)
    return sub


def _exec_nodes(nodes: list, path: Path, ns: dict):
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    # The sources start with `from __future__ import annotations`: annotations stay unevaluated
    code = compile(mod, str(path), "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)
    exec(code, ns)


class _Device:
    def __init__(self, index):
        self.index = index


class _Torch:
    """torch.device(x).index is the only torch use on the exercised host paths"""
    @staticmethod
    def device(x):
        check(x == "cuda:0", f"unexpected device {x!r}")
        return _Device(0)


class FakeTensor:
    """Contiguous tensor over a symbolic byte address: data_ptr / stride / shape / element_size"""

    def __init__(self, base: int, shape: tuple, element_size: int):
        self._base = base
        self.shape = tuple(shape)
        self._es = element_size
        strides, acc = [], 1
        for n in reversed(self.shape):
            strides.append(acc)
            acc *= n
        self._strides = tuple(reversed(strides))

    def data_ptr(self):
        return self._base

    def stride(self, dim: int):
        return self._strides[dim]

    def element_size(self):
        return self._es


# Job descriptors: field order of the gdn.cuh constructors (gdn.cuh:284, 298, 319-320)
class ConvRewindJob:
    FIELDS = ("src", "dst", "dim", "cdim", "stride")

    def __init__(self, *args):
        check(len(args) == 5, "ConvRewindJob takes 5 arguments")
        for f, v in zip(self.FIELDS, args):
            check(isinstance(v, int), f"ConvRewindJob.{f} not an int: {v!r}")
            setattr(self, f, v)

    def key(self):
        return ("ConvRewindJob",) + tuple(getattr(self, f) for f in self.FIELDS)


class StateRewindJob(ConvRewindJob):
    FIELDS = ("src", "dst", "num_elements")

    def __init__(self, *args):
        check(len(args) == 3, "StateRewindJob takes 3 arguments")
        for f, v in zip(self.FIELDS, args):
            check(isinstance(v, int), f"StateRewindJob.{f} not an int: {v!r}")
            setattr(self, f, v)

    def key(self):
        return ("StateRewindJob",) + tuple(getattr(self, f) for f in self.FIELDS)


class StateReplayJob(ConvRewindJob):
    FIELDS = ("mixed_qkv", "g", "beta", "state", "row", "seqlen", "steps", "num_k_heads",
              "num_v_heads")

    def __init__(self, *args):
        check(len(args) == 9, "StateReplayJob takes 9 arguments")
        for f, v in zip(self.FIELDS, args):
            check(isinstance(v, int), f"StateReplayJob.{f} not an int: {v!r}")
            setattr(self, f, v)

    def key(self):
        return ("StateReplayJob",) + tuple(getattr(self, f) for f in self.FIELDS)


class Ext:
    """exllamav3_ext stub: the host halves of the batched launchers (their TORCH_CHECKs) append
    each launch to a log that the device ports below interpret"""
    ConvRewindJob = ConvRewindJob
    StateRewindJob = StateRewindJob
    StateReplayJob = StateReplayJob

    def __init__(self):
        self.log = []

    def batched_conv_rewind(self, jobs, device_index):
        if not jobs:                                            # gdn.cu:2867
            return
        for j in jobs:
            check(type(j) is ConvRewindJob, "batched_conv_rewind: not a ConvRewindJob")
            if not j.cdim <= CONV1D_MAX_K:                      # gdn.cu:2880
                raise RuntimeError("batched_conv_rewind: cdim exceeds CONV1D_MAX_K")
        self.log.append(("conv_rewind", device_index, tuple(jobs)))

    def batched_state_rewind(self, jobs, device_index):
        if not jobs:                                            # gdn.cu:2892
            return
        for j in jobs:
            check(type(j) is StateRewindJob, "batched_state_rewind: not a StateRewindJob")
            if j.num_elements % 4 != 0:                         # gdn.cu:2905
                raise RuntimeError("batched_state_rewind: num_elements must be a multiple of 4")
        self.log.append(("state_rewind", device_index, tuple(jobs)))

    def batched_state_replay(self, jobs, device_index):
        if not jobs:                                            # gdn.cu:2953
            return
        for j in jobs:
            check(type(j) is StateReplayJob, "batched_state_replay: not a StateReplayJob")
            # gdn.cu:2966
            if not (j.mixed_qkv and j.g and j.beta and j.state):
                raise RuntimeError("batched_state_replay: null pointer")
            # gdn.cu:2967-2968
            if not (1 <= j.steps and j.steps <= j.seqlen and 0 <= j.row):
                raise RuntimeError("batched_state_replay: steps must be in 1..seqlen")
            # gdn.cu:2969-2970
            if not (j.num_k_heads >= 1 and j.num_v_heads % j.num_k_heads == 0
                    and j.num_v_heads <= 65535):
                raise RuntimeError("batched_state_replay: bad head counts")
        self.log.append(("state_replay", device_index, tuple(jobs)))


def load_host(tree: Path, recurrent_util: Path) -> dict:
    ns = {"__name__": "gdn_replay_host", "torch": _Torch}
    rewind_py = tree / "generator/gdn_rewind.py"
    gdn_py = tree / "modules/gated_delta_net.py"
    a = ast.parse(rewind_py.read_text())
    b = ast.parse(gdn_py.read_text())
    u = ast.parse(recurrent_util.read_text())
    # gated_delta_net.py:31-59, 62-69, 78-128 (GDNState), 179-368 (GDNLayerState)
    _exec_nodes([
        _top(b, gdn_py, "_collect_rewind_jobs"),
        _top(b, gdn_py, "_dispatch_rewind_jobs"),
        _class_subset(b, gdn_py, "GDNState", ("rewind", "post_advance")),
        _class_subset(b, gdn_py, "GDNLayerState", (
            "replay_job", "check_history_rows", "rewind", "rewind_conv_job", "rewind_state_job",
            "rewind_replay_job",
        )),
    ], gdn_py, ns)
    # gdn_rewind.py:18-112 (_ReplayView, RewindPlans)
    _exec_nodes([_top(a, rewind_py, "_ReplayView"), _top(a, rewind_py, "RewindPlans")],
                rewind_py, ns)
    # recurrent_util.py:66-75 (advance_recurrent_states; 73-74 advance position, last_history)
    adv = _top(u, recurrent_util, "advance_recurrent_states")
    src = ast.get_source_segment(recurrent_util.read_text(), adv)
    for stmt in ("r.position += seqlen", "r.last_history = (seqlen - 1) if history else 0"):
        check(stmt in src, f"{recurrent_util}: advance_recurrent_states lacks `{stmt}`")
    _exec_nodes([adv], recurrent_util, ns)
    return ns


# ---------------------------------------------------------------------------------------------
# Device memory: byte address -> value. Rows of the recurrent state are collapsed to one cell
# (the trace of replayed rule inputs); any read of an unwritten address fails closed.

class Memory:
    def __init__(self):
        self.cells = {}

    def read(self, addr: int):
        check(addr in self.cells, f"device read of unmapped address {addr:#x}")
        return self.cells[addr]

    def write(self, addr: int, value):
        check(addr in self.cells, f"device write to unmapped address {addr:#x}")
        self.cells[addr] = value


def ptr(base: int, index: int, es: int) -> int:
    """typed pointer arithmetic: (T*) base + index"""
    return base + index * es


# ---------------------------------------------------------------------------------------------
# Device ports (gdn.cu, pinned post-image)

def conv1d_update_history(mem: Memory, x_base: int, conv_state: int, slots: list, dim: int,
                          seqlen: int, state_size: int, K: int, bsz: int):
    """conv1d_update_kernel<ACT, HISTORY = true> (gdn.cu:1839-1925): the window write-back"""
    for b in range(bsz):                                        # blockIdx.y
        for d in range(dim):                                    # gdn.cu:1855-1856
            slot = slots[b] if slots else b                     # gdn.cu:1858
            x_d = ptr(x_base, (b * dim + d) * seqlen, BF16)     # gdn.cu:1860
            state_d = ptr(conv_state, (slot * dim + d) * state_size, BF16)  # gdn.cu:1861
            old_state = [None] * CONV1D_MAX_K
            for k in range(CONV1D_MAX_K):                       # gdn.cu:1874-1875
                if k < K:
                    old_state[k] = mem.read(ptr(state_d, k, BF16))
            total = K + seqlen                                  # gdn.cu:1914
            write_size = state_size if state_size < total else total   # gdn.cu:1915
            dst_start = state_size - write_size                 # gdn.cu:1916
            src_start = total - write_size                      # gdn.cu:1917
            for j in range(write_size):                         # gdn.cu:1918
                src_t = src_start + j                           # gdn.cu:1920
                # gdn.cu:1921
                v = old_state[src_t] if src_t < K else mem.read(ptr(x_d, src_t - K, BF16))
                mem.write(ptr(state_d, dst_start + j, BF16), v)  # gdn.cu:1922


def gdn_conv_ba_history(mem: Memory, qkv_base: int, conv_state: int, slots: list, F: int,
                        S: int, state_size: int, K: int, B: int):
    """gdn_conv_ba_kernel<HISTORY = true> (gdn.cu:2275-2476): sh_seq staging and write-back"""
    check(S <= CONV_BA_MAX_S, "gdn_conv_ba: too many rows")    # gdn.cu:2529
    check(state_size >= K, "conv_state must have at least K entries")   # gdn.cu:2535
    HISTORY = True
    d0 = 0                                                      # one block of channels
    for b in range(B):
        slot = slots[b] if slots else b                         # gdn.cu:2401
        state_c = ptr(conv_state, (slot * F + d0) * state_size, BF16)   # gdn.cu:2402
        channels = min(128, F - d0)                             # gdn.cu:2469 (CONV_BA_THREADS)
        sh_seq = [[None] * (K + S) for _ in range(channels)]
        for t in range(channels):                               # threadIdx.x, d = d0 + t
            d = d0 + t
            state_d = ptr(state_c, t * state_size, BF16)        # gdn.cu:2406
            x_d = ptr(qkv_base, b * S * F + d, F32)             # gdn.cu:2407
            seq = sh_seq[t]                                     # gdn.cu:2408
            for kk in range(K):                                 # gdn.cu:2421-2424
                seq[kk] = mem.read(ptr(state_d, kk, BF16))
            xb = [mem.read(ptr(x_d, s * F, F32)) for s in range(S)]    # gdn.cu:2434-2435
            for s in range(S):                                  # gdn.cu:2438-2442
                seq[K + s] = xb[s]
        total = K + S                                           # gdn.cu:2465
        write_size = (state_size if state_size < total else total) if HISTORY else K  # 2466
        dst_start = state_size - write_size if HISTORY else 0   # gdn.cu:2467
        src_start = total - write_size                          # gdn.cu:2468
        for e in range(channels * write_size):                  # gdn.cu:2470
            c = e // write_size                                 # gdn.cu:2472
            j = e - c * write_size                              # gdn.cu:2473
            # gdn.cu:2474
            mem.write(ptr(state_c, c * state_size + dst_start + j, BF16), sh_seq[c][src_start + j])


def gated_delta_rule_128_reg(MODE: int, mem: Memory, mixed_qkv: int, g: int, beta: int,
                             slot_state: int, steps: int, num_k_heads: int, num_v_heads: int,
                             state_size: int, head: int, step):
    """gated_delta_rule_128_reg<MODE> (gdn.cu:889-1032), the lane of thread elem = 0 with the
    state row collapsed to one cell; `step` is the recurrent update. Returns (per-step outputs,
    element offsets of mixed_qkv each step read its token from)"""
    check(MODE in (RULE_VERIFY, RULE_COMMIT), f"unported rule mode {MODE}")
    group = num_v_heads // num_k_heads                          # gdn.cu:913
    k_head = head // group                                      # gdn.cu:918
    elem = 0                                                    # gdn.cu:933 (collapsed)
    state = mem.read(ptr(slot_state, elem, F32))                # gdn.cu:936
    outs, reads = [], []
    for s in range(steps):                                      # gdn.cu:938
        gl_q = ptr(mixed_qkv, k_head * HEAD_DIM, BF16)          # gdn.cu:940
        x = mem.read(gl_q)                                      # gdn.cu:944 (token s's input)
        g_h = mem.read(ptr(g, head, F32))                       # gdn.cu:990
        beta_h = mem.read(ptr(beta, head, BF16))                # gdn.cu:991
        check(g_h == x and beta_h == x, f"rule step {s}: q/g/beta rows disagree ({x}, {g_h}, {beta_h})")
        reads.append((gl_q - QKV_BASE) // BF16)
        state = step(state, x)                                  # gdn.cu:1004
        # gdn.cu:1005: history rows are written only by RULE_HISTORY
        if MODE != RULE_COMMIT:                                 # gdn.cu:1012
            outs.append(state)                                  # gdn.cu:1014-1019 (row s)
        mixed_qkv = ptr(mixed_qkv, 2 * HEAD_DIM * num_k_heads + HEAD_DIM * num_v_heads, BF16)  # 1022
        g = ptr(g, num_v_heads, F32)                            # gdn.cu:1023
        beta = ptr(beta, num_v_heads, BF16)                     # gdn.cu:1024
    if MODE == RULE_INPLACE or MODE == RULE_COMMIT:             # gdn.cu:1027
        mem.write(ptr(slot_state, elem, F32), state)            # gdn.cu:1030
    return outs, reads


def rule_verify_kernel(mem: Memory, mixed_qkv: int, g: int, beta: int, recurrent_state: int,
                       bsz: int, seqlen: int, num_k_heads: int, num_v_heads: int, slots: list,
                       history_stride: int, step):
    """cuda_recurrent_gated_delta_rule_kernel_128_reg<RULE_VERIFY> (gdn.cu:1034-1074)"""
    group = num_v_heads // num_k_heads                          # gdn.cu:1057
    state_size = 1                                              # gdn.cu:1058, collapsed row
    slot_size = history_stride * state_size                     # gdn.cu:1059
    rows = []
    for bi in range(bsz):                                       # gdn.cu:1061
        # gdn.cu:1062
        qkv = ptr(mixed_qkv, bi * seqlen * (3 * HEAD_DIM * num_k_heads + HEAD_DIM * (num_v_heads - num_k_heads)), BF16)
        gg = ptr(g, bi * seqlen * (group * num_k_heads), F32)   # gdn.cu:1063
        bb = ptr(beta, bi * seqlen * (group * num_k_heads), BF16)   # gdn.cu:1064
        state_slot = slots[bi] if slots else bi                 # gdn.cu:1065
        slot_state = ptr(recurrent_state, state_slot * slot_size, F32)  # gdn.cu:1066
        outs, reads = gated_delta_rule_128_reg(                 # gdn.cu:1069-1073
            RULE_VERIFY, mem, qkv, gg, bb, slot_state, seqlen, num_k_heads, num_v_heads,
            state_size, 0, step,
        )
        check(reads == [(bi * seqlen + s) * F_QKV for s in range(seqlen)],
              f"verify read token rows {reads}")
        rows.append(outs)
    return rows


def batched_conv_rewind_kernel(mem: Memory, jobs: tuple):
    """batched_conv_rewind_kernel (gdn.cu:2826-2846)"""
    for j in jobs:                                              # gdn.cu:2829-2831
        for d in range(j.dim):                                  # gdn.cu:2833-2834
            s = ptr(j.src, d * j.stride, BF16)                  # gdn.cu:2836
            t = ptr(j.dst, d * j.stride, BF16)                  # gdn.cu:2837
            reg = [None] * CONV1D_MAX_K
            for k in range(CONV1D_MAX_K):                       # gdn.cu:2841-2842
                if k < j.cdim:
                    reg[k] = mem.read(ptr(s, k, BF16))
            for k in range(CONV1D_MAX_K):                       # gdn.cu:2844-2845
                if k < j.cdim:
                    mem.write(ptr(t, k, BF16), reg[k])


def batched_state_replay_kernel(mem: Memory, jobs: tuple, step):
    """batched_state_replay_kernel (gdn.cu:2923-2949) into gated_delta_rule_128_reg<RULE_COMMIT>"""
    for j in jobs:                                              # gdn.cu:2926-2928
        F = 2 * HEAD_DIM * j.num_k_heads + HEAD_DIM * j.num_v_heads    # gdn.cu:2932
        check(F == F_QKV, f"replay F {F} != staging width {F_QKV}")
        row_tokens = j.row * j.seqlen                           # gdn.cu:2933
        for head in range(j.num_v_heads):                       # blockIdx.y, gdn.cu:2929
            _, reads = gated_delta_rule_128_reg(                # gdn.cu:2934-2948
                RULE_COMMIT, mem,
                ptr(j.mixed_qkv, row_tokens * F, BF16),         # gdn.cu:2936
                ptr(j.g, row_tokens * j.num_v_heads, F32),      # gdn.cu:2937
                ptr(j.beta, row_tokens * j.num_v_heads, BF16),  # gdn.cu:2938
                j.state,                                        # gdn.cu:2939
                j.steps,                                        # gdn.cu:2941
                j.num_k_heads, j.num_v_heads,
                j.num_v_heads * HEAD_DIM * HEAD_DIM,            # gdn.cu:2945
                head, step,
            )
            # token index read at step s: (offset / F) - row * seqlen
            for s, off in enumerate(reads):
                check(off % F == 0 and off // F - j.row * j.seqlen == s,
                      f"replay step {s} read element {off}")


def interpret(mem: Memory, log: list, step):
    for kind, device_index, jobs in log:
        check(device_index == 0, f"launch on device {device_index}")
        if kind == "conv_rewind":
            batched_conv_rewind_kernel(mem, jobs)
        elif kind == "state_replay":
            batched_state_replay_kernel(mem, jobs, step)
        else:
            fail(f"unexpected {kind} launch after a history-free verify")


# ---------------------------------------------------------------------------------------------
# One round of the instance

def snoc(state: tuple, x: int) -> tuple:
    """the instance's recurrent update: the state is the trace of consumed rule inputs"""
    return state + (x,)


class _Model:
    loaded_tp = False


class _Cache:
    def __init__(self, layer):
        self.model = _Model()
        self._layer = layer

    def get_all_recurrent_layers(self):
        return {0: self._layer}


class _Module:
    conv_kernel_size = K_CONV
    num_k_heads = NUM_K_HEADS
    num_v_heads = NUM_V_HEADS
    fdim_qkv = CONV_CHANNELS


class _Ids:
    def __init__(self, shape):
        self.shape = shape


class Engine:
    """one slot and one GDN layer: device memory, host state objects, the ext launch log"""

    def __init__(self, ns: dict, ext: Ext):
        self.ns = ns
        self.ext = ext
        self.mem = Memory()
        layer = object.__new__(ns["GDNLayerState"])
        layer.module = _Module()
        state_size = K_CONV + MAX_HISTORY
        # gated_delta_net.py:189-198: (max_batch_size, fdim_qkv, K + H) bf16 and
        # (max_batch_size, H + 1, nv, hk, hv) fp32 with the head block collapsed to one cell
        layer.conv_state = FakeTensor(CONV_BASE, (1, CONV_CHANNELS, state_size), BF16)
        layer.recurrent_state = FakeTensor(RS_BASE, (1, MAX_HISTORY + 1, 1, 1, 1), F32)
        layer.device = "cuda:0"
        layer.max_history = MAX_HISTORY
        layer.pending = {}
        layer.flushed = set()
        self.layer = layer
        state = object.__new__(ns["GDNState"])
        state.slot = SLOT
        state.position = START_POSITION
        state.cache = _Cache(layer)
        state.last_history = 0
        state.exported = False
        self.state = state
        for i in range(state_size):
            self.mem.cells[ptr(CONV_BASE, i, BF16)] = 100 + i
        for r in range(MAX_HISTORY + 1):
            self.mem.cells[ptr(RS_BASE, r * layer.recurrent_state.stride(1), F32)] = () if r == 0 else None

    def row0(self) -> tuple:
        rs = self.layer.recurrent_state
        return self.mem.read(ptr(rs.data_ptr(), SLOT * rs.stride(0), F32))

    def conv(self) -> list:
        cs = self.layer.conv_state
        return [self.mem.read(ptr(cs.data_ptr(), SLOT * cs.stride(0) + i, BF16))
                for i in range(cs.shape[-1])]

    def verify_forward(self, window: int) -> list:
        """The history verify forward of window + 1 tokens on the bc path with replay_verify
        (gated_delta_net.py:1146-1164), then advance_recurrent_states. Returns the verify
        kernel's per-row outputs"""
        seqlen = window + 1
        state, layer, mem = self.state, self.layer, self.mem
        rsg = [state]
        slots = [r.slot for r in rsg]
        # gated_delta_net.py:1109-1112: no other pending verify; this slot gets fresh history
        check(not layer.pending, "a verify is already pending")
        layer.flushed.difference_update(r.slot for r in rsg)
        cs = layer.conv_state
        state_size = cs.shape[-1]

        # conv window write-back, both kernels on copies of the pre-verify buffer
        results = []
        for kernel in ("conv1d_update", "gdn_conv_ba"):
            m = Memory()
            m.cells = dict(mem.cells)
            if kernel == "conv1d_update":
                x = 0x60000     # x (bsz, dim, seqlen) bf16, gdn.cu:1843
                for s in range(seqlen):
                    m.cells[ptr(x, (0 * CONV_CHANNELS + 0) * seqlen + s, BF16)] = 200 + s
                conv1d_update_history(m, x, cs.data_ptr(), slots, CONV_CHANNELS, seqlen,
                                      state_size, K_CONV, 1)
            else:
                qkv = 0x70000   # qkv [B, S, F] fp32, gdn.cu:2407
                for s in range(seqlen):
                    m.cells[ptr(qkv, (0 * seqlen + s) * CONV_CHANNELS + 0, F32)] = 200 + s
                gdn_conv_ba_history(m, qkv, cs.data_ptr(), slots, CONV_CHANNELS, seqlen,
                                    state_size, K_CONV, 1)
            results.append([m.cells[ptr(cs.data_ptr(), i, BF16)] for i in range(state_size)])
        check(results[0] == results[1], f"conv write-backs differ: {results}")
        for i, v in enumerate(results[0]):
            mem.write(ptr(cs.data_ptr(), i, BF16), v)

        # the verify's rule inputs (replay_statics): conv_out [bsz, seqlen, F] bf16 as
        # gdn.cu:2452 stores it, g [bsz, seqlen, H] fp32, beta [bsz, seqlen, H] bf16;
        # token j's rule input is j
        conv_out = FakeTensor(QKV_BASE, (1, seqlen, F_QKV), BF16)
        g = FakeTensor(G_BASE, (1, seqlen, NUM_V_HEADS), F32)
        beta = FakeTensor(BETA_BASE, (1, seqlen, NUM_V_HEADS), BF16)
        for b in range(1):
            for s in range(seqlen):
                mem.cells[ptr(QKV_BASE, (b * seqlen + s) * F_QKV + 0, BF16)] = s   # gdn.cu:2452
                for h in range(NUM_V_HEADS):
                    mem.cells[ptr(G_BASE, (b * seqlen + s) * NUM_V_HEADS + h, F32)] = s
                    mem.cells[ptr(BETA_BASE, (b * seqlen + s) * NUM_V_HEADS + h, BF16)] = s

        # RULE_VERIFY launch (gdn.cu:1377 / 1519 / 1525 with replay_verify)
        row0_before = self.row0()
        rows = rule_verify_kernel(mem, QKV_BASE, G_BASE, BETA_BASE,
                                  layer.recurrent_state.data_ptr(), 1, seqlen, NUM_K_HEADS,
                                  NUM_V_HEADS, slots, MAX_HISTORY + 1, snoc)
        check(self.row0() == row0_before, "RULE_VERIFY wrote row 0")

        # gated_delta_net.py:1161-1163
        for row, r in enumerate(rsg):
            layer.pending[r.slot] = (conv_out, g, beta, row, seqlen)
        # recurrent_util.py:66-75 with recurrent_history (draft verify)
        self.ns["advance_recurrent_states"](
            _Ids((len(rsg), seqlen)), {"recurrent_states": rsg, "recurrent_history": True}, None)
        return rows[0]


def run_path(ns: dict, window: int, num_rejected: int, planned: bool):
    ext = Ext()
    ns["ext"] = ext
    eng = Engine(ns, ext)
    verify_rows = eng.verify_forward(window)
    refused = None
    try:
        if planned:
            # generator.py:1277, 1296-1297
            plans = ns["RewindPlans"]()
            prepared = plans.prepare(eng.state, window)
            check(prepared is not None, f"RewindPlans.prepare declined window {window}")
            plans.rewind(prepared, num_rejected)
        else:
            # generator.py:1299
            eng.state.rewind(num_rejected)
    except REFUSALS as e:
        refused = type(e).__name__
    interpret(eng.mem, ext.log, snoc)
    log_keys = [(kind, dev, tuple(j.key() for j in jobs)) for kind, dev, jobs in ext.log]
    return {
        "verify": verify_rows,
        "refused": refused,
        "log": log_keys,
        "row0": eng.row0(),
        "conv": eng.conv(),
        "pos": eng.state.position,
        "lh": eng.state.last_history,
    }


def nats(xs) -> str:
    return "[" + ",".join(str(x) for x in xs) + "]"


def engine_table(ns: dict, num_rejected) -> str:
    out = []
    for k in WINDOWS:
        verify = None
        for c in range(k + 2):
            for eos in (0, 1):
                n = num_rejected(k, c, bool(eos))
                a = run_path(ns, k, n, planned=True)
                b = run_path(ns, k, n, planned=False)
                check(a["verify"] == b["verify"], f"k={k}: verify outputs differ between paths")
                check((a["refused"] is None) == (b["refused"] is None),
                      f"k={k} c={c} eos={eos}: planned refused={a['refused']}, "
                      f"fallback refused={b['refused']}")
                check(a["log"] == b["log"], f"k={k} c={c} eos={eos}: launch logs differ\n"
                      f"  planned  {a['log']}\n  fallback {b['log']}")
                fields = ("row0", "conv", "pos", "lh") if a["refused"] is None else ("row0",)
                for f in fields:
                    check(a[f] == b[f], f"k={k} c={c} eos={eos}: {f} differs between paths")
                if verify is None:
                    verify = a["verify"]
                    out.append(f"k={k} verify=" + "".join(nats(t) for t in verify) + "\n")
                check(a["verify"] == verify, f"k={k} c={c} eos={eos}: verify outputs changed")
                head = f"k={k} c={c} eos={eos} "
                if a["refused"] is not None:
                    out.append(head + "refused row0=" + nats(a["row0"]) + "\n")
                else:
                    out.append(head + "row0=" + nats(a["row0"]) + " conv=" + nats(a["conv"]) +
                               f" pos={a['pos']} lh={a['lh']}\n")
    return "".join(out)


def bend_table() -> str:
    p = subprocess.run([BEND, TABLE], cwd=REPO, capture_output=True, text=True, timeout=300)
    check(p.returncode == 0, f"{BEND} {TABLE} exited {p.returncode}: {p.stderr.strip()}")
    try:
        s = json.loads(p.stdout.strip())
    except json.JSONDecodeError as e:
        fail(f"bend output is not one quoted string: {e}")
    check(isinstance(s, str), "bend output is not a string")
    return s


def main(argv: list) -> int:
    check(len(argv) <= 3, "usage: gdn_replay_diff.py [TREE_ROOT] [RECURRENT_UTIL_PY]")
    tree = Path(argv[1] if len(argv) > 1 else DEFAULT_TREE)
    recurrent_util = Path(argv[2] if len(argv) > 2 else DEFAULT_RECURRENT_UTIL)
    verify_pins(tree, recurrent_util)
    num_rejected = load_num_rejected(tree)
    ns = load_host(tree, recurrent_util)
    ours = engine_table(ns, num_rejected)
    theirs = bend_table()
    if ours != theirs:
        sys.stdout.writelines(difflib.unified_diff(
            theirs.splitlines(keepends=True), ours.splitlines(keepends=True),
            "GDN_REPLAY_TABLE.bend", "engine"))
        return 1
    print(f"gdn_replay_diff: {len(ours.splitlines())} rows match")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
