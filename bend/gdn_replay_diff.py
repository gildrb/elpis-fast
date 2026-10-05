#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite differential check of bend/gdn_replay.bend against the pinned ExLlamaV3.

Regenerates the table bend/GDN_REPLAY_TABLE.bend prints, but from the engine's own
code: the host job builders (RewindPlans, GDNState / GDNLayerState,
_collect_rewind_jobs, _dispatch_rewind_jobs, advance_recurrent_states) are extracted
with `ast` from the pinned sources and executed by bend/pysubset.py against stub
torch/ext objects; the launched jobs are then interpreted by line-by-line Python ports
of the CUDA integer indexing of gdn.cu. The result is compared byte for byte with the
Bend program's output. This is differential evidence on a finite instance, not a proof
of equivalence.

Usage: python3 bend/gdn_replay_diff.py TREE_ROOT RECURRENT_UTIL_PY
  TREE_ROOT: OUT/patched of bend/engine_trees.py.
  RECURRENT_UTIL_PY: OUT/patched/cache/recurrent_util.py (no patch changes it; it is
  the stock file).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pysubset
import source_link

REPO = source_link.REPO
TABLE = "bend/GDN_REPLAY_TABLE.bend"

# Pins: post-images of the full patch series, read from the manifests (never
# hardcoded here). exl3-ext patches also change generator/gdn_rewind.py and
# generator/generator.py, so their shipped post-images are in exl3-ext.json, not in
# exl3-patches.json.
MANIFEST_FILES = {
    "patches/exl3-ext/exl3-ext.json": (
        "generator/gdn_rewind.py",
        "generator/generator.py",
        "modules/gated_delta_net.py",
        "exllamav3_ext/gdn.cu",
        "exllamav3_ext/gdn.cuh",
    ),
}
# cache/recurrent_util.py is not patched by any series, so no manifest pins it: this
# is the upstream exllamav3 1.5.0 file (engine revision 355c6ee1 in exl3-patches.json
# `engine`)
RECURRENT_UTIL_SHA256 = (
    "1d6922211adf2e4e514f3a8ddaf5943259bab276ce921dfb3d730b8d4762ba66"
)
SHA256_HEX_LEN = 64

# Instance (GDN_REPLAY_TABLE.bend:3-8)
K_CONV = 4  # module.conv_kernel_size
MAX_HISTORY = 7  # state_size = K + H = 11 conv columns (gated_delta_net.py:190)
START_POSITION = 1000
WINDOWS = range(8)
SLOT = 0

# gdn.cu constants
HEAD_DIM = 128  # gdn.cu:907, 2931
CONV1D_MAX_K = 16  # gdn.cu:1828
CONV_BA_MAX_S = 16  # gdn.cu:2268
CONV_BA_THREADS = 128  # gdn.cu:2469
RULE_INPLACE, RULE_VERIFY, RULE_COMMIT = (
    0,
    2,
    3,
)  # gdn.cu:883 (RULE_HISTORY = 1 unported)
MAX_V_HEADS = 65535  # gdn.cu:2969-2970
STATE_REWIND_ALIGN = 4  # gdn.cu:2905

NUM_K_HEADS = 1
NUM_V_HEADS = 1
CONV_CHANNELS = 1  # module.fdim_qkv of the instance: one conv channel is modeled
# Width of one token row of the rule kernel's mixed_qkv staging (gdn.cu:1022 per-step
# advance, gdn.cu:2932 F); the one-channel conv instance keeps the real rule-input
# layout
F_QKV = 2 * HEAD_DIM * NUM_K_HEADS + HEAD_DIM * NUM_V_HEADS

BF16, F32 = 2, 4  # element sizes in bytes
# Symbolic, disjoint byte addresses of the device tensors
CONV_BASE = 0x0
RS_BASE = 0x10000
QKV_BASE = 0x20000
G_BASE = 0x40000
BETA_BASE = 0x50000
X_BASE = 0x60000  # conv1d_update's x (bsz, dim, seqlen) bf16, gdn.cu:1843
CONV_QKV_BASE = 0x70000  # gdn_conv_ba's qkv [B, S, F] fp32, gdn.cu:2407
CONV_INPUT0 = 200  # value of the first conv input token
CONV_STATE0 = 100  # value of the first pre-verify conv column

REFUSALS = (IndexError, AssertionError, RuntimeError)
BEND_TIMEOUT = 300
ARGC = 3  # the script and its two arguments

type Step = Callable[[object, object], object]


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"gdn_replay_diff: {msg}"
    raise SystemExit(text)


# ---------------------------------------------------------------------------------
# Pins


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file.

    Returns:
        The hex digest.

    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_files(manifest: str) -> dict[str, object]:
    """Return the `files` table of a patch manifest.

    Returns:
        The table.

    """
    files = json.loads((REPO / manifest).read_text(encoding="utf-8"))["files"]
    if not isinstance(files, dict):
        fail(f"{manifest}: no files table")
    return {str(k): v for k, v in files.items()}


def verify_pins(tree: Path, recurrent_util: Path) -> None:
    """Fail unless every quoted source has its pinned hash."""
    for manifest, keys in MANIFEST_FILES.items():
        files = manifest_files(manifest)
        for key in keys:
            if key not in files:
                fail(f"{manifest} has no pin for {key}")
            entry = files[key]
            pin = entry.get("post") if isinstance(entry, dict) else None
            if not (isinstance(pin, str) and len(pin) == SHA256_HEX_LEN):
                fail(f"{manifest}: bad post pin for {key}")
            path = tree / key
            if not path.is_file():
                fail(f"missing pinned source {path}")
            got = sha256(path)
            if got != pin:
                fail(f"{path}: sha256 {got} != post pin {pin} ({manifest})")
    got = sha256(recurrent_util)
    if got != RECURRENT_UTIL_SHA256:
        fail(f"{recurrent_util}: sha256 {got} != {RECURRENT_UTIL_SHA256}")


# ---------------------------------------------------------------------------------
# Decision: generator.py's own expression

NUM_REJECTED_TEXT = (
    "num_rejected = window + 1 - count if not eos and count <= window else 0"
)


class NumRejected(Protocol):
    """The round's decision: tokens to rewind after a verify of `window` drafts."""

    def __call__(self, window: int, count: int, *, eos: bool) -> int:
        """Return the number of rejected tokens."""
        ...


def load_num_rejected(tree: Path) -> NumRejected:
    """Return generator.py:1295 (pinned post-image), the round's decision, verbatim.

    Returns:
        The decision as a function.

    """
    path = tree / "generator/generator.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines, 1) if NUM_REJECTED_TEXT in line]
    if len(hits) != 1:
        fail(f"{path}: expected one `{NUM_REJECTED_TEXT}` line, found {hits}")
    text = lines[hits[0] - 1].strip()
    stmt = ast.parse(text).body
    if not (
        len(stmt) == 1
        and isinstance(stmt[0], ast.Assign)
        and [t.id for t in stmt[0].targets if isinstance(t, ast.Name)]
        == ["num_rejected"]
    ):
        fail(f"{path}:{hits[0]}: not a plain num_rejected assignment")
    value = ast.get_source_segment(text, stmt[0].value)
    if value is None:
        fail(f"{path}:{hits[0]}: no source for the assigned value")
    code = pysubset.compile_expr(value)

    def num_rejected(window: int, count: int, *, eos: bool) -> int:
        n = code({"window": window, "count": count, "eos": eos})
        if not isinstance(n, int):
            fail(f"{path}:{hits[0]}: num_rejected is not an int: {n!r}")
        return n

    return num_rejected


# ---------------------------------------------------------------------------------
# Host code: extracted from the pinned sources and executed against stubs


def _top(tree_ast: ast.Module, path: Path, name: str) -> ast.ClassDef | ast.FunctionDef:
    nodes = [
        n
        for n in tree_ast.body
        if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name == name
    ]
    if len(nodes) != 1:
        fail(f"{path}: expected one top-level `{name}`, found {len(nodes)}")
    return nodes[0]


def _class_subset(
    tree_ast: ast.Module, path: Path, name: str, methods: tuple[str, ...]
) -> ast.ClassDef:
    cls = _top(tree_ast, path, name)
    if not (isinstance(cls, ast.ClassDef) and not cls.bases and not cls.decorator_list):
        fail(f"{path}: `{name}` is not a plain class")
    body: list[ast.stmt] = []
    for m in methods:
        defs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == m]
        if len(defs) != 1:
            fail(f"{path}: expected one {name}.{m}, found {len(defs)}")
        body.append(defs[0])
    sub = ast.ClassDef(
        name=cls.name,
        bases=[],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )
    ast.copy_location(sub, cls)
    return sub


def _exec_nodes(nodes: list[ast.stmt], ns: dict[str, object]) -> None:
    # The sources start with `from __future__ import annotations`: pysubset never
    # evaluates annotations
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    pysubset.exec_block(mod, ns)


@dataclass(frozen=True)
class _Device:
    index: int


class _Torch:
    """torch.device(x).index is the only torch use on the exercised host paths."""

    @staticmethod
    def device(x: object) -> _Device:
        """Return the stub device of "cuda:0".

        Returns:
            The device.

        """
        if x != "cuda:0":
            fail(f"unexpected device {x!r}")
        return _Device(0)


class FakeTensor:
    """A contiguous tensor over a symbolic byte address.

    Supports data_ptr / stride / shape / element_size.
    """

    def __init__(self, base: int, shape: tuple[int, ...], element_size: int) -> None:
        """Lay out `shape` contiguously at byte address `base`."""
        self._base = base
        self.shape = tuple(shape)
        self._es = element_size
        strides: list[int] = []
        acc = 1
        for n in reversed(self.shape):
            strides.append(acc)
            acc *= n
        self._strides = tuple(reversed(strides))

    def data_ptr(self) -> int:
        """Return the base address.

        Returns:
            The address.

        """
        return self._base

    def stride(self, dim: int) -> int:
        """Return the stride of `dim` in elements.

        Returns:
            The stride.

        """
        return self._strides[dim]

    def element_size(self) -> int:
        """Return the element size in bytes.

        Returns:
            The size.

        """
        return self._es


# Job descriptors: field order of the gdn.cuh constructors (gdn.cuh:284, 298, 319-320)


def job_ints(name: str, fields: tuple[str, ...], args: tuple[object, ...]) -> list[int]:
    """Check a job constructor's arguments: one int per field.

    Returns:
        The arguments.

    """
    if len(args) != len(fields):
        fail(f"{name} takes {len(fields)} arguments")
    out: list[int] = []
    for f, v in zip(fields, args, strict=True):
        if not isinstance(v, int):
            fail(f"{name}.{f} not an int: {v!r}")
        out.append(v)
    return out


class ConvRewindJob:
    """ext.ConvRewindJob."""

    FIELDS = ("src", "dst", "dim", "cdim", "stride")

    def __init__(self, *args: object) -> None:
        """Hold the constructor's fields."""
        self.src, self.dst, self.dim, self.cdim, self.stride = job_ints(
            "ConvRewindJob", self.FIELDS, args
        )

    def key(self) -> tuple[object, ...]:
        """Return the job's identity.

        Returns:
            The type name and the fields.

        """
        return ("ConvRewindJob", self.src, self.dst, self.dim, self.cdim, self.stride)


class StateRewindJob:
    """ext.StateRewindJob."""

    FIELDS = ("src", "dst", "num_elements")

    def __init__(self, *args: object) -> None:
        """Hold the constructor's fields."""
        self.src, self.dst, self.num_elements = job_ints(
            "StateRewindJob", self.FIELDS, args
        )

    def key(self) -> tuple[object, ...]:
        """Return the job's identity.

        Returns:
            The type name and the fields.

        """
        return ("StateRewindJob", self.src, self.dst, self.num_elements)


class StateReplayJob:
    """ext.StateReplayJob."""

    FIELDS = (
        "mixed_qkv",
        "g",
        "beta",
        "state",
        "row",
        "seqlen",
        "steps",
        "num_k_heads",
        "num_v_heads",
    )

    def __init__(self, *args: object) -> None:
        """Hold the constructor's fields."""
        (
            self.mixed_qkv,
            self.g,
            self.beta,
            self.state,
            self.row,
            self.seqlen,
            self.steps,
            self.num_k_heads,
            self.num_v_heads,
        ) = job_ints("StateReplayJob", self.FIELDS, args)

    def key(self) -> tuple[object, ...]:
        """Return the job's identity.

        Returns:
            The type name and the fields.

        """
        return (
            "StateReplayJob",
            self.mixed_qkv,
            self.g,
            self.beta,
            self.state,
            self.row,
            self.seqlen,
            self.steps,
            self.num_k_heads,
            self.num_v_heads,
        )


type Job = ConvRewindJob | StateRewindJob | StateReplayJob
type Launch = tuple[str, object, tuple[Job, ...]]


def job_list(jobs: object, what: str) -> list[object]:
    """Return a launcher's job list argument as a list.

    Returns:
        The jobs.

    """
    if not isinstance(jobs, Iterable):
        fail(f"{what}: jobs is not a list")
    return list(jobs)


class Ext:
    """exllamav3_ext stub: the host halves of the batched launchers.

    Their TORCH_CHECKs run here; each launch is appended to a log that the device
    ports below interpret.
    """

    ConvRewindJob = ConvRewindJob
    StateRewindJob = StateRewindJob
    StateReplayJob = StateReplayJob

    def __init__(self) -> None:
        """Start with an empty launch log."""
        self.log: list[Launch] = []

    def batched_conv_rewind(self, jobs: object, device_index: object) -> None:
        """Check and log a batched conv rewind launch.

        Raises:
            RuntimeError: A job's cdim exceeds CONV1D_MAX_K.

        """
        if not jobs:  # gdn.cu:2867
            return
        checked: list[Job] = []
        for j in job_list(jobs, "batched_conv_rewind"):
            if type(j) is not ConvRewindJob:
                fail("batched_conv_rewind: not a ConvRewindJob")
            if not j.cdim <= CONV1D_MAX_K:  # gdn.cu:2880
                msg = "batched_conv_rewind: cdim exceeds CONV1D_MAX_K"
                raise RuntimeError(msg)
            checked.append(j)
        self.log.append(("conv_rewind", device_index, tuple(checked)))

    def batched_state_rewind(self, jobs: object, device_index: object) -> None:
        """Check and log a batched state rewind launch.

        Raises:
            RuntimeError: A job's num_elements is not a multiple of 4.

        """
        if not jobs:  # gdn.cu:2892
            return
        checked: list[Job] = []
        for j in job_list(jobs, "batched_state_rewind"):
            if type(j) is not StateRewindJob:
                fail("batched_state_rewind: not a StateRewindJob")
            if j.num_elements % STATE_REWIND_ALIGN != 0:  # gdn.cu:2905
                msg = "batched_state_rewind: num_elements must be a multiple of 4"
                raise RuntimeError(msg)
            checked.append(j)
        self.log.append(("state_rewind", device_index, tuple(checked)))

    def batched_state_replay(self, jobs: object, device_index: object) -> None:
        """Check and log a batched state replay launch."""
        if not jobs:  # gdn.cu:2953
            return
        checked: list[Job] = []
        for j in job_list(jobs, "batched_state_replay"):
            if type(j) is not StateReplayJob:
                fail("batched_state_replay: not a StateReplayJob")
            check_replay_job(j)
            checked.append(j)
        self.log.append(("state_replay", device_index, tuple(checked)))


def check_replay_job(j: StateReplayJob) -> None:
    """Run the TORCH_CHECKs of batched_state_replay on one job.

    Raises:
        RuntimeError: A check fails.

    """
    # at gdn.cu:2966
    if not (j.mixed_qkv and j.g and j.beta and j.state):
        msg = "batched_state_replay: null pointer"
        raise RuntimeError(msg)
    # at gdn.cu:2967-2968
    if not (j.steps >= 1 and j.steps <= j.seqlen and j.row >= 0):
        msg = "batched_state_replay: steps must be in 1..seqlen"
        raise RuntimeError(msg)
    # at gdn.cu:2969-2970
    if not (
        j.num_k_heads >= 1
        and j.num_v_heads % j.num_k_heads == 0
        and j.num_v_heads <= MAX_V_HEADS
    ):
        msg = "batched_state_replay: bad head counts"
        raise RuntimeError(msg)


def load_host(tree: Path, recurrent_util: Path) -> dict[str, object]:
    """Run the quoted host code of the pinned sources against the stubs.

    Returns:
        The namespace with the host definitions.

    """
    ns: dict[str, object] = {"__name__": "gdn_replay_host", "torch": _Torch}
    rewind_py = tree / "generator/gdn_rewind.py"
    gdn_py = tree / "modules/gated_delta_net.py"
    a = ast.parse(rewind_py.read_text(encoding="utf-8"))
    b = ast.parse(gdn_py.read_text(encoding="utf-8"))
    util_text = recurrent_util.read_text(encoding="utf-8")
    u = ast.parse(util_text)
    # gated_delta_net.py:31-59, 62-69, 78-128 (GDNState), 179-368 (GDNLayerState)
    _exec_nodes(
        [
            _top(b, gdn_py, "_collect_rewind_jobs"),
            _top(b, gdn_py, "_dispatch_rewind_jobs"),
            _class_subset(b, gdn_py, "GDNState", ("rewind", "post_advance")),
            _class_subset(
                b,
                gdn_py,
                "GDNLayerState",
                (
                    "replay_job",
                    "check_history_rows",
                    "rewind",
                    "rewind_conv_job",
                    "rewind_state_job",
                    "rewind_replay_job",
                ),
            ),
        ],
        ns,
    )
    # from gdn_rewind.py:18-112 (_ReplayView, RewindPlans)
    _exec_nodes(
        [_top(a, rewind_py, "_ReplayView"), _top(a, rewind_py, "RewindPlans")], ns
    )
    # recurrent_util.py:66-75 (advance_recurrent_states; 73-74 advance position,
    # last_history)
    adv = _top(u, recurrent_util, "advance_recurrent_states")
    src = ast.get_source_segment(util_text, adv) or ""
    for stmt in (
        "r.position += seqlen",
        "r.last_history = (seqlen - 1) if history else 0",
    ):
        if stmt not in src:
            fail(f"{recurrent_util}: advance_recurrent_states lacks `{stmt}`")
    _exec_nodes([adv], ns)
    return ns


# ---------------------------------------------------------------------------------
# Device memory: byte address -> value. Rows of the recurrent state are collapsed to
# one cell (the trace of replayed rule inputs); any read of an unwritten address
# fails closed.


class Memory:
    """Symbolic device memory: byte address -> value."""

    def __init__(self) -> None:
        """Start with no mapped address."""
        self.cells: dict[int, object] = {}

    def read(self, addr: int) -> object:
        """Return the value at a mapped address.

        Returns:
            The value.

        """
        if addr not in self.cells:
            fail(f"device read of unmapped address {addr:#x}")
        return self.cells[addr]

    def write(self, addr: int, value: object) -> None:
        """Write a mapped address."""
        if addr not in self.cells:
            fail(f"device write to unmapped address {addr:#x}")
        self.cells[addr] = value


def ptr(base: int, index: int, es: int) -> int:
    """Return typed pointer arithmetic: (T*) base + index.

    Returns:
        The byte address.

    """
    return base + index * es


# ---------------------------------------------------------------------------------
# Device ports (gdn.cu, pinned post-image)


@dataclass(frozen=True)
class ConvWriteBack:
    """The arguments of a HISTORY conv launch (conv1d_update or gdn_conv_ba)."""

    x: int  # conv input base address
    conv_state: int
    slots: list[int]
    channels: int  # conv1d_update's dim, gdn_conv_ba's F
    seqlen: int  # gdn_conv_ba's S
    state_size: int
    k: int  # kernel size K
    batch: int  # conv1d_update's bsz, gdn_conv_ba's B


def conv1d_update_history(mem: Memory, p: ConvWriteBack) -> None:
    """conv1d_update_kernel<ACT, HISTORY = true> (gdn.cu:1839-1925): the write-back."""
    for b in range(p.batch):  # blockIdx.y
        for d in range(p.channels):  # gdn.cu:1855-1856
            slot = p.slots[b] if p.slots else b  # gdn.cu:1858
            x_d = ptr(p.x, (b * p.channels + d) * p.seqlen, BF16)  # gdn.cu:1860
            # at gdn.cu:1861
            state_d = ptr(p.conv_state, (slot * p.channels + d) * p.state_size, BF16)
            old_state: list[object] = [None] * CONV1D_MAX_K
            for k in range(CONV1D_MAX_K):  # gdn.cu:1874-1875
                if k < p.k:
                    old_state[k] = mem.read(ptr(state_d, k, BF16))
            total = p.k + p.seqlen  # gdn.cu:1914
            write_size = min(total, p.state_size)  # gdn.cu:1915
            dst_start = p.state_size - write_size  # gdn.cu:1916
            src_start = total - write_size  # gdn.cu:1917
            for j in range(write_size):  # gdn.cu:1918
                src_t = src_start + j  # gdn.cu:1920
                # at gdn.cu:1921
                v = (
                    old_state[src_t]
                    if src_t < p.k
                    else mem.read(ptr(x_d, src_t - p.k, BF16))
                )
                mem.write(ptr(state_d, dst_start + j, BF16), v)  # gdn.cu:1922


def conv_ba_stage(
    mem: Memory, p: ConvWriteBack, state_c: int, b: int, channels: int
) -> list[list[object]]:
    """Return gdn_conv_ba's sh_seq of batch row `b` (gdn.cu:2406-2442).

    Returns:
        Per channel: the K state columns, then the S inputs.

    """
    d0 = 0  # one block of channels
    sh_seq: list[list[object]] = [[None] * (p.k + p.seqlen) for _ in range(channels)]
    for t in range(channels):  # threadIdx.x, d = d0 + t
        d = d0 + t
        state_d = ptr(state_c, t * p.state_size, BF16)  # gdn.cu:2406
        x_d = ptr(p.x, b * p.seqlen * p.channels + d, F32)  # gdn.cu:2407
        seq = sh_seq[t]  # gdn.cu:2408
        for kk in range(p.k):  # gdn.cu:2421-2424
            seq[kk] = mem.read(ptr(state_d, kk, BF16))
        # at gdn.cu:2434-2435
        xb = [mem.read(ptr(x_d, s * p.channels, F32)) for s in range(p.seqlen)]
        for s in range(p.seqlen):  # gdn.cu:2438-2442
            seq[p.k + s] = xb[s]
    return sh_seq


def gdn_conv_ba_history(mem: Memory, p: ConvWriteBack) -> None:
    """gdn_conv_ba_kernel<HISTORY = true> (gdn.cu:2275-2476): staging and write-back."""
    if p.seqlen > CONV_BA_MAX_S:  # gdn.cu:2529
        fail("gdn_conv_ba: too many rows")
    if p.state_size < p.k:  # gdn.cu:2535
        fail("conv_state must have at least K entries")
    # HISTORY = true: the gdn.cu:2466-2467 branches for HISTORY = false are not ported
    d0 = 0  # one block of channels
    for b in range(p.batch):
        slot = p.slots[b] if p.slots else b  # gdn.cu:2401
        state_c = ptr(p.conv_state, (slot * p.channels + d0) * p.state_size, BF16)
        # gdn.cu:2402 above; gdn.cu:2469 (CONV_BA_THREADS)
        channels = min(CONV_BA_THREADS, p.channels - d0)
        sh_seq = conv_ba_stage(mem, p, state_c, b, channels)
        total = p.k + p.seqlen  # gdn.cu:2465
        write_size = min(total, p.state_size)  # gdn.cu:2466
        dst_start = p.state_size - write_size  # gdn.cu:2467
        src_start = total - write_size  # gdn.cu:2468
        for e in range(channels * write_size):  # gdn.cu:2470
            c = e // write_size  # gdn.cu:2472
            j = e - c * write_size  # gdn.cu:2473
            # at gdn.cu:2474
            mem.write(
                ptr(state_c, c * p.state_size + dst_start + j, BF16),
                sh_seq[c][src_start + j],
            )


@dataclass(frozen=True)
class RuleLaunch:
    """The arguments of gated_delta_rule_128_reg<MODE> for one head."""

    mode: int
    mixed_qkv: int
    g: int
    beta: int
    slot_state: int
    steps: int
    num_k_heads: int
    num_v_heads: int
    state_size: int
    head: int


def gated_delta_rule_128_reg(
    mem: Memory, p: RuleLaunch, step: Step
) -> tuple[list[object], list[int]]:
    """gated_delta_rule_128_reg<MODE> (gdn.cu:889-1032), the lane of thread elem = 0.

    The state row is collapsed to one cell; `step` is the recurrent update.

    Returns:
        The per-step outputs and the element offsets of mixed_qkv each step read its
        token from.

    """
    if p.mode not in {RULE_VERIFY, RULE_COMMIT}:
        fail(f"unported rule mode {p.mode}")
    group = p.num_v_heads // p.num_k_heads  # gdn.cu:913
    k_head = p.head // group  # gdn.cu:918
    elem = 0  # gdn.cu:933 (collapsed)
    state = mem.read(ptr(p.slot_state, elem, F32))  # gdn.cu:936
    mixed_qkv, g, beta = p.mixed_qkv, p.g, p.beta
    outs: list[object] = []
    reads: list[int] = []
    for s in range(p.steps):  # gdn.cu:938
        gl_q = ptr(mixed_qkv, k_head * HEAD_DIM, BF16)  # gdn.cu:940
        x = mem.read(gl_q)  # gdn.cu:944 (token s's input)
        g_h = mem.read(ptr(g, p.head, F32))  # gdn.cu:990
        beta_h = mem.read(ptr(beta, p.head, BF16))  # gdn.cu:991
        if not (g_h == x and beta_h == x):
            fail(f"rule step {s}: q/g/beta rows disagree ({x}, {g_h}, {beta_h})")
        reads.append((gl_q - QKV_BASE) // BF16)
        state = step(state, x)  # gdn.cu:1004
        # gdn.cu:1005: history rows are written only by RULE_HISTORY
        if p.mode != RULE_COMMIT:  # gdn.cu:1012
            outs.append(state)  # gdn.cu:1014-1019 (row s)
        # at gdn.cu:1022
        mixed_qkv = ptr(
            mixed_qkv, 2 * HEAD_DIM * p.num_k_heads + HEAD_DIM * p.num_v_heads, BF16
        )
        g = ptr(g, p.num_v_heads, F32)  # gdn.cu:1023
        beta = ptr(beta, p.num_v_heads, BF16)  # gdn.cu:1024
    if p.mode in {RULE_INPLACE, RULE_COMMIT}:  # gdn.cu:1027
        mem.write(ptr(p.slot_state, elem, F32), state)  # gdn.cu:1030
    return outs, reads


@dataclass(frozen=True)
class VerifyLaunch:
    """The arguments of the RULE_VERIFY kernel launch."""

    mixed_qkv: int
    g: int
    beta: int
    recurrent_state: int
    bsz: int
    seqlen: int
    num_k_heads: int
    num_v_heads: int
    slots: list[int]
    history_stride: int


def rule_verify_kernel(mem: Memory, p: VerifyLaunch, step: Step) -> list[list[object]]:
    """cuda_recurrent_gated_delta_rule_kernel_128_reg<RULE_VERIFY> (gdn.cu:1034-1074).

    Returns:
        The per-step outputs of each batch row.

    """
    group = p.num_v_heads // p.num_k_heads  # gdn.cu:1057
    state_size = 1  # gdn.cu:1058, collapsed row
    slot_size = p.history_stride * state_size  # gdn.cu:1059
    rows: list[list[object]] = []
    for bi in range(p.bsz):  # gdn.cu:1061
        token_width = 3 * HEAD_DIM * p.num_k_heads + HEAD_DIM * (
            p.num_v_heads - p.num_k_heads
        )
        qkv = ptr(p.mixed_qkv, bi * p.seqlen * token_width, BF16)  # gdn.cu:1062
        gg = ptr(p.g, bi * p.seqlen * (group * p.num_k_heads), F32)  # gdn.cu:1063
        bb = ptr(p.beta, bi * p.seqlen * (group * p.num_k_heads), BF16)  # 1064
        state_slot = p.slots[bi] if p.slots else bi  # gdn.cu:1065
        slot_state = ptr(p.recurrent_state, state_slot * slot_size, F32)  # 1066
        outs, reads = gated_delta_rule_128_reg(  # gdn.cu:1069-1073
            mem,
            RuleLaunch(
                mode=RULE_VERIFY,
                mixed_qkv=qkv,
                g=gg,
                beta=bb,
                slot_state=slot_state,
                steps=p.seqlen,
                num_k_heads=p.num_k_heads,
                num_v_heads=p.num_v_heads,
                state_size=state_size,
                head=0,
            ),
            step,
        )
        if reads != [(bi * p.seqlen + s) * F_QKV for s in range(p.seqlen)]:
            fail(f"verify read token rows {reads}")
        rows.append(outs)
    return rows


def batched_conv_rewind_kernel(mem: Memory, jobs: tuple[Job, ...]) -> None:
    """batched_conv_rewind_kernel (gdn.cu:2826-2846)."""
    for j in jobs:  # gdn.cu:2829-2831
        if not isinstance(j, ConvRewindJob):
            fail(f"conv rewind launch with {j.key()}")
        for d in range(j.dim):  # gdn.cu:2833-2834
            s = ptr(j.src, d * j.stride, BF16)  # gdn.cu:2836
            t = ptr(j.dst, d * j.stride, BF16)  # gdn.cu:2837
            reg: list[object] = [None] * CONV1D_MAX_K
            for k in range(CONV1D_MAX_K):  # gdn.cu:2841-2842
                if k < j.cdim:
                    reg[k] = mem.read(ptr(s, k, BF16))
            for k in range(CONV1D_MAX_K):  # gdn.cu:2844-2845
                if k < j.cdim:
                    mem.write(ptr(t, k, BF16), reg[k])


def batched_state_replay_kernel(mem: Memory, jobs: tuple[Job, ...], step: Step) -> None:
    """batched_state_replay_kernel (gdn.cu:2923-2949) into RULE_COMMIT."""
    for j in jobs:  # gdn.cu:2926-2928
        if not isinstance(j, StateReplayJob):
            fail(f"state replay launch with {j.key()}")
        # at gdn.cu:2932
        f = 2 * HEAD_DIM * j.num_k_heads + HEAD_DIM * j.num_v_heads
        if f != F_QKV:
            fail(f"replay F {f} != staging width {F_QKV}")
        row_tokens = j.row * j.seqlen  # gdn.cu:2933
        for head in range(j.num_v_heads):  # blockIdx.y, gdn.cu:2929
            _, reads = gated_delta_rule_128_reg(  # gdn.cu:2934-2948
                mem,
                RuleLaunch(
                    mode=RULE_COMMIT,
                    mixed_qkv=ptr(j.mixed_qkv, row_tokens * f, BF16),  # gdn.cu:2936
                    g=ptr(j.g, row_tokens * j.num_v_heads, F32),  # gdn.cu:2937
                    beta=ptr(j.beta, row_tokens * j.num_v_heads, BF16),  # gdn.cu:2938
                    slot_state=j.state,  # gdn.cu:2939
                    steps=j.steps,  # gdn.cu:2941
                    num_k_heads=j.num_k_heads,
                    num_v_heads=j.num_v_heads,
                    state_size=j.num_v_heads * HEAD_DIM * HEAD_DIM,  # gdn.cu:2945
                    head=head,
                ),
                step,
            )
            # token index read at step s: (offset / F) - row * seqlen
            for s, off in enumerate(reads):
                if not (off % f == 0 and off // f - j.row * j.seqlen == s):
                    fail(f"replay step {s} read element {off}")


def interpret(mem: Memory, log: list[Launch], step: Step) -> None:
    """Run the logged launches on the device ports."""
    for kind, device_index, jobs in log:
        if device_index != 0:
            fail(f"launch on device {device_index}")
        if kind == "conv_rewind":
            batched_conv_rewind_kernel(mem, jobs)
        elif kind == "state_replay":
            batched_state_replay_kernel(mem, jobs, step)
        else:
            fail(f"unexpected {kind} launch after a history-free verify")


# ---------------------------------------------------------------------------------
# One round of the instance


def snoc(state: object, x: object) -> object:
    """Return the instance's recurrent update: the trace of consumed rule inputs.

    Returns:
        The state with `x` appended.

    """
    if not isinstance(state, tuple):
        fail(f"recurrent state {state!r} is not a trace")
    return (*state, x)


class _Model:
    loaded_tp = False


class _Cache:
    def __init__(self, layer: object) -> None:
        self.model = _Model()
        self._layer = layer

    def get_all_recurrent_layers(self) -> dict[int, object]:
        return {0: self._layer}


class _Module:
    conv_kernel_size = K_CONV
    num_k_heads = NUM_K_HEADS
    num_v_heads = NUM_V_HEADS
    fdim_qkv = CONV_CHANNELS


@dataclass(frozen=True)
class _Ids:
    shape: tuple[int, int]


@runtime_checkable
class HostLayer(Protocol):
    """The GDNLayerState fields the link sets and reads."""

    conv_state: FakeTensor
    recurrent_state: FakeTensor
    pending: dict[int, tuple[object, ...]]
    flushed: set[int]


@runtime_checkable
class HostState(Protocol):
    """The GDNState fields and method the link uses."""

    slot: int
    position: int
    last_history: int

    def rewind(self, num_tokens: int) -> object:
        """Rewind the last `num_tokens` tokens."""
        ...


@runtime_checkable
class HostPlans(Protocol):
    """The RewindPlans methods the link calls."""

    def prepare(self, state: object, window: int) -> object:
        """Plan the rewind of a verify round."""
        ...

    def rewind(self, prepared: object, num_tokens: int) -> object:
        """Launch a planned rewind."""
        ...


def host_instance(ns: dict[str, object], name: str, **fields: object) -> object:
    """Allocate an instance of the host class `name` without __init__, set `fields`.

    Returns:
        The instance.

    """
    cls = ns[name]
    if not isinstance(cls, type):
        fail(f"{name} is not a class")
    obj = object.__new__(cls)
    for field, value in fields.items():
        setattr(obj, field, value)
    return obj


def host_call(ns: dict[str, object], name: str, *args: object) -> object:
    """Call the host function or class `name`.

    Returns:
        The result.

    """
    fn = ns[name]
    if not callable(fn):
        fail(f"{name} is not callable")
    return fn(*args)


class Engine:
    """One slot and one GDN layer: device memory, host state objects, the launch log."""

    def __init__(self, ns: dict[str, object], ext: Ext) -> None:
        """Set up the instance's layer and state on fresh device memory."""
        self.ns = ns
        self.ext = ext
        self.mem = Memory()
        state_size = K_CONV + MAX_HISTORY
        # gated_delta_net.py:189-198: (max_batch_size, fdim_qkv, K + H) bf16 and
        # (max_batch_size, H + 1, nv, hk, hv) fp32 with the head block collapsed to
        # one cell
        layer = host_instance(
            ns,
            "GDNLayerState",
            module=_Module(),
            conv_state=FakeTensor(CONV_BASE, (1, CONV_CHANNELS, state_size), BF16),
            recurrent_state=FakeTensor(RS_BASE, (1, MAX_HISTORY + 1, 1, 1, 1), F32),
            device="cuda:0",
            max_history=MAX_HISTORY,
            pending={},
            flushed=set(),
        )
        if not isinstance(layer, HostLayer):
            fail("GDNLayerState instance lacks its fields")
        self.layer = layer
        state = host_instance(
            ns,
            "GDNState",
            slot=SLOT,
            position=START_POSITION,
            cache=_Cache(layer),
            last_history=0,
            exported=False,
        )
        if not isinstance(state, HostState):
            fail("GDNState instance lacks its fields")
        self.state = state
        for i in range(state_size):
            self.mem.cells[ptr(CONV_BASE, i, BF16)] = CONV_STATE0 + i
        for r in range(MAX_HISTORY + 1):
            row = ptr(RS_BASE, r * layer.recurrent_state.stride(1), F32)
            self.mem.cells[row] = () if r == 0 else None

    def row0(self) -> object:
        """Return row 0 of the slot's recurrent state.

        Returns:
            The row's trace.

        """
        rs = self.layer.recurrent_state
        return self.mem.read(ptr(rs.data_ptr(), SLOT * rs.stride(0), F32))

    def conv(self) -> list[object]:
        """Return the slot's conv state columns.

        Returns:
            The columns.

        """
        cs = self.layer.conv_state
        return [
            self.mem.read(ptr(cs.data_ptr(), SLOT * cs.stride(0) + i, BF16))
            for i in range(cs.shape[-1])
        ]

    def conv_write_back(self, seqlen: int, slots: list[int]) -> None:
        """Run both HISTORY conv kernels on copies of the pre-verify buffer."""
        mem = self.mem
        cs = self.layer.conv_state
        state_size = cs.shape[-1]
        results = []
        for kernel in ("conv1d_update", "gdn_conv_ba"):
            m = Memory()
            m.cells = dict(mem.cells)
            if kernel == "conv1d_update":
                x = X_BASE
                for s in range(seqlen):
                    addr = ptr(x, (0 * CONV_CHANNELS + 0) * seqlen + s, BF16)
                    m.cells[addr] = CONV_INPUT0 + s
                conv1d_update_history(
                    m,
                    ConvWriteBack(
                        x=x,
                        conv_state=cs.data_ptr(),
                        slots=slots,
                        channels=CONV_CHANNELS,
                        seqlen=seqlen,
                        state_size=state_size,
                        k=K_CONV,
                        batch=1,
                    ),
                )
            else:
                qkv = CONV_QKV_BASE
                for s in range(seqlen):
                    addr = ptr(qkv, (0 * seqlen + s) * CONV_CHANNELS + 0, F32)
                    m.cells[addr] = CONV_INPUT0 + s
                gdn_conv_ba_history(
                    m,
                    ConvWriteBack(
                        x=qkv,
                        conv_state=cs.data_ptr(),
                        slots=slots,
                        channels=CONV_CHANNELS,
                        seqlen=seqlen,
                        state_size=state_size,
                        k=K_CONV,
                        batch=1,
                    ),
                )
            results.append([
                m.cells[ptr(cs.data_ptr(), i, BF16)] for i in range(state_size)
            ])
        if results[0] != results[1]:
            fail(f"conv write-backs differ: {results}")
        for i, v in enumerate(results[0]):
            mem.write(ptr(cs.data_ptr(), i, BF16), v)

    def verify_forward(self, window: int) -> list[object]:
        """Run the history verify forward of window + 1 tokens.

        The bc path with replay_verify (gated_delta_net.py:1146-1164), then
        advance_recurrent_states.

        Returns:
            The verify kernel's per-row outputs.

        """
        seqlen = window + 1
        state, layer, mem = self.state, self.layer, self.mem
        rsg = [state]
        slots = [r.slot for r in rsg]
        # gated_delta_net.py:1109-1112: no other pending verify; this slot gets fresh
        # history
        if layer.pending:
            fail("a verify is already pending")
        layer.flushed.difference_update(r.slot for r in rsg)

        # conv window write-back, both kernels on copies of the pre-verify buffer
        self.conv_write_back(seqlen, slots)

        # the verify's rule inputs (replay_statics): conv_out [bsz, seqlen, F] bf16 as
        # gdn.cu:2452 stores it, g [bsz, seqlen, H] fp32, beta [bsz, seqlen, H] bf16;
        # token j's rule input is j
        conv_out = FakeTensor(QKV_BASE, (1, seqlen, F_QKV), BF16)
        g = FakeTensor(G_BASE, (1, seqlen, NUM_V_HEADS), F32)
        beta = FakeTensor(BETA_BASE, (1, seqlen, NUM_V_HEADS), BF16)
        for b in range(1):
            for s in range(seqlen):
                # at gdn.cu:2452
                mem.cells[ptr(QKV_BASE, (b * seqlen + s) * F_QKV + 0, BF16)] = s
                for h in range(NUM_V_HEADS):
                    head_index = (b * seqlen + s) * NUM_V_HEADS + h
                    mem.cells[ptr(G_BASE, head_index, F32)] = s
                    mem.cells[ptr(BETA_BASE, head_index, BF16)] = s

        # RULE_VERIFY launch (gdn.cu:1377 / 1519 / 1525 with replay_verify)
        row0_before = self.row0()
        rows = rule_verify_kernel(
            mem,
            VerifyLaunch(
                mixed_qkv=QKV_BASE,
                g=G_BASE,
                beta=BETA_BASE,
                recurrent_state=layer.recurrent_state.data_ptr(),
                bsz=1,
                seqlen=seqlen,
                num_k_heads=NUM_K_HEADS,
                num_v_heads=NUM_V_HEADS,
                slots=slots,
                history_stride=MAX_HISTORY + 1,
            ),
            snoc,
        )
        if self.row0() != row0_before:
            fail("RULE_VERIFY wrote row 0")

        # as gated_delta_net.py:1161-1163
        for row, r in enumerate(rsg):
            layer.pending[r.slot] = (conv_out, g, beta, row, seqlen)
        # recurrent_util.py:66-75 with recurrent_history (draft verify)
        host_call(
            self.ns,
            "advance_recurrent_states",
            _Ids((len(rsg), seqlen)),
            {"recurrent_states": rsg, "recurrent_history": True},
            None,
        )
        return rows[0]


@dataclass(frozen=True)
class PathResult:
    """The outcome of one rewind path of a round."""

    verify: list[object]
    refused: str | None
    log: list[tuple[str, object, tuple[tuple[object, ...], ...]]]
    row0: object
    conv: list[object]
    pos: int
    lh: int

    def fields(self) -> dict[str, object]:
        """Return the compared state fields by name.

        Returns:
            row0, conv, pos and lh.

        """
        return {"row0": self.row0, "conv": self.conv, "pos": self.pos, "lh": self.lh}


def rewind(
    ns: dict[str, object], eng: Engine, window: int, num_rejected: int, *, planned: bool
) -> None:
    """Rewind the round: by RewindPlans (planned) or by GDNState.rewind."""
    if planned:
        # as generator.py:1277, 1296-1297
        plans = host_call(ns, "RewindPlans")
        if not isinstance(plans, HostPlans):
            fail("RewindPlans lacks prepare / rewind")
        prepared = plans.prepare(eng.state, window)
        if prepared is None:
            fail(f"RewindPlans.prepare declined window {window}")
        plans.rewind(prepared, num_rejected)
    else:
        # as generator.py:1299
        eng.state.rewind(num_rejected)


def run_path(
    ns: dict[str, object], window: int, num_rejected: int, *, planned: bool
) -> PathResult:
    """Run one verify round and its rewind, planned or by GDNState.rewind.

    Returns:
        The outcome.

    """
    ext = Ext()
    ns["ext"] = ext
    eng = Engine(ns, ext)
    verify_rows = eng.verify_forward(window)
    refused = None
    try:
        rewind(ns, eng, window, num_rejected, planned=planned)
    except REFUSALS as e:
        refused = type(e).__name__
    interpret(eng.mem, ext.log, snoc)
    log_keys = [
        (kind, dev, tuple(j.key() for j in jobs)) for kind, dev, jobs in ext.log
    ]
    return PathResult(
        verify=verify_rows,
        refused=refused,
        log=log_keys,
        row0=eng.row0(),
        conv=eng.conv(),
        pos=eng.state.position,
        lh=eng.state.last_history,
    )


def nats(xs: object) -> str:
    """Return a trace or column list in the Bend table's list syntax.

    Returns:
        The list text.

    """
    if not isinstance(xs, tuple | list):
        fail(f"{xs!r} is not a list")
    return "[" + ",".join(str(x) for x in xs) + "]"


def compare_paths(a: PathResult, b: PathResult, where: str) -> None:
    """Fail unless the planned and the fallback rewind agree."""
    if a.verify != b.verify:
        fail(f"{where.split(' ', maxsplit=1)[0]}: verify outputs differ between paths")
    if (a.refused is None) != (b.refused is None):
        fail(f"{where}: planned refused={a.refused}, fallback refused={b.refused}")
    if a.log != b.log:
        fail(f"{where}: launch logs differ\n  planned  {a.log}\n  fallback {b.log}")
    fields = ("row0", "conv", "pos", "lh") if a.refused is None else ("row0",)
    af, bf = a.fields(), b.fields()
    for f in fields:
        if af[f] != bf[f]:
            fail(f"{where}: {f} differs between paths")


def engine_table(ns: dict[str, object], num_rejected: NumRejected) -> str:
    """Regenerate the Bend table from the engine's code.

    Returns:
        The table text.

    """
    out: list[str] = []
    for k in WINDOWS:
        verify = None
        for c in range(k + 2):
            for eos in (0, 1):
                n = num_rejected(k, c, eos=bool(eos))
                a = run_path(ns, k, n, planned=True)
                b = run_path(ns, k, n, planned=False)
                where = f"k={k} c={c} eos={eos}"
                compare_paths(a, b, where)
                if verify is None:
                    verify = a.verify
                    out.append(
                        f"k={k} verify=" + "".join(nats(t) for t in verify) + "\n"
                    )
                if a.verify != verify:
                    fail(f"{where}: verify outputs changed")
                head = f"{where} "
                if a.refused is not None:
                    out.append(head + "refused row0=" + nats(a.row0) + "\n")
                else:
                    out.append(
                        head
                        + "row0="
                        + nats(a.row0)
                        + " conv="
                        + nats(a.conv)
                        + f" pos={a.pos} lh={a.lh}\n"
                    )
    return "".join(out)


def bend_table() -> str:
    """Run the Bend table program.

    Returns:
        Its output string.

    """
    bend = source_link.bend()
    p = source_link.run(
        [bend, TABLE],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=BEND_TIMEOUT,
        check=False,
        cpu_heavy=False,
    )
    if p.returncode != 0:
        fail(f"{bend} {TABLE} exited {p.returncode}: {p.stderr.strip()}")
    try:
        s = json.loads(p.stdout.strip())
    except json.JSONDecodeError as e:
        fail(f"bend output is not one quoted string: {e}")
    if not isinstance(s, str):
        fail("bend output is not a string")
    return s


def main(argv: list[str]) -> int:
    """Run the differential.

    Args:
        argv: The command line.

    Returns:
        The exit status.

    """
    if len(argv) != ARGC:
        fail("usage: gdn_replay_diff.py TREE_ROOT RECURRENT_UTIL_PY")
    tree = Path(argv[1])
    recurrent_util = Path(argv[2])
    verify_pins(tree, recurrent_util)
    num_rejected = load_num_rejected(tree)
    ns = load_host(tree, recurrent_util)
    ours = engine_table(ns, num_rejected)
    theirs = bend_table()
    if ours != theirs:
        sys.stdout.writelines(
            difflib.unified_diff(
                theirs.splitlines(keepends=True),
                ours.splitlines(keepends=True),
                "GDN_REPLAY_TABLE.bend",
                "engine",
            )
        )
        return 1
    sys.stdout.write(f"gdn_replay_diff: {len(ours.splitlines())} rows match\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
