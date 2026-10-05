#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Finite source link of bend/prefill_nosync.bend to the patched engine text.

The model is ext 9502 EXL3_PREFILL_NOSYNC: deferred recurrent-checkpoint copies and
pinned prefill uploads. Text evidence and a differential run, not a proof.

1. Text: every engine line the model transcribes occurs verbatim, once, inside the
   function the model assumes, and in the order the model assumes inside that
   function (RecurrentCache.settle's loop with the `rnd >= before_round` keep test,
   put's self.settle(self.round) before the stash and the pending append,
   get_stashed / prune_stranded / persist settling before they read,
   Generator.iterate's begin_round before iterate_start_jobs and job.prefill before
   recurrent_checkpoint, recurrent_checkpoint's settle before its job loop, the job's
   stash_defer reset / set around the forward, GDNState.stash(defer)'s copies before
   the event record, the pinned uploads before the forward).
2. Differential: the engine's own RecurrentCache class and
   Generator.recurrent_checkpoint (evaluated from the engine text by bend/pysubset.py,
   torch stubbed: a deferred stash's event logs the wait when synchronized) are
   driven through the iterate sequence of the model (begin_round, get_stashed,
   prefill forward with its put, recurrent_checkpoint, decode, requeue put,
   prune_stranded) for every single iterate after a set-up iterate and for 300
   pseudo-random runs of 1..6 iterates; the logged host/stream trace must equal
   Impl.trace of the Bend model (run with the pinned bend) on the same inputs.

`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/prefill_nosync_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
import tempfile
import textwrap
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, Protocol, cast

sys.path.insert(0, str(Path(__file__).resolve().parent / "gen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detrand
import pysubset
import source_link

HERE = Path(__file__).resolve().parent

# (file, function header, [lines in the order the model assumes]); every line must
# occur exactly once in the function
ORDERED = [
    (
        "cache/recurrent.py",
        "    def __init__(",
        [
            "self.pending = []",
            "self.round = 0",
        ],
    ),
    (
        "cache/recurrent.py",
        "    def begin_round(self):",
        [
            "self.round += 1",
        ],
    ),
    (
        "cache/recurrent.py",
        "    def settle(self, before_round: int | None = None):",
        [
            "keep = []",
            "for rnd, stashed in self.pending:",
            "if before_round is not None and rnd >= before_round:",
            "keep.append((rnd, stashed))",
            "continue",
            "events, pinned = stashed.pop(STASH_PENDING)",
            "for event in events:",
            "event.synchronize()",
            "stashed[k] = (rec, conv)",
            "self.pending = keep",
        ],
    ),
    (
        "cache/recurrent.py",
        "    def get_stashed(self, key, default = None):",
        [
            "self.settle()",
            "if key in self:",
            "return self[key]",
        ],
    ),
    (
        "cache/recurrent.py",
        "    def put(self, key, state, defer: bool = False):",
        [
            "self.settle(self.round)",
            "if key in self:",
            "self.move_to_end(key)",
            "stashed_state = state.stash(defer = True) if defer else state.stash()",
            "if STASH_PENDING in stashed_state:",
            "self.pending.append((self.round, stashed_state))",
        ],
    ),
    (
        "cache/recurrent.py",
        "    def prune_stranded(self) -> int:",
        [
            "self.settle()",
            "stranded = [k for k in self if not self.pagetable.is_resumable(k)]",
        ],
    ),
    (
        "generator/generator.py",
        "    def iterate(self) -> list[dict]:",
        [
            "self.recurrent_cache.begin_round()",
            "self.iterate_start_jobs(results)",
            "for job in list(self.active_jobs):",
            "job.prefill(results)",
            "self.recurrent_checkpoint()",
            "self.iterate_gen(results)",
        ],
    ),
    (
        "generator/generator.py",
        "    def recurrent_checkpoint(self):",
        [
            "self.recurrent_cache.settle(self.recurrent_cache.round)",
            "for job in self.active_jobs:",
            "job.maybe_stash_recurrent(self.recurrent_cache)",
        ],
    ),
    (
        "generator/generator.py",
        "    def on_queue_drained(self):",
        [
            "self.recurrent_cache.prune_stranded()",
        ],
    ),
    (
        "generator/generator.py",
        (
            "    def iterate_gen(self, results: list, draft_tokens: torch.Tensor |"
            " None = None):"
        ),
        [
            "if job in requeuing_jobs and self.recurrent_cache is not None:",
            "job.maybe_stash_recurrent(self.recurrent_cache, PAGE_SIZE)",
            "job.deallocate_pages()",
        ],
    ),
    (
        "generator/generator.py",
        "    def iterate_start_jobs(self, results: list):",
        [
            "job.allocate_pages()",
        ],
    ),
    (
        "generator/job.py",
        "    def allocate_pages(self):",
        [
            (
                "seq.allocate_pages(self.pagetable, self.generator.recurrent_cache,"
                " protected_hashes)"
            ),
        ],
    ),
    (
        "generator/pagetable.py",
        "    def allocate_pages(\n        self,\n        pagetable: PageTable,",
        [
            (
                "stashed_recurrent_state ="
                " recurrent_cache.get_stashed(page_hashes[cached_pages - 1])"
            ),
        ],
    ),
    (
        "generator/job.py",
        "    def prefill(self, results: list):",
        [
            "nosync = prefill_nosync_enabled()",
            "self.stash_defer = False",
            (
                "recurrent_last_page = False\n"
                "            if self.generator.recurrent_cache is not None:"
            ),
            "recurrent_last_page = True",
            "if prefill_end > prefill_start:",
            "block_table = seq.block_index_tensor.pin_memory()",
            (
                "cache_seqlens = torch.tensor([prefill_start], dtype ="
                " torch.int32).pin_memory()"
            ),
            'params["pinned_upload"] = True',
            "self.generator.model.prefill(input_ids = prefill_ids, params = params)",
            "seq.kv_position = prefill_end",
            "self.stash_defer = (",
            'nosync and getattr(self.recurrent_state, "can_defer_stash", False) and',
            "not self.generator.model.loaded_tp",
            "if recurrent_last_page:",
            "self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)",
        ],
    ),
    (
        "generator/job.py",
        "    def maybe_stash_recurrent(self, cache, interval = None):",
        [
            "cache.put(page.phash, self.recurrent_state, defer = self.stash_defer)",
        ],
    ),
    (
        "generator/persist.py",
        "    def capture(self, verify: bool = False) -> Capture | None:",
        [
            "torch.cuda.synchronize(self.segments[0].device)",
            "self.rc.settle()",
            '_require(not self.rc.pending, "deferred checkpoints left unsettled")',
            "ordered, stashes, skipped = self.anchored()",
        ],
    ),
    (
        "modules/gated_delta_net.py",
        "    def stash(self, defer: bool = False):",
        [
            "if defer:",
            (
                'raise RuntimeError("GDNState: a deferred stash needs a local'
                ' (non-TP) cache")'
            ),
            (
                "pinned = {k: l.stash(self.slot, pinned = True) "
                "for k, l in layers.items()}"
            ),
            "event = torch.cuda.Event()",
            "event.record(torch.cuda.current_stream(dev))",
            "stashed[STASH_PENDING] = (events, pinned)",
            "elif not self.cache.model.loaded_tp:",
            "stashed[k] = l.stash(self.slot)",
        ],
    ),
    (
        "modules/gated_delta_net.py",
        "    def stash(self, slot, position: int = 0, pinned: bool = False):",
        [
            "self.commit_pending((slot,))",
            "rec_h = torch.empty(rec.shape, dtype = rec.dtype, pin_memory = True)",
            "conv_h = torch.empty(conv.shape, dtype = conv.dtype, pin_memory = True)",
            "rec_h.copy_(rec, non_blocking = True)",
            "conv_h.copy_(conv, non_blocking = True)",
            "return rec_h, conv_h",
            "return rec.cpu(), conv.cpu()",
        ],
    ),
    (
        "modules/embedding.py",
        "    def forward(",
        [
            'elif params.get("pinned_upload") and x.device.type == "cpu":',
            "x = x.pin_memory()",
        ],
    ),
]
# module-level / class-level lines (exactly once in the file)
ONCE = [
    ("cache/recurrent.py", 'STASH_PENDING = "ext9502_pending"'),
    ("modules/gated_delta_net.py", "    can_defer_stash = True"),
    ("generator/job.py", "def prefill_nosync_enabled() -> bool:"),
]

MUTATIONS = {
    # put stashes without settling earlier rounds first
    "put_no_settle": (
        "cache/recurrent.py",
        "        self.settle(self.round)\n        if key in self:",
        "        if key in self:",
    ),
    # settle keeps only later rounds: the current round's deferred copies are waited
    # at once
    "keep_gt": ("cache/recurrent.py", "rnd >= before_round", "rnd > before_round"),
    # recurrent_checkpoint settles after its stash loop
    "ckpt_settle_after": (
        "generator/generator.py",
        (
            "        self.recurrent_cache.settle(self.recurrent_cache.round)\n"
            "        for job in self.active_jobs:\n"
            "            job.maybe_stash_recurrent(self.recurrent_cache)\n"
        ),
        (
            "        for job in self.active_jobs:\n"
            "            job.maybe_stash_recurrent(self.recurrent_cache)\n"
            "        self.recurrent_cache.settle(self.recurrent_cache.round)\n"
        ),
    ),
    # get_stashed reads without settling
    "get_no_settle": (
        "cache/recurrent.py",
        "        self.settle()\n        if key in self:",
        "        if key in self:",
    ),
    # the event is recorded before the copies are enqueued
    "event_first": (
        "modules/gated_delta_net.py",
        (
            "            pinned = {k: l.stash(self.slot, pinned = True) for k, l in"
            " layers.items()}\n"
            "            events = []\n"
            "            for dev in {l.recurrent_state.device for l in"
            " layers.values()}:\n"
            "                event = torch.cuda.Event()\n"
            "                event.record(torch.cuda.current_stream(dev))\n"
            "                events.append(event)\n"
        ),
        (
            "            events = []\n"
            "            for dev in {l.recurrent_state.device for l in"
            " layers.values()}:\n"
            "                event = torch.cuda.Event()\n"
            "                event.record(torch.cuda.current_stream(dev))\n"
            "                events.append(event)\n"
            "            pinned = {k: l.stash(self.slot, pinned = True) for k, l in"
            " layers.items()}\n"
        ),
    ),
    # begin_round after iterate_start_jobs
    "begin_late": (
        "generator/generator.py",
        (
            "        if self.recurrent_cache is not None:\n"
            "            self.recurrent_cache.begin_round()\n"
            "        self.iterate_start_jobs(results)\n"
        ),
        (
            "        self.iterate_start_jobs(results)\n"
            "        if self.recurrent_cache is not None:\n"
            "            self.recurrent_cache.begin_round()\n"
        ),
    ),
}
FILES = sorted({f for f, _, _ in ORDERED} | {f for f, _ in ONCE})


MUTATE_ARGC = 3
RANDOM_RUNS = 300
MAX_ITERATES = 6
# probabilities of the random runs' flags, in percent
P_RS = 20
P_FW = 70
P_EN = 80
P_GF = 50
P_RE = 15
PERCENT = 100

type Ev = tuple[object, ...]
type It = tuple[bool, bool, str, str, bool, bool, str, bool]


def fail(msg: str) -> NoReturn:
    """Stop with the link's failure message.

    Args:
        msg: What failed.

    Raises:
        SystemExit: Always.

    """
    text = f"prefill_nosync_diff: FAIL: {msg}"
    raise SystemExit(text)


def function(text: str, header: str, fname: str) -> str:
    """Return the source of the function starting at `header`.

    The function ends at the next def / decorator / class at its indentation.

    Args:
        text: The file's text.
        header: The function's first line(s).
        fname: The file name, for messages.

    Returns:
        The function's source.

    """
    if text.count(header) != 1:
        fail(
            f"{fname}: header {header.splitlines()[0]!r} occurs "
            f"{text.count(header)} times (want 1)"
        )
    start = text.index(header)
    indent = header[: len(header) - len(header.lstrip())]
    ends = [
        m.start() for m in re.finditer(rf"\n{indent}(def |@|class )", text[start + 1 :])
    ]
    return text[start : start + 1 + ends[0]] if ends else text[start:]


def check_ordered(text: str, fname: str, header: str, lines: list[str]) -> int:
    """Check `lines` occur once each, in order, in the function at `header`.

    Args:
        text: The file's text.
        fname: The file name, for messages.
        header: The function's first line(s).
        lines: The transcribed lines, in model order.

    Returns:
        The number of lines checked.

    """
    body = function(text, header, fname)
    where = f"{fname} {header.strip().splitlines()[0]}"
    pos = []
    for line in lines:
        c = body.count(line)
        if c != 1:
            fail(f"{where}: {line!r} occurs {c} times (want 1)")
        pos.append(body.index(line))
    if pos != sorted(pos):
        bad = next(i for i in range(1, len(pos)) if pos[i] < pos[i - 1])
        fail(f"{where}: {lines[bad - 1]!r} must precede {lines[bad]!r}")
    return len(lines)


def text_checks(src: dict[str, str]) -> int:
    """Check every ORDERED and ONCE line.

    Args:
        src: The engine files' text, by name.

    Returns:
        The number of lines checked.

    """
    n = 0
    for fname, header, lines in ORDERED:
        n += check_ordered(src[fname], fname, header, lines)
    for fname, line in ONCE:
        if src[fname].count(line) != 1:
            fail(f"{fname}: {line!r} occurs {src[fname].count(line)} times (want 1)")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Differential run: the engine's RecurrentCache + recurrent_checkpoint against
# the Bend model

PU = ("No", "Hit", "New")


class _Torch:
    """Torch stub for RecurrentCache.settle's installs.

    The pinned dicts are empty here, so it is never called.
    """

    @staticmethod
    def empty(*_a: object, **_k: object) -> NoReturn:
        """Reject any call.

        Raises:
            AssertionError: Always.

        """
        msg = "unexpected torch.empty"
        raise AssertionError(msg)


class Cache(Protocol):
    """The engine RecurrentCache operations the differential drives."""

    def __setitem__(self, key: str, value: dict[str, int], /) -> None:
        """Store `value` under `key`."""

    def begin_round(self) -> None:
        """Start an iterate round."""

    def get_stashed(self, key: str) -> object:
        """Return the settled stash of `key`."""

    def prune_stranded(self) -> object:
        """Drop stashes no page table can resume."""

    def put(self, key: str, state: State, *, defer: bool) -> None:
        """Stash `state` under `key`."""


type Checkpoint = Callable[..., object]


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    """Return the top-level class `name` of `tree`.

    Args:
        tree: The parsed file.
        name: The class name.

    Returns:
        The class definition.

    """
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def drop_super_init(cls: ast.ClassDef) -> None:
    """Remove the `super().__init__()` that starts the class's `__init__`.

    pysubset rejects the dunder attribute. Dropping the call is behaviour-neutral:
    the base is OrderedDict, whose `__new__` already initializes a fresh instance,
    so a no-argument `OrderedDict.__init__` on it changes nothing.

    Args:
        cls: The class definition, edited in place.

    """
    init = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ),
        None,
    )
    calls = [
        n
        for n in ast.walk(cls)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Call)
        and isinstance(n.func.value.func, ast.Name)
        and n.func.value.func.id == "super"
    ]
    if init is None or not init.body:
        fail("RecurrentCache: no __init__")
    first = init.body[0]
    if (
        len(calls) != 1
        or not isinstance(first, ast.Expr)
        or ast.unparse(first) != "super().__init__()"
        or first.value is not calls[0]
    ):
        fail("RecurrentCache: want super().__init__() once, first in __init__")
    init.body.pop(0)


def engine_cache(src: dict[str, str]) -> tuple[type, Checkpoint]:
    """Build the engine's RecurrentCache class and recurrent_checkpoint function.

    Args:
        src: The engine files' text, by name.

    Returns:
        The class and the (unbound) function.

    """
    cls = class_node(ast.parse(src["cache/recurrent.py"]), "RecurrentCache")
    ns: dict[str, object] = {
        "OrderedDict": OrderedDict,
        "torch": _Torch,
        "STASH_PENDING": "ext9502_pending",
        "PAGE_SIZE": 256,
        "note_freed": lambda _n: None,
        "mp_cache_recurrent_del": None,
        "malloc_trim": lambda: None,
    }
    drop_super_init(cls)
    pysubset.exec_block(ast.Module(body=[cls], type_ignores=[]), ns)
    gcls = class_node(ast.parse(src["generator/generator.py"]), "Generator")
    rc_fn = next(
        n
        for n in gcls.body
        if isinstance(n, ast.FunctionDef) and n.name == "recurrent_checkpoint"
    )
    gns: dict[str, object] = {}
    pysubset.exec_block(ast.Module(body=[rc_fn], type_ignores=[]), gns)
    rc_cls, ckpt = ns["RecurrentCache"], gns["recurrent_checkpoint"]
    if not isinstance(rc_cls, type) or not callable(ckpt):
        fail("engine RecurrentCache / recurrent_checkpoint did not build")
    return rc_cls, ckpt


@dataclass
class Log:
    """The trace and the counters of one differential run."""

    tr: list[Ev] = field(default_factory=list)
    ver: int = 0
    sid: int = 0
    key: int = 0


@dataclass
class Event:
    """A deferred stash's event: synchronizing logs the wait."""

    log: Log
    s: int

    def synchronize(self) -> None:
        """Log the wait on the stash's copies."""
        self.log.tr.append(("Wt", self.s))


@dataclass
class State:
    """The recurrent state: a stash logs its copy (and wait unless deferred)."""

    log: Log

    def stash(self, *, defer: bool = False) -> dict[str, object]:
        """Stash the state.

        Args:
            defer: Defer the copy's wait to an event.

        Returns:
            The stash.

        """
        s = self.log.sid
        self.log.sid += 1
        self.log.tr.append(("Cp", s, self.log.ver, defer))
        d: dict[str, object] = {"position": 0, "checkpoint_size": 0}
        if defer:
            d["ext9502_pending"] = ([Event(self.log, s)], {})
        else:
            self.log.tr.append(("Wt", s))  # .cpu() synchronizes
        return d


@dataclass
class Job:
    """The one job: stashes its state as `u` says, deferred per `stash_defer`."""

    log: Log
    state: State
    stash_defer: bool = False
    u: str = "No"

    def put(self, cache: Cache, u: str) -> None:
        """Put the state in `cache` under the key `u` names (none for "No").

        Args:
            cache: The recurrent cache.
            u: "No", "Hit" (the seed key) or "New" (a fresh key).

        """
        if u == "No":
            return
        if u == "Hit":
            key = "seed"
        else:
            key = f"k{self.log.key}"
            self.log.key += 1
        cache.put(key, self.state, defer=self.stash_defer)

    def maybe_stash_recurrent(self, cache: Cache, _interval: object = None) -> None:
        """Stash for the checkpoint, as `u` says.

        Args:
            cache: The recurrent cache.
            _interval: Unused.

        """
        self.put(cache, self.u)


@dataclass
class Jobs:
    """The active jobs: iterating logs the checkpoint loop."""

    log: Log
    job: Job

    def __iter__(self) -> Iterator[Job]:
        """Log the checkpoint loop.

        Returns:
            An iterator over the one job.

        """
        self.log.tr.append(("Ck",))
        return iter([self.job])


@dataclass
class Gen:
    """The generator fields recurrent_checkpoint reads."""

    recurrent_cache: Cache
    active_jobs: Jobs


def run_iterate(gen: Gen, job: Job, ckpt: Checkpoint, it: It) -> None:
    """Drive one iterate of the model's sequence through the engine.

    Args:
        gen: The generator stub.
        job: The job.
        ckpt: The engine's recurrent_checkpoint.
        it: The iterate's choices.

    """
    rs, fw, lp, cp, en, gf, rq, re_ = it
    rc, log = gen.recurrent_cache, job.log
    log.tr.append(("Bg",))
    rc.begin_round()
    if rs:
        log.tr.append(("Rd0",))
        rc.get_stashed("seed")
        log.tr.append(("Rd",))
    job.stash_defer = False
    if fw:
        log.ver += 1
        log.tr.extend((("Up", log.ver), ("Fw", log.ver)))
        job.stash_defer = en
        job.put(rc, lp)
    log.tr.append(("Ck0",))
    job.u = cp
    ckpt(gen)
    if gf:
        log.ver += 1
        log.tr.append(("Dc", log.ver))
    job.put(rc, rq)
    if re_:
        log.tr.append(("Rd0",))
        rc.prune_stranded()
        log.tr.append(("Rd",))


def py_trace(rc_cls: type, ckpt: Checkpoint, its: list[It]) -> list[Ev]:
    """Return the engine's host/stream trace of the iterates `its`.

    Args:
        rc_cls: The engine's RecurrentCache.
        ckpt: The engine's recurrent_checkpoint.
        its: The iterates.

    Returns:
        The trace.

    """
    log = Log()
    rc = cast("Cache", rc_cls(SimpleNamespace(loaded_tp=False), 1 << 60))
    rc["seed"] = {"position": 0, "checkpoint_size": 0}
    job = Job(log, State(log))
    gen = Gen(rc, Jobs(log, job))
    for it in its:
        run_iterate(gen, job, ckpt, it)
    return log.tr


BEND_BOOL = {True: "True{}", False: "False{}"}


def bend_its(its: list[It]) -> str:
    """Return the Bend list literal of the iterates `its`.

    Args:
        its: The iterates.

    Returns:
        The literal.

    """
    cells = [
        f"Spec.Iter{{{BEND_BOOL[rs]}, {BEND_BOOL[fw]}, Spec.{lp}{{}}, Spec.{cp}{{}}, "
        f"{BEND_BOOL[en]}, {BEND_BOOL[gf]}, Spec.{rq}{{}}, {BEND_BOOL[re_]}}}"
        for (rs, fw, lp, cp, en, gf, rq, re_) in its
    ]
    return "[" + ", ".join(cells) + "]"


EV = re.compile(r"(Bg|Rd0|Rd|Up|Fw|Dc|Ck0|Ck|Cp|Wt)\{([^{}]*(?:\{\}[^{}]*)*)\}")


def ev_value(a: str) -> bool | int:
    """Return the value of one printed event field.

    Args:
        a: The field text.

    Returns:
        The bool or int.

    """
    if a == "True{}":
        return True
    if a == "False{}":
        return False
    return int(a.rstrip("n"))


def parse_traces(out: str) -> list[list[Ev]]:
    """Parse the Bend program's printed traces.

    Args:
        out: The program's stdout.

    Returns:
        The traces.

    """
    out = out.replace("prefill_nosync_spec.", "")
    if not out.startswith("[[") and out.strip() != "[]":
        fail(f"unexpected bend output {out[:200]!r}")
    traces = []
    for chunk in re.findall(r"\[([^\[\]]*)\]", out.strip()[1:-1]):
        tr: list[Ev] = []
        for name, args in EV.findall(chunk):
            vals = [
                ev_value(a) for a in [x.strip() for x in args.split(",") if x.strip()]
            ]
            tr.append((name, *vals))
        traces.append(tr)
    return traces


def bend_traces(cases: list[list[It]]) -> list[list[Ev]]:
    """Run Impl.trace of the Bend model on every case.

    Args:
        cases: The runs.

    Returns:
        The model's traces.

    """
    with tempfile.TemporaryDirectory() as td:
        for f in ("prefill_nosync.bend", "prefill_nosync_spec.bend"):
            shutil.copy(HERE / f, Path(td) / f)
        body = ",\n   ".join(f"Impl.trace({bend_its(c)}, Impl.init())" for c in cases)
        (Path(td) / "link.bend").write_text(
            textwrap.dedent("""\
            import Base
            import ./prefill_nosync.bend as Impl
            import ./prefill_nosync_spec.bend as Spec

            def main() -> List<&2, List<&2, Spec.Ev>>:
              [""")
            + body
            + "]\n"
        )
        r = source_link.run(
            [source_link.bend(), "link.bend"],
            cwd=td,
            capture_output=True,
            text=True,
            check=False,
        )
    if r.returncode != 0:
        fail(f"bend link.bend exited {r.returncode}: {(r.stdout + r.stderr)[-1500:]}")
    return parse_traces(r.stdout)


def random_run(rng: detrand.SplitMix64) -> list[It]:
    """Return one pseudo-random run of 1..MAX_ITERATES iterates.

    Args:
        rng: The generator.

    Returns:
        The run.

    """
    return [
        (
            rng.below(PERCENT) < P_RS,
            rng.below(PERCENT) < P_FW,
            rng.choice(PU),
            rng.choice(PU),
            rng.below(PERCENT) < P_EN,
            rng.below(PERCENT) < P_GF,
            rng.choice(PU),
            rng.below(PERCENT) < P_RE,
        )
        for _ in range(rng.randint(1, MAX_ITERATES))
    ]


def cases() -> list[list[It]]:
    """Return every single iterate after a set-up iterate, then the random runs.

    Returns:
        The runs.

    """
    setup: It = (False, True, "New", "New", True, True, "New", False)
    singles: list[It] = [
        (rs, fw, lp, cp, en, gf, rq, re_)
        for rs in (False, True)
        for fw in (False, True)
        for lp in PU
        for cp in PU
        for en in (False, True)
        for gf in (False, True)
        for rq in PU
        for re_ in (False, True)
    ]
    out = [[setup, it] for it in singles]
    rng = detrand.SplitMix64(9502)
    out.extend(random_run(rng) for _ in range(RANDOM_RUNS))
    return out


def parse_args(argv: list[str]) -> tuple[str | None, Path]:
    """Parse the command line.

    Args:
        argv: The command line.

    Returns:
        The mutation name (or None) and the engine package directory.

    """
    args = argv[1:]
    mutate = None
    if len(args) == MUTATE_ARGC and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: prefill_nosync_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    return mutate, Path(args[0])


def apply_mutation(src: dict[str, str], mutate: str) -> None:
    """Apply the mutation `mutate` to `src` in place.

    Args:
        src: The engine files' text, by name.
        mutate: The mutation name.

    """
    if mutate not in MUTATIONS:
        fail(f"unknown mutation {mutate!r} (known: {', '.join(MUTATIONS)})")
    f, old, new = MUTATIONS[mutate]
    if src[f].count(old) != 1:
        fail(f"mutation anchor {old!r} occurs {src[f].count(old)} times in {f}")
    src[f] = src[f].replace(old, new)
    sys.stdout.write(f"prefill_nosync_diff: applied mutation {mutate}\n")


def differential(src: dict[str, str]) -> None:
    """Compare the engine's traces with Impl.trace on every case.

    Args:
        src: The engine files' text, by name.

    """
    rc_cls, ckpt = engine_cache(src)
    cs = cases()
    want = bend_traces(cs)
    if len(want) != len(cs):
        fail(f"bend returned {len(want)} traces for {len(cs)} cases")
    events = 0
    for c, w in zip(cs, want, strict=True):
        got = py_trace(rc_cls, ckpt, c)
        if got != w:
            k = next(
                (i for i, (x, y) in enumerate(zip(got, w, strict=False)) if x != y),
                min(len(got), len(w)),
            )
            fail(
                f"engine trace differs from Impl.trace at event {k} for {c}:\n"
                f"  engine {got[max(0, k - 3) : k + 4]}\n"
                f"  model  {w[max(0, k - 3) : k + 4]}"
            )
        events += len(w)
    sys.stdout.write(
        "prefill_nosync_diff: engine RecurrentCache + recurrent_checkpoint == "
        f"Impl.trace on {len(cs)} runs ({events} events)\n"
    )


def main(argv: list[str]) -> None:
    """Run the source link.

    Args:
        argv: The command line.

    """
    mutate, root = parse_args(argv)
    src = {f: (root / f).read_text() for f in FILES}
    if mutate is not None:
        apply_mutation(src, mutate)
    n = text_checks(src)
    sys.stdout.write(
        f"prefill_nosync_diff: {n} transcribed lines found once each, in model order\n"
    )
    differential(src)
    sys.stdout.write("prefill_nosync_diff: OK\n")


if __name__ == "__main__":
    main(sys.argv)
