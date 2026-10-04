#!/usr/bin/env python3
"""
Finite source link of bend/prefill_nosync.bend (ext 9502 EXL3_PREFILL_NOSYNC: deferred recurrent-checkpoint copies and
pinned prefill uploads) to the patched engine text. Text evidence and a differential run, not a proof.

1. Text: every engine line the model transcribes occurs verbatim, once, inside the function the model assumes, and in
   the order the model assumes inside that function (RecurrentCache.settle's loop with the `rnd >= before_round` keep
   test, put's self.settle(self.round) before the stash and the pending append, get_stashed / prune_stranded / persist
   settling before they read, Generator.iterate's begin_round before iterate_start_jobs and job.prefill before
   recurrent_checkpoint, recurrent_checkpoint's settle before its job loop, the job's stash_defer reset / set around the
   forward, GDNState.stash(defer)'s copies before the event record, the pinned uploads before the forward).
2. Differential: the engine's own RecurrentCache class and Generator.recurrent_checkpoint (exec'd from the engine text,
   torch stubbed: a deferred stash's event logs the wait when synchronized) are driven through the iterate sequence of
   the model (begin_round, get_stashed, prefill forward with its put, recurrent_checkpoint, decode, requeue put,
   prune_stranded) for every single iterate after a set-up iterate and for 300 pseudo-random runs of 1..6 iterates;
   the logged host/stream trace must equal Impl.trace of the Bend model (run with the pinned bend) on the same inputs.

`--mutate NAME` applies a deliberate source mutation that must be rejected.

Usage: python3 bend/prefill_nosync_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR
"""
from __future__ import annotations

import ast
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_link  # noqa: E402

# (file, function header, [lines in the order the model assumes]); every line must occur exactly once in the function
ORDERED = [
    ("cache/recurrent.py", "    def __init__(", [
        "self.pending = []",
        "self.round = 0",
    ]),
    ("cache/recurrent.py", "    def begin_round(self):", [
        "self.round += 1",
    ]),
    ("cache/recurrent.py", "    def settle(self, before_round: int | None = None):", [
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
    ]),
    ("cache/recurrent.py", "    def get_stashed(self, key, default = None):", [
        "self.settle()",
        "if key in self:",
        "return self[key]",
    ]),
    ("cache/recurrent.py", "    def put(self, key, state, defer: bool = False):", [
        "self.settle(self.round)",
        "if key in self:",
        "self.move_to_end(key)",
        "stashed_state = state.stash(defer = True) if defer else state.stash()",
        "if STASH_PENDING in stashed_state:",
        "self.pending.append((self.round, stashed_state))",
    ]),
    ("cache/recurrent.py", "    def prune_stranded(self) -> int:", [
        "self.settle()",
        "stranded = [k for k in self if not self.pagetable.is_resumable(k)]",
    ]),
    ("generator/generator.py", "    def iterate(self) -> list[dict]:", [
        "self.recurrent_cache.begin_round()",
        "self.iterate_start_jobs(results)",
        "for job in list(self.active_jobs):",
        "job.prefill(results)",
        "self.recurrent_checkpoint()",
        "self.iterate_gen(results)",
    ]),
    ("generator/generator.py", "    def recurrent_checkpoint(self):", [
        "self.recurrent_cache.settle(self.recurrent_cache.round)",
        "for job in self.active_jobs:",
        "job.maybe_stash_recurrent(self.recurrent_cache)",
    ]),
    ("generator/generator.py", "    def on_queue_drained(self):", [
        "self.recurrent_cache.prune_stranded()",
    ]),
    ("generator/generator.py", "    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):", [
        "if job in requeuing_jobs and self.recurrent_cache is not None:",
        "job.maybe_stash_recurrent(self.recurrent_cache, PAGE_SIZE)",
        "job.deallocate_pages()",
    ]),
    ("generator/generator.py", "    def iterate_start_jobs(self, results: list):", [
        "job.allocate_pages()",
    ]),
    ("generator/job.py", "    def allocate_pages(self):", [
        "seq.allocate_pages(self.pagetable, self.generator.recurrent_cache, protected_hashes)",
    ]),
    ("generator/pagetable.py", "    def allocate_pages(\n        self,\n        pagetable: PageTable,", [
        "stashed_recurrent_state = recurrent_cache.get_stashed(page_hashes[cached_pages - 1])",
    ]),
    ("generator/job.py", "    def prefill(self, results: list):", [
        "nosync = prefill_nosync_enabled()",
        "self.stash_defer = False",
        "recurrent_last_page = False\n            if self.generator.recurrent_cache is not None:",
        "recurrent_last_page = True",
        "if prefill_end > prefill_start:",
        "block_table = seq.block_index_tensor.pin_memory()",
        "cache_seqlens = torch.tensor([prefill_start], dtype = torch.int32).pin_memory()",
        "params[\"pinned_upload\"] = True",
        "self.generator.model.prefill(input_ids = prefill_ids, params = params)",
        "seq.kv_position = prefill_end",
        "self.stash_defer = (",
        "nosync and getattr(self.recurrent_state, \"can_defer_stash\", False) and",
        "not self.generator.model.loaded_tp",
        "if recurrent_last_page:",
        "self.maybe_stash_recurrent(self.generator.recurrent_cache, PAGE_SIZE)",
    ]),
    ("generator/job.py", "    def maybe_stash_recurrent(self, cache, interval = None):", [
        "cache.put(page.phash, self.recurrent_state, defer = self.stash_defer)",
    ]),
    ("generator/persist.py", "    def capture(self, verify: bool = False) -> Capture | None:", [
        "torch.cuda.synchronize(self.segments[0].device)",
        "self.rc.settle()",
        "_require(not self.rc.pending, \"deferred checkpoints left unsettled\")",
        "ordered, stashes, skipped = self.anchored()",
    ]),
    ("modules/gated_delta_net.py", "    def stash(self, defer: bool = False):", [
        "if defer:",
        "raise RuntimeError(\"GDNState: a deferred stash needs a local (non-TP) cache\")",
        "pinned = {k: l.stash(self.slot, pinned = True) for k, l in layers.items()}",
        "event = torch.cuda.Event()",
        "event.record(torch.cuda.current_stream(dev))",
        "stashed[STASH_PENDING] = (events, pinned)",
        "elif not self.cache.model.loaded_tp:",
        "stashed[k] = l.stash(self.slot)",
    ]),
    ("modules/gated_delta_net.py", "    def stash(self, slot, position: int = 0, pinned: bool = False):", [
        "self.commit_pending((slot,))",
        "rec_h = torch.empty(rec.shape, dtype = rec.dtype, pin_memory = True)",
        "conv_h = torch.empty(conv.shape, dtype = conv.dtype, pin_memory = True)",
        "rec_h.copy_(rec, non_blocking = True)",
        "conv_h.copy_(conv, non_blocking = True)",
        "return rec_h, conv_h",
        "return rec.cpu(), conv.cpu()",
    ]),
    ("modules/embedding.py", "    def forward(", [
        "elif params.get(\"pinned_upload\") and x.device.type == \"cpu\":",
        "x = x.pin_memory()",
    ]),
]
# module-level / class-level lines (exactly once in the file)
ONCE = [
    ("cache/recurrent.py", "STASH_PENDING = \"ext9502_pending\""),
    ("modules/gated_delta_net.py", "    can_defer_stash = True"),
    ("generator/job.py", "def prefill_nosync_enabled() -> bool:"),
]

MUTATIONS = {
    # put stashes without settling earlier rounds first
    "put_no_settle": ("cache/recurrent.py", "        self.settle(self.round)\n        if key in self:", "        if key in self:"),
    # settle keeps only later rounds: the current round's deferred copies are waited at once
    "keep_gt": ("cache/recurrent.py", "rnd >= before_round", "rnd > before_round"),
    # recurrent_checkpoint settles after its stash loop
    "ckpt_settle_after": ("generator/generator.py",
                          "        self.recurrent_cache.settle(self.recurrent_cache.round)\n"
                          "        for job in self.active_jobs:\n"
                          "            job.maybe_stash_recurrent(self.recurrent_cache)\n",
                          "        for job in self.active_jobs:\n"
                          "            job.maybe_stash_recurrent(self.recurrent_cache)\n"
                          "        self.recurrent_cache.settle(self.recurrent_cache.round)\n"),
    # get_stashed reads without settling
    "get_no_settle": ("cache/recurrent.py", "        self.settle()\n        if key in self:", "        if key in self:"),
    # the event is recorded before the copies are enqueued
    "event_first": ("modules/gated_delta_net.py",
                    "            pinned = {k: l.stash(self.slot, pinned = True) for k, l in layers.items()}\n"
                    "            events = []\n"
                    "            for dev in {l.recurrent_state.device for l in layers.values()}:\n"
                    "                event = torch.cuda.Event()\n"
                    "                event.record(torch.cuda.current_stream(dev))\n"
                    "                events.append(event)\n",
                    "            events = []\n"
                    "            for dev in {l.recurrent_state.device for l in layers.values()}:\n"
                    "                event = torch.cuda.Event()\n"
                    "                event.record(torch.cuda.current_stream(dev))\n"
                    "                events.append(event)\n"
                    "            pinned = {k: l.stash(self.slot, pinned = True) for k, l in layers.items()}\n"),
    # begin_round after iterate_start_jobs
    "begin_late": ("generator/generator.py",
                   "        if self.recurrent_cache is not None:\n            self.recurrent_cache.begin_round()\n"
                   "        self.iterate_start_jobs(results)\n",
                   "        self.iterate_start_jobs(results)\n"
                   "        if self.recurrent_cache is not None:\n            self.recurrent_cache.begin_round()\n"),
}
FILES = sorted({f for f, _, _ in ORDERED} | {f for f, _ in ONCE})


def fail(msg: str) -> None:
    raise SystemExit(f"prefill_nosync_diff: FAIL: {msg}")


def function(text: str, header: str, fname: str) -> str:
    """The source of the function starting at `header` (up to the next def / decorator at its indentation)."""
    if text.count(header) != 1:
        fail(f"{fname}: header {header.splitlines()[0]!r} occurs {text.count(header)} times (want 1)")
    start = text.index(header)
    indent = header[:len(header) - len(header.lstrip())]
    ends = [m.start() for m in re.finditer(rf"\n{indent}(def |@|class )", text[start + 1:])]
    return text[start:start + 1 + ends[0]] if ends else text[start:]


def text_checks(src: dict[str, str]) -> int:
    n = 0
    for fname, header, lines in ORDERED:
        body = function(src[fname], header, fname)
        pos = []
        for line in lines:
            c = body.count(line)
            if c != 1:
                fail(f"{fname} {header.strip().splitlines()[0]}: {line!r} occurs {c} times (want 1)")
            pos.append(body.index(line))
            n += 1
        if pos != sorted(pos):
            bad = next(i for i in range(1, len(pos)) if pos[i] < pos[i - 1])
            fail(f"{fname} {header.strip().splitlines()[0]}: {lines[bad - 1]!r} must precede {lines[bad]!r}")
    for fname, line in ONCE:
        if src[fname].count(line) != 1:
            fail(f"{fname}: {line!r} occurs {src[fname].count(line)} times (want 1)")
        n += 1
    return n


# ---------------------------------------------------------------------------------------------
# Differential run: the engine's RecurrentCache + recurrent_checkpoint against the Bend model

PU = ("No", "Hit", "New")


class _Torch:
    """torch stub for RecurrentCache.settle's installs (pinned dicts are empty here, so never called)."""

    @staticmethod
    def empty(*a, **k):
        raise AssertionError("unexpected torch.empty")


def engine_cache(src: dict[str, str]):
    tree = ast.parse(src["cache/recurrent.py"])
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RecurrentCache")
    ns = {"OrderedDict": OrderedDict, "torch": _Torch, "STASH_PENDING": "ext9502_pending", "PAGE_SIZE": 256,
          "note_freed": lambda n: None, "mp_cache_recurrent_del": None, "malloc_trim": lambda: None}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "recurrent.py", "exec"), ns)
    gen = ast.parse(src["generator/generator.py"])
    gcls = next(n for n in gen.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
    rc_fn = next(n for n in gcls.body if isinstance(n, ast.FunctionDef) and n.name == "recurrent_checkpoint")
    gns: dict = {}
    exec(compile(ast.Module(body=[rc_fn], type_ignores=[]), "generator.py", "exec"), gns)
    return ns["RecurrentCache"], gns["recurrent_checkpoint"]


def py_trace(rc_cls, ckpt_fn, its: list[tuple]) -> list[tuple]:
    tr: list[tuple] = []
    model = type("M", (), {"loaded_tp": False})()
    rc = rc_cls(model, 1 << 60)
    rc["seed"] = {"position": 0, "checkpoint_size": 0}
    st = {"ver": 0, "sid": 0, "key": 0}

    class Event:
        def __init__(self, s):
            self.s = s

        def synchronize(self):
            tr.append(("Wt", self.s))

    class State:
        def stash(self, defer: bool = False):
            s = st["sid"]
            st["sid"] += 1
            tr.append(("Cp", s, st["ver"], defer))
            d = {"position": 0, "checkpoint_size": 0}
            if defer:
                d["ext9502_pending"] = ([Event(s)], {})
            else:
                tr.append(("Wt", s))  # .cpu() synchronizes
            return d

    state = State()

    class Job:
        stash_defer = False
        u = "No"

        def maybe_stash_recurrent(self, cache, interval=None):
            put(cache, self.u)

    job = Job()

    def put(cache, u):
        if u == "No":
            return
        if u == "Hit":
            key = "seed"
        else:
            key = f"k{st['key']}"
            st["key"] += 1
        cache.put(key, state, defer=job.stash_defer)

    class Jobs:
        def __iter__(self):
            tr.append(("Ck",))
            return iter([job])

    gen = type("G", (), {})()
    gen.recurrent_cache = rc
    gen.active_jobs = Jobs()

    for (rs, fw, lp, cp, en, gf, rq, re_) in its:
        tr.append(("Bg",))
        rc.begin_round()
        if rs:
            tr.append(("Rd0",))
            rc.get_stashed("seed")
            tr.append(("Rd",))
        job.stash_defer = False
        if fw:
            st["ver"] += 1
            tr.append(("Up", st["ver"]))
            tr.append(("Fw", st["ver"]))
            job.stash_defer = en
            put(rc, lp)
        tr.append(("Ck0",))
        job.u = cp
        ckpt_fn(gen)
        if gf:
            st["ver"] += 1
            tr.append(("Dc", st["ver"]))
        put(rc, rq)
        if re_:
            tr.append(("Rd0",))
            rc.prune_stranded()
            tr.append(("Rd",))
    return tr


def b(x: bool) -> str:
    return "True{}" if x else "False{}"


def bend_its(its: list[tuple]) -> str:
    cells = [f"Spec.Iter{{{b(rs)}, {b(fw)}, Spec.{lp}{{}}, Spec.{cp}{{}}, {b(en)}, {b(gf)}, Spec.{rq}{{}}, {b(re_)}}}"
             for (rs, fw, lp, cp, en, gf, rq, re_) in its]
    return "[" + ", ".join(cells) + "]"


EV = re.compile(r"(Bg|Rd0|Rd|Up|Fw|Dc|Ck0|Ck|Cp|Wt)\{([^{}]*(?:\{\}[^{}]*)*)\}")


def parse_traces(out: str) -> list[list[tuple]]:
    out = out.replace("prefill_nosync_spec.", "")
    if not out.startswith("[[") and out.strip() != "[]":
        fail(f"unexpected bend output {out[:200]!r}")
    traces = []
    for chunk in re.findall(r"\[([^\[\]]*)\]", out.strip()[1:-1]):
        tr = []
        for name, args in EV.findall(chunk):
            vals = []
            for a in [x.strip() for x in args.split(",") if x.strip()]:
                vals.append(True if a == "True{}" else False if a == "False{}" else int(a.rstrip("n")))
            tr.append((name, *vals))
        traces.append(tr)
    return traces


def bend_traces(cases: list[list[tuple]]) -> list[list[tuple]]:
    with tempfile.TemporaryDirectory() as td:
        for f in ("prefill_nosync.bend", "prefill_nosync_spec.bend"):
            shutil.copy(HERE / f, Path(td) / f)
        body = ",\n   ".join(f"Impl.trace({bend_its(c)}, Impl.init())" for c in cases)
        (Path(td) / "link.bend").write_text(textwrap.dedent("""\
            import Base
            import ./prefill_nosync.bend as Impl
            import ./prefill_nosync_spec.bend as Spec

            def main() -> List<&2, List<&2, Spec.Ev>>:
              [""") + body + "]\n")
        r = subprocess.run(source_link.locked([source_link.bend(), "link.bend"]), cwd=td, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"bend link.bend exited {r.returncode}: {(r.stdout + r.stderr)[-1500:]}")
    return parse_traces(r.stdout)


def cases() -> list[list[tuple]]:
    setup = (False, True, "New", "New", True, True, "New", False)
    singles = [(rs, fw, lp, cp, en, gf, rq, re_)
               for rs in (False, True) for fw in (False, True) for lp in PU for cp in PU for en in (False, True)
               for gf in (False, True) for rq in PU for re_ in (False, True)]
    out = [[setup, it] for it in singles]
    rng = random.Random(9502)
    for _ in range(300):
        out.append([(rng.random() < 0.2, rng.random() < 0.7, rng.choice(PU), rng.choice(PU), rng.random() < 0.8,
                     rng.random() < 0.5, rng.choice(PU), rng.random() < 0.15) for _ in range(rng.randint(1, 6))])
    return out


def main(argv: list[str]) -> None:
    args = argv[1:]
    mutate = None
    if len(args) == 3 and args[0] == "--mutate":
        mutate, args = args[1], args[2:]
    if len(args) != 1:
        fail("usage: prefill_nosync_diff.py [--mutate NAME] ENGINE_PACKAGE_DIR")
    root = Path(args[0])
    src = {f: (root / f).read_text() for f in FILES}
    if mutate is not None:
        if mutate not in MUTATIONS:
            fail(f"unknown mutation {mutate!r} (known: {', '.join(MUTATIONS)})")
        f, old, new = MUTATIONS[mutate]
        if src[f].count(old) != 1:
            fail(f"mutation anchor {old!r} occurs {src[f].count(old)} times in {f}")
        src[f] = src[f].replace(old, new)
        print(f"prefill_nosync_diff: applied mutation {mutate}")
    n = text_checks(src)
    print(f"prefill_nosync_diff: {n} transcribed lines found once each, in model order")
    rc_cls, ckpt_fn = engine_cache(src)
    cs = cases()
    want = bend_traces(cs)
    if len(want) != len(cs):
        fail(f"bend returned {len(want)} traces for {len(cs)} cases")
    events = 0
    for c, w in zip(cs, want):
        got = py_trace(rc_cls, ckpt_fn, c)
        if got != w:
            k = next((i for i, (x, y) in enumerate(zip(got, w)) if x != y), min(len(got), len(w)))
            fail(f"engine trace differs from Impl.trace at event {k} for {c}:\n  engine {got[max(0, k - 3):k + 4]}\n"
                 f"  model  {w[max(0, k - 3):k + 4]}")
        events += len(w)
    print(f"prefill_nosync_diff: engine RecurrentCache + recurrent_checkpoint == Impl.trace on {len(cs)} runs "
          f"({events} events)")
    print("prefill_nosync_diff: OK")


if __name__ == "__main__":
    main(sys.argv)
