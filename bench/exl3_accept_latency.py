# Copyright (c) 2026 Gil Rodrigues
r"""Per-call latency of the served EXL3 greedy acceptance vs a Python reference.

Run inside the served image (its python 3.13 + torch; CPU only), one core:
  docker run --rm --network none -v ART:/art/a:ro -v OUT:/out \
    --entrypoint /usr/bin/taskset qwen-inference:exl3 -c 6 \
    /opt/venv/bin/python -I -B /out/exl3_accept_latency.py /out/r.json /art/a [...]
Each artifact directory is fully admitted through its own loader, and both
implementations must reproduce the pinned table and agree on every timed
input. Batches of 20000 calls alternate order every round (3 warmup + 40).
Cases: *_server = the engine's GreedyAccept.__call__ (tensor rows, tolist,
sorted stop tuple, loader, ctypes); *_loader = Acceptor.accept on lists;
*_ctypes_floor = the exported function on a prefilled buffer; py_server /
py_list = the serial generator decision in Python with the same inputs.
"""

from __future__ import annotations

import gc
import importlib.util
import itertools
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from exllamav3.generator.greedy_accept import GreedyAccept

if TYPE_CHECKING:
    from types import ModuleType

STOP_HIT_RATE = 0.02
SHORT_BUDGET_RATE = 0.05
CHECKPOINT_RATE = 0.1

RawItem = tuple[list[int], list[int], set[int], int, int]
ListItem = tuple[list[int], list[int], tuple[int, ...], int, int]
TensorItem = tuple[torch.Tensor, torch.Tensor, set[int], int, int]
Call = Callable[..., object]
Cases = dict[str, tuple[Call, str]]
Items = dict[str, list[ListItem] | list[TensorItem]]


def load(directory: str) -> ModuleType:
    """Load the acceptance loader shipped in an artifact directory.

    Args:
        directory: The artifact directory.

    Returns:
        The executed loader module.

    Raises:
        RuntimeError: If no loader can be built for the directory's file.

    """
    spec = importlib.util.spec_from_file_location(
        f"loader_{abs(hash(directory))}", Path(directory) / "exl3_bend_accept.py"
    )
    if spec is None or spec.loader is None:
        msg = f"no loader for {directory}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ref_accept(
    verify_ids: Sequence[int],
    proposals: Sequence[int],
    stops: Collection[int],
    budget: int,
    checkpoint: int,
) -> tuple[int, bool]:
    """Decide serially: stop token, budget, final, mismatch/checkpoint.

    Args:
        verify_ids: The k + 1 target ids.
        proposals: The k proposed ids.
        stops: The stop ids.
        budget: The remaining token budget.
        checkpoint: The forced stop position, 0 for none.

    Returns:
        The (count, eos) verdict.

    """
    k = len(proposals)
    for i in range(k):
        t = verify_ids[i]
        n = i + 1
        if t in stops or n >= budget:
            return n, True
        if proposals[i] != t or n == checkpoint:
            return n, False
    return k + 1, (verify_ids[k] in stops or k + 1 >= budget)


def ref_server(
    verify_ids: torch.Tensor,
    proposals: torch.Tensor,
    stop_tokens: set[int],
    budget: int,
    checkpoint: int,
) -> tuple[int, bool]:
    """Decide in Python with the call shape of GreedyAccept.__call__.

    Args:
        verify_ids: The k + 1 target ids.
        proposals: The k proposed ids.
        stop_tokens: The stop ids.
        budget: The remaining token budget.
        checkpoint: The forced stop position, 0 for none.

    Returns:
        The (count, eos) verdict.

    """
    return ref_accept(
        verify_ids.tolist(), proposals.tolist(), stop_tokens, budget, checkpoint
    )


def served(acceptor: object) -> GreedyAccept:
    """Bind the engine's own GreedyAccept.__call__ to an admitted acceptor.

    Args:
        acceptor: The admitted acceptor.

    Returns:
        The engine object, constructed without re-admission.

    """
    g = GreedyAccept.__new__(GreedyAccept)
    g.acceptor = acceptor
    return g


def inputs_table_k7(rng: random.Random, n: int) -> list[RawItem]:
    """Draw k = 7 inputs of the admission table section A (binary alphabet).

    Args:
        rng: The input generator.
        n: The input count.

    Returns:
        The inputs.

    """
    out = []
    for _ in range(n):
        x = rng.randrange(1 << 15)
        v = [(17, 248046)[(x >> (2 * i)) & 1] for i in range(8)]
        p = [(17, 248046)[(x >> (2 * i + 1)) & 1] for i in range(7)]
        budget = rng.choice([*range(1, 10), 262144])
        cp = rng.randrange(8)
        out.append((v, p, {248046}, budget, cp))
    return out


def inputs_served(rng: random.Random, n: int) -> list[RawItem]:
    """Draw k = 7 rounds shaped like serving.

    Random vocab ids, accepted prefix geometric (mean ~3.5 of 7), stop set of
    two ids, large budget, rare checkpoint and rare stop hit.

    Args:
        rng: The input generator.
        n: The input count.

    Returns:
        The inputs.

    """
    stops = {248044, 248046}
    out = []
    for _ in range(n):
        v = [rng.randrange(248000) for _ in range(8)]
        acc = min(7, int(rng.expovariate(1 / 3.5)))
        p = [v[i] if i < acc else (v[i] + 1) % 248000 for i in range(7)]
        if rng.random() < STOP_HIT_RATE:
            v[rng.randrange(8)] = 248046
        budget = rng.randrange(1, 64) if rng.random() < SHORT_BUDGET_RATE else 4096
        cp = rng.randrange(1, 8) if rng.random() < CHECKPOINT_RATE else 0
        out.append((v, p, stops, budget, cp))
    return out


def as_tensors(items: list[RawItem]) -> list[TensorItem]:
    """Convert the id lists of each input to int64 tensor rows.

    Args:
        items: The inputs.

    Returns:
        The inputs with tensor ids.

    """
    v = torch.tensor([it[0] for it in items], dtype=torch.long)
    p = torch.tensor([it[1] for it in items], dtype=torch.long)
    return [(v[i], p[i], it[2], it[3], it[4]) for i, it in enumerate(items)]


def stop_tuple(items: list[RawItem]) -> list[ListItem]:
    """Replace each input's stop set by its sorted tuple.

    Args:
        items: The inputs.

    Returns:
        The inputs with stop tuples.

    """
    return [(v, p, tuple(sorted(s)), b, c) for v, p, s, b, c in items]


def run_batch(
    fn: Call, items: list[ListItem] | list[TensorItem], sink: list[object]
) -> int:
    """Time one batch of calls.

    Args:
        fn: The call under test.
        items: The inputs.
        sink: Receives every result so no call is dead.

    Returns:
        The elapsed nanoseconds.

    """
    t0 = time.perf_counter_ns()
    for a, b, c, d, e in items:
        sink.append(fn(a, b, c, d, e))
    return time.perf_counter_ns() - t0


def bench(
    cases: Cases, items_by_case: Items, rounds: int = 40, warm: int = 3
) -> dict[str, list[float]]:
    """Time every case in alternating order.

    Args:
        cases: Name to (call, input kind).
        items_by_case: Input kind to inputs.
        rounds: The measured rounds.
        warm: The warmup rounds.

    Returns:
        Name to per-round nanoseconds per call.

    """
    raw: dict[str, list[float]] = {name: [] for name in cases}
    names = list(cases)
    for r in range(warm + rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            fn, kind = cases[name]
            items = items_by_case[kind]
            sink: list[object] = []
            gc.disable()
            dt = run_batch(fn, items, sink)
            gc.enable()
            if r >= warm:
                raw[name].append(dt / len(items))
    return raw


def summarize(raw: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Summarize per-round timings.

    Args:
        raw: Name to per-round nanoseconds per call.

    Returns:
        Name to median, min, max and round count.

    """
    return {
        k: {
            "median_ns": statistics.median(v),
            "min_ns": min(v),
            "max_ns": max(v),
            "n": len(v),
        }
        for k, v in raw.items()
    }


def ctypes_floor(fn: Callable[[object], object], cells: object) -> Call:
    """Bind the exported function to a prefilled cell buffer.

    Args:
        fn: The exported glue function.
        cells: The prefilled cell buffer.

    Returns:
        A five-argument call that ignores its arguments.

    """

    def call(_a: object, _b: object, _c: object, _d: object, _e: object) -> object:
        return fn(cells)

    return call


def admit_case(
    idx: int, directory: str, items: Items, report: dict[str, object], cases: Cases
) -> None:
    """Admit one artifact directory, check it and register its cases.

    Args:
        idx: The directory's position on the command line.
        directory: The artifact directory.
        items: Input kind to inputs.
        report: The report to extend.
        cases: The timing cases to extend.

    Raises:
        AssertionError: If an implementation disagrees with the table or another.

    """
    mod = load(directory)
    acc = mod.admit(directory)  # full admission: hashes + complete table differential
    # Full-output equality over the complete admission input set (3.26M
    # cells) for both implementations against the pinned Bend table.
    table_bytes = (Path(directory) / mod.TABLE_NAME).read_bytes()
    if mod.render(acc.accept).encode() != table_bytes:
        raise AssertionError
    if idx == 0:
        if mod.render(ref_accept).encode() != table_bytes:
            msg = "python reference differs"
            raise AssertionError(msg)
        report["python_reference_table_equal"] = True
    report[f"bend{idx}_table_equal"] = True
    g = served(acc)
    # Equality on the timing inputs too (lists and tensors, server path).
    for kind in ("table", "served"):
        got_b = list(itertools.starmap(g, items[f"{kind}_tensor"]))
        got_l = list(itertools.starmap(acc.accept, items[f"{kind}_list"]))
        want = list(itertools.starmap(ref_server, items[f"{kind}_tensor"]))
        if not got_b == want == got_l:
            raise AssertionError(kind)
    for kind in ("table", "served"):
        cases[f"bend{idx}_server_{kind}"] = (g, f"{kind}_tensor")
        cases[f"bend{idx}_loader_{kind}"] = (acc.accept, f"{kind}_list")
    # ctypes floor: the exported function on a prefilled cell buffer.
    fn, cells = acc.function, acc.cells
    acc.accept(*items["served_list"][0])
    cases[f"bend{idx}_ctypes_floor"] = (ctypes_floor(fn, cells), "served_list")


def main(argv: list[str]) -> None:
    """Admit, check and time every artifact directory, then write the report.

    Args:
        argv: The report path, then the artifact directories.

    """
    out_path = argv[0]
    dirs = argv[1:]
    rng = random.Random(20260926)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  seeded RNG generates reproducible benchmark inputs
    table = inputs_table_k7(rng, 20000)
    real = inputs_served(rng, 20000)
    items: Items = {
        "table_list": stop_tuple(table),
        "served_list": stop_tuple(real),
        "table_tensor": as_tensors(table),
        "served_tensor": as_tensors(real),
    }
    report: dict[str, object] = {
        "python": sys.version,
        "torch": torch.__version__,
        "dirs": dirs,
    }
    cases: Cases = {}
    for idx, d in enumerate(dirs):
        admit_case(idx, d, items, report, cases)
    for kind in ("table", "served"):
        cases[f"py_server_{kind}"] = (ref_server, f"{kind}_tensor")
        cases[f"py_list_{kind}"] = (ref_accept, f"{kind}_list")
    raw = bench(cases, items)
    summary = summarize(raw)
    report["summary"] = summary
    report["raw_ns_per_call"] = raw
    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, v in summary.items():
        sys.stdout.write(
            f"{k:28s} median {v['median_ns']:8.1f} ns  "
            f"min {v['min_ns']:8.1f}  max {v['max_ns']:8.1f}\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
