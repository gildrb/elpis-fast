# Copyright (c) 2026 Gil Rodrigues
r"""Autoresearch GPU workload: cold prefill and decode, in process, one image.

Runs inside the candidate image in place of the server (GPU window, service
stopped). Builds the served ``exl3_server.Server`` (target + DFlash2 draft,
context 262144, cache 270336, cq 3, no prefix cache) and drives its generator
directly, one greedy job at a time. The page table is reset before every job:
every prompt is a cold prefill.

Prompts: the frozen C1 corpus (``bench/throughput-prompts.jsonl``) repeated;
content = nonce line + corpus prefix + C1 instruction, rendered through the
served chat template; the cut makes the rendered length land in
[depth, depth + 2].

Order: warm-up (1K x 32 tokens, 8K x 1 token; untimed), then
  prefill  8192 x2, 32768 x2, 131072 x1 (1 token each: TTFT),
  decode   1024 8192 32768 32768 8192 1024 (256 tokens each, no stop),
  native   262000 x1, 128 tokens (TTFT and decode at native context).
Each decode depth runs one prompt twice.

  python -I -B /work/ar_gpu.py --dry-run   (CPU: size prompts -> /out/plan.json)
  python -I -B /work/ar_gpu.py             (GPU: -> /out/result.json)
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import torch
from exllamav3 import Config, Generator, Job, Tokenizer
from exllamav3.generator.sampler.presets import ArgmaxSampler

if TYPE_CHECKING:
    from types import ModuleType

SERVER_PY = Path("/opt/qwen/serve/exl3_server.py")
TARGET = "/models/qwen38-27b-exl3"
DRAFT = "/models/dflash2-exl3"
CORPUS = Path("/work/throughput-prompts.jsonl")
OUT = Path("/out")
PLAN = OUT / "plan.json"
RESULT = OUT / "result.json"
CORPUS_REPEAT = 384
DEPTH_TOLERANCE = 2
BATCH_ROWS = 1
MATRIX_DIMS = 2
NONCE = "[ar run {repetition} of {repetitions} at depth {depth}]"
INSTRUCTION = (
    "End of reference material. Write a careful technical summary of the main "
    "ideas above, then give one original worked Python example with tests."
)
DECODE_TOKENS = 256
NATIVE_TOKENS = 128
MAX_CHUNK = 2048
NVML_PERIOD_S = 0.05
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
THROTTLE_POWER = 0x4
THROTTLE_THERMAL = 0x60

type Event = dict[str, object]
type Record = dict[str, object]


@dataclass(frozen=True)
class Row:
    """One timed or warm-up request."""

    kind: str
    depth: int
    rep: int
    reps: int
    new_tokens: int

    @property
    def key(self) -> str:
        """Plan key: depth/rep/reps."""
        return f"{self.depth}/{self.rep}/{self.reps}"


WARM = (Row("warm", 1024, 2, 3, 32), Row("warm", 8192, 2, 3, 1))
PREFILL = (
    Row("prefill", 8192, 0, 2, 1),
    Row("prefill", 8192, 1, 2, 1),
    Row("prefill", 32768, 0, 2, 1),
    Row("prefill", 32768, 1, 2, 1),
    Row("prefill", 131072, 0, 1, 1),
)
DECODE = tuple(
    Row("decode", d, 0, 1, DECODE_TOKENS)
    for d in (1024, 8192, 32768, 32768, 8192, 1024)
)
NATIVE = (Row("native", 262000, 0, 1, NATIVE_TOKENS),)
ROWS = WARM + PREFILL + DECODE + NATIVE


class BenchError(RuntimeError):
    """The workload cannot produce a valid measurement."""


def call_untyped(function: object) -> object:
    """Call a function whose static type is not a plain callable.

    Returns:
        The call's result.

    Raises:
        BenchError: The value is not callable.

    """
    if not callable(function):
        msg = "not callable"
        raise BenchError(msg)
    return function()


@dataclass(frozen=True)
class Engine:
    """The parts of the loaded ``exl3_server.Server`` this workload drives."""

    gen: Generator
    tokenizer: Tokenizer

    def iterate(self) -> list[Event]:
        """Run one generator step.

        ``iterate`` is wrapped by ``torch.inference_mode``, which type checkers
        see as the decorator object; narrow it here.

        Returns:
            The step's events.

        Raises:
            BenchError: The step is not callable or returned other data.

        """
        events = call_untyped(self.gen.iterate)
        if not isinstance(events, list):
            msg = "generator iterate did not return a list"
            raise BenchError(msg)
        out: list[Event] = []
        for e in events:
            if not isinstance(e, dict):
                msg = "generator event is not a dict"
                raise BenchError(msg)
            out.append({str(k): v for k, v in e.items()})
        return out


def load_module(name: str, path: Path) -> ModuleType:
    """Import a file as a module.

    Returns:
        The executed module.

    Raises:
        BenchError: The file cannot be loaded.

    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load {path}"
        raise BenchError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def corpus_document() -> str:
    """Join the frozen C1 prompts, repeated to cover 262K tokens.

    Returns:
        The corpus document.

    Raises:
        BenchError: A corpus line has no first-turn text.

    """
    prompts: list[str] = []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        conv = json.loads(line).get("conversations")
        if not conv or not isinstance(conv[0].get("value"), str):
            msg = "invalid corpus line"
            raise BenchError(msg)
        prompts.append(conv[0]["value"])
    return "\n\n".join(prompts * CORPUS_REPEAT)


def content_for(doc: str, row: Row, cut: int) -> str:
    """Build the request content of a row at a corpus cut.

    Returns:
        Nonce line, corpus prefix, blank line, instruction.

    """
    nonce = NONCE.format(repetition=row.rep + 1, repetitions=row.reps, depth=row.depth)
    return nonce + "\n" + doc[:cut] + "\n\n" + INSTRUCTION


def sha_json(value: object) -> str:
    """Hash a value's JSON text.

    Returns:
        The sha256 hex digest.

    """
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class Renderer:
    """The served chat rendering (parse_chat + hf_chat_template)."""

    def __init__(self, srv: ModuleType, tokenizer: Tokenizer) -> None:
        """Bind the server module and the target tokenizer."""
        self.srv = srv
        self.tok = tokenizer

    def ids(self, content: str) -> torch.Tensor:
        """Render one user message with the generation prompt.

        Returns:
            Token ids, shape (1, n).

        Raises:
            BenchError: The template returns another shape.

        """
        body = {
            "model": self.srv.MODEL_NAME,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
        }
        chat = self.srv.parse_chat(body)
        ids = self.tok.hf_chat_template(
            chat.messages, add_generation_prompt=True, **chat.template_kwargs
        )
        if not isinstance(ids, torch.Tensor):
            msg = f"chat template returned {type(ids).__name__}"
            raise BenchError(msg)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != MATRIX_DIMS or ids.shape[0] != BATCH_ROWS:
            msg = f"unexpected rendered shape {tuple(ids.shape)}"
            raise BenchError(msg)
        return ids

    def plan_entry(self, doc: str, row: Row) -> dict[str, object]:
        """Size one row: the smallest cut whose length reaches the depth.

        Returns:
            The cut, the token count and the ids' sha256.

        Raises:
            BenchError: The corpus is too short or the length overshoots.

        """
        lo, hi = 0, len(doc)
        if self.ids(content_for(doc, row, hi)).shape[-1] < row.depth:
            msg = f"corpus too short for depth {row.depth}"
            raise BenchError(msg)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.ids(content_for(doc, row, mid)).shape[-1] >= row.depth:
                hi = mid
            else:
                lo = mid
        ids = self.ids(content_for(doc, row, hi))
        n = int(ids.shape[-1])
        if not row.depth <= n <= row.depth + DEPTH_TOLERANCE:
            msg = f"{row.key}: rendered {n} tokens, not within +{DEPTH_TOLERANCE}"
            raise BenchError(msg)
        return {"cut": hi, "n_ids": n, "ids_sha256": sha_json(ids.flatten().tolist())}

    def planned(self, doc: str, row: Row, entry: dict[str, object]) -> torch.Tensor:
        """Render a row at its planned cut and check it against the plan.

        Returns:
            Token ids, shape (1, n).

        Raises:
            BenchError: The ids differ from the dry-run plan.

        """
        cut = entry["cut"]
        if not isinstance(cut, int):
            msg = f"{row.key}: bad plan cut"
            raise BenchError(msg)
        ids = self.ids(content_for(doc, row, cut))
        if (
            ids.shape[-1] != entry["n_ids"]
            or sha_json(ids.flatten().tolist()) != entry["ids_sha256"]
        ):
            msg = f"{row.key}: rendered ids differ from the dry-run plan"
            raise BenchError(msg)
        return ids


class Nvml:
    """Device 0 power, clocks, throttle reasons and temperature."""

    def __init__(self) -> None:
        """Open NVML and read the enforced power limit.

        Raises:
            BenchError: NVML is unavailable.

        """
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.h = ctypes.c_void_p()
        if (
            self.lib.nvmlInit_v2() != 0
            or self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) != 0
        ):
            msg = "NVML init failed"
            raise BenchError(msg)
        lim = ctypes.c_uint()
        if self.lib.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(lim)) != 0:
            msg = "NVML power limit unavailable"
            raise BenchError(msg)
        self.power_limit_w = lim.value / 1000.0

    def sample(self) -> tuple[float, int, int, int, int]:
        """Read power W, SM MHz, memory MHz, throttle reasons, temperature C.

        Returns:
            One sample.

        Raises:
            BenchError: A query failed.

        """
        p, c, m, t = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        r = ctypes.c_ulonglong()
        lib, h = self.lib, self.h
        if (
            lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) != 0
            or lib.nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, ctypes.byref(c)) != 0
            or lib.nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM, ctypes.byref(m)) != 0
            or lib.nvmlDeviceGetCurrentClocksThrottleReasons(h, ctypes.byref(r)) != 0
            or lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)) != 0
        ):
            msg = "NVML sample failed"
            raise BenchError(msg)
        return p.value / 1000.0, c.value, m.value, r.value, t.value


class Sampler:
    """Background NVML sampling while one request runs."""

    def __init__(self, nvml: Nvml) -> None:
        """Prepare an empty sample list."""
        self.nvml = nvml
        self.samples: list[tuple[float, int, int, int, int]] = []
        self.err: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> Self:
        """Start sampling.

        Returns:
            This sampler.

        """
        self.thread.start()
        return self

    def _loop(self) -> None:
        try:
            while not self.stop.is_set():
                self.samples.append(self.nvml.sample())
                self.stop.wait(NVML_PERIOD_S)
        except BenchError as e:
            self.err = repr(e)

    def __exit__(self, *_: object) -> None:
        """Stop sampling."""
        self.stop.set()
        self.thread.join()

    def summary(self) -> dict[str, float]:
        """Summarize the samples.

        Returns:
            Means and maxima.

        Raises:
            BenchError: Sampling failed or took no samples.

        """
        if self.err is not None or not self.samples:
            msg = f"NVML sampler: {self.err or 'no samples'}"
            raise BenchError(msg)
        s = self.samples
        n = len(s)
        return {
            "n": n,
            "power_w_mean": sum(x[0] for x in s) / n,
            "sm_mhz_mean": sum(x[1] for x in s) / n,
            "mem_mhz_mean": sum(x[2] for x in s) / n,
            "temp_c_max": max(x[4] for x in s),
            "power_cap_frac": sum(1 for x in s if x[3] & THROTTLE_POWER) / n,
            "thermal_frac": sum(1 for x in s if x[3] & THROTTLE_THERMAL) / n,
        }


@dataclass
class Trace:
    """Tokens and decode-round timings of one job."""

    ident: str
    tokens: list[int] = field(default_factory=list)
    first: float | None = None
    final: Event | None = None
    round_ms: list[float] = field(default_factory=list)
    round_n: list[int] = field(default_factory=list)

    def consume(self, events: list[Event], t0: float, t1: float) -> None:
        """Take one iterate()'s events; a round that emitted decode tokens is timed.

        Raises:
            BenchError: The generator reported an error.

        """
        n = 0
        for e in events:
            if e.get("stage") == "error":
                msg = f"job error: {e.get('error')!r}"
                raise BenchError(msg)
            if e.get("identifier") != self.ident:
                continue
            ids = e.get("token_ids")
            if isinstance(ids, torch.Tensor):
                self.tokens += ids.flatten().tolist()
                if self.first is None:
                    self.first = t1
                else:
                    n += int(ids.numel())
            if e.get("eos"):
                self.final = e
        if n:
            self.round_ms.append(1e3 * (t1 - t0))
            self.round_n.append(n)


def run_request(engine: Engine, nvml: Nvml, ids: torch.Tensor, row: Row) -> Record:
    """Run one cold greedy job; time TTFT and every decode round.

    Returns:
        The request record.

    Raises:
        BenchError: The generator was busy or the job did not finish.

    """
    gen = engine.gen
    if gen.num_remaining_jobs():
        msg = "generator not idle"
        raise BenchError(msg)
    gen.pagetable.reset_page_table()
    trace = Trace(uuid.uuid4().hex)
    job = Job(
        input_ids=ids,
        max_new_tokens=row.new_tokens,
        sampler=ArgmaxSampler(),
        stop_conditions=[],
        decode_special_tokens=False,
        identifier=trace.ident,
    )
    torch.cuda.synchronize()
    with Sampler(nvml) as smp:
        ts = time.perf_counter()
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            t0 = time.perf_counter()
            events = engine.iterate()
            trace.consume(events, t0, time.perf_counter())
        te = time.perf_counter()
    if trace.final is None or trace.first is None:
        msg = "incomplete job"
        raise BenchError(msg)
    nv = smp.summary()
    ttft = trace.first - ts
    sys.stdout.write(
        f"[{row.kind}] {row.key}: {ids.shape[-1]} tokens, TTFT {ttft:.2f} s, "
        f"{len(trace.round_ms)} rounds, {sum(trace.round_n)} decode tokens, "
        f"{nv['power_w_mean']:.0f} W, SM {nv['sm_mhz_mean']:.0f} MHz\n"
    )
    sys.stdout.flush()
    return {
        "kind": row.kind,
        "key": row.key,
        "depth": row.depth,
        "prompt_tokens": int(ids.shape[-1]),
        "ttft_s": ttft,
        "total_s": te - ts,
        "tokens": trace.tokens,
        "tokens_sha256": sha_json(trace.tokens),
        "decode_tokens": sum(trace.round_n),
        "rounds": len(trace.round_ms),
        "round_ms": trace.round_ms,
        "round_ms_median": statistics.median(trace.round_ms)
        if trace.round_ms
        else None,
        "decode_s": sum(trace.round_ms) / 1e3,
        "accepted_draft_tokens": trace.final.get("accepted_draft_tokens"),
        "rejected_draft_tokens": trace.final.get("rejected_draft_tokens"),
        "nvml": nv,
    }


def served(srv: ModuleType) -> Engine:
    """Build the served engine exactly as the server does.

    Returns:
        The Server instance.

    Raises:
        BenchError: The generator is not the served configuration.

    """
    ns = argparse.Namespace(
        target=TARGET,
        draft=DRAFT,
        model_name=srv.MODEL_NAME,
        max_model_len=srv.CONTEXT,
        cache_tokens=srv.CACHE_TOKENS,
        cq=3,
        prefix_cache=None,
    )
    server = srv.Server(ns)
    gen, tok = server.gen, server.tokenizer
    if not isinstance(gen, Generator) or not isinstance(tok, Tokenizer):
        msg = "unexpected Server.gen / Server.tokenizer types"
        raise BenchError(msg)
    if (
        gen.max_chunk_size != MAX_CHUNK
        or getattr(gen, "tree", False) is not True
        or getattr(gen, "tree_force_chain", True) is not False
    ):
        msg = "not the served configuration (chunk 2048, token tree on)"
        raise BenchError(msg)
    return Engine(gen, tok)


def plan_prompts(srv: ModuleType) -> None:
    """Size every row's prompt on the CPU and write the plan."""
    render = Renderer(srv, Tokenizer.from_config(Config.from_directory(TARGET)))
    doc = corpus_document()
    plan = {row.key: render.plan_entry(doc, row) for row in dict.fromkeys(ROWS)}
    PLAN.write_text(json.dumps(plan, indent=1))
    for key, entry in plan.items():
        sys.stdout.write(f"[plan] {key}: {entry['n_ids']} tokens\n")
    sys.stdout.write("DRY-RUN OK\n")


def measure(srv: ModuleType, res: dict[str, object], runs: list[Record]) -> None:
    """Load the engine and run every row, saving after each one."""
    plan = json.loads(PLAN.read_text())
    nvml = Nvml()
    res["power_limit_w"] = nvml.power_limit_w
    t0 = time.perf_counter()
    engine = served(srv)
    res["load_s"] = time.perf_counter() - t0
    render = Renderer(srv, engine.tokenizer)
    doc = corpus_document()
    prompts = {r.key: render.planned(doc, r, plan[r.key]) for r in dict.fromkeys(ROWS)}
    for row in ROWS:
        runs.append(run_request(engine, nvml, prompts[row.key], row))
        RESULT.write_text(json.dumps(res, indent=1))
    res["complete"] = True


def main() -> int:
    """Run the dry run or the measurement.

    Returns:
        The process exit status.

    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.environ["QWEN_PREFIX_PERSIST"] = "0"
    srv = load_module("exl3_server", SERVER_PY)
    if args.dry_run:
        plan_prompts(srv)
        return 0
    runs: list[Record] = []
    res: dict[str, object] = {"complete": False, "errors": [], "runs": runs}
    status = 0
    try:
        measure(srv, res, runs)
    except (BenchError, RuntimeError, OSError, KeyError, ValueError) as e:
        res["errors"] = [{"error": repr(e), "trace": traceback.format_exc()}]
        sys.stdout.write(f"FAIL: {e!r}\n")
        status = 1
    RESULT.write_text(json.dumps(res, indent=1))
    sys.stdout.write(f"AR-GPU {'DONE' if status == 0 else 'FAILED'} -> {RESULT}\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
