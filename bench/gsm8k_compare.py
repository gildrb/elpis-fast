# Copyright (c) 2026 Gil Rodrigues
r"""Replay r0b0tlab's GSM8K acceptance workload against the serving endpoint.

The workload is r0b0tlab/qwen38-exl3-dflash2 ``scripts/acceptance_check.py``: the
first ``--n`` GSM8K test questions, each sent as the raw prompt
``<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n`` with greedy
decoding and 512 new tokens. Like that script's ExLlamaV3 ``generate()`` default,
the server's raw ``/v1/completions`` encodes special-token strings as plain text, so
both see the same token IDs. Requests run one at a time with no retries.

It reports r0b0tlab's two metrics, the mean of per-request tok/s (completion tokens
over the request wall, send to complete response) and the mean per-request
acceptance length (committed tokens per verify round), plus pooled rates and GPU
board energy integrated by ``bench.power`` over each request interval. Standard
library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bench.exl3 import (
    DRAFT_PROPOSALS,
    MODEL,
    Client,
    canonical,
    digest,
    integer,
    loads,
    mapping,
    request_bytes,
    save,
    sequence,
    text,
)
from bench.power import NANOSECONDS, PowerSampler, integrate_power

# openai/grade-school-math grade_school_math/data/test.jsonl at this commit; its
# first 40 questions equal the Hugging Face openai/gsm8k test split's first 40.
DATASET_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
DATASET_ROWS = 1319
PROMPT = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
MAX_NEW_TOKENS = 512
# Each sample spawns nvidia-smi; 1 s keeps the sampler's own host load low.
DEFAULT_POWER_INTERVAL_SECONDS = 1.0
HTTP_OK = 200
FINISH_REASONS = frozenset({"stop", "length"})


@dataclass(frozen=True)
class Arguments:
    """Validated command line."""

    api_key_file: Path
    data: Path
    out: Path
    n: int
    gpu: str
    power_interval: float


@dataclass(frozen=True)
class Row:
    """One completed request; rates derive from exact counters and monotonic times."""

    index: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    rounds: int
    committed: int
    started_ns: int
    finished_ns: int
    text_sha256: str

    @property
    def wall_seconds(self) -> float:
        """Send to complete response body."""
        return (self.finished_ns - self.started_ns) / NANOSECONDS

    @property
    def tok_per_s(self) -> float:
        """Completion tokens per request wall second (r0b0tlab's per-request rate)."""
        return self.completion_tokens / self.wall_seconds

    @property
    def acceptance_length(self) -> float:
        """Committed tokens per native verify round."""
        return self.committed / self.rounds


def option[T](namespace: argparse.Namespace, name: str, kind: type[T]) -> T:
    """Read one parsed option, requiring the type argparse was told to produce.

    Returns:
        The option value.

    Raises:
        TypeError: The option is missing or of another type.

    """
    value: object = getattr(namespace, name, None)
    if not isinstance(value, kind):
        message = f"Option {name} did not parse to {kind.__name__}"
        raise TypeError(message)
    return value


def parse_arguments() -> Arguments:
    """Parse the command line.

    Returns:
        The typed arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--api-key-file", type=Path, required=True)
    _ = parser.add_argument(
        "--data", type=Path, required=True, help=f"copy of {DATASET_URL}"
    )
    _ = parser.add_argument("--out", type=Path, required=True, help="new record path")
    _ = parser.add_argument("--n", type=int, default=40)
    _ = parser.add_argument("--gpu", default="0", help="GPU index for power samples")
    _ = parser.add_argument(
        "--power-interval",
        type=float,
        default=DEFAULT_POWER_INTERVAL_SECONDS,
        help="seconds between board-power samples (0.1..60)",
    )
    namespace = parser.parse_args()
    return Arguments(
        api_key_file=option(namespace, "api_key_file", Path),
        data=option(namespace, "data", Path),
        out=option(namespace, "out", Path),
        n=option(namespace, "n", int),
        gpu=option(namespace, "gpu", str),
        power_interval=option(namespace, "power_interval", float),
    )


def questions(path: Path, count: int) -> list[str]:
    """Load the first ``count`` pinned GSM8K test questions, stripped as upstream.

    Returns:
        The question strings in file order.

    Raises:
        ValueError: If the file is not the pinned GSM8K test set or ``count`` is
            out of range.

    """
    if digest(path) != DATASET_SHA256:
        msg = f"GSM8K file is not {DATASET_URL}"
        raise ValueError(msg)
    rows = path.read_bytes().decode("utf-8").splitlines()
    if len(rows) != DATASET_ROWS:
        msg = "GSM8K test file has an unexpected row count"
        raise ValueError(msg)
    if not (1 <= count <= DATASET_ROWS):
        msg = f"--n must be 1..{DATASET_ROWS}"
        raise ValueError(msg)
    return [text(mapping(loads(row))["question"]).strip() for row in rows[:count]]


def request(client: Client, index: int, question: str) -> Row:
    """Send one greedy raw completion, timed from send to the complete body.

    Returns:
        The validated request record.

    Raises:
        ValueError: If the reply is not one finished choice with consistent
            native speculative counters.

    """
    payload = request_bytes({
        "model": MODEL,
        "prompt": PROMPT.format(question=question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
    })
    started = time.monotonic_ns()
    status, raw = client.exchange("POST", "/v1/completions", payload)
    finished = time.monotonic_ns()
    if status != HTTP_OK:
        msg = f"Request {index} returned HTTP {status}"
        raise ValueError(msg)
    body = mapping(loads(raw))
    choices = sequence(body.get("choices"))
    if len(choices) != 1:
        msg = f"Request {index} did not return exactly one choice"
        raise ValueError(msg)
    choice = mapping(choices[0])
    finish = text(choice.get("finish_reason"))
    if finish not in FINISH_REASONS:
        msg = f"Request {index} finished with {finish}"
        raise ValueError(msg)
    usage = mapping(body.get("usage"))
    completion = integer(usage.get("completion_tokens"))
    spec = mapping(usage.get("exl3_spec"))
    if set(spec) != {"rounds", "committed"}:
        msg = "Unexpected exl3_spec fields"
        raise ValueError(msg)
    rounds, committed = integer(spec["rounds"]), integer(spec["committed"])
    if not (
        1 <= rounds <= committed <= completion <= MAX_NEW_TOKENS
        and committed <= (DRAFT_PROPOSALS + 1) * rounds
    ):
        msg = f"Request {index} has inconsistent native speculative counters"
        raise ValueError(msg)
    return Row(
        index=index,
        prompt_tokens=integer(usage.get("prompt_tokens")),
        completion_tokens=completion,
        finish_reason=finish,
        rounds=rounds,
        committed=committed,
        started_ns=started,
        finished_ns=finished,
        text_sha256=hashlib.sha256(text(choice.get("text")).encode()).hexdigest(),
    )


def measure(
    client: Client, prompts: list[str], gpu: str, interval: float
) -> tuple[list[Row], list[float | None], dict[str, object]]:
    """Run every request under one board-power sampler.

    Returns:
        The rows, each row's joules (``None`` unless its interval is fully covered)
        and the sampler summary.

    """
    rows: list[Row] = []
    with PowerSampler(gpu=gpu, interval_seconds=interval) as sampler:
        for index, question in enumerate(prompts):
            rows.append(request(client, index, question))
    joules: list[float | None] = []
    for row in rows:
        value = integrate_power(
            sampler.samples, row.started_ns, row.finished_ns, sampler.max_gap_ns
        )["joules"]
        joules.append(value if isinstance(value, float) else None)
    return rows, joules, sampler.summary()


def summarize(rows: list[Row], joules: list[float | None]) -> dict[str, object]:
    """Pool the rows; energy only when every request interval is fully covered.

    Returns:
        The summary record.

    """
    covered = [value for value in joules if value is not None]
    energy = sum(covered) if len(covered) == len(rows) else None
    tokens = sum(row.completion_tokens for row in rows)
    wall = sum(row.wall_seconds for row in rows)
    rounds = sum(row.rounds for row in rows)
    return {
        "n": len(rows),
        "max_new_tokens": MAX_NEW_TOKENS,
        "mean_acceptance_length": sum(row.acceptance_length for row in rows)
        / len(rows),
        "pooled_acceptance_length": sum(row.committed for row in rows) / rounds,
        "mean_request_tok_per_s": sum(row.tok_per_s for row in rows) / len(rows),
        "pooled_tok_per_s": tokens / wall,
        "pooled_ms_per_round": 1000 * wall / rounds,
        "hit_token_cap": sum(
            1 for row in rows if row.completion_tokens >= MAX_NEW_TOKENS
        ),
        "prompt_tokens_min_max": [
            min(row.prompt_tokens for row in rows),
            max(row.prompt_tokens for row in rows),
        ],
        "completion_tokens": tokens,
        "wall_seconds": wall,
        "joules": energy,
        "mean_watts": None if energy is None else energy / wall,
        "tok_per_joule": None if energy is None else tokens / energy,
        "energy_coverage": f"{len(covered)} of {len(rows)} requests fully covered",
    }


def main() -> None:
    """Run the replay once and write one exclusive-create JSON record.

    Raises:
        ValueError: If the record exists or the endpoint is unhealthy or serves
            another model.

    """
    arguments = parse_arguments()
    if arguments.out.exists():
        msg = "Output record already exists"
        raise ValueError(msg)
    prompts = questions(arguments.data, arguments.n)
    client = Client(arguments.api_key_file.read_text(encoding="utf-8").strip())
    health = client.json("GET", "/health", None)
    if health != {"status": "ok"}:
        msg = "Endpoint is not healthy"
        raise ValueError(msg)
    served = sequence(client.json("GET", "/v1/models", None)["data"])
    names = [text(mapping(item)["id"]) for item in served]
    if names != [MODEL]:
        msg = f"Endpoint must serve only {MODEL}"
        raise ValueError(msg)
    rows, joules, power = measure(
        client, prompts, arguments.gpu, arguments.power_interval
    )
    summary = summarize(rows, joules)
    record = {
        "schema_version": 1,
        "producer": "bench.gsm8k_compare",
        "workload": "r0b0tlab acceptance_check.py: GSM8K test prefix, raw prompt",
        "dataset": {"url": DATASET_URL, "sha256": DATASET_SHA256},
        "summary": summary,
        "power": power,
        "rows": [
            {
                "index": row.index,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "finish_reason": row.finish_reason,
                "rounds": row.rounds,
                "committed": row.committed,
                "acceptance_length": row.acceptance_length,
                "wall_seconds": row.wall_seconds,
                "tok_per_s": row.tok_per_s,
                "joules": value,
                "text_sha256": row.text_sha256,
            }
            for row, value in zip(rows, joules, strict=True)
        ],
    }
    _ = canonical(record)
    save(arguments.out, record)
    _ = sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
