# Copyright (c) 2026 inference contributors.
"""Cold-prefill TTFT ladder for the EXL3 lane; see autoresearch.sh.

Every row sends one non-streaming 1-token request (TTFT) and then the same
prompt with a 32-token budget (continuation). A unique leading nonce line makes
each row's first 256-token KV page, and so every chained page after it, differ
from every earlier request of the ladder: each TTFT request is a cold prefill.
Rows are frozen and rendered before any generation and replayed from raw files.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from typing import TYPE_CHECKING

from bench import exl3
from bench.exl3 import integer, mapping, sequence

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from bench.tokenizer import RawTokenizer

PROTOCOL = "exl3-prefill-ttft-v1"
HTTP_OK = 200
NANOSECONDS = 1_000_000_000
# (raw-content depth, repetitions), ascending depth; repetitions run consecutively.
LADDER = ((8192, 3), (32768, 3), (131072, 2), (262000, 1))
# The corpus always covers 1.05 x the deepest frozen row, whatever ladder is planned.
DEEPEST = 262000
COVERAGE_PERCENT = 105
BUDGETS = {"ttft": 1, "continuation": 32}
NONCE = "[prefill measurement {rep} of {reps} at depth {depth}]"
SAMPLING = {"temperature": 0, "top_p": 1, "n": 1, "stream": False}
ORDER = "ascending_depth_then_repetition; per row ttft then continuation"
CACHE_POLICY = (
    "unique leading nonce per row: its first 256-token KV page, and every chained "
    "page after it, differs from every earlier request of this ladder, so each TTFT "
    "request is a cold prefill with no prefix reuse; the continuation repeats that "
    "prompt and may reuse its pages; no flush, no warmup"
)
TTFT_SCOPE = (
    "streaming TTFT is unavailable (the server buffers SSE), so TTFT is the wall "
    "time of a 1-token non-streaming request: prefill plus the first verify round "
    "plus HTTP, monotonic from request send through complete response body"
)
PRIMARY_SCOPE = (
    "geometric mean over the ladder depths of prefill_tok_s_<depth> = sum of "
    "native prompt_tokens / sum of TTFT wall seconds at that depth"
)
ROW_FILES = tuple(
    f"{kind}-{name}.json"
    for kind in BUDGETS
    for name in ("request", "render-response", "response", "timing")
)


def check_ladder(ladder: tuple[tuple[int, int], ...]) -> None:
    """Accept only a nonempty ascending ladder inside the frozen corpus coverage.

    Raises:
        ValueError: If the ladder is empty, unordered or outside coverage.

    """
    if not bool(ladder):
        msg = "Empty prefill ladder"
        raise ValueError(msg)
    if not all(type(depth) is int and type(reps) is int for depth, reps in ladder):
        msg = "Prefill ladder entries must be exact integers"
        raise ValueError(msg)
    if not all(reps > 0 for _, reps in ladder):
        msg = "Prefill repetitions must be positive"
        raise ValueError(msg)
    depths = [depth for depth, _ in ladder]
    if depths != sorted(set(depths)):
        msg = "Prefill ladder depths must be strictly ascending"
        raise ValueError(msg)
    if not (depths[0] > 0 and depths[-1] <= DEEPEST):
        msg = "Prefill depth outside the frozen corpus coverage"
        raise ValueError(msg)


def metric_names(ladder: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    """Name the primary metric, then per-depth prefill rate, TTFT and reuse wall.

    Args:
        ladder: The planned (depth, repetitions) ladder.

    Returns:
        The metric names in report order.

    """
    check_ladder(ladder)
    return (
        "prefill_tok_s",
        *(f"prefill_tok_s_{depth}" for depth, _ in ladder),
        *(f"ttft_s_{depth}" for depth, _ in ladder),
        *(f"reuse_request_s_{depth}" for depth, _ in ladder),
    )


def corpus(source: Path, tokenizer: RawTokenizer) -> tuple[str, dict[str, object]]:
    """Join the C1 corpus prompts as C1 does, repeated to cover 1.05 x 262000.

    Args:
        source: The frozen throughput prompt file.
        tokenizer: The raw-content tokenizer.

    Returns:
        The corpus document and its coverage record.

    Raises:
        ValueError: If the corpus is empty or too short.

    """
    prompts = exl3.corpus_prompts(source)
    required = -(-DEEPEST * COVERAGE_PERCENT // 100)
    # One pass plus its trailing separator is the repeating unit of the join.
    unit = tokenizer.count("\n\n".join(prompts) + "\n\n")
    if not (unit > 0):
        msg = "Frozen corpus pass has no tokens"
        raise ValueError(msg)
    factor = -(-required // unit)
    document = "\n\n".join(prompts * factor)
    tokens = tokenizer.count(document)
    if not (tokens >= required):
        msg = f"Prefill corpus covers {tokens} tokens, short of {required}"
        raise ValueError(msg)
    return document, {
        "prompts": len(prompts),
        "join": "\n\n",
        "pass_tokens": unit,
        "repetitions": factor,
        "corpus_tokens": tokens,
        "required_tokens": required,
        "coverage": f"{COVERAGE_PERCENT}% of {DEEPEST}",
    }


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _row_content(
    tokenizer: RawTokenizer, document: str, depth: int, repetition: int, reps: int
) -> tuple[str, dict[str, object]]:
    """Size one row's nonce-led content and start its plan record.

    Args:
        tokenizer: The raw-content tokenizer.
        document: The frozen corpus.
        depth: The raw-content depth target.
        repetition: The zero-based repetition index.
        reps: The repetitions at this depth.

    Returns:
        The row content and its initial plan record.

    """
    nonce = NONCE.format(rep=repetition + 1, reps=reps, depth=depth)
    content, cut, count = exl3.sized_content(tokenizer, document, depth, nonce)
    return content, {
        "depth_target": depth,
        "repetition": repetition,
        "repetitions": reps,
        "nonce": nonce,
        "corpus_prefix_chars": cut,
        "raw_content_tokens": count,
        "content_sha256": _sha(content.encode()),
    }


def _render_row(
    row: Path, client: exl3.Client, content: str
) -> tuple[list[int], dict[str, object]]:
    """Freeze both request bodies of a row and their rendered IDs.

    Args:
        row: The row directory.
        client: The server client.
        content: The row content.

    Returns:
        The rendered prompt IDs and the request and render digests.

    Raises:
        ValueError: If no request was rendered.

    """
    fields: dict[str, object] = {}
    rendered: list[int] | None = None
    for kind, budget in BUDGETS.items():
        payload = exl3.request_bytes({
            "model": exl3.MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": budget,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
        })
        exl3.write_new(row / f"{kind}-request.json", payload)
        status, raw = client.exchange("POST", "/v1/chat/completions/render", payload)
        exl3.write_new(row / f"{kind}-render-response.json", raw)
        if status != HTTP_OK:
            msg = f"Render returned HTTP {status}"
            raise ValueError(msg)
        ids = exl3.token_ids(raw)
        if not (rendered is None or ids == rendered):
            msg = "TTFT and continuation requests render different prompts"
            raise ValueError(msg)
        rendered = ids
        fields[f"{kind}_request_sha256"] = _sha(payload)
        fields[f"{kind}_render_response_sha256"] = _sha(raw)
    if rendered is None:
        msg = "Prefill row rendered no request"
        raise ValueError(msg)
    return rendered, fields


def plan(
    directory: Path,
    client: exl3.Client,
    tokenizer: RawTokenizer,
    ladder: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    """Freeze every prompt, both request bodies and their rendered IDs per row.

    Args:
        directory: The new plan directory.
        client: The server client.
        tokenizer: The raw-content tokenizer.
        ladder: The (depth, repetitions) ladder.

    Returns:
        The saved plan.

    Raises:
        ValueError: If the ladder, a render or a rendered prompt is invalid.

    """
    check_ladder(ladder)
    directory.mkdir(mode=0o700)
    document, coverage = corpus(
        directory.parent / "sources/bench/throughput-prompts.jsonl", tokenizer
    )
    rows: list[dict[str, object]] = []
    for depth, reps in ladder:
        for repetition in range(reps):
            started = time.monotonic_ns()
            content, record = _row_content(tokenizer, document, depth, repetition, reps)
            row = directory / exl3.row_name(depth, repetition)
            row.mkdir(mode=0o700)
            rendered, fields = _render_row(row, client, content)
            record.update(fields)
            if not (len(rendered) + BUDGETS["continuation"] <= exl3.CONTEXT):
                msg = "Rendered prompt plus continuation budget exceeds native context"
                raise ValueError(msg)
            record["rendered_prompt_tokens"] = len(rendered)
            record["rendered_token_ids_sha256"] = _sha(exl3.canonical(rendered))
            record["planning_ns"] = time.monotonic_ns() - started
            rows.append(record)
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "ladder": [[depth, reps] for depth, reps in ladder],
        "order": ORDER,
        "concurrency": 1,
        "output_budgets": dict(BUDGETS),
        "nonce_template": NONCE,
        "instruction": exl3.INSTRUCTION,
        "raw_content_tolerance_tokens": 2,
        "raw_content_tokenizer_sha256": tokenizer.sha256,
        "corpus": coverage,
        "sampling": dict(SAMPLING),
        "cache_policy": CACHE_POLICY,
        "ttft_scope": TTFT_SCOPE,
        "rows": rows,
    }
    exl3.save(directory / "plan.json", result)
    return result


def run(
    directory: Path,
    client: exl3.Client,
    frozen: dict[str, object],
    checkpoint: Callable[[], object],
) -> None:
    """Send each row's two frozen requests once, in order; retain raw bytes first.

    Args:
        directory: The plan directory.
        client: The server client.
        frozen: The frozen plan.
        checkpoint: Called before each row.

    Raises:
        ValueError: If a request fails.

    """
    for value in sequence(frozen["rows"]):
        planned = mapping(value)
        checkpoint()
        row = directory / exl3.row_name(
            integer(planned["depth_target"]), integer(planned["repetition"])
        )
        for kind in BUDGETS:
            payload = (row / f"{kind}-request.json").read_bytes()
            started_unix = time.time_ns()
            started = time.monotonic_ns()
            status, raw = client.exchange("POST", "/v1/chat/completions", payload)
            received = time.monotonic_ns()
            exl3.write_new(row / f"{kind}-response.json", raw)
            exl3.save(
                row / f"{kind}-timing.json",
                {
                    "status": status,
                    "request_started_unix_ns": started_unix,
                    "request_started_monotonic_ns": started,
                    "response_received_monotonic_ns": received,
                    "clock": (
                        "monotonic from request send through complete response body"
                    ),
                },
            )
            if status != HTTP_OK:
                msg = f"Prefill {kind} request returned HTTP {status}"
                raise ValueError(msg)
        exl3.save(row / "row.json", measured_row(directory, planned))


def _rendered_ids(row: Path, kind: str, planned: dict[str, object]) -> int:
    """Check one request's frozen bytes and rendered IDs against the plan.

    Args:
        row: The row directory.
        kind: The request kind.
        planned: The frozen row plan.

    Returns:
        The planned rendered prompt token count.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    payload = (row / f"{kind}-request.json").read_bytes()
    if _sha(payload) != planned[f"{kind}_request_sha256"]:
        msg = f"Prefill {kind} request bytes differ from the frozen plan"
        raise ValueError(msg)
    render = (row / f"{kind}-render-response.json").read_bytes()
    if _sha(render) != planned[f"{kind}_render_response_sha256"]:
        msg = f"Prefill {kind} render receipt differs from the frozen plan"
        raise ValueError(msg)
    ids = exl3.token_ids(render)
    rendered = integer(planned["rendered_prompt_tokens"])
    if not (
        len(ids) == rendered
        and _sha(exl3.canonical(ids)) == planned["rendered_token_ids_sha256"]
    ):
        msg = f"Prefill {kind} rendered IDs differ from the frozen plan"
        raise ValueError(msg)
    return rendered


def _timing(row: Path, kind: str) -> tuple[object, int, int]:
    """Read one request's retained timing.

    Args:
        row: The row directory.
        kind: The request kind.

    Returns:
        The raw Unix start, then monotonic start and receipt times in ns.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    timing = exl3.document(row / f"{kind}-timing.json")
    if timing.get("status") != HTTP_OK:
        msg = f"Prefill {kind} request failed"
        raise ValueError(msg)
    started = integer(timing.get("request_started_monotonic_ns"))
    received = integer(timing.get("response_received_monotonic_ns"))
    if not (received - started > 0):
        msg = f"Nonpositive prefill {kind} wall time"
        raise ValueError(msg)
    return timing.get("request_started_unix_ns"), started, received


def _choice(response: dict[str, object], kind: str) -> tuple[object, dict[str, object]]:
    """Validate the single assistant choice of one response.

    Args:
        response: The parsed chat completion.
        kind: The request kind.

    Returns:
        The finish reason and the generated text fields.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    choices = sequence(response.get("choices"))
    if not (
        response.get("object") == "chat.completion"
        and response.get("model") == exl3.MODEL
        and len(choices) == 1
    ):
        msg = "Expected one chat completion"
        raise ValueError(msg)
    choice = mapping(choices[0])
    message = mapping(choice.get("message"))
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not (
        choice.get("index") == 0
        and message.get("role") == "assistant"
        and set(message) == {"role", "content", "reasoning_content"}
        and isinstance(content, str)
        and (reasoning is None or isinstance(reasoning, str))
    ):
        msg = f"Incomplete or unexpected prefill {kind} choice"
        raise ValueError(msg)
    return choice.get("finish_reason"), {
        "content": content,
        "reasoning_content": reasoning,
    }


def _usage(
    response: dict[str, object], kind: str, rendered: int, finish: object
) -> tuple[int, int, object]:
    """Validate one response's native usage against the rendered request.

    Args:
        response: The parsed chat completion.
        kind: The request kind.
        rendered: The planned rendered prompt token count.
        finish: The choice finish reason.

    Returns:
        The prompt tokens, completion tokens and speculative counters.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    usage = mapping(response.get("usage"))
    if frozenset(usage) not in {
        frozenset({"prompt_tokens", "completion_tokens", "total_tokens"}),
        frozenset({
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "exl3_spec",
        }),
    }:
        msg = f"Unexpected prefill {kind} usage fields"
        raise ValueError(msg)
    prompt = integer(usage["prompt_tokens"])
    completion = integer(usage["completion_tokens"])
    if prompt != rendered:
        msg = f"Native {kind} prompt usage differs from the rendered request"
        raise ValueError(msg)
    budget = BUDGETS[kind]
    if not (
        integer(usage["total_tokens"]) == prompt + completion
        and 1 <= completion <= budget
        and isinstance(finish, str)
        and finish in {"stop", "length"}
        and (finish == "stop" or completion == budget)
        and (kind != "ttft" or completion == 1)
    ):
        msg = f"Invalid prefill {kind} completion usage or finish"
        raise ValueError(msg)
    spec = (
        exl3.spec_counters(usage["exl3_spec"], completion)
        if "exl3_spec" in usage
        else None
    )
    return prompt, completion, spec


def _exchange(row: Path, kind: str, planned: dict[str, object]) -> dict[str, object]:
    """Validate one request of a row entirely from its retained raw files.

    Args:
        row: The row directory.
        kind: The request kind.
        planned: The frozen row plan.

    Returns:
        The validated exchange record.

    """
    rendered = _rendered_ids(row, kind, planned)
    started_unix, started, received = _timing(row, kind)
    response = mapping(exl3.loads((row / f"{kind}-response.json").read_bytes()))
    finish, generated = _choice(response, kind)
    prompt, completion, spec = _usage(response, kind, rendered, finish)
    return {
        "request_sha256": planned[f"{kind}_request_sha256"],
        "response_sha256": exl3.digest(row / f"{kind}-response.json"),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "finish_reason": finish,
        "request_started_unix_ns": integer(started_unix),
        "request_started_monotonic_ns": started,
        "response_received_monotonic_ns": received,
        "wall_ns": received - started,
        "exl3_spec": spec,
        "text": generated,
        "text_sha256": _sha(exl3.canonical(generated)),
    }


def measured_row(directory: Path, planned: dict[str, object]) -> dict[str, object]:
    """Validate one row's TTFT and continuation from retained raw files.

    Args:
        directory: The plan directory.
        planned: The frozen row plan.

    Returns:
        The measured row.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    row = directory / exl3.row_name(
        integer(planned["depth_target"]), integer(planned["repetition"])
    )
    ttft = _exchange(row, "ttft", planned)
    continuation = _exchange(row, "continuation", planned)
    if not (
        integer(ttft["response_received_monotonic_ns"])
        <= integer(continuation["request_started_monotonic_ns"])
    ):
        msg = "Prefill requests overlapped; concurrency must be one"
        raise ValueError(msg)
    ttft_wall = integer(ttft["wall_ns"])
    return {
        "depth_target": planned["depth_target"],
        "repetition": planned["repetition"],
        "rendered_prompt_tokens": planned["rendered_prompt_tokens"],
        "ttft": ttft,
        "continuation": continuation,
        "ttft_seconds": ttft_wall / 1_000_000_000,
        "prefill_tok_s": integer(ttft["prompt_tokens"]) * 1_000_000_000 / ttft_wall,
        "continuation_seconds": integer(continuation["wall_ns"]) / 1_000_000_000,
        "ttft_text": ttft["text"],
        "continuation_text": continuation["text"],
        "continuation_text_sha256": continuation["text_sha256"],
    }


def _require_serial(rows: list[dict[str, object]]) -> None:
    """Require every request to finish before the next one starts.

    Args:
        rows: The measured rows in plan order.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    spans = [
        (
            integer(mapping(row[kind])["request_started_monotonic_ns"]),
            integer(mapping(row[kind])["response_received_monotonic_ns"]),
        )
        for row in rows
        for kind in BUDGETS
    ]
    if not all(
        previous[1] <= current[0] for previous, current in itertools.pairwise(spans)
    ):
        msg = "Prefill requests overlapped; concurrency must be one"
        raise ValueError(msg)


def _depth_metrics(
    rows: list[dict[str, object]], ladder: tuple[tuple[int, int], ...]
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, object]]:
    """Pool the measured rows of each ladder depth.

    Args:
        rows: The measured rows.
        ladder: The (depth, repetitions) ladder.

    Returns:
        The per-depth prefill rates, mean TTFTs, mean reuse walls and pooled sums.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    rates: dict[str, float] = {}
    ttfts: dict[str, float] = {}
    reuses: dict[str, float] = {}
    pooled: dict[str, object] = {}
    for depth, reps in ladder:
        selected = [row for row in rows if row["depth_target"] == depth]
        if len(selected) != reps:
            msg = "Incomplete prefill ladder"
            raise ValueError(msg)
        prompt = sum(integer(mapping(row["ttft"])["prompt_tokens"]) for row in selected)
        ttft_wall = sum(integer(mapping(row["ttft"])["wall_ns"]) for row in selected)
        reuse_wall = sum(
            integer(mapping(row["continuation"])["wall_ns"]) for row in selected
        )
        if not (prompt > 0 and ttft_wall > 0 and reuse_wall > 0):
            msg = "Missing positive prefill window"
            raise ValueError(msg)
        rates[f"prefill_tok_s_{depth}"] = prompt * 1_000_000_000 / ttft_wall
        ttfts[f"ttft_s_{depth}"] = ttft_wall / 1_000_000_000 / reps
        reuses[f"reuse_request_s_{depth}"] = reuse_wall / 1_000_000_000 / reps
        pooled[str(depth)] = {
            "prompt_tokens": prompt,
            "ttft_wall_ns": ttft_wall,
            "continuation_wall_ns": reuse_wall,
            "repetitions": reps,
        }
    return rates, ttfts, reuses, pooled


def admit(
    evidence: exl3.Evidence,
    directory: Path,
    ladder: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    """Recompute every row and the per-depth and primary metrics from raw files.

    Args:
        evidence: The evidence retainer.
        directory: The plan directory.
        ladder: The (depth, repetitions) ladder.

    Returns:
        The admitted prefill result.

    Raises:
        ValueError: If a retained artifact differs from the plan or is invalid.

    """
    check_ladder(ladder)
    frozen = exl3.document(evidence.retain(directory / "plan.json"))
    if not (
        frozen.get("protocol") == PROTOCOL
        and frozen.get("ladder") == [[depth, reps] for depth, reps in ladder]
        and frozen.get("output_budgets") == BUDGETS
    ):
        msg = "Unexpected prefill plan"
        raise ValueError(msg)
    planned_rows = [mapping(value) for value in sequence(frozen["rows"])]
    if [(row["depth_target"], row["repetition"]) for row in planned_rows] != [
        (depth, rep) for depth, reps in ladder for rep in range(reps)
    ]:
        msg = "Prefill plan order differs"
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for planned in planned_rows:
        row = measured_row(directory, planned)
        name = exl3.row_name(
            integer(planned["depth_target"]), integer(planned["repetition"])
        )
        for filename in ROW_FILES:
            evidence.retain(directory / name / filename)
        if exl3.document(evidence.retain(directory / name / "row.json")) != row:
            msg = "Prefill producer row differs from raw replay"
            raise ValueError(msg)
        rows.append(row)
    _require_serial(rows)
    rates, ttfts, reuses, pooled = _depth_metrics(rows, ladder)
    primary = math.exp(
        math.fsum(math.log(rate) for rate in rates.values()) / len(rates)
    )
    metrics = {"prefill_tok_s": primary, **rates, **ttfts, **reuses}
    if not (
        tuple(metrics) == metric_names(ladder)
        and all(math.isfinite(value) and value > 0 for value in metrics.values())
    ):
        msg = "Prefill metrics are incomplete or nonpositive"
        raise ValueError(msg)
    return {
        "protocol": PROTOCOL,
        "ladder": [[depth, reps] for depth, reps in ladder],
        "rows": rows,
        "metrics": metrics,
        "pooled": pooled,
        "primary_scope": PRIMARY_SCOPE,
        "metric_scope": {
            "prefill_tok_s_<depth>": (
                "sum native prompt_tokens / sum TTFT wall seconds at that depth"
            ),
            "ttft_s_<depth>": "mean TTFT wall seconds at that depth",
            "reuse_request_s_<depth>": (
                "mean continuation wall seconds at that depth (32-token budget, "
                "same prompt right after its TTFT request); informational"
            ),
        },
        "cache_policy": CACHE_POLICY,
        "ttft_scope": TTFT_SCOPE,
    }
