# Copyright (c) 2026 Gil Rodrigues
r"""Measure the endpoint on LocalMaxxing's canonical prompts; build speed-test payloads.

Fixed before measuring:

- Prompts: ``canonicalPrompts`` ``reasoning-v1`` and ``code-v1`` from a saved copy of
  https://www.localmaxxing.com/api/agent-context, each checked against its pinned
  SHA-256. Every request sends one user message: a leading
  ``[LocalMaxxing cache-bust nonce: <32 hex>]`` line, then the canonical text.
- Greedy chat completions (temperature 0), batch 1, the served chat template (thinking
  on), ``max_tokens`` 320.
- Per prompt: one excluded warmup, then five timed pairs: a 1-token request, then the
  full request, each with its own nonce. The server's SSE is buffered, so the first
  token's time is not observable inside the full request; TTFT is the paired 1-token
  request's wall time (send to complete body: prefill plus the first verify round,
  which emits the first token).
- Per pair: tokSOut = (completion tokens - 1) / (full wall - TTFT); tokSPrefill = the
  1-token request's prompt tokens / TTFT; tokSTotal = (prompt + completion tokens) /
  full wall. The pair with the median tokSOut is reported; every request is recorded.
- Speculative counts are the server's native per-request counters (verify rounds,
  committed tokens); every round drafts 7 tokens.
- Board power: ``bench.power`` over the whole run, integrated over each prompt's timed
  full requests. VRAM: nvidia-smi ``memory.used`` after every request (it includes the
  preallocated cache).
- Refuses unless the GPU runs the lane's declared state and that state is stock: power
  limit equal to the card's default, core and memory clock offsets 0.

Standard library only; the record and payloads carry no credentials.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bench.exl3 import (
    CONTEXT,
    DRAFT_PROPOSALS,
    MANIFEST,
    MODEL,
    ROOT,
    Client,
    _gpu,
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
from bench.gsm8k_compare import option
from bench.power import NANOSECONDS, PowerSampler, integrate_power

AGENT_CONTEXT_URL = "https://www.localmaxxing.com/api/agent-context"
PROMPT_SHA256 = {
    "reasoning-v1": "9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445",
    "code-v1": "a737db738bd4e37881d1737f8dc51cae0295eff9ece6551d39826ce8ed43a690",
}
NONCE_LINE = "[LocalMaxxing cache-bust nonce: {nonce}]\n"
MAX_TOKENS = 320
TIMED_PAIRS = 5
POWER_INTERVAL_SECONDS = 0.25
GPU = "0"
HTTP_OK = 200
FINISH_REASONS = frozenset({"stop", "length"})
ENTRYPOINT = ROOT / "serve/exl3-entrypoint.sh"
PATCH_SERIES = (ROOT / "patches/exl3/series", ROOT / "patches/exl3-ext/series")
COMMAND = (
    "/opt/venv/bin/python -B /opt/qwen/serve/exl3_server.py"
    " --target /models/qwen38-27b-exl3 --draft /models/dflash2-exl3"
    " --model-name qwen3.8-27b --max-model-len 262144 --cache-tokens 270336"
    " --cq 3 --host 0.0.0.0 --port 18020"
)
ENGINE_REPOSITORY = "https://github.com/r0b0tlab/exllamav3"
ENGINE_VERSIONS = {"355c6ee10fbd25b79070316a81ea0708cc18155a": "1.5.0"}
QUANTIZATION = "EXL3-4.00bpw"
STATE_FIELDS = (
    "name",
    "vbios_version",
    "power.limit",
    "power.default_limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "fan.speed",
    "clocks_event_reasons.active",
    "memory.used",
    "memory.total",
)
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
DMI_MEMORY_SIZE = re.compile(r"^E: MEMORY_DEVICE_\d+_SIZE=(\d+)$", re.MULTILINE)
NOTES_LIMIT = 2000
TIMINGS_LIMIT = 8192
PROMPT_SAMPLE_LIMIT = 2000
OUTPUT_SAMPLE_LIMIT = 4000


@dataclass(frozen=True)
class Arguments:
    """Validated command line."""

    api_key_file: Path
    context: Path
    image_id: str
    out: Path


@dataclass(frozen=True)
class Prompt:
    """One pinned canonical prompt."""

    identifier: str
    text: str
    sha256: str
    min_output_tokens: int


@dataclass(frozen=True)
class Reply:
    """One chat completion, timed from send to the complete body."""

    message: str
    started_ns: int
    finished_ns: int
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    rounds: int
    committed: int
    output: str
    usage: dict[str, object]
    vram_mib: int

    @property
    def wall_seconds(self) -> float:
        """Send to complete response body."""
        return (self.finished_ns - self.started_ns) / NANOSECONDS

    def record(self) -> dict[str, object]:
        """Every observed field, including the exact message and output text.

        Returns:
            The JSON-ready request record.

        """
        return {
            "message": self.message,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "wall_seconds": self.wall_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "rounds": self.rounds,
            "committed": self.committed,
            "output": self.output,
            "output_sha256": sha256(self.output),
            "usage": self.usage,
            "vram_mib_after": self.vram_mib,
        }


@dataclass(frozen=True)
class Pair:
    """A 1-token TTFT request followed by the full request."""

    probe: Reply
    full: Reply

    @property
    def ttft_seconds(self) -> float:
        """Wall time of the 1-token request."""
        return self.probe.wall_seconds

    @property
    def tok_s_out(self) -> float:
        """Tokens after the first over the full request's time after TTFT."""
        return (self.full.completion_tokens - 1) / (
            self.full.wall_seconds - self.ttft_seconds
        )

    @property
    def tok_s_prefill(self) -> float:
        """The 1-token request's prompt tokens over its wall time."""
        return self.probe.prompt_tokens / self.ttft_seconds

    @property
    def tok_s_total(self) -> float:
        """Prompt plus completion tokens over the full request wall."""
        return (
            self.full.prompt_tokens + self.full.completion_tokens
        ) / self.full.wall_seconds

    @property
    def tok_s_request(self) -> float:
        """Completion tokens over the full request wall, prefill included."""
        return self.full.completion_tokens / self.full.wall_seconds


@dataclass(frozen=True)
class Setup:
    """Served identity, host facts and launch recipe shared by every payload."""

    target_repository: str
    target_revision: str
    draft_repository: str
    draft_revision: str
    engine_revision: str
    engine_version: str
    patches: int
    command: str
    image_id: str
    gpu_name: str
    vram_gib: int
    power_limit_watts: float
    cpu: str
    ram_gib: int
    os: str


def sha256(value: str) -> str:
    """Hash UTF-8 text with SHA-256.

    Returns:
        The lowercase hex digest.

    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_arguments() -> Arguments:
    """Parse the command line.

    Returns:
        The typed arguments.

    Raises:
        ValueError: If the image ID is not ``sha256:<64 lowercase hex>``.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--api-key-file", type=Path, required=True)
    _ = parser.add_argument(
        "--context", type=Path, required=True, help=f"saved copy of {AGENT_CONTEXT_URL}"
    )
    _ = parser.add_argument(
        "--image-id", required=True, help="the served image ID (sha256:<64 hex>)"
    )
    _ = parser.add_argument("--out", type=Path, required=True, help="new record path")
    namespace = parser.parse_args()
    arguments = Arguments(
        api_key_file=option(namespace, "api_key_file", Path),
        context=option(namespace, "context", Path),
        image_id=option(namespace, "image_id", str),
        out=option(namespace, "out", Path),
    )
    if not (IMAGE_ID.fullmatch(arguments.image_id) is not None):
        msg = "--image-id must be sha256:<64 lowercase hex>"
        raise ValueError(msg)
    return arguments


def catalog(path: Path) -> tuple[list[Prompt], list[str]]:
    """Load the pinned canonical prompts and LocalMaxxing's discrete GPU names.

    Returns:
        The prompts in pinned order and the accepted GPU names.

    Raises:
        ValueError: If a pinned prompt is missing, duplicated, changed or needs more
            than the output budget.

    """
    context = mapping(loads(path.read_bytes()))
    found: dict[str, Prompt] = {}
    for entry in sequence(context.get("canonicalPrompts")):
        item = mapping(entry)
        identifier = text(item.get("id"))
        if identifier not in PROMPT_SHA256:
            continue
        body = text(item.get("text"))
        digest_hex = sha256(body)
        if not (
            digest_hex == PROMPT_SHA256[identifier] and item.get("sha256") == digest_hex
        ):
            msg = f"Canonical prompt {identifier} differs from its pinned SHA-256"
            raise ValueError(msg)
        minimum = integer(item.get("minOutputTokens"))
        if not (1 <= minimum <= MAX_TOKENS):
            msg = f"{identifier} needs more than {MAX_TOKENS} output tokens"
            raise ValueError(msg)
        if not (identifier not in found):
            msg = f"Duplicate canonical prompt {identifier}"
            raise ValueError(msg)
        found[identifier] = Prompt(identifier, body, digest_hex, minimum)
    if set(found) != set(PROMPT_SHA256):
        msg = "The saved agent context lacks a pinned canonical prompt"
        raise ValueError(msg)
    hardware = mapping(context.get("hardwareOptions"))
    names = [text(name) for name in sequence(hardware.get("discreteGpuNames"))]
    return [found[identifier] for identifier in PROMPT_SHA256], names


def executable(name: str) -> str:
    """Resolve a host program on PATH.

    Args:
        name: The program name.

    Returns:
        The absolute path of the program.

    Raises:
        FileNotFoundError: If the program is not on PATH, as exec would report.

    """
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    return found


def query(fields: tuple[str, ...]) -> dict[str, str]:
    """Read named nvidia-smi fields of the measured board.

    Returns:
        Field name to its unformatted value.

    Raises:
        ValueError: If nvidia-smi returns another number of fields.

    """
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: nvidia-smi from PATH + fixed query flags, no shell
        [
            executable("nvidia-smi"),
            f"--id={GPU}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != len(fields):
        msg = "nvidia-smi returned an unexpected field count"
        raise ValueError(msg)
    return dict(zip(fields, values, strict=True))


def answer_text(choice: dict[str, object]) -> str:
    """Join the reply message's reasoning and content.

    Args:
        choice: The single chat completion choice.

    Returns:
        The reasoning text (empty when null) followed by the content.

    Raises:
        ValueError: If ``reasoning_content`` is neither text nor null.
        TypeError: If ``content`` is not text.

    """
    answer = mapping(choice.get("message"))
    reasoning = answer.get("reasoning_content")
    content = answer.get("content")
    if reasoning is None:
        thinking = ""
    elif isinstance(reasoning, str):
        thinking = reasoning
    else:
        msg = "reasoning_content must be text or null"
        raise ValueError(msg)
    if not isinstance(content, str):
        msg = "content must be text"
        raise TypeError(msg)
    return thinking + content


def spec_counts(
    usage: dict[str, object], completion: int, max_tokens: int, prompt: Prompt
) -> tuple[int, int]:
    """Validate the server's speculative counters against the token counts.

    Args:
        usage: The reply's usage object.
        completion: The reply's completion token count.
        max_tokens: The request's output budget.
        prompt: The prompt the reply answers.

    Returns:
        The verify rounds and the committed tokens.

    Raises:
        ValueError: If the counters are not exactly rounds and committed, or are
            inconsistent with the token counts.

    """
    spec = mapping(usage.get("exl3_spec"))
    if set(spec) != {"rounds", "committed"}:
        msg = "Unexpected exl3_spec fields"
        raise ValueError(msg)
    rounds, committed = integer(spec["rounds"]), integer(spec["committed"])
    if not (
        1 <= completion <= max_tokens
        and 0 <= rounds <= committed <= completion
        and committed <= (DRAFT_PROPOSALS + 1) * rounds
    ):
        msg = (
            f"{prompt.identifier} reply has inconsistent token or speculative counters"
        )
        raise ValueError(msg)
    return rounds, committed


def reply_choice(body: dict[str, object]) -> tuple[str, str]:
    """Validate the reply's single choice.

    Args:
        body: The chat completion response object.

    Returns:
        The finish reason and the reasoning text followed by the content.

    Raises:
        ValueError: If there is not exactly one choice or it did not finish.

    """
    choices = sequence(body.get("choices"))
    if len(choices) != 1:
        msg = "Expected exactly one choice"
        raise ValueError(msg)
    choice = mapping(choices[0])
    finish = text(choice.get("finish_reason"))
    if finish not in FINISH_REASONS:
        msg = f"Unexpected finish reason {finish}"
        raise ValueError(msg)
    return finish, answer_text(choice)


def chat(client: Client, prompt: Prompt, max_tokens: int) -> Reply:
    """Send one greedy chat completion with a fresh nonce line.

    Returns:
        The validated reply, with VRAM read after the body arrived.

    Raises:
        ValueError: If the endpoint does not return HTTP 200.

    """
    message = NONCE_LINE.format(nonce=secrets.token_hex(16)) + prompt.text
    payload = request_bytes({
        "model": MODEL,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    started = time.monotonic_ns()
    status, raw = client.exchange("POST", "/v1/chat/completions", payload)
    finished = time.monotonic_ns()
    if status != HTTP_OK:
        msg = f"{prompt.identifier} request returned HTTP {status}"
        raise ValueError(msg)
    body = mapping(loads(raw))
    finish, output = reply_choice(body)
    usage = mapping(body.get("usage"))
    completion = integer(usage.get("completion_tokens"))
    rounds, committed = spec_counts(usage, completion, max_tokens, prompt)
    return Reply(
        message=message,
        started_ns=started,
        finished_ns=finished,
        prompt_tokens=integer(usage.get("prompt_tokens")),
        completion_tokens=completion,
        finish_reason=finish,
        rounds=rounds,
        committed=committed,
        output=output,
        usage=usage,
        vram_mib=int(query(("memory.used",))["memory.used"]),
    )


def measure(client: Client, prompt: Prompt) -> tuple[Reply, list[Pair]]:
    """Run the excluded warmup, then the timed pairs, back to back.

    Returns:
        The warmup reply and the timed pairs in order.

    Raises:
        ValueError: If a full request is below the verified-run minimum or not
            longer than its TTFT request.

    """
    warmup = chat(client, prompt, MAX_TOKENS)
    pairs: list[Pair] = []
    for _ in range(TIMED_PAIRS):
        probe = chat(client, prompt, 1)
        full = chat(client, prompt, MAX_TOKENS)
        if not (
            full.rounds >= 1
            and full.completion_tokens >= prompt.min_output_tokens
            and bool(full.output)
        ):
            msg = f"{prompt.identifier} full request is below the verified-run minimum"
            raise ValueError(msg)
        if not (full.wall_seconds > probe.wall_seconds):
            msg = f"{prompt.identifier} TTFT is not shorter than the full request"
            raise ValueError(msg)
        pairs.append(Pair(probe, full))
    return warmup, pairs


def median_index(pairs: list[Pair]) -> int:
    """Find the pair with the median tokSOut (odd pair count).

    Returns:
        The index into ``pairs``.

    Raises:
        ValueError: If the pair count is even.

    """
    if len(pairs) % 2 != 1:
        msg = "The median needs an odd number of pairs"
        raise ValueError(msg)
    order = sorted(range(len(pairs)), key=lambda index: pairs[index].tok_s_out)
    return order[len(order) // 2]


def pooled_watts(sampler: PowerSampler, pairs: list[Pair]) -> float:
    """Divide board energy over the timed full requests by their summed wall time.

    Returns:
        Mean board watts.

    Raises:
        TypeError: If the power samples do not cover a timed request.

    """
    joules = 0.0
    seconds = 0.0
    for pair in pairs:
        value = integrate_power(
            sampler.samples,
            pair.full.started_ns,
            pair.full.finished_ns,
            sampler.max_gap_ns,
        )["joules"]
        if not isinstance(value, float):
            msg = "Power samples do not cover a timed request"
            raise TypeError(msg)
        joules += value
        seconds += pair.full.wall_seconds
    return joules / seconds


def patch_count() -> int:
    """Count the patches named by the image's patch series files.

    Returns:
        The number of non-comment, non-blank series lines.

    """
    patches = 0
    for series in PATCH_SERIES:
        patches += sum(
            1
            for line in series.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return patches


def cpu_model() -> str:
    """Read the host CPU model name.

    Returns:
        The first ``model name`` of /proc/cpuinfo.

    Raises:
        ValueError: If /proc/cpuinfo has no model name.

    """
    cpu = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    if not bool(cpu):
        msg = "/proc/cpuinfo has no model name"
        raise ValueError(msg)
    return cpu


def os_pretty_name() -> str:
    """Read the host distribution name.

    Returns:
        ``PRETTY_NAME`` of /etc/os-release.

    """
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            release[key] = value.strip().strip('"')
    return text(release.get("PRETTY_NAME"))


def installed_memory_bytes() -> int:
    """Sum the DMI memory device sizes.

    Returns:
        Installed memory in bytes, a whole number of GiB.

    Raises:
        TypeError: If a matched DMI memory size is not text.
        ValueError: If DMI reports no memory or not a whole number of GiB.

    """
    dmi = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: udevadm from PATH + fixed DMI sysfs path, no shell
        [executable("udevadm"), "info", "/sys/devices/virtual/dmi/id"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    sizes: list[int] = []
    for match in DMI_MEMORY_SIZE.finditer(dmi):
        size = match.group(1)
        if not isinstance(size, str):
            msg = "Unreadable DMI memory size"
            raise TypeError(msg)
        sizes.append(int(size))
    if sizes == []:
        msg = "DMI reports no memory devices"
        raise ValueError(msg)
    ram_bytes = sum(sizes)
    if ram_bytes % 2**30 != 0:
        msg = "Installed memory is not a whole number of GiB"
        raise ValueError(msg)
    return ram_bytes


def setup(image_id: str, gpu: dict[str, str]) -> Setup:
    """Collect the served identity, launch recipe and host facts.

    Returns:
        The shared payload inputs.

    Raises:
        ValueError: If the engine, target, launch recipe or board memory differ
            from what the payloads record.

    """
    manifest = mapping(loads(MANIFEST.read_bytes()))
    sources = mapping(manifest.get("source_revisions"))
    target = mapping(sources.get("target"))
    draft = mapping(sources.get("draft"))
    engine = text(manifest.get("engine_revision"))
    if engine not in ENGINE_VERSIONS:
        msg = f"No recorded ExLlamaV3 version for {engine}"
        raise ValueError(msg)
    target_repository = text(target.get("repository"))
    if not target_repository.endswith("-" + QUANTIZATION):
        msg = f"The target is not an {QUANTIZATION} repository"
        raise ValueError(msg)
    recipe = " ".join(
        ENTRYPOINT.read_text(encoding="utf-8").replace("\\\n", " ").split()
    )
    if f"exec {COMMAND}" not in recipe:
        msg = "serve/exl3-entrypoint.sh no longer launches the recorded command"
        raise ValueError(msg)
    patches = patch_count()
    cpu = cpu_model()
    pretty = os_pretty_name()
    ram_bytes = installed_memory_bytes()
    memory_total = int(gpu["memory.total"])
    if memory_total % 1024 != 0:
        msg = "Board memory is not a whole number of GiB"
        raise ValueError(msg)
    return Setup(
        target_repository=target_repository,
        target_revision=text(target.get("revision")),
        draft_repository=text(draft.get("repository")),
        draft_revision=text(draft.get("revision")),
        engine_revision=engine,
        engine_version=ENGINE_VERSIONS[engine],
        patches=patches,
        command=COMMAND,
        image_id=image_id,
        gpu_name=gpu["name"],
        vram_gib=memory_total // 1024,
        power_limit_watts=float(gpu["power.limit"]),
        cpu=cpu,
        ram_gib=ram_bytes // 2**30,
        os=f"{pretty}, Linux {platform.release()}",
    )


def notes(prompt: Prompt, pairs: list[Pair], chosen: Pair, shared: Setup) -> str:
    """Compose the method and setup for LocalMaxxing's 2000-character notes field.

    Returns:
        The notes text.

    Raises:
        ValueError: If the notes exceed 2000 characters.

    """
    rates = sorted(pair.tok_s_out for pair in pairs)
    value = (
        f"Stock GPU: {shared.power_limit_watts:.0f} W power limit (the card's "
        "default), core and memory clock offsets 0; custom quiet fan curve. Isolated "
        "server container, no other GPU work. Qwen3.8-27B as r0b0tlab EXL3 4.00 bpw "
        f"@{shared.target_revision[:8]} at native context 262,144 with a 3-bit KV "
        "cache (270,336 tokens preallocated; peak VRAM includes it). Draft: r0b0tlab "
        f"DFlash2 EXL3 4.00 bpw @{shared.draft_revision[:8]}. ExLlamaV3 "
        f"{shared.engine_revision[:7]} (r0b0tlab community, native DFlash2) + "
        f"{shared.patches} patches: 8-row dynamic token-tree verification (anchor + 7 "
        "nodes), acceptance proved in Bend; greedy output identical with all drafts "
        f"rejected. Method: canonical {prompt.identifier} with a leading cache-bust "
        "nonce line, served chat template (thinking on), temperature 0, max_tokens "
        f"{MAX_TOKENS}, batch 1. One excluded warmup, then {len(pairs)} timed pairs (a "
        "1-token request, then the full request; each its own nonce). SSE is buffered, "
        "so TTFT = wall time of the paired 1-token request (prefill + the first verify "
        "round). tokSOut = (outputTokens - 1) / (full wall - TTFT); tokSPrefill = "
        "prompt tokens of the 1-token request / TTFT (includes HTTP and the first "
        "token); tokSTotal = (prompt + output tokens) / full wall. Reported: the "
        f"median-tokSOut pair; tokSOut over {len(pairs)} pairs "
        f"{rates[0]:.1f}-{rates[-1]:.1f}; whole request including prefill "
        f"{chosen.tok_s_request:.1f} tok/s. Spec counts: the server's per-request "
        f"counters; {DRAFT_PROPOSALS} drafted tokens per verify round. Power: "
        f"nvidia-smi power.draw every {POWER_INTERVAL_SECONDS} s, integrated over the "
        "timed full requests."
    )
    if not (len(value) <= NOTES_LIMIT):
        msg = "Notes exceed LocalMaxxing's 2000 characters"
        raise ValueError(msg)
    return value


def timings(pairs: list[Pair], chosen: int) -> dict[str, object]:
    """Collect server usage objects verbatim plus client walls, within 8192 bytes.

    Returns:
        The engineTimingsRaw object.

    Raises:
        ValueError: If the canonical object exceeds 8192 bytes.

    """
    value: dict[str, object] = {
        "source": (
            "elpis exl3_server usage objects (verbatim) and client monotonic walls"
        ),
        "reported_pair": chosen,
        "pairs": [
            {
                "ttft_request": {
                    "usage": pair.probe.usage,
                    "wall_ms": round(pair.probe.wall_seconds * 1000, 2),
                },
                "full_request": {
                    "usage": pair.full.usage,
                    "wall_ms": round(pair.full.wall_seconds * 1000, 2),
                    "finish_reason": pair.full.finish_reason,
                },
                "tok_s_out": round(pair.tok_s_out, 2),
            }
            for pair in pairs
        ],
    }
    if not (len(canonical(value)) <= TIMINGS_LIMIT):
        msg = "engineTimingsRaw exceeds 8192 bytes"
        raise ValueError(msg)
    return value


def payload(
    prompt: Prompt, pairs: list[Pair], watts: float, peak_mib: int, shared: Setup
) -> dict[str, object]:
    """Build one POST /api/speed-tests body for the median pair; no credentials.

    Returns:
        The payload.

    """
    index = median_index(pairs)
    chosen = pairs[index]
    full = chosen.full
    drafted = DRAFT_PROPOSALS * full.rounds
    accepted = full.committed - full.rounds
    return {
        "hfId": shared.target_repository,
        "modelRevision": shared.target_revision,
        "hardware": {
            "hwClass": "DISCRETE_GPU",
            "gpuName": shared.gpu_name,
            "gpuCount": 1,
            "vramGb": shared.vram_gib,
            "cpu": shared.cpu,
            "ramGb": shared.ram_gib,
            "os": shared.os,
            "powerWatts": shared.power_limit_watts,
        },
        "engineName": "exllamav3",
        "engineVersion": (
            f"{shared.engine_version} (r0b0tlab community {shared.engine_revision[:7]} "
            f"+ {shared.patches} elpis patches)"
        ),
        "engineRepository": ENGINE_REPOSITORY,
        "engineCommit": shared.engine_revision,
        "engineBuild": f"elpis image {shared.image_id}",
        "backend": "cuda",
        "quantization": QUANTIZATION,
        "promptTokens": full.prompt_tokens,
        "outputTokens": full.completion_tokens,
        "contextLength": CONTEXT,
        "batchSize": 1,
        "prefillTokens": 0,
        "ttftMs": round(chosen.ttft_seconds * 1000, 1),
        "tokSOut": round(chosen.tok_s_out, 1),
        "tokSPrefill": round(chosen.tok_s_prefill, 1),
        "tokSTotal": round(chosen.tok_s_total, 1),
        "peakVramGb": round(peak_mib / 1024, 2),
        "gpuPowerWatts": [round(watts, 1)],
        "promptSha256": prompt.sha256,
        "promptSample": full.message[:PROMPT_SAMPLE_LIMIT],
        "outputSha256": sha256(full.output),
        "outputSample": full.output[:OUTPUT_SAMPLE_LIMIT],
        "engineTimingsRaw": timings(pairs, index),
        "notes": notes(prompt, pairs, chosen, shared),
        "engineFlags": {
            "commandSnippet": shared.command,
            "kvCacheDtype": "exl3-q3 (3-bit K/V)",
            "specDecoding": True,
            "specMethod": "DFlash2",
            "specModel": shared.draft_repository,
            "specNumTokens": DRAFT_PROPOSALS,
            "specDraftTokens": drafted,
            "specAcceptedTokens": accepted,
            "specAcceptanceRate": round(accepted / drafted, 6),
            "specMeanAcceptedLength": round(full.committed / full.rounds, 6),
            "temperature": 0,
            "concurrency": 1,
            "extraFlags": (
                "EXL3_TREE=1 (image default): 8-row dynamic token-tree verification, "
                "anchor + 7 nodes"
            ),
        },
    }


def connect(api_key_file: Path) -> Client:
    """Open the endpoint client and check it serves only the recorded model.

    Args:
        api_key_file: File holding the endpoint API key.

    Returns:
        The checked client.

    Raises:
        ValueError: If the endpoint is unhealthy or serves another model.

    """
    client = Client(api_key_file.read_text(encoding="utf-8").strip())
    health = client.json("GET", "/health", None)
    if health != {"status": "ok"}:
        msg = "Endpoint is not healthy"
        raise ValueError(msg)
    served = sequence(client.json("GET", "/v1/models", None)["data"])
    names = [text(mapping(item)["id"]) for item in served]
    if names != [MODEL]:
        msg = f"Endpoint must serve only {MODEL}"
        raise ValueError(msg)
    return client


def prompt_results(
    prompt: Prompt,
    result: tuple[Reply, list[Pair]],
    sampler: PowerSampler,
    shared: Setup,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one prompt's record entry and summary entry.

    Args:
        prompt: The measured prompt.
        result: The prompt's warmup reply and timed pairs.
        sampler: The power sampler that ran over the measurement.
        shared: The shared payload inputs.

    Returns:
        The record entry and the summary entry.

    """
    warmup, pairs = result
    watts = pooled_watts(sampler, pairs)
    peak = max(
        [warmup.vram_mib]
        + [reply.vram_mib for pair in pairs for reply in (pair.probe, pair.full)]
    )
    body = payload(prompt, pairs, watts, peak, shared)
    index = median_index(pairs)
    measured: dict[str, object] = {
        "sha256": prompt.sha256,
        "warmup": warmup.record(),
        "pairs": [
            {
                "ttft_request": pair.probe.record(),
                "full_request": pair.full.record(),
                "ttft_seconds": pair.ttft_seconds,
                "tok_s_out": pair.tok_s_out,
                "tok_s_prefill": pair.tok_s_prefill,
                "tok_s_total": pair.tok_s_total,
                "tok_s_request": pair.tok_s_request,
            }
            for pair in pairs
        ],
        "reported_pair": index,
        "mean_watts_full_requests": watts,
        "peak_vram_mib": peak,
        "payload": body,
    }
    rates = [pair.tok_s_out for pair in pairs]
    summary: dict[str, object] = {
        "tokSOut": body["tokSOut"],
        "tokSOut_min_max": [round(min(rates), 1), round(max(rates), 1)],
        "ttftMs": body["ttftMs"],
        "tokSPrefill": body["tokSPrefill"],
        "tokSTotal": body["tokSTotal"],
        "whole_request_tok_s": round(pairs[index].tok_s_request, 1),
        "accepted_length": round(
            pairs[index].full.committed / pairs[index].full.rounds, 3
        ),
        "outputTokens": body["outputTokens"],
        "promptTokens": body["promptTokens"],
        "watts": round(watts, 1),
        "peakVramGb": body["peakVramGb"],
    }
    return measured, summary


def main() -> None:
    """Measure once and write one exclusive-create JSON record with the payloads.

    Raises:
        ValueError: If the record exists, the GPU is not in its stock declared
            state or a LocalMaxxing GPU, or its policy changed while measuring.

    """
    arguments = parse_arguments()
    if arguments.out.exists():
        msg = "Output record already exists"
        raise ValueError(msg)
    prompts, gpu_names = catalog(arguments.context)
    declared = _gpu()
    before = query(STATE_FIELDS)
    if before["power.limit"] != before["power.default_limit"]:
        msg = "The power limit differs from the card's default (not stock)"
        raise ValueError(msg)
    if not (
        declared["core_clock_offset_mhz"] == 0
        and declared["memory_clock_offset_mhz"] == 0
    ):
        msg = "Clock offsets are not stock"
        raise ValueError(msg)
    if before["name"] not in gpu_names:
        msg = f"{before['name']} is not a LocalMaxxing GPU name"
        raise ValueError(msg)
    shared = setup(arguments.image_id, before)
    client = connect(arguments.api_key_file)
    results: dict[str, tuple[Reply, list[Pair]]] = {}
    with PowerSampler(gpu=GPU, interval_seconds=POWER_INTERVAL_SECONDS) as sampler:
        for prompt in prompts:
            results[prompt.identifier] = measure(client, prompt)
    after = query(STATE_FIELDS)
    if _gpu() != declared:
        msg = "The GPU policy changed during the measurement"
        raise ValueError(msg)
    measured: dict[str, object] = {}
    summary: dict[str, object] = {}
    for prompt in prompts:
        measured[prompt.identifier], summary[prompt.identifier] = prompt_results(
            prompt, results[prompt.identifier], sampler, shared
        )
    record = {
        "schema_version": 1,
        "producer": "bench.localmaxxing",
        "protocol": {
            "max_tokens": MAX_TOKENS,
            "timed_pairs": TIMED_PAIRS,
            "nonce_line": NONCE_LINE,
            "power_interval_seconds": POWER_INTERVAL_SECONDS,
        },
        "agent_context": {
            "url": AGENT_CONTEXT_URL,
            "sha256": digest(arguments.context),
        },
        "image_id": arguments.image_id,
        "gpu_policy": declared,
        "gpu_before": before,
        "gpu_after": after,
        "power": sampler.summary(),
        "power_samples": sampler.records(),
        "prompts": measured,
        "summary": summary,
    }
    _ = canonical(record)
    save(arguments.out, record)
    _ = sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
