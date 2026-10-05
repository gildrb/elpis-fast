# Copyright (c) 2026 Gil Rodrigues
"""Finite EXL3 + Bend lane: one explicitly selected suite; see autoresearch.sh.

Suites: ``broad`` (native tasksets + C1 whole requests) and ``prefill`` (the
cold-prefill TTFT ladder in ``bench.prefill``). This supervisor never operates
Docker lifecycle, promotion, power policy or the maintenance guardian. The
latter must independently recover the owned candidate. Only the unchanged
native taskset producers, frozen requests and raw-evidence replays admit results.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import re
import select
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bench import exl3, prefill, process
from bench.exl3 import mapping, number, sequence

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType

    from bench.tokenizer import RawTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = exl3.ENDPOINT
OPERATOR = Path("/run/user/1000/elpis-autoresearch-operator.json")
LIMIT_SECONDS = 2400
OWNER_UID = 1000
PRIVATE_DIRECTORY_MODE = 0o700
# Printable ASCII without space (``!`` through ``~``) for the bearer key.
KEY_FIRST_CHAR = 33
KEY_LAST_CHAR = 126
# Path characters: no C0 controls (below space) and no DEL.
FIRST_PRINTABLE = 32
DELETE = 127
# /proc/locks row: ordinal, class, mode, access, pid, major:minor:inode, ...
LOCK_ROW_FIELDS = 6
LOCK_DEVICE_PARTS = 3
# The required ``--suite NAME`` prefix of the command line.
SUITE_ARGUMENTS = 2
RECOVERY_HEADROOM_SECONDS = 120
TERMINATION_SECONDS = 20
LEASE = Path("/run/user/1000/qwen-packed64-docker-gpu0-maintenance.lock")
LAUNCH_LOCK = Path("/mnt/ssd/storage/ai/qwen3.8-27b/qwen-inference-launch.lock")
LAUNCH_IDENTITY = {
    "dev": 66305,
    "ino": 23726380,
    "uid": 1000,
    "mode": 0o600,
    "nlink": 1,
}
TASKSET_METRICS = ("output_tok_s", "reward", "truncated")
POOLED_SCOPE = (
    "every native model call of all four tasksets pooled: "
    "sum of completion tokens / sum of model-call wall time"
)
# Every rejected observation is recorded privately; nothing is retried.
FAILURES = (
    OSError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    subprocess.SubprocessError,
)


def file_identity(info: os.stat_result) -> dict[str, int]:
    """Match the guardian's inode/owner/private-mode identity contract.

    Returns:
        The device, inode, owner, permission bits and link count.

    """
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def private_read(path: Path, *, key: bool = False) -> tuple[bytes, dict[str, int]]:
    """Read one bounded private file, refusing symlinks and identity races.

    Returns:
        The file bytes and the identity they were read under.

    Raises:
        ValueError: If the file is unsafe, oversized or changed while reading.

    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        identity = file_identity(info)
        if not (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) in ((0o400, 0o600) if key else (0o600,))
        ):
            msg = "Unsafe private file"
            raise ValueError(msg)
        raw = stream.read(4097 if key else 65537)
        if not (len(raw) <= (4096 if key else 65536)):
            msg = "Private file exceeds bound"
            raise ValueError(msg)
        if identity != file_identity(path.stat(follow_symlinks=False)):
            msg = "Private file changed while reading"
            raise ValueError(msg)
    return raw, identity


def private_directory(path: Path) -> dict[str, int]:
    """Require the existing canonical private directory used by the guardian.

    Returns:
        The directory identity in the guardian's status format.

    Raises:
        ValueError: If the directory is not canonical, owned and private.

    """
    if not (path.is_absolute() and path.resolve(strict=True) == path):
        msg = "Maintenance directory must be canonical and absolute"
        raise ValueError(msg)
    info = path.stat(follow_symlinks=False)
    if not (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == PRIVATE_DIRECTORY_MODE
    ):
        msg = "Unsafe maintenance directory"
        raise ValueError(msg)
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "uid": info.st_uid,
        "mode": PRIVATE_DIRECTORY_MODE,
    }


def process_start(pid: int) -> str:
    """Bind a live process to its Linux start-time tick, not just a reused PID.

    Returns:
        The process start time in clock ticks, as text.

    Raises:
        ValueError: If the process is a zombie or dead.

    """
    fields = (
        (Path("/proc") / str(pid) / "stat")
        .read_text(encoding="utf-8")
        .rsplit(")", 1)[1]
        .split()
    )
    if not (fields[0] not in {"Z", "X"}):
        msg = "Guardian process is not live"
        raise ValueError(msg)
    return fields[19]


def api_key(path: Path) -> str:
    """Load the validated private bearer key; it is never written to evidence.

    Returns:
        The key as ASCII text.

    Raises:
        ValueError: If the key is empty or not printable ASCII without spaces.

    """
    raw, _ = private_read(path, key=True)
    key = raw.removesuffix(b"\n")
    if not (bool(key) and all(KEY_FIRST_CHAR <= char <= KEY_LAST_CHAR for char in key)):
        msg = "Invalid private API key"
        raise ValueError(msg)
    return key.decode("ascii")


@dataclass(frozen=True)
class Settings:
    """Explicit operator-owned inputs; workload/endpoint are not tunable knobs."""

    container: str
    key_file: Path
    maintenance: Path
    output: Path
    operator_raw: bytes
    operator_identity: dict[str, int]
    operator_sha256: str

    @classmethod
    def descriptor(cls) -> Settings:
        """Load the one private descriptor without inferring any operator input.

        Returns:
            The validated operator settings.

        Raises:
            ValueError: If the descriptor or any input path is invalid.

        """
        if OPERATOR.resolve(strict=True) != OPERATOR:
            msg = "Operator path is not canonical"
            raise ValueError(msg)
        raw, identity = private_read(OPERATOR)
        value = mapping(exl3.loads(raw))
        if set(value) != {
            "schema_version",
            "container_id",
            "api_key_file",
            "maintenance_directory",
            "output_directory",
        }:
            msg = "Unexpected operator descriptor keys"
            raise ValueError(msg)
        if exl3.integer(value.get("schema_version")) != 1:
            msg = "Unsupported operator descriptor schema"
            raise ValueError(msg)
        container = exl3.text(value.get("container_id"))
        if not (re.fullmatch(r"[0-9a-f]{64}", container) is not None):
            msg = "Owned container must be its full immutable ID"
            raise ValueError(msg)
        values = [
            exl3.text(value.get(name))
            for name in ("api_key_file", "maintenance_directory", "output_directory")
        ]
        if not all(
            all(ord(char) >= FIRST_PRINTABLE and ord(char) != DELETE for char in value)
            for value in values
        ):
            msg = "Invalid input path"
            raise ValueError(msg)
        paths = [Path(value) for value in values]
        if not all(
            path.is_absolute() and str(path) == value
            for path, value in zip(paths, values, strict=True)
        ):
            msg = "Input paths must be canonical and absolute"
            raise ValueError(msg)
        if paths[0].resolve(strict=True) != paths[0]:
            msg = "Key path must be canonical"
            raise ValueError(msg)
        if paths[2].parent.resolve(strict=True) != paths[2].parent:
            msg = "Output parent must already exist and be canonical"
            raise ValueError(msg)
        if paths[2].resolve(strict=False) != paths[2]:
            msg = "Output path must be canonical"
            raise ValueError(msg)
        _ = api_key(paths[0])
        return cls(
            container,
            paths[0],
            paths[1],
            paths[2],
            raw,
            identity,
            hashlib.sha256(raw).hexdigest(),
        )


def _check_operator(settings: Settings) -> None:
    """Require the guardian owner and the unchanged operator descriptor.

    Args:
        settings: The operator settings loaded at startup.

    Raises:
        ValueError: If the owner or the descriptor differs.

    """
    if not (os.getuid() == os.geteuid() == OWNER_UID):
        msg = "Guardian contract requires owner uid1000"
        raise ValueError(msg)
    if OPERATOR.resolve(strict=True) != OPERATOR:
        msg = "Operator path is not canonical"
        raise ValueError(msg)
    operator_raw, operator_identity = private_read(OPERATOR)
    if not (
        operator_identity == settings.operator_identity
        and operator_raw == settings.operator_raw
    ):
        msg = "Operator descriptor changed during benchmark"
        raise ValueError(msg)


def _maintenance_state(settings: Settings) -> dict[str, object]:
    """Load the guardian status and require it armed on this candidate.

    Args:
        settings: The operator settings.

    Returns:
        The guardian's maintenance status.

    Raises:
        ValueError: If the window is not armed on this candidate and boot.

    """
    directory = private_directory(settings.maintenance)
    raw, _ = private_read(settings.maintenance / "status.json")
    state = mapping(exl3.loads(raw))
    _ = exl3.canonical(state)
    if not (
        state.get("schema") == 1
        and state.get("state") == "armed"
        and state.get("phase") == "candidate_started"
    ):
        msg = "Maintenance window is not armed on candidate"
        raise ValueError(msg)
    if state.get("directory_identity") != directory:
        msg = "Maintenance directory identity changed"
        raise ValueError(msg)
    if (
        state.get("boot_id")
        != Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    ):
        msg = "Foreign maintenance boot"
        raise ValueError(msg)
    if os.path.lexists(settings.maintenance / "control.json"):
        msg = "Recovery/promotion already queued"
        raise ValueError(msg)
    candidate = mapping(state.get("candidate"))
    if candidate.get("id") != settings.container:
        msg = "Candidate is not owned by this guardian"
        raise ValueError(msg)
    if state.get("lease_path") != str(LEASE):
        msg = "Unexpected guardian lease"
        raise ValueError(msg)
    return state


def _lease_identity(settings: Settings, state: dict[str, object]) -> dict[str, int]:
    """Require the lease, launch lock and operation mutex the guardian recorded.

    Args:
        settings: The operator settings.
        state: The guardian's maintenance status.

    Returns:
        The maintenance lease identity.

    Raises:
        ValueError: If any lock identity changed.

    """
    _, lease_identity = private_read(LEASE)
    _, launch_identity = private_read(LAUNCH_LOCK)
    _, mutex_identity = private_read(settings.maintenance / "operation.lock")
    if not (
        lease_identity == state.get("lease_identity")
        and launch_identity == state.get("launch_lock_identity") == LAUNCH_IDENTITY
        and mutex_identity == state.get("operation_mutex_identity")
    ):
        msg = "Maintenance lock identity changed"
        raise ValueError(msg)
    return lease_identity


def _holds_lease(pid: int, lease_identity: dict[str, int]) -> bool:
    """Read flock ownership from the kernel, not from an unlocked file.

    Args:
        pid: The guardian PID.
        lease_identity: The maintenance lease identity.

    Returns:
        Whether the guardian holds a write flock on the lease inode.

    """
    held = False
    for row in Path("/proc/locks").read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if (
            len(fields) < LOCK_ROW_FIELDS
            or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]
            or fields[4] != str(pid)
        ):
            continue
        device = fields[5].split(":")
        if len(device) == LOCK_DEVICE_PARTS and (
            int(device[0], 16),
            int(device[1], 16),
            int(device[2]),
        ) == (
            os.major(lease_identity["dev"]),
            os.minor(lease_identity["dev"]),
            lease_identity["ino"],
        ):
            held = True
    return held


def _check_guardian(state: dict[str, object], lease_identity: dict[str, int]) -> None:
    """Require the recorded live guardian process to hold the lease.

    Args:
        state: The guardian's maintenance status.
        lease_identity: The maintenance lease identity.

    Raises:
        ValueError: If the guardian process changed or does not hold the lease.

    """
    pid = exl3.integer(state.get("guardian_pid"))
    if not (pid > 1 and process_start(pid) == state.get("guardian_start")):
        msg = "Guardian process changed"
        raise ValueError(msg)
    if (Path("/proc") / str(pid)).stat().st_uid != os.getuid():
        msg = "Foreign guardian owner"
        raise ValueError(msg)
    if not _holds_lease(pid, lease_identity):
        msg = "Guardian does not hold the maintenance lease"
        raise ValueError(msg)


def guard(settings: Settings) -> dict[str, object]:
    """Verify existing guardian ownership without acquiring or changing its locks.

    Returns:
        The guardian's armed maintenance status.

    Raises:
        ValueError: If too little recovery headroom remains.

    """
    _check_operator(settings)
    state = _maintenance_state(settings)
    lease_identity = _lease_identity(settings, state)
    _check_guardian(state, lease_identity)
    if not (
        number(state.get("deadline_monotonic")) - time.monotonic()
        > RECOVERY_HEADROOM_SECONDS
    ):
        msg = "Insufficient recovery headroom"
        raise ValueError(msg)
    return state


def verify_container(settings: Settings, state: dict[str, object]) -> None:
    """Check only public Docker identity fields before authenticated API probes.

    Raises:
        ValueError: If the instance is not the guardian's exclusively published
            candidate.

    """
    template = (
        '{"id":{{json .Id}},"image":{{json .Image}},"name":{{json .Name}},'
        '"running":{{json .State.Running}},"ports":{{json .NetworkSettings.Ports}},'
        '"network":{{json .HostConfig.NetworkMode}}}'
    )
    value = mapping(
        json.loads(
            exl3.docker(
                "container",
                "inspect",
                "--format",
                template,
                settings.container,
            )
        )
    )
    candidate = mapping(state.get("candidate"))
    if not (
        all(value.get(key) == candidate.get(key) for key in ("id", "image", "name"))
        and value.get("running") is True
        and value.get("network") != "host"
    ):
        msg = "Owned serving instance differs from guardian candidate"
        raise ValueError(msg)
    if mapping(value.get("ports")).get("18020/tcp") != [
        {"HostIp": "127.0.0.1", "HostPort": "18020"}
    ]:
        msg = "Owned candidate must exclusively publish the fixed loopback endpoint"
        raise ValueError(msg)


def run_producer(
    settings: Settings,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> None:
    """Run once, preserving private native output; no retries or score selection.

    Raises:
        ValueError: If the native command exits nonzero.

    """
    guard(settings)
    with (
        (settings.output / "logs" / f"{name}.stdout").open("xb") as stdout,
        (settings.output / "logs" / f"{name}.stderr").open("xb") as stderr,
    ):
        result = process.run(
            [
                process.resolve(command[0], cwd=cwd, path=env.get("PATH", os.defpath)),
                *command[1:],
            ],
            check=False,
            launch=process.Launch(
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            ),
        )
    exl3.save(
        settings.output / name / "provenance/producer-exit.json",
        {"returncode": result.returncode},
    )
    if result.returncode != 0:
        msg = f"Native {name} command failed; retained artifacts are incomplete"
        raise ValueError(msg)
    guard(settings)


def _call_observation(
    value: object,
    ordinal: int,
    index: int,
    task: dict[str, object],
    trace: dict[str, object],
) -> tuple[dict[str, object], int, int, float]:
    """Validate one native model call of one episode trace.

    Args:
        value: The raw model-call record.
        ordinal: The episode's line number in the trace file.
        index: The call's position in the trace.
        task: The episode's task record.
        trace: The episode's single trace.

    Returns:
        The call observation, its completion tokens, its input tokens and its
        wall duration in seconds.

    Raises:
        ValueError: If the call failed, did not finish or has invalid usage or
            clocks.

    """
    call = mapping(value)
    # Native write_episode(exclude_none=True) omits a successful error.
    if call.get("error") is not None:
        msg = "Failed or incomplete model call; no filtering or retries"
        raise ValueError(msg)
    finish = call.get("finish_reason")
    if not (isinstance(finish, str) and finish in {"stop", "length"}):
        msg = "Incomplete native model-call finish"
        raise ValueError(msg)
    usage = mapping(call.get("usage"))
    tokens = exl3.integer(usage.get("completion_tokens"))
    if not (tokens >= 0):
        msg = "Negative model-call completion usage"
        raise ValueError(msg)
    prompt_tokens = exl3.integer(usage.get("prompt_tokens"))
    cached_value = usage.get("cached_input_tokens")
    cached_tokens = 0 if cached_value is None else exl3.integer(cached_value)
    if not (prompt_tokens >= 0 and cached_tokens >= 0):
        msg = "Negative native prompt/cache usage"
        raise ValueError(msg)
    input_tokens = prompt_tokens + cached_tokens
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None and not (0 <= exl3.integer(reasoning) <= tokens):
        msg = "Reasoning usage must be a subset, never additional tokens"
        raise ValueError(msg)
    span = mapping(call.get("time"))
    start, end = number(span.get("start")), number(span.get("end"))
    duration = number(end - start)
    if not (0 < start < end and duration > 0):
        msg = "Missing, nonfinite or reversed native model-call wall clocks"
        raise ValueError(msg)
    observation: dict[str, object] = {
        "episode": ordinal,
        "task_key": exl3.text(task.get("key")),
        "task_sha256": exl3.text(task.get("hash")),
        "trace_id": trace.get("id"),
        "call": index,
        "completion_tokens": tokens,
        "input_tokens": input_tokens,
        "uncached_prompt_tokens": prompt_tokens,
        "cached_input_tokens": cached_value,
        "reasoning_tokens_subset": reasoning,
        "start_unix_seconds": start,
        "end_unix_seconds": end,
        "duration_wall_seconds": duration,
        "finish_reason": call["finish_reason"],
    }
    return observation, tokens, input_tokens, duration


def _episode_observations(
    raw: bytes, ordinal: int
) -> list[tuple[dict[str, object], int, int, float]]:
    """Validate one native episode line and every model call of its trace.

    Args:
        raw: One JSON line of the native trace file.
        ordinal: The line number of the episode.

    Returns:
        Each call's observation, completion tokens, input tokens and duration.

    Raises:
        ValueError: If the episode or its single trace failed or has no calls.

    """
    episode_value: object = json.loads(raw, object_pairs_hook=exl3.pairs)
    episode = mapping(episode_value)
    task = mapping(episode.get("task"))
    if not (episode.get("ok") is True and not sequence(episode.get("errors"))):
        msg = "Model-call metric cannot exclude an operationally failed episode"
        raise ValueError(msg)
    traces = sequence(episode.get("traces"))
    if len(traces) != 1:
        msg = "Expected the native single-agent trace"
        raise ValueError(msg)
    trace = mapping(traces[0])
    if not (trace.get("ok") is True and not sequence(trace.get("errors"))):
        msg = "Model-call metric cannot exclude a failed native trace"
        raise ValueError(msg)
    calls = sequence(trace.get("calls"))
    if not bool(calls):
        msg = "Missing native model-call observations"
        raise ValueError(msg)
    return [
        _call_observation(value, ordinal, index, task, trace)
        for index, value in enumerate(calls)
    ]


def model_call_observations(
    evidence: exl3.Evidence,
    directory: Path,
    expected_episode_count: int,
) -> dict[str, object]:
    """Pool every native model call, including graded incorrect/truncated answers.

    Verifiers ModelCall.time is Unix wall time from request send through fully
    received response. Usage.completion_tokens already includes reasoning tokens.
    This is whole-model-call throughput, not SSE committed-decode/GPU timing.

    Returns:
        The pooled throughput, totals, every call observation and their scope.

    Raises:
        ValueError: If the trace file, episode count or durations are invalid.

    """
    if not (type(expected_episode_count) is int and expected_episode_count > 0):
        msg = "Expected a positive exact native episode count"
        raise ValueError(msg)
    paths = list(directory.glob("*/traces.jsonl"))
    if len(paths) != 1:
        msg = "Missing unique native trace file"
        raise ValueError(msg)
    path = evidence.retain(paths[0])
    observations: list[dict[str, object]] = []
    durations: list[float] = []
    total_tokens = 0
    total_input_tokens = 0
    episodes = 0
    with path.open("rb") as stream:
        for ordinal, raw in enumerate(stream):
            for observation, tokens, input_tokens, duration in _episode_observations(
                raw, ordinal
            ):
                total_tokens += tokens
                total_input_tokens += input_tokens
                durations.append(duration)
                observations.append(observation)
            episodes += 1
    if episodes != expected_episode_count:
        msg = f"Model-call metric requires all {expected_episode_count} native episodes"
        raise ValueError(msg)
    seconds = number(math.fsum(durations))
    if not (seconds > 0):
        msg = "No positive native model-call duration"
        raise ValueError(msg)
    return {
        "model_call_output_tok_s": number(total_tokens / seconds),
        "completion_tokens": total_tokens,
        "input_tokens": total_input_tokens,
        "model_call_wall_seconds": seconds,
        "call_count": len(observations),
        "episode_count": episodes,
        "length_truncated_calls": sum(
            call["finish_reason"] == "length" for call in observations
        ),
        "calls": observations,
        "scope": (
            "whole native model-call wall time including prefill/decode/HTTP; "
            "not decode-only, monotonic or GPU time"
        ),
        "usage_semantics": (
            "completion_tokens includes reasoning; "
            "optional reasoning_tokens is not added again"
        ),
        "input_usage_semantics": (
            "native prompt_tokens excludes cache reads; input_tokens adds reported "
            "cached_input_tokens back, without claiming omitted cache telemetry "
            "is zero"
        ),
        "clock": (
            "native Unix wall seconds, unmodified; "
            "positive finite end minus start per call"
        ),
    }


Window = tuple[int, int]
Metrics = dict[str, float]


@dataclass(frozen=True)
class Suite:
    """One frozen workload: protocol, metrics, sources and its three worker phases.

    ``freeze`` binds suite inputs before any generation and returns the suite's
    workload fields; ``collect`` sends the frozen workload once; ``admit``
    replays raw evidence into metrics plus the suite's admitted.json fields.
    """

    name: str
    protocol: str
    scope: str
    order: tuple[str, ...]
    primary_metric: str
    primary_scope: str
    metric_names: tuple[str, ...]
    optional_metrics: tuple[str, ...]
    sources: tuple[str, ...]
    record: dict[str, object]
    trees: tuple[str, ...]
    freeze: Callable[[Settings, exl3.Client, RawTokenizer], dict[str, object]]
    collect: Callable[[Settings, exl3.Client], None]
    admit: Callable[
        [Settings, exl3.Evidence, dict[str, object], Window],
        tuple[Metrics, dict[str, object]],
    ]


def _broad_freeze(
    settings: Settings, client: exl3.Client, tokenizer: RawTokenizer
) -> dict[str, object]:
    """Bind taskset inputs and C1 prompts/IDs before any generation.

    Returns:
        The frozen taskset inputs and the C1 plan binding.

    """
    tasksets: dict[str, object] = {}
    for taskset in exl3.TASKSETS:
        inputs = exl3.freeze_taskset(settings.output / taskset.name, taskset)
        tasksets[taskset.name] = {
            "profile": taskset.config,
            "tasks": taskset.tasks,
            "output_budget": taskset.output_tokens,
            "files_sha256": inputs["files_sha256"],
            "local_dataset": inputs["local_dataset"],
            "selection": inputs["selection"],
            "call_sampling": exl3.expected_call_sampling(taskset),
        }
    plan = exl3.plan_c1(settings.output / "c1", client, tokenizer)
    return {
        "tasksets": tasksets,
        "c1": {
            "plan_sha256": exl3.digest(settings.output / "c1/plan.json"),
            "rows": plan["rows"],
        },
    }


def _broad_collect(settings: Settings, client: exl3.Client) -> None:
    """Run every native taskset once, then the frozen C1 rows."""
    for taskset in exl3.TASKSETS:
        group = settings.output / taskset.name
        command, cwd = exl3.native_command(group, taskset)
        run_producer(
            settings,
            taskset.name,
            command,
            cwd,
            exl3.native_environment(group, client.key),
        )
        exl3.save(
            group / f"provenance/{taskset.name}.evaluation-inputs-after.json",
            exl3.taskset_inputs(group / "provenance", taskset),
        )
    exl3.run_c1(
        settings.output / "c1",
        client,
        exl3.document(settings.output / "c1/plan.json"),
        lambda: guard(settings),
    )


def _broad_admit(
    settings: Settings,
    evidence: exl3.Evidence,
    frozen_workload: dict[str, object],
    window: Window,
) -> tuple[Metrics, dict[str, object]]:
    """Replay every taskset, the pooled primary and C1 from raw evidence.

    Returns:
        The admitted metrics and the suite's admitted.json fields.

    Raises:
        ValueError: If a taskset lost rollouts, its inputs or the C1 plan changed,
            or no positive pooled duration remains.

    """
    frozen_tasksets = mapping(frozen_workload["tasksets"])
    qualities: dict[str, object] = {}
    native_calls: dict[str, object] = {}
    metrics: Metrics = {}
    pooled_tokens = 0
    pooled_durations: list[float] = []
    for taskset in exl3.TASKSETS:
        group = settings.output / taskset.name
        quality = exl3.admit_taskset(evidence, group, taskset, window)
        calls = model_call_observations(evidence, group / taskset.name, taskset.tasks)
        if not (
            calls["call_count"] == taskset.tasks
            and quality["rollouts"] == taskset.tasks
            and calls["length_truncated_calls"] == quality["truncated_rollouts"]
        ):
            msg = (
                f"{taskset.name} must retain all {taskset.tasks} "
                "graded one-call rollouts"
            )
            raise ValueError(msg)
        if (
            mapping(frozen_tasksets[taskset.name])["files_sha256"]
            != exl3.document(
                group / f"provenance/{taskset.name}.evaluation-inputs-before.json"
            )["files_sha256"]
        ):
            msg = f"{taskset.name} inputs differ from the pre-suite frozen workload"
            raise ValueError(msg)
        metrics[f"{taskset.metric}_output_tok_s"] = number(
            calls["model_call_output_tok_s"]
        )
        metrics[f"{taskset.metric}_reward"] = number(quality["weighted_reward_mean"])
        metrics[f"{taskset.metric}_truncated"] = float(
            exl3.integer(calls["length_truncated_calls"])
        )
        pooled_tokens += exl3.integer(calls["completion_tokens"])
        pooled_durations.extend(
            number(mapping(call)["duration_wall_seconds"])
            for call in sequence(calls["calls"])
        )
        qualities[taskset.name] = quality
        native_calls[taskset.name] = calls
    pooled_seconds = number(math.fsum(pooled_durations))
    if not (pooled_seconds > 0):
        msg = "No positive pooled native model-call duration"
        raise ValueError(msg)
    metrics["model_call_output_tok_s"] = number(pooled_tokens / pooled_seconds)
    native_calls["pooled"] = {
        "model_call_output_tok_s": metrics["model_call_output_tok_s"],
        "completion_tokens": pooled_tokens,
        "model_call_wall_seconds": pooled_seconds,
        "call_count": len(pooled_durations),
        "tasksets": [taskset.name for taskset in exl3.TASKSETS],
        "scope": POOLED_SCOPE,
    }
    c1 = exl3.admit_c1(evidence, settings.output / "c1")
    if mapping(frozen_workload["c1"])["plan_sha256"] != exl3.digest(
        settings.output / "c1/plan.json"
    ):
        msg = "C1 plan differs from the pre-suite frozen workload"
        raise ValueError(msg)
    guard(settings)
    metrics.update({
        name: number(value) for name, value in mapping(c1["metrics"]).items()
    })
    return metrics, {
        "quality_scope": (
            "sampled native tasksets (3 AIME25, 20 MMLU-Pro, 6 I3 Logic, "
            "3 LiveCodeBench); native rewards per taskset, no combined quality score"
        ),
        "tasksets": qualities,
        "native_model_calls": native_calls,
        "c1": c1,
        "not_measured": {
            "ttft": c1["ttft"],
            "committed_decode_tps": c1["committed_decode_tps"],
            "power_energy": "not sampled by this lane",
            "context_capacity": (
                "/v1/models max_model_len is the reported limit, "
                "not a 262144-token capacity test"
            ),
        },
    }


def _prefill_freeze(
    settings: Settings, client: exl3.Client, tokenizer: RawTokenizer
) -> dict[str, object]:
    """Size, freeze and render every ladder row before any generation.

    Returns:
        The prefill plan binding.

    """
    plan = prefill.plan(settings.output / "prefill", client, tokenizer, prefill.LADDER)
    return {
        "prefill": {
            "plan_sha256": exl3.digest(settings.output / "prefill/plan.json"),
            "rows": plan["rows"],
        },
    }


def _prefill_collect(settings: Settings, client: exl3.Client) -> None:
    """Send each frozen row's TTFT then continuation request once, in order."""
    prefill.run(
        settings.output / "prefill",
        client,
        exl3.document(settings.output / "prefill/plan.json"),
        lambda: guard(settings),
    )


def _prefill_admit(
    settings: Settings,
    evidence: exl3.Evidence,
    frozen_workload: dict[str, object],
    window: Window,
) -> tuple[Metrics, dict[str, object]]:
    """Replay every ladder row from raw evidence inside the identity window.

    Returns:
        The admitted metrics and the suite's admitted.json fields.

    Raises:
        ValueError: If the plan changed or a request falls outside the window.

    """
    result = prefill.admit(evidence, settings.output / "prefill", prefill.LADDER)
    if mapping(frozen_workload["prefill"])["plan_sha256"] != exl3.digest(
        settings.output / "prefill/plan.json"
    ):
        msg = "Prefill plan differs from the pre-suite frozen workload"
        raise ValueError(msg)
    if not all(
        window[0]
        < exl3.integer(mapping(mapping(row)[kind])["request_started_unix_ns"])
        < window[1]
        for row in sequence(result["rows"])
        for kind in prefill.BUDGETS
    ):
        msg = "Prefill requests fall outside the identity capture window"
        raise ValueError(msg)
    guard(settings)
    metrics = {
        name: number(value) for name, value in mapping(result["metrics"]).items()
    }
    return metrics, {
        "prefill": result,
        "not_measured": {
            "streaming_ttft": prefill.TTFT_SCOPE,
            "committed_decode_tps": (
                "unavailable: EXL3 exposes no incremental committed counters"
            ),
            "prefix_reuse": (
                "the continuation is expected to reuse the TTFT prefix; usage "
                "carries no cache telemetry, so reuse is not observed"
            ),
            "quality": "this suite measures no task quality or reward",
            "power_energy": "not sampled by this lane",
        },
    }


BROAD = Suite(
    name="broad",
    protocol="exl3-native-broad-c1-request-v5",
    scope="exl3_bend_sampled_broad_tasksets_and_c1_whole_request_not_full_qualification",
    order=(*(taskset.name for taskset in exl3.TASKSETS), "c1"),
    primary_metric="model_call_output_tok_s",
    primary_scope=POOLED_SCOPE,
    metric_names=(
        "model_call_output_tok_s",
        *(
            f"{taskset.metric}_{name}"
            for taskset in exl3.TASKSETS
            for name in TASKSET_METRICS
        ),
        *(f"c1_request_tok_s_{depth}" for depth in exl3.DEPTHS),
    ),
    optional_metrics=("spec_accept_length",),
    sources=(
        "autoresearch.sh",
        "bench/autoresearch.py",
        "bench/autoresearch_worker.py",
        "bench/exl3.py",
        "bench/process.py",
        "bench/throughput-prompts.jsonl",
        "prepare/exl3-manifest.json",
    ),
    record={
        "tasksets": [
            {
                "taskset": taskset.name,
                "profile": taskset.config,
                "tasks": taskset.tasks,
                "rollouts": 1,
                "shuffle": True,
                "seed": 0,
                "output_budget": taskset.output_tokens,
                "sampling": (
                    "eval/configs/local.toml; greedy, thinking enabled; "
                    "only max_tokens set per taskset"
                ),
            }
            for taskset in exl3.TASKSETS
        ],
        "c1": {
            "protocol": exl3.C1_PROTOCOL,
            "depths": list(exl3.DEPTHS),
            "repetitions": exl3.REPETITIONS,
            "output_budget": exl3.OUTPUT_TOKENS,
            "concurrency": 1,
            "order": "depth_then_repetition",
            "sampling": {"temperature": 0, "top_p": 1, "n": 1, "stream": False},
            "cache_policy": "deterministic_per_row_nonce_no_flush_no_warmup",
        },
    },
    trees=(
        *(
            f"{taskset.name}/{part}"
            for taskset in exl3.TASKSETS
            for part in ("provenance", taskset.name)
        ),
        "c1",
    ),
    freeze=_broad_freeze,
    collect=_broad_collect,
    admit=_broad_admit,
)
PREFILL = Suite(
    name="prefill",
    protocol="exl3-native-prefill-ttft-v1",
    scope="exl3_cold_prefill_ttft_ladder_nonstreaming_not_capacity_or_quality",
    order=("prefill",),
    primary_metric="prefill_tok_s",
    primary_scope=prefill.PRIMARY_SCOPE,
    metric_names=prefill.metric_names(prefill.LADDER),
    optional_metrics=(),
    sources=(
        "autoresearch.sh",
        "bench/autoresearch.py",
        "bench/autoresearch_worker.py",
        "bench/exl3.py",
        "bench/prefill.py",
        "bench/process.py",
        "bench/throughput-prompts.jsonl",
        "prepare/exl3-manifest.json",
    ),
    record={
        "prefill": {
            "protocol": prefill.PROTOCOL,
            "ladder": [[depth, reps] for depth, reps in prefill.LADDER],
            "output_budgets": dict(prefill.BUDGETS),
            "concurrency": 1,
            "order": prefill.ORDER,
            "sampling": dict(prefill.SAMPLING),
            "cache_policy": prefill.CACHE_POLICY,
        },
    },
    trees=("prefill",),
    freeze=_prefill_freeze,
    collect=_prefill_collect,
    admit=_prefill_admit,
)
SUITES = {suite.name: suite for suite in (BROAD, PREFILL)}


@dataclass
class Interruption:
    """Let the supervisor terminate only its own process family on interruption."""

    signum: int | None = None

    def request(self, signum: int, _frame: FrameType | None) -> None:
        """Defer shutdown to the supervised polling loop."""
        if self.signum is None:
            self.signum = signum


def _task_children(pid: int, task: Path) -> list[int]:
    """Read the children forked by one thread of a process.

    Args:
        pid: The process whose thread is read.
        task: The thread's /proc task directory.

    Returns:
        The child PIDs; none when the thread is gone.

    Raises:
        FileNotFoundError: If this process's own main thread is unreadable.
        ProcessLookupError: If this process's own main thread is unreadable.

    """
    try:
        listing = (task / "children").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        if pid == os.getpid() and task.name == str(pid):
            raise
        return []
    return [int(child) for child in listing.split()]


def _process_children(pid: int) -> list[int]:
    """Include children forked by any thread, regardless of their sessions.

    Returns:
        The child PIDs of every thread; none when the process is gone.

    Raises:
        FileNotFoundError: If this process's own threads are unreadable.
        ProcessLookupError: If this process's own threads are unreadable.

    """
    children: list[int] = []
    try:
        for task in (Path("/proc") / str(pid) / "task").iterdir():
            children.extend(_task_children(pid, task))
    except (FileNotFoundError, ProcessLookupError):
        if pid == os.getpid():
            raise
    return children


def _enable_subreaper() -> None:
    """Keep orphaned worker descendants under this dedicated Linux supervisor.

    Raises:
        OSError: If prctl cannot set or read the child subreaper flag.
        ValueError: If pidfd or prctl support is missing, the supervisor already
            has children, or the subreaper flag did not take effect.

    """
    if not (
        sys.platform == "linux"
        and hasattr(os, "pidfd_open")
        and hasattr(os, "P_PIDFD")
        and hasattr(signal, "pidfd_send_signal")
    ):
        msg = "Linux pidfd ownership is required before launching the worker"
        raise ValueError(msg)
    if _process_children(os.getpid()):
        msg = "Dedicated supervisor already owns children"
        raise ValueError(msg)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "prctl"):
        msg = "Linux subreaper setup is unavailable"
        raise ValueError(msg)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    # PR_SET_CHILD_SUBREAPER and PR_GET_CHILD_SUBREAPER from linux/prctl.h.
    if prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    enabled = ctypes.c_int()
    if prctl(37, ctypes.addressof(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    if enabled.value != 1:
        msg = "Linux child subreaper was not enabled"
        raise ValueError(msg)


def _pidfd_pid(descriptor: int) -> int:
    """Read the kernel identity of a pidfd; reaped processes report minus one.

    Returns:
        The PID the descriptor refers to, or -1 once it was reaped.

    """
    identity = dict(
        row.split(":", 1)
        for row in (Path("/proc/self/fdinfo") / str(descriptor))
        .read_text(encoding="ascii")
        .splitlines()
    )
    return int(identity.get("Pid", ""))


def _verified_children(
    current: int,
    parent: int,
    supervisor: int,
    descriptor: int,
    descriptors: dict[int, int],
) -> list[int] | None:
    """Check a pidfd-pinned process is still the expected family member.

    Args:
        current: The candidate descendant PID.
        parent: The pinned parent it was discovered under.
        supervisor: This supervisor's PID.
        descriptor: The candidate's pidfd.
        descriptors: The already pinned descendants.

    Returns:
        The candidate's children, or None when it is no longer that member.

    Raises:
        ValueError: If the pidfd names another process or the owner changed.

    """
    process = Path("/proc") / str(current)
    fields = (process / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
    owner = process.stat().st_uid
    if int(fields[1]) != parent:
        return None
    children = _process_children(current)
    # A pidfd survives reaping/PID reuse; reject /proc data from a
    # replacement process before trusting its identity or children.
    pinned_pid = _pidfd_pid(descriptor)
    if pinned_pid == -1:
        return None
    if pinned_pid != current:
        msg = "Unexpected pidfd process identity"
        raise ValueError(msg)
    if parent != supervisor and _pidfd_pid(descriptors[parent]) != parent:
        return None
    if owner != os.getuid():
        msg = "Worker descendant changed its owner uid"
        raise ValueError(msg)
    return children


def _pin(
    current: int,
    parent: int,
    supervisor: int,
    descriptors: dict[int, int],
    pending: list[tuple[int, int]],
) -> None:
    """Pin one verified descendant and queue its children; close any rejected pidfd.

    Args:
        current: The candidate descendant PID.
        parent: The pinned parent it was discovered under.
        supervisor: This supervisor's PID.
        descriptors: The pinned descendants, extended in place.
        pending: The traversal queue, extended in place.

    """
    descriptor: int | None = os.pidfd_open(current)
    try:
        children = _verified_children(
            current, parent, supervisor, descriptor, descriptors
        )
        if children is not None:
            descriptors[current] = descriptor
            descriptor = None
            pending.extend((child, current) for child in children)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _pin_descendants(
    supervisor: int,
    pending: list[tuple[int, int]],
    descriptors: dict[int, int],
    deadline: float,
) -> None:
    """Walk the pending family tree until it is exhausted or the deadline passes.

    Args:
        supervisor: This supervisor's PID.
        pending: The traversal queue of (PID, parent PID).
        descriptors: The pinned descendants, extended in place.
        deadline: The monotonic traversal deadline.

    """
    while pending and time.monotonic() < deadline:
        current, parent = pending.pop()
        if current in descriptors:
            continue
        try:
            _pin(current, parent, supervisor, descriptors, pending)
        except (FileNotFoundError, ProcessLookupError):
            continue


def owned_processes(deadline: float) -> dict[int, int]:
    """Pin verified descendants of this supervisor, including adopted orphans.

    Returns:
        Each pinned descendant PID mapped to its pidfd.

    """
    supervisor = os.getpid()
    pending = [(child, supervisor) for child in _process_children(supervisor)]
    descriptors: dict[int, int] = {}
    try:
        _pin_descendants(supervisor, pending, descriptors, deadline)
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise
    return descriptors


def _adopt(
    descriptors: dict[int, int], poller: select.poll, deadline: float
) -> set[int]:
    """Register newly pinned family members; close duplicate pidfds.

    Args:
        descriptors: The watched family, extended in place.
        poller: The exit poller, extended in place.
        deadline: The monotonic cleanup deadline.

    Returns:
        The pidfds added in this round.

    """
    added: set[int] = set()
    for pid, descriptor in owned_processes(deadline).items():
        if pid in descriptors:
            os.close(descriptor)
        else:
            descriptors[pid] = descriptor
            poller.register(descriptor, select.POLLIN)
            added.add(descriptor)
    return added


def _reap(child: subprocess.Popen[bytes], pid: int, descriptor: int) -> bool:
    """Reap one family member whose pidfd reported exit.

    Args:
        child: The worker root process.
        pid: The exited member's PID.
        descriptor: The exited member's pidfd.

    Returns:
        Whether the member is finished.

    """
    if pid == child.pid and child.returncode is None:
        return child.poll() is not None
    # A ChildProcessError means its parent still owns reaping, or already reaped
    # it. Rediscovery will pin it again if it becomes adopted.
    with contextlib.suppress(ChildProcessError):
        os.waitid(os.P_PIDFD, descriptor, os.WEXITED | os.WNOHANG)
    return True


def _signal_family(
    child: subprocess.Popen[bytes],
    descriptors: dict[int, int],
    added: set[int],
    term_deadline: float,
    exited: dict[int, int],
) -> tuple[list[int], bool]:
    """Reap exited members and signal live ones.

    New members get SIGTERM; every live member gets SIGKILL after the term
    deadline.

    Args:
        child: The worker root process.
        descriptors: The watched family.
        added: The pidfds added in this round.
        term_deadline: The monotonic SIGKILL deadline.
        exited: The poll events of exited pidfds.

    Returns:
        The finished PIDs and whether any live member was seen.

    """
    finished: list[int] = []
    saw_live = False
    for pid, descriptor in descriptors.items():
        if descriptor in exited:
            if _reap(child, pid, descriptor):
                finished.append(pid)
            continue
        saw_live = True
        try:
            if descriptor in added:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            if time.monotonic() >= term_deadline:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        except ProcessLookupError:
            continue
    return finished, saw_live


def terminate(child: subprocess.Popen[bytes], deadline: float) -> bool:
    """Reap the owned family within the reserve; report any live processes seen.

    Returns:
        Whether any live family member was seen.

    Raises:
        ValueError: If the family is not reaped within its reserve.

    """
    cleanup_deadline = min(deadline, time.monotonic() + TERMINATION_SECONDS)
    term_deadline = cleanup_deadline - 1
    descriptors: dict[int, int] = {}
    poller = select.poll()
    saw_live = False
    try:
        while True:
            added = _adopt(descriptors, poller, cleanup_deadline)
            # Only Popen may reap its root and update its cached return code.
            child.poll()
            exited = dict(poller.poll(0))
            finished, live = _signal_family(
                child, descriptors, added, term_deadline, exited
            )
            saw_live = saw_live or live
            for pid in finished:
                descriptor = descriptors.pop(pid)
                poller.unregister(descriptor)
                os.close(descriptor)
            # A traversal can race adoption. An empty direct-child list is the
            # final proof that no unvisited descendant can still fork or reparent.
            if (
                not descriptors
                and child.returncode is not None
                and not _process_children(os.getpid())
            ):
                return saw_live
            if not (time.monotonic() < cleanup_deadline):
                msg = (
                    "Worker family could not be terminated and reaped within its "
                    "reserve"
                )
                raise ValueError(msg)
            time.sleep(min(0.05, max(0.0, cleanup_deadline - time.monotonic())))
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def supervise(settings: Settings, suite: Suite, started: float) -> int:
    """Enforce one deadline across capture, every frozen workload and admission.

    Returns:
        Zero once the admitted metrics are printed.

    Raises:
        ValueError: If the output exists or no guarded time remains.

    """
    if os.path.lexists(settings.output):
        msg = "Output directory must be fresh"
        raise ValueError(msg)
    state = guard(settings)
    deadline = min(
        started + LIMIT_SECONDS,
        number(state.get("deadline_monotonic")) - RECOVERY_HEADROOM_SECONDS,
    )
    if not (deadline - time.monotonic() > TERMINATION_SECONDS):
        msg = "No guarded execution time remains"
        raise ValueError(msg)
    settings.output.mkdir(mode=0o700)
    try:
        return supervise_created(settings, suite, started, state, deadline)
    except FAILURES as error:
        exl3.save(
            settings.output / "failure.json",
            {
                "status": "rejected",
                "error_type": type(error).__name__,
                "reason": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def _bind_worker(child: subprocess.Popen[bytes], settings: Settings) -> None:
    """Send the operator descriptor binding to the worker and close its stdin.

    Args:
        child: The worker root process.
        settings: The operator settings.

    Raises:
        RuntimeError: If the worker has no stdin pipe.

    """
    if child.stdin is None:
        msg = "Missing supervisor binding pipe"
        raise RuntimeError(msg)
    with child.stdin:
        child.stdin.write(
            exl3.canonical({
                "identity": settings.operator_identity,
                "sha256": settings.operator_sha256,
            })
        )


def _watch_worker(
    child: subprocess.Popen[bytes],
    settings: Settings,
    state: dict[str, object],
    deadline: float,
    interruption: Interruption,
) -> None:
    """Poll the worker until it exits, rechecking deadline and guardian ownership.

    Args:
        child: The worker root process.
        settings: The operator settings.
        state: The guardian status at supervisor start.
        deadline: The monotonic supervisor deadline.
        interruption: The deferred signal record.

    Raises:
        ValueError: If interrupted, out of time or the guardian ownership changed.

    """
    while child.poll() is None:
        if interruption.signum is not None:
            msg = "Canonical benchmark interrupted"
            raise ValueError(msg)
        if not (time.monotonic() < deadline - TERMINATION_SECONDS):
            msg = (
                "Canonical benchmark exceeded its guarded deadline; no partial metrics"
            )
            raise ValueError(msg)
        current = guard(settings)
        if not all(
            current.get(key) == state.get(key)
            for key in (
                "candidate",
                "guardian_pid",
                "guardian_start",
                "deadline_monotonic",
                "lease_identity",
            )
        ):
            msg = "Maintenance ownership changed during benchmark"
            raise ValueError(msg)
        time.sleep(
            min(0.5, max(0.0, deadline - TERMINATION_SECONDS - time.monotonic()))
        )


def _record_measurement(
    settings: Settings, suite: Suite, started: float
) -> dict[str, float]:
    """Bind the admitted metrics to the frozen workload and save measurement.json.

    Args:
        settings: The operator settings.
        suite: The selected suite.
        started: The monotonic supervisor start time.

    Returns:
        The ordered admitted metrics plus the elapsed seconds.

    Raises:
        ValueError: If the admission is incomplete, unbound or has unexpected
            metrics.

    """
    admitted = exl3.document(settings.output / "admitted.json")
    if not (
        admitted.get("status") == "complete_admitted_measurement"
        and admitted.get("schema_version") == 1
        and admitted.get("protocol") == suite.protocol
        and admitted.get("scope") == suite.scope
        and admitted.get("order") == list(suite.order)
        and admitted.get("primary_metric") == suite.primary_metric
    ):
        msg = "Missing complete frozen EXL3 admission"
        raise ValueError(msg)
    benchmark = exl3.document(settings.output / "benchmark.json")
    if not (
        admitted.get("benchmark_sha256")
        == exl3.digest(settings.output / "benchmark.json")
        and admitted.get("workload_sha256")
        == benchmark.get("workload_sha256")
        == hashlib.sha256(exl3.canonical(benchmark.get("workload"))).hexdigest()
    ):
        msg = "Admitted measurement is not bound to this complete frozen workload"
        raise ValueError(msg)
    values = mapping(admitted.get("metrics"))
    names = set(suite.metric_names)
    if not (names <= set(values) <= names | set(suite.optional_metrics)):
        msg = "Missing or unexpected admitted metrics"
        raise ValueError(msg)
    ordered = [
        *suite.metric_names,
        *(name for name in suite.optional_metrics if name in values),
    ]
    metrics = {name: number(values[name]) for name in ordered}
    metrics["elapsed_seconds"] = time.monotonic() - started
    exl3.save(
        settings.output / "measurement.json",
        {
            "schema_version": 1,
            "status": "complete_admitted_measurement",
            "protocol": suite.protocol,
            "workload_sha256": admitted["workload_sha256"],
            "metrics": metrics,
            "admitted_sha256": exl3.digest(settings.output / "admitted.json"),
            "elapsed_scope": (
                "entire canonical command through raw-evidence admission; "
                "not decode-only time"
            ),
            "finished_monotonic": time.monotonic(),
        },
    )
    return metrics


def supervise_created(
    settings: Settings,
    suite: Suite,
    started: float,
    state: dict[str, object],
    deadline: float,
) -> int:
    """Use only the fresh output directory exclusively created by this invocation.

    Returns:
        Zero once the admitted metrics are printed.

    Raises:
        ValueError: If the worker failed, left descendants or finished outside the
            deadline, or finalization was late or interrupted.

    """
    _enable_subreaper()
    (settings.output / "logs").mkdir(mode=0o700)
    sources = exl3.snapshot_sources(settings.output, suite.sources)
    exl3.save(
        settings.output / "supervisor.json",
        {
            "schema_version": 1,
            "protocol": suite.protocol,
            "scope": suite.scope,
            "started_monotonic": started,
            "deadline_monotonic": deadline,
            "hard_limit_seconds": LIMIT_SECONDS,
            "recovery_headroom_seconds": RECOVERY_HEADROOM_SECONDS,
            "termination_reserve_seconds": TERMINATION_SECONDS,
            "owned_container_id": settings.container,
            "endpoint": ENDPOINT,
            "maintenance_directory": str(settings.maintenance),
            "operator_descriptor": {
                "path": str(OPERATOR),
                "identity": settings.operator_identity,
                "sha256": settings.operator_sha256,
            },
            "guardian_before": state,
            **suite.record,
            "order": list(suite.order),
            "sources": sources,
        },
    )
    interruption = Interruption()
    previous = {
        sig: signal.signal(sig, interruption.request)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with (
            (settings.output / "logs/worker.stdout").open("xb") as stdout,
            (settings.output / "logs/worker.stderr").open("xb") as stderr,
        ):
            child = process.start(
                [
                    process.resolve("nix"),
                    "develop",
                    "--offline",
                    "--no-write-lock-file",
                    "-c",
                    sys.executable,
                    "-m",
                    "bench.autoresearch_worker",
                    "--suite",
                    suite.name,
                ],
                process.Launch(
                    cwd=ROOT,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                ),
            )
            try:
                _bind_worker(child, settings)
                _watch_worker(child, settings, state, deadline, interruption)
            except BaseException:
                terminate(child, deadline)
                raise
            descendants_survived = terminate(child, deadline)
            if child.returncode != 0:
                msg = "Canonical worker failed; inspect private evidence"
                raise ValueError(msg)
            if descendants_survived:
                msg = "Canonical worker left live descendants; no admitted metrics"
                raise ValueError(msg)
        if not (interruption.signum is None and time.monotonic() < deadline):
            msg = "Canonical benchmark finished outside its guarded deadline"
            raise ValueError(msg)
        guard(settings)
        metrics = _record_measurement(settings, suite, started)
        if not (time.monotonic() < deadline and interruption.signum is None):
            msg = "Finalization exceeded deadline or was interrupted"
            raise ValueError(msg)
        sys.stdout.write(
            "".join(f"METRIC {name}={value:.17g}\n" for name, value in metrics.items())
        )
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def arguments(values: list[str]) -> Suite:
    """Parse the required suite choice (no default).

    Returns:
        The selected suite.

    Raises:
        ValueError: If the suite is missing or unknown, or arguments follow it.

    """
    if not bool(values):
        msg = "A suite is required: --suite broad|prefill"
        raise ValueError(msg)
    if not (values[0] == "--suite" and len(values) >= SUITE_ARGUMENTS):
        msg = "Expected --suite broad|prefill first"
        raise ValueError(msg)
    suite = SUITES.get(values[1])
    if suite is None:
        msg = f"Unknown suite {values[1]!r}; expected broad or prefill"
        raise ValueError(msg)
    if values[2:]:
        msg = "Unexpected arguments after the suite"
        raise ValueError(msg)
    return suite


def main() -> int:
    """Run the finite supervisor; preserve sanitized failure diagnostics privately.

    Returns:
        The process exit status.

    """
    started = time.monotonic()
    os.umask(0o077)
    try:
        suite = arguments(sys.argv[1:])
    except ValueError as error:
        sys.stderr.write(f"{error}; use bash autoresearch.sh --help\n")
        return 2
    try:
        return supervise(Settings.descriptor(), suite, started)
    except FAILURES:
        sys.stderr.write(
            "EXL3 autoresearch rejected; no admitted metrics. "
            "Inspect private artifacts.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
