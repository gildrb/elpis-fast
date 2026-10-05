# Copyright (c) 2026 inference contributors.
"""Opt-in measurement around the unchanged native evaluator; never a scorer.

Run via eval/scripts/run --measure-power. The public summary contains only
numeric observations, counts and fixed labels. Native traces and detailed
measurement records stay in the private run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from bench.power import NANOSECONDS, PowerSampler, clock_anchor, integrate_power

if TYPE_CHECKING:
    from types import FrameType

CLOCK_TOLERANCE_NS = 50_000_000
PHASES = ("boot", "setup", "agent", "finalize", "scoring")
VERIFIERS_REVISION = "ef47b2e96284a00bdcfc1012b9624b0c41ee6a0e"
TRUNCATING_STOP_CONDITIONS = frozenset({
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
})
COST_SCOPE = "whole_native_cli_including_startup_scoring_failures_retries_teardown"
SUCCESS_DEFINITION = "single_trace_weighted_reward_exactly_1_and_episode_and_trace_ok"
ATTRIBUTION_SCOPE = (
    "persisted_trace_envelopes_only_not_discarded_retries_or_full_episodes"
)
TERMINATION_GRACE_SECONDS = 10
POLL_SECONDS = 0.2
REAP_TIMEOUT_SECONDS = 5
MISSING_EXECUTABLE_EXIT = 127
SIGNAL_EXIT_BASE = 128


class InvalidJSONTypeError(TypeError, ValueError):
    """A JSON value has the wrong type; still a ValueError for existing callers."""


def mapping(value: object) -> dict[str, object]:
    """Validate an object without trusting arbitrary JSON types.

    Returns:
        A copy of the object with string keys.

    Raises:
        InvalidJSONTypeError: The value is not an object with string keys.

    """
    if not isinstance(value, dict):
        msg = "Expected JSON object"
        raise InvalidJSONTypeError(msg)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            msg = "Expected string JSON keys"
            raise InvalidJSONTypeError(msg)
        result[key] = item
    return result


def sequence(value: object) -> list[object]:
    """Validate a JSON array.

    Returns:
        A copy of the array.

    Raises:
        InvalidJSONTypeError: The value is not an array.

    """
    if not isinstance(value, list):
        msg = "Expected JSON array"
        raise InvalidJSONTypeError(msg)
    return list(value)


def number(value: object) -> float:
    """Reject missing, boolean and nonfinite numeric data.

    Returns:
        The value as a finite float.

    Raises:
        InvalidJSONTypeError: The value is not a non-boolean number.
        ValueError: The value is not finite.

    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        msg = "Expected finite number"
        raise InvalidJSONTypeError(msg)
    result = float(value)
    if not math.isfinite(result):
        msg = "Expected finite number"
        raise ValueError(msg)
    return result


@dataclass
class EpisodeObservation:
    """Sanitized observations of one upstream episode, not a new score."""

    ordinal: int
    operational_ok: bool
    recorded_error_count: int
    trace_count: int
    trace_rewards: list[float | None]
    reward_components: list[list[dict[str, float | None]]]
    weighted_reward: float | None
    full_reward_success: bool | None
    truncated: bool
    start_unix_ns: int | None
    end_unix_ns: int | None
    timing_failure: str | None


@dataclass
class TraceObservation:
    """Sanitized observations of one persisted trace."""

    error_count: int
    components: list[dict[str, float | None]]
    reward: float | None
    truncated: bool
    envelope: tuple[int, int] | None


def trace_rewards(
    trace: dict[str, object],
) -> tuple[list[dict[str, float | None]], float | None]:
    """Read reward components; a missing component leaves the sum unknown.

    Returns:
        The components and their weighted sum, or None when any is missing.

    """
    parts: list[dict[str, float | None]] = []
    values: list[float] = []
    for reward in mapping(trace.get("rewards", {})).values():
        if reward is None:
            parts.append({"score": None, "weight": None, "value": None})
            continue
        record = mapping(reward)
        score = number(record.get("score"))
        weight = number(record.get("weight", 1.0))
        weighted = number(score * weight)
        parts.append({"score": score, "weight": weight, "value": weighted})
        values.append(weighted)
    total = number(math.fsum(values)) if parts and len(parts) == len(values) else None
    return parts, total


def trace_truncated(trace: dict[str, object]) -> bool:
    """Detect a stop condition or final completed call cut off by a budget.

    Returns:
        Whether the trace was truncated.

    """
    calls = [mapping(call) for call in sequence(trace.get("calls", []))]
    completed_calls = [call for call in calls if call.get("error") is None]
    stop_condition = trace.get("stop_condition")
    return (
        isinstance(stop_condition, str) and stop_condition in TRUNCATING_STOP_CONDITIONS
    ) or bool(completed_calls and completed_calls[-1].get("finish_reason") == "length")


def trace_envelope(trace: dict[str, object]) -> tuple[int, int]:
    """Read the serialized trace start and latest finished phase end.

    Returns:
        The envelope start and end in Unix nanoseconds.

    Raises:
        ValueError: A phase is unfinished or reversed, or no envelope exists.

    """
    timing = mapping(trace.get("timing"))
    start = number(timing.get("start"))
    phase_ends: list[float] = []
    for phase in PHASES:
        span = mapping(timing.get(phase, {}))
        lo = number(span.get("start", 0))
        hi = number(span.get("end", 0))
        if lo == 0 and hi == 0:
            continue
        # An unfinished phase is not a serialized episode end.
        if lo < start or hi < lo or lo <= 0:
            msg = "Incomplete or reversed trace phase"
            raise ValueError(msg)
        phase_ends.append(hi)
    if start <= 0 or not phase_ends:
        msg = "No usable trace envelope"
        raise ValueError(msg)
    return round(start * NANOSECONDS), round(max(phase_ends) * NANOSECONDS)


def observe_trace(trace: dict[str, object]) -> TraceObservation:
    """Read pinned Trace fields; timing failures only void the envelope.

    Returns:
        The trace observation.

    """
    error_count = len(sequence(trace.get("errors", [])))
    components, reward = trace_rewards(trace)
    truncated = trace_truncated(trace)
    try:
        envelope: tuple[int, int] | None = trace_envelope(trace)
    except ValueError:
        envelope = None
    return TraceObservation(error_count, components, reward, truncated, envelope)


def observe_episode(value: object, ordinal: int) -> EpisodeObservation:
    """Read pinned Episode/Trace fields; preserve fractional and missing rewards.

    Returns:
        The episode observation.

    Raises:
        InvalidJSONTypeError: The operational status is not a boolean.

    """
    episode = mapping(value)
    ok = episode.get("ok")
    if not isinstance(ok, bool):
        msg = "Missing episode operational status"
        raise InvalidJSONTypeError(msg)
    traces = [mapping(trace) for trace in sequence(episode.get("traces"))]
    error_count = len(sequence(episode.get("errors", [])))
    observed = [observe_trace(trace) for trace in traces]
    envelopes = [trace.envelope for trace in observed if trace.envelope is not None]
    timing_failure = None
    if not traces:
        timing_failure = "no_trace_timestamps"
    elif len(envelopes) != len(observed):
        timing_failure = "missing_or_invalid_trace_timestamps"
    rewards = [trace.reward for trace in observed]
    # Existing profiles are single-agent. Do not invent a multi-agent reducer.
    weighted_reward = rewards[0] if len(rewards) == 1 else None
    success = False if not ok else None
    if weighted_reward is not None and 0 <= weighted_reward <= 1:
        success = ok and traces[0].get("ok") is True and weighted_reward == 1.0  # ruff: ignore[float-equality-comparison]  reward is exactly 1.0 only for full success; partial credit must not count
    return EpisodeObservation(
        ordinal=ordinal,
        operational_ok=ok,
        recorded_error_count=error_count + sum(trace.error_count for trace in observed),
        trace_count=len(traces),
        trace_rewards=rewards,
        reward_components=[trace.components for trace in observed],
        weighted_reward=weighted_reward,
        full_reward_success=success,
        truncated=any(trace.truncated for trace in observed),
        start_unix_ns=min(lo for lo, _ in envelopes)
        if timing_failure is None
        else None,
        end_unix_ns=max(hi for _, hi in envelopes) if timing_failure is None else None,
        timing_failure=timing_failure,
    )


@dataclass
class NativeTraces:
    """Observations of one native traces file, counting malformed records."""

    episodes: list[EpisodeObservation] = field(default_factory=list)
    malformed: int = 0

    def consume(self, ordinal: int, raw: bytes) -> None:
        """Observe one raw record, counting it if malformed."""
        try:
            decoded: object = json.loads(raw)
            self.episodes.append(observe_episode(decoded, ordinal))
        except (ValueError, UnicodeError, OverflowError):
            self.malformed += 1


def serving_c1(serve: object) -> bool:
    """Check the resolved serving pool for single concurrency.

    Returns:
        Whether serving and its worker pool are both C1.

    """
    server = mapping(serve)
    pool = mapping(server.get("pool"))
    return server.get("max_concurrent") == 1 and pool.get("num_workers") == 1


def load_episodes(
    root: Path,
) -> tuple[list[EpisodeObservation], dict[str, object], bool]:
    """Consume only one fresh native run, counting malformed/torn records.

    Returns:
        The episodes, trace evidence, and whether the resolved run is C1.

    """
    paths = sorted(root.glob("*/traces.jsonl"))
    traces = NativeTraces()
    evidence: dict[str, object] = {"failures": [], "malformed_records": 0}
    failures: list[str] = []
    evidence["failures"] = failures
    if len(paths) != 1:
        failures.append("no_unique_native_traces_file")
        return traces.episodes, evidence, False
    digest = hashlib.sha256()
    try:
        with paths[0].open("rb") as stream:
            for ordinal, raw in enumerate(stream, 1):
                digest.update(raw)
                traces.consume(ordinal, raw)
    except OSError:
        failures.append("native_traces_read_failed")
    evidence["traces_sha256"] = digest.hexdigest()
    evidence["malformed_records"] = traces.malformed
    if traces.malformed:
        failures.append("malformed_native_records")
    if not traces.episodes:
        failures.append("no_native_episodes")
    c1 = False
    try:
        config = mapping(
            json.loads((paths[0].parent / "configs/resolved/eval.json").read_bytes())
        )
        serve = config.get("serve")
        c1 = config.get("max_concurrent") == 1
        if serve is not None:
            c1 = serving_c1(serve) and c1
    except (OSError, ValueError, UnicodeError):
        failures.append("resolved_concurrency_unavailable")
    return traces.episodes, evidence, c1


@dataclass
class Interruption:
    """Mutable signal state, including whether a forwarded signal raced exit."""

    signum: int | None = None
    delivery_races: int = 0

    def request(self, signum: int, _frame: FrameType | None) -> None:
        """Defer process control and evidence writing to the normal flow."""
        if self.signum is None:
            self.signum = signum

    def forward(self, pid: int, signum: int) -> None:
        """Signal only the native process group we started."""
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            self.delivery_races += 1


@dataclass
class NativeProcess:
    """Supervision state of the native command."""

    interruption: Interruption
    code: int = MISSING_EXECUTABLE_EXIT
    launch_failure: str | None = None

    def run(self, command: list[str], cwd: str) -> None:
        """Run the command in its own session, recording any launch failure."""
        try:
            with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: native eval command assembled by this runner from its arguments, no shell
                command,
                cwd=cwd,
                start_new_session=True,
            ) as child:
                self.supervise(child)
        except OSError:
            self.launch_failure = "native_process_launch_or_control_failed"

    def supervise(self, child: subprocess.Popen[bytes]) -> None:
        """Wait for the child, forwarding a requested signal then escalating."""
        interruption = self.interruption
        termination_deadline = None
        killed = False
        try:
            while True:
                if interruption.signum is not None and termination_deadline is None:
                    interruption.forward(child.pid, interruption.signum)
                    termination_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
                if (
                    termination_deadline is not None
                    and time.monotonic() >= termination_deadline
                    and not killed
                ):
                    interruption.forward(child.pid, signal.SIGKILL)
                    killed = True
                try:
                    self.code = child.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # Also reap descendants if the parent exits on a signal.
            if interruption.signum is not None or child.poll() is None:
                interruption.forward(child.pid, signal.SIGKILL)
                child.wait(timeout=REAP_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class NativeRun:
    """One measured native run and its clock anchors."""

    root: Path
    environment: str
    start: dict[str, int]
    end: dict[str, int]
    process: NativeProcess


@dataclass
class Attribution:
    """Per-episode interval evidence and its attributed totals."""

    details: list[dict[str, object]] = field(default_factory=list)
    attributed_ns: int = 0
    attributed_joules: float = 0.0
    observed_intervals: int = 0
    timing_failures: dict[str, int] = field(default_factory=dict)

    def integrate(self, sampler: PowerSampler, lo: int, hi: int) -> dict[str, object]:
        """Integrate one trace envelope and add it to the attributed totals.

        Returns:
            The envelope's power interval.

        """
        task_interval = integrate_power(sampler.samples, lo, hi, sampler.max_gap_ns)
        self.attributed_ns += hi - lo
        observed = task_interval["observed_joules"]
        if observed is not None:
            self.attributed_joules += number(observed)
            self.observed_intervals += 1
        return task_interval

    def record(
        self,
        episode: EpisodeObservation,
        task_interval: dict[str, object] | None,
        reason: str | None,
    ) -> None:
        """Count a timing failure and keep the episode's private details."""
        if reason is not None:
            self.timing_failures[reason] = self.timing_failures.get(reason, 0) + 1
        self.details.append({
            "episode_ordinal": episode.ordinal,
            "operational_ok": episode.operational_ok,
            "recorded_error_count": episode.recorded_error_count,
            "trace_count": episode.trace_count,
            "trace_weighted_rewards": episode.trace_rewards,
            "reward_components": episode.reward_components,
            "full_reward_success": episode.full_reward_success,
            "truncated": episode.truncated,
            "interval_scope": "persisted_trace_envelope_not_complete_attempt",
            "interval": task_interval,
            "timing_failure": reason,
        })


def clock_drift(sampler: PowerSampler, end: dict[str, int], offset: int) -> int:
    """Measure the largest Unix-to-monotonic offset change during the run.

    Returns:
        The maximum absolute drift in nanoseconds.

    """
    offsets = [end["unix_time_ns"] - end["monotonic_ns"]]
    offsets.extend(
        sample.unix_time_ns - sample.monotonic_ns for sample in sampler.samples
    )
    return max((abs(value - offset) for value in offsets), default=0)


def envelopes_overlap(episodes: list[EpisodeObservation], offset: int) -> bool:
    """Detect overlapping trace envelopes on the monotonic clock.

    Returns:
        Whether any two envelopes overlap.

    """
    spans = sorted(
        (episode.start_unix_ns - offset, episode.end_unix_ns - offset)
        for episode in episodes
        if episode.start_unix_ns is not None and episode.end_unix_ns is not None
    )
    return any(right[0] < left[1] for left, right in pairwise(spans))


def run_failure(*, c1: bool, clock_stable: bool, overlap: bool) -> str | None:
    """Name the first run-level reason that forbids any attribution.

    Returns:
        The reason, or None when episodes may be attributed.

    """
    if not c1:
        return "resolved_run_not_c1"
    if not clock_stable:
        return "host_clock_discontinuity"
    if overlap:
        return "overlapping_trace_envelopes"
    return None


def attribute_episodes(
    run: NativeRun,
    sampler: PowerSampler,
    episodes: list[EpisodeObservation],
    offset: int,
    failure: str | None,
) -> Attribution:
    """Integrate each attributable trace envelope inside the measured run.

    Returns:
        The per-episode details and attributed totals.

    """
    attribution = Attribution()
    for episode in episodes:
        reason = failure if failure is not None else episode.timing_failure
        task_interval = None
        if (
            reason is None
            and episode.start_unix_ns is not None
            and episode.end_unix_ns is not None
        ):
            lo = episode.start_unix_ns - offset
            hi = episode.end_unix_ns - offset
            if (
                lo < run.start["monotonic_ns"]
                or hi > run.end["monotonic_ns"]
                or hi <= lo
            ):
                reason = "trace_envelope_outside_measured_run"
            else:
                task_interval = attribution.integrate(sampler, lo, hi)
        attribution.record(episode, task_interval, reason)
    return attribution


@dataclass(frozen=True)
class Outcomes:
    """Task-success counts and whether per-success ratios are valid."""

    successes: int
    unknown_success: int
    denominator_complete: bool
    ratio_valid: bool
    rewards: list[float]


def task_outcomes(
    episodes: list[EpisodeObservation],
    trace_evidence: dict[str, object],
    native_exit_code: int,
) -> Outcomes:
    """Count successes and decide whether the success denominator is usable.

    Returns:
        The task outcomes.

    """
    successes = sum(episode.full_reward_success is True for episode in episodes)
    unknown_success = sum(episode.full_reward_success is None for episode in episodes)
    denominator_complete = not trace_evidence["failures"] and not unknown_success
    return Outcomes(
        successes=successes,
        unknown_success=unknown_success,
        denominator_complete=denominator_complete,
        ratio_valid=denominator_complete and successes > 0 and native_exit_code == 0,
        rewards=[
            episode.weighted_reward
            for episode in episodes
            if episode.weighted_reward is not None
        ],
    )


def ratio_reasons(outcomes: Outcomes, native_exit_code: int) -> list[str]:
    """Name every reason per-success ratios are unavailable.

    Returns:
        The reasons, empty when ratios are available.

    """
    reasons: list[str] = []
    if outcomes.successes == 0:
        reasons.append("zero_successful_task_rollouts")
    if not outcomes.denominator_complete:
        reasons.append("incomplete_success_denominator")
    if native_exit_code != 0:
        reasons.append("native_run_did_not_exit_successfully")
    return reasons


def efficiency_report(
    run: NativeRun, sampler: PowerSampler, interrupted_signal: int | None
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Attribute C1 trace envelopes only; all attempt costs remain in totals.

    Returns:
        The public summary and the private per-episode details.

    """
    episodes, trace_evidence, c1 = load_episodes(run.root)
    power = sampler.summary(run.start["monotonic_ns"], run.end["monotonic_ns"])
    interval = mapping(power["interval"])
    offset = run.start["unix_time_ns"] - run.start["monotonic_ns"]
    drift = clock_drift(sampler, run.end, offset)
    clock_stable = drift <= CLOCK_TOLERANCE_NS
    failure = run_failure(
        c1=c1,
        clock_stable=clock_stable,
        overlap=envelopes_overlap(episodes, offset),
    )
    attribution = attribute_episodes(run, sampler, episodes, offset, failure)
    outcomes = task_outcomes(episodes, trace_evidence, run.process.code)
    rewards = outcomes.rewards
    joules = interval["joules"]
    wall = number(interval["wall_seconds"])
    summary: dict[str, object] = {
        "schema_version": 1,
        "environment": run.environment,
        "verifiers_revision": VERIFIERS_REVISION,
        "native_exit_code": run.process.code,
        "interrupted_signal": interrupted_signal,
        "cost_scope": COST_SCOPE,
        "energy_scope": "selected_gpu_board_not_system_no_idle_subtraction",
        "success_definition": SUCCESS_DEFINITION,
        "success_unit": "task_rollout_not_unique_question_or_operational_completion",
        "success_denominator_complete": outcomes.denominator_complete,
        "successful_task_rollouts": outcomes.successes,
        "unknown_success_task_rollouts": outcomes.unknown_success,
        "recorded_task_rollouts": len(episodes),
        "operational_completions": sum(episode.operational_ok for episode in episodes),
        "operational_failures": sum(not episode.operational_ok for episode in episodes),
        "episodes_with_recorded_errors": sum(
            episode.recorded_error_count > 0 for episode in episodes
        ),
        "truncated_task_rollouts": sum(episode.truncated for episode in episodes),
        "fractional_reward_task_rollouts": sum(0 < value < 1 for value in rewards),
        "reward_observations": len(rewards),
        "weighted_reward_sum": math.fsum(rewards) if rewards else None,
        "weighted_reward_mean": math.fsum(rewards) / len(rewards) if rewards else None,
        "total_wall_seconds": wall,
        "total_joules": joules,
        "observed_joules": interval["observed_joules"],
        "seconds_per_successful_task_rollout": wall / outcomes.successes
        if outcomes.ratio_valid
        else None,
        "joules_per_successful_task_rollout": number(joules) / outcomes.successes
        if outcomes.ratio_valid and joules is not None
        else None,
        "time_ratio_available": outcomes.ratio_valid,
        "energy_ratio_available": outcomes.ratio_valid and joules is not None,
        "ratio_unavailable_reasons": ratio_reasons(outcomes, run.process.code),
        "power_measurement_status": "complete" if joules is not None else "incomplete",
        "resolved_c1": c1,
        "clock_alignment": {
            "method": "host_unix_to_monotonic_offset_at_cli_start",
            "max_observed_drift_ns": drift,
            "tolerance_ns": CLOCK_TOLERANCE_NS,
            "boundary_uncertainty_ns": max(
                run.start["uncertainty_ns"], run.end["uncertainty_ns"]
            ),
            "stable": clock_stable,
        },
        "trace_attribution": {
            "scope": ATTRIBUTION_SCOPE,
            "attributed_wall_seconds": attribution.attributed_ns / NANOSECONDS,
            "unattributed_wall_seconds": wall - attribution.attributed_ns / NANOSECONDS,
            "attributed_observed_joules": attribution.attributed_joules
            if attribution.observed_intervals
            else None,
            "timing_failures": attribution.timing_failures,
        },
        "trace_evidence": trace_evidence,
        "power": {key: value for key, value in power.items() if key != "gpu"},
    }
    return summary, attribution.details


def save_json(path: Path, value: object) -> None:
    """Write private JSON, refusing to overwrite earlier evidence."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_evidence(run: NativeRun, sampler: PowerSampler) -> None:
    """Persist samples, then the efficiency evidence, and point to it."""
    evidence_dir = run.root / "measurement"
    # Persist collection first, even if native trace parsing later fails.
    save_json(
        evidence_dir / "power-samples.json",
        {
            "schema_version": 1,
            "gpu": sampler.gpu,
            "start": run.start,
            "end": run.end,
            "samples": sampler.records(),
        },
    )
    interruption = run.process.interruption
    summary, details = efficiency_report(run, sampler, interruption.signum)
    summary["launch_failure"] = run.process.launch_failure
    summary["signal_delivery_races"] = interruption.delivery_races
    save_json(evidence_dir / "task-intervals.json", details)
    save_json(evidence_dir / "efficiency.json", summary)
    if (
        summary["total_joules"] is None
        or sampler.limit_reached
        or any(sample.error for sample in sampler.samples)
    ):
        sys.stderr.write(
            "Power measurement incomplete; inspect measurement/efficiency.json"
            " (native rewards unchanged).\n"
        )
    if not summary["success_denominator_complete"]:
        sys.stderr.write(
            "Task-success denominator incomplete; inspect native trace evidence"
            " (native rewards unchanged).\n"
        )
    sys.stderr.write(
        f"Per-environment efficiency evidence: {evidence_dir / 'efficiency.json'}\n"
    )


@dataclass
class Arguments:
    """Typed namespace for the existing eval wrapper's optional sidecar."""

    environment: str = ""
    output_dir: str = ""
    working_directory: str = ""
    gpu: str = "0"
    interval_seconds: float = 0.5
    max_seconds: float = 86400.0
    max_samples: int = 200_000
    command: list[str] | None = None


def parse_arguments() -> tuple[Arguments, list[str], PowerSampler]:
    """Parse the sidecar options and the native command after them.

    Returns:
        The options, the native command, and the validated power sampler.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--working-directory", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=86400.0)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = Arguments()
    parser.parse_args(namespace=args)
    command = args.command or []
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or re.fullmatch(r"[a-z0-9-]+", args.environment) is None:
        parser.error("A native command and a safe environment label are required")
    try:
        sampler = PowerSampler(
            args.gpu, args.interval_seconds, args.max_seconds, args.max_samples
        )
    except ValueError as error:
        parser.error(str(error))
    return args, command, sampler


def main() -> int:
    """Run exactly the supplied native command and report measurement separately.

    Returns:
        The native exit status, or 128 plus a received signal number.

    """
    args, command, sampler = parse_arguments()
    os.umask(0o077)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Fresh invocation only; do not mix resumed attempt costs.
    (root / "measurement").mkdir()
    process = NativeProcess(Interruption())
    previous = {
        sig: signal.signal(sig, process.interruption.request)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with sampler:
            start = clock_anchor()
            try:
                process.run(command, args.working_directory)
            finally:
                end = clock_anchor()
        write_evidence(NativeRun(root, args.environment, start, end, process), sampler)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if process.interruption.signum is not None:
        return SIGNAL_EXIT_BASE + process.interruption.signum
    code = process.code
    return SIGNAL_EXIT_BASE - code if code < 0 else code


if __name__ == "__main__":
    raise SystemExit(main())
