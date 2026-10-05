# Copyright (c) 2026 inference contributors.
"""Bounded board-power observations and gap-aware, clipped trapezoidal integration."""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Self, final

NANOSECONDS = 1_000_000_000
MIN_INTERVAL_SECONDS = 0.1
MAX_INTERVAL_SECONDS = 60
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 604800
MIN_SAMPLES = 2
MAX_SAMPLES = 1_000_000
GPU_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-:."
)


@dataclass(frozen=True)
class PowerSample:
    """One host observation; query midpoint timestamps are not device timestamps."""

    monotonic_ns: int
    unix_time_ns: int
    query_started_ns: int
    query_finished_ns: int
    watts: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid numeric observations rather than integrating them.

        Raises:
            ValueError: If the timestamp, power value or error state is invalid.

        """
        if not self.query_started_ns <= self.monotonic_ns <= self.query_finished_ns:
            msg = "Sample timestamp must lie within its query interval"
            raise ValueError(msg)
        if self.watts is not None and (not math.isfinite(self.watts) or self.watts < 0):
            msg = "Observed watts must be finite and nonnegative"
            raise ValueError(msg)
        if self.watts is not None and self.error is not None:
            msg = "Failed samples cannot also provide power"
            raise ValueError(msg)


def clock_anchor() -> dict[str, int]:
    """Pair the host Unix and monotonic clocks, with read uncertainty in ns.

    Returns:
        The midpoint monotonic time, Unix time and read uncertainty in ns.

    """
    before = time.monotonic_ns()
    unix_time = time.time_ns()
    after = time.monotonic_ns()
    return {
        "monotonic_ns": (before + after) // 2,
        "unix_time_ns": unix_time,
        "uncertainty_ns": (after - before + 1) // 2,
    }


def integrate_power(
    samples: list[PowerSample], start_ns: int, end_ns: int, max_gap_ns: int
) -> dict[str, object]:
    """Integrate only adjacent valid observations, clipped to [start, end].

    No extrapolation, bridging failed reads, or filling long gaps. Full-interval
    joules are null unless every nanosecond is covered. Zero observed power is
    valid, but no observation is never represented as zero energy.

    Args:
        samples: Observations in strictly increasing monotonic order.
        start_ns: Interval start on the monotonic clock.
        end_ns: Interval end on the monotonic clock.
        max_gap_ns: Longest sample spacing that may be integrated.

    Returns:
        Energy, coverage and gap evidence for the interval.

    Raises:
        ValueError: If the interval, gap bound or sample order is invalid.

    """
    if end_ns < start_ns or max_gap_ns <= 0:
        msg = "Invalid integration interval or maximum gap"
        raise ValueError(msg)
    if any(b.monotonic_ns <= a.monotonic_ns for a, b in pairwise(samples)):
        msg = "Power samples must be strictly monotonic"
        raise ValueError(msg)
    covered_ns = 0
    joules = 0.0
    cursor = start_ns
    gaps: list[dict[str, object]] = []
    for left, right in pairwise(samples):
        lo = max(start_ns, left.monotonic_ns)
        hi = min(end_ns, right.monotonic_ns)
        if hi <= lo:
            continue
        if lo > cursor:
            gaps.append({
                "start_ns": cursor,
                "end_ns": lo,
                "reason": "unobserved_boundary",
            })
        span = right.monotonic_ns - left.monotonic_ns
        left_watts = left.watts
        right_watts = right.watts
        if left_watts is None or right_watts is None:
            gaps.append({"start_ns": lo, "end_ns": hi, "reason": "failed_sample"})
        elif span > max_gap_ns:
            gaps.append({"start_ns": lo, "end_ns": hi, "reason": "sampling_gap"})
        else:
            # Both endpoints are observed; no nominal/average fill value.
            slope = (right_watts - left_watts) / span
            low_watts = left_watts + slope * (lo - left.monotonic_ns)
            high_watts = left_watts + slope * (hi - left.monotonic_ns)
            joules += (low_watts + high_watts) * 0.5 * (hi - lo) / NANOSECONDS
            covered_ns += hi - lo
        cursor = hi
    if cursor < end_ns:
        gaps.append({
            "start_ns": cursor,
            "end_ns": end_ns,
            "reason": "unobserved_boundary",
        })
    elapsed_ns = end_ns - start_ns
    complete = elapsed_ns > 0 and covered_ns == elapsed_ns
    return {
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "wall_seconds": elapsed_ns / NANOSECONDS,
        "observed_joules": joules if covered_ns else None,
        "joules": joules if complete else None,
        "covered_seconds": covered_ns / NANOSECONDS,
        "uncovered_seconds": (elapsed_ns - covered_ns) / NANOSECONDS,
        "coverage_fraction": covered_ns / elapsed_ns if elapsed_ns else None,
        "complete": complete,
        "gaps": gaps,
    }


def _parse_power(stdout: str) -> tuple[float | None, str | None]:
    """Parse one nvidia-smi power reading.

    Args:
        stdout: The query output.

    Returns:
        The observed watts, or the failure reason.

    """
    try:
        value = float(stdout.strip())
    except ValueError:
        return None, "invalid_power_output"
    if math.isfinite(value) and value >= 0:
        return value, None
    return None, "invalid_power_value"


def _query_failure(exc: subprocess.TimeoutExpired | ValueError | OSError) -> str:
    """Name a failed nvidia-smi query.

    Args:
        exc: The exception the query raised.

    Returns:
        The recorded failure reason.

    """
    if isinstance(exc, FileNotFoundError):
        return "nvidia_smi_not_found"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "nvidia_smi_timeout"
    if isinstance(exc, ValueError):
        return "invalid_power_output"
    return "nvidia_smi_os_error"


@final
class PowerSampler:
    """Sample one selected board on absolute monotonic deadlines, within bounds."""

    def __init__(
        self,
        gpu: str = "0",
        interval_seconds: float = 0.5,
        max_seconds: float = 86400.0,
        max_samples: int = 200_000,
    ) -> None:
        """Validate collection bounds; this never changes device power policy.

        Args:
            gpu: One GPU index, UUID or PCI bus ID.
            interval_seconds: Sampling period.
            max_seconds: Collection duration bound.
            max_samples: Collection sample-count bound.

        Raises:
            ValueError: If a bound or the GPU selector is invalid.

        """
        if (
            not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS
            or not MIN_DURATION_SECONDS <= max_seconds <= MAX_DURATION_SECONDS
        ):
            msg = "Power interval must be 0.1..60 s and duration 1..604800 s"
            raise ValueError(msg)
        if not MIN_SAMPLES <= max_samples <= MAX_SAMPLES:
            msg = "Power sample limit must be 2..1000000"
            raise ValueError(msg)
        if not gpu or any(c not in GPU_ID_CHARACTERS for c in gpu):
            msg = "Use a single GPU index, UUID, or PCI bus ID"
            raise ValueError(msg)
        self.gpu = gpu
        self.interval_ns = int(interval_seconds * NANOSECONDS)
        self.max_gap_ns = self.interval_ns * 3
        self.max_ns = int(max_seconds * NANOSECONDS)
        self.max_samples = max_samples
        self.samples: list[PowerSample] = []
        self.limit_reached: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._deadline_ns = 0
        self.started: dict[str, int] = {}
        self.ended: dict[str, int] = {}

    def _query(self) -> tuple[float | None, str | None]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None, "nvidia_smi_not_found"
        try:
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  argv: nvidia-smi from PATH + fixed power.draw query, no shell
                [
                    executable,
                    f"--id={self.gpu}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            return None, _query_failure(exc)
        if result.returncode:
            return None, f"nvidia_smi_exit_{result.returncode}"
        return _parse_power(result.stdout)

    def _sample(self) -> None:
        if len(self.samples) >= self.max_samples:
            self.limit_reached = "sample_limit"
            return
        if time.monotonic_ns() >= self._deadline_ns:
            self.limit_reached = "duration_limit"
            return
        before = clock_anchor()
        watts, error = self._query()
        after = clock_anchor()
        self.samples.append(
            PowerSample(
                monotonic_ns=(before["monotonic_ns"] + after["monotonic_ns"]) // 2,
                unix_time_ns=(before["unix_time_ns"] + after["unix_time_ns"]) // 2,
                query_started_ns=before["monotonic_ns"],
                query_finished_ns=after["monotonic_ns"],
                watts=watts,
                error=error,
            )
        )

    def _loop(self) -> None:
        deadline = time.monotonic_ns() + self.interval_ns
        while not self._stop.wait(max(0, deadline - time.monotonic_ns()) / NANOSECONDS):
            self._sample()
            if self.limit_reached:
                return
            deadline += self.interval_ns
            # Missed deadlines are visible as gaps; never issue catch-up bursts.
            now = time.monotonic_ns()
            if deadline <= now:
                deadline += (
                    (now - deadline) // self.interval_ns + 1
                ) * self.interval_ns

    def __enter__(self) -> Self:
        """Bracket the measured interval with a first observation.

        Returns:
            This sampler.

        """
        self._deadline_ns = time.monotonic_ns() + self.max_ns
        self._sample()
        self.started = clock_anchor()
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        """Stop and join collection, including on exceptions; bracket the end."""
        self.ended = clock_anchor()
        self._stop.set()
        self._thread.join()
        self._sample()

    def summary(
        self, start_ns: int | None = None, end_ns: int | None = None
    ) -> dict[str, object]:
        """Return evidence only after cleanup, never a fabricated zero sample.

        Args:
            start_ns: Interval start; defaults to the sampler entry time.
            end_ns: Interval end; defaults to the sampler exit time.

        Returns:
            Power statistics and interval integration evidence.

        Raises:
            RuntimeError: If the sampler has not been closed.

        """
        if not self.ended:
            msg = "Power summary requires a closed sampler"
            raise RuntimeError(msg)
        interval = integrate_power(
            self.samples,
            self.started["monotonic_ns"] if start_ns is None else start_ns,
            self.ended["monotonic_ns"] if end_ns is None else end_ns,
            self.max_gap_ns,
        )
        values = sorted(
            sample.watts for sample in self.samples if sample.watts is not None
        )
        errors: dict[str, int] = {}
        for sample in self.samples:
            if sample.error is not None:
                errors[sample.error] = errors.get(sample.error, 0) + 1
        return {
            "schema_version": 1,
            "scope": "selected_gpu_board_not_system_or_decode_only",
            "integration": "piecewise_linear_trapezoids_no_extrapolation",
            "timestamp_basis": "host_query_midpoint_not_device_timestamp",
            "units": {"power": "W", "energy": "J", "duration": "s", "timestamps": "ns"},
            "gpu": self.gpu,
            "sampling_interval_seconds": self.interval_ns / NANOSECONDS,
            "max_gap_seconds": self.max_gap_ns / NANOSECONDS,
            "sample_count": len(self.samples),
            "valid_sample_count": len(values),
            "sample_failures": errors,
            "limit_reached": self.limit_reached,
            "watts_min": values[0] if values else None,
            "watts_p50": values[len(values) // 2] if values else None,
            "watts_p90": values[min(len(values) - 1, len(values) * 9 // 10)]
            if values
            else None,
            "watts_max": values[-1] if values else None,
            "interval": interval,
        }

    def records(self) -> list[dict[str, int | float | str | None]]:
        """Return timestamped raw observations without subprocess output or secrets.

        Returns:
            One record per observation.

        """
        return [
            {
                "monotonic_ns": sample.monotonic_ns,
                "unix_time_ns": sample.unix_time_ns,
                "query_started_ns": sample.query_started_ns,
                "query_finished_ns": sample.query_finished_ns,
                "watts": sample.watts,
                "error": sample.error,
            }
            for sample in self.samples
        ]
