# Copyright (c) 2026 Gil Rodrigues
"""The autoresearch worker: freeze, collect and admit one suite, once.

``bench.autoresearch`` (the supervisor) starts this module under ``nix develop``
as ``python -m bench.autoresearch_worker --suite NAME``. Only this module loads
the served tokenizer, so the supervisor stays importable by host Python without
the ``tokenizers`` wheel's native runtime.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from bench import exl3
from bench.autoresearch import (
    FAILURES,
    ROOT,
    Settings,
    Suite,
    Window,
    api_key,
    arguments,
    guard,
    verify_container,
)
from bench.exl3 import mapping, sequence
from bench.tokenizer import RawTokenizer


def _identity_tokenizer(
    settings: Settings, identity: dict[str, object], candidate: dict[str, object]
) -> RawTokenizer:
    """Check the captured instance against the candidate and load its tokenizer.

    Args:
        settings: The operator settings.
        identity: The before-capture serving identity.
        candidate: The guardian's candidate record.

    Returns:
        The captured serving tokenizer.

    Raises:
        ValueError: If the capture selected another container or image.

    """
    instance = mapping(identity["container"])
    if not (
        instance.get("id") == settings.container
        and instance.get("image") == candidate.get("image")
    ):
        msg = "Captured identity selected another container or image"
        raise ValueError(msg)
    tokenizer_record = mapping(identity["tokenizer"])
    return RawTokenizer(
        Path(exl3.text(tokenizer_record["host_path"])),
        exl3.text(tokenizer_record["sha256"]),
    )


def _freeze_workload(
    settings: Settings,
    suite: Suite,
    client: exl3.Client,
    tokenizer: RawTokenizer,
    before_path: Path,
) -> tuple[dict[str, object], str, list[object]]:
    """Freeze the suite inputs, recheck the sources and save benchmark.json.

    Args:
        settings: The operator settings.
        suite: The selected suite.
        client: The endpoint client.
        tokenizer: The captured serving tokenizer.
        before_path: The before-capture identity file.

    Returns:
        The frozen workload, its SHA-256 and the snapshotted source records.

    Raises:
        ValueError: If a source changed after its initial snapshot.

    """
    identity_sha256 = exl3.digest(before_path)
    supervisor = exl3.document(settings.output / "supervisor.json")
    fields = suite.freeze(settings, client, tokenizer)
    sources = sequence(supervisor["sources"])
    for value in sources:
        item = mapping(value)
        name = exl3.text(item["path"]).removeprefix("sources/")
        if item.get("sha256") != exl3.digest(ROOT / name):
            msg = f"Benchmark source changed after its initial snapshot: {name}"
            raise ValueError(msg)
    workload: dict[str, object] = {
        "protocol": suite.protocol,
        "order": list(suite.order),
        "primary_metric": suite.primary_metric,
        "primary_scope": suite.primary_scope,
        "identity_before_sha256": identity_sha256,
        **fields,
        "sources": sources,
    }
    workload_sha256 = hashlib.sha256(exl3.canonical(workload)).hexdigest()
    benchmark = {
        **supervisor,
        "workload": workload,
        "workload_sha256": workload_sha256,
    }
    exl3.save(settings.output / "benchmark.json", benchmark)
    return workload, workload_sha256, sources


def _capture_after(
    settings: Settings,
    client: exl3.Client,
    before: dict[str, object],
    identity: dict[str, object],
    before_path: Path,
) -> tuple[exl3.Evidence, Window]:
    """Capture the after-identity and open the evidence closure.

    Args:
        settings: The operator settings.
        client: The endpoint client.
        before: The before-capture envelope.
        identity: The before-capture serving identity.
        before_path: The before-capture identity file.

    Returns:
        The evidence closure holding both captures and the identity window.

    Raises:
        ValueError: If the measurement changed the serving instance.

    """
    after_path = settings.output / "identity-after.json"
    after = exl3.capture_file(settings.container, client, after_path, before_path)
    evidence = exl3.Evidence(settings.output / "admitted.json")
    for path in (before_path, after_path):
        evidence.retain(path)
    if after["identity"] != identity:
        msg = "Measurement changed serving instance"
        raise ValueError(msg)
    window = (
        exl3.integer(before["finished_unix_ns"]),
        exl3.integer(after["started_unix_ns"]),
    )
    return evidence, window


def _retain_evidence(
    settings: Settings,
    suite: Suite,
    evidence: exl3.Evidence,
    sources: list[object],
) -> None:
    """Retain the snapshotted sources, run records, logs and suite trees.

    Args:
        settings: The operator settings.
        suite: The selected suite.
        evidence: The evidence closure.
        sources: The snapshotted source records.

    Raises:
        ValueError: If a snapshotted source changed.

    """
    for value in sources:
        item = mapping(value)
        path = settings.output / exl3.text(item["path"])
        if exl3.digest(evidence.retain(path)) != item["sha256"]:
            msg = f"Snapshotted source changed: {path}"
            raise ValueError(msg)
    for path in (
        settings.output / "benchmark.json",
        settings.output / "supervisor.json",
    ):
        evidence.retain(path)
    evidence.tree(settings.output / "sources")
    evidence.tree(settings.output / "logs")
    for tree in suite.trees:
        evidence.tree(settings.output / tree)


def worker(settings: Settings, suite: Suite) -> int:
    """Collect the frozen workload once, then replay raw-evidence admission.

    Returns:
        Zero once admitted.json is written.

    """
    state = guard(settings)
    verify_container(settings, state)
    candidate = mapping(state.get("candidate"))
    client = exl3.Client(api_key(settings.key_file))
    before_path = settings.output / "identity-before.json"
    before = exl3.capture_file(settings.container, client, before_path, None)
    identity = mapping(before["identity"])
    tokenizer = _identity_tokenizer(settings, identity, candidate)
    workload, workload_sha256, sources = _freeze_workload(
        settings, suite, client, tokenizer, before_path
    )
    suite.collect(settings, client)
    evidence, window = _capture_after(settings, client, before, identity, before_path)
    metrics, admitted = suite.admit(settings, evidence, workload, window)
    _retain_evidence(settings, suite, evidence, sources)
    exl3.save(
        settings.output / "admitted.json",
        {
            "schema_version": 1,
            "status": "complete_admitted_measurement",
            "protocol": suite.protocol,
            "scope": suite.scope,
            "order": list(suite.order),
            "workload_sha256": workload_sha256,
            "benchmark_sha256": exl3.digest(settings.output / "benchmark.json"),
            "identity": identity,
            "metrics": metrics,
            "primary_metric": suite.primary_metric,
            **admitted,
            "evidence_sha256": evidence.hashes,
        },
    )
    return 0


def _run_worker(settings: Settings, suite: Suite) -> int:
    """Check the supervisor's descriptor binding, then run the worker once.

    Args:
        settings: The operator settings.
        suite: The selected suite.

    Returns:
        The worker's exit status.

    Raises:
        ValueError: If the binding differs from the supervisor's descriptor.
        FAILURES: Any rejected observation, re-raised once worker-failure.json is
            saved.

    """
    binding = mapping(exl3.loads(sys.stdin.buffer.read(4097)))
    if binding != {
        "identity": settings.operator_identity,
        "sha256": settings.operator_sha256,
    }:
        msg = "Worker operator descriptor differs from supervisor startup"
        raise ValueError(msg)
    try:
        return worker(settings, suite)
    except FAILURES as error:
        exl3.save(
            settings.output / "worker-failure.json",
            {
                "status": "rejected",
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise


def main() -> int:
    """Run the worker once; preserve sanitized failure diagnostics privately.

    Returns:
        The process exit status.

    """
    os.umask(0o077)
    try:
        suite = arguments(sys.argv[1:])
    except ValueError as error:
        sys.stderr.write(f"{error}; use bash autoresearch.sh --help\n")
        return 2
    try:
        return _run_worker(Settings.descriptor(), suite)
    except FAILURES:
        sys.stderr.write(
            "EXL3 autoresearch rejected; no admitted metrics. "
            "Inspect private artifacts.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
