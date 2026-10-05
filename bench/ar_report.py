# Copyright (c) 2026 Gil Rodrigues
"""Turn one ``bench/ar_gpu.py`` result into autoresearch METRIC lines.

  python3 -I -B bench/ar_report.py RESULT.json REFERENCE.json
  python3 -I -B bench/ar_report.py RESULT.json REFERENCE.json --write-reference

Primary: speed_score = sqrt(prefill_tok_s x decode_tok_s), where
  prefill_tok_s = geomean over 8K/32K/128K of (prompt tokens / TTFT),
  decode_tok_s  = geomean over 1K/8K/32K of (decode tokens / decode-round time).
Quality proxies against the reference run (greedy text identity):
  ref_decode_equal (0-3 depths), ref_first_token_equal (0-5 prefill rows).
Rejects (exit 1, no METRIC): an incomplete run, a power limit other than
350 W, or a decode depth whose two repeats differ.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

POWER_LIMIT_W = 350.0
POWER_TOLERANCE_W = 0.5
PREFILL_DEPTHS = (8192, 32768, 131072)
DECODE_DEPTHS = (1024, 8192, 32768)
DECODE_REPEATS = 2

type Run = dict[str, object]


class ReportError(ValueError):
    """The result is not a valid measurement."""


def num(run: Run, key: str) -> float:
    """Read a numeric field.

    Returns:
        The value as float.

    Raises:
        ReportError: The field is missing or not a number.

    """
    v = run.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        msg = f"{run.get('key')}: {key} is not a number"
        raise ReportError(msg)
    return float(v)


def text(run: Run, key: str) -> str:
    """Read a string field.

    Returns:
        The value.

    Raises:
        ReportError: The field is missing or not a string.

    """
    v = run.get(key)
    if not isinstance(v, str):
        msg = f"{run.get('key')}: {key} is not a string"
        raise ReportError(msg)
    return v


def first_token(run: Run) -> int:
    """Read the first generated token id.

    Returns:
        The token id.

    Raises:
        ReportError: The run has no tokens.

    """
    toks = run.get("tokens")
    if not isinstance(toks, list) or not toks or not isinstance(toks[0], int):
        msg = f"{run.get('key')}: no tokens"
        raise ReportError(msg)
    return toks[0]


def load_runs(path: Path) -> list[Run]:
    """Load and admit a result file.

    Returns:
        The runs.

    Raises:
        ReportError: The run is incomplete or at another power limit.

    """
    res = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(res, dict):
        msg = f"{path}: not an object"
        raise ReportError(msg)
    if res.get("complete") is not True or res.get("errors"):
        msg = f"{path}: incomplete run: {res.get('errors')}"
        raise ReportError(msg)
    limit = res.get("power_limit_w")
    if not isinstance(limit, float) or abs(limit - POWER_LIMIT_W) > POWER_TOLERANCE_W:
        msg = f"power limit {limit} W, not {POWER_LIMIT_W} W"
        raise ReportError(msg)
    runs = res.get("runs")
    if not isinstance(runs, list):
        msg = "runs is not a list"
        raise ReportError(msg)
    out: list[Run] = []
    for r in runs:
        if not isinstance(r, dict):
            msg = "run is not an object"
            raise ReportError(msg)
        out.append({str(k): v for k, v in r.items()})
    return out


def of(runs: list[Run], kind: str, depth: int) -> list[Run]:
    """Select the runs of one kind and depth.

    Returns:
        The selected runs, in run order.

    """
    return [r for r in runs if r.get("kind") == kind and r.get("depth") == depth]


def geomean(values: list[float]) -> float:
    """Geometric mean of positive values.

    Returns:
        The geometric mean.

    """
    return math.exp(statistics.fmean(math.log(v) for v in values))


def reference_of(runs: list[Run]) -> dict[str, object]:
    """Extract the identity reference from a run.

    Returns:
        Decode text hashes per depth and first token per prefill row.

    """
    return {
        "decode": {
            str(d): text(of(runs, "decode", d)[0], "tokens_sha256")
            for d in DECODE_DEPTHS
        },
        "first_token": {
            text(r, "key"): first_token(r)
            for d in PREFILL_DEPTHS
            for r in of(runs, "prefill", d)
        },
    }


def metrics(runs: list[Run], ref: dict[str, object] | None) -> dict[str, float]:
    """Compute every METRIC value.

    Returns:
        Metric name to value.

    Raises:
        ReportError: A row is missing or decode repeats differ.

    """
    m: dict[str, float] = {}
    pre: list[float] = []
    for d in PREFILL_DEPTHS:
        rows = of(runs, "prefill", d)
        if not rows:
            msg = f"no prefill rows at {d}"
            raise ReportError(msg)
        tps = sum(num(r, "prompt_tokens") for r in rows) / sum(
            num(r, "ttft_s") for r in rows
        )
        m[f"prefill_tok_s_{d}"] = tps
        m[f"ttft_s_{d}"] = statistics.fmean(num(r, "ttft_s") for r in rows)
        pre.append(tps)
    dec: list[float] = []
    power: list[float] = []
    for d in DECODE_DEPTHS:
        rows = of(runs, "decode", d)
        if len(rows) != DECODE_REPEATS:
            msg = f"{len(rows)} decode rows at {d}, expected {DECODE_REPEATS}"
            raise ReportError(msg)
        if len({text(r, "tokens_sha256") for r in rows}) != 1:
            msg = f"decode repeats at {d} differ: greedy output is not deterministic"
            raise ReportError(msg)
        rounds = [x for r in rows for x in _floats(r, "round_ms")]
        toks = sum(num(r, "decode_tokens") for r in rows)
        tps = toks / sum(num(r, "decode_s") for r in rows)
        m[f"decode_tok_s_{d}"] = tps
        m[f"round_ms_{d}"] = statistics.median(rounds)
        m[f"tokens_per_round_{d}"] = toks / len(rounds)
        dec.append(tps)
        power += [_nvml(r, "power_w_mean") for r in rows]
    m["prefill_tok_s"] = geomean(pre)
    m["decode_tok_s"] = geomean(dec)
    m["speed_score"] = math.sqrt(m["prefill_tok_s"] * m["decode_tok_s"])
    m["decode_power_w"] = statistics.fmean(power)
    m["thermal_frac_max"] = max(_nvml(r, "thermal_frac") for r in runs)
    if ref is not None:
        mine = reference_of(runs)
        for part, name in (
            ("decode", "ref_decode_equal"),
            ("first_token", "ref_first_token_equal"),
        ):
            a, b = mine[part], ref.get(part)
            if not isinstance(a, dict) or not isinstance(b, dict):
                msg = f"reference has no {part}"
                raise ReportError(msg)
            m[name] = float(sum(1 for k, v in a.items() if b.get(k) == v))
    return m


def _floats(run: Run, key: str) -> list[float]:
    v = run.get(key)
    if not isinstance(v, list) or not all(isinstance(x, int | float) for x in v):
        msg = f"{run.get('key')}: {key} is not a list of numbers"
        raise ReportError(msg)
    return [float(x) for x in v]


def _nvml(run: Run, key: str) -> float:
    nv = run.get("nvml")
    if not isinstance(nv, dict):
        msg = f"{run.get('key')}: no nvml summary"
        raise ReportError(msg)
    return num({str(k): v for k, v in nv.items()}, key)


def report(result: Path, reference: Path, *, write: bool) -> None:
    """Print METRIC lines, or write the reference.

    Raises:
        ReportError: The reference is not an object.

    """
    runs = load_runs(result)
    if write:
        reference.write_text(
            json.dumps(reference_of(runs), indent=1) + "\n", encoding="utf-8"
        )
        sys.stdout.write(f"wrote {reference}\n")
        return
    ref = None
    if reference.exists():
        ref = json.loads(reference.read_text(encoding="utf-8"))
        if not isinstance(ref, dict):
            msg = f"{reference} is not an object"
            raise ReportError(msg)
    for k, v in sorted(metrics(runs, ref).items()):
        sys.stdout.write(f"METRIC {k}={v:.6g}\n")


def main() -> int:
    """Parse arguments and report.

    Returns:
        The process exit status.

    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--write-reference", action="store_true")
    a = ap.parse_args()
    try:
        report(a.result, a.reference, write=a.write_reference)
    except (ReportError, OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"ar_report: {e}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
