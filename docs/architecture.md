# Architecture

## Fixed target

Qwen3.8-27B EXL3 4.00 bpw + native DFlash2 draft on one NVIDIA RTX 3090 (SM86,
24 GiB). Optimize for one active agent: C1 latency, useful code/reasoning output
and reliably usable context toward **262144 total tokens**. A smaller passing
context is a measured operating point, never a new model limit.

## Ownership

| Boundary | Owner |
|---|---|
| Execution, cache, speculative decoding | ExLlamaV3 at `355c6ee10fbd25b79070316a81ea0708cc18155a` with native DFlash2 from the authenticated base image; unchanged in `baseline` builds, patched by the SHA-pinned `patches/exl3` series in `candidate` builds |
| OpenAI-compatible API | `serve/exl3_server.py`, baked by `Dockerfile.exl3` / `docker/build-exl3.sh` |
| Launch recipe | `serve/exl3-entrypoint.sh`: context 262144, cache 270336, CQ3, greedy, one sequence |
| Weights | Pinned public EXL3 target/draft revisions; every file SHA256-bound in `prepare/exl3-manifest.json` and rehashed at every start |
| Runtime | Docker Compose; Nix supplies development tools, the `.#bend` toolchain and a thin adapter |
| GPU driver, storage, fan and optional power cap | Host |
| Capability tasks, scoring, rewards and traces | Pinned Prime Envs + Verifiers in `eval/` |
| Throughput measurement | `bash autoresearch.sh` EXL3 lane ([protocol](benchmarks.md)) |
| Proven decision logic | Bend modules under `bend/`, checked through `PROOF.bend` |

No private engine fork, vendored upstream source tree or runtime source overlay.
The base image ID defines the engine and its dependency closure; the manifest
defines the weights. Record both with every run.

## Bend's role

Bend states laws for decisions the serving path must get right and proves the
production definitions against them (`LAWS.bend` → `PROOF.bend`). The current
focus is EXL3 greedy speculative acceptance: the accepted-prefix and bonus-token
decision for each DFlash2 block, proved in Bend and executed in the serving path,
and, for the 8-row tree verify, the matching-path acceptance and the TreeDesc
derivation (`bend/exl3_tree_accept.bend`) with tree speculation invariance
(`bend/spec_inv_tree_laws.bend`). Both artifacts are built into one acceptor root
by `bend/exl3_build.py` with the pinned Bend 2.0.34.
A Bend proof covers the Bend definition; it does not certify
the CUDA kernels, the Python server or measured speed. Those need their own
evidence.

## Invariants

1. Compare target-only and DFlash2 with identical weights, prompts, templates,
   sampling and budgets. Greedy speculative decoding must emit exactly the
   target's greedy tokens; acceptance commits only verified tokens.
2. Prove capacity with actual deep prompts, completed prefill and meaningful
   generation, not server metadata alone.
3. Freeze each A/B workload and quality decision rule before observing results.
   Keep per-environment scores, errors and truncation rates; no combined
   intelligence score. Report uncertainty and failed experiments.
4. Bind results to recipe, model, image and evaluator identities. Never multiply
   gains from unlike fixtures or transfer another engine's results to this one.

See [benchmarks](benchmarks.md), [Docker deployment](docker.md) and
[development](development.md).
