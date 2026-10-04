# Standard model-quality evaluation

Run `eval/scripts/setup`, then `eval/scripts/data aime24`,
`eval/scripts/sandbox`, and `eval/scripts/run smoke`. Enter the pinned development shell first:

```sh
nix develop . --no-write-lock-file
```

Use the Git-backed `.` flake reference: `path:.` also copies ignored datasets
and source checkouts into the Nix store.

`bench/` measures the inference implementation. `eval/` invokes upstream
Tasksets, harnesses, scorers, traces and rewards. There are no local benchmark
prompts, generators, answer parsers or scoring rules.

## Install and smoke-test

```sh
eval/scripts/setup
eval/scripts/data aime24
eval/scripts/sandbox
export QWEN_API_KEY_FILE=/absolute/path/to/serving/state/api-key
eval/scripts/run smoke
```

The key file must belong to you, have mode 0400 or 0600, and contain the serving
key. The wrapper reads it into `QWEN_API_KEY`; no key goes into command arguments
or saved config. Do not run with shell tracing or paste keys into commands.
The server must already be running. Evaluation never starts, replaces or
reconfigures it and never obtains model weights.

For a model-free config check:

```sh
eval/scripts/run smoke --dry-run
```

The wrapper uses the installed native `eval` CLI from the pinned checkout.
It normally runs from the Prime Envs root; GraphWalks uses an isolated data
working directory as described below. Its smoke invocation is equivalent to the current upstream guidance:

```sh
uv run --no-sync eval aime24 -n 3 -r 1 --no-rich -v --no-push
```

Our native TOML additionally selects the authenticated local model, C1, fixed
sampling, a tool-free harness, and the local Docker runtime. No compatibility
proxy or upstream source patch is installed. Verifiers' own interception layer
is part of its standard harness, not a repository-specific adapter.

## What is pinned

| Input | Source of truth |
|---|---|
| Prime Envs | `prime-envs.lock`: `https://github.com/PrimeIntellect-ai/prime-envs`, `c4d04dfe212c153a587ea4ce072ae6753e74d6e9` |
| Verifiers | `prime-envs.lock`: `https://github.com/PrimeIntellect-ai/verifiers`, `ef47b2e96284a00bdcfc1012b9624b0c41ee6a0e`, version `0.3.2.dev86` |
| Environment packages | Editable packages from that exact Prime Envs checkout; versions in `pyproject.toml` and `uv.lock` |
| Python/tools | Repository `flake.lock`; Python 3.12 for eval, uv 0.12.1; Python 3.13 remains the repository development interpreter |
| Python dependencies | `eval/uv.lock`, including the build tools (`build` group); separate native uv script locks in `runtime/` for the actual harness/scoring subprocesses |
| Task data | `datasets.lock`: HF commit, exact source files and hashes; MRCR GCS generation, size and SHA256 |
| Sandbox | Digest-pinned base and uv images in `runtime/Dockerfile`; helper-image pins in `runtime/images.lock`; actual built image ID saved in each resolved Docker launch |

Versions at this Prime Envs revision: AIME24/25/26 `0.1.0`, I3 Logic `0.2.1`,
LiveCodeBench `0.1.0`, MMLU-Pro `0.1.0`, MRCR v2 `0.1.0`, GraphWalks `0.1.0`.

Source checkouts under `.sources/` are obtained by `scripts/setup`, never vendored
into Git. Existing wrong-revision or dirty checkouts fail instead of being reset.
The separate eval project is intentional: upstream's root uv lock does **not**
include Verifiers or these environment packages. We lock their actual combined
installation instead of relying on editable `pip install` resolution at run time.
The editables build without isolation from the hash-locked `build` dependency
group, so uv never fetches an unlocked build requirement. Verifiers' dynamic
version comes from static `dependency-metadata`; `scripts/setup` fails if
hatchling builds different metadata.

The chosen Verifiers commit includes upstream's fix to reuse an installed uv.
The earlier 0.3.1 release upgraded uv during each harness setup. Native runtime
script locks and `UV_LOCKED=true` also prevent independent PEP 723 dependency
drift. `UV_FROZEN` is not exported. The evaluator uses a private HOME to avoid user-local
platform configuration shadowing. Both source and dependency pins
must be reviewed together on upgrades; nothing silently follows `main`.

### Dataset acquisition is not evaluation logic

`scripts/data` downloads unchanged upstream dataset assets through Hugging Face
or generation-pinned GCS URLs. It constructs no tasks and evaluates no answers.
AIME supplies revision knobs. Current I3 Logic, LiveCodeBench and GraphWalks do
not. Their standard HF `refs/main` entries are explicitly bound to the locked
commit **only inside this repository's owned cache**. MMLU-Pro hardcodes both its
Hub name `TIGER-Lab/MMLU-Pro` and revision `b189ec76…`; the lock pins that same
revision (README plus the `test` and `validation` parquet files its default config
declares). All runs use HF offline
mode and verify the source files before loading them. Each invocation has a fresh derived Arrow cache; only verified raw assets are
reused. Shared/user HF caches are not modified. A mismatching cache fails; it is never silently repaired or refreshed.

The wrapper passes AIME and I3 Logic the verified local snapshot through their
native dataset-name configuration. A fresh offline `datasets.load_dataset`
cannot resolve Hub metadata from raw cached files alone. Local snapshot loading
keeps the upstream README/schema and data unchanged; the actual absolute path
and revision are saved in the resolved config. I3 uses the raw `logic/`
subdirectory with the native parquet builder's `default` config; no rows are
rewritten. GraphWalks hardcodes `openai/graphwalks` and MMLU-Pro hardcodes
`TIGER-Lab/MMLU-Pro`, so their invocations use a fresh working directory with
that relative path symlinked to the verified snapshot. Hugging Face natively
resolves local paths before Hub names (the hardcoded revision argument is then
not consulted, which is why the lock pins exactly that revision). That
working directory is recorded in provenance and remains outside every virtual
environment. No cached Arrow artifacts or task banks are imported.

Data preparation can be scoped to one Taskset:

```sh
eval/scripts/data i3-logic
eval/scripts/data livecodebench
eval/scripts/data mmlu-pro
eval/scripts/data --check livecodebench
```

LiveCodeBench v6 downloads about **4.49 GB** of upstream test data even for a
small evaluation. This is upstream loader behavior. Keep dataset files and
native traces ignored. No private task banks are created.

## Profiles

```sh
eval/scripts/run quick                  # all five model-only Tasksets
eval/scripts/run quick i3-logic         # one unchanged profile entry
eval/scripts/run full aime25            # full AIME25, 20 rollouts per task
```

Prepare each selected dataset first. For the complete core suite, prepare
`aime24`, `aime25`, `aime26`, `i3-logic`, and `livecodebench`; build the coding
sandbox with `eval/scripts/sandbox`. `eval/scripts/data all` also downloads
optional long-context data. `run` never downloads missing task data.

| Profile | Task selection | Rollouts per task | Purpose |
|---|---|---|---|
| `smoke` | First 3 AIME24 tasks; no shuffle | 1 | Prove loading → local inference → upstream scoring → saved traces |
| `quick` | Native fixed-seed shuffle: 10 each AIME24/25/26, 24 I3 Logic, 12 LiveCodeBench | 1 | 66 fixed comparison tasks; not a broad intelligence score |
| `full` | All tasks after each upstream environment's configured filters | 20 for each AIME; 1 I3 Logic; 2 LiveCodeBench | Per-environment estimates with representative rollout counts |
| `tiny` + `broad` | Native fixed-seed shuffle: 3 AIME25 (32768 budget); 10 MMLU-Pro (8192), 4 I3 Logic (16384), 3 LiveCodeBench (16384) | 1 | The broad autoresearch suite tasksets (`python -m bench.autoresearch --suite broad`; [../docs/benchmarks.md](../docs/benchmarks.md)); sampled, not qualification |

Verifiers owns task shuffling (seed 0 at this revision). There is no local sampler
or selection state. Quick comparisons reuse exactly the same datasets, order,
configs, sampling and budgets. The roughly one-hour quick estimate was measured
with C1 and the earlier 8192-token cap; the current 32768-token cap admits longer
episodes, so budget more, plus startup/scoring overhead; actual time depends on
the model. Full is deliberately expensive: AIME alone is
1800 rollouts. Run one environment at a time when practical.

Upstream recommends usually **more than 500 total rollouts** for full runs, not a
particular universal repetition count. Each AIME has 30 tasks × 20 = 600 rollouts.
Large I3 Logic uses one rollout. LiveCodeBench uses two with its official v6 date
filter (inclusive 2024-08-01 through 2025-05-01): 454 tasks × 2 = 908 rollouts
at the pinned data revision. Report the actual post-filter
count from traces, not the unfiltered dataset size. Repetitions use stochastic
sampling; they do not create new independent questions.

`configs/local.toml` defines the model `qwen3.8-27b`, endpoint
`http://127.0.0.1:18020/v1`, client type `eval`, `api_key_var = "QWEN_API_KEY"`,
greedy decoding (temperature 0, top-p 1, no top-k), min-p 0, no frequency/presence
penalties, repetition penalty 1, thinking enabled, and at most 32768 output tokens
per call; a profile may lower only `sampling.max_tokens` (`broad` 8192/16384,
`diverse` 8192).
C1 and one server worker avoid oversubscribing the 3090. Core harness `null` has
one model turn and no model tools. All profiles use the same locked Docker
runtime. Current AIME tasks require a network policy that the host subprocess
runtime cannot enforce; using Docker preserves that upstream policy rather
than relaxing it. Docker isolation alone does not make a tool-free run agentic.

AIME measures competition math. I3 Logic adds diverse task-specific reasoning;
it is a public training split, not a contamination-free held-out intelligence
test. LiveCodeBench adds single-turn competitive-programming generation with
upstream hidden-test scoring. HumanEval is not used. I3 Code substantially
overlaps this coding capability. Other reasoning candidates were not added just
to inflate benchmark count.

## Optional tools and long-context profiles

```sh
eval/scripts/data mrcr-v2
eval/scripts/data graphwalks
eval/scripts/sandbox
eval/scripts/run agentic                # 8 tasks per entry, 1 rollout each
eval/scripts/run agentic-full mrcr-v2-256k
```

These profiles are **model + upstream harness + tools + sandbox** evaluations.
They never run as part of `smoke`, `quick`, or `full`.

| Config entry | Actual Taskset | Evidence |
|---|---|---|
| `mrcr-v2-64k` | `mrcr-v2`, 8 needles, `32k-64k` source bucket | Transcript coreference/retrieval; official prefix-gated SequenceMatcher reward |
| `mrcr-v2-256k` | `mrcr-v2`, 8 needles, `128k-256k` source bucket | Same capability over larger source files |
| `graphwalks` | `graphwalks`, BFS and parents, exact scoring | Complementary graph traversal over uploaded edge lists |

Agentic full loads each complete configured bucket/taskset: MRCR has 8 rollouts
per task (85 × 8 = 680 in the smaller bucket; 141 × 8 = 1128 in the larger);
GraphWalks has 1150 tasks × 1. Both use the upstream bash harness, local Docker,
16 model turns, at most 32768 output tokens across the episode, and the same
32768-token per-call cap. No paid search, remote Prime sandbox, or LLM judge is
configured. Untrusted model code never runs in the host subprocess runtime.

**These are not direct model-window tests.** At the pinned Prime Envs revision,
MRCR v2, GraphWalks, and LongBench-Pro upload context to files and ask an agent to
search them. A `128k-256k` file bucket neither sends that many tokens to Qwen nor
proves it fits the configured model window. LongBench-Pro adds document-understanding
metrics, but also file/REPL harness behavior, excluded task families and extra
dependencies; GraphWalks is the simpler complementary choice here. Direct-context
rows live in `eval/direct/` and the EXL3 benchmark lane (`bash autoresearch.sh`).

The sandbox build uses upstream program bytes and native uv locks only. It does
not alter an environment or scorer. Its build context excludes datasets, model
weights, keys and results. The wrapper substitutes the built immutable image ID
into a copied native launch TOML. Verifiers also hardcodes two helper image tags;
`scripts/sandbox` verifies their exact pinned image IDs and refuses to replace a
pre-existing different tag. Use an isolated Docker daemon if other workloads
need different versions. No serving containers are stopped or modified.

## Results and provenance

Each invocation creates a fresh ignored `runs/PROFILE.RANDOM/` directory:

```text
provenance/                       # source/data/dependency/config/recipe identities
TASKSET/RUN/
  configs/eval.toml                # upstream launch config
  configs/resolved/eval.json       # authoritative complete resolved config
  traces.jsonl                    # native episodes, messages, rewards and errors
  logs/attempt_1/eval.log
```

The repository records its checkout commit, dirty status and diff hash, the EXL3
model manifest (`prepare/exl3-manifest.json`), source lock, dataset lock,
dependency locks and launch configs. Before each taskset it records the serving
container's ID, image ID, running state, start time and restart count
(`QWEN_SERVING_CONTAINER`, else the one running container publishing the
`configs/local.toml` endpoint) plus SHA256 of the frozen evaluator inputs; a changed
container or input after the run fails the invocation. That identifies the
running container, not the checkout that built it. Set `QWEN_RECIPE_ID` for
direct-lane runs and preserve deployment image and model-inventory evidence with
any published comparison. A dirty checkout is diagnostic evidence, not a
reproducible release claim.

Inspect the resolved JSON before trusting a run: model, client URL/key-variable,
sampling, taskset filters, harness, runtime image/workdir/resources and rollout
counts must match the intended launch. Inspect several native traces as well.
The wrapper fails on missing prerequisites or any native episode `ok=false`;
a score of zero without an operational error remains a valid upstream result.

Use native per-environment rewards and error records. The upstream evaluation
guide's read-only reward command is:

```sh
jq -s '[.[].traces[] | [.rewards[]? | .score * .weight] | add // 0] | if length > 0 then add / length else 0 end' RUN/traces.jsonl
```

Report task count, rollout count, reward breakdown, operational failures and
sampling limits **per environment**. Do not merge tasks into a synthetic overall
score. Do not omit truncations or compare different output limits as though they
were identical configurations. Raw upstream records remain authoritative.

All runs use `--no-push`. Traces may contain full copyrighted/public task data and
sensitive endpoint output; keep them private by default. Commit only small,
sanitized summaries with complete provenance when explicitly choosing to publish.
Each wrapper invocation starts a new run. Do not edit a saved resolved run for a
new comparison. Advanced native resume must use the recorded working directory,
`uv run --project /absolute/path/to/eval --no-sync`, the provider key variable,
and the same offline dataset paths and runtime locks. Follow the pinned
upstream evaluation guide for that workflow; it is not a separate local replay
implementation.

## Validation

See [validation.md](validation.md) for the actual checks and remaining blockers.
Use `run PROFILE --dry-run` to validate all native configs without a GPU or key.
The runtime benchmark path remains independent: see
[../docs/benchmarks.md](../docs/benchmarks.md).
