# Evaluation integration validation

Validated on 2026-09-13. This is integration evidence, not a published model-quality
comparison or qualification of the concurrently changing serving recipe.
This document covers the current pinned integration in `eval/scripts` and
`eval/configs`. Concurrent historical adapters under `eval/direct` are separate
work and were not audited here. Their dependency metadata was preserved.

## Passed

- Exact Prime Envs and Verifiers source checkouts; locked evaluator installation
  (`verifiers 0.3.2.dev86`); `setup --check` confirms source, dependency and native
  runtime-export identities without changing them. A second, newly created
  temporary virtual environment installed all 112 platform packages from the
  same lock and passed version/import/native CLI help checks; the normal
  environment was not replaced.
- All 17 profile entries passed native configuration validation: the smoke
  ran end to end, and final quick/full/agentic/agentic-full wrapper dry runs
  passed with the actual local dataset routing. All core entries use
  Docker with the tool-free `null` harness; optional entries use `bash`.
- All pinned source datasets downloaded and hash-checked. Native offline loaders
  were exercised with fresh derived caches: AIME24/25/26, I3 Logic, LiveCodeBench,
  GraphWalks. MRCR generation-pinned source files are present and verified.
- Sandbox image built from pinned Python/uv images and native uv script locks.
  Both fixed upstream helper images and `sandbox --check` pass. The image
  also passed native locked script-environment checks with network disabled.
  No serving
  container was stopped, replaced or reconfigured by evaluation.
- Authenticated **AIME24 3 × 1** evaluation completed through Prime Envs against
  `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`. Three native episodes have
  `ok=true`; two calls reached the 8192-token limit. Truncation is retained in
  the official records, not treated as an infrastructure success score.

The successful local run is ignored:
`eval/runs/smoke.1YiECYH9/aime24/aime24--qwen3.8-27b--null--152f0430/`.
Its parent contains provenance. The resolved JSON confirms the intended model,
client URL/key-variable, three tasks, one rollout, C1, temperature 0.6, top-p
0.95, top-k 20, 8192-token budget, thinking enabled, `null`, local Docker,
`/workspace`, 2 CPUs, 4 GB, restricted networking, static one-worker serving
and `push=false`.

The exact sandbox image for that run was
`sha256:aedce3045f6e36ec790d545666ca680cf3b1695fa8e08fccc3baf28b377d56dd`.
The model was the already-running 64K baseline, not the newly edited candidate.
Its observed stock image was
`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`,
with a mounted Nix source overlay. Its serving-recipe commit was not established.
The recorded checkout was dirty. Do not use this smoke run for a release claim
or matched recipe comparison.

## 2026-09-25: MMLU-Pro and the broad lane profiles

- `mmlu-pro 0.1.0` (editable, pinned Prime Envs checkout) was added to
  `pyproject.toml`; `uv lock --offline` added only that package (dependency:
  `verifiers`, already locked) and `uv sync --locked --offline` installed it.
  `setup --check` passes.
- `datasets.lock` pins `TIGER-Lab/MMLU-Pro` at
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, the revision the taskset hardcodes:
  `README.md` (configs metadata), `data/test-00000-of-00001.parquet` and
  `data/validation-00000-of-00001.parquet` (the default config's two splits).
  `data mmlu-pro` downloaded and `data --check mmlu-pro` verified them.
- The native offline loader (relative `TIGER-Lab/MMLU-Pro` link in a fresh
  working directory, fresh Arrow cache, HF offline) selected the 20 seed-0
  shuffled zero-shot tasks, and `run tiny --dry-run` and `run broad --dry-run`
  passed. For every lane taskset the lane's offline selection replay produced
  exactly the resolved config of the native `--dry-run`; the AIME25 selection
  and resolved config are byte-identical to the historical v1 lane's. No model
  endpoint was used; no task or scorer was changed.

## 2026-10-04: audit of the current pins

- `setup --check`, `data --check all` (all seven Hub snapshots and both MRCR
  files) and `run PROFILE --dry-run` for all eight profiles (21 entries) pass.
  Every resolved config names `qwen3.8-27b` at `http://127.0.0.1:18020/v1` with
  `QWEN_API_KEY`, greedy sampling and the profile's budget. `direct/setup --check`
  and `direct/run smoke|quick|full --dry-run` pass.
- The serving-container default `qwen-inference-inference-1` was stale (a
  never-started old container). `run` now records the container named by
  `QWEN_SERVING_CONTAINER` only if it publishes the `configs/local.toml`
  endpoint, and otherwise the one running container that does.
- `sandbox --check` failed ("Sandbox build inputs changed"), so every non-dry
  `run` stopped before its first native command. The recorded image
  `sha256:facda9cb…` was built from the pre-MMLU-Pro `pyproject.toml`/`uv.lock`;
  only those two recipe inputs changed, and neither enters the build context.
  `eval/scripts/sandbox` (in `nix develop`) rebuilt and recorded
  `sha256:5962c155…`. Against `facda9cb…`, all 10,400 non-`.pyc` files have equal
  content except 35 uv HTTP-cache `.http` entries (fetch metadata).
  `sandbox --check` passes. Later lane runs record the new image ID.

## 2026-10-04: hash-locked build tools

- Before: `uv sync` built the editable Verifiers and Prime Envs packages in
  isolated build environments. `build-constraint-dependencies` pinned hatchling,
  hatch-vcs and setuptools-scm by version only. uv fetched them and their
  dependencies without hashes. uv 0.12.1 has no hashed build-constraint form.
- Now: `pyproject.toml` (and `direct/mrcr`, `direct/graphwalks`) lists the build
  tools in a default `build` dependency group and sets `no-build-isolation`.
  `uv.lock` records them with hashes (editables 0.6, hatchling 1.32.0, hatch-vcs
  0.5.0, setuptools-scm 9.2.2, setuptools 84.0.0, packaging 26.3, pathspec 1.1.1,
  pluggy 1.6.0, tomlkit 0.15.1, trove-classifiers 2026.6.1.19). Every locked
  hash matches the PyPI JSON API. uv installs these first and then builds the
  editables in the project environment, so no build fetches an unlocked file.
- Verifiers has a dynamic version. `[[tool.uv.dependency-metadata]]` gives uv
  its static metadata, so the lock does not need a build. uv does not compare it
  with the build. `setup` and `direct/setup` now build the metadata with the locked
  hatchling and fail on any difference (shown with a changed version and with a
  removed requirement).
- `uv lock --offline` changed `uv.lock` only by these additions, the
  dependency-metadata entry and the Verifiers lock metadata. No runtime package
  version changed. A new virtual environment synced offline with
  `--locked`; uv reported "Prepared 9 packages without build isolation".
- `setup --check`, `direct/setup --check`, `run tiny --dry-run` and
  `direct/run smoke --dry-run` pass.
- `pyproject.toml` and `uv.lock` are sandbox recipe inputs, so `sandbox --check`
  failed after this change. `eval/scripts/sandbox` re-recorded the image as
  `sha256:d6a4517c…`. Its RootFS layer list is identical to `sha256:5962c155…`;
  neither file enters the image build context. `sandbox --check` passes.
- `direct/setup` now runs uv with `--directory`: uv 0.12.1 reports the
  dynamic-version lock as stale with `--project` from another directory. It also
  sets `GIT_LFS_SKIP_SMUDGE=1` and uses the Nix `python3.12`.

## 2026-10-04: Dependabot advisories

- urllib3 2.7.0 -> 2.8.0 (GHSA-vxq7-64xx-v4gw, GHSA-8988-9cw3-xx77,
  GHSA-gh4c-6fx4-qh6g) and PyJWT 2.14.0 -> 2.15.0 (GHSA-42vr-xj54-vc7v) in
  `uv.lock`, `direct/graphwalks/uv.lock` and `direct/mrcr/uv.lock`. `uv lock
  --upgrade-package urllib3==2.8.0 --upgrade-package pyjwt==2.15.0` (uv 0.12.1,
  `nix develop`) changed only these two packages and their hashes. No
  `pyproject.toml` changed. The `build` group is unchanged and stays hashed.
- datasets stays 4.6.1 in `direct/`. GHSA-379c-qx7v-6h59 is fixed in 5.0.1, but
  the pinned Verifiers 0.1.15.dev17 declares `datasets>=3.0.0,<4.7.0`.
  `direct/setup --check` runs `uv pip check` and fails on that conflict. The
  advisory is a path traversal through `file_name` metadata in folder-based
  builders (imagefolder, audiofolder, videofolder). The direct runtime does not
  use them:
  - A search of the pinned Verifiers tree (`977e3fc4`, both `verifiers` and
    `verifiers-graphwalks` checkouts), `graphwalks.py`, `mrcr_v2.py` and
    `eval/direct/{run,setup,inspect_dataset.py,prepare_transport.py,configs}`
    finds no `imagefolder`, `audiofolder`, `videofolder`, `FolderBasedBuilder`,
    `file_name` or `save_to_disk`.
  - The one `push_to_hub` is `verifiers/utils/save_utils.py:884`. It runs only if
    `save_to_hf_hub` is true (`utils/eval_utils.py:1134`,
    `envs/environment.py:1102`). All 11 `direct/configs/*/*.toml` set
    `save_to_hf_hub = false` (line 11).
  - GraphWalks: `graphwalks.py:188` calls `load_dataset("openai/graphwalks")`.
    `run` links that name to the snapshot `f338bb26…`. It holds only `README.md`
    and two parquet files, sha256 `54036036…` and `53787943…` in
    `datasets.lock`. The smoke dry run recorded `builder_name: parquet`, 1150
    rows.
  - MRCR: `mrcr_v2.py` does not import `datasets`. It reads the two CSV files
    with `csv.DictReader` (`mrcr_v2.py:113-114`). `datasets.lock` pins their GCS
    generation, size and sha256 (`c6be39bc…`, `fe3b726c…`). `run` checks them
    with `scripts/data --check` before it loads them. The download at
    `mrcr_v2.py:99-102` runs only for a missing file.
- `setup` and `direct/setup` (sync mode) installed the new versions. `setup
  --check`, `direct/setup --check`, `run tiny --dry-run` and `direct/run smoke
  --dry-run` pass.
- `eval/scripts/sandbox` re-recorded the image as `sha256:93ac158c…`. Its RootFS
  layer list is identical to `sha256:d6a4517c…`. `sandbox --check` passes.

## Failures found and fixed during integration

A raw HF snapshot is not enough for a fresh offline repository-name lookup.
Native local dataset paths fix AIME; I3 needs its `logic/` subdirectory and the
local parquet builder's `default` config. GraphWalks uses native local relative
path discovery from its recorded per-run working directory. No tasks, questions,
answers or scorers were changed.

The current AIME TaskData requires a network policy. Verifiers correctly rejects
its execution on the subprocess runtime. Docker preserves that policy. Earlier
failed attempts remain ignored and were not silently retried or counted as model
quality results. A missing key file fails before any model request.

## Repository checks

ShellCheck with `-x` and shell syntax checks pass for the evaluation scripts.
No throughput or capacity figures were remeasured during this refactor.

Nix flake validation passed on a source-only snapshot that excluded ignored
caches, datasets and traces and included the concurrent untracked runtime files.
Git-backed development-shell evaluation also passed. Do not use `path:.` on a
checkout containing private ignored runs.

Full repository checks were run read-only. The inspected tree reported **164
Ruff diagnostics**, **9 ty diagnostics**, and one formatting failure in
since-removed model-preparation code. These concern retained/concurrent runtime
and preparation code, not newly added evaluation Python (there is none). Strict
rules were not disabled and authenticated/vendor inputs were not rewritten for lint.
No repository test suite is configured; no tests were added or represented as
passing. Runtime files were changing concurrently, so these counts identify the
inspected state rather than a promise about later edits.

## Not claimed

Quick/full quality suites and agentic model runs have not been executed. Native
configuration and dataset-loader checks are not substitutes for those runs.
No 64K/240K capacity or model-quality parity claim follows from a 3-task smoke.
Per-environment source counts at these pins include 11060 filtered I3 tasks,
454 filtered LiveCodeBench tasks, 1150 GraphWalks tasks, and MRCR bucket sizes
85 and 141. Full rollout settings exceed the upstream >500-rollout guideline.

To repeat the end-to-end check from a clean checkout:

```sh
nix develop . --no-write-lock-file
eval/scripts/setup
eval/scripts/data aime24
eval/scripts/sandbox
export QWEN_API_KEY_FILE=/absolute/path/to/serving/state/api-key
eval/scripts/run smoke
```

## Opt-in task time and board energy

The measurement path is `eval/scripts/run PROFILE [TASKSET] --measure-power`.
It wraps the same pinned native command without changing tasks, retries, rewards,
sampling, runtime or serving policy. Use an exclusive quiet endpoint and select
the actual serving GPU; no board limit or service is modified:

```sh
eval/scripts/run quick aime24 --measure-power --power-gpu=0 \
  --power-interval-seconds=0.5 --power-max-seconds=86400 \
  --power-max-samples=200000
```

`eval/scripts/run --help` documents the bounds. Collection stops at its duration
or sample bound; evaluation continues, with uncovered time explicitly reported.
Each `nvidia-smi` query has a two-second timeout. INT, TERM and HUP are forwarded
to the native process group; teardown has a ten-second grace period before kill,
and the collector is joined before evidence is written. SIGKILL cannot execute
cleanup. `--dry-run` cannot be combined with measurement. Native exit status and
raw rewards are retained; measurement failure is a warning and a separate status,
never an invented benchmark failure score or zero-joule observation.

The private run's `TASKSET/measurement/` contains:

- `power-samples.json`: selected board, host Unix/monotonic clock anchors, power
  in watts, each query's monotonic start/end and midpoint, and categorized read
  failures. These are host observations, not device timestamps.
- `task-intervals.json`: episode ordinals, operational status, original numeric
  reward components (including fractional scores), error counts, truncation and
  attributable trace-envelope intervals. No prompts or error text are copied.
- `efficiency.json`: sanitized per-environment machine-readable totals and
  denominators. Publish only this summary with separately reviewed deployment
  provenance; keep native traces and detailed records private.

Energy is the clipped trapezoidal integral between adjacent valid observations.
There is no extrapolation, idle subtraction, interpolation across failed reads,
or bridging gaps longer than three sample periods. `observed_joules` describes
only covered intervals; `total_joules` and joules/success are null unless the whole
native-command interval is covered. Coverage, gaps, read failures and collection
limits remain in the evidence. The scope is selected-board energy, not whole-host
or decode-only energy.

`successful_task_rollouts` counts single-trace episodes with both native
operational statuses true and the native weighted reward exactly 1; `ok=true`
alone is not solved. Fractional rewards remain fractional. Repeated rollouts are
not unique questions. Zero successes, unsupported/missing reward denominators,
or an interrupted/failed native CLI produce null efficiency ratios. Total wall
time and energy include startup, scoring, failed attempts, retry/backoff and
teardown, not just successful episodes.

At the pinned Verifiers revision, episodes have no serialized whole-episode
start/end. Only each persisted trace's start and phase spans are available.
Task attribution therefore uses **persisted trace envelopes**, never claims
discarded retry or whole-episode durations, and reports unattributed overhead.
Attribution requires resolved C1, nonoverlapping spans inside the measured run,
and host Unix/monotonic clock drift no greater than 50 ms. Missing/invalid spans
or clock jumps disable attribution explicitly without changing native scores
or whole-run monotonic energy. This does not correct an unobserved clock jump
between samples; query intervals and clock uncertainty are retained.

This path's implementation is not a new measurement result or qualification.
Verification must exercise synthetic linear/clipped intervals, failed-read and
long-gap coverage, zero/fractional/full rewards, missing or torn native traces,
overlapping timestamps, clock jumps, collection bounds and signal cleanup before
using a real isolated GPU run as efficiency evidence.
