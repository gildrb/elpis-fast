# Measurement protocol

**Status:** EXL3 with native DFlash2 speculative decoding (greedy, one sequence,
native context 262144, CQ3 cache) on one RTX 3090. `bash autoresearch.sh` runs the
cold-prefill suite `exl3-native-prefill-ttft-v1` (§2a); the broad suite
`exl3-native-broad-c1-request-v5` (§2) stays selectable with `--suite broad`.

| Protocol | Tasks | Declared power, clock offsets |
|---|---|---|
| v2 | AIME 2025 ×3, MMLU-Pro ×20, I3 Logic ×6, LiveCodeBench ×3 + C1 | 350 W; offsets not declared (stock) |
| v3 | v2 with MMLU-Pro ×10, I3 Logic ×4 (the first tasks of the same native shuffles; v2 did not fit its 2400 s deadline at 250 W) | 250 W; offsets not declared (stock) |
| v4 | v3 tasks | 250 W; core 0, memory −1500 MHz |
| **v5** (since 2026-09-28) | v3 tasks | **350 W; core 0, memory 0** |
| **prefill-ttft-v1** (`exl3-native-prefill-ttft-v1`, since 2026-09-29) | Cold-prefill TTFT ladder 8192 ×3, 32768 ×3, 131072 ×2, 262000 ×1; no tasksets | 350 W; core 0, memory 0 |

- Each version is a new comparison segment with a fresh baseline; numbers do not carry across versions or to `exl3-native-math3-c1-request-v1` (math-only primary).
- `aime25_*` keeps the v1 math definition (same producer, tasks, config and budget): a whole-stack comparison only.
- No lane here is full quality, capacity or promotion qualification.

## 1. Freeze the comparison

Use one RTX 3090, an exclusive quiet endpoint, and exact image, engine patch
manifest, Bend acceptance identity and target/draft inventory hashes. Save
tokenizer identity, sampling settings, actual input IDs, output budget, cache
condition and power policy. Record failures and truncations explicitly;
operational completion is not a successful quality result. Never combine
percentage gains from different prompts or configurations, and never compare
numbers across runner-configuration changes.

Quality uses the frozen **Prime Envs + Verifiers** profiles under `eval/`
(prime-envs `c4d04dfe`, verifiers `ef47b2e9`). `bench/` orchestrates those
unchanged native tasks; it does not define their prompts, graders or a combined
intelligence score. No lane uses an LLM judge. GPQA is excluded: its dataset is
gated and its scorer can fall back to a remote LLM judge.

## 2. Frozen autoresearch lane

`python -m bench.autoresearch --suite broad` (`bench/autoresearch.py`
supervisor/worker, `bench/exl3.py` identity, native taskset and C1 logic; same
environment as `autoresearch.sh`) runs exactly once, in this order:

| Order | Workload | Frozen selection and settings |
| --- | --- | --- |
| 1 | `aime25` | `eval/configs/tiny/aime25.toml` unchanged: 3 native seed-zero shuffled tasks, 32768 output-token budget |
| 2 | `mmlu-pro` | `eval/configs/broad/mmlu-pro.toml`: 10 tasks, zero-shot, 8192 budget; boxed-letter math-verify scoring |
| 3 | `i3-logic` | `eval/configs/broad/i3-logic.toml`: 4 tasks, 16384 budget |
| 4 | `livecodebench` | `eval/configs/broad/livecodebench.toml`: 3 tasks, 16384 budget, official v6 date filter (2024-08-01 through 2025-05-01, as `quick`); hidden tests in the sandbox |
| 5 | C1 depth matrix | Raw-content depths 1024/8192/32768 (±2 tokens), five repetitions each in depth-then-repetition order, 1024 output-token budget, greedy, `top_p` 1, `n` 1, normal EOS, non-streaming |

Every taskset uses one rollout per task, one model call per episode (null
harness), `max_concurrent` 1, the local Docker runtime image pinned by
`eval/.cache/sandbox-image`, native Docker timeouts (setup 600 s, rollout
1800 s, scoring 600 s) and `eval/configs/local.toml` sampling (greedy,
thinking enabled). A taskset profile may set only `sampling.max_tokens` (its
per-call budget, equal to `env.agent.max_output_tokens`); any other sampling
difference rejects the run.

### Operator contract

Main installs a fresh private descriptor at
`/run/user/1000/elpis-autoresearch-operator.json` (uid 1000, mode 0600, regular
file, no symlink) with exactly `schema_version` 1, the full 64-hex
`container_id`, canonical absolute `api_key_file` (0400/0600),
`maintenance_directory` of an already armed guardian window, and a not yet
existing `output_directory`. The supervisor binds the descriptor's inode and
digest, verifies guardian lease/launch-lock/mutex ownership on every poll,
requires the container to be the guardian candidate publishing only
`127.0.0.1:18020`, and enforces `min(2400 s, guardian deadline - 120 s)` over
capture, every workload and admission. The worker runs inside pinned offline
Nix. There is no retry, task reduction, warmup, flush, capacity probe, power
change, deployment or promotion.

Prerequisites: prepared offline `eval/.venv` (including `tokenizers` and the
`mmlu-pro` package), pinned Prime/Verifiers sources, verified AIME25, MMLU-Pro,
I3 Logic and LiveCodeBench snapshots (`eval/scripts/data --check <taskset>`), the
pinned local sandbox image in `eval/.cache/sandbox-image`, host `docker` and
`nvidia-smi`, and a healthy owned EXL3 instance serving `qwen3.8-27b` with
target/draft mounted under `/models`. The server must accept the native client's
identity sampling fields (`top_p` 1, `min_p` 0, frequency/presence penalty 0,
repetition penalty 1) and reject other values.

### Serving identity

Before any workload and again after all of them, the worker captures and requires
byte-identical identity (`identity-before.json`, `identity-after.json`):

- Docker: full container ID, name, immutable image ID, creation and start time,
  PID, restart count, entrypoint/command, public `QWEN_*` environment, network,
  published ports, read-only root and mounts; image labels and layer digests.
- One read-only in-container probe (`docker exec` of the image's
  `/opt/venv/bin/python -I -B`): SHA256 of every file under `/opt/qwen` (baked
  server, launchers, engine patch manifest, Bend artifacts) and
  `/model-preparation`; the installed `exllamav3` package tree and its
  `exllamav3_ext` extension; engine distribution version and install URL; all
  installed distributions; target/draft inventories (every file size; SHA256 of
  each file up to 64 MiB, checked against `prepare/exl3-manifest.json`; weights
  are rehashed by the image's own startup inventory).
- Engine patch manifest `/opt/qwen/exl3-patches.json`: when present, its bytes
  must match image label `io.elpis.exl3.patches-sha256` and every listed
  installed file must rehash to its recorded post-patch SHA256. When absent the
  record is explicit `null` and the label must be absent.
- Bend acceptance identity `/opt/qwen/bend-exl3/identity.json`: when present,
  schema `elpis-exl3-bend-accept/1`, `identity_sha256` must recompute over the
  document's other keys and every listed artifact must match its baked bytes.
  When absent the record is explicit `null`.
- Authenticated `/health` and `/v1/models` (`max_model_len` 262144 is the
  reported limit, not a capacity test).
- Host `nvidia-smi`: exactly one `NVIDIA GeForce RTX 3090`, power limit and
  enforced limit 350 W (declared operating point; never changed here), UUID,
  bus, driver and memory.
- The host tokenizer file behind the target mount, whose bytes must equal both
  the served file and its manifest pin.

### Native tasksets and the primary

Each taskset `<ts>` is invoked directly as
`uv run --project eval --no-sync eval @ eval/configs/local.toml @ <launch>
-o <group>/<ts> --no-rich --no-push <data flags>`, where `<launch>` is its profile
with only the sandbox image replaced by the pin. Offline data loading is exactly
`eval/scripts/run`'s (`eval/README.md`): AIME25 gets
`--env.taskset.dataset-name <snapshot>` and I3 Logic
`--env.taskset.dataset.name <snapshot>/logic --env.taskset.dataset.subset default`,
both from `eval/.sources/prime-envs`; LiveCodeBench reads the owned HF cache whose
`refs/main` must name the locked revision; MMLU-Pro, which hardcodes
`TIGER-Lab/MMLU-Pro` and its revision, runs from `<group>/local-datasets`, where
that relative path links to the verified snapshot, and the taskset module's
hardcoded name and revision must equal the lock entry.

Before any generation, every taskset's evaluator/source/config/data hash closure
is frozen and its native seed-zero selection and resolved config are replayed
offline with the CLI's deep merge (`<ts>.evaluation-inputs-before.json`); the
same record is recomputed after that taskset's run and must be identical.
Admission then replays each taskset from raw files: the resolved config equals
the frozen plan (model, taskset, harness, sampling, budget, runtime); the log
states exactly one `<N>x1` run; exactly the planned tasks in planned order with
matching content hashes; every episode and trace `ok`, complete and scored by
the pinned Verifiers commit; exactly one call per episode with the expected
model, endpoint, wire sampling (including its budget) and a `stop`/`length`
finish; and every trace inside the identity capture window. Any failure rejects
the whole run.

`model_call_output_tok_s` is the **primary**: the completion tokens of every
native model call of all four tasksets divided by the sum of their native
model-call wall intervals (`bench.autoresearch.model_call_observations`). These
intervals cover request send through the fully received response, including
prefill, decode and HTTP; they are not decode-only or GPU time. The primary
therefore weights tasksets by their generated tokens. `<ts>_output_tok_s` is the
same ratio over one taskset's calls, `<ts>_reward` the mean native weighted
reward of its episodes and `<ts>_truncated` the number of its calls with finish
reason `length`, for `<ts>` in `aime25`, `mmlu_pro`, `i3_logic`,
`livecodebench`. Incorrect and length-truncated but operationally complete
graded answers stay in numerators, denominators and rewards.

Replaying the historical run-g7kafqt AIME25 traces through this admission path
reproduces `aime25_output_tok_s` 19714 tokens / 125.18 s = 157.4852 tok/s,
reward 1.0 and no truncation (a v1 run, not a baseline for this protocol).

### C1 whole-request secondaries

Before any generation, the worker sizes each of the 15 prompts from the frozen
`bench/throughput-prompts.jsonl` corpus (deterministic nonce
`[measurement run R of 5 at depth D]`, fixed instruction) to raw-content depth
within two tokens using the served tokenizer bytes on CPU, writes the exact
request bytes, and retains the server's `/v1/chat/completions/render` token IDs.
Raw-content depth is not total templated depth; both counts are recorded.

Each row is one non-streaming `/v1/chat/completions` request. Its wall time runs
from request send through the complete response body. Admission replays every
row from raw files: HTTP 200, one assistant choice, `stop`/`length` finish
(`length` only at the 1024 budget), `prompt_tokens` equal to the rendered ID
count, consistent totals, and sequential non-overlapping requests.

`c1_request_tok_s_<depth>` = sum of the five rows' completion tokens / sum of
their request wall times. It is **whole-request output throughput including
prefill**, not TTFT and not decode-only. TTFT and committed decode rates are
unavailable on this transport and are never reported or estimated.

### Speculative acceptance

When every C1 row's usage carries `exl3_spec` = `{rounds, committed}` (native
verify rounds and tokens committed by them; `rounds <= committed <=
min(completion_tokens, 8 * rounds)`), `spec_accept_length` = total committed /
total rounds over all 15 C1 rows is reported. It is absent when the server does
not report the field; partial reporting rejects the run. Native taskset traces do
not retain this field.

### Time budget

Expected wall time is about 25 minutes, inside the 2400 s hard deadline:
AIME25 ~150 s (measured on g7kafqt: 125 s of calls), MMLU-Pro ~310 s, I3 Logic
~210 s and LiveCodeBench ~290 s including sandbox scoring (estimates: about
2000/5000/9000 completion tokens per call at 160–185 tok/s plus ~4 s of Docker
episode overhead), C1 ~340 s (measured) and ~190 s of identity captures, Nix,
offline selection replays, input hashing (LiveCodeBench's 4.49 GB source is
hashed and loaded before and after its run) and evaluator startups. That leaves
roughly 900 s for longer outputs; output-bound worst cases (every call reaching
its budget, about 2600 s of generation alone) exceed the deadline. Such a run
is rejected, never shortened or retried.

Measured end to end: #67 (v4, 250 W) 1,649 s; #68 (v5, 350 W) 1,312 s.

### Admission and output

Artifacts: `supervisor.json`, `benchmark.json` (frozen workload and
`workload_sha256`), `identity-before.json`, `identity-after.json`,
`<ts>/{provenance,<ts>}` for each taskset (plus `mmlu-pro/local-datasets`),
`c1/{plan.json,depth-D-rep-R/}`, `logs/`, `sources/`, `admitted.json` (all
metrics, per-taskset/per-call/per-row records and the retained evidence hash
closure) and `measurement.json`. Only a complete admitted run prints `METRIC`
lines, in this order: `model_call_output_tok_s`; `<ts>_output_tok_s`,
`<ts>_reward`, `<ts>_truncated` per taskset in lane order; the three
`c1_request_tok_s_*`; optional `spec_accept_length`; and `elapsed_seconds`
(whole command, not a per-lane clock). Any rejection writes `failure.json` (and
`worker-failure.json` from the worker) and exits nonzero.

Not measured by this lane: TTFT, committed decode throughput, power/energy and
262144-token capacity.

## 2a. Cold-prefill suite (`exl3-native-prefill-ttft-v1`)

`bash autoresearch.sh` = `python -m bench.autoresearch --suite prefill` (`bench/prefill.py`). The suite is required, with no default. Operator contract, serving identity, guard checkpoints, 2400 s deadline and evidence retention are those of §2.

| Depth (raw content, ±2 tokens) | Repetitions | Rendered prompt tokens (live plan, 2026-09-29) |
|---|---|---|
| 8192 | 3 | 8244 |
| 32768 | 3 | 32821 |
| 131072 | 2 | 131122 |
| 262000 | 1 | 262052 (+ 32 ≤ 262144) |

| Item | Frozen definition |
|---|---|
| Row content | `[prefill measurement R of N at depth D]`, newline, corpus prefix, blank line, the C1 instruction; sized by binary search on the served tokenizer (no specials) |
| Corpus | the 8 `bench/throughput-prompts.jsonl` prompts joined by a blank line as in C1, repeated to ≥ 1.05 × 262000 tokens (382 passes, 275803 tokens); short coverage rejects |
| Requests per row | TTFT: `max_tokens` 1; then continuation: same messages, `max_tokens` 32. Greedy, `top_p` 1, `n` 1, non-streaming, concurrency 1, ascending depth, repetitions consecutive |
| Freeze | both request bodies written and rendered (`/v1/chat/completions/render`) before any generation; rendered ID count and sha256 recorded; both renders must be identical |
| Clock | monotonic, request send through complete response body |
| Row admission | TTFT: `prompt_tokens` = rendered, `completion_tokens` 1, totals consistent, finish `length`/`stop`, `exl3_spec` per the C1 rule. Continuation: `prompt_tokens` = rendered, 1–32 tokens, `stop` or exactly 32. No overlapping requests; every request inside the identity window |
| Recorded per row | TTFT and continuation text (`content`, `reasoning_content`), continuation text sha256, `exl3_spec` |

| METRIC (print order) | Definition |
|---|---|
| `prefill_tok_s` (primary) | geometric mean over the four depths of `prefill_tok_s_<d>` |
| `prefill_tok_s_<d>` | sum of native `prompt_tokens` / sum of TTFT wall seconds at depth d |
| `ttft_s_<d>` | mean TTFT wall seconds at depth d |
| `reuse_request_s_<d>` | mean continuation wall seconds at depth d (informational) |
| `elapsed_seconds` | whole command, as §2 |

| Guarantee / limit | Detail |
|---|---|
| Cold prefill | the leading nonce makes each row's first 256-token KV page unique, so no page of any TTFT request matches an earlier request of the ladder; no flush, no warmup |
| Deterministic nonces | a server instance that already served this ladder would reuse pages; run on a freshly started candidate |
| TTFT scope | streaming TTFT is unavailable (buffered SSE): TTFT = wall of a 1-token non-streaming request = prefill + first verify round + HTTP |
| Reuse | continuation reuse is expected, not observed: usage has no cache telemetry |
| Not measured | quality, reward, decode throughput, power/energy, 262144-token capacity |
| Comparability | new segment; not comparable with v5 or any earlier protocol |
| Time budget | estimate from observed cold TTFTs (8K ~6 s, 32K ~25 s, 262K ~445 s): ~1,000–1,200 s including ~45 s of CPU planning and identity captures, inside the 2400 s deadline; an overrun is rejected, never shortened |

## 3. Power

| Since | Declared cap | Core / memory offset | Fan curve (host, CoolerControl) | Why |
|---|---|---|---|---|
| 2026-09-25 | 250 W | 0 / −1500 MHz (memory from 2026-09-26) | 100 % at 75 °C | 350 W with that curve was too loud |
| 2026-09-28 (v5) | 350 W | 0 / 0 (stock) | ≤ 80 % up to 84 °C, 100 % at 90 °C; above 83 °C the card lowers its own clocks | quiet at 350 W; 350 W measured 0.492 vs 0.463 tok/J at 250 W (`cs5`, §6) |

Memory offset at 350 W (RoundBench, tree3s, C1 1K / 8K, 12 paired reps per arm, 2026-09-28):

| Offset | Δ tok/s vs −1500 | Δ tok/J vs −1500 | SM MHz mean (1K) | Bit-exact | Xid |
|---|---|---|---|---|---|
| **0** | **+5.51 ±0.16 / +5.21 ±0.11 %** | **+6.20 ±0.69 / +6.57 ±0.54 %** | 1,473 | yes | 0 |
| −500 | +3.72 / +3.62 % | +4.42 / +4.55 % | 1,508 | yes | 0 |
| −1000 | +1.80 / +1.72 % | +1.66 / +2.27 % | 1,537 | yes | 0 |
| −1500 | ref: 27.15 / 27.84 ms per round | ref: 0.429 / 0.357 | 1,557 | ref | 0 |
| −2000 | −1.84 / −1.73 % | −2.57 / −2.11 % | 1,582 | yes | 0 |

- At 350 W the round is memory-bandwidth sensitive: memory clock beats SM clock. At 250 W the reverse held (§6).
- Positive offsets (overclock) are not used: stock clocks only (user, 2026-09-28: stability first). GDDR6X temperature is not readable on this card (NVML: not supported).

Core offset at 350 W, memory 0 (RoundBench, tree3s, C1 1K / 8K, 10 paired reps per arm, 2026-09-28):

| Offset | Δ tok/s vs 0 | Δ tok/J vs 0 | SM MHz mean (1K) | Bit-exact | Xid |
|---|---|---|---|---|---|
| **0** (stock) | ref: 25.81 / 26.55 ms per round | ref: 0.452 / 0.378 | 1,492 | ref | 0 |
| +60 | +0.49 ±0.14 / +0.62 ±0.14 % | +0.98 ±0.68 / +0.58 ±0.55 % | 1,530 | yes | 0 |
| +120 | +0.92 ±0.68 / +1.03 ±0.19 % | +1.73 ±1.07 / +1.59 ±0.65 % | 1,557 | yes | 0 |
| +150 | +0.79 ±0.48 / +0.85 ±0.38 % | +2.22 ±0.78 / +1.61 ±0.86 % | 1,583 | yes | 0 |

- Not adopted: stock clocks only (user, 2026-09-28: stability over ≤ 1 % tok/s; +225 MHz faulted with Xid 109 at 250 W).

- Applied by the host (`nvidia-quiet-power-limit.service`, boot and resume); the lane checks it, never changes it.
- Segments at different caps are not comparable.
- Bounded cap windows (tok/J sweeps): host `nvidia-power-window WATTS SECONDS` (250-350 W, ≤ 2 h, auto-restore).
- Containers never touch host power or fans; tok/J needs watts, clocks, temperature and throttle reasons over a stated interval.

## 4. Engine state at the end of segment 4 (2026-09-25, 350 W)

**Best kept configuration:** candidate image `qwen-inference:exl3-cand-g7kafqt`,
`sha256:f2e72ec7c9b65aa0478119b7be980763162ed7837b1f2288ece6d46abbc795e7`. It was built
from this commit's `patches/` by `bash docker/build-exl3.sh candidate-ext`. The live
deployment (see [docker.md](docker.md)) is unchanged and still serves the promoted image
`fca4c263`; nothing was promoted.

- **exl3 series:** 0001 greedy batched verify, 0002 host overlap, 0003 draft graph,
  0005 DFlash2 reference draft block mask (window (W-1, W-1), bidirectional block).
- **exl3-ext series,** each patch after the one before it:
  - 0001–0004 GDN verify/commit;
  - 3001/3002 attention dequant and GQA split;
  - 5001 GDN small fusions;
  - 2001 M ≤ 16 GEMM;
  - 6001 draft graph;
  - 7001 norm/residual fusion;
  - 3004 attention partial bounds;
  - 3005 row-invariant split;
  - 9002 int4 draft head;
  - 8201 persistent fused MLP;
  - 3003 CUDA verify attention over fixed absolute 512-token chunks;
  - 5101 fused GDN front end, recurrence and norm;
  - 2102 grouped m16 qkv(+z) projections;
  - 8202 persistent layer tail (o/out-proj, residual, norm and MLP in one kernel).

Primary `model_call_output_tok_s` of the kept runs (each run is one full lane of
the earlier protocol `exl3-native-math3-c1-request-v1`, math-only primary; not
comparable with the broad v2 lane above, which needs its own baseline):

| Run | Image | Primary tok/s | tok/J | Note |
|---|---|---|---|---|
| #43 | g7n | 131.62 | 0.409 | segment baseline |
| #45 | g7j | 132.50 | 0.408 | 3004/3005 correctness, 9002 |
| #47 | g7km | 137.47 | 0.426 | 0005, 8201 (bit-exact) |
| #48 | g7kma | 141.73 | 0.432 | 3003 (numerics change) |
| #50 | g7kafq | 149.06 | 0.460 | 5101 (bit-exact), 2102 (numerics change) |
| #53 | g7kafqt | 157.49 | 0.494 | 8202 (numerics change) |

Numerics-changing keeps change the greedy text and with it the math call lengths.
Part of the primary gain from #48 on is a shorter long call, not faster steps. The
step-time traces below are text-independent.

Verify-step time from the CUPTI kernel traces
(`evidence/kernel-trace-N/trace.log`, unprofiled median). The floor is the
Bend-proven byte count of one round (`bend/roofline.bend`, 9002's smaller draft
head subtracted) at 875.6 GB/s:

| Image | Power | Step ms at depth 107 / 8190 / 32728 | Share of DRAM floor |
|---|---|---|---|
| g3 | 280 W | 35.1 / 38.0 / 45.7 | 51 / 47 / 40 % |
| g7n | 280 W | 30.9 / 32.6 / 36.7 | 58 / 55 / 50 % |
| g7j | 350 W | 28.6 / 30.1 / 34.3 | 61 / 59 / 53 % |
| g7kafqt | 350 W | 26.9 / 27.2 / 30.3 | 65 / 65 / 60 % |

The largest remaining losses per step at short depth:
- the MLP GEMMs (~78 % of DRAM bandwidth);
- the GDN projections (~64 %);
- ~1.3 ms of GPU idle between graphs;
- ~2.5 ms of latency-bound small kernels (GDN recurrence, norms, attention).

**Gates held for every kept change:**
- Bend gate green.
- The patch's Bend differential against the shipped source where the module has one
  (`bend/*_diff.py`; 5101 is checked by its quoted source fragments instead).
- Invariance: `ops/autoresearch/invariance.py run`. Target ids under the capped and
  all-wrong draft arms must equal the normal arm on all 15 parity cases, so the
  output is independent of the draft. Bit-exact candidates must also have ids
  identical to their base.
- tiny-math 3/3.

**Open items:**
- 8202 schedule laws: the baked header is byte-identical to
  `TAIL_M16_SCHED_TABLE.bend` output, but four laws were still unproven at the end of
  segment 4. They are proven and wired in segment 5 (see §5).
- Measured and dropped:
  - device-side acceptance with speculative next draft (+0.35 %, flat);
  - one whole-target CUDA graph (−4.2 %);
  - draft window > 2048 (8192: −28 % tokens per round);
  - L2 prefetch into inter-GEMM windows (within noise).

`ops/autoresearch/` is a snapshot of the operator scripts used for this segment.
They are run from `/tmp`: copy them back there, `invariance.py` to
`/tmp/kernel-work/Invariance/` and `kernel_trace.py` to `/tmp/kernel-work/KernelTrace/`.
- `mkcand.py` and `gpu-window.sh` precreate and run guarded GPU windows around the
  retained guardian.
- `build-one.sh` builds a candidate from the committed series plus extra patches.
- `ar-serve.sh` and `ar-when-built.sh` open the timing window for `bash autoresearch.sh`.
- `gpu-inv.sh` and `inv-compare.py` run the invariance gate.
- `ar-energy.py` and `step-report.py` give per-call tok/J and step efficiency.

Launch every script that owns a GPU window through `detach.sh`. A caller that is
killed mid-window leaves the guardian dead, and its library then refuses recovery.
In that case restore by hand, as the guardian would: stop the candidate, start
container `b5e51bc1…`, and require an authenticated `GET /v1/models` of 200.

## 5. Segment 5: broad lane (protocol `exl3-native-broad-c1-request-v2`, 350 W)

| Run | Image | Primary tok/s | tok/J | aime25 / mmlu-pro / i3-logic / lcb tok/s | Rewards | Note |
|---|---|---|---|---|---|---|
| #54 | g7kafqt | 137.39 | 0.426 | 158.34 / 132.07 / 151.37 / 117.15 | 3/3, 12/20, 2/6, 1/3 | baseline |
| #55 | c3006 | 139.54 | 0.431 | 152.83 / 129.65 / 155.51 / 118.14 | 3/3, 13/20, 2/6, 1/3 | 3006 (numerics change) |
| #56 | cs5 | 144.13 | 0.444 | 158.90 / 139.36 / 157.01 / 121.15 | 3/3, 12/20, 1/6, 1/3 | 8204, 9003b, 3007, 2105, 5106 |

3006 replaces 3003's fixed 512-token chunks in the verify attention split with
strided absolute 64-token tiles and one partial slot per split CTA
(`bend/attn_stride*.bend`).

**Judging numerics-changing keeps on this lane.** A numerics change alters the greedy
text, and with it the call lengths and depths of every taskset. On #55, aime25 wrote 60 %
more tokens than on #54, so per-taskset tok/s moves with the text mix and not only with
step time. The taskset traces carry no round counts. Step time therefore comes from the
C1 rows by regression: per depth, fit wall = intercept + rounds × step over both runs'
five rows, with one shared intercept (prefill plus per-token work, since every row
commits 1024 tokens) and one slope per image. #54 → #55: 27.55 → 26.82 ms at 1K
(−2.6 % ± 0.3), 26.30 → 26.34 ms at 8K (+0.2 % ± 0.1), 31.18 → 30.32 ms at 32K
(−2.8 % ± 0.4). These match 3006's component harness (−0.59, −0.08, −0.97 ms per round).

**#56 (cs5).** Five patches on c3006:
- 8204: the 8201/8202 kernels issue their input loads before the weight-ring prologue and
  publish phase-1 group completion once per block after its last flush, instead of with
  a fence at every mid-slice group boundary. Bit-exact.
- 9003b: the draft's q/k/v and K/V-refresh projections (and fc at 3-8 rows) run on the
  grouped m16g kernel. Draft numerics only; target text is draft-independent.
- 3007: deinterleave + RoPE + quantized cache append as one kernel, and the output gate
  inside the attention combine. Bit-exact.
- 2105: the m16g split-K partition weights each SM's second CTA at 0.91 of the first
  (the second CTA streams ~10 % slower). Numerics change; error vs fp32 no worse.
- 5106: the GDN verify kernel's b/a GEMV splits K across all 16 warps (one load round
  instead of five serial ~1 µs rounds). Numerics change; b/a error vs fp64 lower.

C1 regression against #55: 26.57 → 26.15 ms at 1K (−1.6 % ± 0.2), 26.31 → 25.81 ms at 8K
(−1.9 % ± 0.1), 30.20 → 29.86 ms at 32K (−1.1 % ± 0.5). Invariance 45/45. The primary's
+3.3 % is larger than the step gain because the text mix moved again. Rewards differ
from #55 only by single-task flips in both directions (i3-logic `cipher` went 0 → 1 → 0
over #54-#56; `numbrix` already hit the 16384 budget on #55), which a numerics change
causes at these sample sizes.

**Bend coverage after #56.** `bend/mlp_m16_defer*` (8204's deferred publish) and
`bend/attn_pre*` (3007's fused pre-attention kernel) are wired; their differentials and
`attn_stride_diff`, `tail_m16_sched_diff`, `mlp_m16_sched_diff` and `gemm_m16_group_diff`
pass on the #56 tree. Two gaps remained after #56 and are closed since:

- 2105's weighted partition: `bend/gemm_m16_wpart*` (22 laws) proves W, start and the
  closed-form owner generically in (w0, w1, nsm, G, total). The owner is the largest block
  whose start is at most x; ownership is exactly once; equal weights reproduce v2; and the
  host guard, contributor slots and `max_contrib` are bounded. The served GDN and attention
  bundles at (100, 91) satisfy the guard and fit the launch, with `max_contrib` 7 and 8.
  `bend/gemm_m16_wpart_diff.py` quotes 84 lines of the 2105 patch and matches the Bend
  table byte for byte for 8 configurations. Those include 9003b's draft shapes at (1, 1),
  so it also serves as the draft-shape differential. One mutation (`<` → `<=` in the owner
  threshold) survives because it is equivalent: an exhaustive comparison of 65.7 M owner
  evaluations finds no difference.

5106's K-split GEMV is covered since: `bend/gdn_ba_ksplit*` proves that every half2
element of every b/a task is multiplied once, by one (warp, lane, iteration). The per-task
sum order (lane chain, shuffle tree, warps 0-15, bias) is fixed and independent of the
number of rows. The reduction reads only partials completed before the barrier, and more
than 8 rows or K > 5120 falls back to 5101's one-warp-per-task GEMV (still described by
`gdn_fast`'s `task_exactly_once`). `bend/gdn_ba_ksplit_diff.py` quotes the served kernel
lines and matches the Bend table byte for byte (801 lines).

**Differentials.** `bend/attn_chunk_diff.py` quotes 3003's chunk loop, which 3006
removes, so it applies only to trees before 3006. The attn_chunk laws stay in the gate as
the model of the removed partition, as attn_split did when 3003 replaced it.
`bend/attn_stride_diff.py` is the differential for the current tree.

**8202 schedule laws.** `bend/tail_m16_sched*.bend` are wired into `LAWS.bend`/`PROOF.bend`.
The four laws left open in segment 4 (`mc0_served`, `f0_partials_once`,
`norm_reads_partials`, `norm_cells_once`) are proven. `mc0_served` uses a scaling lemma so
the checker evaluates the owner bound at total 960 and G 41 instead of the served 3840 and
164. `bend/tail_m16_sched_diff.py` extracts the baked `exl3_tail_m16_sched.h` from the
committed 8202 patch and compares it byte for byte with the Bend table output.

**Measured and dropped in this segment:**
- 5102, the GDN commit replay folded into the next verify. Bit-exact, but the GDN part of
  a round grew 0.5-0.75 ms (a 3 MB state write per layer inside a latency-bound 48-SM
  kernel), against 0.49 ms of replay removed.
- 2103, the m16g projection without grid barriers. Bit-exact but slower: 79 against
  70 µs per GDN launch. Its slow blocks are the ones whose slice crosses a group
  boundary, where 2103 added a mid-loop fence. It is not SM speed: blockIdx lands on the
  same SM in every launch, and a plain stream of equal contiguous slices runs every SM
  within 2.5 % of the median at 811 GB/s. In 2102 the loop itself streams at ~95 % of the
  DRAM rate; the time goes to the kernel start (~8 µs of launch skew and the grid barrier;
  2104 below shows reordering the input loads does not shorten it), the finish (~5 µs) and
  the launch ramp (~5 µs).
- 8203, dataflow counters instead of the 8202 tail's barriers A/B/C. Bit-exact but
  +1.6 µs per layer. Its stamps show where the tail's time goes: 217 µs per layer against
  a 170 µs DRAM floor, with block spreads of 7.7 / 30 / 20 µs at the ends of the o_proj,
  gate+up and down loops.
- 2104, m16g start reorder (activation and Hadamard-scale loads before the weight-ring
  prologue, counter finish instead of the second grid barrier). Bit-exact, no gain: the
  input transform is still ready 8.4 µs after launch.
- 5103, GDN verify split over v-columns (192 blocks). Bit-exact; faster only at 1-2 rows,
  slower from 4 rows on.
- 5105, GDN verify prologue reorder. Bit-exact; only issuing the state loads after the
  conv helps (−0.65 µs per launch at 8 rows). The fix for that kernel was 5106.
- 5104, GDN commit replay from the verify's stored v′. Bit-exact, but the verify's extra
  stores (+29 µs per round) cancel the replay saving at ~3.4 committed tokens per round.
- 9004, streaming kernel for the draft's dynconv projections. Correct per call, but it
  packed its weight copy inside the loader's deferred-load bracket, before the weight was
  read, so the draft ran on garbage: acceptance fell from 4.82 to 3.48 tokens per round
  while the target text stayed identical. Fixed, it saves only 11 µs per round at the
  draft's fixed 8 rows, so it is not used. A component harness must load weights through
  the served path and gate the draft's end-to-end acceptance.
- 8205, the same slot weighting for the 8201/8202 kernels. Correct, and the per-block
  stamps move barrier D ~6 µs earlier per tail launch, but the 64-layer chain measured
  +1.2 µs per launch at every weighted setting. Not kept until that gap is explained.

## 6. Segment 6: 250 W, protocol `exl3-native-broad-c1-request-v3`

The declared power limit is 250 W (the user's host policy; 350 W was too loud). v3 is v2
with mmlu-pro cut to 10 tasks and i3-logic to 4, the first ones of the same seed-0
shuffles, so the run fits its deadline at the lower clock. Nothing here compares with
segment 5.

| Run | Image | Primary tok/s | tok/J | aime25 / mmlu-pro / i3-logic / lcb tok/s | Rewards | Note |
|---|---|---|---|---|---|---|
| #57 | c0 | 101.46 | 0.411 | 114.69 / 94.12 / 116.15 / 83.87 | 3/3, 0.8, 1/4, 1/3 | baseline (the cs5 stack) |
| #58 | cs10 | 106.13 | 0.432 | 116.24 / 103.83 / 119.65 / 87.93 | 3/3, 0.8, 2/4, 1/3 | 8205b, 9005c, 3010, 3011, 2106, 2107 |

tok/J for both rows comes from the guardian's 5 s power samples of each window (the
250 ms sampler was down during #58); the same method gives #57's earlier 0.410.

**The 250 W regime.** At 250 W the card sits at the cap for the whole run and serves at
~0.9-1.05 GHz. A kernel trace of the same image at 250 vs 350 W (kernel-trace-9 vs 8):
the round grows ×1.27; compute-bound kernels (attention split, norms, GDN conv) ×1.32-1.42;
the m16 weight-streaming loops ×1.21-1.24; the int4 draft head, DRAM-bound, ×1.04. The
m16 hot loops turn issue-bound: ~205 instructions per 512 B warp-iteration need ~1.19 GHz
to keep DRAM busy. Replacing the trellis decode with a trivial one saves ~7 ms of a
~36 ms round. At the cap, time tracks energy per round (~9 J): removing idle time barely
helps (a 3.4 ms per-round sleep costs 1.9 ms; dropping the draft-id host sync, 0).

**#58 (cs10).** On c0:
- 8205b: weighted slot partition for the 8201/8202 kernels. Not bit-exact (corrected 2026-09-29; this line said "Bit-exact"): the default non-uniform weights change the split-K summation order (8205b's own note); fp16 runs stay 64 values, slot sums fp32. Per that note, uniform weights (`EXL3_M16_WEIGHTED=0`) are 8204 bit for bit.
- 9005c: pruned int4 draft head (896 of 1940 blocks, adaptive switch-back). Draft only;
  target text is draft-independent. `bend/draft_head_idmap*`.
- 3010: CUDA prefill attention that sums Q·Kᵀ and P·V in fp16 over 32-value spans, flushed into fp32 (corrected 2026-09-29; this line said "fp16-accumulated PV"). Numerics change: teacher-forced
  KL below the chunk-size noise floor at 32K and 131K. At 250 W a 262136-token prompt
  prefills in 537 s instead of 722 s. Attention alone is 1.55-1.65× faster, under the
  pre-registered 1.8× bar; kept on the time-to-first-token result.
- 3011: wide-tile fp16-accumulate GEMM route for prefill linears, bit-exact, −21.6 % per
  2048-row chunk.
- 2106 / 2107: instruction diets of the m16 hot loops (loop bookkeeping as masked
  offsets and countdowns; trellis bit extraction as byte permutes; per-lane load
  addresses hoisted). Bit-exact. Tail 211-226 → 201-202 instructions per iteration;
  64-layer tail chain −13.3 µs per layer at the cap.

C1 regression against #57: 35.27 → 33.82 ms at 1K (−4.1 % ± 1.5), 37.92 → 32.99 ms at 8K
(−13.0 % ± 7.3); 32K is dominated by its prefill intercept (3010/3011) and not separable.
Invariance 45/45. Rewards differ by one i3-logic task (a numerics change via 3010's
prefill).

**Bend coverage (2.0.29).** Wired with this keep: `bend/m16_wsched*` (8205b),
`bend/draft_head_idmap*` (9005c), `bend/pattn_sched*` (3010), `bend/hgemm_wide*` (3011),
`bend/m16_diet*` (2106: iteration skeleton, ring/x offsets, fold and chunk tests, issue
countdowns and phase-exit cursor equal to the undieted kernels'; no out-of-ring offset,
every weight tile issued once) and
`bend/m16_diet2*` (2107: permute extraction equals the shift reference for every offset;
hoisted load addresses equal the per-iteration ones).

5108 (cs11, #61: 109.65 vs #60's 107.03) rewrites the commit replay's input loader: 16-byte
gathers of only the k and v elements the commit reads, instead of 3S dependent 2-byte loads
per thread that also fetched the unused q. Bit-exact (48-layer state hashes at 1-8 steps,
e2e ids). The replay kernel goes from 620 to 402 µs at 8 committed tokens and 490 to
396 µs at 5; it is unchanged at 3 or fewer. `bend/gdn_replay_gather*` proves, about its
Bend model, that the loader stores exactly the cells the commit reads, each once, from the
same conv_out elements, widened to the same fp32 bits, and that EXL3_GDN_REPLAY_GATHER
parses as 0 or 1 (default 1). `bend/gdn_replay_gather_diff.py` checks that the kernel
source still contains the 18 transcribed expressions (conformance of the transcription,
not a proof of the CUDA code).

2113 (cs12, #63: 111.85 vs cs11's 109.65 / 109.81, identical accept length and rewards) adds
`discard.global.L2` of the split-K partial-slot lines after each finish has
read them for the last time: m16g, the 8201 draft MLP (pair and down finishes) and the layer
tail (F0, pair and down finishes). Once summed, those lines are dead, but L2 would otherwise
write them back to DRAM. Nothing arithmetic moves, so the change is bit-exact. The bx2113
check (PASS) compared on and off bit for bit:
- all 64 layers of tail, 8201 and m16g, plus the draft m16g shapes;
- every row count 1-8;
- back-to-back launches;
- 64-layer graph chains captured with the discard on and replayed 30 times.

RoundBench on − off, 4 reps in fresh processes: −0.496 [−0.608, −0.384] ms per round at 1K
and −0.396 [−0.717, −0.075] at 8K in normal mode, −0.506 / −0.378 in wrong0. Ids and
committed tokens per round are identical. `EXL3_SPLITK_DISCARD` is 0 or 1 (default 1,
anything else refused); `torch.ops.exl3_m16.discard` sets it before graph capture.

`bend/m16_discard*` proves the following about its Nat model of the discard loop, the
finish loads, the contributor stores and a per-line event timeline:
- each finish discards exactly the 128-byte lines of its own chunk in the slots it summed;
- every element it loads lies in those lines;
- a line and a slot name one job (rows < 16, chunks < 4, contributors < mc, the workspaces
  laid out apart);
- every cell of such a line is stored again by the next launch's contributors before any load;
- the loads come before the warp barrier and the discard, and each load observes its own
  launch's store (or its slot-0 sum) with the discard on or off;
- the switch parses to 0 or 1, default 1.

Premises, not modelled: the kernels' synchronisation (grid.sync, the cnt1 counters,
`__syncwarp`, stream order) orders the steps as the model lists them; PTX discard semantics;
the 128-byte workspace alignment (a host TORCH_CHECK); and nc ≤ mc from the partition
models. `bend/m16_discard_diff.py` checks that the patched sources still contain the 39
transcribed expressions and one discard call per finish (m16g 1, 8201 2, tail 3). That is
conformance of the transcription, not a proof of the CUDA code.

**GPU clocks at 250 W.** Core clock offsets do nothing here: the card already runs at the
bottom of its voltage curve (+75/+150 MHz: no change; +225 MHz: Xid 109 fault). A lower
memory clock moves watts to the SMs: −1000 MHz cut the round by 0.77-0.90 ms (~2.3 %),
bit-exact, SM clock 910 → 945 MHz. Not part of #58. It is host policy since 2026-09-26 14:10
(the NixOS power-limit service applies it at boot and resume), and protocol
`exl3-native-broad-c1-request-v4` (the v3 tasks) declares it: the lane reads the core and
memory offsets through NVML and refuses to measure unless they are 0 / −1500 MHz: a second
sweep on cs10 (reference −1000) gave −1500 a further −0.25 to −0.46 ms per round (two of four
intervals clear of 0) and −2000 nothing beyond −1500, all bit-exact. v4 is a new segment; v3
numbers do not carry over.

**Measured and dropped at 250 W:** 8208 (no tail start barrier; +3.7 µs per layer); the GDN
conv and b/a move into the m16g epilogue (probe: the remaining core is 23 µs per call,
realistic gain 0.2-0.4 ms per round); fp16-accumulated PV in decode attention (0.2 % at
8K); dequantizing the 3-bit cache once per round (11× the traffic); the draft-id handoff
without host sync and graphing the draft preamble (idle removal gives nothing at the cap);
producer-aligned split-K for the MLP tail; the 2109 lm_head instruction trim (−1.2 %).

**8-row dynamic tree verify (ext 9008 / 3012 / 5109 / 3013, exl3 0006; `EXL3_TREE`, default on).**
The same 8 verify rows carry a draft token tree instead of a 7-token chain. The anchor is row 0, and the 7 nodes are chosen best-first
by calibrated cumulative probability over the DFlash2 selector lattice (frozen constants, deterministic tie-break).
The tree is built on the GPU (9008). The target verifies each row at position base + depth: attention folds in the
row's ancestors in logical order (3012), and GDN runs the conv window along the ancestors and the recurrence from the
parent state (5109). The proven tree acceptance commits the maximal matching root path under the chain rules (eos,
budget, checkpoint). The commit replays GDN along the path, moves the path's CQ3 K/V rows to base + depth (3013), and
refreshes the draft cache from the path rows. `EXL3_TREE` and the test hook `EXL3_TREE_FORCE_CHAIN` parse strictly as
0 or 1. Unset, the tree is on wherever it is supported (single-sequence DFlash2 with the int4 head, window 7, GDN target),
otherwise off; `EXL3_TREE=1` on an unsupported configuration refuses to start.

Verification (tree3 image, 250 W / mem −1500, 2026-09-27; `/tmp/gpu-queue/done/0070a-tree3-rungs.out`):
- per slice, bitwise GPU differentials:
  - 9008: builder mode 0 equals the greedy walk on 14,301 rounds;
  - 3012: all 5040 tree shapes, every row equal to the chain kernel on the row's path; CHAIN equals cs12;
  - 5109: T1-T5; CHAIN equals cs12 including the conv state;
  - 3013: T1-T4.
- rung 2: the forced chain (`EXL3_TREE_FORCE_CHAIN=1`) equals cs12 on the 17 alpha-2032 prompts (lane ×15, C1 1K/8K, capped at
  4096 / 1024 tokens): the same ids, every round's (position, count), every round's drafted ids, and the same usage.
- rung 3: the dynamic tree produces the same ids and finish reasons as cs12 on those prompts.
- rung 4: lane-mix committed tokens per round 4.675 vs 4.309 (+8.48 %; the lattice simulation gave +8.53 %).
  Per task: aime +5.7, i3-logic +5.9, livecodebench +12.3, mmlu-pro +9.5, C1 1K +12.9, C1 8K +13.7 %.
- rung 5 (RoundBench, tree3s image, `sw` payload, off vs tree): ms per round is unchanged at 1K (34.287 vs 34.229, Δ −0.059,
  95 % CI [−0.310, +0.192]) and 8K (35.031 vs 35.014); committed tokens per round go up by 14.7 % (1K) and 14.1 % (8K).
- rung 6 (lane, tree3s `sha256:ec9751b0…`, #67): **122.33** vs cs12's 111.85 (+9.4 %), 0.495 vs 0.452 tok/J, accept length
  3.933 vs 3.521. Rewards and truncation counts match. All 85 completion texts (lane traces, C1 responses) are byte-identical
  to cs12's.

Proof boundary. `bend PROOF.bend` covers:
- the Bend tree-acceptance leaf and descriptor derivation (`exl3_tree_accept*`), against their independent list
  reference;
- the Nat models of the tree attention passes (`attn_tree*`) and of the GDN tree program and commit maps (`gdn_tree*`);
- the agreement of the two descriptor references (`tree_desc_agree*`);
- round-level speculation invariance over trees (`spec_inv_tree*`), under the hypothesis that a verify row's output
  depends only on its absolute prefix.

It does not prove the CUDA or Python code. The kernels' conformance to the models, the floating-point facts that hypothesis
rests on (an all-masked attention pass leaves (m, l, acc) unchanged; PEEK and ADV share one arithmetic body), and the
host plumbing are established by the bitwise differentials and rungs above, which are evidence, not proofs.

## 7. Protocol v5: 350 W (2026-09-28)

Declared 350 W, core 0, memory 0 (§3). Image `tree3s` (`sha256:ec9751b0…`), #67's stack unchanged.

| Run | Lane tok/s | tok/J | aime25 / mmlu-pro / i3-logic / lcb tok/s | C1 1K / 8K / 32K tok/s | Accept length | Rewards, truncations | Texts |
|---|---|---|---|---|---|---|---|
| #68 | **161.79** | **0.497** | 170.86 / 157.01 / 181.45 / 138.96 | 185.30 / 73.25 / 28.98 | 3.933 | = #67 | 85/85 byte-identical to #67 (250 W) |

- GSM8K (`bench/gsm8k_compare.py`, 5 isolated runs): 202.9 tok/s median (200.2-203.7), 0.617 tok/J; wall = 87 ms + 26.34 ms × rounds over 200 requests; answers identical to all 16 runs at 250 W.
- RoundBench (memory sweep, offset-0 arm, 12 paired reps): 25.73 ms per round at 1K, 26.46 at 8K.
- 32K prompt (*computed*: C1 regression intercept, 95 % CI): 26.2 ± 2.4 s at 350 W vs 35.3 ± 4.1 s at 250 W.
- Lane thermals (quiet curve): 77 °C median, 83 °C max; fan 75 % median, 79 % max. 60-min soak: 81 °C median, 84 °C max; fan 79 % median; thermal slowdown 4 of 657 samples.

**Measured and dropped: tree policy** (CPU simulation on the cs12 lattice capture; lane-mix committed tokens per round; leave-one-prompt-out fits, 95 % paired bootstrap over prompts; the shipped policy simulates 4.677 vs 4.675 measured on the GPU):

| Change | Δ tokens / round |
|---|---|
| Calibration (T, q) refit per held-out prompt | −0.02 % (−0.06..+0.01) |
| Depth prior on the tree shape | −0.17 % (−0.46..+0.04) |
| Confidence-keyed children per node | −0.06 % (−0.10..−0.01) |
| 3 or 5-6 children per popped node | −0.11 % / −0.02 % |
| Suffix/copy candidates from the full history (longest earlier match ≤ 32 tokens, merged into the 8 rows) | −0.02 % (−0.10..+0.03); used in 1.4 % of rounds |
| Oracle 8-row tree (true path always present) | +43.8 %: the headroom is candidate quality (the draft), not the shape |

## 8. Segment 12: cold prefill (`exl3-native-prefill-ttft-v1`, 350 W, 2026-09-29)

§2a suite; TTFT = 1-token request wall time (prefill + first verify round + HTTP), cold (unique leading nonce).

| Run | Image | Stack | Prefill tok/s (geomean) | TTFT 8K / 32K / 128K / 262K s | Texts |
|---|---|---|---|---|---|
| #70 | `tree3s` `ec9751b0…` | #68 | 977.9 | 5.89 / 25.80 / 153.78 / 435.05 | baseline |
| #71 | `p3020` `023c8662…` | + `3020-prefill-pattn8` (8-warp prefill attention, same per-element order) | **1072.8** (+9.7 %) | 5.81 / 24.44 / 135.00 / 365.90 | 9/9 rows = #70 |
| #72 | `p3020f` `5174b18b…` | + `5111-gdn-prefill-fuse-nom4096` (GDN conv reads the fp32 qkv, writes contiguous q/k/v) + `5112-mlp-act-fuse` (SiLU·up in the gate/up GEMM store) | **1087.8** (+11.2 %) | 5.69 / 24.09 / 133.65 / 362.47 | 9/9 rows = #70 |
| #73 | `p3021p` `d48879d1…` | + `3021c-prefill-pattn8-int8qk` (int8 Q·Kᵀ in prefill attention; numerics change, user-approved) | **1157.3** (+18.3 %) | 5.63 / 23.27 / 122.91 / 321.96 | first token 9/9 = #70; 32-token continuations 5/9 = #70, the rest diverge after 22-52 characters (paraphrases) |
| #74 | `p3021r` `91b01c53…` | #73 rebuilt after the eta → elpis rename (commit 14a15c6; acceptor tables unchanged) | 1156.0 | 5.65 / 23.28 / 122.92 / 321.96 | 9/9 rows = #73 byte for byte |

- 3020 exactness: kernel 76/76 bitwise vs 3010; served ids identical on the lane mix (forced chain and tree, 57,651 ids).
- 5111 / 5112 exactness: request-scoped prefill end state, served ids and rounds identical with each kill switch (`EXL3_GDN_PREFILL_FUSE`, `EXL3_MLP_ACT_FUSE`) on vs off at 32K (ABBA) and 128K (AB), proved on stacks that also carried 5110; the kept 5111 drops only 5110's b/a-split hunk. #72's 9 rows match #70.
- `5110-prefill-m4096` (merge aligned 2048 pieces into 4096): prefill state, ids and rounds identical at 32K/128K (TTFT ×0.974 / ×0.970), but the 262K run ended in a swallowed job error; not kept.
- 3021c (int8 Q·Kᵀ): the 3-bit K cache in the H32 basis is exactly int8 codes × one scale per 32-group, so only Q is quantized (per row); int32 partials, exact conversion. Kernel ×0.78-0.79 of 3020. V and P·V stay fp16; decode and verify attention are unchanged. Quality gate (`Int8Gate`, harness validated: capture = plain forward bit for bit, head = served m=1 head): teacher-forced KL vs the served route on held-out 8K / 32K / 128K documents 0.087 / 0.357 / 0.125 against exact-numerics floors (prefill chunk 1024 vs 2048; pre-3010 attention) of 0.093-0.096 / 0.355-0.389 / 0.115-0.135; top-1 0.949 / 0.902 / 0.951 vs floors 0.951 / 0.899-0.903 / 0.936-0.954. It misses the pre-registered strict rule on one of 9 checks (top-1 at 8K, by 0.11 pt) and passes the calibrated rule R2, which was set after seeing this result. Continuation KL (base's 256 greedy tokens) ×0.96 / ×1.09 of the worse floor; draft acceptance 3.33 vs 3.38 (within band). Broad suite on `p3021p`: AIME25 1.0, MMLU-Pro 0.8, LiveCodeBench 0.333 = #68 (per task identical); I3 Logic 0.25 vs 0.5: task 1 answered correctly at 14,217 of 16,384 tokens in #68 and hit the 16,384 cap here; total I3 tokens 39,751 vs 39,876. Lane primary 161.68 vs 161.79 tok/s.
