# elpis-fast

**The fastest decode and prefill for 4-bit Qwen3.8-27B on one RTX 3090 that preserve the model's quality, proven with Bend**

- Model: Qwen3.8-27B, EXL3 4.00 bpw weights, 3-bit KV cache.
- Context: 262,144 tokens (native).
- Speculation: DFlash2 draft, 8-row token tree, greedy.
- Prefill: int8 Q·Kᵀ attention, double-buffered K/V tiles.
- Decode: int8 Q·Kᵀ in the verify attention.
- GPU: one RTX 3090, 350 W cap.
- Proof: `bend PROOF.bend` (Bend 2.0.35): ALL PROOFS CHECK. `--verdict` (Lean 4.34.0 kernel): every proof module.

[elpis](https://github.com/gildrb/elpis) is the lossless sibling: same model, same benchmarks, every changed op proven no less accurate than stock.

## What fast means

elpis-fast = the fastest decode and the fastest prefill that a quality gate accepts.

| Rule | Value |
|---|---|
| Keep a speedup if | it is faster and the output is unchanged, or it is faster and it passes the quality gate |
| Draft | never changes the output (proven, tested) |
| Precision | may be lower than stock; measured, not bounded |

Quality gates (`Int8Gate` for prefill, `DecGate` for decode):

| | |
|---|---|
| Input | one held-out document per depth: 8K, 32K, 128K tokens |
| Method | teacher-forced logits vs the served route; continuations of 512 tokens |
| Floors | two exact-numerics variants of the served route (prefill chunk 1024 vs 2048; pre-3010 attention) |
| Rule P (strict, set before the runs) | P1 mean KL, P2 p99 KL, P3 top-1 within the floors; P4 draft acceptance within a band of base; P5 decode unchanged |
| Rule R2 (calibrated) | C1 continuation KL, C2 prompt KL, C3 top-1, C4 \|ΔNLL\|, C5 p99 KL, each with a margin over the floors; + P4, P5; thresholds set after earlier candidates were seen |
| Rule R3 (set before the runs) | R2's limits on 13 held-out documents (6 at 8K, 4 at 32K, 3 at 128K); paired block bootstrap, 2,000 replicates; a check passes only if its 90 % interval is inside the limit |
| int8 Q·Kᵀ (shipped) | passes R2 and R3; strict P: P3 fails at 8K under R2; P2 at 8K unresolved under R3 |
| int8 MLP (+ int8 Q·Kᵀ) | fails R2 at 8K: top-1 0.92907 < 0.94556, \|ΔNLL\| 0.2056 > 0.0804; R3 unresolved (C5 at 8K: interval −0.054 … +1.020); strict P1, P2 fail at 8K, 32K, 128K; not kept |
| int8 MLP at 8K / 32K only, int8 Q·Kᵀ at 128K+ | R3 unresolved (same C5 at 8K); not kept |
| int8 attention + GDN linears, MLP fp16 (5115) | +7 % prefill tok/s; R3 FAIL (C2 prompt KL at 8K / 32K / 128K, C5 at 8K / 32K / 128K); not kept |
| int8 GDN linears only (5115, `EXL3_INT8_ATTN=0`) | +5 % prefill tok/s; R3 FAIL (C2 at 8K and 128K); not kept |
| `DecGate` rule D1 (set before the runs) | R3's limits and bootstrap on the decode verify kernel: teacher-forced 512-token continuation in 8-row verify rounds after a cold prefill; 2 documents at 8K, 2 at 32K, 1 at 128K, 1 at 261,600; floors = Triton verify attention, half the split count |
| decode int8 Q·Kᵀ (3032, shipped) | D1 PASS at 8K, 32K, 128K, 262K; draft acceptance 3.507 vs band ≥ 3.398; strict P1-P3 fail or unresolved at some depths |
| decode fp16-accumulated P·V (3032 mode 2, with or without int8 Q·Kᵀ) | D1 unresolved; not shipped (kill switch `EXL3_AV_FAST`) |

Fast does not mean:

- Lossless: the int8 Q·Kᵀ prefill and the int8 Q·Kᵀ decode attention change the text. Autoresearch suite: first token 5/5 equal; 256-token decode texts 2/4 equal to `pfast4`.
- Error-bounded: prefill attention error vs fp64 is 3.6-33× stock (median 25×). No proof covers precision.
- Lossless against BF16: the 4.00 bpw weights and the 3-bit KV cache cost quality.

## Speed

Target: most tok/s and shortest time to first token (TTFT) at 262,144 context, one RTX 3090, 350 W.

Status: `pfast5` = `pfast4` + 3022 (prefill attention: double-buffered K/V tiles, same arithmetic, same token ids) + 3032 (decode verify attention: int8 Q·Kᵀ; changes the text, passes `DecGate`).

`pfast4` = `pfast2` + 9503f (prefill staging scratch: span rounded up to a multiple of 64 pages, not to a power of two) + 9503b (4096-row prefill merge for every prompt up to 262,143 tokens; staging bound 1,088 pages) + 9601 (tree verify pipe: verify inputs staged on the device, host check after launch, discard and recompute on mismatch). None of these three patches changes the token ids ([Proofs and tests](#proofs-and-tests)).

Autoresearch suite (`bash autoresearch.sh`, in process, no HTTP; 2026-10-05/06, 350 W):

| 350 W | elpis-fast `pfast5` | elpis-fast `pfast4` | elpis `p3031b` (#78) |
|---|---|---|---|
| Prefill tok/s, geomean over 8K / 32K / 128K / 262K | **1,220.7** | 1,197.2 | 997.4 |
| TTFT, 8K / 32K / 128K / 262K | 5.54 / 21.87 / 114.34 / **299.53** s | 5.55 / 22.08 / 117.27 / 312.41 s | 5.82 / 24.58 / 149.90 / 434.25 s |
| Decode tok/s, geomean over 1K / 8K / 32K / 262K | **125.7** | 120.1 | 82.7 |
| Decode, median ms per verify round, 1K / 8K / 32K / 262K | 25.44 / 26.06 / 28.08 / **45.59** | 25.75 / 26.38 / 28.93 / 48.28 | 26.70 / 27.27 / 29.40 / 112.94 |
| Lane scores: AIME 2025 · MMLU-Pro · I3 Logic · LiveCodeBench v6 | 3/3 · 8/10 · 2/4 · 1/3 = 14/20 | 3/3 · 8/10 · 1/4 · 1/3 = 13/20 | 3/3 · 8/10 · 2/4 · 1/3 = 14/20 |

- Decode tok/s depends on the text (tokens per round). Ms per round is the speed of the engine. `pfast5` changes the text.
- 262K decode: 128 tokens after a 262,000-token cold prefill.
- Tried after `pfast5`, not kept: int8 linears in prefill (fail R3, see above); int8 P·V in prefill attention (3023: prefill +1.2 %, 262K TTFT 292.0 s; text changes, gain too small for a numerics change); 8192-row prefill merge (9506: exact, flat); checkpoint settle without a second host copy (9504: exact, flat). Data: [benchmarks §11](docs/benchmarks.md#11-autoresearch-pfast5-350-w-2026-10-06).

Served suites, `pfast4` and elpis `p3031b` (#78) back to back, 2026-10-05, 350 W:

| 350 W | elpis-fast `pfast4` | elpis `p3031b` (#78) |
|---|---|---|
| Cold prefill, geomean over 8K / 32K / 128K / 262K | 1,200.0 tok/s | 1,003.4 tok/s |
| TTFT, 8K / 32K / 128K / 262K | 5.45 / 22.35 / 118.00 / 311.83 s | 5.61 / 25.08 / 150.15 / 434.09 s |
| Decode, median ms per verify round (mean of 4 windows), 1K / 8K / 32K context | 25.35 / 25.98 / 28.18 | 25.64 / 26.28 / 28.54 |
| Lane scores: AIME 2025 · MMLU-Pro · I3 Logic · LiveCodeBench v6 | 3/3 · 8/10 · 1/4 · 1/3 = 13/20 | 3/3 · 8/10 · 2/4 · 1/3 = 14/20 (#78, earlier session) |
| Prefill attention error vs fp64, relative to stock | 25× (median) | 0.890× (mean) |

| Workload, 350 W | tok/s | tok/J | Image |
|---|---|---|---|
| Lane: 20 calls, thinking on, whole request | 161.88 | not recorded | `pfast5` |
| Lane, same suite | 162.49 | not recorded | `pfast4` |
| GSM8K: 40 questions, 512 tokens, median of 3 runs | 205.3 | 0.630 | `pfast4` |
| GSM8K, same session | 193.7 | 0.586 | elpis `p3031b` (#78) |
| C1: 1K / 8K / 32K prompt, 1,024 tokens out, whole request | 166.7 / 76.9 / 32.8 | not recorded | `pfast5` |
| C1, same suite | 179.3 / 76.9 / 32.4 | not recorded | `pfast4` |

- Tokens per round change with the text. Compare tok/s only on the same text. `pfast5`, `pfast4` and `p3031b` give different text. Lane acceptance length: 3.886 (`pfast5`), 3.951 (`pfast4`).
- Decode per round: `p3031b` takes 1.1-1.3 % more time than `pfast4` at each depth.
- Native capacity, 262,136 + 8 tokens: minimum allocator headroom 583 → 1,077 MiB (`pfast2` → `pfast2` + 9503f / 9503b).
- Data: [benchmarks §10](docs/benchmarks.md#10-phase-2-pfast4-350-w-2026-10-05), [§11](docs/benchmarks.md#11-autoresearch-pfast5-350-w-2026-10-06).

## Compared

Figures of other projects come from their repositories. elpis-fast did not run them again. No other result uses the same prompts, power cap and metric. This is not a ranking.

| One RTX 3090 | Weights | Speculation | Context / KV | Power | Reported tok/s |
|---|---|---|---|---|---|
| elpis-fast (measured) | 4.00 bpw | DFlash2 + 8-row tree | 262,144 / 3-bit | 350 W cap | 161.9 lane (`pfast5`); 205.3 GSM8K (`pfast4`); whole request |
| elpis (measured) | 4.00 bpw | DFlash2 + 8-row tree | 262,144 / 3-bit | 350 W cap | 158.7 lane (#78); 193.7 GSM8K (#78, same session as `pfast4`); whole request |
| [trellis-serve](https://github.com/0xSero/trellis-serve/tree/1ace59c4b43ca16a50fb6b7acf8b3fd7e2351f96) README (MTP) | 3.00 bpw | MTP, 3 steps / 4 tokens | 212,992 / fp8 | not published | 96.2 prose, 141.1 code (thinking off); 141.3 prose, 129.3 code (thinking on); decode only ([sweep](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/speed-sweep/qwen38-27b-exl3-3bpw-mtp-vision-rtx3090-sglang-tp1-sweep.json)) |
| trellis-serve fastest recipe ([DFlash2](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/recipe/qwen38-27b-exl3-3bpw-dflash2-rtx3090-sglang-tp1.json), status "candidate") | 3.00 bpw | DFlash2, 5.0 bpw draft, block 8 | 131,072 / fp8 | not published | 98.2 prose, 225.1 code (thinking off); 227.0 prose, 195.4 code (thinking on); decode only |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | 4.00 bpw | DFlash2 | 8,192 / FP16 | 350 W cap | 162.9 GSM8K |

| One RTX 3090 | 32K prompt: TTFT | Longest prompt shown |
|---|---|---|
| elpis-fast (`pfast5`, autoresearch suite, no HTTP) | 21.87 s | 262,000 tokens: 299.53 s |
| elpis-fast (`pfast4`, served suite) | 22.35 s | 262,052 tokens: 311.83 s |
| elpis (#78, same session) | 25.08 s | 262,052 tokens: 434.09 s |
| trellis-serve MTP | 21.9 s (1,497 tok/s) | 208,858 tokens: 294 s |
| trellis-serve DFlash2 | 35.9 s (914 tok/s) | 126,782 tokens: 187 s |
| r0b0tlab | not published (150K prompt: 594 tok/s) | 262,080 tokens (needle test) |

- Same test: GSM8K, 350 W cap, same 40 questions: elpis-fast 205.3, r0b0tlab 162.9 tok/s (+26.0 %). [`bench/gsm8k_compare.py`](bench/gsm8k_compare.py) runs the workload of r0b0tlab's [`acceptance_check.py`](https://github.com/r0b0tlab/qwen38-exl3-dflash2/blob/main/scripts/acceptance_check.py).
- Metric: trellis-serve reports decode only. elpis-fast divides by the full request time: prefill and HTTP included.
- Bits: 3.00 bpw reads 25 % fewer weight bits per token than 4.00 bpw (*computed*), at a larger quantization error.
- Clocks: trellis-serve publishes no power limit. Its DFlash2 soak held the SMs at 1.74 GHz. elpis-fast: 1.51 GHz median per call (#68 lane).

## Proofs and tests

| Claim | Evidence | Result |
|---|---|---|
| Acceptance logic | `exl3_accept`, `exl3_tree_accept` = list references; emitted to C; checked by table at build | proven |
| The draft never changes the output, if each verify row depends only on its prefix (`~rinv`) | Bend: `spec_inv`, `spec_inv_tree` over chain and 8-row tree; `~rinv` is a hypothesis | proven under `~rinv` |
| Evidence for `~rinv` | Bend: `attn_rowinv_laws`, `gdn_rounds_laws` per kernel; the link to `~rinv` is not proven | per kernel |
| Each round makes progress; generation ends | Bend: round model `exl3_round`, without `~rinv` | proven |
| Kernel schedules: each output computed once, in a fixed order | Bend: `gemm_m16_*`, `mlp_m16_*`, `tail_m16_sched`, `m16_wsched*`, `m16_diet*`, `attn_split` / `chunk` / `stride` / `pre` / `tree` / `bounds`, `gdn_*`, `pattn_sched`, `pattn8_sched`, `hgemm_wide`, `norm_fuse`, `draft_head_*`, `draft_mask` | proven |
| Prefill patches: 3020 = 3010 per warp (and 3021c's schedule), 5111 / 5112 index maps and terms, 5110g/h merge = two 2048-row steps under its guard, 9502 copies settled before use, 3021c int32 core exact, 9503f / 9503b staging covers the span, 3022 two-stage K/V ring: every read sees its own tile, no load races a read | Bend: `pattn8_sched`, `gdn_conv_qkv`, `act_fuse`, `prefill_merge`, `prefill_membound`, `prefill_nosync`, `pattn8i_int`, `qc_staging`, `pattn8i_pipe` | proven |
| Decode int8 Q·Kᵀ (3032): K bytes = exact 3-bit codes; one group IMMA = the exact integer dot product; each score depends only on its row's q and its token's K (row invariance kept); tile / pass / slot schedule unchanged | Bend: `attn_fast` | proven |
| Tree verify pipe (9601): for every staged input, faults included, a round emits the sequential round's tokens; a mismatch always discards and recomputes; GDN conv windows restored on discard | Bend: `tree_pipe` (refines `spec_inv_tree`; a mutation witness shows that the check is necessary) | proven |
| Full contract | `bend PROOF.bend`: 50 law modules, 52 proof modules; `--verdict`: every proof module, in smaller runs (one module or one gate per run) | ALL PROOFS CHECK |
| The draft never changes the output, on the GPU | 15 prompts × normal / capped / all-rejected draft, tree and chain (`pfast4`) | 90/90 |
| 9601 recovery, on the GPU | 5 injected faults: 243 rounds discarded and recomputed; token ids and per-round records = the fault-free run | identical |
| Kernel changes | GDN state hashes, 1-8 steps (5108); 64 layers × rows 1-8 × 30 graph replays (2113); all 5,040 tree shapes vs the chain kernel (3012) | bit-exact |
| Exact prefill patches 3020, 5111, 5112, 5110g, 9502, 5110h, 3022 | prefill suite, 9 rows, vs the previous image (3022: autoresearch suite, 4 decode texts and 5 first tokens) | byte-identical |
| 9503f / 9503b | 262,136 + 8 tokens: prefill state, ids, rounds vs `pfast2` | identical |
| Decode speedups `cs10` → `tree3s` keep the text | lane 20 + C1 15 answers | byte-identical |
| Power and clocks keep the text | 250 W vs 350 W; memory offsets 0 … −2000 MHz | byte-identical |

- [`LAWS.bend`](LAWS.bend) = contract. [`PROOF.bend`](PROOF.bend) = proofs. Order for each engine change: law → proof → measurement.
- Bend proves facts about Bend models of the kernels. Source links and GPU tests check the link from model to CUDA. No proof covers this link.
- No law covers numeric precision.

## How

| Lever | Measured |
|---|---|
| DFlash2 draft: one pass proposes 7 tokens | 1.00 → 3.39 tokens per round at 1K; round time unchanged (≈34 ms, 250 W) |
| 8-row token tree: 7 nodes, best first; commit the longest matching path + 1 | 3.39 → 3.89 tokens per round at 1K; lane 111.85 → 122.33 tok/s (250 W) |
| int8 Q·Kᵀ in prefill attention (3021c): Q per row and K per key requantized to int8; exact int32 accumulation and conversion | kernel ×0.78-0.79 of 3020; 262K TTFT 362.5 → 322.0 s |
| Prefill: 8-warp attention (3020), fused GDN conv (5111), fused SiLU · up (5112), 4096-row merge (5110g, 5110h, 9503b), no host syncs (9502), 64-page staging scratch (9503f) | geomean 977.9 → 1,200.0 tok/s (#70 → `pfast4`) |
| Prefill attention, two-stage K/V ring (3022): the next tile loads during the whole compute of the current tile; one barrier per tile, not three | 262K TTFT 312.4 → 300.5 s; 128K 117.3 → 115.5 s |
| Tree verify pipe (9601): host check after the verify launch, not before | −0.69 / −0.51 / −0.49 % ms per round at 1K / 8K / 32K (two sessions, pooled) |
| Decode verify attention, int8 Q·Kᵀ (3032): Q per (row, 32-dim group) to int8, the 3-bit K codes used as exact int8 (2c − 7), integer IMMA | ms per round −4.8 % at 262K, −0.9 % at 32K, −0.5 % at 1K / 8K |
| 54 engine patches: 5 [`patches/exl3`](patches/exl3) + 49 [`patches/exl3-ext`](patches/exl3-ext) | per patch: [docs/benchmarks.md](docs/benchmarks.md) |
| 350 W cap, quiet fans (≤ 80 % to 84 °C) | lane +32.3 % tok/s vs 250 W at equal tok/J (0.497 vs 0.495) |
| Stock memory clock at 350 W | +5.4 % tok/s, +6.4 % tok/J vs −1500 MHz ([sweep](docs/benchmarks.md#3-power)) |

## Run

1. Make a private `QWEN_STATE_ROOT` with `models/`, `cache/`, `prefix-cache/` (mode 0700), `api-key` and `qwen-inference-launch.lock` ([docs/docker.md](docs/docker.md)).
2. Install Docker with Compose and Buildx, the NVIDIA Container Toolkit (CDI `nvidia.com/gpu=0`) and Nix with flakes.
3. Download the pinned weights and check every byte:

```sh
export QWEN_STATE_ROOT=/absolute/path/to/state
hf download r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw --revision 3f1771b8c21f83cbb8e82169559ced9f38ca04e5 --local-dir "$QWEN_STATE_ROOT/models/qwen38-27b-exl3"
hf download r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw --revision 265b5240592907d2d55ff0dc4d5f66569692604d --local-dir "$QWEN_STATE_ROOT/models/dflash2-exl3"
python3 -B prepare/verify-models.py --target "$QWEN_STATE_ROOT/models/qwen38-27b-exl3" --draft "$QWEN_STATE_ROOT/models/dflash2-exl3" --representation exl3
```

4. Build from source and start:

```sh
nix develop --no-write-lock-file -c true    # get the pinned Nix toolchain once
bash docker/fetch-base.sh                   # the only network step: download and sha256-check every base input
bash docker/build-base.sh                   # no network; tags qwen-elpis:exl3-base only if its content manifest matches the pin
bash docker/build-exl3.sh candidate-ext qwen-inference:exl3
export QWEN_IMAGE=qwen-inference:exl3
export QWEN_IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$QWEN_IMAGE")"   # sha256:<64 hex>
export QWEN_ALLOW_UNQUALIFIED=1
docker compose --project-name qwen-inference up --no-build --pull never --detach --wait
```

- API: `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`. Do not run another inference service on the same GPU.
- Prefix cache: kept across restarts, bound to `QWEN_IMAGE_ID`. `QWEN_PREFIX_PERSIST=0` turns it off.
- Docker owns runtime and restarts. Nix pins the tools, the `.#bend` toolchain and a Compose adapter ([nix/STANDALONE.md](nix/STANDALONE.md)).

## Measure

```sh
bash autoresearch.sh   # build this checkout + one GPU window: prefill and decode tok/s at 1K-262K (bench/ar_gpu.py)
bash bench/lane.sh     # the served prefill suite; protocol: docs/benchmarks.md; scoring: Prime Envs + Verifiers (eval/README.md)
curl -o gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
python3 -m bench.gsm8k_compare --api-key-file /path/to/api-key --data gsm8k-test.jsonl --out gsm8k.json
```

## Prove

```sh
nix run .#bend -- PROOF.bend                     # every law, TypeScript checker (about 25 min)
nix run .#bend-verdict -- bend/tree_pipe_proof.bend --verdict   # one proof module, checked again by the Lean-proven kernel; run each bend/*_proof.bend of PROOF.bend
nix develop --no-write-lock-file -c python3 -I -B bend/engine_trees.py OUT                # stock + patched ExLlamaV3 from the pinned tarball
nix develop --no-write-lock-file -c python3 -I -B bend/pattn_sched_diff.py OUT/patched    # one source link: Bend model text vs patched source
```

Every source link, its engine tree and its inputs: [docs/development.md](docs/development.md#source-links-bend_diffpy).

## Setup

| | |
|---|---|
| GPU | RTX 3090 24 GiB (GA102, SM86), VBIOS 94.02.42.80.1F, PCIe 4.0 ×16, driver 595.71.05 |
| Power, clocks | 350 W cap since 2026-09-28 (250 W before); core and memory offsets 0; the lane checks all three via NVML |
| Host | Ryzen 7 5800X (8 cores / 16 threads), 125.7 GiB, NixOS 26.05, Linux 6.18.50 |
| Runtime | rootless Docker 29.7.2, CDI, read-only root; Ubuntu 24.04 CUDA base; Python 3.13.10, PyTorch 2.10.0+cu130, CUDA 13.0.96, cuBLAS 13.1.0.3, Triton 3.6.0 |
| Engine | ExLlamaV3 1.5.0 `355c6ee` (r0b0tlab `community`, native DFlash2) + 5 [`patches/exl3`](patches/exl3) + 49 [`patches/exl3-ext`](patches/exl3-ext), SHA256-pinned |
| Server | [`serve/exl3_server.py`](serve/exl3_server.py): authenticated OpenAI-compatible `/v1` chat/completions + tool calls, greedy, one sequence |
| Target | [`r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw) @ `3f1771b8` (`Qwen/Qwen3.8-27B`): 48 Gated DeltaNet + 16 full-attention layers, hidden 5,120, vocab 248,320; 4.00 bpw, 6 bpw head; 16.5 GB |
| Draft | [`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw) @ `265b5240` (`incoai/Qwen3.8-27B-DFlash2`): 5 sliding-attention layers, block 8, reads target layers 5/19/33/47/61, top-16 selector; 1.25 GB |
| Recipe | [`serve/exl3-entrypoint.sh`](serve/exl3-entrypoint.sh): context 262,144, cache 270,336, 3-bit KV (12 KiB per token: 16 of 64 layers keep KV); 8 verify rows per round; each file hashed again against [`prepare/exl3-manifest.json`](prepare/exl3-manifest.json) at start |

## Limitations

```
- Greedy only. One sequence at a time.
- One 270K-token prefix cache for all clients. A request with another prefix can evict a long
  session; its next turn re-prefills (197 s at 193K). See docs/benchmarks.md §12.
- Chat stream=true streams reasoning and content as generated; tool calls arrive at the end of the turn.
- Prefill and decode verify attention compute Q·Kᵀ in int8. Outputs can differ from fp16.
  EXL3_AV_FAST=0 restores the exact decode attention kernel.
- Exact logit ties can depend on max_tokens (cs12: token 39457 at 8192 vs 54185 at 256,
  p = 0.28775 each). Compare outputs only at equal request parameters.
```

## References

- Protocol, every run, every change, every dropped attempt: [docs/benchmarks.md](docs/benchmarks.md)
- Architecture and the Bend proof boundary: [docs/architecture.md](docs/architecture.md)
- Deployment: [docs/docker.md](docs/docker.md)
- Evaluator: [eval/README.md](eval/README.md)
- All docs: [docs/README.md](docs/README.md)
- Lossless sibling: [elpis](https://github.com/gildrb/elpis)
- Security reports: [SECURITY.md](SECURITY.md)
- License: MIT ([LICENSE](LICENSE))
