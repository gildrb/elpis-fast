# elpis-fast

**The fastest decode and the fastest prefill for Qwen3.8-27B on one RTX 3090 that keep the model's quality.**

- Model: Qwen3.8-27B, EXL3 4.00 bpw weights, 3-bit KV cache.
- Context: 262,144 tokens (native).
- Speculation: DFlash2 draft, 8-row token tree, greedy.
- Prefill: int8 Q·Kᵀ attention.
- GPU: one RTX 3090, 350 W cap.
- Proof: `bend PROOF.bend --verdict` (Bend 2.0.35, Lean 4.34.0 kernel): ALL PROOFS CHECK.

[elpis](https://github.com/gildrb/elpis) is the lossless sibling: same model, same benchmarks, every changed op proven no less accurate than stock.

## What fast means

elpis-fast = the fastest decode and the fastest prefill that a quality gate accepts.

| Rule | Value |
|---|---|
| Keep a speedup if | it is faster and the output is unchanged, or it is faster and it passes the quality gate |
| Draft | never changes the output (proven, tested) |
| Precision | may be lower than stock; measured, not bounded |

Quality gate (`Int8Gate`):

| | |
|---|---|
| Input | one held-out document per depth: 8K, 32K, 128K tokens |
| Method | teacher-forced logits vs the served route; continuations of 512 tokens |
| Floors | two exact-numerics variants of the served route (prefill chunk 1024 vs 2048; pre-3010 attention) |
| Rule P (strict, set before the runs) | P1 mean KL, P2 p99 KL, P3 top-1 within the floors; P4 draft acceptance within a band of base; P5 decode unchanged |
| Rule R2 (calibrated) | C1 continuation KL, C2 prompt KL, C3 top-1, C4 \|ΔNLL\|, C5 p99 KL, each with a margin over the floors; + P4, P5; thresholds set after earlier candidates were seen |
| int8 Q·Kᵀ (shipped) | passes R2; fails P3 at 8K: top-1 0.94946 < floor 0.95056 |
| int8 MLP + int8 Q·Kᵀ | fails R2 at 8K: top-1 0.92907 < 0.94556, \|ΔNLL\| 0.2056 > 0.0804; not kept |

Fast does not mean:

- Lossless: the int8 Q·Kᵀ prefill changes the text. Prefill suite: first token 9/9 equal to fp16; 32-token continuations: 4/9 differ after 22-52 characters.
- Error-bounded: prefill attention error vs fp64 is 3.6-33× stock (median 25×). No proof covers precision.
- Lossless against BF16: the 4.00 bpw weights and the 3-bit KV cache cost quality.

## Speed

Target: most tok/s and shortest time to first token (TTFT) at 262,144 context, one RTX 3090, 350 W.

Status: `pfast2` = `pfast1` (#74 + 5110g + 9501b) + 9502 (prefill without host syncs) + 5110h (4096-row prefill merge, memory-bound). `pfast2` changes prefill only: 9/9 suite rows byte-equal to `pfast1`.

| 350 W | elpis-fast `pfast2` | elpis `p3031b` (#78) |
|---|---|---|
| Cold prefill, geomean over 8K / 32K / 128K / 262K | 1,196.6 tok/s | 1,014.7 tok/s |
| TTFT, 8K / 32K / 128K / 262K | 5.45 / 22.45 / 118.63 / 312.58 s | 5.60 / 24.42 / 148.75 / 431.47 s |
| Decode, median ms per verify round, 1K / 8K / 32K context | 25.50 / 26.24 / 28.39 (`pfast1`) | 25.75 / 26.40 / 28.60 |
| Lane scores: AIME 2025 · MMLU-Pro · I3 Logic · LiveCodeBench v6 | 3/3 · 8/10 · 1/4 · 1/3 = 13/20 (`p3021p`, #73) | 3/3 · 8/10 · 2/4 · 1/3 = 14/20 |
| Prefill attention error vs fp64, relative to stock | 25× (median) | 0.890× (mean) |

| Workload, 350 W | tok/s | tok/J | Run |
|---|---|---|---|
| Lane: 20 calls, thinking on, whole request | 161.68 | not recorded | #73 (`p3021p`) |
| GSM8K: 40 questions, 512 tokens, median of 5 runs | 202.9 | 0.617 | #68 (`tree3s`) |
| C1: 1K / 8K / 32K prompt, 1,024 tokens out, whole request | 185.3 / 73.3 / 29.0 | not recorded | #68 (`tree3s`) |

- #68 and #73 use the decode stack of `pfast2`.
- Tokens per round change with the text. Compare tok/s only on the same text.
- `p3021p` I3 task 1 hit the 16,384-token cap; on `tree3s` it was correct at 14,217 tokens.
- Data: [benchmarks §7](docs/benchmarks.md#7-protocol-v5-350-w-2026-09-28), [§8](docs/benchmarks.md#8-segment-12-cold-prefill-exl3-native-prefill-ttft-v1-350-w-2026-09-29).

## Compared

Figures of other projects come from their repositories. elpis-fast did not run them again. No other result uses the same prompts, power cap and metric. This is not a ranking.

| One RTX 3090 | Weights | Speculation | Context / KV | Power | Reported tok/s |
|---|---|---|---|---|---|
| elpis-fast (measured) | 4.00 bpw | DFlash2 + 8-row tree | 262,144 / 3-bit | 350 W cap | 161.7 lane (#73); 202.9 GSM8K (#68); whole request |
| elpis (measured) | 4.00 bpw | DFlash2 + 8-row tree | 262,144 / 3-bit | 350 W cap | 158.7 lane (#78); whole request |
| [trellis-serve](https://github.com/0xSero/trellis-serve/tree/1ace59c4b43ca16a50fb6b7acf8b3fd7e2351f96) README (MTP) | 3.00 bpw | MTP, 3 steps / 4 tokens | 212,992 / fp8 | not published | 96.2 prose, 141.1 code (thinking off); 141.3 prose, 129.3 code (thinking on); decode only ([sweep](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/speed-sweep/qwen38-27b-exl3-3bpw-mtp-vision-rtx3090-sglang-tp1-sweep.json)) |
| trellis-serve fastest recipe ([DFlash2](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/recipe/qwen38-27b-exl3-3bpw-dflash2-rtx3090-sglang-tp1.json), status "candidate") | 3.00 bpw | DFlash2, 5.0 bpw draft, block 8 | 131,072 / fp8 | not published | 98.2 prose, 225.1 code (thinking off); 227.0 prose, 195.4 code (thinking on); decode only |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | 4.00 bpw | DFlash2 | 8,192 / FP16 | 350 W cap | 162.9 GSM8K |

| One RTX 3090 | 32K prompt: TTFT | Longest prompt shown |
|---|---|---|
| elpis-fast (`pfast2`) | 22.45 s | 262,052 tokens: 312.58 s |
| elpis (#78) | 24.4 s (1,344 tok/s) | 262,052 tokens: 431.5 s |
| trellis-serve MTP | 21.9 s (1,497 tok/s) | 208,858 tokens: 294 s |
| trellis-serve DFlash2 | 35.9 s (914 tok/s) | 126,782 tokens: 187 s |
| r0b0tlab | not published (150K prompt: 594 tok/s) | 262,080 tokens (needle test) |

- Same test: GSM8K, 350 W cap, same 40 questions: elpis-fast 202.9, r0b0tlab 162.9 tok/s (+24.6 %). [`bench/gsm8k_compare.py`](bench/gsm8k_compare.py) runs the workload of r0b0tlab's [`acceptance_check.py`](https://github.com/r0b0tlab/qwen38-exl3-dflash2/blob/main/scripts/acceptance_check.py).
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
| Kernel schedules: each output computed once, in a fixed order | Bend: `gemm_m16_*`, `mlp_m16_*`, `tail_m16_sched`, `m16_wsched*`, `m16_diet*`, `attn_split` / `chunk` / `stride` / `pre` / `tree` / `bounds`, `gdn_*`, `pattn_sched`, `hgemm_wide`, `norm_fuse`, `draft_head_*`, `draft_mask` | proven |
| Full contract | `bend PROOF.bend --verdict`: 39 law modules, 41 proof modules | ALL PROOFS CHECK |
| The draft never changes the output, on the GPU | 15 prompts × normal / capped / all-rejected draft (`cs10`) | 45/45 |
| Kernel changes | GDN state hashes, 1-8 steps (5108); 64 layers × rows 1-8 × 30 graph replays (2113); all 5,040 tree shapes vs the chain kernel (3012) | bit-exact |
| Exact prefill patches 3020, 5111, 5112, 5110g, 9502, 5110h | prefill suite, 9 rows, vs the previous image | byte-identical |
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
| int8 Q·Kᵀ in prefill attention (3021c): the 3-bit K codes are exact int8; only Q is quantized, per row | kernel ×0.78-0.79 of 3020; 262K TTFT 362.5 → 322.0 s |
| Prefill: 8-warp attention (3020), fused GDN conv (5111), fused SiLU · up (5112), 4096-row merge (5110g, 5110h), no host syncs (9502) | geomean 977.9 → 1,196.6 tok/s (#70 → `pfast2`) |
| 49 engine patches: 5 [`patches/exl3`](patches/exl3) + 44 [`patches/exl3-ext`](patches/exl3-ext) | per patch: [docs/benchmarks.md](docs/benchmarks.md) |
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
bash autoresearch.sh   # the lane; protocol: docs/benchmarks.md; scoring: Prime Envs + Verifiers (eval/README.md)
curl -o gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
python3 -m bench.gsm8k_compare --api-key-file /path/to/api-key --data gsm8k-test.jsonl --out gsm8k.json
```

## Prove

```sh
nix run .#bend -- PROOF.bend                     # every law, TypeScript checker (about 8 min)
nix run .#bend-verdict -- PROOF.bend --verdict   # the same, checked again by the Lean-proven kernel (about 1.6 h)
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
| Engine | ExLlamaV3 1.5.0 `355c6ee` (r0b0tlab `community`, native DFlash2) + 5 [`patches/exl3`](patches/exl3) + 44 [`patches/exl3-ext`](patches/exl3-ext), SHA256-pinned |
| Server | [`serve/exl3_server.py`](serve/exl3_server.py): authenticated OpenAI-compatible `/v1` chat/completions + tool calls, greedy, one sequence |
| Target | [`r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw) @ `3f1771b8` (`Qwen/Qwen3.8-27B`): 48 Gated DeltaNet + 16 full-attention layers, hidden 5,120, vocab 248,320; 4.00 bpw, 6 bpw head; 16.5 GB |
| Draft | [`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw) @ `265b5240` (`incoai/Qwen3.8-27B-DFlash2`): 5 sliding-attention layers, block 8, reads target layers 5/19/33/47/61, top-16 selector; 1.25 GB |
| Recipe | [`serve/exl3-entrypoint.sh`](serve/exl3-entrypoint.sh): context 262,144, cache 270,336, 3-bit KV (12 KiB per token: 16 of 64 layers keep KV); 8 verify rows per round; each file hashed again against [`prepare/exl3-manifest.json`](prepare/exl3-manifest.json) at start |

## Limitations

```
- Greedy only. One sequence at a time.
- Chat stream=true sends buffered SSE. The first event is not the first token.
- Prefill computes Q·Kᵀ in int8. Outputs can differ from an fp16 prefill.
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
