# elpis-fast: speed-first Qwen3.8-27B on one RTX 3090 at 350 W, 262K context

**EXL3 4.00 bpw · DFlash2 + 8-row token tree · 262,144 context · one RTX 3090 at 350 W, quiet fans · the draft never changes the output (acceptance proved in Bend) · prefill trades precision for speed: int8 Q·Kᵀ and fp16 sums, so outputs can differ from full-precision prefill.**

elpis-fast is the speed-first build. Its accurate sibling is [elpis](https://github.com/gildrb/elpis): same model, same benchmarks, never less precise than stock ExLlamaV3.

| | elpis-fast (this repo) | elpis |
|---|---|---|
| Rule | fastest serving whose quality is measured | every speedup is Bend-proven to keep the output bit for bit, or measured at least as accurate as the stock kernel it replaces; nothing computes less precisely than stock ExLlamaV3 |
| Speculative decoding | the draft never changes the output: Bend proof + bitwise tests | same |
| Prefill arithmetic | int8 Q·Kᵀ and fp16 P·V sums; outputs can differ from full-precision prefill | fp32 sums in prefill attention (3022 v4); prefill GEMMs use stock's own scheme, bit-exact with stock; no int8 |
| Evidence | teacher-forced KL within exact-numerics floors; broad-suite rewards; attention error vs fp64 3.6-33× stock's (median 25×) | vs fp64: prefill attention error ≤ stock Triton in 16/16 cells, decode error ≤ stock on every op; draft on/off byte-identity 90/90 |
| Model | EXL3 4.00 bpw weights, 3-bit KV cache (not identical to BF16) | same |

## How it compares with other RTX 3090 results for this model

Other projects' figures below are quoted from their repositories, not re-run here. No
other result shares elpis's prompts, power cap and metric, so read this as what each
project reports, not as a ranking.

| One RTX 3090 | Weights | Speculation | Context / KV | Power | Reported tok/s |
|---|---|---|---|---|---|
| **elpis, this repo** (measured) | 4.00 bpw | DFlash2 + 8-row token tree | 262,144 / 3-bit | 350 W cap; SM 1.24-1.62 GHz per call (median 1.51) | **161.8** reasoning + code lane, **202.9** GSM8K (whole request, prefill included); decode only: 151.1 at 1K (RoundBench), 214.8 on GSM8K (*computed*) |
| [trellis-serve](https://github.com/0xSero/trellis-serve/tree/1ace59c4b43ca16a50fb6b7acf8b3fd7e2351f96) README headline, by 0xSero | 3.00 bpw | MTP, 3 steps / 4 tokens | 212,992 / fp8 | not published | 96.2 prose, 141.1 code (thinking off); 141.3 prose, 129.3 code (thinking on); decode only ([sweep](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/speed-sweep/qwen38-27b-exl3-3bpw-mtp-vision-rtx3090-sglang-tp1-sweep.json)) |
| trellis-serve's fastest 3090 recipe ([DFlash2](https://github.com/0xSero/local-ai-registry/blob/c6e6f4c796304229a3c11442af6f09673180d4f6/data/registry/recipe/qwen38-27b-exl3-3bpw-dflash2-rtx3090-sglang-tp1.json), registry status "candidate") | 3.00 bpw | DFlash2, 5.0 bpw draft, block 8 | 131,072 / fp8 | not published; SM 1.74 GHz in its soak | 98.2 prose, 225.1 code (thinking off); 227.0 prose, 195.4 code (thinking on); decode only |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | 4.00 bpw | DFlash2 | 8,192 / FP16 in this run | 350 W cap | 162.9 GSM8K (in-process, per request) |

What differs:

- **Prompts.** Only GSM8K is shared: elpis 202.9 vs r0b0tlab 162.9 (+24.6 %), both at a 350 W cap, same
  40 questions ([details](#gsm8k-the-same-test-as-r0b0tlabs-published-number)).
  trellis-serve's prose and code panel prompts are not published.
- **Metric.** trellis-serve reports decode only: (completion tokens − 1) / (last − first
  streamed token). elpis's lane and GSM8K rates divide by the whole request wall time; its
  decode-only figures: RoundBench 3.89 tokens per 25.7 ms round (1K); GSM8K 5.66 tokens per
  26.3 ms round (*computed*, regression over 200 requests).
- **Power.** elpis and r0b0tlab: 350 W cap. trellis-serve publishes no power limit; its DFlash2
  soak held the SMs at 1.74 GHz, against elpis's 1.51 GHz median per call. elpis at 250 W:
  −24 % tok/s at the same tok/J ([250 W vs 350 W](#250-w-vs-350-w)).
- **Bits.** 3.00 bpw reads 25 % fewer weight bits per token than 4.00 bpw (*computed*),
  at a larger quantization error; quality is not compared here.
- **Cost per verify step** (*computed*: reported tok/s ÷ reported accept length, assuming
  the accept length counts the bonus token): trellis-serve 23.4-24.9 ms at short context,
  elpis 25.7 ms at 1K and 26.5 ms at 8K (RoundBench, 350 W). Tokens per step at temperature 0:
  trellis-serve 2.25-3.48 (MTP) and 2.45-5.65 (DFlash2) on its panel; elpis 3.93 on the lane's
  C1 rows, 5.66 on GSM8K.

Prefill and long context (others quoted; elpis-fast and elpis measured: cold prompt, time of a 1-token request, 350 W, image `pfast1` and elpis run #77):

| One RTX 3090 | 32K prompt: time to first token | Longest prompt shown |
|---|---|---|
| **elpis-fast** (350 W, int8 Q·Kᵀ) | 22.7 s (1,448 tok/s) | 262,052 tokens: 323.0 s to first token |
| **elpis** (350 W, fp32 attention sums) | 23.9 s (1,371 tok/s) | 262,052 tokens: 427.3 s to first token |
| trellis-serve MTP (README headline) | 21.9 s (1,497 tok/s) | 208,858 tokens: 294 s to first token |
| trellis-serve DFlash2 | 35.9 s (914 tok/s) | 126,782 tokens: 187 s to first token |
| r0b0tlab | not published (150K prompt: 594 tok/s) | 262,080 tokens (needle test) |

elpis also reports energy: 0.497 tok/J on the lane, 0.617 on GSM8K (350 W); no other row publishes tok/J.

In short:

- Same test, same 350 W cap: elpis 202.9 vs r0b0tlab 162.9 tok/s (+24.6 %).
- Decode only: trellis-serve DFlash2 225-227 tok/s on its code and thinking-on prose panels (3.00 bpw, 131K window, SM 1.74 GHz, power unpublished); elpis 214.8 on GSM8K (*computed*; 4.00 bpw, 262K window).
- Prefill at 32K: trellis-serve MTP fastest (21.9 s); elpis-fast 22.7 s; elpis 24.5 s; trellis-serve DFlash2 35.9 s.
- trellis-serve's README headline (MTP): 96-141 tok/s decode.

## Speed

**Target:** most tok/s at the native 262K context, one RTX 3090, 350 W. **Status:** best image `pfast1` = #74's stack (`tree3s` decode + prefill patches 3020/5111/5112/3021c, where 3021c computes prefill Q·Kᵀ in int8) + 5110g + 9501b; 350 W, memory offset 0, 2026-09-30. Live serving runs `p3021r` (#74). Decode rows were measured on `tree3s`, whose decode path is unchanged.

| Workload | tok/s | Tokens / round | tok/J |
|---|---|---|---|
| **Lane**: 20 calls, AIME 2025 · MMLU-Pro · I3 Logic · LiveCodeBench v6, thinking on, whole request (#68) | **161.79** | — | **0.497** |
| **GSM8K**: 40 questions, 512 tokens, median of 5 runs | **202.9** | 5.66 | **0.617** |
| C1 whole request, 1K / 8K / 32K-token prompt, 1,024 tokens out | 185.3 / 73.3 / 29.0 | 5.81 / 3.40 / 3.37 | 0.578 / 0.217 / 0.085 |
| Prefill, cold 8K / 32K / 128K / 262K prompt (1-token request, `pfast1`) | 1,477 / 1,448 / 1,060 / 811 | — | — |
| Decode only, 1K / 8K context (RoundBench, 12 reps) | 151.1 / 125.2 | 3.89 / 3.31 | 0.456 / 0.380 |

**elpis-fast vs elpis** (same prompts, same protocol, 350 W):

| | elpis-fast `pfast1` | elpis `p9502` (#77) |
|---|---|---|
| Cold prefill, geomean over 8K / 32K / 128K / 262K | 1,164.9 tok/s (+13.0 %) | 1,031.2 tok/s |
| Time to first token, 8K / 32K / 128K / 262K | 5.58 / 22.67 / 123.66 / 322.98 s | 5.50 / 23.94 / 146.25 / 427.29 s |
| Decode: median ms per verify round at 1K / 8K / 32K context, 256 tokens (mean of three 350 W windows per image; elpis measured on `p9501x4`, #76) | 25.50 / 26.24 / 28.39 | 25.73 / 26.19 / 28.36 |
| Scores: AIME 2025 · MMLU-Pro · I3 Logic · LiveCodeBench | 3/3 · 8/10 · 1/4 · 1/3 (`p3021p`, #73: same prefill arithmetic, without 5110g / 9501b) | 3/3 · 8/10 · 1/4 · 1/3 |
| Prefill attention error vs fp64, relative to stock Triton | median 25× stock (3.6-33×) | ≤ stock in 16/16 cells |

- The decode rows compare the same 8-row verify per round. Tokens per round follow each build's own text, so tok/s is not comparable across builds.

<details><summary>Lane per task (#68, 350 W)</summary>

| Task set | Calls | Budget / call | Output tokens | Wall s | tok/s | Mean W | tok/J | Score | At budget |
|---|---|---|---|---|---|---|---|---|---|
| AIME 2025 | 3 | 32,768 | 36,985 | 216.5 | 170.86 | 324.2 | 0.527 | 3/3 | 0 |
| MMLU-Pro | 10 | 8,192 | 11,991 | 76.4 | 157.01 | 304.4 | 0.516 | 8/10 | 1 |
| I3 Logic | 4 | 16,384 | 39,876 | 219.8 | 181.45 | 328.3 | 0.553 | 2/4 | 1 |
| LiveCodeBench v6 | 3 | 16,384 | 36,018 | 259.2 | 138.96 | 329.6 | 0.422 | 1/3 | 2 |
| **All** | **20** | | **124,870** | **771.8** | **161.79** | **325.2** | **0.497** | | **4** |

- tok/s = completion tokens (reasoning included) / request wall time, prefill and HTTP included.
- Power: host `nvidia-smi` sampler, 250 ms, over each call.

</details>

<details><summary>Decode rounds (RoundBench: fresh process per repetition, 90 s heat-up)</summary>

| Power | Depth | Mode | ms / round | Tokens / round | tok/s (*computed*) | J / round | tok/J |
|---|---|---|---|---|---|---|---|
| 350 W | 1K | tree | 25.73 | 3.89 | 151.1 | 8.52 | 0.456 |
| 350 W | 8K | tree | 26.46 | 3.31 | 125.2 | 8.72 | 0.380 |
| 250 W | 1K | tree | 34.23 | 3.89 | 113.6 | 8.63 | 0.451 |
| 250 W | 8K | tree | 35.01 | 3.31 | 94.5 | 8.71 | 0.380 |
| 250 W | 1K | chain (`EXL3_TREE=0`) | 34.29 | 3.39 | 98.9 | 8.65 | 0.392 |
| 250 W | 8K | chain (`EXL3_TREE=0`) | 35.03 | 2.90 | 82.8 | 8.78 | 0.330 |
| 250 W | 1K | every draft token rejected (`cs12`) | 33.89 | 1.00 | 29.5 | 8.49 | 0.118 |
| 250 W | 8K | every draft token rejected (`cs12`) | 34.71 | 1.00 | 28.8 | 8.65 | 0.116 |

- 350 W: memory offset 0, 12 paired reps; 250 W: memory −1500, 4 reps.

</details>

## The draft never changes the output: proof and tests

| Claim | Proof | Result |
|---|---|---|
| The draft never changes the text | invariance gate (`cs10`): 15 prompts × normal / capped / all-rejected draft; not yet re-run on the int8-prefill stack (#73/#74) | 45/45 identical token ids |
| Decode speedups since `cs10` never changed the text | lane 20 + C1 15 answers, `cs10` → `cs11` → `cs12` → `tree3s` | byte-identical |
| Earlier kept speedups changed the arithmetic, so the text can differ | decode 3003, 2102, 8202, 3006, 2105, 5106, 8205b (split-K order); prefill 3010 (Q·Kᵀ and P·V summed in fp16 over 32-value spans) | accuracy recorded only for 2105 (vs fp32: no worse), 5106 (vs fp64: lower) and 3010 (teacher-forced KL below the chunk-size noise floor) |
| Exact prefill patches 3020 / 5111 / 5112 never change the text | prefill suite, 9 rows: #71 and #72 vs #70 | byte-identical |
| **Exception:** int8 prefill Q·Kᵀ (3021c, live since #73) changes the text | prefill suite, 9 rows: #73 vs #70 | first token 9/9 identical; 32-token continuations 4/9 differ after 22-52 characters |
| Power and clocks never change the text | lane 20 + C1 15 answers, 250 W (#67) vs 350 W (#68); memory offsets 0 … −2000 (RoundBench ids) | byte-identical |
| | GSM8K 40 answers × 21 runs, `cs12` + `tree3s`, 250 W + 350 W | byte-identical |
| Forced chain = old engine | `EXL3_TREE_FORCE_CHAIN=1` vs `cs12`, 17 prompts: ids, every round, drafted ids, usage | identical |
| Accept / commit logic | `bend PROOF.bend` (Bend 2.0.34): 41 modules, chain + tree acceptance, speculation invariance over trees | "ALL PROOFS CHECK" |
| Kernel changes | GDN state hashes, 1-8 steps (5108); 64 layers × rows 1-8 × 30 graph replays (2113); all 5,040 tree shapes vs the chain kernel (3012) | bit-exact |
| The tree costs no time (250 W) | tree − chain, ms per round, 4 fresh processes | 1K: −0.06 (95 % CI −0.31..+0.19); 8K: −0.02 (−0.21..+0.18) |
| Scores, `tree3s` (#68) | AIME 2025 3/3 · MMLU-Pro 8/10 · I3 Logic 2/4 · LiveCodeBench 1/3 | = `cs12` |
| Scores, int8 prefill `p3021p` (#73) | AIME 2025 3/3 · MMLU-Pro 8/10 · I3 Logic 1/4 · LiveCodeBench 1/3 | I3 task 1: correct at 14,217 tokens on `tree3s`; hit the 16,384-token cap on `p3021p` |

- [`LAWS.bend`](LAWS.bend) = contract; [`PROOF.bend`](PROOF.bend) = proofs. Order for every engine change: law → proof → measurement.
- Proofs cover Bend models of the logic and of each modelled kernel's schedule: which outputs it computes, each exactly once, in which summation order, and row invariance for verify attention. Not proven: that the CUDA code matches those models, and the floating-point values themselves; both are tested (bitwise differentials above). The speculation proof assumes each verify row's token depends only on its prefix (`~rinv`, [`bend/spec_inv_tree_laws.bend`](bend/spec_inv_tree_laws.bend)); the kernel row-invariance laws support that assumption, but the link between them is a prose argument (the tree design note, not in this repo), not a proof.

## 262K context

| | |
|---|---|
| Context / cache | 262,144 / 270,336 tokens, 3-bit K and V |
| Longest prompt run | 262,000 tokens: 322.0 s to first token at 350 W (#73; 435.0 s before the prefill patches, #70); 262,136 tokens: 537 s at 250 W (`cs10`) |
| 32K prompt to first token | 23.3 s at 350 W (#73); 25.8 s before the prefill patches (#70) |
| KV size | 12 KiB / token: only 16 of 64 layers keep KV; 3.1 GiB at 270,336 tokens (*computed*) |
| GPU memory, live | 21,888 / 24,576 MiB |

## GSM8K, the same test as r0b0tlab's published number

[`bench/gsm8k_compare.py`](bench/gsm8k_compare.py) = r0b0tlab's [`acceptance_check.py`](https://github.com/r0b0tlab/qwen38-exl3-dflash2/blob/main/scripts/acceptance_check.py) workload: first 40 GSM8K test questions (pinned), raw ChatML, greedy, 512 tokens, mean of per-request tok/s.

| | r0b0tlab (published) | elpis `cs12`, 250 W (11 runs) | elpis `tree3s`, 250 W (5 runs) | elpis `tree3s`, 350 W (live, 5 runs) |
|---|---|---|---|---|
| Power | 350 W cap; 336 W mean (their telemetry, another run) | 250 W cap; 249.4 W median | 250 W cap; 249.4 W median | 350 W cap; 329.5 W median |
| Engine | ExLlamaV3 `355c6ee`, unpatched | + 36 patches | + 41 patches | + 41 patches |
| Context / KV | 8,192 / FP16 | 262,144 (cache 270,336) / 3-bit | same | same |
| Transport | in-process `generate()` | HTTP `/v1/completions` | same | same |
| tok/s | 162.9 | 148.8 median (147.9-154.3) | 153.3 median (151.9-159.7) | **202.9** median (200.2-203.7) |
| Tokens / round | 5.657 | 5.552 | 5.695 | 5.695 |
| Answers at the 512-token cap | 5/40 | 4/40 | 4/40 | 4/40 |
| tok/J | not published | 0.598 median (0.593-0.623) | 0.616 median (0.610-0.646) | **0.617** median (0.604-0.637) |

- Same test, same 350 W cap: elpis +24.6 % over r0b0tlab.
- Answers: byte-identical across all 21 elpis runs (both images, both caps).
- Tree gain (250 W): +3.0 % here vs +9.4 % on the lane; the chain already commits 5.55 of 8 tokens per round.
- Spread: fresh window, quiet host fastest (250 W: `cs12` 153.9-154.3, `tree3s` 159.7; 350 W: 203.7); warm card slower.

## 250 W vs 350 W

Same image (`tree3s`), token-identical outputs. 250 W: memory −1500 MHz, old fan curve (100 % at 75 °C). 350 W: memory 0, quiet curve (≤ 80 % to 84 °C).

| Workload | 250 W | 350 W | Δ |
|---|---|---|---|
| Lane tok/s (#67 → #68) | 122.33 | **161.79** | +32.3 % |
| Lane tok/J | 0.495 | **0.497** | +0.4 % |
| GSM8K tok/s (median of 5 runs) | 153.3 | **202.9** | +32.4 % |
| GSM8K tok/J | 0.616 | **0.617** | +0.2 % |
| C1 whole request 1K / 8K / 32K, tok/s | 140.8 / 54.3 / 21.9 | 185.3 / 73.3 / 29.0 | +31.6 / +34.9 / +32.3 % |
| Decode round 1K / 8K (RoundBench) | 34.23 / 35.01 ms | 25.73 / 26.46 ms | −24.8 / −24.4 % |
| 32K prompt to first token (*computed*) | ≈35.3 s | ≈26.2 s | −26 % |
| Board power, lane mean | 247.1 W | 325.2 W | +31.6 % |
| SM clock, lane (median per call) | 1,029 MHz | 1,511 MHz | +46.8 % |
| GPU temperature, lane (median / max) | 67 / 68 °C | 77 / 83 °C | +10 °C |
| Fan, lane (median / max) | 71 / 77 % | 75 / 79 % | +4 points |

## How

| Lever | Measured |
|---|---|
| DFlash2 draft: one pass proposes 7 tokens from the target's hidden states | 1.00 → 3.39 tokens / round, 29.5 → 98.9 tok/s (1K, 250 W, ~34 ms / round either way) |
| 8-row token tree: 7 nodes best-first; commit the longest matching root path + 1 | 3.39 → 3.89 tokens / round (1K); lane 111.85 → 122.33 tok/s (250 W); round time unchanged |
| 1-8 verify rows share one weight pass (16-row tensor-core tiles) | 16 rows: +21 % per verify forward |
| 4-bit EXL3 weights | 15.4 GiB target + 1.2 GiB draft |
| 41 engine patches, each bit-exact or numerics-gated | per area below |
| 350 W cap, quiet fan curve (≤ 80 % to 84 °C) | lane +32.3 % tok/s at equal tok/J (0.497 vs 0.495); fan 75 vs 71 % median ([250 W vs 350 W](#250-w-vs-350-w)) |
| Memory clock: stock at 350 W (−1500 MHz was best at 250 W) | 350 W, vs −1500: 0 = +5.4 % tok/s, +6.4 % tok/J; −2000: −1.8 % ([sweep](docs/benchmarks.md#3-power)). 250 W: −1500 saved 1.0-1.4 ms / round (*computed* sum of two sweeps); core offsets: none (+225 MHz: Xid 109) |

<details><summary>45 engine patches</summary>

| Area | Patches | What | Measured when kept |
|---|---|---|---|
| Verify loop | exl3 0001-0003, 0005, 0006 | batched greedy verify via the Bend acceptor, host/GPU overlap, draft CUDA graph, DFlash2 block mask, tree verify via the proved tree acceptor | |
| Token tree | 9008, 3012, 5109, 3013 | GPU tree builder, ancestor-masked verify attention, GDN along each row's ancestors, commit of the accepted path's K/V | lane 111.85 → 122.33 tok/s |
| Layer tail + MLP | 2001, 7001, 8201, 8202, 8204, 8205b, 2106, 2107, 2113 | M ≤ 16 GEMMs, fused persistent MLP, one kernel per layer tail, weighted split-K, instruction diets, L2 discard of dead split-K partials | 2106/2107: −13.3 µs / layer; 2113: −0.40..−0.50 ms / round |
| Projections | 2102, 2105, 9003b | grouped m16 qkv(+z) GEMMs, target and draft | |
| Attention | 3001-3007, 3010, 3020, 3021c | exact dequant, GQA split, row-invariant strided verify attention, CUDA prefill attention, 8 warps per prefill CTA (same per-element order), int8 Q·Kᵀ in prefill (the 3-bit K codes used exactly, Q quantized per row) | 3010: 262,136-token prefill 722 → 537 s (250 W); 3020: 262K first token 435.0 → 365.9 s; 3021c: 362.5 → 322.0 s (350 W) |
| Gated DeltaNet | 0001-0004, 5001, 5101, 5106, 5108, 5111 | history-free verify, commit replay, fused conv/recurrence/norm, b/a K-split, replay gather, prefill conv reads the fp32 projection | 5108: replay 620 → 402 µs (8 tokens); 5111 + 5112: 262K first token 365.9 → 362.5 s |
| Draft | 6001, 9002, 9005c | draft graph, int4 draft head, head pruned to 896 blocks | full head: −2.7 % tok/s (1K) |
| Prefill GEMM | 3011, 5112 | fp16-accumulate wide tiles; SiLU · up applied in the gate/up GEMM store | −21.6 % per 2,048-row chunk |

</details>

<details><summary>Where one verify round goes (CUPTI trace, <code>cs12</code>, 250 W: 31.77 ms / round, 6.17 tokens / round)</summary>

| Phase | ms / round | % | Note |
|---|---|---|---|
| Layer tails: out projection + residual + norm + MLP | 17.10 | 53.8 | 64 fused persistent kernels (261.5 µs each) + pre-mixer norms |
| Gated DeltaNet input projections (qkv + z) | 4.15 | 13.0 | 48 grouped m16 GEMMs |
| Draft: forward, int4 head, walk, KV refresh | 3.02 | 9.5 | |
| Output head (248,320 × 5,120, 6 bpw) | 1.69 | 5.3 | |
| Gated DeltaNet conv + recurrence + norm | 1.54 | 4.8 | 48 fused kernels |
| Attention qkv projections | 1.23 | 3.9 | 16 grouped m16 GEMMs |
| Attention (split + combine + pre) | 0.52 | 1.6 | 1.50 ms at 8K context |
| GDN commit replay + conv rewind | 0.44 | 1.4 | |
| Sampler + copies | 0.05 | 0.2 | |
| GPU idle | 1.66 | 5.2 | lead, host-late gaps, gaps < 10 µs |

- ≈ 14 GB of quantized weights read per round (shape-derived).
- At 250 W, time per round tracks energy per round (8.5-8.7 J), not DRAM bandwidth.
- At 350 W the round is memory-bandwidth sensitive: stock memory clock +5.4 % tok/s vs −1500 MHz ([sweep](docs/benchmarks.md#3-power)).

</details>

## History

Kept lane runs. Compare within one protocol only.

| Run | Image | Protocol | Power | tok/s | tok/J |
|---|---|---|---|---|---|
| #53 | g7kafqt | v1: AIME 2025 ×3 + C1 (math only) | 350 W | 157.49 | 0.494 |
| #56 | cs5 | v2: AIME ×3, MMLU-Pro ×20, I3 Logic ×6, LiveCodeBench ×3 + C1 | 350 W | 144.13 | 0.444 |
| #57 | c0 (cs5 stack) | v3: v2 with MMLU-Pro ×10, I3 Logic ×4 | 250 W | 101.46 | 0.411 |
| #58 | cs10 | v3 | 250 W | 106.13 | 0.432 |
| #60 | cs10 | v4: v3 + declared memory offset −1500 MHz | 250 W | 107.03 | 0.433 |
| #61, #62 | cs11 | v4 | 250 W | 109.65, 109.81 | 0.444, 0.444 |
| #63 | cs12 | v4 | 250 W | 111.85 | 0.452 |
| #67 | tree3s | v4 | 250 W | 122.33 | 0.495 |
| **#68** | **tree3s** (decode stack of the live image) | v5: v4 tasks at 350 W, memory offset 0 | 350 W | **161.79** | **0.497** |

Cold prefill (protocol `exl3-native-prefill-ttft-v1`, 350 W): geometric mean of prompt tokens / time to first token over 8K, 32K, 128K and 262K prompts.

| Run | Image | Stack | Prefill tok/s | 32K / 262K first token |
|---|---|---|---|---|
| #70 | tree3s | #68 | 977.9 | 25.8 / 435.0 s |
| #71 | p3020 | + 3020 | 1,072.8 | 24.4 / 365.9 s |
| #72 | p3020f | + 5111, 5112 | 1,087.8 | 24.1 / 362.5 s |
| **#73** | **p3021p (live)** | + 3021c (int8 Q·Kᵀ in prefill; outputs change slightly, [quality checks](docs/benchmarks.md#8-segment-12-cold-prefill-exl3-native-prefill-ttft-v1-350-w-2026-09-29)) | **1,157.3** | **23.3 / 322.0 s** |

Every change and every dropped attempt: [docs/benchmarks.md](docs/benchmarks.md).

## Run it

1. Weights (pinned below) into a private `QWEN_STATE_ROOT`: `models/qwen38-27b-exl3/`, `models/dflash2-exl3/`, `cache/`, `api-key`, launch lock ([docs/docker.md](docs/docker.md)). Any byte mismatch: refuses to start.
2. Docker Compose + NVIDIA Container Toolkit (CDI `nvidia.com/gpu=0`); authenticated base image present locally.
3. Build and start:

```sh
bash docker/build-exl3.sh candidate-ext qwen-inference:exl3   # or: baseline, candidate
export QWEN_STATE_ROOT=/absolute/path/to/state QWEN_IMAGE=qwen-inference:exl3
export QWEN_ALLOW_UNQUALIFIED=1
docker compose --project-name qwen-inference up --no-build --pull never --detach --wait
```

- API: `http://127.0.0.1:18020/v1`, model `qwen3.8-27b`. Never beside another inference service.
- Docker owns runtime and restarts; Nix pins tools, the `.#bend` toolchain and a Compose adapter ([nix/STANDALONE.md](nix/STANDALONE.md)).

## Measure

```sh
bash autoresearch.sh   # the lane; protocol: docs/benchmarks.md; scoring: Prime Envs + Verifiers (eval/README.md)
curl -o gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
python3 -m bench.gsm8k_compare --api-key-file /path/to/api-key --data gsm8k-test.jsonl --out gsm8k.json
```

## Setup

| | |
|---|---|
| GPU | RTX 3090 24 GiB (GA102, SM86), VBIOS 94.02.42.80.1F, PCIe 4.0 ×16, driver 595.71.05 |
| Power, clocks | **350 W** cap since 2026-09-28 (250 W before); core +0; memory +0 (stock; −1500 MHz at 250 W); NixOS `nvidia-quiet-power-limit.service`; the lane checks all three via NVML |
| Fans | CoolerControl: 70 % at 70 °C, 77 % at 80 °C, 80 % at 84 °C, 100 % at 90 °C; above 83 °C the card lowers its clocks |
| Under load, 350 W | lane (#68): 325.2 W mean; SM per-call mean 1,240-1,618 MHz (median 1,511); 77 °C median, 83 °C max; fan 75 % median, 79 % max. 60-min sustained soak (fine-tune data job): 81 °C median, 84 °C max; fan 79 % median, 80 % max; thermal slowdown 4 of 657 samples |
| Under load, 250 W (#67) | 247.1 W mean; SM per-call mean 957-1,194 MHz (median 1,029); 67 °C median, 68 °C max; fan 71 % median, 77 % max (old curve) |
| Host | Ryzen 7 5800X (8 cores / 16 threads), 125.7 GiB, NixOS 26.05, Linux 6.18.50 |
| Runtime | rootless Docker 29.7.2, CDI, read-only root; Ubuntu 24.04 CUDA base; Python 3.13.10, PyTorch 2.10.0+cu130, CUDA 13.0.96, cuBLAS 13.1.0.3, Triton 3.6.0 |
| Engine | ExLlamaV3 1.5.0 `355c6ee` (r0b0tlab `community`, native DFlash2) + 5 [`patches/exl3`](patches/exl3) + 36 [`patches/exl3-ext`](patches/exl3-ext), SHA256-pinned |
| Server | [`serve/exl3_server.py`](serve/exl3_server.py): authenticated OpenAI-compatible `/v1` chat/completions + tool calls, greedy, one sequence |
| Target | [`r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw) @ `3f1771b8` (`Qwen/Qwen3.8-27B`): 48 Gated DeltaNet + 16 full-attention layers, hidden 5,120, vocab 248,320; 4.00 bpw, 6 bpw head; 16.5 GB |
| Draft | [`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw) @ `265b5240` (`incoai/Qwen3.8-27B-DFlash2`): 5 sliding-attention layers, block 8, reads target layers 5/19/33/47/61, top-16 selector; 1.25 GB |
| Recipe | [`serve/exl3-entrypoint.sh`](serve/exl3-entrypoint.sh): context 262,144, cache 270,336, 3-bit KV; 8 verify rows / round (anchor + 7 tree nodes); every file rehashed against [`prepare/exl3-manifest.json`](prepare/exl3-manifest.json) at start |

## Limitations

```
- Unqualified: 262K prompts run only in the prefill lane (one 262,000-token prompt per run); sustained 262K capacity, quality and speed are not qualified.
- Prefill attention computes Q·Kᵀ in int8 (3021c): outputs differ slightly from the fp16 route (KL inside
  the exact-numerics noise floor; broad suite equal except one I3 task that hit the token cap).
- Greedy only, one sequence at a time.
- Chat stream=true is buffered SSE: first event is not TTFT.
- Exact logit ties can depend on max_tokens (cs12: token 39457 at 8192 vs 54185 at 256,
  p = 0.28775 each; normal and all-rejected draft agree). Compare at equal request params.
- Speed varies with the text (3.37-6.69 tokens / round), host CPU load and card temperature.
- Quality: lane samples only. Full Prime Envs suite and 4-bit vs BF16 loss: not measured.
- Long-context reasoning with 3-bit KV vs fp16 KV: not measured.
- GDDR6X temperature is not readable on this card; memory runs at the stock clock.
```

## In progress

| Work | Status | Measured so far |
|---|---|---|
| Draft fine-tune on agent traffic (tool calls, SWE turns) | data done: 2,869 prompts, 1.66M tokens, disjoint from all eval sets; training next | pilot, live engine: agent tokens / round +4.22 % (95 % CI +3.14..+5.65), control −0.01 % (−0.66..+0.60); generic self-distillation +0.23 % (−0.13..+0.62, replay estimate): dropped |
| Draft precision 4 / 5 / 6 / 8 bpw | queued | — |
| Prefill: merge aligned 2,048-row chunks into 4,096 (5110) | 262K error was out-of-memory in the first verify round; fix under test | 32K / 128K: identical outputs, first token ×0.974 / ×0.970 |
| Prefix cache kept across server restarts | built; restart test passed 9/9; identity test queued | next turn after a restart: 34K 1.3 s (cold 26.5 s), 262K 3.3 s |
| int8 P·V in prefill attention; int8 MLP in prefill | built; quality gates queued | int8 MLP (all linears): 32K 24.4 → 17.4 s, failed quality; MLP-only variants under test |

## References

- Protocol, every change, every dropped attempt: [docs/benchmarks.md](docs/benchmarks.md)
- Architecture and the Bend proof boundary: [docs/architecture.md](docs/architecture.md)
- Deployment: [docs/docker.md](docs/docker.md)
- Evaluator: [eval/README.md](eval/README.md)
- All docs: [docs/README.md](docs/README.md)
