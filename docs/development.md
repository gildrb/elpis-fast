# Strict Python development

Use the pinned Nix dev shell and its Python 3.13. uv owns the project
virtual environment and lock. Do not install tools in an agent/global Python.
Use the Git-backed `.` flake reference for development. `path:.` includes ignored
evaluation datasets and caches in its source snapshot, which can be many GB.
New files referenced by the flake must be tracked before Git-backed evaluation;
do not stage unrelated user changes. Evaluation scripts run from the checkout
and are not embedded in the development shell.

```console
nix develop . --no-write-lock-file -c uv sync --locked --python python3.13 --no-managed-python
nix develop . --no-write-lock-file -c uv run --locked --python python3.13 --no-managed-python ruff check .
nix develop . --no-write-lock-file -c uv run --locked --python python3.13 --no-managed-python ruff format --check .
nix develop . --no-write-lock-file -c uv run --locked --python python3.13 --no-managed-python ty check .
```

Add `--offline` after `uv run` once the locked environment is available.
Ruff 0.16.10 and ty 0.0.80 are exact development pins. numpy 2.5.3 and
tokenizers 0.23.2 match the serving image (`docker/base/requirements.lock`)
so ty resolves them. Serving runtime pins are separate from the upstream
evaluation environment pins described in [eval/README.md](../eval/README.md).

`ty check .` skips the files that import torch, exllamav3, JSON Schema or
Verifiers (`[tool.ty.src] exclude`). They are checked in their own
environments, not skipped. `bash check.sh IMAGE` runs every gate: ruff, format
and ty on the host, ty in `IMAGE` (an image that
`docker/build-exl3.sh candidate-ext` built from this checkout) and ty in the
`eval/direct` venv (`eval/direct/setup` makes it). CI
(`.github/workflows/lint.yml`) runs the host gates on every push to `main` and
every pull request.

## Policy

Ruff selects ALL rules, including preview rules, targets Python 3.13 and
formats docstring code. Rule names are used because the pinned Ruff's
RUF201 rejects rule codes in selectors. ty sets every diagnostic to error
and treats any remaining warning as a failed check. The installed ty has
`--error all` and `--error-on-warning`; it has no `--strict` switch.
These settings do not prove that every dynamic Python boundary is typed.

The only lint choices are mutually exclusive docstring conventions and
formatter-owned styles. D211 (no blank line before class docstrings) wins
over D203; D212 (summary on first line) wins over D213. Following the pinned
Ruff rule documentation's formatter-compatibility advice, the formatter owns
indentation (E111, E114, E117, W191, D206), quotes (D300, Q000, Q001, Q002,
Q003), and trailing commas (COM812, COM819). Their named lint counterparts
are disabled, not source-level errors. ISC001 and ISC002 remain enabled;
the default multiline concatenation setting is formatter-compatible.
One rule is off: `suspicious-subprocess-import` (S404) flags every
`import subprocess`.

No file exclusions, per-file exemptions, type ignores or workarounds. A
finding is fixed in the code. The one exception is S603
(`subprocess-without-shell-equals-true`): it flags every call whose argv is
not all string literals, so no compliant form exists for a tool that runs
compilers. Every process launch goes through one audited helper, and only the
helper's call carries `# ruff: ignore[subprocess-without-shell-equals-true]`
with its reason: `bend/source_link.py` `run` (every source link),
`bench/process.py` `run` and `start` (bench and eval), and the single call in
each of `bend/exl3_build.py`, `bend/build_toolchain.py` and
`docker/base/lock.py`. Each helper validates argv: a list, an absolute
executable, no shell, explicit `check`.
`grep -rn 'ruff: ignore' --include='*.py' .` lists them all.

## Scope and honest failures

Run checks read-only before editing. Default checks cover authored Python.
Canonical serving uses the existing immutable EXL3 engine image plus the baked
`serve/exl3_server.py`, not runtime source overlays. The guarded
`docker/build-exl3.sh` `baseline` build preserves that engine (`candidate` applies
the SHA-pinned `patches/exl3` series) and adds the hash-pinned
JSON Schema dependencies in `serve/exl3-requirements.txt`. The Python runtime
inside that image is separate from the repository's development environment.

The CPU development environment does not install serving torch, EXL3 or
transformers. The files that import them are type-checked in the serving
image (see above). Bend sources are checked with the pinned
`.#bend` toolchain (see [README, Prove](../README.md#prove)), not the Python gates;
`nix run .#bend-verdict -- PROOF.bend --verdict` rechecks them with Bend's
Lean-proven kernel (same Bend, plus the pinned Lean 4.34.0 from `nix/lean4.nix`).

Upstream evaluation environments use their separate setup and locked
dependencies under `eval/`. Keep full check logs private and group authored
findings by file when assigning cleanup. Do not claim passing quality gates
while diagnostics remain.

## Source links (`bend/*_diff.py`)

A source link compares a Bend model with the engine source text. It is
evidence for `H_conform`, not a proof. Run every link from the repository root
inside the dev shell. The shell supplies `bend` 2.0.35, `c++` (clang 19),
`patch` and Python 3.13. Each link resolves `bend` from `PATH` and stops if
`bend version` is not exactly `bend 2.0.35` (`bend/source_link.py`). Every
process a link starts goes through `source_link.run`, without a shell, with the
executable resolved to an absolute path. Compilers and `bend` runs take the
host CPU lock when `$XDG_RUNTIME_DIR/elpis-gpu.lock` exists, is a regular file
you own, and neither group nor others can write it: the link waits while
`$XDG_RUNTIME_DIR/elpis-gpu.pending` exists (a GPU timing window is queued),
holds a shared `flock` on the lock file while the child runs, and lowers
itself, and so every later child, to nice 19. Without such a lock file the
links run them directly.

### 1. Make the engine trees

```console
nix develop --offline --no-write-lock-file -c python3 -I -B bend/engine_trees.py OUT
nix develop --offline --no-write-lock-file -c python3 -I -B bend/engine_trees.py OUT3005 --through 3005-attn-row-invariant-split.patch
```

- Input: the ExLlamaV3 `355c6ee` commit tarball. `docker/base/sources.lock`
  (`archives.exllamav3`) pins its URL and SHA-256. The script downloads it over
  HTTPS. With `--archive FILE` it reads a local copy instead, for example the
  file that `docker/fetch-base.sh` stores under `build/base-inputs/`. The
  SHA-256 must agree before the script reads the archive.
- `OUT/stock`: the stock `exllamav3` package directory. The script checks the
  extension tree against `patches/exl3-ext/exl3-ext.json`.
- `OUT/patched`: `OUT/stock` plus `patches/exl3/series`, then
  `patches/exl3-ext/series`, with every patch hash, pre-image and post-image
  checked (the same checks as the image build).
- `--through PATCH`: stop after that `patches/exl3-ext` patch. Some links
  document an earlier kernel. For a partial series the script checks the patch
  hashes and the pre-images only; the manifests have no partial post-images.
- On any error the script stops and keeps no output.

### 2. Run the links

Prefix each command with
`nix develop --offline --no-write-lock-file -c python3 -I -B`. `OUTn` is a tree
made with `--through` the patch whose number is `n`.

| Link | Arguments | Notes |
|---|---|---|
| `attn_pre_diff.py`, `attn_stride_diff.py`, `pattn_sched_diff.py`, `gdn_ba_ksplit_diff.py`, `mlp_m16_defer_diff.py`, `gemm_m16_wpart_diff.py`, `m16_discard_diff.py` | `OUT/patched` | |
| `pattn8_sched_diff.py`, `act_fuse_diff.py`, `gdn_conv_qkv_diff.py`, `prefill_nosync_diff.py` | `OUT/patched` | `pattn8_sched_diff.py` also checks 3021c's `pattn8i_kernel.cuh` schedule. `prefill_nosync_diff.py` runs the engine's `RecurrentCache` against the Bend trace. |
| `pattn8i_int_diff.py` | `OUT/patched` | Builds and runs host C with the shell's `c++` (exhaustive i2f and fp16-scale checks). |
| `pattn8i_pipe_diff.py` | `OUT/patched` | 3022's ping-pong loop, prologue, stage addressing and shared-memory defines against `bend/pattn8i_pipe.bend`; reads the 3022 patch to check it changes only schedule lines. |
| `prefill_merge_diff.py` | `OUT9502/patched` | 5110h replaces `M4096_MAX_PROMPT = 131072`. |
| `prefill_membound_diff.py` | `OUT5110h/patched` | 9503b changes the stage bound. |
| `qc_staging_diff.py` | `OUT/patched` with 9503f and 9503b in the series | Evaluates the engine's `qc_staging_pages` against the Bend model. |
| `tree_pipe_diff.py` | `OUT/patched` | Pins 9601's stage kernel, host upload, readback check and settle (including the GDN conv window save / restore). Proof gate: `nix run .#bend-verdict -- bend/tree_pipe_gate.bend --verdict`. |
| `draft_mask_diff.py` | `--engine OUT/patched` | |
| `m16_diet_diff.py`, `m16_diet2_diff.py` | `--tree OUT/patched` | The 2106 / 2107 patch defaults to the tracked file. |
| `m16_wsched_diff.py` | `--tree OUT/patched` | Reference: `bend/gen/m16_wsched_ref.py`. |
| `hgemm_wide_diff.py` | `OUT/patched/exllamav3_ext` | |
| `gdn_replay_diff.py` | `OUT/patched OUT/patched/cache/recurrent_util.py` | |
| `attn_chunk_diff.py` | `OUT3003/patched` | 3003 kernel; 3006 replaces it. |
| `attn_rowinv_diff.py` | `OUT3005/patched/modules/attention_fn/triton_paged.py` | The file must have the pinned 3005 hash. |
| `gdn_replay_gather_diff.py` | `OUT5108/patched` | 5108 kernel; 5109 changes one quoted line. |
| `norm_fuse_diff.py` | `OUT/stock` | The link applies 7001 itself. |
| `mlp_m16_sched_diff.py`, `tail_m16_sched_diff.py` | none | Inputs are tracked files. |
| `draft_head_idmap_diff.py` | `OUT/patched` | Needs numpy, see below. |

`draft_head_idmap_diff.py` runs the engine's Python with a numpy stand-in for
torch. The dev shell has no numpy. Use the numpy of the pinned nixpkgs
(`flake.lock`). If it is not in the local store, Nix fetches it once from the
signed binary cache:

```console
nix develop --offline --no-write-lock-file -c nix shell --impure --expr 'let p = (builtins.getFlake "git+file://${toString ./.}").inputs.nixpkgs.legacyPackages.x86_64-linux; in p.python313.withPackages (ps: [ ps.numpy ])' -c python3 -I -B bend/draft_head_idmap_diff.py OUT/patched
```

### Inputs outside the repository

- `draft_head_idmap_diff.py --order-json FILE`: the full block order from a
  corpus run. The repository cannot make it.

### Links that do not pass today

- `attn_split_diff.py` quotes an `av_split_len` revision of 3003 that no
  tracked patch contains. No tree passes. Its model `bend/attn_split.bend`
  documents that earlier, round-relative split. `bend/attn_chunk.bend` models
  the tracked 3003 and `bend/attn_stride.bend` the 3006 kernel.

`gemm_m16_group_diff.py` takes the tracked 2102 patch:
`python3 -I -B bend/gemm_m16_group_diff.py patches/exl3-ext/2102-proj-m16-grouped-v2-on3003-5101.patch`.

### Generators (`bend/gen/`)

| Generator | Output | Check |
|---|---|---|
| `gen_table.py --out-dir D` | `exl3_tree_table.bend`, `EXL3_TREE_ACCEPT.bend`, `EXL3_TREE_ACCEPT_SPEC.bend` | `cmp D/<file> bend/<file>` |
| `roofline_impl.py` (input `roofline_inventory.json`, made by `roofline_inventory.py` from the model files) | `roofline.bend` on stdout | `cmp` with `bend/roofline.bend` |
| `mlp_m16_sched_ref.py`, `m16_wsched_ref.py` (`wpart.py`) | independent references | used by the links above |
