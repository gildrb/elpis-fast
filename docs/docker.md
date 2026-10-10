# Canonical EXL3 Docker deployment

`Dockerfile.exl3`, `docker/build-exl3.sh`, `docker-compose.yml` and
`serve/exl3-entrypoint.sh` define the serving recipe. Nix consumes the same
prebuilt-image Compose configuration; it does not provide another engine.

The fixed recipe is Qwen3.8-27B EXL3 plus native DFlash2, one sequence, CQ3,
context **262144** and cache **270336**. These settings are not a full-context
capacity, performance or model-quality qualification. There is no smaller-context,
target-only, different-KV or alternate-engine fallback.

## Prepare and build

1. Install Docker Compose v2, Buildx, a compatible NVIDIA driver and NVIDIA
   Container Toolkit with CDI `nvidia.com/gpu=0`. Drivers, GPU reservation,
   storage, fans and power remain operator-owned.
2. Supply an existing canonical absolute private `QWEN_STATE_ROOT` containing
   `models/qwen38-27b-exl3/`, `models/dflash2-exl3/`, private writable `cache/`,
   an operator-owned mode-0700 `prefix-cache/` directory and an operator-owned
   mode-0600 regular `api-key`. The key is 1..4096 bytes, printable ASCII without
   whitespace, with at most one trailing newline. Compose creates none of these
   paths. Startup neither downloads nor converts models.
3. Create `qwen-inference-launch.lock` only if absent, with exclusive creation
   and mode 0600. Never replace or unlink its inode while any deployment can
   exist. Reuse the existing shared lock when migrating an occupied service.
4. For rootless Docker, `QWEN_CONTAINER_USER=0:0` maps to the operator. For
   rootful Docker, set the state owner's numeric `UID:GID`; do not weaken state
   permissions to work around an ownership mismatch.
5. Build the base image from source. Network access is used only in step 5a.
   a. `bash docker/fetch-base.sh` pulls the three pinned images by digest
      (CUDA 13.0.0 devel and runtime, uv 0.9.15) and downloads every file that
      `docker/base/sources.lock` names into `build/base-inputs/`: the exllamav3
      commit tarball (355c6ee), the CPython 3.13.10 build that uv installs, the
      Ubuntu `.deb` files from a fixed snapshot.ubuntu.com time, the base wheels
      (torch and CUDA wheels from download.pytorch.org/whl/cu130, all other wheels
      from PyPI) and the serve wheels. Each file must match its sha256, else the
      script stops.
   b. `bash docker/build-base.sh` builds `docker/base/Dockerfile` with no network
      in any RUN, `SOURCE_DATE_EPOCH` from the checked-out commit,
      `rewrite-timestamp=true` and no attestations. It tags `qwen-elpis:exl3-base`
      only if the image's content manifest equals the pin in
      `docker/base/engine-manifest.json`.

   The content manifest (`docker/base/manifest.py`) is the sha256 of a sorted
   listing: the installed Debian packages with versions, and the sha256, mode
   and path of every file and symlink under `/opt/uv-python` and `/opt/venv`
   (without `__pycache__`). nvcc does not give byte-identical output from build
   to build: nvcc puts its process ID into local symbol names, and cicc orders
   registers by memory layout (ASLR), which changes the PTX of some
   `gdn_conv_rule_norm_kernel` instances. ptxas also writes source mtimes into
   the `-lineinfo` tables; `patches/exl3-ext/ext.py` sets fixed mtimes. A fixed
   layout needs ASLR off, and Docker's default seccomp profile blocks that in
   RUN steps. So the compiled `exllamav3_ext` shared object and the exllamav3
   `RECORD` (which holds its hash) are not part of the pin. The base build
   records their sha256 in `/opt/elpis-base-manifest.json`, and
   `patches/exl3-ext/ext.py` requires the installed shared object to match that
   record. To check a base by hand, run `manifest.py check` in it:

```sh
docker run --rm --network none --user 0:0 \
  --mount type=bind,source="$PWD/docker/base/manifest.py",target=/tmp/manifest.py,readonly \
  --entrypoint /opt/venv/bin/python qwen-elpis:exl3-base \
  -I -B /tmp/manifest.py check /opt/elpis-base-manifest.json
```

   `docker/base/lock.py` wrote `requirements.lock` and `sources.lock` from
   `requirements.in`, `apt.in` and `serve/exl3-requirements.txt`. It needs network
   access and is not part of a build. Run it only to change an input.

```sh
bash docker/fetch-base.sh       # the only step with network access
bash docker/build-base.sh       # tags qwen-elpis:exl3-base
# CPU image build, not a GPU launch:
bash docker/build-exl3.sh candidate-ext qwen-inference:exl3   # or: baseline, candidate, candidate-rebuilt
export QWEN_STATE_ROOT=/absolute/private/qwen-state
export QWEN_IMAGE=qwen-inference:exl3
docker compose --project-name qwen-inference config --quiet
```

Use the build scripts, not `docker compose build` or a direct Dockerfile build.
`build-exl3.sh` first builds the `base-manifest` target: `manifest.py check` runs
in the base and its content sha256 must equal the pin. The value becomes the
`io.elpis.exl3.base-manifest-sha256` label of the image. The script also requires
the base tag to name the same image before and after the `--pull=false` build with
the daemon-backed default builder, checks the baked labels and only then assigns
the output tag. It refuses an output tag that names the base. The operator must
exclusively control image tags during the build: the before/after check detects
ordinary retagging, not a hostile Docker operator. Compose has no build stanza and
uses `pull_policy: never`.

The `baseline` target retains the base's installed native EXL3/DFlash2 engine
unchanged. The `candidate` target additionally applies the SHA-pinned
`patches/exl3` series to the installed ExLlamaV3 and embeds the Bend acceptance
artifacts (recorded by `/opt/qwen/exl3-patches.json` and image labels). The
`candidate-ext` target is `candidate` plus `exllamav3_ext` recompiled for sm_86
from the pinned `355c6ee` sources with the SHA-pinned `patches/exl3-ext` series;
`candidate-rebuilt` recompiles the same sources unpatched, as the toolchain
control. Both extension targets build in two phases: the first exports the
composed engine manifest, whose SHA-256 becomes the final image's
`io.elpis.exl3.patches-sha256` label. All targets bake the standalone
`serve/exl3_server.py`, startup guard, healthcheck and model inventory, rather
than mounting server code from `/tmp`. No RUN step has network access: the
hash-pinned JSON Schema wheels of `serve/exl3-requirements.txt` come from
`build/base-inputs/serve-wheels/`. The extension stages use the devel image of the
base build stage with its own g++, not an apt install. Serving uses offline
Hugging Face/Transformers settings and disables telemetry. Torch and
transformers are never rebuilt or replaced; only the extension targets replace
the installed `exllamav3_ext` shared object.

`prepare/exl3-manifest.json` records engine/model revisions and SHA256 identities.
Every start authenticates all 13 target and three draft runtime files, including
weights, config, quantization, tokenizer, merges and chat template. The verifier
rejects symlinks, unexpected loadable files, mismatches and changes during hashing.
Only inert publication files and a real `.cache` directory are allowed outside
the inventory. Model-byte identity is not a numerical or quality qualification.

## Ordinary Compose ownership and lifecycle

Do not launch beside another inference service on the same GPU or port. After
explicit exclusive-GPU approval:

```sh
export QWEN_ALLOW_UNQUALIFIED=1
docker compose --project-name qwen-inference up --no-build --pull never \
  --detach --wait --wait-timeout 1200
```

The acknowledgment permits operation without full qualification. All model,
credential and ownership checks still apply. The EXL3 launcher has one fixed
recipe and rejects retired serving controls instead of ignoring them.

The ordinary image entrypoint is `docker/entrypoint.sh`: it acquires the shared
lifetime lock and execs the baked `/opt/qwen/serve/entrypoint.sh`, which is the
EXL3 launcher in this image. The lock excludes only deployments sharing that
inode, not other states, accounts, daemons or GPU applications.

Compose keeps loopback-only `127.0.0.1:${QWEN_PORT:-18020}`, read-only models,
credential and root filesystem, private writable cache, restricted tmpfs,
`cap_drop: ALL`, `no-new-privileges`, init, an eight-CPU/48-GiB limit,
`restart: unless-stopped` and a 60-second stop budget. The image routes compiler
and library caches into `/cache` and uses CDI's `/usr/local/nvidia/lib64` driver
path. The served model is `qwen3.8-27b` at `http://127.0.0.1:18020/v1`.

The launcher passes `--cpu-cache-gib 8`: a host-RAM page tier (engine
`generator/cpu_cache.py`) of 8 GiB pinned memory, inside the container's 48-GiB
limit. Prefix pages evicted from the 270K-token GPU cache move there and come
back on the next prefix hit, so a long session that another client pushed out
does not re-prefill (`docs/benchmarks.md` §12). The persistent prefix cache
(`exllamav3.generator.persist`, patch 9501b) refuses to run with the tier, so the
launcher no longer passes `--prefix-cache`; Compose still binds `prefix-cache/`.
`--prefix-cache DIR` without the tier keeps the old behaviour: bound to the image
ID (`QWEN_IMAGE_ID`, full `sha256:<64 hex>`), `candidate-ext` only, off with
`QWEN_PREFIX_PERSIST=0`.

The healthcheck authenticates `/health` and the expected `/v1/models` entry. It
runs every 30 seconds, with a 310-second probe budget, 315-second Docker timeout,
20-minute startup grace and three failures to unhealthy. Docker does not restart
a merely unhealthy container. A failed `up --wait` can leave a running container;
inspect it rather than assuming cleanup. A manual stop remains stopped.

## API and tool-call boundaries

The authenticated API supports chat and raw text completions. Generation is
**greedy only** (`temperature=0`), `n=1`, with one native sequence executing at a
time. Input tokens plus the requested output budget must fit 262144. Request
bodies are bounded at 32 MiB. Unsupported fields and invalid option types are
rejected, not silently ignored. This is not a claim of complete OpenAI API parity.

Chat tools use the authenticated model's native template and native XML-like
function-call syntax. Responses expose OpenAI-style `tool_calls`, including JSON
argument strings and call IDs; assistant-call and matching tool-result history
can be submitted for continuation. Clients execute tools; the server does not
execute their functions. Supported selection is `auto`, `none`, `required` or a
named function; `parallel_tool_calls` and the selection are stated in the prompt.
JSON Schema validation is offline. Generation is not constrained, and what the model
writes never fails the request: a parameter value that is not valid JSON or breaks
its schema is returned as the raw string (the call is kept, also when the whole call
breaks the schema; `strict: true` is accepted but not enforced), and markup that
cannot be parsed, an undeclared function, text after a call or an unmet
`tool_choice` / `parallel_tool_calls` returns the decoded text as plain `content`
without `tool_calls` (logged as `[tool] unparsed: <reason>`). Tool `parameters` must be
a direct `type: object` with parameter schemas in its root `properties`,
`patternProperties` or `additionalProperties`. Root `allOf`, `anyOf`, `oneOf`, `not`,
`if`/`then`/`else` and `dependentSchemas` are accepted only when they constrain the
object (for example `required`) and declare no parameter schemas; a root `$ref` is
rejected.

The server matches tool-schema `pattern` and `patternProperties` with the `regex`
module from the base image, not Python `re`. All matches for one response share a
2-second budget. When the budget runs out, the request fails with HTTP 400
`pattern_timeout`. A tool schema that contains both `unevaluatedProperties` and
`patternProperties` is rejected. A client must send its request line and headers
within 60 seconds (this includes keep-alive idle time) and its body within 300
seconds; else the server closes the connection (HTTP 408 for a late body).
A streaming write that the client does not take within 60 seconds ends the stream;
other response writes have no timeout. A request waits at most 7200 seconds for its
generation (HTTP 504 `generation_timeout`). A client that disconnects or stops
reading a stream cancels its job: queued jobs are skipped, a running job stops at
the next generator step (`[serve] request cancelled: client disconnected`). A reset
or closed idle keep-alive connection is closed without a traceback.

Chat `stream=true` is incremental SSE (`X-EXL3-Transport: streaming`): chunked
transfer coding for HTTP/1.1 requests, a close-delimited body for HTTP/1.0. The role
chunk is sent at once; `reasoning_content` and `content` deltas follow as tokens are
generated (a tail that could begin `</think>` or `<tool_call>` waits for the next
tokens). From `<tool_call>` on, the text is held and parsed at the end of the turn;
tool-call deltas, the finish reason, optional usage and `[DONE]` follow. After 10
seconds without a write (queue wait, prefill, held tool calls) the server sends an
SSE comment `: keep-alive`. A failure after the status line is an SSE `error` event
that ends the stream. Usage reports prefix-cache hits as
`prompt_tokens_details.cached_tokens`. Raw completion streaming is unsupported.
Native thinking controls remain
in `chat_template_kwargs`; OpenAI top-level `reasoning_effort` is also accepted:
`none` disables thinking, `minimal` maps to `low`, `high` and `max` map to `xhigh`,
and `low`, `medium` and `xhigh` are unchanged. Invalid values or conflicting
nested controls return HTTP 400. Existing nested OMP controls are unchanged;
reasoning and final content remain separate channels.

## Current persistent live deployment

The current authenticated promotion is `qwen-exl3-serving-12` (2026-10-10), container
`d7826c4b604a0c9d84dee74c1184c5f00dfb0c4a1ded1b9a9113bec242a71a57`, image
`sha256:cad5583694e7e35735d27facc97f9ee5274c25388b7fb0bf9e21f3d4f42f94f6`
(`qwen-inference:exl3-cand-tier1`): the `pfast5` `candidate-ext` engine
(`docs/benchmarks.md` §11) with the server and launcher of commit 3b6b46b (streaming
SSE, cancel on disconnect, 8 GiB host page tier; §12). Startup log:
`host page tier 8 GiB: 1783 pages of 4816896 bytes` (456K tokens). The canonical tag
`qwen-inference:exl3` names this image; `qwen-inference:exl3-previous` is the
serving-11 image `serve-fix1` (`sha256:35957772…`, same engine, server 2ae8d35, no tier).
The persistent configuration is
`/mnt/ssd/storage/ai/qwen3.8-27b/exl3-serving-12/compose.json` (Compose project
`elpis-exl3-serving-12`, network `elpis_default`), alongside unchanged copies of
`launch-gate.py`, `recovery.py` and `operate.py` and its promoted `cutover-window-1/`.
It sequentially reuses `/mnt/ssd/storage/ai/qwen3.8-27b/exl3-serving-1/cache`; preserve
the old state. Promotions are made by
`ADAPTER_REV=<commit> /tmp/elpis-promote.sh TAG IMAGE` (operator tooling outside the
repo), which replays the serving-2 guardian procedure below with automatic guardian
rollback. It pins the adapter to `git show <commit>:serve/exl3_server.py`, checks the
`base-manifest-sha256` label against `docker/base/engine-manifest.json`, runs CPU smokes
adapted to this server, verifies the mise Hermes CLI through its stream-json events,
and replays a captured Autolith request. `PROMOTE_QUIET=<script>` replaces the
quiet-host wait (promotion measures no speed). serving-11 (`serve-fix1`) and serving-10
(container `7b98d3dc…`, image `sha256:bc21628f…`, p3021r server hotfix) are retained
for rollback.
The previous `qwen-exl3-serving-9` (container
`b930391224e8806e925fdd128c34446c52673959ae3451388870b249d48b121a`, image p3021r,
Compose project `elpis-exl3-serving-9`), `qwen-exl3-serving-8` (container
`bf7087ac57f2dad0113149301e7d7e55b0a05e6a78e0b82cd00cd42269a57ab5`, image p3021p,
Compose project `eta-exl3-serving-8`), `qwen-exl3-serving-7` (container
`ebb819cf17a5754f6b9f37c8188d31f73b7e1503597c7eec7ee03d6bfa1611ee`, image p3020fh),
`qwen-exl3-serving-6` (container
`e358fe89dd5d985b60594da0131a0d367d297be740e5accc54a31bea5f608c7f`, image tree3s),
`qwen-exl3-serving-5` (container
`ca966e6f5407ff3bd27cc435c9a9ad90da7d4562ec65a3cd8d7ffe30d1c884a2`, image cs12),
`qwen-exl3-serving-4` (container
`22154417d27bc0ce2455d3f60ff1e3743dd7c40aca8ef0f502450a2463ffa945`, image cs11),
`qwen-exl3-serving-3` (container
`288862e7573f5dec9abb9ecc06bb2d144c9926c1f8fb972efafcadf1fa0a235a`, image cs10), all on
network `eta_default`, and `qwen-exl3-serving-2` (container
`b5e51bc1f6ac85b1b2af8db3612ff300190145397bb48b31e4cb7bf0f2d30f28`, network
`litos_default`) are stopped and retained; their promoted windows still admit them for
rollback, so keep `litos_default` while any retained container uses it.

The original cutover promoted `qwen-exl3-serving-1`, container
`e22faabe5bc9244d88581e4abcab1459d545b82b154b731ea4ccb2de642efb7c`, with image
`sha256:f4bdcfb444f3215e57d23c67edd6929c4afa87eb95f5594d1df7bf438a19ae80`.
Its controls and evidence remain under
`/mnt/ssd/storage/ai/qwen3.8-27b/exl3-serving-1/`. That container is stopped and
retained, not deleted.

This live deployment deliberately differs from an ordinary root Compose launch:
the persistent guardian gate validates candidate identity and its promoted window,
acquires the **same existing shared lock inode**, then execs
`bash /opt/qwen/serve/entrypoint.sh` with the lock inherited. It must not call the
Docker ownership wrapper again and try to acquire a second lock. The server code
is still baked in the image; only host-owned guardian controls are mounted at
`/maintenance-control`, read-only. Preserve the control files, promoted receipt,
operation mutex and launch-lock inode. Do not hand-author authorization states,
replace the gate with a direct launch, or start a competing root Compose/Nix
service. Future maintenance must use the retained deployment and its approved
ownership procedure. Restart configuration is not a demonstrated cold-boot test.

Current private evidence is in `exl3-serving-10/evidence/` under the state root:
`main-verification.json`, `promotion-receipt.json`, `cutover-receipt.json`,
`endpoint-smoke.json`, `live-tool-smoke.json`, `hermes-real-tool-turns.json`,
`hermes-interactive-stream.jsonl`, `omp-smoke.jsonl`, `autolith-request.json`,
`autolith-replay.json`, `installed-exl3-server.py`, `guardian.log` and `thermal.csv`.
The guardian exited 0 with state `promoted_authenticated_main_verified`. Real Hermes
gateway and interactive (2026.9.24) terminal-tool turns, an OMP 18.4.12 read-tool round
trip and the captured Autolith 0.57.0 request (55 tools; HTTP 400 on serving-9) passed
against the new image. After promotion, an interactive Autolith 0.57.0 turn called
`search.content` (the root-`oneOf` tool) and answered from its result. No Telegram
delivery was repeated.

serving-2's evidence (`exl3-serving-2/evidence/`: `main-verification.json`,
`promotion-receipt.json`, `telegram-delivery.json`) records the top-level reasoning
compatibility change, Hermes 0.21.3/0.21.4 and OMP 18.2.11 turns, Telegram API checks
and one approved outbound delivery; **no fresh inbound user-to-bot exchange has been
exercised**. See [client routing](#host-client-routing) and
[verification boundaries](development.md#exl3-cutover-verification-status).

Original `exl3-serving-1/evidence/` retains `promotion-receipt.json`,
`main-promotion-verification.json`, `live-tool-smoke.json` and
`post-promotion-health.stderr`. That smoke exercised a named addition call
(`19 + 23`), client execution and continuation returning `42`, buffered tool SSE,
authentication, model identity and schema-error handling; authenticated health
and runtime CPU protocol proof passed. Neither promotion establishes full-context
capacity, quality or performance qualification. Ruff ALL style findings and host
ty dependency blockers remain; the upstream torch `inference_mode` issue is a
historical known runtime typing limitation, not a current runtime ty rerun.

## Host client routing

OMP's host runtime `~/.omp/agent/models.yml` defines `qwen-local/qwen3.8-27b`
using chat completions, the private key via `!cat`, temperature 0, the existing
nested Qwen thinking dialect and a ten-minute buffered first-event floor.
Explicit `thinking.requiresEffort: false` allows thinking off. The existing
cloud default is preserved: select `omp --model qwen-local/qwen3.8-27b` or choose
that model with `/model`.

Hermes's host runtime `config.yaml` uses local `api_mode: chat_completions` and
temperature 0. Its durable host-owned overlay in
`~/nix/modules/nixos/local-ai-backend.nix` was updated without OS activation;
the pinned installed old profile may still regenerate the old configuration.
Generic dotfiles were not changed. The live gateway 0.21.3 was not restarted;
its persisted session explicitly selects `custom:local/qwen3.8-27b` at
`http://127.0.0.1:18020/v1`. Its original request failed HTTP 400 on the old server
but now works with top-level reasoning compatibility. Interactive 0.21.4 had
previously recovered automatically from the rejected effort; gateway 0.21.3 did
not. These are verified host settings, not defaults installed by root Compose.
