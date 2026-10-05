#!/usr/bin/env bash
# Finite EXL3 + Bend native cold-prefill TTFT measurement; never a deployment/promotion command.
set -euo pipefail
set +x
umask 077
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Usage: bash bench/lane.sh
Protocol exl3-native-prefill-ttft-v1 (suite prefill): cold prefill throughput of
EXL3 + native DFlash2 serving, with the Bend acceptance identity and engine patch
manifest recorded when the image bakes them (explicit null when absent). A new
comparison segment: not comparable to exl3-native-broad-c1-request-v5 or any
earlier segment (different workload and primary); it needs a fresh baseline.
The broad decode suite (exl3-native-broad-c1-request-v5) is unchanged and stays
selectable only as python -m bench.autoresearch --suite broad; this script always
runs --suite prefill.
Required private operator descriptor (no inferred inputs or environment fallback):
  /run/user/1000/elpis-autoresearch-operator.json
Exactly these JSON keys (replace placeholders; schema_version is integer 1):
  {"schema_version":1,"container_id":"<full 64-hex candidate ID>",
   "api_key_file":"/private/api-key",
   "maintenance_directory":"/private/existing-armed-maintenance",
   "output_directory":"/private/parent/new-autoresearch"}
Main atomically installs a NEW uid1000-owned regular 0600 descriptor for each run.
No symlinks; paths must be canonical and absolute. Key mode is 0400/0600.
The window must already be armed; output must not exist. Never edit the descriptor
during a run: supervisor/worker bind its file identity and exact content digest.
Prerequisites: prepared offline eval/.venv (with tokenizers), Docker access
(read-only inspect and one read-only in-container hashing probe via docker exec),
host nvidia-smi, and a healthy owned EXL3 instance at http://127.0.0.1:18020
serving qwen3.8-27b with max_model_len 262144, target/draft mounted under /models,
on one RTX 3090 at 350 W with clock offsets core 0 / memory 0 MHz (host policy;
checked, never set).
Rootless Docker is fixed to unix:///run/user/1000/docker.sock.
The prepared Python supervisor enters pinned offline Nix only for its worker;
Nix startup and owned-process cleanup are inside the whole-command deadline.
Main must already own the maintenance window and perform recovery afterwards.
The deadline is min(2400 seconds, guardian remaining time minus 120 seconds).
There is no retry, row reduction, capacity probe, deployment or promotion.
Workload: a ladder of raw-content depths (served tokenizer, no specials, +-2
tokens), ascending, repetitions consecutive:
  8192 x3, 32768 x3, 131072 x2, 262000 x1   (9 rows)
Each row's content is the nonce line "[prefill measurement R of N at depth D]",
a prefix of the frozen bench/throughput-prompts.jsonl corpus (the C1 prompts
repeated to cover 1.05 x 262000 tokens), a blank line and the fixed C1
instruction. The unique leading nonce makes every row's first 256-token KV page
unique, so every TTFT request is a cold prefill with no prefix reuse (no flush,
no warmup). All request bytes are frozen and rendered through
/v1/chat/completions/render before any generation (rendered + 32 <= 262144).
Per row, sequentially (concurrency 1, greedy, top_p 1, n 1, non-streaming):
  TTFT request          max_tokens 1  (cold prefill + first verify round + HTTP)
  continuation request  max_tokens 32 (same prompt; reuses the prefix just computed)
Wall time is monotonic from request send through complete response body. Streaming
TTFT is unavailable (the server buffers SSE), so TTFT is that 1-token request.
Only complete raw-evidence-admitted measurements print METRIC name=value:
  prefill_tok_s (primary: geometric mean over the four depths of
    prefill_tok_s_<d>),
  prefill_tok_s_<d> (sum of native prompt_tokens / sum of TTFT wall seconds),
  ttft_s_<d> (mean TTFT wall seconds),
  reuse_request_s_<d> (mean continuation wall seconds; informational),
    for d in 8192, 32768, 131072, 262000,
  elapsed_seconds.
No quality, reward, decode throughput or power is measured by this suite.
Artifacts: OUTPUT/{prefill,logs,sources}, identity-before/after.json,
supervisor.json, benchmark.json, admitted.json, measurement.json;
failure.json/worker-failure.json on rejection (nonzero exit, no METRIC lines).
USAGE
    exit 0
fi
[[ $# -eq 0 ]] || { printf 'Use bash bench/lane.sh --help\n' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -x "$root/eval/.venv/bin/python" ]] || { printf 'Prepared eval/.venv is required; no installation is performed.\n' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME QWEN_API_KEY OPENAI_API_KEY DOCKER_CONTEXT DOCKER_TLS_VERIFY DOCKER_CERT_PATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LC_ALL=C
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
export DOCKER_HOST=unix:///run/user/1000/docker.sock
cd -- "$root"
exec "$root/eval/.venv/bin/python" -m bench.autoresearch --suite prefill
