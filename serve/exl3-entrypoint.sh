#!/usr/bin/env bash
# Called with the lifetime ownership lock already held by Docker or the guardian.
set -euo pipefail
if (( $# > 0 )); then
  echo "EXL3 launch arguments are fixed; command overrides are not supported." >&2
  exit 1
fi
if [[ "${QWEN_ALLOW_UNQUALIFIED:-0}" != 1 ]]; then
  echo "This unqualified EXL3 candidate requires QWEN_ALLOW_UNQUALIFIED=1." >&2
  exit 1
fi

if [[ ! -f /app/api_key.txt || -L /app/api_key.txt ||
      "$(stat -c %a /app/api_key.txt)" != 600 ||
      "$(stat -c %u /app/api_key.txt)" != "$(id -u)" ]]; then
  echo "Mount the operator-owned mode-0600 regular API key read-only." >&2
  exit 1
fi

/opt/venv/bin/python -B /model-preparation/verify-models.py \
  --target /models/qwen38-27b-exl3 --draft /models/dflash2-exl3 \
  --representation exl3

echo "UNQUALIFIED EXL3 + native DFlash2: native context262144/cache270336/CQ3; no capacity, quality or performance claim." >&2
# Host-RAM page tier: evicted prefix pages wait in 8 GiB of pinned host memory and come back on a
# prefix hit. The persistent prefix cache refuses to run with the tier on, so /prefix-cache is unused.
exec /opt/venv/bin/python -B /opt/qwen/serve/exl3_server.py \
  --target /models/qwen38-27b-exl3 --draft /models/dflash2-exl3 \
  --model-name qwen3.8-27b --max-model-len 262144 --cache-tokens 270336 \
  --cq 3 --host 0.0.0.0 --port 18020 --cpu-cache-gib 8
