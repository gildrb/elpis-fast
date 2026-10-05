#!/usr/bin/env bash
# Every Python gate: ruff, format and ty on the host, ty in the serving image
# and in the eval/direct venv. Run inside `nix develop`.
#   bash check.sh IMAGE
# IMAGE: an image that docker/build-exl3.sh candidate-ext built from this checkout.
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: bash check.sh IMAGE" >&2
  exit 2
fi
image="$1"
root="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
cd -- "$root"
run=(uv run --locked --python python3.13 --no-managed-python)
"${run[@]}" ruff check .
"${run[@]}" ruff format --check .
"${run[@]}" ty check .
docker run --rm --network none -v "$root:/w:ro" -v "$root/.venv/bin/ty:/usr/local/bin/ty:ro" \
  -w /w --entrypoint ty "$image" check --python /opt/venv/bin/python3 \
  serve/exl3_server.py bench/exl3_accept_latency.py
eval/direct/setup --check
"${run[@]}" ty check --python eval/direct/mrcr/.venv --python-version 3.12 eval/direct
