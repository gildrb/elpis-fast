#!/usr/bin/env bash
# Build the EXL3 base image from docker/base/ with no network, after
# docker/fetch-base.sh. The image gets the tag only if its content manifest
# equals the pin in docker/base/engine-manifest.json.
#   bash docker/build-base.sh [<output-image-tag>]   (default qwen-elpis:exl3-base)
set -euo pipefail
umask 077
if (( $# > 1 )); then
  echo "Usage: bash docker/build-base.sh [<output-image-tag>]" >&2
  exit 1
fi
image="${1:-qwen-elpis:exl3-base}"
root="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/..")"
inputs="$root/build/base-inputs"
if [[ ! -d "$inputs" ]]; then
  echo "Missing $inputs: run bash docker/fetch-base.sh first." >&2
  exit 1
fi
# File times in the image are clamped to the time of the checked-out commit.
epoch="$(git -C "$root" log -1 --format=%ct HEAD)"
if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
  echo "Refusing base build: no commit time for SOURCE_DATE_EPOCH." >&2
  exit 1
fi
pinned="$(python3 -I -B -c '
import json, re, sys
pin = json.load(open(sys.argv[1], "rb"))["content_sha256"]
if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin):
    sys.exit("build-base.sh: bad content_sha256 pin")
print(pin)
' "$root/docker/base/engine-manifest.json")"

scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
# The daemon-backed default builder, local images only, no attestations; every
# RUN in docker/base/Dockerfile also sets --network=none.
DOCKER_BUILDKIT=1 docker buildx build --builder default --pull=false --network=none \
  --provenance=false --sbom=false \
  --build-arg "SOURCE_DATE_EPOCH=$epoch" \
  --build-context "base-inputs=$inputs" \
  --file "$root/docker/base/Dockerfile" --target base \
  --output type=image,rewrite-timestamp=true,unpack=false \
  --metadata-file "$scratch/build.json" \
  "$root/docker/base"
# With rewrite-timestamp the image ID is the digest of the rewritten manifest.
built_id="$(python3 -I -B -c '
import json, sys
print(json.load(open(sys.argv[1], "rb"))["containerimage.digest"])
' "$scratch/build.json")"
if [[ ! "$built_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Refusing base build: no output image ID." >&2
  exit 1
fi
# Recompute the content manifest with the repository's manifest.py; it must equal
# the record in the image and the tracked pin.
docker run --rm --network none --user 0:0 --read-only \
  --mount "type=bind,source=$root/docker/base/manifest.py,target=/tmp/manifest.py,readonly" \
  --entrypoint /opt/venv/bin/python "$built_id" \
  -I -B /tmp/manifest.py check /opt/elpis-base-manifest.json > "$scratch/record.json"
actual="$(python3 -I -B -c '
import json, sys
print(json.load(open(sys.argv[1], "rb"))["content_sha256"])
' "$scratch/record.json")"
if [[ "$actual" != "$pinned" ]]; then
  printf 'Refusing base build: content manifest %s differs from the pin %s.\n' \
    "$actual" "$pinned" >&2
  exit 1
fi
docker image tag "$built_id" "$image"
printf 'Built base %s as %s; content manifest %s; per-build record:\n' \
  "$built_id" "$image" "$actual"
python3 -I -B -c '
import json, sys
for path, sha in sorted(json.load(open(sys.argv[1], "rb"))["per_build"].items()):
    print(f"  {sha}  {path}")
' "$scratch/record.json"
