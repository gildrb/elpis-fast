#!/usr/bin/env bash
# CPU-only image build with no network in any RUN. Never pulls or retags the base.
# The base is the source-built qwen-elpis:exl3-base (docker/build-base.sh), or the
# tag in EXL3_BASE_IMAGE. Its identity is its content manifest, not its image ID.
#   baseline          - installed engine unchanged
#   candidate         - baseline + the sha-pinned patches/exl3 series and proven acceptance artifact
#   candidate-rebuilt - candidate + exllamav3_ext recompiled from the unpatched pinned sources
#   candidate-ext     - candidate + exllamav3_ext recompiled with the patches/exl3-ext series
set -euo pipefail
umask 077
variants=(baseline candidate candidate-rebuilt candidate-ext)
valid=0
for candidate_variant in "${variants[@]}"; do
  [[ "${1:-}" == "$candidate_variant" ]] && valid=1
done
if (( $# != 2 || ! valid )); then
  choices="${variants[*]}"
  echo "Usage: bash docker/build-exl3.sh <${choices// /|}> <output-image-tag>" >&2
  exit 1
fi
variant="$1"
image="$2"
base="${EXL3_BASE_IMAGE:-qwen-elpis:exl3-base}"
root="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/..")"
serve_wheels="$root/build/base-inputs/serve-wheels"
if [[ ! -d "$serve_wheels" ]]; then
  echo "Missing $serve_wheels: run bash docker/fetch-base.sh first." >&2
  exit 1
fi
pinned="$(python3 -I -B -c '
import json, re, sys
pin = json.load(open(sys.argv[1], "rb"))["content_sha256"]
if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{64}", pin):
    sys.exit("build-exl3.sh: bad content_sha256 pin")
print(pin)
' "$root/docker/base/engine-manifest.json")"
base_id="$(docker image inspect --format '{{.Id}}' "$base")"
existing_id="$(docker image ls --no-trunc --format '{{.ID}}' --filter "reference=$image")"
if [[ "$image" == "$base" || "$existing_id" == "$base_id" ]]; then
  echo "Refusing to overwrite a tag of the base image." >&2
  exit 1
fi
epoch="$(git -C "$root" log -1 --format=%ct HEAD)"
if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
  echo "Refusing EXL3 build: no commit time for SOURCE_DATE_EPOCH." >&2
  exit 1
fi

# The base tag must name the same image before and after the build. This detects
# ordinary retagging during the build; the content manifest is the identity.
check_base() {
  if [[ "$(docker image inspect --format '{{.Id}}' "$base")" != "$base_id" ]]; then
    echo "Refusing EXL3 build: the base tag moved during the build." >&2
    exit 1
  fi
}

label() {
  docker image inspect --format "{{ index .Config.Labels \"$2\" }}" "$1"
}

build() {
  DOCKER_BUILDKIT=1 docker buildx build --builder default --pull=false --network=none \
    --provenance=false --sbom=false \
    --build-arg "BASE=$base" --build-arg "SOURCE_DATE_EPOCH=$epoch" \
    --build-context "serve-wheels=$serve_wheels" \
    --file "$root/Dockerfile.exl3" "${build_args[@]}" "$@" "$root"
}

patches_sha=""
build_args=()
if [[ "$variant" != baseline ]]; then
  patches_sha="$(sha256sum -- "$root/patches/exl3/exl3-patches.json")"
  patches_sha="${patches_sha%% *}"
  build_args=(--build-arg "EXL3_PATCHES_SHA256=$patches_sha")
  # Proof gate, emitted C, library and admission, all from the pinned flake toolchain;
  # the image build then re-verifies every output against the manifest pins.
  rm -rf -- "$root/build/bend-exl3"
  (cd -- "$root" && nix develop --offline --no-write-lock-file -c \
    python3 -B bend/exl3_build.py --output build/bend-exl3)
fi

# Use the daemon-backed default builder, not an independently selected remote one.
# Require exclusive operator control of image tags throughout this build; the
# before/after checks detect ordinary retagging, not a hostile Docker operator.
check_base
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
# Phase zero: manifest.py recomputes the base's content listing and per-build
# hashes inside the base and requires its own record; the content sha256 must
# equal the tracked pin. It becomes the image's base-manifest label.
build --target base-manifest --output "type=local,dest=$scratch/base"
base_sha="$(python3 -I -B -c '
import json, sys
print(json.load(open(sys.argv[1], "rb"))["content_sha256"])
' "$scratch/base/base-manifest.json")"
if [[ "$base_sha" != "$pinned" ]]; then
  printf 'Refusing EXL3 build: base content manifest %s differs from the pin %s.\n' \
    "$base_sha" "$pinned" >&2
  exit 1
fi
build_args+=(--build-arg "BASE_MANIFEST_SHA256=$base_sha")
if [[ "$variant" == candidate-rebuilt || "$variant" == candidate-ext ]]; then
  # Both phases read the patch tools from one snapshot (named build context
  # `patches`), so concurrent repository edits cannot split them. Phase one
  # compiles the extension and exports the engine manifest composed around the
  # built shared object; its SHA-256 becomes the final image's label, and the
  # final stage recomposes the manifest from the installed object.
  snap="$scratch/patches"
  mkdir -p -- "$snap/exl3" "$snap/exl3-ext"
  shopt -s nullglob
  cp -- "$root"/patches/exl3/{series,exl3-patches.json,apply.py} "$root"/patches/exl3/*.patch \
    "$snap/exl3/"
  cp -- "$root"/patches/exl3-ext/{series,exl3-ext.json,ext.py} "$root"/patches/exl3-ext/*.patch \
    "$snap/exl3-ext/"
  shopt -u nullglob
  # Upstream setup.py comes from the pinned source archive (docker/base/sources.lock);
  # ext.py checks it against exl3-ext.json's setup_py_sha256.
  python3 -I -B -c '
import hashlib, json, sys, tarfile
lock, inputs, out = sys.argv[1:]
pin = json.load(open(lock, "rb"))["archives"]["exllamav3"]
path = inputs + "/" + pin["file"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != pin["sha256"]:
    sys.exit(f"build-exl3.sh: {path} differs from docker/base/sources.lock")
with tarfile.open(path, "r:gz") as archive:
    root = archive.next()
    top = root.name.rstrip("/") if root is not None and root.isdir() else ""
    if not top or "/" in top:
        sys.exit(f"build-exl3.sh: {path} has no single top directory")
    member = archive.getmember(top + "/setup.py")
    source = archive.extractfile(member) if member.isfile() else None
    if source is None:
        sys.exit(f"build-exl3.sh: {path} has no regular setup.py")
    open(out, "xb").write(source.read())
' "$root/docker/base/sources.lock" "$root/build/base-inputs" "$snap/exl3-ext/setup.py"
  build_args+=(--build-context "patches=$snap")
  kind=rebuilt
  [[ "$variant" == candidate-ext ]] && kind=patched
  build --target "ext-$kind-manifest" --output "type=local,dest=$scratch/manifest"
  patches_sha="$(sha256sum -- "$scratch/manifest/exl3-patches.json")"
  patches_sha="${patches_sha%% *}"
  build_args+=(--build-arg "EXL3_ENGINE_MANIFEST_SHA256=$patches_sha")
fi
build --target "$variant" --output type=image,rewrite-timestamp=true,unpack=false \
  --metadata-file "$scratch/build.json"
check_base
# With rewrite-timestamp the image ID is the digest of the rewritten manifest.
built_id="$(python3 -I -B -c '
import json, sys
print(json.load(open(sys.argv[1], "rb"))["containerimage.digest"])
' "$scratch/build.json")"
if [[ ! "$built_id" =~ ^sha256:[0-9a-f]{64}$ ||
      "$(label "$built_id" io.elpis.exl3.base-manifest-sha256)" != "$pinned" ||
      "$(label "$built_id" io.elpis.exl3.variant)" != "$variant" ||
      "$(label "$built_id" io.elpis.exl3.patches-sha256)" != "$patches_sha" ]]; then
  echo "Refusing EXL3 build: missing output identity or baked provenance." >&2
  exit 1
fi
# Publish the local tag only after both base checks and build succeed.
docker image tag "$built_id" "$image"
printf 'Built %s %s as %s from base %s (content manifest %s)\n' \
  "$variant" "$built_id" "$image" "$base" "$pinned"
