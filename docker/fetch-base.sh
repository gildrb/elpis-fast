#!/usr/bin/env bash
# The only network step of the EXL3 base build. It pulls the three pinned images
# by digest and downloads every file that docker/base/sources.lock names into
# build/base-inputs/. Each file must match its sha256, else the script stops and
# keeps no partial file. The base and image builds then run with no network.
#   bash docker/fetch-base.sh
set -euo pipefail
umask 022
if (( $# != 0 )); then
  echo "Usage: bash docker/fetch-base.sh" >&2
  exit 1
fi
root="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/..")"
lock="$root/docker/base/sources.lock"
requirements="$root/docker/base/requirements.lock"
out="$root/build/base-inputs"

# Print the lock as `image <ref>` and `file <path> <sha256> <url>` lines. The
# program is fixed; the lock is only data. Paths must stay inside the output.
listing="$(python3 -I -B -c '
import json, re, sys
lock = json.load(open(sys.argv[1], "rb"))
safe = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+~%-]*")
image = re.compile(r"[a-z0-9./:_-]+@sha256:[0-9a-f]{64}")
def emit(record):
    parts = record["file"].split("/")
    assert all(safe.fullmatch(part) for part in parts), record["file"]
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), record["file"]
    assert record["url"].startswith("https://"), record["url"]
    print("file", record["file"], record["sha256"], record["url"])
assert lock["schema"] == 1
for ref in lock["images"].values():
    assert image.fullmatch(ref), ref
    print("image", ref)
for record in lock["archives"].values():
    emit(record)
for group in (lock["apt"]["packages"], lock["wheels"], lock["serve_wheels"]):
    for record in group:
        emit(record)
' "$lock")"

# requirements.lock must name exactly the wheels and hashes of sources.lock.
python3 -I -B -c '
import json, sys
lock = json.load(open(sys.argv[1], "rb"))
pinned = sorted(
    "{}=={} --hash=sha256:{}".format(w["name"], w["version"], w["sha256"])
    for w in lock["wheels"]
)
lines = sorted(l for l in open(sys.argv[2]).read().splitlines() if l and not l.startswith("#"))
if pinned != lines:
    sys.exit("fetch-base.sh: requirements.lock differs from sources.lock")
' "$lock" "$requirements"

mkdir -p -- "$out"
declare -A wanted=()
while read -r kind first second third; do
  if [[ "$kind" == image ]]; then
    docker pull --quiet -- "$first" > /dev/null
    continue
  fi
  path="$first" sha="$second" url="$third"
  wanted["$path"]=1
  dest="$out/$path"
  if [[ -f "$dest" ]] && echo "$sha  $dest" | sha256sum --check --status --strict; then
    continue
  fi
  mkdir -p -- "$(dirname -- "$dest")"
  rm -f -- "$dest" "$dest.part"
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --retry 3 --output "$dest.part" -- "$url"
  if ! echo "$sha  $dest.part" | sha256sum --check --status --strict; then
    rm -f -- "$dest.part"
    echo "fetch-base.sh: sha256 mismatch for $url" >&2
    exit 1
  fi
  mv -- "$dest.part" "$dest"
done <<< "$listing"

# The build context must hold exactly the locked files.
while IFS= read -r -d '' found; do
  relative="${found#"$out"/}"
  if [[ -z "${wanted[$relative]:-}" ]]; then
    echo "fetch-base.sh: unexpected file $found; remove it" >&2
    exit 1
  fi
done < <(find "$out" -type f -print0)
printf 'Fetched and verified %d files in %s\n' "${#wanted[@]}" "$out"
