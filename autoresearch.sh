#!/usr/bin/env bash
# Autoresearch harness for elpis-fast: decode + prefill speed of this checkout.
#   1. build this checkout's candidate-ext image (docker/build-exl3.sh, under the CPU lock);
#   2. one guarded GPU window (host ops /tmp/gpu-batch.py -> /tmp/gpu-window.sh: maintenance
#      lease, cool-down, live baseline restored after) runs bench/ar_gpu.py in the image:
#      cold prefill 8K x2 / 32K x2 / 128K, decode 1K / 8K / 32K x2 (256 tokens),
#      native 262K (TTFT + 128 decode tokens);
#   3. bench/ar_report.py prints METRIC lines: prefill_tok_s and decode_tok_s (geomeans,
#      262K included); speed_score = sqrt(prefill x decode) only decides keep/discard.
#      Quality vs bench/ar_reference.json (fast baseline) and bench/ar_lossless.json
#      (elpis lossless image): NAME_text_equal (0-4), NAME_prefix_frac, NAME_first_token_equal.
# Evidence: /tmp/kernel-work/AR/run-STAMP/{build.log,spec.json,batch.log,gpu.log,io/}.
# About 16 min when the build is cached (+ about 10 min when the ext recompiles);
# give run_experiment a 2400 s timeout. The body is functions: bash parses it whole,
# so editing this file during a run cannot change the running harness.
set -euo pipefail
main() {
umask 022
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# The Bend acceptor root (proof gates + emitted C, about 7 min) depends only on bend/
# and the flake toolchain: build it once per content hash, reuse it after.
bkey=$(cd -- "$root" && find bend flake.nix flake.lock -type f ! -name '*.pyc' -print0 |
	sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
bcache=/tmp/kernel-work/AR/bend-exl3-$bkey
cd -- "$root"
# Fixed SOURCE_DATE_EPOCH (session baseline f230543): commits alone keep the ext build cached.
export DOCKER_HOST=unix:///run/user/1000/docker.sock GPU_COOL_GAP=120 \
	EXL3_SOURCE_DATE_EPOCH=1791224614
stamp=$(date +%Y%m%d-%H%M%S)
out=/tmp/kernel-work/AR/run-$stamp
tag=qwen-inference:ar-candidate
mkdir -p /tmp/kernel-work/AR
mkdir "$out" "$out/io"
t0=$(date +%s)

if [[ -n ${AR_IMAGE:-} ]]; then
	gpu_run "$AR_IMAGE" "$out"
	return
fi
echo "build $(date +%T) -> $out/build.log (bend root ${bcache##*/}: $([[ -d $bcache ]] && echo cached || echo new))"
prebuilt=()
[[ -d $bcache ]] && prebuilt=(env "EXL3_BEND_PREBUILT=$bcache")
if ! /tmp/cpu-lock.sh "${prebuilt[@]}" bash docker/build-exl3.sh candidate-ext "$tag" \
	>"$out/build.log" 2>&1; then
	tail -n 40 "$out/build.log" >&2
	echo "BUILD FAILED" >&2
	exit 1
fi
[[ -d $bcache ]] || cp -a -- build/bend-exl3 "$bcache"
img=$(docker image inspect -f '{{.Id}}' "$tag")
echo "image $img ($(( $(date +%s) - t0 )) s)"
gpu_run "$img" "$out"
}

# gpu_run IMAGE OUT: one guarded window running bench/ar_gpu.py in IMAGE, then METRIC lines.
# AR_IMAGE=sha256:... bash autoresearch.sh skips the build and measures that image
# (reference runs: the fast baseline and the elpis lossless image).
gpu_run() {
local img=$1 out=$2 stamp=${2##*/run-}

python3 -I -B - "$out" "$root" "$img" "$stamp" >"$out/spec.json" <<'PY'
import json, sys
out, root, img, stamp = sys.argv[1:]
cmd = ["/opt/venv/bin/python", "-I", "-B", "/work/ar_gpu.py"]
print(json.dumps({"batch": f"ar-{stamp}", "payloads": [{
    "name": f"qwen-exl3-ar-{stamp}", "image": img, "window": f"comp-window-ar-{stamp}",
    "lease": 1800, "run_timeout": 1500, "gpu_seconds": 900, "log": f"{out}/gpu.log",
    "cmd": cmd, "dry_cmd": [*cmd, "--dry-run"], "dry_timeout": 900,
    "mounts": [f"{root}/bench:/work:ro", f"{out}/io:/out"],
    "outputs": [f"{out}/io/result.json"]}]}, indent=1))
PY

# One GPU user at a time: wait for other batches and approvals to clear.
while pgrep -f '[g]pu-batch.py /' >/dev/null || grep -qv '^#' /tmp/elpis-gpu-allow 2>/dev/null; do
	sleep 30
done
echo "gpu window $(date +%T) -> $out/batch.log"
if ! python3 /tmp/gpu-batch.py "$out/spec.json" >"$out/batch.log" 2>&1; then
	tail -n 60 "$out/batch.log" >&2
	echo "GPU BATCH FAILED" >&2
	exit 1
fi
grep -E '^\[(prefill|decode|native)\]' "$out/gpu.log" || true
refs=()
for r in fast:bench/ar_reference.json lossless:bench/ar_lossless.json; do
	[[ -f ${r#*:} ]] && refs+=(--ref "${r%%:*}=${r#*:}")
done
python3 -I -B bench/ar_report.py "$out/io/result.json" "${refs[@]}"
echo "METRIC elapsed_s=$(( $(date +%s) - t0 ))"
}
main "$@"
