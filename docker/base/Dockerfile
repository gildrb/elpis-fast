# EXL3 base image, built from source with no network. It follows the runtime
# stage of the r0b0tlab recipe (github.com/r0b0tlab/qwen38-exl3-dflash2,
# container/Dockerfile at 75d4ec5a), without its example files, server and
# entrypoint. elpis does not use them.
#
# Build only with docker/build-base.sh after docker/fetch-base.sh. Every input
# is pinned: the images below by digest, all other files by sha256 in sources.lock,
# and they come from the `base-inputs` build context. No RUN has network access.
ARG SOURCE_DATE_EPOCH

FROM ghcr.io/astral-sh/uv:0.9.15@sha256:4c1ad814fe658851f50ff95ecd6948673fffddb0d7994bdb019dcb58227abd52 AS uv

# Build stages. The devel image has g++ 13.3.0-6ubuntu2~24.04; the recipe's apt
# step changed nothing it needs. ninja comes from the locked ninja wheel.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS python
ARG SOURCE_DATE_EPOCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    MAX_JOBS=8 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/venv/bin
COPY --from=uv /uv /usr/local/bin/uv

# uv checks the CPython archive against the sha256 in its own download table.
RUN --network=none \
    --mount=type=bind,from=base-inputs,source=python,target=/inputs/python \
    UV_PYTHON_INSTALL_MIRROR=file:///inputs/python \
    uv venv --python 3.13.10 /opt/venv && \
    grep -qx 'home = /opt/uv-python/cpython-3.13.10-linux-x86_64-gnu/bin' /opt/venv/pyvenv.cfg

FROM python AS build
ARG SOURCE_DATE_EPOCH

# The three install steps of the recipe, with the same requests, so the same
# packages carry REQUESTED. manifest.py first requires the wheel directory to hold
# exactly the locked files, each with its pinned sha256. The constraints then fix
# every version, and uv checks each wheel against the hash in requirements.lock.
RUN --network=none \
    --mount=type=bind,from=base-inputs,source=wheels,target=/inputs/wheels \
    --mount=type=bind,source=requirements.lock,target=/inputs/requirements.lock \
    --mount=type=bind,source=sources.lock,target=/inputs/sources.lock \
    --mount=type=bind,source=manifest.py,target=/inputs/manifest.py \
    set -eu; \
    /opt/venv/bin/python -I -B /inputs/manifest.py inputs /inputs/sources.lock wheels /inputs/wheels; \
    install() { \
      uv pip install --python /opt/venv/bin/python --offline --no-index \
        --find-links /inputs/wheels --verify-hashes \
        --constraint /inputs/requirements.lock "$@"; \
    }; \
    install torch==2.10.0; \
    install setuptools wheel ninja tokenizers numpy rich typing_extensions safetensors \
      pillow pyyaml marisa-trie pydantic llguidance triton blessed prompt_toolkit \
      pyperclip requests huggingface_hub hf_transfer; \
    install transformers; \
    uv pip check --python /opt/venv/bin/python; \
    /opt/venv/bin/python -I -B /inputs/manifest.py installed /inputs/sources.lock

# exllamav3 from the pinned commit tarball, compiled for sm_86 like the recipe.
# The tarball's file times are the commit time, so the -lineinfo line tables that
# ptxas writes do not depend on the build time.
RUN --network=none \
    --mount=type=bind,from=base-inputs,source=exllamav3-355c6ee.tar.gz,target=/inputs/exllamav3.tar.gz \
    --mount=type=bind,source=sources.lock,target=/inputs/sources.lock \
    set -eu; \
    sha="$(/opt/venv/bin/python -I -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["archives"]["exllamav3"]["sha256"])' /inputs/sources.lock)"; \
    echo "$sha  /inputs/exllamav3.tar.gz" | sha256sum --check --strict; \
    mkdir -p /src/exllamav3; \
    tar -xzf /inputs/exllamav3.tar.gz -C /src/exllamav3 --strip-components=1 --no-same-owner; \
    cd /src/exllamav3; \
    uv pip install --python /opt/venv/bin/python --offline --no-index \
      --no-build-isolation --no-deps .

RUN --network=none \
    --mount=type=bind,source=sources.lock,target=/inputs/sources.lock \
    --mount=type=bind,source=manifest.py,target=/inputs/manifest.py \
    /opt/venv/bin/python -I -B /inputs/manifest.py normalise /inputs/sources.lock

# Check the .deb files here, where Python exists, and pass on only checked files.
FROM python AS debs
RUN --network=none \
    --mount=type=bind,from=base-inputs,source=debs,target=/inputs/debs \
    --mount=type=bind,source=sources.lock,target=/inputs/sources.lock \
    --mount=type=bind,source=manifest.py,target=/inputs/manifest.py \
    /opt/venv/bin/python -I -B /inputs/manifest.py inputs /inputs/sources.lock apt /inputs/debs && \
    cp -a /inputs/debs /debs

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3 AS base
ARG SOURCE_DATE_EPOCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# The recipe's `apt-get install ca-certificates libgomp1 gcc libc6-dev`, from the
# locked .deb files only. The logs and caches that hold build times are removed.
RUN --network=none \
    --mount=type=bind,from=debs,source=/debs,target=/inputs/debs \
    apt-get install -y --no-install-recommends /inputs/debs/*.deb && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*.bin /var/cache/apt/archives/partial \
      /var/log/apt /var/log/dpkg.log /var/log/alternatives.log \
      /var/cache/debconf/*-old /var/lib/dpkg/*-old /var/cache/ldconfig/aux-cache

COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /opt/venv /opt/venv

RUN --network=none \
    useradd --create-home --uid 10001 runner && \
    mkdir -p /models && chown runner:runner /models && \
    rm -f /var/log/lastlog /var/log/faillog /etc/shadow- /etc/gshadow- \
      /etc/passwd- /etc/group- /etc/subuid- /etc/subgid-

ENV PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu

# The recipe's smoke import (no GPU needed), then the per-build record that
# docker/build-exl3.sh and patches/exl3-ext/ext.py authenticate.
RUN --network=none \
    --mount=type=bind,source=manifest.py,target=/inputs/manifest.py \
    python -I -B -c "import torch, exllamav3; from exllamav3.ext import exllamav3_ext as e; print('smoke ok:', torch.__version__, 'shapes', e.exl3_gemm_num_kernel_shapes())" && \
    python -I -B /inputs/manifest.py write /opt/elpis-base-manifest.json
USER runner
