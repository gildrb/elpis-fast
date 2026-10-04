# Third-party source notices

Serving uses the ExLlamaV3 engine and its native DFlash2 extension as installed
in the source-built base image of `docker/base/`. `patches/exl3/` and
`patches/exl3-ext/` hold explicit diffs against that installed engine, not copied
upstream files; the engine retains its upstream license.

## Base image recipe and its contents

The repository holds the recipe and the locks, not the images. An image that
you build from them contains third-party software under these terms:

- `docker/base/Dockerfile` follows the runtime stage of the r0b0tlab recipe
  (https://github.com/r0b0tlab/qwen38-exl3-dflash2, `container/Dockerfile` at
  commit 75d4ec5a). MIT License, Copyright (c) 2026 r0b0tlab contributors.
- ExLlamaV3, community branch of https://github.com/r0b0tlab/exllamav3 at commit
  355c6ee10fbd25b79070316a81ea0708cc18155a, from Turboderp's ExLlamaV3. MIT
  License, Copyright (c) 2025 Turboderp. The build compiles its `exllamav3_ext`
  CUDA extension from this source.
- The base starts from `nvidia/cuda:13.0.0-runtime-ubuntu24.04` and builds in
  `nvidia/cuda:13.0.0-devel-ubuntu24.04` (both pinned by digest). The CUDA
  libraries in them are under the NVIDIA CUDA Toolkit End User License Agreement
  (https://docs.nvidia.com/cuda/eula/index.html); Attachment A lists the
  redistributable runtime libraries. The images also carry
  `/NGC-DL-CONTAINER-LICENSE`. The compiled extension links against the CUDA
  runtime, and nvcc from the devel image builds it.
- The CUDA, cuDNN, NCCL and other `nvidia-*` Python wheels and `torch` come from
  https://download.pytorch.org/whl/cu130 under their own licenses, which this
  repository has not reviewed for redistribution.
- Ubuntu 24.04 packages (gcc, libc6-dev, binutils and their dependencies, listed in
  `docker/base/apt.in`) are under GPL, LGPL and other free licenses. Their
  sources are on https://launchpad.net and in the Ubuntu archive.
- CPython 3.13.10 (python-build-standalone release 20251202, as installed by uv
  0.9.15) is under the Python Software Foundation License and the licenses of
  its bundled libraries.
- The other Python distributions in `docker/base/requirements.lock` keep their
  own licenses. `serve/exl3_server.py` imports `regex` (2026.9.10, Apache-2.0),
  which the base venv contains as a transformers dependency.

## Methodology and evaluation

The measurement methodology is informed by
https://github.com/syv-ai/qwen38-27b-rtx3090. Its vLLM performance measurements
are not measurements of this EXL3 deployment.

Model-quality evaluation uses external upstream Prime Envs environments.
Their pinned source references and setup are documented in `eval/README.md`.
Environment code and data retain their upstream licenses and usage conditions.
These notices do not change the licenses of model weights or other dependencies.
