#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux2014_x86_64:latest}"
PY_TAG="${PY_TAG:-cp310-cp310}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"

DOCKER_ENV=()
if [[ -n "$PIP_INDEX_URL" ]]; then
  DOCKER_ENV+=("-e" "PIP_INDEX_URL=$PIP_INDEX_URL")
fi
if [[ -n "$PIP_TRUSTED_HOST" ]]; then
  DOCKER_ENV+=("-e" "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST")
fi

docker run --rm \
  "${DOCKER_ENV[@]}" \
  -v "$ROOT:/io" \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    cd /io
    rm -rf build build_ffmpeg_install dist wheelhouse src/compressed_video_preinfer/libs compressed_video_preinfer.egg-info
    /opt/python/$PY_TAG/bin/python -m pip install -U pip setuptools wheel build 'numpy==1.26.4'
    bash scripts/build_patched_ffmpeg.sh
    /opt/python/$PY_TAG/bin/python -m build --wheel
    LD_LIBRARY_PATH=/io/build_ffmpeg_install/lib:/io/src/compressed_video_preinfer/libs \
      auditwheel repair -w wheelhouse dist/*.whl
    auditwheel show wheelhouse/*.whl
    ls -lh wheelhouse
  "

