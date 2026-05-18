#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Build cp310 first (with FFmpeg compile)
echo "========================================"
echo "Building cp310-cp310 ..."
echo "========================================"
PY_TAG=cp310-cp310 bash scripts/build_manylinux_wheel.sh

# Build remaining versions reusing FFmpeg
for tag in cp311-cp311 cp312-cp312 cp313-cp313; do
  echo "========================================"
  echo "Building $tag ..."
  echo "========================================"
  REUSE_FFMPEG=1 PY_TAG=$tag bash scripts/build_manylinux_wheel.sh
done

echo "All wheels built:"
ls -lh wheelhouse/
