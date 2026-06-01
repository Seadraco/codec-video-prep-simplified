#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${X86_64_WHEELHOUSE:-$ROOT/wheelhouse_x86_64}"
PY_TAGS="${PY_TAGS:-cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313}"
FFMPEG_BUILD_SCRIPT="${FFMPEG_BUILD_SCRIPT:-build_pixel_ffmpeg.sh}"

rm -rf "$OUT"
mkdir -p "$OUT"

first=1
for tag in $PY_TAGS; do
  echo "========================================"
  echo "Building x86_64 $tag"
  echo "FFmpeg build script: $FFMPEG_BUILD_SCRIPT"
  echo "========================================"
  if [[ "$first" == "1" ]]; then
    FFMPEG_BUILD_SCRIPT="$FFMPEG_BUILD_SCRIPT" PY_TAG="$tag" bash "$ROOT/scripts/build_manylinux_wheel.sh"
    first=0
  else
    FFMPEG_BUILD_SCRIPT="$FFMPEG_BUILD_SCRIPT" REUSE_FFMPEG=1 PY_TAG="$tag" bash "$ROOT/scripts/build_manylinux_wheel.sh"
  fi
  cp "$ROOT"/wheelhouse/*.whl "$OUT"/
done

echo "All x86_64 wheels:"
ls -lh "$OUT"
