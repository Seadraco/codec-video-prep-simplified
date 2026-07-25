#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL_DIR="${1:?wheel directory is required}"
IMAGE="${2:?manylinux image is required}"
PY_TAGS="${PY_TAGS:-cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313}"

docker run --rm \
  -e "WHEEL_DIR=$WHEEL_DIR" \
  -e "PY_TAGS=$PY_TAGS" \
  -v "$ROOT:/io" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    for tag in $PY_TAGS; do
      wheel="$(find "/io/$WHEEL_DIR" -maxdepth 1 -name "*-${tag}-manylinux*.whl" -print -quit)"
      test -n "$wheel"
      /opt/python/$tag/bin/python -m pip install --no-cache-dir "$wheel"
      /opt/python/$tag/bin/python -c "import codec_video_prep.cv_reader_fast; from codec_selector.core.config import BitcostReadinessConfig; c=BitcostReadinessConfig(video=\"v\", out_dir=\"o\", selector_mode=\"diverse_mixed_simple\").normalized(); assert c.selector_mode == \"diverse_mixed_simple\""
      /opt/python/$tag/bin/codec-video-prep --help >/dev/null
      /opt/python/$tag/bin/python -m pip uninstall -y codec-video-prep
    done
  '
