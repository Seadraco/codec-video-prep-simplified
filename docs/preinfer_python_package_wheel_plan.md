# Pre-Infer Pipeline Python Package And Wheel Plan

## Goal

Package the optimized pre-infer pipeline as a Python library that can be installed with `pip`.

Target user experience:

```bash
pip install compressed-video-preinfer

cv-preinfer \
  --video /path/to/video.mp4 \
  --out_dir /tmp/preinfer_out \
  --num_sampled_frames 1024 \
  --group_size 32 \
  --images_per_group 4 \
  --max_pixels 153664
```

Python API:

```python
from compressed_video_preinfer import run_preinfer

result = run_preinfer(
    video="/path/to/video.mp4",
    out_dir="/tmp/preinfer_out",
)

print(result.timings)
```

The package should only keep the optimized path:

- `read_video_fast_selected`
- `thread_type=frame`
- `thread_count=16`
- frame threading disables decoder-internal target-only accounting by default
- FFmpeg decode-side scale
- `mb` bitcost path
- precomputed block scores
- optimized readiness grouping and canvas generation

## Recommended Package Layout

```text
compressed-video-preinfer/
  pyproject.toml
  setup.py
  README.md
  MANIFEST.in
  src/
    compressed_video_preinfer/
      __init__.py
      api.py
      config.py
      cli.py
      pipeline.py
      video_processor.py
      frame_utils.py
      patch_utils.py
      ffmpeg_utils.py
      doctor.py
      cv_reader_fast.cpython-*.so
      libs/
        libavcodec.so.*
        libavformat.so.*
        libavutil.so.*
        libswresample.so.*
        libswscale.so.*
  native/
    cv_reader_fast.cpp
  ffmpeg_patch/
    h264_cabac.c
    h264_cavlc.c
    h264_picture.c
    h264dec.c
    h264dec.h
    patch.sh
  ffmpeg/
    ffmpeg-snapshot.tar.bz2
  scripts/
    build_patched_ffmpeg.sh
    build_wheel_linux.sh
```

## Public API

Use a config object so CLI, Python API, and benchmarks share one execution path.

```python
from dataclasses import dataclass

@dataclass
class PreinferConfig:
    video: str
    out_dir: str
    num_sampled_frames: int = 1024
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 153664
    min_group_frames: int = 8
    max_group_frames: int = 64
    thread_type: str = "frame"
    thread_count: int = 16
    bitcost_grid: str = "mb"
```

Recommended top-level API:

```python
def run_preinfer(
    video: str,
    out_dir: str,
    num_sampled_frames: int = 1024,
    group_size: int = 32,
    images_per_group: int = 4,
    patch: int = 14,
    max_pixels: int = 153664,
    min_group_frames: int = 8,
    max_group_frames: int = 64,
):
    ...
```

The CLI should be a thin wrapper around the same function.

## Optimized Defaults

The packaged library should avoid exposing old paths as defaults.

Use these defaults:

```text
use_fast_selected = True
thread_type = "frame"
thread_count = 16
bitcost_grid = "mb"
parallel_decode_cv_reader = False
canvas_format = "jpg"
```

Native behavior:

```text
if thread_type == frame and export_bitcost:
    disable decoder-internal target-only accounting by default
```

Keep an environment override for debugging:

```bash
CVR_DISABLE_TARGET_ONLY=0  # force target-only on
CVR_DISABLE_TARGET_ONLY=1  # force target-only off
```

## Bundle Patched FFmpeg

For scheme C, the wheel should include patched FFmpeg libraries so users do not need to build FFmpeg manually.

Build FFmpeg as shared libraries:

```bash
./configure \
  --prefix=/opt/cvpreinfer/ffmpeg_install \
  --enable-shared \
  --disable-static \
  --disable-programs \
  --disable-doc \
  --disable-debug \
  --enable-avcodec \
  --enable-avformat \
  --enable-avutil \
  --enable-swresample \
  --enable-swscale \
  --enable-protocol=file \
  --enable-demuxer=mov \
  --enable-demuxer=matroska \
  --enable-demuxer=h264 \
  --enable-parser=h264 \
  --enable-decoder=h264 \
  --enable-decoder=hevc
```

If more containers are needed, add corresponding demuxers.

Required bundled libraries:

```text
libavcodec.so.*
libavformat.so.*
libavutil.so.*
libswresample.so.*
libswscale.so.*
```

## Build Patched FFmpeg Script

Create `scripts/build_patched_ffmpeg.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/ffmpeg"
INSTALL="$ROOT/build_ffmpeg_install"

rm -rf "$BUILD" "$INSTALL"
mkdir -p "$BUILD"

tar xf "$ROOT/ffmpeg/ffmpeg-snapshot.tar.bz2" -C "$BUILD" --strip-components=1

export FFMPEG_PATCH_DIR="$ROOT/ffmpeg_patch"
export FFMPEG_INSTALL_DIR="$BUILD"
bash "$FFMPEG_PATCH_DIR/patch.sh"

cd "$BUILD"
./configure \
  --prefix="$INSTALL" \
  --enable-shared \
  --disable-static \
  --disable-programs \
  --disable-doc \
  --disable-debug \
  --enable-avcodec \
  --enable-avformat \
  --enable-avutil \
  --enable-swresample \
  --enable-swscale \
  --enable-protocol=file \
  --enable-demuxer=mov \
  --enable-demuxer=matroska \
  --enable-demuxer=h264 \
  --enable-parser=h264 \
  --enable-decoder=h264 \
  --enable-decoder=hevc

make -j"$(nproc)"
make install

mkdir -p "$ROOT/src/compressed_video_preinfer/libs"
cp "$INSTALL"/lib/libavcodec.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libavformat.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libavutil.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libswresample.so* "$ROOT/src/compressed_video_preinfer/libs/"
cp "$INSTALL"/lib/libswscale.so* "$ROOT/src/compressed_video_preinfer/libs/"
```

## Build The Native Extension

Place the extension inside the package:

```text
compressed_video_preinfer/cv_reader_fast.cpython-*.so
compressed_video_preinfer/libs/libavcodec.so.*
```

Use rpath so the extension loads FFmpeg libraries from the wheel:

```text
-Wl,-rpath,$ORIGIN/libs
```

Example `setup.py`:

```python
from pathlib import Path

import numpy
from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).parent
ffmpeg_root = ROOT / "build_ffmpeg_install"

ext = Extension(
    "compressed_video_preinfer.cv_reader_fast",
    sources=["native/cv_reader_fast.cpp"],
    include_dirs=[
        str(ffmpeg_root / "include"),
        numpy.get_include(),
    ],
    library_dirs=[
        str(ffmpeg_root / "lib"),
    ],
    libraries=[
        "avformat",
        "avcodec",
        "swresample",
        "swscale",
        "avutil",
    ],
    language="c++",
    extra_compile_args=["-std=c++11", "-O3"],
    extra_link_args=[
        "-Wl,-rpath,$ORIGIN/libs",
        "-Wl,-Bsymbolic",
    ],
)

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[ext],
    package_data={
        "compressed_video_preinfer": ["libs/*.so*"],
    },
)
```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools", "wheel", "numpy"]
build-backend = "setuptools.build_meta"

[project]
name = "compressed-video-preinfer"
version = "0.1.0"
description = "Optimized compressed video pre-inference pipeline"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "opencv-python-headless",
  "Pillow",
]

[project.scripts]
cv-preinfer = "compressed_video_preinfer.cli:main"
cv-preinfer-doctor = "compressed_video_preinfer.doctor:main"
```

## Build Manylinux Wheels

Install build tools:

```bash
pip install build cibuildwheel auditwheel
```

Start with Python 3.11 only:

```bash
CIBW_BUILD="cp311-manylinux_x86_64" \
cibuildwheel --platform linux --output-dir wheelhouse
```

Then expand to Python 3.10 and 3.12:

```bash
CIBW_BUILD="cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64" \
cibuildwheel --platform linux --output-dir wheelhouse
```

Recommended `cibuildwheel` config:

```toml
[tool.cibuildwheel.linux]
before-build = "bash scripts/build_patched_ffmpeg.sh"
repair-wheel-command = "auditwheel repair -w {dest_dir} {wheel}"
```

## Wheel Validation

Check the wheel:

```bash
auditwheel show wheelhouse/*.whl
twine check wheelhouse/*.whl
```

Install in a clean container:

```bash
docker run --rm -it -v "$PWD/wheelhouse:/wheelhouse" -v /video_vit:/video_vit python:3.11 bash

pip install /wheelhouse/compressed_video_preinfer-0.1.0-*.whl
cv-preinfer-doctor
```

Run a real video:

```bash
cv-preinfer \
  --video /video_vit/test.mp4 \
  --out_dir /tmp/preinfer_out \
  --num_sampled_frames 1024 \
  --group_size 32 \
  --images_per_group 4 \
  --max_pixels 153664
```

Expected outputs:

```text
meta.json
src_patch_position.npy
canvas_*.jpg
```

`meta.json` should contain:

```json
{
  "timings": {
    "decode_frames": 0.0,
    "cv_reader_fetch_bitcost": 0.0,
    "process_groups_make_canvases": 0.0,
    "total": 0.0
  }
}
```

## Doctor Command

Add `cv-preinfer-doctor` to make installation issues easy to diagnose.

It should check:

```text
cv_reader_fast import
bundled FFmpeg libraries exist
extension loads bundled FFmpeg libraries
ffmpeg/ffprobe command availability, if CLI decode still shells out
small H.264 bitcost export smoke test
thread_type default is frame
thread_count default is 16
target-only is disabled under frame threading
```

Example output:

```text
cv_reader_fast import: OK
bundled FFmpeg libs: OK
H.264 bitcost export: OK
thread_type: frame
thread_count: 16
frame-thread target-only policy: disabled
```

## Release Options

Internal release via GitHub:

```bash
pip install https://github.com/<org>/<repo>/releases/download/v0.1.0/compressed_video_preinfer-0.1.0-cp311-cp311-manylinux_x86_64.whl
```

Test PyPI:

```bash
twine upload --repository testpypi wheelhouse/*.whl
```

PyPI:

```bash
twine upload wheelhouse/*.whl
```

## Recommended Implementation Order

1. Create the package skeleton under `src/compressed_video_preinfer`.
2. Move the optimized pipeline into `run_preinfer(config)`.
3. Keep only the optimized path as default.
4. Move `tool/benchmark/cv_reader_fast.cpp` to `native/cv_reader_fast.cpp`.
5. Move H.264 FFmpeg patches into `ffmpeg_patch/`.
6. Add `scripts/build_patched_ffmpeg.sh`.
7. Add `setup.py` and `pyproject.toml`.
8. Build FFmpeg shared libraries and compile the extension locally.
9. Test `pip install -e .`.
10. Add `cv-preinfer` and `cv-preinfer-doctor`.
11. Build `cp311-manylinux_x86_64` wheel.
12. Test the wheel in a clean container.
13. Expand to Python 3.10 and 3.12.
14. Publish to GitHub Release or PyPI.

## First Version Scope

Keep the first release narrow:

```text
Linux x86_64
Python 3.10 / 3.11
H.264 optimized path
VideoMME-style 1024 sampled frames
bundled patched FFmpeg
```

Defer these until the package is stable:

```text
macOS wheels
Windows wheels
full HEVC support validation
automatic GPU inference integration
multiple old pipeline compatibility modes
```

