# codec-video-prep

Codec-aware video preprocessing for training and inference. Extracts codec-level bitcost information from H.264 / HEVC videos and turns it into patch-canvases ready for downstream vision models.

## What it does

- **Patched FFmpeg decoder** – Instruments the H.264 / HEVC decoder to export per-macroblock (H.264) or per-CTU (HEVC) **bitcost** maps during decoding.
- **Fast C++ extension** (`cv_reader_fast`) – Decodes video with loop-filter / IDCT skipped and optionally returns bitcost data as NumPy arrays.
- **Readiness grouping** – Groups frames by compressibility (bitcost) so that hard-to-decode regions get more patches.
- **Top-K patch selection** – Selects the most informative 2×2 patch blocks from each group and packs them into JPG/PNG canvases.
- **One-command pipeline** – From a raw video to a folder of canvases + metadata in a single call.

## Install

### From wheel (recommended)

```bash
python -m pip install codec_video_prep-*.whl
```

Verify the installation:

```bash
codec-video-prep-doctor
```

### Build from source

1. Build the patched FFmpeg shared libraries:

```bash
bash scripts/build_patched_ffmpeg.sh
```

2. Build and install the Python package:

```bash
python -m pip install -e .
```

## Quick start (CLI)

```bash
codec-video-prep \
  --video /path/to/video.mp4 \
  --out_dir ./preinfer_out \
  --num_sampled_frames 1024 \
  --group_size 32 \
  --images_per_group 4 \
  --max_pixels 153664
```

Output directory will contain:

- `canvas_*.jpg` – Packed patch canvases
- `meta.json` – Full metadata, timing, and group info
- `frame_ids.npy` – Sampled frame indices
- `src_patch_position.npy` – Patch source positions

## Python API

### High-level one-shot call

```python
from codec_video_prep import run_preinfer

result = run_preinfer(
    video="/path/to/video.mp4",
    out_dir="./preinfer_out",
    num_sampled_frames=1024,
    group_size=32,
    images_per_group=4,
    patch=14,
    max_pixels=153664,
    min_group_frames=8,
    max_group_frames=64,
    bitcost_grid="adaptive",
)

print(result.out_dir)       # output directory
print(result.meta_path)     # path to meta.json
print(result.timings)       # timing breakdown
```

### Low-level fast decoder

```python
from codec_video_prep import cv_reader_fast

# Decode all frames with bitcost export
frames = cv_reader_fast.read_video_fast(
    path="/path/to/video.mp4",
    thread_count=16,
    export_bitcost=1,
    thread_type="auto",
)

# Decode selected frames only
selected = cv_reader_fast.read_video_fast_selected(
    path="/path/to/video.mp4",
    frame_ids=[0, 30, 60, 90],
    thread_count=16,
    export_bitcost=1,
)
```

Each frame dict contains:

| Key | Description |
|-----|-------------|
| `frame_idx` | Frame index |
| `pict_type` | `'I'`, `'P'` or `'B'` |
| `width` / `height` | Frame resolution |
| `codec_name` | Decoder name (`h264`, `hevc`, …) |
| `bitcost` | Dict with MB/CTU bitcost arrays (when `export_bitcost=1`) |

## Project structure

```
├── src/codec_video_prep/    # Python package
│   ├── api.py                        # run_preinfer() entrypoint
│   ├── cli.py                        # codec-video-prep CLI
│   ├── doctor.py                     # codec-video-prep-doctor diagnostics
│   ├── config.py                     # PreinferConfig
│   └── libs/                         # Bundled FFmpeg .so files
├── codec_selector/                   # Frame sampling / grouping / patch selection
│   ├── core/                         # Pipeline, probe, decode, config
│   ├── plugins/                      # Samplers, scorers, groupers, selectors, packers
│   └── codec_patch_gop/              # Legacy GOP-based utilities
├── native/                           # C++ Python extension
│   └── cv_reader_fast.cpp            # Fast decoder with bitcost export
├── ffmpeg_patch/                     # FFmpeg source patches
│   ├── h264_*.c                      # H.264 bitcost instrumentation
│   ├── hevc_*.c                      # HEVC bitcost instrumentation
│   └── patch.sh                      # Patch application script
├── scripts/
│   ├── build_patched_ffmpeg.sh       # Build patched FFmpeg libs
│   └── build_manylinux_wheel.sh      # Build manylinux wheel
├── setup.py                          # setuptools build (C++ extension + FFmpeg libs)
└── pyproject.toml                    # PEP 517 project metadata
```

## Build a manylinux wheel

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
PIP_TRUSTED_HOST=mirrors.aliyun.com \
bash scripts/build_manylinux_wheel.sh
```

Output:

```
wheelhouse/codec_video_prep-0.1.0-cp310-cp310-manylinux2014_x86_64.whl
```

Install and check:

```bash
python -m pip install wheelhouse/codec_video_prep-*.whl
codec-video-prep-doctor
```

To target a different Python ABI, set `PY_TAG`:

```bash
PY_TAG=cp311-cp311 bash scripts/build_manylinux_wheel.sh
```

## Diagnostics

`codec-video-prep-doctor` checks:

- `cv_reader_fast` C extension can be imported
- Bundled FFmpeg shared libraries are present
- Threading defaults (auto thread type, 16 threads)

## Backward Compatibility

The old import path and CLI names are kept as aliases:

- `compressed_video_preinfer`
- `cv-preinfer`
- `cv-preinfer-doctor`

## Requirements

- Python ≥ 3.10
- numpy >= 1.23, < 2.0
- opencv-python-headless < 4.12
- Pillow
- Patched FFmpeg shared libraries (built automatically by `scripts/build_patched_ffmpeg.sh`)
