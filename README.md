# codec-video-prep

Codec-aware video preprocessing for training and inference. Extracts codec-level bitcost information from **H.264 / HEVC / VP9** videos and turns it into patch-canvases ready for downstream vision models.

## What it does

- **Patched FFmpeg decoder** – Instruments the H.264 / HEVC / VP9 decoder to export per-macroblock (H.264) or per-CTU (HEVC) **bitcost** maps during decoding.
- **Fast C++ extension** (`cv_reader_fast`) – Decodes video with loop-filter / IDCT skipped and optionally returns bitcost data as NumPy arrays.
- **Readiness grouping** – Groups frames by compressibility (bitcost) so that hard-to-decode regions get more patches.
- **Top-K patch selection** – Selects the most informative 2×2 patch blocks from each group and packs them into JPG/PNG canvases.
- **One-command pipeline** – From a raw video to a folder of canvases + metadata in a single call.

## Install

### From wheel (recommended)

```bash
python -m pip install codec_video_prep-*.whl
```

Choose the wheel that matches your Python ABI and CPU architecture. Current
Linux wheels are built for:

- `manylinux2014_x86_64` / `manylinux_2_17_x86_64`: x86_64 Linux with glibc >= 2.17.
- `manylinux2014_aarch64` / `manylinux_2_17_aarch64`: aarch64 Linux with glibc >= 2.17.

The aarch64 wheels bundle aarch64 FFmpeg shared libraries built in a
manylinux2014/aarch64 environment, so the Linux wheel baseline is consistent
across x86_64 and aarch64.

Verify the installation:

```bash
codec-video-prep-doctor
```

### Build from source

1. Build the patched FFmpeg shared libraries:

   - **Pixel-capable** (recommended — supports both bitcost and BGR pixel export):
     ```bash
     bash build_pixel_ffmpeg.sh
     ```
   - **Legacy skip-IDCT** (faster bitcost-only scan, no pixel output):
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

### Decode backends

Two decode backends are available:

| Backend | Description | Best for |
|---------|-------------|----------|
| `ffmpeg_native` (default) | FFmpeg subprocess decode + `cv_reader_fast` bitcost scan | General use |
| `cv_reader_pixels` | Single-pass decode via `cv_reader_fast` that returns **both** bitcost and BGR pixels | Speed (~1.8–1.9× faster end-to-end) |

Switch backend:

```bash
codec-video-prep --decode_backend cv_reader_pixels ...
```

### Parallel segment decoding

For long videos with dense frame sampling, the bitcost-scan step dominates total time. You can split the workload into N parallel decode segments using `ProcessPoolExecutor`:

```bash
codec-video-prep \
  --decode_backend cv_reader_pixels \
  --parallel_segments 4 \
  --threads_per_segment 4 \
  --segment_guard_frames 30 \
  ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--parallel_segments` | `0` (disabled) | Number of parallel segments. Set to `0` or `1` to use serial decoding. |
| `--threads_per_segment` | `4` | FFmpeg `thread_count` inside each worker process. |
| `--segment_guard_frames` | `30` | Extra frames decoded before/after each segment boundary to compensate for seek-to-keyframe inaccuracy. |

> **Note:** Parallel segment decoding incurs process-spawn overhead. For short clips (< a few thousand frames) serial decoding is usually faster. The benefit appears on long videos with dense sampling (e.g. 10k+ frames).

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
    decode_backend="cv_reader_pixels",   # or "ffmpeg_native"
    parallel_segments=4,                  # 0 = serial
    threads_per_segment=4,
    segment_guard_frames=30,
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

# Decode selected frames only (bitcost + optional pixels)
selected = cv_reader_fast.read_video_fast_selected(
    path="/path/to/video.mp4",
    frame_ids=[0, 30, 60, 90],
    thread_count=16,
    export_bitcost=1,
    export_pixels=1,   # also return BGR pixels
    out_w=224,         # optional resize width
    out_h=224,         # optional resize height
)

# Segment seek + decode (used internally for parallel workers)
segment = cv_reader_fast.read_video_fast_selected_segment(
    path="/path/to/video.mp4",
    frame_ids=[30, 60, 90],
    seek_frame=0,       # seek target (decoder lands on nearest keyframe before this)
    end_frame=120,      # stop after this frame index
    thread_count=4,
    export_bitcost=1,
    export_pixels=1,
    out_w=224,
    out_h=224,
)
```

Each frame dict contains:

| Key | Description |
|-----|-------------|
| `frame_idx` | Frame index |
| `pict_type` | `'I'`, `'P'` or `'B'` |
| `width` / `height` | Frame resolution |
| `codec_name` | Decoder name (`h264`, `hevc`, `vp9`, …) |
| `bitcost` | Dict with MB/CTU bitcost arrays (when `export_bitcost=1`) |
| `pixels` | `(H, W, 3)` uint8 BGR array (when `export_pixels=1`)

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
│   └── cv_reader_fast.cpp            # Fast decoder with bitcost + pixel export, segment seek API
├── ffmpeg_patch/                     # FFmpeg source patches
│   ├── bitcost_only/                 # Pixel-capable patches (H.264 + HEVC + VP9, keeps full IDCT)
│   │   ├── h264_cabac.c / h264_cavlc.c
│   │   ├── hevcdec.c / hevcdec.h / hevc_refs.c
│   │   ├── vp9.c / vp9dec.h / vp9shared.h
│   │   └── h264_bitcost_only.patch
│   └── full_skip/                    # Legacy skip-IDCT patches (faster, no pixel output)
│       ├── h264_*.c
│       ├── hevc_*.c
│       └── patch.sh
├── scripts/
│   ├── build_patched_ffmpeg.sh       # Build legacy skip-IDCT FFmpeg libs
│   ├── build_pixel_ffmpeg.sh         # Build pixel-capable FFmpeg libs
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
- Linux:
  - x86_64 wheels require glibc >= 2.17.
  - aarch64 wheels require glibc >= 2.17.
- numpy >= 1.23, < 2.0
- opencv-python-headless < 4.12
- Pillow
- Patched FFmpeg shared libraries (built automatically by `scripts/build_patched_ffmpeg.sh`)
