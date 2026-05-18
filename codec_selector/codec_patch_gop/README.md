# Codec Patch GOP

Codec-aware patch dataset generator with GOP-based sampling.

## Overview

This package provides tools for extracting patches from videos using motion vector and residual-based scoring. It replaces the monolithic `generate_codec_patch_uniform_resize.py` with a modular structure.

## Structure

```
codec_patch_gop/
├── __init__.py          # Package init
├── utils.py             # General utilities (jsonl, hash, path)
├── frame_utils.py       # Frame processing (resize, letterbox, bad frame detection)
├── patch_utils.py       # Patch packing and canvas saving
├── video_probe.py       # ffprobe and video metadata
├── energy_sampling.py   # Energy-based sampling algorithms
├── video_processor.py   # Core video processing (process_one_video)
└── main.py              # CLI entry point
```

## Usage

### Basic Usage

```bash
python -m codec_patch_gop.main \
    --jsonl input.jsonl \
    --out_root ./output \
    --num_workers 8
```

### New: Collage Patch Order

Control how patches are arranged in the output canvases:

```bash
# Default: time-based ordering (keeps temporal continuity)
python -m codec_patch_gop.main --collage_patch_order time ...

# Raster ordering by (h, w) - better for visualization
python -m codec_patch_gop.main --collage_patch_order wh ...
```

## Docs

- [Frame Sampling Strategy](./FRAME_SAMPLING_STRATEGY.md)

## Migration Status

| Module | Status | Notes |
|--------|--------|-------|
| utils.py | ✅ Complete | All utility functions |
| frame_utils.py | ✅ Complete | Frame processing |
| patch_utils.py | ✅ Complete | Patch packing with placement_order |
| video_probe.py | ✅ Complete | ffprobe functions |
| energy_sampling.py | ✅ Complete | Sampling algorithms |
| video_processor.py | 🟡 Skeleton | Needs process_one_video extraction |
| main.py | 🟡 Skeleton | CLI args defined, needs processing loop |

## Original File

The original monolithic file remains at:
`tool/generate_codec_patch_uniform_resize.py`

To complete the migration, extract `process_one_video` function (lines ~2045-3144) into `video_processor.py`.

## Dependencies

- numpy
- opencv-python
- pillow (optional, for JPEG writing)
- cv_reader (optional, for MV/Residual extraction)
