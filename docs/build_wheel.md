# Build The Wheel

Build a Python 3.10 manylinux wheel:

```bash
cd /path/to/cv_pre_infer

# Optional: specify a PyPI mirror via environment variables
# export PIP_INDEX_URL=https://pypi.org/simple
# export PIP_TRUSTED_HOST=pypi.org
bash scripts/build_manylinux_wheel.sh
```

Output:

```text
wheelhouse/codec_video_prep-0.1.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

Install and check:

```bash
python -m pip install wheelhouse/codec_video_prep-0.1.0-*.whl
codec-video-prep-doctor
```

Run:

```bash
codec-video-prep \
  --video /path/to/video.mp4 \
  --out_dir ./preinfer_out \
  --num_sampled_frames 1024 \
  --group_size 32 \
  --images_per_group 4 \
  --max_pixels 153664
```

To build another Python ABI, set `PY_TAG`, for example:

```bash
PY_TAG=cp311-cp311 bash scripts/build_manylinux_wheel.sh
```

