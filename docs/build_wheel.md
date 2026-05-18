# Build The Wheel

Build a Python 3.10 manylinux wheel:

```bash
cd /path/to/cv_pre_infer

PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
PIP_TRUSTED_HOST=mirrors.aliyun.com \
bash scripts/build_manylinux_wheel.sh
```

Output:

```text
wheelhouse/compressed_video_preinfer-0.1.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

Install and check:

```bash
python -m pip install wheelhouse/compressed_video_preinfer-0.1.0-*.whl
cv-preinfer-doctor
```

Run:

```bash
cv-preinfer \
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

