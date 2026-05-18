# Bitcost Readiness Pipeline Timing Report

## Test Setup

- Video: `/ov2/dataset_mid_source_split/2_3_m_pandas70m/1800000_1849999/1832000_1833999/00006/0000606_st293.63_d180.00.mp4`
- Duration bucket: 180s
- Sampling: `--frame_sampling_mode fps --sample_fps 4`
- Grouping: `--grouping_mode readiness`
- Sampled frames: 723
- Output geometry: `[280, 504]`
- Baseline codec/grid: H.264, `--bitcost_grid sub`

## Pipeline Stages

| Stage | What it does |
| --- | --- |
| `probe_video` | Uses ffprobe metadata: fps, frame count, resolution, codec, bitrate. |
| `sample_frames` | Chooses candidate frame ids from the video timeline. |
| `decode_frames` | Uses ffmpeg to decode selected frame pixels for final patch/canvas extraction. |
| `cv_reader_fetch_bitcost` | Uses patched cv_reader/FFmpeg to parse bitstream and extract bitcost maps for selected frames. |
| `prepare_frames` | Resizes and pads decoded BGR frames to patch/block-aligned geometry. |
| `bitcost_to_score_maps` | Converts raw bitcost maps into normalized per-pixel score maps. |
| `build_groups` | Splits sampled frames into fixed/readiness groups using score-map statistics. |
| `process_groups_make_canvases` | Selects top-k 2x2 blocks per group, extracts patches, and packs in-memory canvases. |
| `save_canvases` | Writes canvas images to disk. |
| `save_numpy_outputs` | Writes `src_patch_position.npy` and `frame_ids.npy`. |

## Timing Summary

| Version | Total | Main notes |
| --- | ---: | --- |
| Original baseline | 14.883s | Sequential pipeline, repeated grouping computation, Python-side frame resize. |
| Optimized grouping | 11.276s | Precomputed per-frame block scores; `build_groups` dropped sharply. |
| Parallel decode/cv_reader | 10.888s | Small gain only; decode and cv_reader contend for IO/CPU. |
| Serial + ffmpeg preprocess | 8.101s | ffmpeg decodes directly to target size and default returns to serial execution. |
| Reused block scores in selector | 7.662s | Current best result; grouping and selector share one block-score precompute, and group frame/score copies are avoided. |

## Stage-Level Comparison

| Stage | Original baseline | After grouping opt | Parallel decode/cv | ffmpeg preprocess | Current best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decode_frames` | 3.477s | 3.232s | 5.709s | 1.309s | 1.409s |
| `cv_reader_fetch_bitcost` | 3.024s | 3.019s | 3.109s | 3.091s | 3.069s |
| Decode/cv wall time | 6.501s serial | 6.251s serial | 5.710s parallel | 4.400s serial | 4.477s serial |
| `prepare_frames` | 1.306s | 1.300s | 1.302s | 0.001s | 0.001s |
| `bitcost_to_score_maps` | 0.399s | 0.400s | 0.522s | 0.406s | 0.408s |
| `precompute_block_scores` | N/A | included in grouping | included in grouping | included in grouping | 0.188s |
| `build_groups` | 3.762s | 0.470s | 0.470s | 0.469s | 0.094s |
| `process_groups_make_canvases` | 2.233s | 2.243s | 2.270s | 2.149s | 1.884s |
| `save_canvases` | 0.576s | 0.533s | 0.536s | 0.587s | 0.510s |
| `total` | 14.883s | 11.276s | 10.888s | 8.101s | 7.662s |

## Optimizations Applied

### 1. Removed repeated `build_groups` work

Before, readiness grouping repeatedly recomputed:

```text
score_map -> patch_scores -> block_scores -> top-k
```

for many candidate group ends.

Now each frame's block scores are precomputed once and reused by threshold estimation and group building.

Result:

```text
build_groups: 3.762s -> 0.470s
```

### 2. Made decode/cv_reader parallel optional

Parallel execution was tested, but on this server it gives only a small and unstable gain because both tasks read and parse the same video concurrently.

Observed:

```text
serial after grouping opt: 11.276s
parallel decode/cv:        10.888s
```

The pipeline now defaults to serial execution:

```text
parallel_decode_cv_reader = False
```

and can be enabled explicitly with:

```bash
--parallel_decode_cv_reader
```

### 3. Moved frame resize into ffmpeg

Before, ffmpeg decoded full-size selected frames and Python/OpenCV resized and padded all 723 frames.

Now ffmpeg applies scale during decode, so Python `prepare_frames` is effectively a shape check/no-op for normal resized mode.

Result:

```text
decode_frames:  3.809s -> 1.309s
prepare_frames: 1.419s -> 0.001s
total:         12.422s -> 8.101s
```

This is controlled by:

```bash
--ffmpeg_preprocess_frames
--no-ffmpeg_preprocess_frames
```

Default is enabled.

### 4. Tested cv_reader export narrowing

`cv_reader_fetch_bitcost` can now request only the bitcost map needed by `--bitcost_grid`, and the C++ extension was prototyped with `bitcost_export`:

```text
0 = all
1 = mb_bit_cost
2 = sub_mb_bit_cost
3 = ctu_bit_cost
```

However, `mb`-only export barely changed `cv_reader` time:

```text
all export count-only: 3.075s
mb-only count-only:    3.036s
```

Conclusion: Python/numpy result construction is not the main cv_reader bottleneck.

### 5. Tested yasm-enabled patched FFmpeg

An isolated `ffmpeg_install_yasm` was built without `--disable-yasm` and cv_reader was linked against it.

Result:

```text
default count-only: 3.036s
yasm count-only:    3.086s
```

Conclusion: `--disable-yasm` is not the cause of the bitcost-only runtime for this workload.

### 6. Reused block scores in selector and removed group copies

After the ffmpeg preprocess optimization, `process_groups_make_canvases` became one of the largest remaining CPU stages.

Two low-risk changes were tested:

- pass group frames and score maps by reference instead of copying every frame/map per group;
- precompute `block_scores_all` once and reuse it in both readiness grouping and top-k selection.

Before this change, the selector recomputed:

```text
score_map -> patch_scores -> block_scores
```

inside every group.

Result:

```text
total:                         8.101s -> 7.662s
process_groups_make_canvases:   2.149s -> 1.884s
build_groups:                   0.469s -> 0.094s
precompute_block_scores:        new 0.188s
```

The output shape stayed stable:

```text
sampled_frames = 723
num_groups = 23
total_images = 92
total_patches = 66240
```

### 7. Tested per-frame seek for sampled frames

We tested whether cv_reader could parse only sampled frames by seeking to each sampled frame separately.

For the first 64 sampled frames:

```text
sequential scan: 0.320s
per-frame seek: 4.085s
```

For all 723 sampled frames:

```text
sequential scan: 3.033s
per-frame seek: 43.603s
```

Conclusion: per-frame seek is much slower for dense 4fps sampling because each seek lands on an earlier keyframe and repeatedly reparses GOPs. Sequential scan remains the right strategy for this workload.

### 8. Tested cv_reader output and bitcost accounting variants

Several experiments were run to isolate the `cv_reader_fetch_bitcost` bottleneck:

```text
all export count-only:      3.072s
mb-only export count-only:  3.034s
no bitcost export:          3.055s
internal mb-only accounting: 2.972s count-only
```

The internal mb-only accounting experiment disabled sub-MB accumulation in a temporary FFmpeg build and kept only MB bitcost. Full pipeline with `--bitcost_grid mb` reached:

```text
total:                   7.680s
cv_reader_fetch_bitcost:  2.933s
```

Conclusion: bitcost map construction, Python callback output, and sub-MB accumulation are not the main source of the remaining 3s cv_reader time. The dominant cost is still patched decoder bitstream parsing and macroblock syntax traversal.

### 9. Tested compact block-score output from cv_reader

An experimental H.264 MB-only API was added to return compact per-frame block scores directly from C++:

```text
read_video_cb_blocks(...)
```

It returned one small `[10, 18]` `block_scores` array per selected frame instead of a full MB bitcost map.

Result:

```text
read_video_cb_blocks:      3.099s
mb-only count callback:    3.038s
```

Conclusion: compact output does not materially reduce cv_reader time, which again points to bitstream parsing rather than Python data transfer as the bottleneck.

### 10. Compared system FFmpeg and patched cv_reader FFmpeg builds

The system ffmpeg used by `decode_frames` is Ubuntu's shared FFmpeg 4.4 build:

```text
/usr/bin/ffmpeg
--enable-shared
--enable-libx264
--enable-libx265
...
```

cv_reader links against the repository's patched FFmpeg 5.1 static build:

```text
ffmpeg/ffmpeg_install/lib/pkgconfig
--enable-pic
--disable-doc
--disable-yasm
--disable-programs
```

The patched build reports:

```text
HAVE_X86ASM 0
HAVE_SSE2_EXTERNAL 0
HAVE_AVX2_EXTERNAL 0
HAVE_INLINE_ASM 1
```

However, the isolated yasm-enabled patched FFmpeg experiment did not improve cv_reader timing, so compile flags are not the primary explanation for this workload.

## Current Bottlenecks

Current best timing:

```text
total:                         7.662s
decode_frames:                 1.409s
cv_reader_fetch_bitcost:        3.069s
prepare_frames:                 0.001s
bitcost_to_score_maps:          0.408s
precompute_block_scores:        0.188s
build_groups:                   0.094s
process_groups_make_canvases:   1.884s
save_canvases:                  0.510s
```

Remaining major costs:

1. `cv_reader_fetch_bitcost`: about 3.1s
   - Mostly patched decoder bitstream parsing and bitcost accounting.
   - Not mainly numpy copy or callback overhead.

2. `process_groups_make_canvases`: about 1.9s
   - Patch extraction, sorting, packing, and per-group in-memory canvas creation.

3. `save_canvases`: about 0.6s
   - Image encoding and disk write.

## Practical Takeaways

- The most successful optimization so far is moving frame resize into ffmpeg.
- `build_groups` is no longer a bottleneck.
- Reusing block scores in both grouping and selector reduced total runtime to 7.662s.
- Parallel decode/cv_reader is not worth making default on this server.
- `cv_reader` bitcost-only time is dominated by bitstream parsing/statistics, not Python-side numpy storage.
- Per-frame seek is much slower for dense sampling.
- Compact block-score output from cv_reader did not materially improve cv_reader time.
- Further speedups likely need one of:
  - lazy decode only for frames that actually contribute selected patches;
  - optimizing `process_groups_make_canvases`;
  - deeper FFmpeg patch changes to reduce bitcost accounting work itself.
