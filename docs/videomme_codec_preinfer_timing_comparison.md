# VideoMME Codec Pre-Inference Timing Comparison

## 测试目的

对比 `llava_onevision2_v3_codec.py` 推理前 online codec 预处理的优化前后耗时。

本次只测试推理前流程，不加载模型、不进入 GPU 推理：

1. online codec pipeline 生成 canvas 和 `src_patch_position.npy`
2. wrapper 风格加载 `meta.json`、canvas、`src_patch_position.npy`

## 测试环境

- Server: `root@172.16.5.106`
- Docker image: `preprocess:mv_res_v1_2604`
- Video dataset: VideoMME
- Video dir: `/vlm/xiangan/huggingface_hevc_vit/videomme/data`
- Benchmark script:
  - `/video_vit/yunyaoyan/code/lmms_ov2_result/benchmarks/benchmark_lmms_codec_preinfer.py`
  - `/video_vit/yunyaoyan/code/lmms_ov2_result/benchmarks/run_videomme_preinfer_batch.py`

## 测试参数

按 `llava_onevision2_v3_codec.py` 的 eval 常见 codec 参数：

```text
codec_target_canvas = 128
codec_max_pixels    = 153664
group_size          = 32
images_per_group    = 4
patch               = 14
min_group_frames    = 8
max_group_frames    = 64
sampled_frames      = 1024
avoid_keyframes     = true
```

`sampled_frames = codec_target_canvas / images_per_group * group_size = 128 / 4 * 32 = 1024`。

## 对比对象

### 优化前

原始 pipeline：

```text
/vlm/yinxie/code/lmms_ov2_result/codec_tools/pipeline/process_video_bitcost_mv_mask_collage.py
```

为了拿到内部阶段耗时，已复制到用户隔离路径并加了 `timing_sec`：

```text
/video_vit/yunyaoyan/code/lmms_ov2_result/codec_tools/pipeline/process_video_bitcost_mv_mask_collage.py
```

未修改 `/vlm/yinxie`。

### 优化后

优化后的 readiness pipeline：

```text
/video_vit/yunyaoyan/Compressed-Video-Reader/tool/pipeline/process_video_bitcost_readiness.py
```

主要包含：

- `avoid_keyframes` 改用 `ffprobe -skip_frame nokey`
- ffmpeg decode 阶段直接 scale 到目标尺寸
- `build_groups` 避免重复计算 block scores
- canvas 组装减少重复 copy
- 默认串行跑 decode 和 cv_reader，保留并行配置

## 总体结果

| Video | Duration | Resolution | Original Total | Optimized Total | Speedup | Original RSS | Optimized RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1NYQf_OXDqI.mp4` | 59.2 min | 1280x714 | 191.49s | 80.82s | 2.37x | 5.57GB | 1.47GB |
| `1wzgMHrkrys.mp4` | 57.9 min | 640x360 | 80.97s | 31.52s | 2.57x | 2.10GB | 1.29GB |
| `-dfvdKf-KR0.mp4` | 54.5 min | 1280x720 | 145.26s | 47.70s | 3.05x | 5.62GB | 1.46GB |
| `068rdc75mHM.mp4` | 53.3 min | 1280x720 | 167.48s | 56.35s | 2.97x | 5.62GB | 1.46GB |
| `2LDriAWltwM.mp4` | 52.8 min | 1280x720 | 142.24s | 50.58s | 2.81x | 5.62GB | 1.46GB |

平均：

```text
Original pre-inference: 145.49s
Optimized pre-inference: 53.39s
Average speedup: 2.73x

Original RSS: ~4.91GB
Optimized RSS: ~1.43GB
Memory reduction: ~71%
```

## 阶段耗时平均值

| Stage | Original Avg | Optimized Avg | Notes |
|---|---:|---:|---|
| `avoid_keyframes` | 82.95s | 4.88s | 最大收益来源，原始版全帧 `ffprobe -show_frames`，优化版只输出 keyframes |
| `fetch_bitcost_and_motion_vectors` / `cv_reader_fetch_bitcost` | 35.38s | 31.87s | 优化版只取 bitcost 相关路径，仍需顺序扫到目标帧附近 |
| `decode_frames` | 14.86s | 11.39s | 优化版 ffmpeg decode 时直接 scale |
| `prepare_frames` | 1.69s | ~0.0007s | resize/pad 已前移到 ffmpeg |
| `bitcost_to_score_maps` | 0.50s | 0.52s | 基本持平 |
| `build_spatial_masks` | 1.70s | N/A | 优化版当前 readiness pipeline 不走原始 spatial mask 逻辑 |
| `build_groups` | 4.64s | 0.13s | 优化版复用预计算 block scores |
| `process_groups_make_canvases` | 2.22s | 2.71s | 优化版与原始输出 canvas 数略有差异，整体同量级 |
| `save_canvases` | 1.12s | 0.80s | 优化版保存 jpg，原始计时版保存 npy |
| `load_codec_result` | ~0.51s | ~0.41s | wrapper 加载结果耗时较小 |

## 单条详细例子

`1NYQf_OXDqI.mp4`，约 59.2 分钟，1280x714。

### 优化前

```text
total:                            190.54s
avoid_keyframes:                  108.02s
decode_frames:                     18.81s
prepare_frames:                     2.09s
fetch_bitcost_and_motion_vectors:  50.73s
bitcost_to_score_maps:              0.57s
build_spatial_masks:                1.96s
build_groups:                       4.44s
process_groups_make_canvases:       2.25s
save_canvases:                      1.21s
RSS:                                5.57GB
```

### 优化后

```text
total:                          80.01s
avoid_keyframes:                 8.65s
decode_frames:                  14.99s
cv_reader_fetch_bitcost:        51.72s
prepare_frames:                  0.001s
bitcost_to_score_maps:           0.56s
precompute_block_scores:         0.27s
build_groups:                    0.12s
process_groups_make_canvases:    2.76s
save_canvases:                   0.83s
RSS:                             1.47GB
```

## 为什么之前没有“解码后到组装拼图”的时间

之前测试优化前脚本时，只能看到外层 benchmark 的：

```text
codec_subprocess_seconds
load_codec_result_seconds
preinfer_total_seconds
```

原因是原始 `/vlm/yinxie` 脚本的 `meta.json` 没有写内部阶段耗时。外层 benchmark 只能知道整个 subprocess 花了多久，无法知道里面 `decode_frames`、`build_groups`、`process_groups_make_canvases` 各自耗时。

现在已经把原始脚本复制到：

```text
/video_vit/yunyaoyan/code/lmms_ov2_result/codec_tools/pipeline/process_video_bitcost_mv_mask_collage.py
```

并在复制版中新增：

```json
"timing_sec": {
  "decode_frames": ...,
  "prepare_frames": ...,
  "fetch_bitcost_and_motion_vectors": ...,
  "bitcost_to_score_maps": ...,
  "build_spatial_masks": ...,
  "build_groups": ...,
  "process_groups_make_canvases": ...,
  "save_canvases": ...
}
```

## 结论

长 VideoMME 上优化收益很明显：

- 平均 pre-inference 时间从 `145.49s` 降到 `53.39s`
- 平均加速约 `2.73x`
- 峰值 RSS 从约 `4.91GB` 降到约 `1.43GB`

最大瓶颈变化：

1. 优化前最大瓶颈是 `avoid_keyframes`，平均 `82.95s`
2. 优化后最大瓶颈重新变成 `cv_reader_fetch_bitcost`，平均 `31.87s`
3. `decode_frames` 仍然是第二梯队瓶颈，平均 `11.39s`

对一小时级长视频来说，目前不是主要 OOM 风险，而是顺序扫描耗时风险。优化后内存已经比较稳定，约 `1.3GB - 1.5GB`；继续提速的核心方向是减少 `cv_reader_fetch_bitcost` 对长视频的顺序扫描成本。
