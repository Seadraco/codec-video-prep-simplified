# 精简 Block 选择器验证摘要

## 基线

- 上游仓库：`YunyaoYan/codec-video-prep`
- 上游提交：`77e8e91`
- 本分支只新增 `diverse_mixed_simple` Block 选择器及对应配置入口。
- FFmpeg 补丁、C++ 解码扩展、抽帧、readiness 分组、Anchor、Canvas 打包、Padding 和位置元数据均未修改。
- 默认 `selector_mode=topk_2x2_bitcost`，因此默认行为仍是官方算法。

## 自动化测试

`pytest` 覆盖以下内容：

- 75% bit-cost 与 25% diversity 的固定预算；
- `pooled4` 与 `full` 两种去重描述子；
- 相邻帧同位置 Block 去重和预算不足回填；
- Anchor-only 输出与官方选择器逐数组一致；
- Canvas 像素与 `patch_position`、`src_patch_position` 逐 Patch 对齐；
- 环境变量与 Python 配置入口；
- 非法配置和不兼容配置检查。

## 真实视频验证

H.264 High 与 HEVC Main 视频均完成 baseline、`pooled4` 和 `full` 三组端到端处理。三组结果满足：

- `frame_ids.npy` 完全一致；
- readiness 分组完全一致；
- Canvas 数量、总 Patch 数和数组形状完全一致；
- 新算法只改变非 Anchor Block 的来源位置；
- baseline 输出与独立官方目录的 Canvas 文件、`frame_ids.npy` 和 `src_patch_position.npy` 逐字节一致。

## 压力配置

使用 128 个候选帧、392×728 处理分辨率、32 张 Canvas：

| 选择器 | Canvas 构造耗时 | 端到端耗时 |
|---|---:|---:|
| 官方 baseline | 0.206 s | 6.078 s |
| `pooled4` | 0.306 s | 6.264 s |
| `full` | 0.366 s | 6.212 s |

该测试中，两种新模式均选择 10,192 个非 Anchor Block，其中 7,644 个来自 bit-cost 队列，2,548 个来自 diversity 队列；没有改变 Canvas 或 token 预算。

这些数字用于检查实现开销和结构正确性，不代表模型精度收益。算法收益仍需在相同模型、prompt、Canvas 预算和 benchmark 配置下进行 A/B 评测。
