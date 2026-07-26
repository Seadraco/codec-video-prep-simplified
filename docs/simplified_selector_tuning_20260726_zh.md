# 精简 Codec Block 选择器调优报告

日期：2026-07-26

> 本文记录 `0.2.5.post3` 的早期固定阈值实验，已经被
> [`0.2.5.post4` 自适应选择器报告](adaptive_selector_search_20260726_zh.md)
> 取代。保留本文是为了说明固定绝对阈值和 always-on 策略为何没有被
> 继续作为推荐配置。

## 1. 结论

在本轮固定视觉预算的实验中，最佳观测配置为：

```bash
export CODEC_SELECTOR_MODE=diverse_mixed_simple
export CODEC_DIVERSITY_FRACTION=0.10
export CODEC_NOVELTY_WEIGHT=0.50
export CODEC_DEDUP_ENABLED=1
export CODEC_DEDUP_DESCRIPTOR=pooled4
export CODEC_DEDUP_THRESHOLD=0.025
```

它保留 90% Public bit-cost Block，只把 10% 预算交给
Novelty/Edge 混合排序，并启用轻量相邻同位置去重。

相对 Public 0.2.5：

| Benchmark | Public | 最佳候选 | 净变化 | 配对修复/退化 | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| RapidVideoQA-200 | 64.00% (128/200) | 65.50% (131/200) | +1.50 pp | 8 / 5 | 0.5811 |
| TempCompass-MC | 74.18% (1172/1580) | 74.30% (1174/1580) | +0.13 pp | 23 / 21 | 0.8804 |

这是两个 Benchmark 上都为正的最佳观测点，但 p 值均远大于 0.05。
因此它是后续研究候选，不是已经证明优于 Public 的统计结论。
包级默认仍是官方 `topk_2x2_bitcost`；只有显式启用研究选择器时，
才使用上述 0.10 配置。

## 2. 公平设置

所有 A/B 实验固定：

- 模型：LLaVA-OneVision-2-8B-Instruct；
- 评测器：官方 `lmms-eval` OV2 模型适配器；
- 推理：`batch_size=1`，任务和模型生成参数使用评测器默认值；
- Attention：FlashAttention 2；
- 候选帧：128；
- `target_canvas=16`；
- `group_size=32`；
- `images_per_group=4`；
- `max_pixels=150000`，`min_pixels=100352`；
- 相同视频、问题、Prompt、Anchor、readiness 分组和模型参数；
- 只修改非 Anchor Block 的选择；
- 每个视频的 Canvas、Patch、位置数组和视觉 Token 预算保持不变。

正式 GPU 推理使用离线预生成 Canvas。每个运行都检查：

- 样本数完整；
- 每个视频离线缓存命中；
- `MISS=0`；
- 在线回退为 0；
- 结果 JSON 和逐题 JSONL 均存在。

曾测试 `batch_size=2/4`，虽然更快，但各有 2/200 个预测相对
`batch_size=1` 发生变化，因此没有用于正式对照。

## 3. 数据集

| Benchmark | 规模 | 视频构成 | 作用 |
|---|---:|---|---|
| RapidVideoQA-200 | 200 题 | 100 短视频题 + 100 长视频题，共 68 个视频 | 快速筛选参数并观察长短视频差异 |
| TempCompass-MC | 1580 题 | 410 个短视频，5 类时序选择题 | 外部验证，避免只对 RapidVideoQA 调参 |

## 4. 分阶段实验

### 4.1 两个改进方向

RapidVideoQA-200：

| 配置 | Diversity | 去重 | 总体 | 短视频 | 长视频 |
|---|---:|---:|---:|---:|---:|
| Public | 0 | 关 | 64.00% | 77.00% | 51.00% |
| 混合选择 | 0.25 | 关 | 65.50% | 78.00% | 53.00% |
| 仅去重 | 0 | 开，0.025 | 64.50% | 76.00% | 53.00% |
| 联合 | 0.25 | 开，0.025 | 63.00% | 75.00% | 51.00% |

结论：两个方向不能简单叠加。25% Diversity 与去重联合反而低于
Public，因此后续先分别调比例、内部权重和去重阈值。

### 4.2 Diversity 比例

固定 `novelty_weight=0.5`，关闭去重：

| Diversity / bit-cost | Rapid 总体 | 短视频 | 长视频 |
|---|---:|---:|---:|
| 0.10 / 0.90 | 64.00% | 76.00% | 52.00% |
| 0.25 / 0.75 | 65.50% | 78.00% | 53.00% |
| 0.40 / 0.60 | 64.50% | 77.00% | 52.00% |

25% 在 Rapid 上最好，但外部验证下降到 TempCompass 73.48%，
低于 Public 0.70 pp，说明较大的 Diversity 预算不稳健。

### 4.3 Diversity 内部权重

固定 `diversity_fraction=0.25`，关闭去重：

| Novelty / Edge | Rapid | TempCompass | Temp 相对 Public |
|---|---:|---:|---:|
| 1.0 / 0.0 | 65.00% | 74.18% | 0.00 pp |
| 0.5 / 0.5 | 65.50% | 73.48% | -0.70 pp |
| 0.0 / 1.0 | 65.50% | 73.42% | -0.76 pp |

纯 Novelty 的跨数据集稳定性最好；纯 Edge 在 TempCompass 的
order 和 speed 题上退化明显。Edge 可作为补充信号，但不应获得过高预算。

### 4.4 去重阈值

固定 `diversity_fraction=0`，使用 `pooled4`：

| 阈值 | Rapid | 去重拒绝数 | TempCompass |
|---|---:|---:|---:|
| 0.015 | 63.00% | 2,810 | 未进入外部验证 |
| 0.025 | 64.50% | 4,486 | 未进入外部验证 |
| 0.040 | 65.50% | 7,162 | 74.05% |

0.040 在 Rapid 上最好，但 TempCompass 比 Public 少 2 题。
去重能改变覆盖分布，却没有形成独立、稳定的准确率收益。

### 4.5 温和联合与描述子

把 Diversity 降到 0.10 后再加入去重：

| 配置 | Rapid | 短/长 | TempCompass |
|---|---:|---:|---:|
| 0.10 + pooled4(0.025) | 65.50% | 78% / 53% | 74.30% |
| 0.10 + full(0.035) | 63.50% | 74% / 53% | 74.18% |

`full` 相对 `pooled4` 在 Rapid 少 4 题，在 TempCompass 少 2 题，
因此最终保留 `pooled4`。

## 5. 题型变化

最佳候选相对 Public：

RapidVideoQA-200 的主要变化：

| 题型 | Public | 最佳候选 | 正确题变化 |
|---|---:|---:|---:|
| Action Recognition | 15/20 | 16/20 | +1 |
| Attribute Perception | 11/18 | 14/18 | +3 |
| Spatial Reasoning | 8/12 | 9/12 | +1 |
| Temporal Reasoning | 6/14 | 7/14 | +1 |
| Action Reasoning | 17/22 | 16/22 | -1 |
| Counting Problem | 9/19 | 8/19 | -1 |
| Object Recognition | 11/22 | 10/22 | -1 |

TempCompass-MC：

| 题型 | Public | 最佳候选 | 正确题变化 |
|---|---:|---:|---:|
| action | 328/338 | 329/338 | +1 |
| attribute_change | 238/288 | 237/288 | -1 |
| direction | 177/335 | 183/335 | +6 |
| order | 238/302 | 236/302 | -2 |
| speed | 191/317 | 189/317 | -2 |

净收益主要来自 direction，其他题型存在互有得失，进一步说明当前
改进仍然是弱效应。

## 6. Canvas 指标

最佳 `pooled4` 候选：

| 指标 | RapidVideoQA | TempCompass |
|---|---:|---:|
| Group 数 | 283 | 1,743 |
| 目标局部 Block | 152,955 | 944,796 |
| bit-cost 选择 | 137,658 | 850,194 |
| diversity 选择 | 15,297 | 94,584 |
| 去重拒绝（去重后计数） | 6,311 | 281,223 |
| bit-cost 回填 | 0 | 18 |
| 每组唯一来源帧 | 27.47 | 26.97 |
| 时间分布归一化熵 | 0.8532 | 0.8632 |
| 单帧最大 Block 占比 | 0.1674 | 0.1337 |
| 平均选中 bit-cost | 399.03 | 485.35 |

TempCompass 的原始相邻差异中有大量值低于 0.025，但最终只对
“即将入选且相邻同位置已经入选”的候选执行拒绝，因此原始 CDF
不能直接当作最终拒绝率。

回填比例为 18/944,796，约 0.0019%，说明 0.025 不会造成预算
难以填满。最终 Canvas、Patch 和视觉 Token 数仍严格不变。

## 7. 时间开销

选择器直接计时：

| 数据集 | pooled4 dedup map | full dedup map | 倍率 |
|---|---:|---:|---:|
| RapidVideoQA（68 视频） | 0.101 s | 5.091 s | 50.2x |
| TempCompass（410 视频） | 0.855 s | 28.471 s | 33.3x |

离线模型评测墙钟：

| 数据集 | Public | pooled4 最佳候选 | full |
|---|---:|---:|---:|
| RapidVideoQA-200 | 不同历史运行，不作墙钟比较 | 204 s | 203 s |
| TempCompass-MC | 498 s | 526 s | 536 s |

模型评测墙钟包含容器启动、模型加载和生成，不是纯选择器耗时。
长视频端到端预处理仍由顺序软件解码主导；描述子优化不会解决
Codec 长视频解码瓶颈。

## 8. 最终判断

1. 25% Diversity 太激进，Rapid 的局部收益不能外推到 TempCompass。
2. Edge-only 和去重-only 都没有稳定跨数据集收益。
3. 10% 温和 Diversity 加 pooled4 去重是唯一两个 Benchmark 都正向的组合。
4. full 更慢且准确率更低，应删除为默认候选，只保留消融能力。
5. 当前提升没有统计显著性，Public 必须继续作为发布和回归控制路径。
6. 下一轮优化应减少参数搜索，重点研究问题相关选择或更强的低成本
   语义信号，而不是继续扩大 Edge/去重强度。
