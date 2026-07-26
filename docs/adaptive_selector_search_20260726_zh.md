# 自适应 Codec Block 选择器实验报告

日期：2026-07-26

## 1. 最终配置

包级默认仍是未修改的 Public `topk_2x2_bitcost`。研究选择器必须显式
启用，当前推荐配置为：

```bash
export CODEC_SELECTOR_MODE=diverse_mixed_simple
export CODEC_DIVERSITY_FRACTION=0.30
export CODEC_NOVELTY_WEIGHT=0.50
export CODEC_DEDUP_ENABLED=1
export CODEC_DEDUP_DESCRIPTOR=pooled4
export CODEC_DEDUP_THRESHOLD_MODE=group_quantile
export CODEC_DEDUP_QUANTILE=0.15
export CODEC_DIVERSITY_ACTIVATION_MODE=sample_stride
export CODEC_DIVERSITY_MIN_SAMPLE_STRIDE_SECONDS=5
```

该配置不是对所有视频强制使用 Diversity。每个 readiness group 先计算
相邻候选帧原始帧号差的中位数，再除以源视频 FPS：

```text
sample_stride_seconds =
    median(abs(diff(group_frame_ids))) / source_fps
```

- 小于 5 秒：调用原始 Public Top-K 路径；
- 大于等于 5 秒：70% bit-cost + 30% Diversity，并启用组内 15%
  分位去重；
- FPS 缺失：保守回到 Public；
- 任一分支均严格保持 Canvas、Patch、位置数组和视觉 Token 预算不变。

## 2. 实验规模

本轮在 RapidVideoQA-200 上完成 104 个不同参数配置，覆盖：

- Diversity 比例 `0.00-0.40`；
- Novelty 权重 `0.00-1.00`；
- 固定 pooled4 MAD 阈值 `0.005-0.15`；
- group quantile 去重；
- pooled4/full 描述子；
- Diversity-only、dedup-only、联合配置和交互点。

随后选择 6 个候选，分别在三个未参与粗筛的集合上交叉验证：

| Benchmark | 题目 | 视频 | 构成 |
|---|---:|---:|---|
| RapidVideoQA-200 | 200 | 68 | 100 短 + 100 长，调参集 |
| ExternalVideoQA-200 | 200 | 68 | 100 短 + 100 长，全新视频 |
| TempCompass-MC | 1580 | 410 | 短视频时序多选 |
| RapidNExTQA-200 | 200 | 200 | 全新短视频，8 类题型 |

所有模型运行固定 OV2、官方 `lmms-eval` 适配器、greedy、
`batch_size=1`、FlashAttention 2、候选帧数、`max_pixels`、Canvas
预算和生成参数。正式对比使用离线预生成 Canvas，要求 `MISS=0`。

## 3. 为什么不采用调参集最高点

Rapid 上的最高观测点是 `d=0.30, q=0.30`：

| 配置 | Rapid | External | TempCompass | NExTQA |
|---|---:|---:|---:|---:|
| Public | 64.00% | 69.00% | 74.18% | 75.50% |
| d=.30, q=.30 | 67.50% | 67.00% | 73.54% | 74.50% |
| d=.30, q=.15 | 67.00% | 67.00% | 73.67% | 73.50% |
| d=.30, 无去重 | 66.00% | 66.00% | 74.05% | 73.50% |
| 固定阈值 .075 | 66.50% | 68.50% | 73.92% | 75.50% |

always-on 候选在 Rapid 上提高，但在三个外部集合上没有稳定收益。
固定 MAD 阈值在数据集之间也不可迁移：相同阈值在 Rapid 的最终拒绝
比例约 10%，在 TempCompass 可接近 50%。因此最终方案使用组内分位数
稳定去重强度，并只在候选帧足够稀疏时激活。

## 4. 5 秒门控结果

16 Canvas 下：

| Benchmark | Public | Adaptive | 变化 | 修复/退化 | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| RapidVideoQA-200 | 64.00% | 66.50% | +2.50 pp | 6 / 1 | 0.1250 |
| ExternalVideoQA-200 | 69.00% | 69.50% | +0.50 pp | 5 / 4 | 1.0000 |
| 两个 Rapid 合并 | 66.50% | 68.00% | +1.50 pp | 11 / 5 | 0.2101 |
| TempCompass-MC | 74.18% | 74.18% | 0.00 pp | 0 / 0 | 1.0000 |
| RapidNExTQA-200 | 75.50% | 75.50% | 0.00 pp | 0 / 0 | 1.0000 |

两个 Rapid 集合合并后，短视频为 79.0% → 79.0%，长视频为
54.0% → 57.0%。视频聚类 Bootstrap 的 95% 区间为
`[-0.50, +3.71] pp`，仍覆盖 0。因此当前结论是“点估计更稳健”，
不是“已经统计证明优于 Public”。

## 5. 门控阈值敏感性

使用 Public 与 always-on 的逐题预测，按视频的真实采样间隔重放不同
门控阈值。因为 `batch_size=1`，并且两个分支的离线资产已逐文件校验，
该重放与重新运行同一分支等价。

- `always-on`：External -2.00 pp、TempCompass -0.51 pp、NExTQA -2.00 pp；
- `1-10 秒`：四个 Benchmark 均处于稳定平台；
- `5 秒`：16 Canvas 时 Rapid/External 各 34 个长视频激活，全部短视频走 Public；
- `20 秒`：大量长视频也回到 Public，Rapid 的收益消失。

5 秒位于 16 Canvas 稳定平台内部，并在 64 Canvas 时只激活仍然
足够稀疏的长视频，避免 always-on 在 Rapid c64 上的退化。

## 6. 多视觉预算验证

视觉 Token 中位数分别为 1440/2880/5760/11520。每个输入的
Adaptive 与 Public 具有完全相同的 Canvas 和视觉 Token 数。

| Canvas 档位 | Rapid Public | Rapid Adaptive | External Public | External Adaptive |
|---:|---:|---:|---:|---:|
| 8 | 59.00% | 60.00% | 64.00% | 64.00% |
| 16 | 64.00% | 66.50% | 69.00% | 69.50% |
| 32 | 65.50% | 65.50% | 63.50% | 67.50% |
| 64 | 67.50% | 68.00% | 66.50% | 66.50% |

5 秒门控的激活视频数：

| Canvas 档位 | Rapid Active/Control | External Active/Control |
|---:|---:|---:|
| 8 | 34 / 34 | 34 / 34 |
| 16 | 34 / 34 | 34 / 34 |
| 32 | 34 / 34 | 34 / 34 |
| 64 | 12 / 56 | 13 / 55 |

External c32 修复 10 题、退化 2 题，exact McNemar
`p=0.0386`；两个 200 题集合合并后为 64.50% → 66.50%，修复
12 题、退化 4 题，`p=0.0768`，视频聚类 Bootstrap 95% 区间为
`[0.00, 4.47] pp`。这是最强的单点结果，但仍应结合其他预算和
Benchmark 判断，不能把一个配置点解释为普遍显著。

## 7. 正确性审计

- Public 控制路径：1268 个关键文件和 7792 个完整资产文件均为
  0 mismatch；
- 16 Canvas 自适应路由：Rapid、External、TempCompass、NExTQA
  合计 746 个视频，分支参考资产 0 mismatch；
- Rapid 8 Canvas：34 个短视频控制分支与对应 Public 资产 0 mismatch；
- 所有视频均无 group 内混合激活；
- 单元测试覆盖 FPS 缺失回退、稠密 Public 路径、稀疏 Active 路径、
  预算保持、Canvas/位置同步和配置校验。

## 8. 使用边界

1. Public 选择器必须继续作为回归和生产控制。
2. 研究配置的收益集中在稀疏采样的长视频；它不是通用的短视频增益。
3. 当前置信区间仍覆盖 0，扩大独立长视频 QA 样本比继续细抠同一调参集
   更有价值。
4. 选择器只优化 Canvas 内容，不解决长视频顺序 CPU 软件解码的主要
   端到端瓶颈。

## 9. 冷预处理时延

使用 2 个短视频和 2 个长视频，Public/Adaptive 交错顺序，单 worker、
无公共解码缓存，完成 3 次有效重复。每类共有 6 个观测：

| 视频 | 模式 | 总预处理 P50/P90 | cv_reader P50/P90 | Selector P50/P90 |
|---|---|---:|---:|---:|
| 短 | Public | 3.718 / 5.539 s | 3.157 / 4.936 s | 0 / 0 ms |
| 短 | Adaptive | 3.769 / 5.835 s | 3.192 / 5.230 s | 0 / 0 ms |
| 长 | Public | 125.101 / 146.387 s | 124.477 / 145.784 s | 0 / 0 ms |
| 长 | Adaptive | 128.553 / 159.363 s | 127.876 / 158.609 s | 148.9 / 175.3 ms |

短视频全部被门控到 Public，因此 Selector 为 0；其成对总时间中位差
`+0.213 s` 是解码波动。长视频成对总时间中位差 `+1.385 s`，其中
cv_reader 中位差为 `+1.408 s`，说明总时间差主要仍来自顺序解码波动。
Active Selector 自身 P50 为 `148.9 ms`，只占 Adaptive 长视频总预处理
P50 的约 `0.12%`。`process_groups_make_canvases` 的成对中位增量为
`76.0 ms`。

因此新选择策略不是 Codec 预处理的主要性能风险；要实质降低 E2E
时延，优先级仍是分段 Seek、并行/硬件解码和避免扫描到末尾候选帧。
