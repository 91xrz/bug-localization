# Chunk 到文件的三种聚合

对每条缺陷报告、每种分块路线，ES 只计算一次当前快照的全部 Chunk 相似度。Max、Top-3 Mean、Normalized LogSumExp 使用完全相同的这份分数；聚合阶段不重新编码、不改变候选或 Chunk，不使用真实标签计算分数，也不启用 RRF。

## 定义

设文件有 n 个 Chunk，它们与报告的原始余弦相似度为 s₁…sₙ，m 为最高相似度：

| 方法 | 文件相似度 S(f) |
| --- | --- |
| Max | max(s₁…sₙ) |
| Top-3 Mean | 降序排列，取前 min(3,n) 个相似度的均值 |
| Normalized LogSumExp | m + τ × [log Σ exp((sᵢ−m)/τ) − log n] |

第三式与 `τ × log[(1/n) × Σ exp(sᵢ/τ)]` 等价。实现先减去最大值再除以温度，所有指数的输入均不大于 0，避免直接计算大指数；求和使用 `math.fsum`。τ 必须是有限正数，默认 0.1。只有一个块时三种方法一致。

归一化使用文件的全部 n 个块，不能只对前三个计算；把同一组分数完整重复若干次不会改变归一化 LogSumExp。它仍会受到低相似度块比例影响，与 Max 的行为不同。

三种排名均按文件分数降序、文件路径升序排序。结果保留块总数、最高分块及前三块 ID，便于解释。主 `ranking.score` 沿用原接口的 `1 + S(f)`，`aggregated_similarity` 为 S(f)；`aggregations.<method>.ranking[].score` 直接记录 S(f)，无偏移。`cosine_similarity` 字段指最佳块的原始相似度，选择其他聚合时不等于文件聚合分数。

## 配置与运行

`configs/retrieval/vector.json` 中：

```json
{
  "file_aggregation": "max",
  "aggregation_temperature": 0.1,
  "enable_fusion": false
}
```

主排名默认 Max，因此原有四路对比维持相同口径；无论主排名选择哪种方法，语义入口都会额外输出全部九组指标。用以下命令绑定当前 44 条固定清单：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_vector_batch.py --selection results\comparison\20261015_214826_508867\selection.json --no-fusion --temperature 0.1
```

本地没有该历史清单时，可按当前固定数据集顺序选择：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_vector_batch.py --repository psf/requests --limit 44 --no-fusion --temperature 0.1
```

`--aggregation top3_mean` 或 `--aggregation normalized_logsumexp` 可切换主排名。`--temperature` 支持未来在验证集比较 0.05、0.1、0.2；本次只评测预设的 0.1，不用测试集挑选温度。

## 输出与验证

结果保存在 `results/vector/batches/<run_id>/`，不覆盖既有实验：

- `aggregation_summary.csv/json`：九组 MAP@10、MRR@10、Top-1/5/10、MeanRecall@10 及成功/失败数量。
- `aggregation_samples.csv/json`：每条报告、分块、聚合的指标；完整 44 条共 396 行。
- `cases/*/<route>_chunk_scores.json`：全部块的 `chunk_id`、路径、原始余弦相似度及 SHA-256。
- `cases/*/<route>_result.json`：三种完整文件排名、各自指标、同一份分数的指纹以及独立核对结果。

返回块 ID 必须与原始清单一一对应，缺失、重复、错文件或非有限分数会报错；scroll 获取全部块，不受单页数量限制。NumPy 独立复算每个块的相似度，并核对三种文件得分与前十顺序，验证结果不参与排名。

评测沿用原有全部选定报告为分母、失败计零及缺失标签不删除的规则。样本、候选、报告、模型、分块参数和 @10 指标保持一致，本次只改变文件级聚合。

## 当前 44 条报告的结果

运行目录：`results/vector/batches/swe_bench_formatted_test_20261015_225004_395621/`。九组均成功 44 条，失败为 0，Normalized LogSumExp 的 τ 固定为 0.1。

| 分块 | 聚合 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 | MeanRecall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 动态 | Max | 0.5650 | 0.6082 | 0.4091 | 0.8636 | 0.9545 | 0.9276 |
| 动态 | Top-3 Mean | **0.5779** | 0.6061 | 0.4091 | 0.8636 | 0.9545 | 0.9290 |
| 动态 | Normalized LogSumExp | 0.3686 | 0.4144 | 0.2273 | 0.6591 | 0.8636 | 0.8366 |
| 方法 | Max | 0.5239 | 0.5684 | 0.4091 | 0.7727 | 0.9091 | 0.8949 |
| 方法 | Top-3 Mean | 0.4690 | 0.5106 | 0.2727 | 0.8182 | 0.9318 | 0.8949 |
| 方法 | Normalized LogSumExp | 0.3405 | 0.3908 | 0.2273 | 0.5909 | 0.8182 | 0.7737 |
| 固定 400 tokens | Max | 0.5466 | 0.5606 | 0.3409 | 0.8864 | 0.9318 | 0.9176 |
| 固定 400 tokens | Top-3 Mean | 0.5360 | 0.5551 | 0.3409 | 0.9091 | 0.9545 | 0.9176 |
| 固定 400 tokens | Normalized LogSumExp | 0.4058 | 0.4467 | 0.2955 | 0.6818 | 0.8636 | 0.8395 |

动态块 Top-3 Mean 的 MAP@10 比 Max 高 0.01285，但 MRR@10 略低；方法和固定块的 Top-3 Mean 在 MAP@10、MRR@10 上均低于各自 Max。τ=0.1 的归一化 LogSumExp 在三种分块下均较弱。当前主排名配置仍保留 Max；这些结果限于 `psf/requests` 的 44 条报告，未进行显著性检验或测试集温度调优。

`aggregation_verification.json` 记录了本次独立核对：132 个分块路线样本的完整 Max 排名与既有实验一致；三种聚合共同使用 171,889 个 ES Chunk 分数，396 份逐报告指标均已从原始分数独立复算。NumPy 与 ES 的最大块分数误差约 `3.21e-7`，396 个前十排名核对全部一致。所有 132 个 ES 索引及已有向量均复用，没有新增模型编码。
