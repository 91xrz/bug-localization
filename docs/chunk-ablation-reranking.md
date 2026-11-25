# 固定 C 候选池的代码块选择对照（500 条）

本实验固定此前 C 组的候选文件池：功能推测 Top-40，按原文文件排名去重补足至 50。仅改变每个候选文件的代码块选择，重排查询全部使用原始报告。

| 方法 | 每文件代码块 | 执行方式 |
| --- | --- | --- |
| C1 | 原文相似度 Top-3 | 复用已审计的 C 组结果 |
| C2 | 功能推测相似度 Top-3 | GPU 0 |
| C3 | 原文 Top-1 优先，按功能推测相似度排名补足至最多 3 个不同代码块 | GPU 1 |

C3 先取原文 Top-1，再取功能推测 Top-2；发生重复时继续扫描功能推测排名，直至凑足 3 个不同 chunk_id。若文件本身不足 3 个代码块，则保留全部。相似度相同按 chunk_id 排序。功能推测只影响证据块选择，生成的推测文本不会送入重排器。

所有文件保持原文排名顺序送入重排器，最终文件得分为所选代码块得分的最大值，文件得分相同时按原文排名排序。代码块送入顺序由各自选择规则决定。候选文件、报告文本、修复标签、模型与运行参数均与 C1 核验；C1 每条结果另存哈希作为固定对照。不同输入批次组合的 BF16 运算可能产生小幅数值差异。

模型仍为本地 Qwen3-Reranker-0.6B，batch=16、max_tokens=1536、BF16。原始报告使用第一阶段原文编码器实际使用的文本，保留原有截断。整个实验不重新生成推测或语义向量。

沿用原 180 条＋新增 320 条的冻结清单，不筛除失败。全局第 308 条（pydata/xarray #5662）因已有功能推测输入超长而失败，C1/C2/C3 均按零分计入主指标分母。新增错误会单独记录，不自动替换样本或降级选择策略。

## 执行与结果

- 配置：`configs/experiments/chunk_ablation500.json`
- 入口：`scripts/experiments/run_chunk_ablation500.py`
- tmux：`irbl-chunk-C2-C3-500`
- 输出目录：`results/chunk_ablation_reranking/20261112_C2_C3_500`
- 基准 C1：`results/mixed_recall_reranking/20261111_CD_500/C`

`status.json` 为整体状态；`C2/progress.json`、`C3/progress.json` 为独立进度。两个工作进程分别使用物理 GPU 0、1，各自保存断点。以相同入口和输出目录可恢复；冻结实现或源数据变化时拒绝恢复。

完成后自动检查固定候选池与原文查询、代码块选择及去重补足、实际代码文本、文件 Max 聚合、平局顺序和指标。`final_summary.json`、`comparison.csv` 汇总 C1/C2/C3 在全部 500、原 180、新增 320 上的 MAP@10、MRR@10、Top-1/5/10、MeanRecall@10 与逐报告 AP 胜平负；`project_summary.json` 为按项目统计。每条结果记录代码块的两路语义分数、选入来源、实际重排输入和分数。

```bash
.venv/bin/python scripts/experiments/run_chunk_ablation500.py \
  --output results/chunk_ablation_reranking/20261112_C2_C3_500
```
