# SWE 全量融合与重排（2026-11-19）

输出目录：`results/reranking_full/20261119_swe2294_gpu0`。全量 2,294 条，原样沿用 `results/recall_full/20261118_swe2294_v2/selection.json` 的顺序和样本身份；不筛选失败样本。独立保存原 500 条开发集和其余 1,794 条汇总，后者不宣称为独立盲测。

## 三组实验

| 实验 | 候选与排序 | 重排证据 |
| --- | --- | --- |
| 4 / F | H0 前 40 文件，S0 去重补齐到 50；池内等权 RRF，k=60，路径打破同分 | 不重排 |
| 5 / R0 | S0 前 50 文件 | 每文件 S0 前 3 片段，Max 聚合，同分按 S0 排名 |
| 6 / FULL | 与 F 相同的候选池、RRF 顺序 | 每文件 S0 前 1 片段，H0 去重补齐到 3；Max 聚合，同分按 RRF 顺序 |

R0、FULL 均使用原报告作为重排查询：从 S0 已保存的 embedding_text 去除固定 instruction，精确复用原语义查询的 512-token 预算内文本。假设仅用于召回、融合和片段选择，不作为重排查询。

使用本地 Qwen3-Reranker-0.6B，一次加载，两组实验顺序推理。batch_size=16、max_tokens=1536、BF16、官方 yes/no logit 分数，与原 500 条实验一致。启动时比对模型文件哈希、torch/transformers 版本、GPU 型号及提示词身份。

## 流式执行与资源

仅用物理 GPU 0（`CUDA_VISIBLE_DEVICES=0`），CPU 检索和 GPU 1 IGSIL 可同时运行。逐条读取已经完成的 S0/H0 检查点，复用候选清单、源码片段和检索分数，不重建索引、不重新编码。R0 不等待 H0；F/FULL 等待两路结果。已就绪样本处理完后，每 30 秒检查新结果。模型推理期间显存和利用率会随批次变化。

H0 上游失败只影响 F/FULL，仍继续运行 R0。依赖失败计入总分母且记零；模型、输入完整性或基础设施异常会停止任务并写出错误，不批量计为样本失败。

逐条保存结果和校验和，原始检索结果不改写。完成全部样本前不发布最终主指标；`metrics_successful_only` 是当前成功子集的临时指标。每条重排结果通过独立的片段、Max 聚合、同分顺序检查；FULL 还核验候选池和 RRF 顺序。

## 启动、检查与续跑

```bash
.venv/bin/python scripts/experiments/run_reranker_full.py prepare \
  --output results/reranking_full/20261119_swe2294_gpu0
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/experiments/run_reranker_full.py run \
  --output results/reranking_full/20261119_swe2294_gpu0
```

运行会话：`irbl-reranker-full-20261119`。日志：输出目录内 `worker.log`。状态：`status/worker.json`、`status/F.json`、`status/R0.json`、`status/FULL.json`。各方法 `samples.json` 按全量位置排序；实际完成顺序取决于上游检查点就绪情况。

断点恢复使用同一条 `run` 命令，不需要重新 prepare；文件锁防止重复启动。恢复时核验协议、结果校验和、原始输入文件哈希。协议冻结后不要修改被记录的源代码或配置。

验证：`test_reranker_full.py` 覆盖假设失败隔离、原报告查询约束、40+10 候选池、完整流式运行、续跑不重复评分和源文件损坏检测；`test_chunk_ablation.py` 验证混合片段去重与独立审计。

## 2026-11-20 双卡续跑

上游全量检索与独立核验已完成。用户授权恢复原报告重排及融合/完整方法，并使用两张 GPU。新增入口 `scripts/experiments/resume_reranker_full_dual.py` 保持原冻结入口及科学配置不变，从原目录继续写入尚未完成的样本。

按仍有任一方法未完成的报告，按原始位置排序后交替分配 GPU 0/1，每个报告只属于一个 worker；该报告的 F/R0/FULL 顺序执行，已完成的方法直接跳过。每张卡的进程仅暴露一张 GPU，使用同一逻辑 cuda:0、相同模型身份、batch=16、原查询与证据选择、Max 聚合及平局规则。汇总由持有原 `worker.lock` 的主管进程统一写入，禁止与旧入口同时启动。

原协议、全部既有结果文件和来源哈希在启动前核验；`dual_resume/protocol.json` 冻结续跑代码及分工，`prior_records.json` 保存恢复前的完整记录快照。两张卡各自保存 model identity 和日志。全部完成后检查原有记录未变、三种方法每种均覆盖 2,294 条、结果校验和与复算指标，输出 `dual_resume/verification.json`。

```bash
.venv/bin/python -u scripts/experiments/resume_reranker_full_dual.py \
  --output results/reranking_full/20261119_swe2294_gpu0
```

tmux：`irbl-reranker-dual-20261120`。日志位于 `dual_resume/supervisor.log`、`gpu0.log`、`gpu1.log`。状态为原 `status/worker.json` 和新增 `status/dual_gpu0.json`、`dual_gpu1.json`。该命令可在相同分工和协议下继续恢复；任一 worker 出错则停止另一 worker 并保留已提交检查点。任务结束后两个 GPU worker 退出并释放其显存。

双卡续跑测试及既有全量重排测试共 5 项通过，覆盖互斥分工、GPU 可见性、原查询、既有记录保持、完整三路输出与重复启动不重新评分。
