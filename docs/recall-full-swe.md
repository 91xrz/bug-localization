# SWE 全量 L0 / S0 / H0（2026-11-18）

本轮按用户授权启动前三条全量实验：原报告 BM25（L0）、原报告语义检索（S0）、直接功能推测语义检索（H0）。范围是本地 SWE 格式化版 test 的全部 **2,294 条、12 个项目**，保留原始行顺序和全部标签，不按结果筛选。尚未完成时不报告最终全量指标。

原始冻结配置：`configs/experiments/recall_full_swe.json`。原实验入口：`scripts/experiments/run_recall_full.py`。**当前续跑入口为 `scripts/experiments/run_recall_paired.py`**，按源码快照共用 S0/H0 的加载与索引，并只允许 GPU 1 执行后续编码和生成。

## 运行和资源

- 正式运行目录：`results/recall_full/20261118_swe2294_v2/`。
- tmux：`irbl-recall-full-20261118`。
- L0 与共享快照检索并行推进；代码及原报告向量继续复用原双卡 0.6B 缓存，不重新编码。检索进程隐藏所有 CUDA 设备，不占用 GPU 0 或 GPU 1 模型显存。
- 功能推测按报告正文哈希分为两份，分别绑定物理 GPU 0、1。先导入兼容的有效生成缓存并识别超长输入。初始门槛为每卡 19,000 MiB；11 月 18 日用户要求启用空闲 GPU 1 后，其门槛单独调整为 **18,500 MiB**，GPU 0 仍为 19,000 MiB。没有停止 BeetleBox 或其他用户任务。
- GPU 0 的 1,142 条生成已完成，用户将 GPU 0 用于其他实验，因此本轮后续不再调度 GPU 0。GPU 1 继续生成剩余报告。
- 独立查询准备阶段先读取兼容的单卡查询缓存。缺失的 H0 查询向量在 GPU 1 生成完成、可用显存至少 2,500 MiB 后编码，严格逐条 singleton batch。实际单卡查询身份独立记录，不伪装成双卡缓存。
- 按仓库与精确提交分组：同一快照内全部报告的查询就绪后，只验证/加载一次源码、分块与代码向量，只建立一次 ES 索引，然后连续完成所有尚未完成的 S0/H0 查询，最后清理该索引。已完成 S0 不重跑；H0 不再等待整个 S0 阶段完成。就绪快照优先执行，汇总仍按原清单顺序。
- 结果逐条原子写入，阶段有独立锁和状态。基础设施或完整性错误停止对应阶段，待恢复条目保留 pending；不把资源错误当作模型失败批量计零。
- 原 9200 服务在首轮启动时已达 1,000 分片上限。初次尝试记录保留在 `results/recall_full/20261118_swe2294/`。正式运行使用已有 19200 服务（`blaze-repair-local`、Elasticsearch 9.5.5），没有删除旧索引或调高原服务分片限制。只清理本轮新建且核验归属的临时索引。

## 冻结口径

- 只使用报告 `body`，不读取可能来自修复 PR 的独立 `title`。
- BM25 保持原词法分析器、512 个不同查询词、k1=1.2、b=0.75；Python 候选及文件内容哈希与全量缓存准备清单逐条一致。
- S0/H0 使用 Qwen3-Embedding-0.6B、AST64、1500 字符 AST 分块、512 总 tokens、64 前缀 tokens、Max 文件聚合。采用已完成全量准备的双 GPU、总 batch=64 向量身份 `c686d774e2241cd23e566db55e98244fa1765229561648f09e9c792bbc50a5b5`。
- 上述身份继续用于全部代码向量和 S0 查询。H0 singleton 查询使用实际单卡身份 `f4d436b7dd6e3e784ecd4f1622daf24e996a734278821bc35ae55f28781ce5bb`。原双卡 DataParallel 的 batch=1 查询也只分发到第一张卡；9 项目 32 条双卡/单卡缓存控制已逐位一致验证。首次真正加载 GPU 1 查询编码器时，还会绕开缓存重新编码一个控制，要求逐位一致才继续。
- 旧 500 条使用单 GPU 向量身份。本轮包括这些样本在内的所有排名都在统一双 GPU 身份下重新评分，不拼接旧排名，不宣称与旧分数逐位一致。查询和分块算法不变。
- H0 使用现有直接推测 Generator、Qwen3-8B BF16、greedy、关闭 thinking、16,384 输入上限及 1,024 输出上限。兼容缓存按模型权重、提示词、参数、环境和原始正文验证；重复正文仍保留为不同评价样本，但共享生成缓存。
- 超长输入、生成格式错误或生成未正常结束在 H0 中记失败，完整分母计零，不回退为 S0。L0/S0 不依赖 H0 成功。
- 固定数据文件、清单、配置和依赖实现 SHA256；改动实验实现后应使用新目录，不能修改原协议强行续跑。

## 产物和查看

`selection.json`、`protocol.json` 保存清单和协议。`status/` 和 `logs/` 分阶段记录状态、显存等待、进度与错误；`L0/`、`S0/`、`H0/` 保存逐样本排名与检查点；`generation/` 保留推测；`assets/` 保留后续融合/重排可复用的候选和完整代码块。

`summary.json`、`comparison.csv`、`project_summary.json` 自动更新：全量 2,294、开发 500、其余 1,794，以及分项目与项目宏平均。未完成的组不发布完整分母最终指标。其余 1,794 条不称为独立盲测集。完成全部路线后自动复算指标、核验候选/标签/结果校验和并写入 `verification.json`。

`paired_protocol.json` 冻结本次共享执行实现、原协议哈希与单查询兼容性证据；`paired_transition.json` 保存切换前 1,147 条检查点哈希和资源状态；`hypothesis_query_vectors/` 保存推测查询向量及其真实身份、生成输入哈希；`snapshot_timings/` 分别记录源码准备、分块、向量读取和各次查询耗时、索引复用情况。

`paired_live_verification.json` 的真实快照验证中，S0 完整排名和全部块分数与先前结果逐项一致。第一路含索引建立为 43.859 秒，第二路复用索引为 5.265 秒，共享准备为 9.891 秒。这是单快照实测，不是全量平均提速保证。

```bash
tmux attach -t irbl-recall-full-20261118
tail -f results/recall_full/20261118_swe2294_v2/logs/L0.log
tail -f results/recall_full/20261118_swe2294_v2/logs/S0.log
tail -f results/recall_full/20261118_swe2294_v2/logs/paired.log
cat results/recall_full/20261118_swe2294_v2/status/queries.json
cat results/recall_full/20261118_swe2294_v2/status/generate0.json
cat results/recall_full/20261118_swe2294_v2/status/generate1.json
```

14:30 后的资源调度调整使用独立包装入口 `scripts/experiments/resume_recall_full_resources.py`，原实验脚本、协议和生成参数仍保持冻结。调度切换过程中原工作进程短暂退出，随后从已验证检查点恢复；已完成样本保留。`resource_events/` 保存当前包装脚本与原协议 SHA256、门槛变化及恢复记录；较早的 `resource_amendment_20261118_gpu1.json` 是被后续完整恢复取代的首次接管尝试记录。

15:37 的 GPU 1 生成发生一次显存不足，保留为运行错误，未计入模型失败。共享检索撤销了旧 S0 的约 384 MiB CUDA 占用，恢复时启用可扩展显存分配以减少碎片。原冻结协议和旧结果不改写。

原调度及工作进程均结束、故障修复且冻结输入保持不变后，用以下入口恢复当前设置；不要与仍在运行的工作进程重复启动。当前子进程使用独立会话，停止调度器不再自动停止其工作进程，续跑前需检查各阶段锁及 PID：

```bash
CUDA_VISIBLE_DEVICES='' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -B -u scripts/experiments/run_recall_paired.py \
  --output results/recall_full/20261118_swe2294_v2
```

已通过 9 项 `test_recall*.py` 测试，涵盖候选一致性、失败与 pending 分母、项目宏平均、检查点篡改拒绝、重复正文确定性分片、按仓库和提交分组、两查询共用一次加载与索引、生成失败不影响原文查询、准备查询的生成来源与向量完整性。
