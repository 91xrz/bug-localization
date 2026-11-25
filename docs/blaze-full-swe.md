# BLAZE 全量 SWE 评测（2026-11-19）

用户要求暂停等待上游检索的重排任务，改跑其他 GPU 实验。本轮在 GPU 0 启动 BLAZE 发布编码器适配版，GPU 1 继续 RRL。全量清单为 SWE 格式化 test 的 2,294 条，输出三路：`blaze_bm25`、`blaze_dense`、`blaze_rrf`。

## 固定方法与资源

沿用 `results/blaze-repair-20261116/retry` 对应的 500 条实验协议：发布编码器和预处理不变，batch_size=4、源码最大 512 tokens、报告最大 1024 tokens、dense/lexical depth=1000、chunk-RRF k=60。参考模型身份、实现哈希及依赖版本在启动时核验。

BLAZE 向量缓存仍按原实现身份校验和复用；缓存未命中的源码片段在 GPU 0 编码。不会复用 Qwen embedding 的向量，也不依赖正在生成的 S0/H0 排名。候选清单逐条与全量缓存准备阶段的清单比较。

使用独立 Elasticsearch 9.5.5 节点：HTTP 29200、transport 29300、cluster `blaze-full-20261119`、2 GB JVM heap、1 个搜索线程。数据目录 `.runtime/blaze-full-20261119-es/`。原 BM25 查询和排序函数不变，三路结果及检查点落盘后，只删除经 fingerprint 和 owner 核验的本次临时索引。断点恢复时清理该专用节点上中断遗留的自有临时索引。现有 ES 19200 全量语义检索继续运行；CPU 和磁盘资源仍共享。

## 运行与恢复

- 配置：`configs/experiments/blaze_full_swe.json`、`configs/retrieval/blaze_full_base.json`。
- 入口：`scripts/experiments/run_blaze_full.py`。
- 结果：`results/blaze/full2294-20261119/`。
- tmux：`irbl-blaze-full-20261119`。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -u scripts/experiments/run_blaze_full.py \
  --output results/blaze/full2294-20261119
```

同一命令支持续跑。每样本三路共用一个原子 `record.json` 检查点，恢复时核验样本身份、结果文件校验和和排名指标。未完整提交的样本重做，向量缓存可以复用。配置或实现变化时拒绝恢复。

`status.json` 显示运行阶段，`worker.log` 保存样本进度，`summary.json` 和各路线的 `samples.json`、`summary.json` 保存完整分母进度。任何路线错误或完整性/基础设施异常会停止任务，尚未完成的样本保持 pending，排查后续跑；不把环境故障批量计为模型失败。未完成前不发布最终全量指标。

退出时停止本次启动的专用 ES 节点，进程退出释放编码器显存；缓存、排名和审计信息保留。

## 被暂停的重排

`results/reranking_full/20261119_swe2294_gpu0` 保留所有已提交结果。暂停时原报告重排约 1,721 条，融合和完整方法各 1,531 条；准确数量以各方法 `samples.json` 为准。恢复命令见其 `status/worker.json` 中的 `resume_command`。本次不自动恢复重排，以免与用户后续 GPU 安排冲突。

## 验证

启动前 11 项既有 BLAZE 测试和 2 项全量调度测试通过。新增检查覆盖临时索引归属、三路完整提交、样本身份与指标篡改、结果文件损坏；实际首批运行另核验共同候选和标签。

## 2026-11-19 暂停与清理

按用户要求暂停全量评测，保留 184 条完整样本的三路结果、历史 500 条结果及全部日志。已清理 BLAZE 专用向量缓存、发布模型文件、旧版本兼容性验证环境、全量专用 ES 数据目录，并删除 9200/19200 中归属核验通过的 503 个 BLAZE 临时索引。19200 服务因仍用于 S0/H0 检索而保留。

清理审计：`results/blaze/cleanup-20261119/cleanup.json`；模型元数据及权重哈希已归档于该目录。代码、配置和共享源码快照保留。将来恢复 BLAZE 前需要重新准备模型，并重建已删除的缓存和索引；不能直接执行上面的续跑命令。
