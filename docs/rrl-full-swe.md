# RRL 全量 SWE 评测（2026-11-19）

用户授权后启动 RRL author-agent 的本地 7B 适配版，使用 SWE 格式化 test 全部 2,294 条、12 个项目。清单复用 `results/recall_full/20261118_swe2294_v2/selection.json`，保留所有样本和原始标签；这是统一候选协议下的适配对比，不是论文 30B/32B 配置的复现。

## 固定协议

- 参考运行：`results/rrl/crossproject500-gpu1-remaining-20261116`（原 500 条实验中 GPU 执行的 497 条）。启动前校验 RRL 实现、候选配置、GGUF、tokenizer，以及实际服务参数与参考一致。
- 模型：Qwen2.5-Coder-7B-Instruct Q4_K_M；上下文 16,384 tokens，单次输出最多 1,024 tokens，每报告最多 24 次模型调用。请求 temperature=0、seed=42、repeat_penalty=1.05。
- 原报告 body-only；共同 Python 候选，保留测试文件。RRL 作者代理执行报告改写、Lucene BM25 检索、文件查看和自检；并非 IGSIL 图搜索流程。
- 物理 GPU 1，模型别名 `rrl-local-7b`，接口 `http://127.0.0.1:8097/v1`。模型文件与 IGSIL 共用，方法实现、接口和结果目录独立。
- 自建每报告 Lucene 索引，不占用正在运行的 ES 语义检索索引；仍会消耗共享 CPU 和磁盘资源。
- 失败计零并保留在完整分母中。未完成前 `metrics` 为 null，成功子集分数仅供诊断。

## 运行产物

- 配置：`configs/experiments/rrl_full_swe.json`。
- 启动入口：`scripts/experiments/run_rrl_full.py`，调用原有 `scripts/retrieval/run_rrl.py`；未修改评分方法。
- 结果：`results/rrl/full2294-20261119/`。
- 协议、状态、日志：`results/rrl/full2294-20261119-launch/`。
- tmux：`irbl-rrl-full-20261119`。

```bash
.venv/bin/python -u scripts/experiments/run_rrl_full.py \
  --output results/rrl/full2294-20261119
```

启动器独占锁防止重复启动。服务只绑定 GPU 1，评测客户端隐藏 CUDA。服务意外退出时停止客户端，避免后续样本被批量记为模型失败。正常结束或启动器收到终止信号后，关闭本次启动的客户端和模型服务，自动释放其显存；不停止其他进程。

结果目录存在时拒绝覆盖。原 RRL 入口不支持直接续跑：若中断，应保留原始输出，按已保存的样本身份构造剩余清单，在新的目录补跑，再核验合并，不直接重启同一输出路径。

## 检查

```bash
cat results/rrl/full2294-20261119-launch/status.json
tail -f results/rrl/full2294-20261119-launch/run.log
cat results/rrl/full2294-20261119/summary.json
```

`launch_protocol.json` 保存配置、清单、实现和服务程序哈希，以及实际服务设置。评测本身保存模型、作者源码和依赖身份、每次请求响应、候选、排名和指标。`shutdown.json` 记录服务退出情况。

启动前 5 项 RRL 回归测试通过，覆盖真实 Lucene 索引与空文件审计、缓存重用、模型调用预算、上下文边界和源码读取范围。
